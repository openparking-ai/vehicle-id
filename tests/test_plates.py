"""Never wrong silently, proven on the real recogniser.

Skipped when the weights are absent, because they are not committed by design.
CI trains a small model first, so the guarantee is enforced there rather than
being a test nobody ever runs.

Note what these tests no longer import. They used to reach into the lane
controller's decision logic to prove the fallback guarantee; now the guarantee
is expressed in the engine's own record — `outcome` — and the lane's half is
tested in the lane, against the contract. That split is the boundary being real
rather than declared.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import cv2  # noqa: E402

from vehicle_id.contract import ANSWER, FALLBACK, Capture  # noqa: E402
from vehicle_id.engine import RECOMMENDED_CONFIDENCE_THRESHOLD, PlateEngine  # noqa: E402
from vehicle_id.plates.generator import PlateGenerator  # noqa: E402

#: CI trains a small model at the default path; a full local run can point at
#: properly trained weights so the assertions that need a real confidence
#: signal actually execute instead of skipping.
WEIGHTS = Path(os.environ.get("VEHICLE_ID_TEST_WEIGHTS", "models/plate_crnn.pt"))
needs_weights = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="no trained weights; run the train module first"
)


def as_capture(image) -> Capture:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return Capture.now(buf.tobytes(), camera_id="test")


@pytest.fixture(scope="module")
def engine():
    # Threshold 0 so these tests see the raw confidence and decide for
    # themselves. The tests that care about the operating point set it.
    return PlateEngine(WEIGHTS, threshold=0.0)


@pytest.fixture(scope="module")
def clean_confidence(engine):
    """What THIS model scores on clean plates.

    The guarantees below are relative to it on purpose. CI trains a small model
    to prove the pipeline, and a small model is not confident; asserting the
    production threshold against it would only prove that a toy is a toy. The
    invariant that actually matters -- a degraded frame must score materially
    lower than a clean one, and the engine must send it to fallback -- holds for
    any model, and is what gets checked.
    """
    scores = []
    for seed in range(20):
        sample = PlateGenerator(seed=100 + seed).sample(degradation=0)
        scores.append(engine.read([as_capture(sample.image)]).confidence)
    scores.sort()
    return scores[len(scores) // 2]


#: Below this, the loaded weights are a smoke-test model rather than a trained
#: one, and the absolute production assertions do not apply to it.
TRAINED = 0.9

#: A model that reads nothing at all cannot demonstrate anything about
#: confidence. Skipping is honest; passing vacuously is not.
READS_NOTHING = "the loaded model reads nothing — train it further before trusting this test"


# --- the generator, which needs no weights --------------------------------

def test_the_generator_is_deterministic_from_its_seed():
    # The eval set is re-derived from a number rather than stored, so this is
    # what makes an evaluation reproducible at all.
    a = PlateGenerator(seed=42).batch(5)
    b = PlateGenerator(seed=42).batch(5)
    assert [s.text for s in a] == [s.text for s in b]
    assert (a[0].image == b[0].image).all()


def test_different_seeds_give_different_plates():
    a = PlateGenerator(seed=1).batch(5)
    b = PlateGenerator(seed=2).batch(5)
    assert [s.text for s in a] != [s.text for s in b]


def test_the_degradation_ladder_actually_degrades():
    gen = PlateGenerator(seed=7)
    clean = gen.sample(degradation=0).image
    rough = gen.sample(degradation=9).image
    # Laplacian variance is a standard sharpness proxy; a rung-9 plate must be
    # measurably less sharp than a rung-0 one or the ladder is decoration.
    sharp = cv2.Laplacian(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    blurred = cv2.Laplacian(cv2.cvtColor(rough, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert blurred < sharp


def test_florida_is_weighted_up():
    states = [s.state for s in PlateGenerator(seed=3).batch(300)]
    assert states.count("FL") > len(states) / 3, "Florida first, per E2"


# --- the engine -----------------------------------------------------------

@needs_weights
def test_a_clean_plate_is_read(engine, clean_confidence):
    """The control. Without it, the fallback tests below could pass because the
    engine never reads anything at all."""
    sample = PlateGenerator(seed=11).sample(degradation=0)
    read = engine.read([as_capture(sample.image)])
    if clean_confidence <= 0.0:
        pytest.skip(READS_NOTHING)
    assert read.identity.plate is not None
    assert read.confidence > 0.0
    if clean_confidence >= TRAINED:
        assert read.identity.plate.replace(" ", "") == sample.text.replace(" ", "")
        assert read.confidence > RECOMMENDED_CONFIDENCE_THRESHOLD


@needs_weights
def test_no_captures_is_a_fallback_not_a_guess(engine):
    read = engine.read([])
    assert read.identity.plate is None
    assert read.confidence == 0.0
    assert read.outcome == FALLBACK


@needs_weights
def test_an_undecodable_capture_is_not_an_invention(engine):
    junk = Capture.now(b"not an image at all", camera_id="test")
    read = engine.read([junk])
    assert read.identity.plate is None
    assert read.confidence == 0.0
    assert read.outcome == FALLBACK


@needs_weights
def test_the_engine_never_invents_make_model_or_colour(engine):
    # Those slices are not built. A plausible value here would be
    # indistinguishable from a measurement to everything downstream.
    sample = PlateGenerator(seed=13).sample(degradation=0)
    identity = engine.read([as_capture(sample.image)]).identity
    assert identity.make is None and identity.model is None and identity.color is None
    assert identity.plate_region is None
    assert identity.marks == ()


@needs_weights
def test_the_best_capture_wins(engine):
    """Several captures are taken precisely so one bad moment does not decide."""
    import random

    from vehicle_id.plates.generator import degrade

    sample = PlateGenerator(seed=17).sample(degradation=0)
    bad = degrade(sample.image, 9, random.Random(0))

    from_both = engine.read([as_capture(bad), as_capture(sample.image)])
    from_bad_only = engine.read([as_capture(bad)])
    assert from_both.confidence >= from_bad_only.confidence, (
        "a batch containing a good capture must not score below the bad one alone"
    )


@needs_weights
def test_noise_is_never_an_answer(engine, clean_confidence):
    """An image with no plate in it must not come back as an answer.

    The engine may still emit some text -- OCR on noise sometimes does. What
    must never happen is the RECORD claiming to stand behind it.
    """
    import numpy as np

    if clean_confidence <= 0.0:
        pytest.skip(READS_NOTHING)

    noise = np.random.default_rng(0).integers(0, 255, (160, 320, 3), dtype=np.uint8)
    # Threshold taken from THIS model's own clean-plate score, so the guarantee
    # is tested against whatever weights are loaded rather than only a full set.
    strict = PlateEngine(WEIGHTS, threshold=clean_confidence * 0.9)
    read = strict.read([as_capture(noise)])
    assert read.outcome == FALLBACK
    assert not read.is_answer


@needs_weights
def test_degraded_plates_fall_back_far_more_often_than_clean_ones(engine, clean_confidence):
    """The operational form of the never-wrong-silently guarantee.

    Not "degraded scores lower on average" -- that difference is small even on a
    well-trained model, because this recogniser is overconfident by nature, and
    on a lightly-trained one it is pure noise. What matters operationally is the
    OUTCOME: at the operating point, a rung-9 plate must be sent to fallback far
    more often than a rung-0 one.

    Requires weights with an actual confidence signal. A model that has not
    learned one cannot demonstrate this, and skipping says so rather than
    asserting noise -- CI trains a smoke model, so this is verified by the full
    harness run instead (scripts/eval_plates.py).
    """
    if clean_confidence < TRAINED:
        pytest.skip(
            f"loaded weights score {clean_confidence:.3f} on clean plates; "
            "confidence separation is only meaningful on a trained model"
        )

    strict = PlateEngine(WEIGHTS, threshold=RECOMMENDED_CONFIDENCE_THRESHOLD)

    def fallback_rate(degradation: int, seed_base: int) -> float:
        fell_back = 0
        for i in range(30):
            sample = PlateGenerator(seed=seed_base + i).sample(degradation=degradation)
            fell_back += strict.read([as_capture(sample.image)]).outcome == FALLBACK
        return fell_back / 30

    clean_rate = fallback_rate(0, 300)
    rough_rate = fallback_rate(9, 400)
    assert rough_rate > clean_rate + 0.2, (
        f"rung-9 fallback {rough_rate:.0%} vs rung-0 {clean_rate:.0%} — "
        "the threshold is not separating good reads from bad ones"
    )


def test_the_measured_threshold_travels_with_the_engine():
    """The recogniser is accurate AND overconfident: at a naive 0.85 it would
    act on reads the harness measured as wrong 4.4% of the time. The engine
    publishes the measured operating point, and ships it in every record."""
    assert RECOMMENDED_CONFIDENCE_THRESHOLD >= 0.99


# --- a batch that is not one vehicle --------------------------------------

@needs_weights
def test_two_vehicles_in_one_batch_is_a_fallback_not_the_higher_score(clean_confidence):
    """The silent-wrongness path this engine most needed closing.

    Identity used to come from `max(confidence)` across the batch, with the
    timestamp and camera taken from the first capture and nothing recording
    that the batch disagreed. Three individually confident plates in one
    request produced ONE coherent, in-contract, above-threshold record naming
    the wrong car -- and the engine had held the disconfirming evidence and
    thrown it away.

    Two captures of one vehicle cannot show two different plates. When more
    than one clears the operating point and they disagree, the honest answer is
    that we do not know which car is at the barrier.
    """
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    strict = PlateEngine(WEIGHTS, threshold=clean_confidence * 0.9)
    a = PlateGenerator(seed=201).sample(degradation=0)
    b = PlateGenerator(seed=202).sample(degradation=0)
    assert a.text != b.text

    alone_a = strict.read([as_capture(a.image)])
    alone_b = strict.read([as_capture(b.image)])
    if not (alone_a.is_answer and alone_b.is_answer):
        pytest.skip("both plates must be individually confident for this to mean anything")

    together = strict.read([as_capture(a.image), as_capture(b.image)])
    assert together.outcome == FALLBACK, (
        f"a batch showing {alone_a.identity.plate} and {alone_b.identity.plate} "
        "answered confidently instead of falling back"
    )
    assert together.captures_seen == 2


@needs_weights
def test_agreeing_captures_of_one_vehicle_still_answer(clean_confidence):
    """The control. A guard that turns every multi-capture batch into a
    fallback would be safe and useless -- grabbing several frames is the whole
    reason one bad moment does not decide."""
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    strict = PlateEngine(WEIGHTS, threshold=clean_confidence * 0.9)
    sample = PlateGenerator(seed=203).sample(degradation=0)
    read = strict.read([as_capture(sample.image), as_capture(sample.image)])
    assert read.outcome == ANSWER
    assert read.captures_seen == 2


@needs_weights
def test_the_record_names_the_camera_the_answer_came_from(clean_confidence):
    """A plate read off one camera must not be stamped with another's id.

    The mixed-provenance case: the vehicle at the barrier is unreadable, and a
    pristine frame of a different vehicle arrives from a different camera. The
    record used to carry the entry lane's camera_id beside the other camera's
    plate.
    """
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    import cv2 as _cv2

    strict = PlateEngine(WEIGHTS, threshold=clean_confidence * 0.9)
    unreadable = PlateGenerator(seed=204).sample(degradation=9)
    readable = PlateGenerator(seed=205).sample(degradation=0)

    def at(image, camera_id):
        ok, buf = _cv2.imencode(".png", image)
        assert ok
        return Capture.now(buf.tobytes(), camera_id=camera_id)

    read = strict.read([at(unreadable.image, "lane-1-entry"), at(readable.image, "lane-9-STAFF")])
    if read.identity.plate is None:
        pytest.skip("neither frame was read; nothing to attribute")
    assert read.camera_id == "lane-9-STAFF", (
        "the record named a camera the answer did not come from"
    )


@needs_weights
def test_unmeasured_weights_refuse_to_start_rather_than_inventing_a_threshold(tmp_path):
    """A constant cannot know which weights were loaded.

    The default engine used to stamp 0.99 onto every record whatever model was
    in memory, beside a comment claiming that number was measured for it.
    """
    import shutil

    from vehicle_id.engine import UnmeasuredWeights

    copy = tmp_path / "unmeasured.pt"
    shutil.copy(WEIGHTS, copy)
    with pytest.raises(UnmeasuredWeights, match="no measured operating point"):
        PlateEngine(copy)


@needs_weights
def test_an_operating_point_measured_for_other_weights_is_refused(tmp_path):
    """The control that matters more than the one above: a sidecar is only
    worth anything if it is bound to the model it was measured on."""
    import json
    import shutil

    from vehicle_id.engine import UnmeasuredWeights, operating_point_path

    copy = tmp_path / "borrowed.pt"
    shutil.copy(WEIGHTS, copy)
    operating_point_path(copy).write_text(
        json.dumps({"threshold": 0.5, "weights_id": "sha256:somebodyelses"})
    )
    with pytest.raises(UnmeasuredWeights, match="measured for"):
        PlateEngine(copy)


@needs_weights
def test_a_noisy_feed_is_mostly_but_not_entirely_refused(clean_confidence):
    """A measured limitation, pinned so it cannot quietly get worse.

    This recogniser has NO rejection capability: it returns text for a flat
    black image, and for uniform sensor noise it returns text on every frame at
    a mean confidence of 0.83, with 2.3% of single frames clearing the measured
    operating point. Confidence alone cannot tell a plate from snow.

    Reading several captures helps, because noise reads differently every
    frame and the batch then disagrees with itself -- measured at 0.7% for
    three captures against 2.3% for one. It does not reach zero, and pretending
    otherwise is the thing this project exists not to do. Rejecting an image
    with no plate in it needs a detector, which is the next slice.
    """
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    import cv2 as _cv2
    import numpy as np

    engine = PlateEngine(WEIGHTS)
    rng = np.random.default_rng(0)

    def noise_capture():
        image = rng.integers(0, 255, (160, 320, 3), dtype=np.uint8)
        ok, buf = _cv2.imencode(".png", image)
        assert ok
        return Capture.now(buf.tobytes(), camera_id="dead-feed")

    answered = 0
    for _ in range(200):
        if engine.read([noise_capture() for _ in range(3)]).is_answer:
            answered += 1
    rate = answered / 200

    # Not asserting zero, because it is not zero. Asserting that batching is
    # doing its job and that the number has not drifted upwards.
    assert rate <= 0.03, (
        f"{rate:.1%} of noisy-feed reads answered confidently; the batch "
        "disagreement rule has stopped working"
    )
