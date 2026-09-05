"""Never wrong silently, proven on the real recogniser.

Skipped when the weights are absent, because they are not committed by design.

**Most of this file does not run in CI, and that is not a hidden fact any more.**
CI trains a 600-step model to prove the pipeline; a model that small answers
almost nothing, so every assertion below that needs a real confidence signal
skips there. The header used to claim the opposite -- "CI trains a small model
first, so the guarantee is enforced there rather than being a test nobody ever
runs" -- and three presence-gate guarantees skipped in every CI run for a
fortnight on the strength of it, with the build green throughout.

Two things now hold that claim honest:

  * whether the gate is CONNECTED needs no weights at all and moved to
    `test_presence_wiring.py`, which always runs;
  * a `@pytest.mark.guarantee` test that skips FAILS the run unless the job
    names its reason in `VEHICLE_ID_ALLOW_SKIPPED` (see `conftest.py`). The
    engine job names "needs weights", so the skips below are declared rather
    than silent.

What is left here is ACCURACY, which genuinely needs weights and is measured by
`scripts/eval_plates.py` and `scripts/eval_presence.py`.

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


@pytest.mark.guarantee
def test_every_template_in_the_tuple_is_actually_sampled():
    """Derived from TEMPLATES, so an addition is noticed without being told.

    The fail-control is NOT removing the template from the tuple -- the tuple is
    both subject and oracle there, so the test would stay green with one fewer
    template. It is setting the template's `weight` to 0.0: it stays in the
    list, the sampler cannot reach it, and this goes red. That is also the
    failure worth catching, because a weight is the one field that can drop a
    template out of training while every list still shows it.
    """
    from vehicle_id.plates.templates import TEMPLATES

    sampled = {s.state for s in PlateGenerator(seed=0).batch(300)}
    missing = [t.state for t in TEMPLATES if t.state not in sampled]
    assert not missing, f"never sampled in 300 draws at seed 0: {missing}"


@pytest.mark.guarantee
def test_the_class_count_is_a_stated_fact():
    """A charset change is a full retrain and a new `weights_id`.

    Stated here so it can never be a silent class-count change discovered as a
    shape mismatch when an older checkpoint refuses to load.
    """
    from vehicle_id.plates.model import NUM_CLASSES
    from vehicle_id.plates.templates import charset

    assert len(charset()) == 36
    assert NUM_CLASSES == 37


@pytest.mark.guarantee
def test_a_template_with_its_own_letters_draws_only_those():
    """A restricted layout must not be trained on registrations it cannot have."""
    from vehicle_id.plates.templates import TEMPLATES

    restricted = [t for t in TEMPLATES if t.letters]
    assert restricted, "no template declares its own letter set; this check is empty"
    samples = PlateGenerator(seed=0).batch(600)
    for template in restricted:
        drawn = {
            ch
            for s in samples
            if s.state == template.state
            for ch in s.text
            if ch.isalpha()
        }
        assert drawn, f"{template.state} never sampled; cannot check its letters"
        assert drawn <= set(template.letters), (
            f"{template.state} drew {sorted(drawn - set(template.letters))}, "
            "which its layout cannot contain"
        )


@pytest.mark.guarantee
def test_the_widest_registration_fits_every_template_text_area():
    """The band and the scale range are chosen together, not separately.

    The generator's scale variation stands in for the font variation it cannot
    model, so it is not narrowed for a banded template. Instead the band is
    sized so the widest rendering still fits. Thickness is not a dimension here:
    it does not change the advance width, measured.
    """
    import cv2

    from vehicle_id.plates.generator import PLATE_W
    from vehicle_id.plates.templates import DIGITS, LETTERS, TEMPLATES

    fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX)

    def widest(pool):
        return max(pool, key=lambda c: max(cv2.getTextSize(c * 8, f, 1.9, 6)[0][0] for f in fonts))

    for template in TEMPLATES:
        area = PLATE_W - template.band
        letter = widest(template.letters or LETTERS)
        digit = widest(DIGITS)
        for pattern in template.patterns:
            specimen = pattern.replace("L", letter).replace("N", digit)
            tw = max(cv2.getTextSize(specimen, f, 1.9, 6)[0][0] for f in fonts)
            # Every template: the registration never leaves the plate.
            assert tw <= area, (
                f"{template.state} pattern {pattern!r} renders {tw}px into a "
                f"{area}px text area (band {template.band}) and would run off the plate"
            )
            # A BANDED template additionally leaves room to centre, which is what
            # bounds the band width. The unbanded templates are NOT held to this:
            # GA's "LLLL NNN" and TX's "NNN LLLL" render 306px into 320 and reach
            # the 8px floor already, unchanged by this round and recorded rather
            # than quietly fixed.
            if template.band:
                assert tw <= area - 16, (
                    f"{template.state} pattern {pattern!r} renders {tw}px into a "
                    f"{area}px text area (band {template.band}); the band is too "
                    "wide for the scale range"
                )


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
    black image, and for uniform sensor noise it returns text on every frame.
    Confidence alone cannot tell a plate from snow.

    Reading several captures helps, because noise reads differently every frame
    and the batch then disagrees with itself. **It does not reach zero**, and
    pretending otherwise is the thing this project exists not to do.

    The rates themselves are not typed here. They are measured by
    `scripts/eval_presence.py` into `docs/measured/presence.json`, because a
    figure that lives only in prose drifts: this docstring said 0.7% while the
    README beside it said 0.3%, and nothing in the repository could tell you
    which was measured. What is asserted here is the SHAPE -- that batching
    helps and that the residual has not drifted upwards.
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


@needs_weights
@pytest.mark.guarantee
def test_the_presence_gate_moves_the_noise_measurement(clean_confidence):
    """D6. A detector that does not move the number is not evidence of anything.

    Whether the gate is CONNECTED is proven in `test_presence_wiring.py` with no
    weights at all. What this adds is the ACCURACY half, on a real recogniser:
    handed a dead feed, does the gate actually stop the confident answers the
    recogniser produces out of noise?

    The rates are read from the evidence file rather than restated, and the
    control is the ungated arm measured in the same run -- if it also answers
    zero, this assertion proves nothing and says so.
    """
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    import cv2 as _cv2
    import numpy as np

    from lanes import lane
    from vehicle_id.presence import PresenceDetector

    rng = np.random.default_rng(0)

    def noise_capture():
        image = rng.integers(0, 255, (160, 320, 3), dtype=np.uint8)
        ok, buf = _cv2.imencode(".png", image)
        assert ok
        return Capture.now(buf.tobytes(), camera_id="dead-feed")

    gated = PlateEngine(WEIGHTS, presence=PresenceDetector(reference=lane(90, seed=1)))
    plain = PlateEngine(WEIGHTS)

    reads = 150
    batches = [[noise_capture() for _ in range(3)] for _ in range(reads)]

    ungated_answers = sum(plain.read(batch).is_answer for batch in batches)
    gated_answers = sum(gated.read(batch).is_answer for batch in batches)

    # The control, measured here rather than assumed. On these weights the
    # ungated arm answers between 1 and 5 times in 150; if it ever answers zero
    # the gated assertion below is vacuous and must not be allowed to pass.
    assert ungated_answers > 0, (
        f"the ungated control answered 0/{reads} noisy reads, so the gated "
        "assertion below would pass with the gate removed and proves nothing"
    )
    assert gated_answers == 0, (
        f"{gated_answers}/{reads} noisy-feed reads got past the presence gate "
        f"(ungated control: {ungated_answers}/{reads})"
    )


@needs_weights
@pytest.mark.guarantee
def test_the_gate_does_not_refuse_a_lane_with_a_vehicle_in_it(clean_confidence):
    """The control that matters more. A gate that rejects everything would post
    a perfect noise score and break the product.

    Note what it does NOT assert: that a plate comes out. The gate is handed the
    LANE view, which is what it is comparing against a reference of that lane
    empty; this recogniser reads a plate-shaped crop. Asserting both here would
    tie the gate's control to the recogniser's framing, and the first version
    did exactly that -- it passed a bare plate crop against a flat black
    "reference", which any gate admitting large changes passes, including one
    that also admits a dead camera.
    """
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    from lanes import lane, vehicle
    from vehicle_id.presence import PresenceDetector

    gated = PlateEngine(WEIGHTS, presence=PresenceDetector(reference=lane(90, seed=1)))
    read = gated.read([as_capture(vehicle(420, 240, seed=40))])

    assert read.presence is True, "the gate refused a lane with a vehicle in it"
    assert read.presence_confidence is not None


@needs_weights
@pytest.mark.guarantee
def test_a_caller_submitting_tight_plate_crops_is_not_measured_not_refused(clean_confidence):
    """The framing the gate cannot serve, and the safe way for it to fail.

    A caller replacing an LPR unit submits a crop around the plate, not a view
    of the lane. There is no empty-lane background in such a frame, so occupancy
    runs to the ceiling and presence is NOT MEASURED. That must be `null` -- the
    old behaviour, a plate read normally -- and never `false`, which at the lane
    would refuse every customer of every crop-submitting deployment.
    """
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    from lanes import lane
    from vehicle_id.presence import PresenceDetector

    gated = PlateEngine(WEIGHTS, presence=PresenceDetector(reference=lane(90, seed=1)))
    sample = PlateGenerator(seed=11).sample(degradation=0)
    read = gated.read([as_capture(sample.image)])

    assert read.presence is not False, (
        "a tight plate crop was reported as an empty lane; every crop-submitting "
        "deployment would refuse every customer"
    )
    assert read.presence is None
    assert read.identity.plate, "a plate crop must still be read when presence is null"
    assert read.is_answer


@needs_weights
@pytest.mark.guarantee
def test_with_no_detector_presence_is_not_measured_and_nothing_changes(clean_confidence):
    """A lane that has not configured a reference view must behave exactly as it
    did before this stage existed."""
    if clean_confidence < TRAINED:
        pytest.skip("needs weights that actually read plates confidently")

    engine = PlateEngine(WEIGHTS)
    sample = PlateGenerator(seed=11).sample(degradation=0)
    read = engine.read([as_capture(sample.image)])
    assert read.presence is None
    assert read.is_answer
