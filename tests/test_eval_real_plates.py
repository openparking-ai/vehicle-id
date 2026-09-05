"""The real-photograph harness: its buckets, its padding, and its refusal.

This file is the plate-string guard. `check-no-real-data.js` is not -- that
script matches e-mail addresses in tracked files and cannot fire on a
registration, so the guarantee that no plate reaches a file lives here.

Three things are proven, and each is proven able to fail:

  * **the buckets**, which are the harness's whole output. A read can be
    textually exact and still FALL BACK under the operating point, so right/wrong
    and ANSWER/FALLBACK are counted as five cells rather than three, and a bug
    that merged two of them would produce a plausible table nothing could
    falsify. The engine's own `recognizer=` injection point makes every cell
    reachable deterministically with NO WEIGHTS, so this never skips and never
    needs an allowance.
  * **the padding**, in both directions. An axis-aligned box around a tilted
    plate is taller than the plate, and past enough tilt it is already narrower
    than the training aspect -- so one fixture wider and one narrower are both
    required, and neither may be dropped: "always pad the height" leaves the
    wide one green, "always pad the width" leaves the narrow one green.
  * **the refusal**, on both sides. Catching a planted registration is half a
    guard; the other half is not eating the object's own fields, and that half
    is DERIVED from the object rather than listed, so a field added in a later
    round is covered without anyone remembering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")
pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from eval_real_plates import (  # noqa: E402
    AS_IS,
    LETTERBOX,
    RegistrationInOutput,
    PathInsideRepository,
    TILTS,
    as_capture,
    build_output,
    classify,
    empty_buckets,
    letterbox_to_training_aspect,
    refuse_repository_paths,
    registration_shaped,
    registration_window,
    training_aspect,
    write_output,
)
from vehicle_id.engine import PlateEngine  # noqa: E402
from vehicle_id.plates.generator import PLATE_H, PLATE_W  # noqa: E402

guarantee = pytest.mark.guarantee


# --- the buckets ---------------------------------------------------------


class StubRecogniser:
    """Anything with `.read(image) -> (text, conf)`.

    The engine exposes `recognizer=` precisely so a question that is not about
    how well the model reads can be answered without a trained model. Which
    bucket a result lands in is such a question.
    """

    def __init__(self, text: str, confidence: float) -> None:
        self._text, self._confidence = text, confidence

    def read(self, image):
        return self._text, self._confidence


def engine_with(text: str, confidence: float, threshold: float) -> PlateEngine:
    return PlateEngine(recognizer=StubRecogniser(text, confidence), threshold=threshold)


def one_capture():
    image = np.full((60, 200, 3), 220, np.uint8)
    return as_capture(image, camera_id="test")


#: (case, stub text, stub confidence, threshold, label) -> the only bucket that moves.
BUCKET_CASES = [
    ("exact_answer", "ABC1234", 0.995, 0.99, "ABC 1234"),
    ("exact_fallback", "ABC1234", 0.995, 0.999, "ABC 1234"),
    ("wrong_answer", "ABC9999", 0.995, 0.99, "ABC 1234"),
    ("wrong_fallback", "ABC9999", 0.995, 0.999, "ABC 1234"),
    ("no_text", "", 0.0, 0.99, "ABC 1234"),
]


@guarantee
@pytest.mark.parametrize("expected,text,conf,threshold,label", BUCKET_CASES)
def test_each_bucket_is_reachable_and_only_it_moves(expected, text, conf, threshold, label):
    """One case per bucket, each asserting the other four stay zero.

    The fail-control is structural: swap two buckets in `classify` and at least
    two of these five go red, because each case pins every cell rather than the
    one it expects.
    """
    engine = engine_with(text, conf, threshold)
    read = engine.read([one_capture()])
    buckets = empty_buckets()
    buckets[classify(read, label)] += 1
    assert buckets[expected] == 1, f"{expected} did not receive the read"
    for other in buckets:
        if other != expected:
            assert buckets[other] == 0, f"{other} moved when only {expected} should have"


@guarantee
def test_an_exact_read_below_the_threshold_is_not_counted_as_wrong():
    """The cell that would eat the answer.

    Merging FALLBACK into "no read", or into "wrong", is the specific mistake
    this five-way split exists to prevent: on real plates almost every correct
    read is expected to land under the operating point measured for these
    weights, and a harness that called those wrong would report the opposite of
    what happened.
    """
    read = engine_with("ABC1234", 0.995, 0.999).read([one_capture()])
    assert read.identity.plate is not None
    assert read.outcome == "fallback"
    assert classify(read, "ABC 1234") == "exact_fallback"


# --- the padding ---------------------------------------------------------


def solid(w: int, h: int, colour=(200, 190, 180)) -> np.ndarray:
    return np.full((h, w, 3), colour, np.uint8)


@guarantee
@pytest.mark.parametrize(
    "w,h,which",
    [
        (900, 190, "wider than the training aspect -> pad the height"),
        (400, 260, "narrower than the training aspect -> pad the width"),
    ],
)
def test_letterbox_reaches_the_training_aspect_from_both_sides(w, h, which):
    """Both branches, because a tilted plate's box can be on either side of it.

    Neither fixture may be dropped. Plant "always pad the height" and the narrow
    fixture goes red while the wide one stays green; plant "always pad the
    width" and the wide one goes red while the narrow one stays green. One
    fixture proves one bug.
    """
    out = letterbox_to_training_aspect(solid(w, h))
    got = out.shape[1] / out.shape[0]
    assert got == pytest.approx(training_aspect(), rel=0.01), which
    assert out.shape[1] >= w and out.shape[0] >= h, "padding may only add"


@guarantee
def test_the_training_aspect_is_read_from_the_generator_not_typed():
    """A second copy of this number would make the claim true by coincidence."""
    assert training_aspect() == PLATE_W / PLATE_H


# --- the refusal, catching -----------------------------------------------


def representative_fixture(condition: str = AS_IS) -> dict:
    """The real object, built by the harness's own builder, with zero counts.

    Not a hand-built dict: the shape comes from the function the real run calls,
    so it cannot drift. The VALUES are supplied here and are deliberately
    representative -- a 16-hex `weights_id` is the form that would exercise the
    length branch, where a placeholder would prove nothing.
    """
    return build_output(
        buckets=empty_buckets(),
        per_tilt={t: {"n": 0, "exact": 0, "wrong": 0, "no_text": 0} for t in TILTS},
        mean_conf_exact=None,
        mean_conf_wrong=None,
        n=0,
        excluded=0,
        weights_id="sha256:0de21983b58b0ecd",
        threshold=0.99,
        condition=condition,
        python_version="3.12.14",
        torch_version="2.14.0",
        package_version="0.1.0",
        script_digest="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        window=registration_window(),
        timestamp="2026-09-04T15:04:05+03:00",
    )


@guarantee
def test_the_writer_refuses_a_registration_at_the_top_level(tmp_path):
    obj = representative_fixture()
    obj["note"] = "ABC1234"
    with pytest.raises(RegistrationInOutput):
        write_output(obj, tmp_path / "out.json")


@guarantee
def test_the_writer_refuses_a_registration_nested_in_a_sub_object(tmp_path):
    """The output nests, so the guard must recurse. Planted one level down."""
    obj = representative_fixture()
    obj["per_tilt"]["slight"]["note"] = "ABC1234"
    with pytest.raises(RegistrationInOutput):
        write_output(obj, tmp_path / "out.json")


@guarantee
def test_the_writer_refuses_a_registration_used_as_a_key(tmp_path):
    obj = representative_fixture()
    obj["per_tilt"]["ABC1234"] = {"n": 0}
    with pytest.raises(RegistrationInOutput):
        write_output(obj, tmp_path / "out.json")


@guarantee
@pytest.mark.parametrize("plate", ["ABC1234", "ABC 1234", "ABC-1234", "abc1234", "AB1234", "ABC12345"])
def test_the_predicate_catches_the_shapes_a_registration_is_written_in(plate):
    assert registration_shaped(plate), f"{plate!r} should be caught"


# --- the refusal, accepting: DERIVED from the object ----------------------


def every_string(node, path="$"):
    """The test's OWN traversal, so it can fail independently of the guard's.

    If this reused the harness's walk, a guard that stopped recursing early
    would still pass a test that recursed correctly -- the two would agree
    because they were one.
    """
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                yield f"{path}.<key>", key
            yield from every_string(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, item in enumerate(node):
            yield from every_string(item, f"{path}[{i}]")


@guarantee
@pytest.mark.parametrize("condition", [AS_IS, LETTERBOX])
def test_no_field_the_object_carries_looks_like_a_registration(condition):
    """(a) Derived, not listed: every string in the real object, keys included.

    A field added in a later round is covered without anyone remembering to add
    it here, and the day a bare hex `weights_id` lands inside the window this
    says so -- naming the field, so "too broad" is diagnosable rather than an
    opaque refusal.
    """
    offenders = [
        f"{path}={value!r}"
        for path, value in every_string(representative_fixture(condition))
        if registration_shaped(value)
    ]
    assert not offenders, "the guard would refuse the object's own fields: " + ", ".join(offenders)


@guarantee
@pytest.mark.parametrize("condition", [AS_IS, LETTERBOX])
def test_the_writer_accepts_the_object_it_will_really_write(tmp_path, condition):
    """(b) One call. The guard must not reject a legitimate run."""
    path = write_output(representative_fixture(condition), tmp_path / "out.json")
    assert json.loads(path.read_text())["crop_condition"] == condition


@guarantee
def test_the_guard_reaches_deeper_than_the_real_object_nests(tmp_path):
    """(c) The recursion-depth control, on the accept side and the catch side.

    A guard that stopped at the depth the real object happens to reach would
    pass every test above. So: one level deeper than the object nests, a clean
    structure is still accepted and a planted registration is still refused.
    """
    clean = representative_fixture()
    clean["per_tilt"]["slight"]["deeper"] = {"deeper_still": {"note": "nothing to see"}}
    write_output(clean, tmp_path / "clean.json")

    planted = representative_fixture()
    planted["per_tilt"]["slight"]["deeper"] = {"deeper_still": {"note": "ABC1234"}}
    with pytest.raises(RegistrationInOutput):
        write_output(planted, tmp_path / "planted.json")


# --- paths ---------------------------------------------------------------


@guarantee
def test_a_path_inside_this_repository_is_refused():
    """The check that must fail: handed this very work tree, it refuses."""
    with pytest.raises(PathInsideRepository):
        refuse_repository_paths(photos=ROOT)


@guarantee
def test_a_path_outside_every_repository_passes(tmp_path):
    """The control for the control: without it the refusal could be uncondtional."""
    refuse_repository_paths(photos=tmp_path, labels=tmp_path / "l.json", out=tmp_path / "o.json")
