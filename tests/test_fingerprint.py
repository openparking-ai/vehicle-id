"""The appearance descriptor, and the matcher that publishes a distance.

Every test here runs with NO trained model, NO weights and NO photograph: the
descriptor is classical, so its properties are provable from images this file
invents. That is the round's whole point about licensing said as a test -- if
any of this needed a checkpoint, the descriptor would have acquired the licence
question the plate recogniser spent a round escaping.

Five things are proven, and each is proven able to fail:

  * **every published term is a DISTANCE**, lower meaning more alike. One of the
    five is natively a similarity and is converted exactly once; a sign left the
    other way round would make a table of five numbers where one column silently
    means the opposite of the other four.
  * **a descriptor round trips**, so the record compared in memory is the record
    written down. The histogram is quantised on the way into the string, and
    without the round trip a stored descriptor would score differently from the
    one that produced it -- in the third decimal place, months later.
  * **descriptors that do not compare REFUSE**, on version and on kind, and the
    version refusal happens before a byte of the payload is read.
  * **there are THREE outcomes.** "Nothing matched" and "there was nothing to
    match" are different answers and are never the same cell.
  * **the chooser refuses**, naming the conflict, when no threshold satisfies
    the constraints it was given -- and still chooses when one does, because
    "refuses" and "refuses everything" are otherwise the same test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from eval_fingerprint import second_view, synthetic_vehicle  # noqa: E402

from vehicle_id.fingerprint import (  # noqa: E402
    DESCRIPTOR_PREFIX,
    DESCRIPTOR_VERSION,
    MEASURABLE_MIN_KEYPOINTS,
    ORB,
    SIFT,
    TERM_NAMES,
    TERMS,
    Descriptor,
    DescriptorComputer,
    IncomparableDescriptors,
    MalformedDescriptor,
    NoOperatingPoint,
    TermDistance,
    choose_operating_point,
    compare,
    compute,
    compute_text,
    decode,
    descriptor_version_of,
    rates_at,
)

guarantee = pytest.mark.guarantee

KINDS_UNDER_TEST = [ORB, SIFT]

#: The descriptor itself needs no torch -- it is classical, which is the point.
#: The engine does, because it also reads plates, so only the engine section at
#: the bottom of this file is gated. The reason is worded to match the allowance
#: the no-engine CI job already names.
needs_engine = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="could not import torch"
)


def a_vehicle(seed: int = 7) -> np.ndarray:
    return synthetic_vehicle(seed)


def the_same_vehicle_again(image: np.ndarray, seed: int = 7) -> np.ndarray:
    return second_view(image, seed)


def a_flat_frame() -> np.ndarray:
    """A night arrival, a plain van filling the crop, a wet windscreen.

    One flat colour: no corners, no edges, nothing for a keypoint detector to
    find. This is the frame the third outcome exists for.
    """
    return np.full((240, 360, 3), 118, np.uint8)


# --- the terms are distances ---------------------------------------------


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
@pytest.mark.parametrize("term", TERM_NAMES)
def test_every_published_term_is_a_distance(kind, term):
    """Lower means more alike, on all five, for both permitted detectors.

    The fail-control is the conversion itself: `colour_intersection` is natively
    a SIMILARITY. Publish `cv2.compareHist(..., HISTCMP_INTERSECT)` raw instead
    of `1 - it` and this parametrisation goes red on that term alone, while the
    other four stay green -- which is exactly the bug being guarded against,
    because a single column meaning the opposite of its neighbours is invisible
    in a table.
    """
    car = a_vehicle(seed=11)
    again = the_same_vehicle_again(car, seed=11)
    other = a_vehicle(seed=12)

    same = compare(compute(car, kind), compute(again, kind))[term]
    different = compare(compute(car, kind), compute(other, kind))[term]

    assert same.measurable and different.measurable, f"{term} was not measurable at all"
    assert same.value < different.value, (
        f"{term}: the same vehicle scored {same.value} and a different one "
        f"{different.value}. Lower must mean more alike."
    )


@guarantee
def test_the_scale_table_covers_every_published_term_and_nothing_else():
    """The units a reader needs are DATA, read by the harness rather than
    restated in it. A term added without its scale would publish five numbers
    where one has no stated meaning."""
    assert tuple(t.name for t in TERMS) == TERM_NAMES
    assert len(set(TERM_NAMES)) == len(TERM_NAMES)


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_a_comparison_is_symmetric(kind):
    """`d(a, b)` and `d(b, a)` are the same number, on every term.

    Not free: a ratio test in one direction keeps different matches from the
    same test in the other, so the structure term matches BOTH ways and averages.
    Drop one direction and this goes red.
    """
    a, b = compute(a_vehicle(3), kind), compute(a_vehicle(4), kind)
    forward, backward = compare(a, b), compare(b, a)
    for term in TERM_NAMES:
        assert forward[term].value == pytest.approx(backward[term].value), term
        assert forward[term].measurable == backward[term].measurable


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_a_vehicle_compared_with_itself_sits_at_the_floor(kind):
    """The degenerate control: without it, "lower is closer" could hold on a
    scale that never actually reaches its lower bound.

    The four histogram and geometry terms are exactly 0. STRUCTURE is not, and
    that is a measured property of Lowe's ratio test rather than a defect: a
    keypoint's best match against its own set is itself at distance 0, and the
    test keeps a match only when the best neighbour beats the second by the
    ratio. Where two keypoints carry the SAME descriptor -- which happens on
    repeated texture -- the second best is also 0, `0 < 0.75 * 0` is false, and
    both are dropped. Measured at 2.7% of keypoints on this fixture.

    So the assertion is what is true: a self-comparison is at the floor for four
    terms and within a small margin of it for the fifth, and it is strictly
    closer than the same vehicle photographed again -- which is the ordering the
    product actually depends on.
    """
    one = compute(a_vehicle(5), kind)
    itself = compare(one, one)
    again = compare(one, compute(the_same_vehicle_again(a_vehicle(5), 5), kind))
    for term in TERM_NAMES:
        assert itself[term].measurable, term
        if term == "structure":
            assert itself[term].value < 0.05, f"{term} floor is {itself[term].value}"
        else:
            assert itself[term].value == pytest.approx(0.0, abs=1e-6), term
        assert itself[term].value <= again[term].value, term


# --- the record ----------------------------------------------------------


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_the_descriptor_round_trips_exactly(kind):
    """Encode, decode, and get the same record back.

    `compute` deliberately returns the DECODED form of its own string, so the
    descriptor compared in memory is byte-for-byte the descriptor stored. Return
    the pre-encoding arrays instead and this goes red on the histogram, because
    quantisation is lossy -- which is the whole reason the round trip is there.
    """
    one = compute(a_vehicle(9), kind)
    again = decode(one.text)
    assert again.kind == one.kind and again.version == one.version
    assert np.array_equal(again.keypoints, one.keypoints)
    assert np.array_equal(again.grid, one.grid)
    assert np.allclose(again.histogram, one.histogram, atol=0, rtol=0)
    assert compare(one, again)["colour_bhattacharyya"].value == pytest.approx(0.0, abs=1e-6)


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_computing_the_same_image_twice_gives_the_same_descriptor(kind):
    """Deterministic, because a descriptor that drifts between two runs of the
    same build cannot be stored and compared later."""
    car = a_vehicle(13)
    assert compute_text(car, kind) == compute_text(car, kind)


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_the_descriptor_is_bounded_and_is_not_an_image(kind):
    """It carries a bounded amount of information, whatever the camera's
    resolution, and it is text.

    A descriptor whose size grew with the frame would be an image by another
    name. The same scene at 360 x 240 and at 2880 x 1920 produces records within
    the same bound, because everything is computed after a fixed resize.
    """
    small = a_vehicle(17)
    large = cv2.resize(small, (2880, 1920), interpolation=cv2.INTER_CUBIC)
    for image in (small, large):
        text = compute_text(image, kind)
        assert text.startswith(f"{DESCRIPTOR_PREFIX}/{DESCRIPTOR_VERSION}:")
        assert text.isascii()
        assert len(text) < 64_000, f"{len(text)} characters is not a compact record"


@guarantee
def test_the_version_is_readable_from_the_prefix_without_decoding():
    """A consumer holding a descriptor from a future build must be able to
    refuse it without trusting one byte of a payload it does not understand."""
    assert descriptor_version_of(compute_text(a_vehicle(2))) == DESCRIPTOR_VERSION
    assert descriptor_version_of(f"{DESCRIPTOR_PREFIX}/9:not-even-base64") == 9
    with pytest.raises(MalformedDescriptor):
        descriptor_version_of("just a string")


# --- the refusals --------------------------------------------------------


@guarantee
def test_two_descriptors_of_different_versions_refuse_to_compare():
    """FAIL-CONTROL 2 of the round: a descriptor compared against a different
    version.

    They do not compare "carefully" and they do not fall back to the terms both
    versions understand. The payload here is deliberately GARBAGE: if the
    version were checked after decoding, this would raise
    `MalformedDescriptor`, and the assertion on the type is what proves the
    refusal happens first.
    """
    from_the_future = f"{DESCRIPTOR_PREFIX}/{DESCRIPTOR_VERSION + 1}:zzzz-not-a-payload"
    ours = compute(a_vehicle(21))
    with pytest.raises(IncomparableDescriptors, match="different build"):
        compare(ours, from_the_future)
    with pytest.raises(IncomparableDescriptors):
        compare(from_the_future, ours)


@guarantee
def test_two_descriptors_of_different_kinds_refuse_to_compare():
    """An ORB structure distance and a SIFT one are different measurements under
    different metrics. Putting them on one axis makes the table an artefact of
    the detector rather than of the vehicle."""
    with pytest.raises(IncomparableDescriptors, match="kinds"):
        compare(compute(a_vehicle(23), ORB), compute(a_vehicle(23), SIFT))


@guarantee
@pytest.mark.parametrize(
    "text",
    [
        "opvid-fp/1:@@@not-base64@@@",
        "opvid-fp/1:",
        "some-other-scheme/1:aGVsbG8",
        "opvid-fp/x:aGVsbG8",
    ],
)
def test_a_malformed_descriptor_is_refused_rather_than_half_read(text):
    with pytest.raises((MalformedDescriptor, IncomparableDescriptors)):
        decode(text)


# --- the third outcome ---------------------------------------------------


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_a_frame_with_nothing_to_match_is_unmeasurable_not_distant(kind):
    """FAIL-CONTROL 3 of the round: a pair forced to the unmeasurable outcome.

    A flat frame yields no keypoints, so there is nothing to match -- which is
    NOT the same as nothing matching. Collapse the two and an unmeasurable
    same-car pair counts as a miss while an unmeasurable different-car pair
    counts as a correct reject: the error flatters the answer in both
    directions at once.

    Return a distance of 1.0 instead of `measurable=False` and this goes red on
    the first assertion.
    """
    flat = compute(a_flat_frame(), kind)
    car = compute(a_vehicle(29), kind)
    assert flat.keypoint_count < MEASURABLE_MIN_KEYPOINTS

    structure = compare(flat, car)["structure"]
    assert structure.measurable is False
    assert structure.value is None
    assert "nothing to match" in structure.reason


@guarantee
@pytest.mark.parametrize("kind", KINDS_UNDER_TEST)
def test_one_unmeasurable_term_does_not_make_the_others_unmeasurable(kind):
    """The three outcomes are PER TERM. A flat frame still has a colour and an
    edge grid, and refusing to publish those because the structure term could
    not be computed would throw away the evidence that exists."""
    result = compare(compute(a_flat_frame(), kind), compute(a_vehicle(31), kind))
    assert result["structure"].measurable is False
    for term in TERM_NAMES:
        if term != "structure":
            assert result[term].measurable is True, term


@guarantee
def test_a_term_cannot_be_unmeasurable_and_carry_a_value():
    """The invariant that keeps the third outcome from decaying back into two.

    `value is None` exactly when `measurable` is false, and an unmeasurable term
    must say why -- because "it could not be measured" with no reason is
    indistinguishable from a bug in the code that measured it.
    """
    with pytest.raises(ValueError, match="unmeasurable but carries"):
        TermDistance(term="structure", value=0.5, measurable=False, reason="x")
    with pytest.raises(ValueError, match="measurable with no value"):
        TermDistance(term="structure", value=None, measurable=True)
    with pytest.raises(ValueError, match="no reason given"):
        TermDistance(term="structure", value=None, measurable=False)


# --- the operating point -------------------------------------------------
#
# Distances, not verdicts. Where a threshold is wanted it comes from a chooser
# that takes its constraints EXPLICITLY -- neither has a default, because the
# number a barrier opens on is not something a helper function decides -- and
# refuses, naming the conflict, when nothing satisfies them.

#: Cleanly separated: every same-car distance below every different-car one.
SEPARATED_SAME = [0.10, 0.12, 0.15, 0.20, 0.22]
SEPARATED_DIFFERENT = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

#: Fully overlapping: the same values in both classes, so the descriptor
#: carries no information about identity at all.
MIXED_SAME = [0.10, 0.50, 0.90]
MIXED_DIFFERENT = [0.10, 0.50, 0.90, 0.10, 0.50, 0.90]


@guarantee
def test_the_chooser_chooses_when_a_threshold_satisfies_both_constraints():
    """The control for the control. Without it, "refuses" and "refuses
    everything" are the same test, and a chooser that never returns would
    pass every refusal assertion in this file."""
    chosen = choose_operating_point(
        SEPARATED_SAME, SEPARATED_DIFFERENT, max_false_match_rate=0.0, max_miss_rate=0.0
    )
    assert 0.22 <= chosen["threshold"] < 0.60
    assert chosen["miss_rate"] == 0.0 and chosen["false_match_rate"] == 0.0
    assert chosen["fitted_on"] == {"same_car": 5, "different_car": 8}


@guarantee
def test_the_chooser_refuses_and_names_the_conflict_from_both_sides():
    """FAIL-CONTROL 4 of the round.

    No threshold can both admit these same-car distances and exclude these
    different-car ones, because they are the same numbers. A refusal is the
    correct output -- and it names what each constraint could achieve ALONE, so
    a reader can see which one to move rather than being told only that it
    said no.
    """
    with pytest.raises(NoOperatingPoint) as refusal:
        choose_operating_point(
            MIXED_SAME, MIXED_DIFFERENT, max_false_match_rate=0.0, max_miss_rate=0.0
        )
    message = str(refusal.value)
    assert "no threshold satisfies both" in message
    assert "false matches at or below" in message
    assert "misses at or below" in message
    assert "3 same-car and 6 different-car" in message


@guarantee
def test_the_chooser_refuses_a_class_it_was_given_nothing_of():
    """A threshold fitted on one class is not a threshold. This is the shape a
    run with every different-car comparison unmeasurable would take, and
    returning "0.0, and it misses nothing" would be a perfect score over an
    empty measurement."""
    with pytest.raises(NoOperatingPoint, match="nothing to choose from"):
        choose_operating_point([0.1], [], max_false_match_rate=0.5, max_miss_rate=0.5)


@guarantee
def test_the_rates_move_the_way_the_search_assumes_they_do():
    """The chooser takes the LARGEST admissible threshold because false matches
    only ever increase with it and misses only ever decrease. That is an
    assumption about the arithmetic, so it is checked rather than believed."""
    thresholds = [i / 20 for i in range(21)]
    rates = [rates_at(t, SEPARATED_SAME, SEPARATED_DIFFERENT) for t in thresholds]
    misses = [m for m, _ in rates]
    false_matches = [f for _, f in rates]
    assert misses == sorted(misses, reverse=True)
    assert false_matches == sorted(false_matches)


@guarantee
def test_a_forced_match_makes_the_chooser_refuse():
    """FAIL-CONTROL 1 of the round: two different cars forced to match.

    The adversarial case stated as arithmetic. If every comparison came back at
    distance 0 -- a matcher that says every car is every other car -- then no
    threshold separates them, and the chooser must REFUSE rather than choose the
    one number that "works" on the same-car side.
    """
    everything_matches_same = [0.0] * 5
    everything_matches_different = [0.0] * 40
    with pytest.raises(NoOperatingPoint):
        choose_operating_point(
            everything_matches_same,
            everything_matches_different,
            max_false_match_rate=0.01,
            max_miss_rate=0.20,
        )


# --- the engine's injection point ----------------------------------------


@guarantee
def test_the_computer_produces_what_the_record_carries():
    """`DescriptorComputer` is the engine's injection point and its output is
    the string the contract's `identity.descriptor` holds -- not the decoded
    record, which is this module's business alone."""
    text = DescriptorComputer(ORB).compute(a_vehicle(37))
    assert isinstance(text, str)
    assert isinstance(decode(text), Descriptor)


@guarantee
def test_an_unknown_kind_is_refused_at_construction():
    with pytest.raises(ValueError, match="kind must be one of"):
        DescriptorComputer("akaze")


# --- the engine, which is where the field becomes reachable ---------------
#
# The descriptor is OFF by default and the engine's behaviour with it off is
# byte-for-byte what it was before this round. That is the claim, so it is the
# thing tested first: a component that changed how every existing deployment
# reads would not be additive whatever the schema version said.


class _StubRecogniser:
    """The engine's documented injection point: `.read(image) -> (text, conf)`."""

    def __init__(self, text="ABC1234", confidence=0.995) -> None:
        self._text, self._confidence = text, confidence

    def read(self, image):
        return self._text, self._confidence


def _a_capture(image: np.ndarray):
    from vehicle_id.contract import Capture

    ok, buf = cv2.imencode(".png", image)
    assert ok
    return Capture.now(buf.tobytes(), camera_id="lane-1")


def _an_engine(**kwargs):
    from vehicle_id.engine import PlateEngine

    return PlateEngine(recognizer=_StubRecogniser(), threshold=0.99, **kwargs)


@guarantee
@needs_engine
def test_the_engine_measures_no_descriptor_unless_it_is_asked_to():
    """NOT MEASURED, and it is null -- the rule every other component obeys.

    Most integrations today hand this engine a tight plate CROP; an appearance
    descriptor computed from one describes a plate, not a vehicle. Whether it is
    worth computing is a property of what the camera is pointed at, which only
    the operator knows, so the default is off.
    """
    read = _an_engine().read([_a_capture(a_vehicle(41))])
    assert read.identity.descriptor is None
    assert read.identity.plate == "ABC1234"
    assert read.is_answer


@guarantee
@needs_engine
def test_the_engine_carries_the_descriptor_when_one_is_injected():
    read = _an_engine(descriptor=DescriptorComputer(ORB)).read([_a_capture(a_vehicle(43))])
    assert read.identity.descriptor is not None
    assert decode(read.identity.descriptor).kind == ORB


@guarantee
@needs_engine
def test_the_descriptor_comes_from_the_capture_the_answer_came_from():
    """The plate and the descriptor on one record describe ONE photograph.

    Two captures, and the recogniser is confident only about the second. The
    record's `camera_id` already names the camera the answer came from; the
    descriptor has to agree with it, or the record describes two vehicles.
    """
    from vehicle_id.contract import Capture
    from vehicle_id.engine import PlateEngine

    class _SecondOneWins:
        def __init__(self) -> None:
            self.seen = 0

        def read(self, image):
            self.seen += 1
            return ("ABC1234", 0.995) if self.seen == 2 else ("ABC1234", 0.10)

    first, second = a_vehicle(45), a_vehicle(46)
    captures = [
        Capture.now(cv2.imencode(".png", first)[1].tobytes(), camera_id="staff"),
        Capture.now(cv2.imencode(".png", second)[1].tobytes(), camera_id="lane-1"),
    ]
    engine = PlateEngine(
        recognizer=_SecondOneWins(), threshold=0.99, descriptor=DescriptorComputer(ORB)
    )
    read = engine.read(captures)
    assert read.camera_id == "lane-1"
    assert read.identity.descriptor == compute_text(second, ORB)
    assert read.identity.descriptor != compute_text(first, ORB)


@guarantee
@needs_engine
def test_a_read_with_no_plate_still_carries_a_descriptor():
    """The whole point of the round: the plate is one component, not the identity.

    A vehicle whose plate cannot be read is still a vehicle that was
    photographed. The outcome is `fallback`, because nothing here stands behind
    an appearance match -- but the evidence is on the record rather than thrown
    away.
    """
    from vehicle_id.engine import PlateEngine

    engine = PlateEngine(
        recognizer=_StubRecogniser(text="", confidence=0.0),
        threshold=0.99,
        descriptor=DescriptorComputer(ORB),
    )
    read = engine.read([_a_capture(a_vehicle(47))])
    assert read.identity.plate is None
    assert read.identity.descriptor is not None
    assert read.outcome == "fallback"


@guarantee
@needs_engine
def test_no_descriptor_is_carried_when_nothing_was_there():
    """`presence is False` says the lane is visible and empty. The contract
    refuses an identity on such a record outright, so a descriptor computed
    anyway would make the engine unable to produce a record at all."""
    from vehicle_id.engine import PlateEngine
    from vehicle_id.presence import Presence

    class _NothingThere:
        def measure(self, images):
            return Presence(present=False, confidence=0.9, reason="empty lane")

    engine = PlateEngine(
        recognizer=_StubRecogniser(),
        threshold=0.99,
        presence=_NothingThere(),
        descriptor=DescriptorComputer(ORB),
    )
    read = engine.read([_a_capture(a_vehicle(49))])
    assert read.presence is False
    assert read.identity.descriptor is None
    assert read.identity.plate is None


class _Throws:
    """One way an injected computer fails: it raises."""

    def compute(self, image):
        raise RuntimeError("no")


class _ReturnsBytes:
    """The other way, and it is the one that got through.

    `.compute(image) -> str` is a PUBLISHED shape, not an enforced one, so a
    third party's computer can return anything. A non-string used to travel
    past `_describe` -- which caught only exceptions -- into `Identity(...)`,
    where the contract's type check raised `ValueError` out of `read()`, a
    method whose own docstring promises it answers rather than raising. The
    engine was only ever this honest about the failures it had imagined.
    """

    def compute(self, image):
        return b"opvid-fp/1:bytes-not-str"


class _ReturnsNone:
    """A computer that answers "I could not", off the published shape."""

    def compute(self, image):
        return None


@guarantee
@needs_engine
@pytest.mark.parametrize(
    "descriptor",
    [
        pytest.param(_Throws(), id="throws"),
        pytest.param(_ReturnsBytes(), id="returns-bytes"),
        pytest.param(_ReturnsNone(), id="returns-none"),
        # Not a computer at all: an operator who read `descriptor=` as naming a
        # kind. It has no `.compute`, so it arrives by the same first branch,
        # and the point is that it produces a record rather than an exception.
        pytest.param("orb", id="not-a-computer-at-all"),
    ],
)
def test_a_descriptor_that_cannot_be_computed_is_null_and_not_an_exception(descriptor):
    """A component, not the answer. The engine's promise is that no ORDINARY
    failure inside `read` raises instead of answering, and no descriptor
    computer must become the first one. The promise is not stated over
    `BaseException` and this test does not assert one: `SystemExit` and
    `KeyboardInterrupt` propagate by design, which `read()`'s docstring says.

    Parametrised over BOTH axes that reach the guard, because it previously
    varied only one of them: the fixture built a computer that THREW, the
    docstring stated the general claim, and the type failure -- the one that
    actually broke the promise -- was never constructed. A fixture is part of
    the measurement, and it has to vary every axis the decision branches on.
    """
    read = _an_engine(descriptor=descriptor).read([_a_capture(a_vehicle(51))])
    assert read.identity.descriptor is None
    assert read.identity.plate == "ABC1234"
    assert read.is_answer
