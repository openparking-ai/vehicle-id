"""The presence gate: is a VEHICLE there, asked before anything is read.

The distinction these tests exist to hold: a car with an unreadable plate is a
legitimate entry and must be admitted to fallback; a metal object on the loop is
not a vehicle and must receive nothing. A gate that cannot tell those apart is
worse than no gate, because it refuses honest customers while still admitting
the fraud.

And the distinction the second version exists to hold: **`False` is a
measurement, not a fallback value.** It says the lane is visible, it matches the
reference, and there is nothing on it -- the only claim that ends a transaction
before it starts. Everything the gate cannot see is `None`. The first version
answered `True` at confidence 1.0 for a dead camera and for an empty lane at
dusk, which is the same error in the other direction: a verdict where there was
no measurement.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")


from lanes import dead_sensor, flat, lane, rain, sensor_noise, vehicle  # noqa: E402
from vehicle_id.presence import (  # noqa: E402
    NO_SIGNAL,
    REFERENCE_NOT_RECOGNISED,
    SCENE_CHANGED,
    PresenceDetector,
)

guarantee = pytest.mark.guarantee


@pytest.fixture
def detector():
    return PresenceDetector(reference=lane(90, seed=1))


# --- the two cases the requirement is stated in --------------------------

@guarantee
def test_a_vehicle_sized_object_is_present(detector):
    result = detector.measure([vehicle(420, 240, seed=2)])
    assert result.present is True
    assert result.occupancy > 0.15


@guarantee
def test_a_hand_held_metal_plate_is_not_a_vehicle(detector):
    """A person on foot holding something over the loop. Roughly plate-sized
    against a lane-filling camera: about 1% of the frame."""
    result = detector.measure([vehicle(80, 40, seed=3)])
    assert result.present is False
    assert result.occupancy < 0.15


@guarantee
def test_the_empty_lane_is_not_a_vehicle(detector):
    assert detector.measure([lane(90, seed=4)]).present is False


# --- not measured is a third state, and it is not "no" --------------------

def test_without_a_reference_presence_is_not_measured_rather_than_false():
    """The failure this prevents is a lane that refuses every customer because
    nobody configured a reference view. A field that was not measured is null --
    the same rule the rest of the contract obeys."""
    result = PresenceDetector().measure([vehicle(420, 240)])
    assert result.present is None
    assert result.confidence is None
    assert "reference" in result.reason


def test_undecodable_captures_are_not_measured_rather_than_absent(detector):
    assert detector.measure([None]).present is None


# --- B1a: a frame carrying no information can never produce a verdict -----

@guarantee
@pytest.mark.parametrize("level", [0, 255])
def test_a_blank_frame_is_a_camera_fault_not_a_vehicle(detector, level):
    """L3's first probe. A dead or covered camera (flat black) and a blown
    exposure (flat white) both used to report `presence=true` at confidence
    1.0, with nothing in frame -- the raw-intensity difference against a lit
    reference filled the whole frame, and a filled frame was read as a car.

    It is not a vehicle. It is also not "no vehicle": nobody can see the lane.
    """
    result = detector.measure([flat(level)])
    assert result.present is None, "a blank frame produced a verdict"
    assert result.camera_health == NO_SIGNAL
    assert result.confidence is None


@guarantee
def test_no_single_guard_is_what_keeps_a_blank_frame_from_answering(detector):
    """F3, and what actually holds the ordering up.

    The concern F3 raised is real in shape: cancelling illumination makes a flat
    frame agree with a lit reference, so a blank-frame check placed AFTER the
    fit would let a dead camera report a confident `False` -- worse than the
    `True` being fixed, because at the lane `False` ends the transaction.

    What is measured here is that it takes more than one guard away to get
    anywhere near that. B1a names the fault first; with B1a defeated the
    illumination fit refuses the frame outright, because a constant image has no
    gain that maps it onto a lit reference. So the answer stays `None` and only
    the REASON changes.

    The control is the second half: the same defeated detector still answers
    `False` for a lane it can genuinely see, so "None everywhere" is not this
    test passing because the detector stopped working.
    """
    blank = flat(0)
    named = detector.measure([blank])
    assert named.present is None
    assert named.camera_health == NO_SIGNAL

    defeated = PresenceDetector(reference=lane(90, seed=1), min_frame_std=0.0)
    fallen_through = defeated.measure([blank])
    assert fallen_through.present is None, (
        "with the blank-frame check defeated a dead camera produced a verdict"
    )
    assert fallen_through.camera_health == REFERENCE_NOT_RECOGNISED, (
        "the second guard is not the one catching it; the reason has moved"
    )

    # The control. A defeated detector that refused everything would pass the
    # assertions above while proving nothing.
    assert defeated.measure([lane(90, seed=23)]).present is False


# --- B1b: a change in the LIGHT is not a change in the SCENE --------------

@guarantee
@pytest.mark.parametrize("level", [25, 40, 55, 70, 110, 140, 180, 220])
def test_the_same_empty_lane_under_different_light_is_still_empty(detector, level):
    """L3's second and third probes. Against a daylight reference, dusk read as
    a vehicle filling 82.2% of the frame and sun on tarmac as 100%, both with
    nothing in the picture, because the comparison was raw intensity and the
    fixed 30-level step turned a global change into a contiguous blob.

    A global gain and offset are fitted and cancelled first, so what is left is
    a change in the scene.
    """
    result = detector.measure([lane(level, seed=10 + level)])
    assert result.present is False, (
        f"an empty lane at light level {level} against a reference at 90 read as "
        f"{result.present} ({result.occupancy})"
    )


def test_the_illumination_fit_does_not_hide_a_vehicle(detector):
    """The control for the test above. A fit that cancelled everything would
    make every scene match and report an empty lane forever."""
    assert detector.measure([vehicle(420, 240, level=55, seed=11)]).present is True
    assert detector.measure([vehicle(420, 240, level=170, seed=12)]).present is True


# --- F2: the whole frame changing is NOT MEASURED, never "no vehicle" -----

@guarantee
def test_a_view_that_no_longer_matches_the_reference_is_not_measured(detector):
    """F2, and the reason the ceiling does not answer `False`.

    A vehicle close enough to fill the view, a camera that was knocked, a scene
    rebuilt overnight: indistinguishable from here. An upper bound that answered
    `False` would turn the van, the truck and the bus that legitimately fill an
    entry lane view into "no vehicle" -- and at the lane that is no ticket, no
    session and no vend for a customer who is really there.
    """
    result = detector.measure([vehicle(636, 356, seed=13)])
    assert result.present is None, "a frame-filling vehicle was called an empty lane"
    assert result.camera_health in (REFERENCE_NOT_RECOGNISED, SCENE_CHANGED)


@guarantee
def test_a_dead_feed_is_a_camera_fault_not_a_verdict(detector):
    """Uniform sensor noise. Not a vehicle, and not an empty lane either.

    Caught as a frame with no picture in it rather than as a frame that does not
    match the reference, and the distinction is load-bearing: a tight plate crop
    does not match a lane reference either, and a caller replacing an LPR unit
    sends nothing else. See the test below.
    """
    result = detector.measure([sensor_noise()])
    assert result.present is None
    assert result.camera_health == NO_SIGNAL


@guarantee
def test_a_real_picture_that_is_not_this_lane_is_not_called_a_dead_feed(detector):
    """The control for the test above, and the reason it is two checks and not
    one. A frame can carry a perfectly good picture and still fail to match the
    reference -- a plate crop, a camera someone knocked. That is NOT MEASURED,
    but it is not an equipment fault, and the engine goes on reading it.
    """
    from vehicle_id.plates.generator import PlateGenerator

    crop = PlateGenerator(seed=11).sample(degradation=0).image
    result = detector.measure([crop])
    assert result.present is None
    assert result.camera_health == REFERENCE_NOT_RECOGNISED, (
        "a plate crop was classified as a dead camera; every caller replacing an "
        "LPR unit would stop getting reads"
    )


def test_a_frame_filling_object_never_reads_as_absent(detector):
    """The property F2 exists for, stated as its own assertion: whatever else
    the gate does with a large object, it must never call it an empty lane."""
    for width in range(400, 641, 20):
        result = detector.measure([vehicle(width, 340, seed=14)])
        assert result.present is not False, (
            f"an object {width}px wide was reported as an empty lane"
        )


# --- what the gate must not do -------------------------------------------

@guarantee
def test_scattered_change_is_not_a_vehicle(detector):
    """Rain or snow: a large fraction of the frame changes, but not contiguously.
    Only the largest connected region counts, which is what separates a car from
    weather.

    Still `False`, not `None`. Rain fragments the picture but leaves most of the
    lane matching the reference and the frame is still a picture, so the lane IS
    visible and it IS empty. That is a measurement, and it is the one that keeps
    a lane working in bad weather instead of falling back all afternoon.
    """
    result = detector.measure([rain(0.45)])
    assert result.present is False, (
        f"rain across ~45% of the frame read as {result.present} "
        f"({result.occupancy})"
    )


@guarantee
def test_a_sensor_dropping_pixels_is_a_fault_not_weather(detector):
    """Salt noise across half the frame is not rain -- no camera emits it. It is
    a failing sensor, and it lands in the camera-fault case rather than being
    reported as a lane somebody can see."""
    result = detector.measure([dead_sensor(0.45)])
    assert result.present is None
    assert result.camera_health == NO_SIGNAL


def test_the_largest_capture_in_a_burst_decides(detector):
    """A vehicle may only be fully in frame for part of the burst. A gate that
    averaged would refuse a car that arrived halfway through."""
    result = detector.measure([lane(90, seed=15), lane(90, seed=16), vehicle(420, 240, seed=17)])
    assert result.present is True


def test_a_blank_capture_does_not_veto_a_burst_that_can_see(detector):
    """One dropped frame in a burst is not a camera fault. The captures that
    carry a picture still decide."""
    result = detector.measure([flat(0), vehicle(420, 240, seed=18)])
    assert result.present is True


def test_the_occupancy_is_reported_whichever_way_the_answer_goes(detector):
    """An operator debugging a gate that refuses everything needs the number,
    not just the verdict."""
    for scene in (vehicle(420, 240, seed=19), vehicle(80, 40, seed=20)):
        assert detector.measure([scene]).occupancy is not None


def test_the_threshold_is_configurable_and_actually_applied():
    """The occupancy threshold is an ASSUMPTION, not a measurement -- it cannot
    be measured without lane footage. So it must be reachable, and a test that
    moves it must change the answer, or it is decoration."""
    scene = [vehicle(200, 120, seed=21)]
    lenient = PresenceDetector(reference=lane(90, seed=1), min_occupancy=0.05)
    strict = PresenceDetector(reference=lane(90, seed=1), min_occupancy=0.60)
    assert lenient.measure(scene).present is True
    assert strict.measure(scene).present is False


def test_a_threshold_pair_that_cannot_be_satisfied_is_refused():
    with pytest.raises(ValueError, match="min_occupancy"):
        PresenceDetector(reference=lane(), min_occupancy=0.9, max_occupancy=0.5)


# --- F5: the confidence arithmetic said one thing and did another ---------

@guarantee
def test_an_entirely_empty_lane_reads_as_certain(detector):
    """F5. The first version divided both sides of the boundary by the wider of
    the two, which capped a `False` verdict at min_occupancy/(1 - min_occupancy)
    -- 0.176 at the default. So a completely empty lane reported 17.6% confident
    that it was empty, three lines under a comment saying such a scene should
    read as certain. The comment was right and the arithmetic was wrong.
    """
    result = detector.measure([lane(90, seed=22)])
    assert result.present is False
    assert result.confidence == pytest.approx(1.0, abs=0.02), (
        f"an entirely unchanged frame reported {result.confidence} confident"
    )


def test_confidence_reaches_zero_at_the_boundary_from_both_sides():
    """Continuity, which is what F4 asks for: the verdict flips at the boundary
    but the number does not jump across it."""
    d = PresenceDetector(reference=lane(90, seed=1))
    assert d._confidence(d.min_occupancy) == pytest.approx(0.0)
    assert d._confidence(d.min_occupancy - 1e-9) == pytest.approx(0.0, abs=1e-6)


@guarantee
def test_confidence_degrades_rather_than_jumping_through_the_transition():
    """F4. Sweep an object through the decision boundary and require the
    confidence to move in steps, not in one leap. A boolean has a cliff at its
    boundary by definition -- that is what a boolean is -- so what is asserted
    is that the NUMBER reported beside it is continuous.
    """
    d = PresenceDetector(reference=lane(90, seed=1))
    seen = []
    for index, width in enumerate(range(120, 481, 10)):
        result = d.measure([vehicle(width, 240, seed=800 + index)])
        if result.occupancy is not None:
            seen.append((result.occupancy, result.confidence))
    seen.sort()
    steps = [abs(seen[i][1] - seen[i - 1][1]) for i in range(1, len(seen))]
    assert max(steps) < 0.25, f"confidence jumped by {max(steps):.3f} between adjacent samples"
