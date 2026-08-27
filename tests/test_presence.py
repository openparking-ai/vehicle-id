"""The presence gate: is a VEHICLE there, asked before anything is read.

The distinction these tests exist to hold: a car with an unreadable plate is a
legitimate entry and must be admitted to fallback; a metal object on the loop is
not a vehicle and must receive nothing. A gate that cannot tell those apart is
worse than no gate, because it refuses honest customers while still admitting
the fraud.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vehicle_id.presence import PresenceDetector  # noqa: E402

W, H = 640, 360


def empty_lane():
    """Tarmac: a flat mid-grey with a little texture, so the reference is not a
    degenerate constant image that any difference would light up."""
    rng = np.random.default_rng(1)
    lane = np.full((H, W, 3), 90, np.uint8)
    noise = rng.integers(-6, 6, (H, W, 3)).astype(np.int16)
    return np.clip(lane.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def with_object(width, height, colour=200):
    """The empty lane with one solid rectangle on it."""
    scene = empty_lane().copy()
    x0 = (W - width) // 2
    y0 = (H - height) // 2
    cv2.rectangle(scene, (x0, y0), (x0 + width, y0 + height), (colour,) * 3, -1)
    return scene


@pytest.fixture
def detector():
    return PresenceDetector(reference=empty_lane())


# --- the two cases the requirement is stated in --------------------------

def test_a_vehicle_sized_object_is_present(detector):
    result = detector.measure([with_object(420, 240)])
    assert result.present is True
    assert result.occupancy > 0.15


def test_a_hand_held_metal_plate_is_not_a_vehicle(detector):
    """A person on foot holding something over the loop. Roughly plate-sized
    against a lane-filling camera: about 1% of the frame."""
    result = detector.measure([with_object(80, 40)])
    assert result.present is False
    assert result.occupancy < 0.15


def test_the_empty_lane_is_not_a_vehicle(detector):
    assert detector.measure([empty_lane()]).present is False


# --- not measured is a third state, and it is not "no" --------------------

def test_without_a_reference_presence_is_not_measured_rather_than_false():
    """The failure this prevents is a lane that refuses every customer because
    nobody configured a reference view. A field that was not measured is null --
    the same rule the rest of the contract obeys."""
    result = PresenceDetector().measure([with_object(420, 240)])
    assert result.present is None
    assert result.confidence is None
    assert "reference" in result.reason


def test_undecodable_captures_are_not_measured_rather_than_absent(detector):
    assert detector.measure([None]).present is None


# --- what the gate must not do -------------------------------------------

def test_scattered_change_is_not_a_vehicle(detector):
    """Rain, snow, sensor noise: a large fraction of the frame changes, but not
    contiguously. Only the largest connected region counts, which is what
    separates a car from weather."""
    rng = np.random.default_rng(2)
    speckled = empty_lane().copy()
    mask = rng.random((H, W)) < 0.45
    speckled[mask] = 255
    result = detector.measure([speckled])
    assert result.present is False, (
        f"speckle covering ~45% of the frame read as a vehicle ({result.occupancy:.1%})"
    )


def test_the_largest_capture_in_a_burst_decides(detector):
    """A vehicle may only be fully in frame for part of the burst. A gate that
    averaged would refuse a car that arrived halfway through."""
    result = detector.measure([empty_lane(), empty_lane(), with_object(420, 240)])
    assert result.present is True


def test_the_occupancy_is_reported_whichever_way_the_answer_goes(detector):
    """An operator debugging a gate that refuses everything needs the number,
    not just the verdict."""
    for scene in (with_object(420, 240), with_object(80, 40)):
        assert detector.measure([scene]).occupancy is not None


def test_the_threshold_is_configurable_and_actually_applied():
    """The occupancy threshold is an ASSUMPTION, not a measurement -- it cannot
    be measured without lane footage. So it must be reachable, and a test that
    moves it must change the answer, or it is decoration."""
    scene = [with_object(200, 120)]
    lenient = PresenceDetector(reference=empty_lane(), min_occupancy=0.05)
    strict = PresenceDetector(reference=empty_lane(), min_occupancy=0.60)
    assert lenient.measure(scene).present is True
    assert strict.measure(scene).present is False
