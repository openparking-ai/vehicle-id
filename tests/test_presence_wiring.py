"""Is the presence gate CONNECTED? Answered without a trained model.

This file exists because the tests that proved the gate was wired into the
engine all skipped in CI. They guarded on `clean_confidence < TRAINED`, CI
trains a 600-step model that is nowhere near it, and so three guarantees never
executed anywhere automated while the build stayed green. CI green proved the
gate was importable.

Whether the gate is connected and whether the recogniser is accurate are two
different questions. The second needs weights. The first does not need any: the
seam is the RECOGNISER, and a stub at a fixed confidence answers it completely.
Accuracy lives in `test_plates.py` and may skip; this may not.

`PlateEngine` refuses to construct on weights whose operating point nobody has
measured, which is correct and is why every engine below passes an explicit
`threshold=` -- the operating point is being SUPPLIED here, not measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

import cv2  # noqa: E402

from lanes import flat, lane, vehicle  # noqa: E402
from vehicle_id.contract import Capture  # noqa: E402
from vehicle_id.engine import PlateEngine  # noqa: E402
from vehicle_id.presence import PresenceDetector  # noqa: E402

guarantee = pytest.mark.guarantee

CONFIDENT = 0.999
THRESHOLD = 0.99

#: Deliberately a path that does not exist. Nothing in this file needs weights,
#: and naming a real file would let these tests start depending on one without
#: anybody noticing -- which is how the previous set ended up unable to run.
NO_WEIGHTS = Path("models/there-are-deliberately-no-weights-here.pt")


class StubRecognizer:
    """Always reads the same plate, at a confidence the test chooses.

    It also counts. "The gate stopped the read" and "the recogniser happened to
    return nothing" produce the same record, and only one of them is the
    guarantee -- so the calls are counted rather than inferred from the output.
    """

    def __init__(self, text: str = "7ABC123", confidence: float = CONFIDENT) -> None:
        self.text = text
        self.confidence = confidence
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return self.text, self.confidence


def capture(image, camera_id: str = "lane-1-entry") -> Capture:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return Capture.now(buf.tobytes(), camera_id=camera_id)


def engine_with(detector, recognizer=None):
    return PlateEngine(
        NO_WEIGHTS,
        threshold=THRESHOLD,
        presence=detector,
        recognizer=recognizer if recognizer is not None else StubRecognizer(),
    )


@pytest.fixture
def gate():
    return PresenceDetector(reference=lane(90, seed=1))


# --- the gate is in the path, and it decides before anything is read ------

@guarantee
def test_an_empty_lane_stops_the_read_before_it_happens(gate):
    """D3. Not "the recogniser found nothing" -- the recogniser was never asked.

    This is the whole point of ordering presence first: the recogniser has no
    rejection stage and reads text out of an empty frame, so the only way to not
    get a plate out of an empty lane is to not run it.
    """
    recognizer = StubRecognizer()
    engine = engine_with(gate, recognizer)

    read = engine.read([capture(lane(90, seed=30))])

    assert read.presence is False
    assert read.identity.plate is None
    assert read.outcome == "fallback"
    assert recognizer.calls == 0, "the recogniser ran on a lane the gate said was empty"


@guarantee
def test_a_vehicle_is_read_normally(gate):
    """The control that matters more. A gate that stopped every read would pass
    the test above and break the product."""
    recognizer = StubRecognizer()
    engine = engine_with(gate, recognizer)

    read = engine.read([capture(vehicle(420, 240, seed=31))])

    assert read.presence is True
    assert read.identity.plate == "7ABC123"
    assert read.is_answer
    assert recognizer.calls == 1


@guarantee
def test_removing_the_gate_changes_the_answer_on_the_same_captures(gate):
    """The wiring assertion proper. Same captures, same recogniser, same
    threshold; the ONLY difference is whether the detector is attached.

    Without this, every assertion above is also satisfied by an engine that
    ignores the gate and happens to agree with it.
    """
    captures = [capture(lane(90, seed=32))]

    gated = engine_with(gate, StubRecognizer())
    ungated = engine_with(None, StubRecognizer())

    assert gated.read(captures).presence is False
    assert gated.read(captures).identity.plate is None

    loose = ungated.read(captures)
    assert loose.presence is None, "presence must be NOT MEASURED with no detector"
    assert loose.identity.plate == "7ABC123", (
        "with the gate removed these captures must still produce a plate -- if "
        "they do not, the test above proves nothing about the gate"
    )
    assert loose.is_answer


# --- a camera fault is not a plate to read, and not "no vehicle" either ---

@guarantee
@pytest.mark.parametrize("level", [0, 255])
def test_a_dead_camera_reads_nothing_and_claims_nothing(gate, level):
    """The record says NOT MEASURED, and no plate comes out of it.

    Both halves matter and they are different claims. `presence: null` is
    honest: nobody can see the lane. Reading anyway would not be -- the
    recogniser returns text for a flat frame, and 1.9% of single frames clear
    the measured operating point, so a dead camera would mint confident plates.
    """
    recognizer = StubRecognizer()
    engine = engine_with(gate, recognizer)

    read = engine.read([capture(flat(level))])

    assert read.presence is None
    assert read.identity.plate is None
    assert read.outcome == "fallback"
    assert recognizer.calls == 0, "a plate was read out of a blank frame"


def test_a_camera_fault_is_not_reported_as_an_empty_lane(gate):
    """The distinction the lane acts on. `false` ends the transaction; `null`
    sends it to a human. A dead camera must produce the second."""
    engine = engine_with(gate, StubRecognizer())
    assert engine.read([capture(flat(0))]).presence is not False


# --- presence travels on the record, all the way out ----------------------

@guarantee
def test_presence_reaches_the_record_and_survives_the_round_trip(gate):
    from vehicle_id.contract import Read

    engine = engine_with(gate, StubRecognizer())
    read = engine.read([capture(vehicle(420, 240, seed=33))])

    payload = read.to_dict()
    assert payload["presence"] is True
    assert 0.0 <= payload["presence_confidence"] <= 1.0
    assert Read.from_dict(payload) == read


@guarantee
def test_a_record_can_never_carry_a_plate_with_presence_false(gate):
    """D2 at the engine rather than at the contract. The contract refuses such a
    record outright; this proves the engine cannot be talked into building one,
    with a recogniser that would happily supply the plate.
    """
    engine = engine_with(gate, StubRecognizer(text="SHOULDNOT", confidence=1.0))
    read = engine.read([capture(lane(90, seed=34))])
    assert read.presence is False
    assert read.identity.plate is None
    assert not read.is_answer
