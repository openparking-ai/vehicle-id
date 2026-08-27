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


# --- G4c: the gate MOVES the number, proven without weights ---------------

@guarantee
def test_the_gate_moves_the_noise_answer_rate_with_no_weights_at_all():
    """D6's wiring half, and the resolution of a conflict in the brief.

    The accuracy version of this lives in `test_plates.py`, needs weights that
    actually read plates, and is allowed to skip in the engine job. That made
    the control which proves the gate does anything the one thing that never
    ran in CI -- and the job that DID publish the number trained a 600-step
    model which answers zero on noise, so its ungated control could not have
    been satisfied either. A comparison of 0.0% against 0.0% is not evidence.

    Whether the gate MOVES the number needs no model. The recogniser is a seam;
    a stub that always answers confidently is the worst case a gate could face,
    and it makes the ungated arm answer 100% by construction rather than by
    luck. That removes the reason this ever skipped.

    Both arms are measured in the same run, so neither number is assumed.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    batches = []
    for _ in range(60):
        frame = rng.integers(0, 255, (160, 320, 3), dtype=np.uint8)
        batches.append([capture(frame, camera_id="dead-feed")])

    gated = engine_with(PresenceDetector(reference=lane(90, seed=1)), StubRecognizer())
    ungated = engine_with(None, StubRecognizer())

    ungated_answers = sum(ungated.read(b).is_answer for b in batches)
    gated_answers = sum(gated.read(b).is_answer for b in batches)

    # The control, measured rather than assumed. If the ungated arm answered
    # zero the assertion below would pass with the gate removed.
    assert ungated_answers == len(batches), (
        f"the ungated control answered {ungated_answers}/{len(batches)}; with a "
        "stub that always answers confidently it must answer every one, or the "
        "assertion below proves nothing"
    )
    assert gated_answers == 0, (
        f"{gated_answers}/{len(batches)} dead-feed reads got past the presence "
        f"gate (ungated control: {ungated_answers}/{len(batches)})"
    )


@guarantee
def test_the_gate_does_not_move_the_number_by_refusing_everything():
    """The control that matters more, again. A gate that stopped every read
    would post a perfect noise score and break the product."""
    gated = engine_with(PresenceDetector(reference=lane(90, seed=1)), StubRecognizer())
    read = gated.read([capture(vehicle(420, 240, seed=44))])
    assert read.presence is True
    assert read.identity.plate == "7ABC123"
    assert read.is_answer


# --- K1b: the disclosure is at the SEAM, and something checks that ---------

@guarantee
def test_the_health_endpoint_states_that_presence_is_unvalidated_and_why():
    """H4's standard, as a check rather than a convention.

    Two documents nobody reads is not the acceptance. `GET /v1/health` is where
    an operator looks, and a capability reported without its status reads as a
    validated one -- so the endpoint carries the disclosure and its named
    limitations, and this is what stops a refactor dropping them silently.

    The previous round put the string in and nothing asserted it. That is the
    same shape as a guarantee that skips: present, plausible, and unproven.
    """
    from vehicle_id.presence import CAMERA_FAULTS_CAVEAT, KNOWN_LIMITS, UNVALIDATED
    from vehicle_id.service import VehicleIdService

    service = VehicleIdService(engine_with(PresenceDetector(reference=lane(90, seed=1)),
                                           StubRecognizer()))
    health = service.health()
    assert health["presence_gate"] is True
    assert health["presence_validation"] == UNVALIDATED
    assert "UNVALIDATED" in health["presence_validation"]
    assert health["presence_limits"] == list(KNOWN_LIMITS)
    assert health["presence_limits"], "the endpoint names no limitations at all"
    # X3a. The count and what it does not mean, in the same payload. An operator
    # reading `camera_faults` is the person who would send a technician, and
    # `reference_not_recognised` also covers heavy weather and an ordinary
    # arrival on low-texture ground under a beam pool.
    assert "camera_faults" in health
    assert health["camera_faults_caveat"] == CAMERA_FAULTS_CAVEAT


# The coverage control for the disclosure -- "every limitation the measurement
# found is named at the seam" -- is
# `tests/test_measured_docs.py::test_every_measured_limitation_is_named_at_the_seam`.
# It lived here and iterated a hard-coded 5-tuple of topic words while promising
# it would go red when a measured limitation was added; it could not, and the
# conflation sat unnamed at the seam through a whole round with this green. It
# derives from `docs/measured/presence.json` now, and it lives in a file with no
# cv2 import so it runs in the job that proves the contract stands alone too.


@guarantee
def test_the_cli_says_it_at_the_moment_somebody_switches_the_gate_on(tmp_path, capsys):
    """The other seam. The contract and the README both carry this, and neither
    is read by the person typing `--empty-lane` at 6am.

    Presence is off by default precisely so that turning it on is a decision,
    and a decision nobody was told about is not one.
    """
    import argparse

    import cv2

    from vehicle_id.cli import _presence
    from vehicle_id.presence import KNOWN_LIMITS

    reference = tmp_path / "empty-lane.png"
    cv2.imwrite(str(reference), lane(90, seed=1))
    detector = _presence(
        argparse.Namespace(empty_lane=reference, min_occupancy=0.15)
    )
    assert detector is not None and detector.has_reference

    said = capsys.readouterr().err
    assert "UNVALIDATED" in said, "the CLI does not say the gate is unvalidated"
    for limit in KNOWN_LIMITS:
        assert limit in said, f"the CLI does not state the limitation: {limit[:50]}..."


@guarantee
def test_the_cli_says_nothing_when_the_gate_is_left_off(capsys):
    """The control. A warning printed unconditionally is noise, and noise is
    what an operator learns to skip past -- which would defeat the seam."""
    import argparse

    from vehicle_id.cli import _presence

    assert _presence(argparse.Namespace(empty_lane=None, min_occupancy=0.15)) is None
    assert capsys.readouterr().err == "", (
        "the CLI warns about a gate that is not switched on"
    )
