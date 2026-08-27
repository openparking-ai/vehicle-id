"""The contract's own guarantees, tested without the engine.

These run with no torch and no OpenCV installed, which is the point: a consumer
integrating against this product needs the contract and nothing else, so the
contract has to be testable on its own. If this file ever needs the engine to
run, the contract has stopped being a contract.
"""

from __future__ import annotations

import json

import pytest

from vehicle_id.contract import (
    ANSWER,
    FALLBACK,
    SCHEMA_VERSION,
    Capture,
    Engine,
    Identity,
    Read,
    utc_now,
)


def a_read(**overrides) -> Read:
    base = dict(
        read_id="r1",
        captured_at=utc_now(),
        camera_id="lane-1",
        identity=Identity(plate="ABC123"),
        confidence=0.995,
        engine=Engine(name="test", version="0.1.0", weights_id="sha256:deadbeef"),
        threshold_applied=0.99,
        outcome=ANSWER,
    )
    base.update(overrides)
    return Read(**base)


def test_unmeasured_fields_are_null_not_absent_and_not_invented():
    identity = Identity(plate="ABC123")
    assert identity.make is None
    assert identity.model is None
    assert identity.color is None
    assert identity.plate_region is None
    # Empty means "none were measured", not "the vehicle had none".
    assert identity.marks == ()


def test_fallback_is_a_first_class_outcome_not_an_error():
    read = a_read(outcome=FALLBACK, confidence=0.4)
    assert not read.is_answer
    # It round-trips, transports and stores exactly like an answer does.
    assert Read.from_dict(read.to_dict()) == read


def test_an_unknown_outcome_is_refused_rather_than_carried():
    with pytest.raises(ValueError, match="outcome must be one of"):
        a_read(outcome="error")


def test_confidence_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match="confidence"):
        a_read(confidence=1.4)


def test_the_record_round_trips_through_json():
    read = a_read(identity=Identity(plate="XY 1234", marks=("dent",)))
    restored = Read.from_dict(json.loads(json.dumps(read.to_dict())))
    assert restored == read
    assert restored.identity.marks == ("dent",)


def test_a_record_from_a_future_schema_is_refused_not_guessed_at():
    payload = a_read().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="unsupported schema_version"):
        Read.from_dict(payload)


def test_the_threshold_travels_with_the_record():
    # A consumer must be able to see the operating point that produced the
    # outcome without knowing it out of band.
    assert "threshold_applied" in a_read().to_dict()


def test_redacted_keeps_the_outcome_and_drops_the_identity():
    read = a_read(identity=Identity(plate="ABC123", marks=("dent",)))
    safe = read.redacted()
    assert safe.identity.plate is None
    assert safe.identity.marks == ()
    assert safe.outcome == read.outcome
    assert safe.confidence == read.confidence
    assert safe.read_id == read.read_id


def test_captured_at_carries_an_offset():
    # A naive timestamp is the bug that surfaces months later, when a lane in
    # one timezone and a consumer in another disagree about when a car arrived.
    stamp = Capture.now(b"x", camera_id="lane-1").captured_at
    assert stamp.endswith("+00:00") or stamp.endswith("Z")


# --- the compatibility promise, which the parser has to actually keep -----

def test_an_added_field_is_ignored_rather_than_rejected():
    """`docs/CONTRACT.md`: additive changes do not bump `schema_version`, and a
    consumer ignores what it does not recognise.

    The day the make/model/colour slice adds a field to `identity` -- which the
    contract explicitly authorises without a bump -- a parser that refused it
    would break every consumer built on this one.
    """
    payload = a_read().to_dict()
    payload["identity"]["body_type"] = "sedan"
    payload["engine"]["runtime"] = "onnx"
    payload["captured_by"] = "a field nobody has invented yet"

    restored = Read.from_dict(payload)
    assert restored.identity.plate == "ABC123"
    assert restored.engine.name == "test"


def test_ignoring_unknown_fields_does_not_mean_ignoring_a_version_bump():
    """The control for the test above.

    Tolerance for new fields must not become tolerance for a record this build
    cannot read. If both were true, the version would mean nothing.
    """
    payload = a_read().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    payload["identity"]["body_type"] = "sedan"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        Read.from_dict(payload)


def test_a_missing_required_field_is_still_an_error():
    """The other control: dropping unknown fields must not quietly become
    dropping the ones that carry meaning."""
    payload = a_read().to_dict()
    del payload["confidence"]
    with pytest.raises(KeyError):
        Read.from_dict(payload)


# --- presence, and the invariant it exists for ----------------------------

def test_presence_false_cannot_carry_an_identity():
    """D2. If nothing was there, there is nothing to have identified.

    A record claiming no vehicle while naming a plate does not describe a bad
    read -- it contradicts itself, and something downstream would believe one
    half of it.
    """
    with pytest.raises(ValueError, match="Nothing was there to identify"):
        a_read(presence=False, identity=Identity(plate="ABC123"), outcome=FALLBACK,
               confidence=0.0)


def test_presence_false_cannot_be_an_answer():
    with pytest.raises(ValueError, match="nothing to stand behind"):
        a_read(presence=False, identity=Identity(), outcome=ANSWER, confidence=0.999)


def test_presence_false_with_an_empty_identity_is_a_perfectly_good_record():
    """The control. A rule that refused every presence=false record would be
    safe and useless -- that record is the whole output of the gate."""
    read = a_read(presence=False, identity=Identity(), outcome=FALLBACK, confidence=0.0)
    assert read.presence is False
    assert Read.from_dict(read.to_dict()) == read


def test_presence_not_measured_is_not_presence_false():
    """The distinction the third state exists for. A lane with no reference view
    must behave as it did before this field existed, not refuse everybody."""
    read = a_read(presence=None, identity=Identity(plate="ABC123"))
    assert read.presence is None
    assert read.vehicle_present is None
    assert read.is_answer


def test_presence_must_be_a_boolean_or_null():
    with pytest.raises(ValueError, match="presence must be"):
        a_read(presence="yes")


def test_presence_survives_the_json_round_trip():
    read = a_read(presence=True, presence_confidence=0.8)
    restored = Read.from_dict(json.loads(json.dumps(read.to_dict())))
    assert restored.presence is True
    assert restored.presence_confidence == 0.8


def test_a_record_from_before_presence_existed_still_parses():
    """Presence is additive, so it does not bump `schema_version` -- which
    means records written without it must keep working."""
    payload = a_read().to_dict()
    del payload["presence"]
    del payload["presence_confidence"]
    assert Read.from_dict(payload).presence is None
