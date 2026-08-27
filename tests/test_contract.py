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
