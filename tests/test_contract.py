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


# --- the appearance descriptor, which is ADDITIVE ------------------------
#
# The descriptor is a component of an identity, like the plate: a garage may run
# on plate alone, on appearance alone, or on both. These tests run in the job
# with no torch and no OpenCV, deliberately -- the claim being made is about the
# CONTRACT, and a claim about the contract that can only be checked with the
# engine installed is not one a consumer can rely on.

#: A descriptor is opaque to the contract. This is a shape, not a real one; the
#: bytes it stands for are `vehicle_id.fingerprint`'s business and nothing here
#: parses them.
A_DESCRIPTOR = "opvid-fp/1:eJwzNDI2MTUzt7A0MjYxNTO3sAQAJ8sD9A"


def test_the_descriptor_is_a_component_of_the_identity_and_defaults_to_null():
    """Null means NOT MEASURED, the same rule every other component obeys."""
    assert Identity().descriptor is None
    assert Identity(plate="ABC123").descriptor is None
    assert Identity(descriptor=A_DESCRIPTOR).descriptor == A_DESCRIPTOR


def test_an_identity_may_carry_a_descriptor_and_no_plate():
    """The point of the whole round: the plate is one component, not the identity.

    A vehicle identified by appearance alone is a complete identity. If this
    ever needs a plate to be valid, the record has stopped modelling what the
    product says it models.
    """
    read = a_read(identity=Identity(descriptor=A_DESCRIPTOR))
    assert read.identity.plate is None
    assert read.identity.descriptor == A_DESCRIPTOR
    assert read.is_answer


def test_a_descriptor_that_is_not_a_string_is_refused():
    """The wire is JSON and `Identity` is frozen, slots and hashable. A dict or
    a list satisfies none of that, and a record carrying one would fail
    somewhere further away than here."""
    for bad in ({"kp": [1, 2]}, [1, 2, 3], 7, 0.5, True):
        with pytest.raises(ValueError, match="identity.descriptor"):
            Identity(descriptor=bad)


def test_an_identity_carrying_a_descriptor_is_still_hashable_and_comparable():
    """`Read.__post_init__` compares an identity with `!=` and the dataclass is
    frozen and hashable. A field that broke either would break the record's own
    presence invariant, which is what that comparison is for."""
    one = Identity(descriptor=A_DESCRIPTOR)
    same = Identity(descriptor=A_DESCRIPTOR)
    other = Identity(descriptor=A_DESCRIPTOR + "x")
    assert one == same and hash(one) == hash(same)
    assert one != other
    assert one != Identity()


def test_the_descriptor_survives_the_json_round_trip_unchanged():
    """A string survives JSON exactly. A tuple of numbers would come back as a
    list, and the two would stop comparing equal -- which is why the field is a
    string and not the tuple that was first proposed."""
    read = a_read(identity=Identity(plate="ABC123", descriptor=A_DESCRIPTOR))
    restored = Read.from_dict(json.loads(json.dumps(read.to_dict())))
    assert restored.identity.descriptor == A_DESCRIPTOR
    assert restored.identity == read.identity


def test_adding_the_descriptor_did_not_move_the_schema_version():
    """The published rule is that additive changes do not bump it, and this is
    the test that keeps the rule honest.

    Bumping it has already broken production once, on 2026-08-31: `from_dict`
    version-matches on `!=`, so every pinned consumer refuses every record and
    every vehicle at every lane falls to a human on every arrival. The lane
    controller and the gate agent each carry their own hardcoded list of known
    versions in their own repositories, and neither is in this one.
    """
    assert SCHEMA_VERSION == 1
    payload = a_read(identity=Identity(plate="ABC123", descriptor=A_DESCRIPTOR)).to_dict()
    assert payload["schema_version"] == 1


def test_a_consumer_pinned_before_the_descriptor_existed_sees_a_record_it_understands():
    """The other half, and the one that makes 'additive' mean anything.

    An OLD consumer handed a NEW record must not refuse it: the version has not
    moved, so it parses, and the field it does not know is dropped rather than
    rejected. `_only_known` is what does that, and it is exercised here on the
    nested identity, which is where the next component will land too.
    """
    payload = a_read(identity=Identity(plate="ABC123", descriptor=A_DESCRIPTOR)).to_dict()
    payload["identity"]["body_type"] = "van"          # a component from a later round
    restored = Read.from_dict(payload)
    assert restored.identity.plate == "ABC123"
    assert restored.identity.descriptor == A_DESCRIPTOR

    # And the reverse: a record written BEFORE the field existed still parses,
    # and the descriptor is null -- not measured, which is exactly true.
    old = a_read().to_dict()
    del old["identity"]["descriptor"]
    assert Read.from_dict(old).identity.descriptor is None


def test_redacted_drops_the_descriptor_with_the_rest_of_the_identity():
    """Nothing in this package logs an identity, and a descriptor is one.

    `redacted()` blanks the whole identity rather than listing fields, which is
    why no logging change was needed for this round -- and this is the test that
    says so rather than the assumption that says so.
    """
    read = a_read(identity=Identity(plate="ABC123", descriptor=A_DESCRIPTOR))
    assert read.redacted().identity == Identity()
    assert read.redacted().identity.descriptor is None
    assert read.redacted().confidence == read.confidence


def test_presence_false_cannot_carry_a_descriptor_either():
    """If nothing was there, there is nothing to have described.

    The invariant is written against the whole identity rather than against the
    plate, so it covers a component that did not exist when it was written. That
    is the property being checked here -- not the descriptor.
    """
    with pytest.raises(ValueError, match="presence is false"):
        a_read(presence=False, outcome=FALLBACK, confidence=0.0,
               identity=Identity(descriptor=A_DESCRIPTOR))
