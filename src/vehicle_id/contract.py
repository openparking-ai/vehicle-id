"""The contract. This module IS the product's public surface.

Everything else in this package is an implementation detail that may be
rewritten. A READ record is not: it is what a consumer integrates against,
ours and a third party's alike, and it is versioned from the first release so
that a change to it is always visible as a change.

Three properties are the whole point, and each is enforced below rather than
documented and hoped for:

  * **A field that was not MEASURED is null.** The engine never emits a value it
    did not measure. `make` is null today because no slice measures it -- not
    because the car had no make. A plausible guess in that field would be
    indistinguishable from a measurement to everything downstream.

  * **`fallback` is an answer, not an error.** A read the engine is not
    confident enough to stand behind is a successful read with
    `outcome = "fallback"`. It is returned, transported and stored exactly like
    any other. An integrator who treats it as a failure will retry it forever;
    one who treats it as an answer will do what the lane does -- ask a human.

  * **The operating threshold travels with the read.** `threshold_applied` is
    part of the record, so a consumer can see the operating point that produced
    the outcome instead of having to know it out of band. It is a measured
    number, not a chosen one -- see `scripts/eval_plates.py`.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime

#: Bumped whenever the record's shape changes in a way a consumer could notice.
#: A consumer that does not recognise the version it is handed should say so
#: and stop, not guess which fields it still understands.
SCHEMA_VERSION = 1

#: The two outcomes, and there is deliberately no third. "error" is not one:
#: an engine that cannot read is an engine that returns `fallback`.
ANSWER = "answer"
FALLBACK = "fallback"
OUTCOMES = (ANSWER, FALLBACK)


def _text(value, field_name: str) -> str:
    """A string, or a refusal that names the field.

    Every one of these used to be copied verbatim out of the request into the
    record: `captured_at: "banana"`, `camera_id: {"a": 1}`, a naive timestamp
    with no offset. All produced a 200 with `outcome: "answer"`, and a consumer
    pricing a stay from two such timestamps has no way to know.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _iso_utc(value, field_name: str) -> str:
    """An ISO 8601 timestamp that carries an offset.

    A naive timestamp is refused rather than assumed to be UTC. Assuming is how
    a lane in one timezone and a consumer in another end up disagreeing about
    when a car arrived, months later, with the money already collected.
    """
    text = _text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not ISO 8601: {text!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} has no UTC offset: {text!r}")
    return text


def _unit_interval(value, field_name: str) -> float:
    """A real number within [0, 1].

    `True` is excluded explicitly. Python says `0.0 <= True <= 1.0`, so a
    `confidence` of `true` -- a field carrying no measurement at all -- used to
    reach the lane as maximum confidence and open a barrier.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be within [0, 1], got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class Capture:
    """One image handed to the engine, plus enough provenance to argue later.

    This is the engine's INPUT, and it is part of the contract for the same
    reason the record is: a consumer has to construct it. It deliberately does
    not describe a camera, a stream or a driver -- the engine reads images and
    knows nothing about where they came from.
    """

    image_bytes: bytes
    captured_at: str
    camera_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, (bytes, bytearray)):
            raise ValueError("image_bytes must be bytes")
        _iso_utc(self.captured_at, "captured_at")
        _text(self.camera_id, "camera_id")

    @staticmethod
    def now(image_bytes: bytes, camera_id: str) -> Capture:
        return Capture(image_bytes=image_bytes, captured_at=utc_now(), camera_id=camera_id)


@dataclass(frozen=True, slots=True)
class Identity:
    """What was measured about the vehicle. Unmeasured stays null.

    `marks` is the one field that is a list rather than a nullable scalar, and
    empty means "none were measured", not "the vehicle had none". The
    distinction matters the day something does measure them.
    """

    plate: str | None = None
    plate_region: str | None = None
    make: str | None = None
    model: str | None = None
    color: str | None = None
    marks: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # An int plate was accepted here and then crashed inside the lane's
        # decision path on `.upper()`. A field that carries an identity has to
        # be text or absent; there is no third thing it could sensibly be.
        for name in ("plate", "plate_region", "make", "model", "color"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"identity.{name} must be a string or null, got {value!r}")
        if not all(isinstance(mark, str) for mark in self.marks):
            raise ValueError(f"identity.marks must all be strings, got {self.marks!r}")


@dataclass(frozen=True, slots=True)
class Engine:
    """Which engine produced this, precisely enough to reproduce it.

    `weights_id` is a digest of the weights actually loaded, not a label
    somebody typed. Two reads that disagree are worth investigating only if you
    can tell whether the same model produced them.
    """

    name: str
    version: str
    weights_id: str | None = None


@dataclass(frozen=True, slots=True)
class Read:
    """One identification. The product's unit of output."""

    read_id: str
    captured_at: str
    camera_id: str
    identity: Identity
    confidence: float
    engine: Engine
    threshold_applied: float
    outcome: str
    #: How many captures the engine was handed for this read. Additive, and
    #: present so that "one confident record" and "the batch disagreed with
    #: itself" are distinguishable after the fact rather than only in a log.
    captures_seen: int = 1
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            # `True == 1` in Python, so a bare equality check accepted
            # `schema_version: true` and let the record through.
            raise ValueError(f"schema_version must be an integer, got {self.schema_version!r}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        _text(self.read_id, "read_id")
        _iso_utc(self.captured_at, "captured_at")
        _text(self.camera_id, "camera_id")
        confidence = _unit_interval(self.confidence, "confidence")
        threshold = _unit_interval(self.threshold_applied, "threshold_applied")

        # The one relationship this record exists to express, and the one thing
        # that used to be unchecked: a record cannot claim the engine stood
        # behind it while carrying a confidence the stated operating point would
        # have rejected. Every combination below reached the lane and opened a
        # barrier -- `answer` at 0.86 against a threshold of 0.99, a threshold
        # of None, of NaN, of -5.
        if self.outcome == ANSWER and confidence < threshold:
            raise ValueError(
                f"outcome is {ANSWER!r} but confidence {confidence} is below the "
                f"stated threshold_applied {threshold}. A record cannot claim an "
                "operating point it did not clear."
            )

    @property
    def is_answer(self) -> bool:
        """True when the engine stands behind this read.

        Provided so that a consumer never has to compare `outcome` to a string
        it typed itself, which is how the check gets silently inverted.
        """
        return self.outcome == ANSWER

    def to_dict(self) -> dict:
        d = asdict(self)
        d["identity"]["marks"] = list(self.identity.marks)
        return d

    @staticmethod
    def from_dict(d: dict) -> Read:
        version = d.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version!r}; "
                f"this build understands {SCHEMA_VERSION}. Refusing to guess "
                "which fields still mean what they used to."
            )
        identity = dict(d["identity"])
        identity["marks"] = tuple(identity.get("marks") or ())
        return Read(
            read_id=d["read_id"],
            captured_at=d["captured_at"],
            camera_id=d["camera_id"],
            identity=_only_known(Identity, identity),
            confidence=d["confidence"],
            engine=_only_known(Engine, d["engine"]),
            threshold_applied=d["threshold_applied"],
            outcome=d["outcome"],
            captures_seen=d.get("captures_seen", 1),
            schema_version=d["schema_version"],
        )

    def redacted(self) -> Read:
        """The same record with the identity blanked, for logs and metrics.

        Keeps the shape, the outcome and the confidence; drops what identifies a
        vehicle. Nothing in this package logs a plate, and this is what it uses
        instead of hand-rolling a dict at each site.
        """
        return replace(self, identity=Identity())


def _only_known(kind, payload: dict):
    """Build `kind` from `payload`, dropping fields this build does not know.

    The contract promises that additive changes do not bump `schema_version`
    and that a consumer should ignore fields it does not recognise. This is the
    reference parser, so it has to actually do that -- and nested objects are
    where it matters most, because `identity` is exactly where the next slice
    adds `body_type` and the rest.

    Refusing an unknown field here would mean every consumer on this parser
    breaks on the first additive change, which is the opposite of what
    versioning it from day one was for.
    """
    known = {f.name for f in fields(kind)}
    return kind(**{k: v for k, v in payload.items() if k in known})


def utc_now() -> str:
    """UTC, ISO 8601, with the offset present.

    A naive timestamp is the bug that surfaces months later, when a lane in one
    timezone and a consumer in another disagree about when a car arrived.
    """
    return datetime.now(UTC).isoformat()


def new_read_id() -> str:
    return uuid.uuid4().hex
