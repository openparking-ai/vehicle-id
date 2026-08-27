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

import uuid
from dataclasses import asdict, dataclass, field, replace
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
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within [0, 1], got {self.confidence!r}")

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
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {d.get('schema_version')!r}; "
                f"this build understands {SCHEMA_VERSION}. Refusing to guess "
                "which fields still mean what they used to."
            )
        identity = dict(d["identity"])
        identity["marks"] = tuple(identity.get("marks") or ())
        return Read(
            read_id=d["read_id"],
            captured_at=d["captured_at"],
            camera_id=d["camera_id"],
            identity=Identity(**identity),
            confidence=d["confidence"],
            engine=Engine(**d["engine"]),
            threshold_applied=d["threshold_applied"],
            outcome=d["outcome"],
            schema_version=d["schema_version"],
        )

    def redacted(self) -> Read:
        """The same record with the identity blanked, for logs and metrics.

        Keeps the shape, the outcome and the confidence; drops what identifies a
        vehicle. Nothing in this package logs a plate, and this is what it uses
        instead of hand-rolling a dict at each site.
        """
        return replace(self, identity=Identity())


def utc_now() -> str:
    """UTC, ISO 8601, with the offset present.

    A naive timestamp is the bug that surfaces months later, when a lane in one
    timezone and a consumer in another disagree about when a car arrived.
    """
    return datetime.now(UTC).isoformat()


def new_read_id() -> str:
    return uuid.uuid4().hex
