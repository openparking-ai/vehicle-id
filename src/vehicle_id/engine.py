"""The engine: captures in, one READ record out.

Today it reads plates. Make, model, colour and the appearance fingerprint are
the next slice; their fields stay null rather than being invented, which is the
contract's first rule and the reason this file returns `Identity()` fields
explicitly instead of leaving them to a default nobody can see.

The threshold is applied HERE, not by the consumer. That is deliberate: the
operating point is a property of the engine and its weights, it was measured
rather than chosen, and an integrator who had to supply it would be guessing at
a number only the harness can produce.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .contract import ANSWER, FALLBACK, Capture, Engine, Identity, Read, new_read_id, utc_now
from .plates.recognizer import DEFAULT_WEIGHTS, PlateRecognizer
from .presence import Presence, PresenceDetector

log = logging.getLogger(__name__)

ENGINE_NAME = "openparking-vehicle-id/plates"

#: The operating point the harness measured for the REFERENCE weights, kept
#: here as documentation of what a trained model looks like: the cheapest
#: threshold whose silent-wrong rate falls below 1%.
#:
#: It is deliberately NOT the default any more. A constant cannot know which
#: weights were loaded, and stamping it onto records produced by a different
#: model is a measurement claim nobody measured. `PlateEngine` reads the
#: operating point recorded beside the weights themselves, and refuses to start
#: if there is not one.
#:
#: It is far above a naive 0.85 because this recogniser is accurate AND
#: overconfident -- its mean confidence barely moves across the ladder while
#: accuracy falls.
REFERENCE_OPERATING_POINT = 0.99

#: Kept as an alias so an existing caller does not silently get a different
#: number; new code should read `threshold_applied` off the record instead.
RECOMMENDED_CONFIDENCE_THRESHOLD = REFERENCE_OPERATING_POINT


class PlateEngine:
    """Plate-only Vehicle ID. Real weights, measured confidence, no guessing."""

    def __init__(
        self,
        weights: Path = DEFAULT_WEIGHTS,
        device: str = "cpu",
        threshold: float | None = None,
        version: str = "",
        presence: PresenceDetector | None = None,
    ) -> None:
        self._recognizer = PlateRecognizer(weights, device=device)
        #: Optional, and when it is absent presence is reported as NOT MEASURED
        #: rather than assumed either way. A lane with no reference view of the
        #: empty tarmac behaves exactly as it did before this stage existed.
        self._presence = presence
        digest = weights_id(weights)
        self.threshold = _resolve_threshold(weights, digest, threshold)
        measured = load_operating_point(weights) or {}
        # Both are MEASURED per weights by the harness. Where an older sidecar
        # does not carry them, the conservative reading applies: a spread of 0
        # means any differing text counts as a different vehicle, and a noise
        # ceiling of 0 means nothing is dismissed as noise. That errs towards
        # falling back, which is the safe direction.
        self.same_vehicle_spread = float(measured.get("same_vehicle_spread") or 0.0)
        #: MEASURED, and recorded rather than used as a filter. For the
        #: reference weights it is 0.9998: uniform sensor noise reads as text
        #: on every frame, and 2.3% of those frames clear the 0.99 operating
        #: point. Dismissing competitors below this ceiling would dismiss
        #: almost all of them, which is the unsafe direction -- so it is
        #: published, not applied. What protects against a dead camera feed is
        #: the batch rule below: noise reads DIFFERENTLY every frame, so a
        #: multi-capture read of a noisy feed disagrees with itself and falls
        #: back.
        self.noise_ceiling = float(measured.get("noise_confidence_ceiling") or 0.0)
        self._engine = Engine(
            name=ENGINE_NAME,
            version=version or _package_version(),
            weights_id=digest,
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    def read(self, captures: Sequence[Capture]) -> Read:
        """Identify from one or more captures of the SAME vehicle.

        Always returns a record. There is no path through this method that
        raises instead of answering, because a consumer at a barrier needs an
        outcome, and "the engine threw" is not one a lane can act on.
        """
        if not captures:
            return self._record([], "", 0.0, None, disagreed=False, presence=_no_presence())

        decoded = [(capture, _decode(capture)) for capture in captures]
        images = [image for _, image in decoded if image is not None]

        # Asked FIRST, and answered separately. If nothing was there, nothing is
        # read: not because reading would fail, but because a plate read out of
        # an empty lane is the silent wrong answer this whole stage exists to
        # prevent. The recogniser has no rejection stage of its own -- it reads
        # text out of sensor noise -- so "nothing is there" has to be decided
        # before it, by something that is looking at the scene rather than at a
        # crop of it.
        presence = (
            self._presence.measure(images) if self._presence is not None else _no_presence()
        )
        if presence.present is False:
            log.info("no vehicle present (%s); not reading", presence.reason)
            return self._record(captures, "", 0.0, None, disagreed=False, presence=presence)

        # Every capture is read, and every result is kept. Taking only the
        # argmax and discarding the rest is what made a batch holding two
        # different vehicles come back as one confident, coherent, WRONG
        # record -- the engine held the disconfirming evidence and threw it
        # away.
        results: list[tuple[str, float, Capture]] = []
        for capture, image in decoded:
            if image is None:
                continue
            text, confidence = self._recognizer.read(image)
            if text:
                results.append((normalise(text), confidence, capture))

        if not results:
            return self._record(captures, "", 0.0, None, disagreed=False, presence=presence)

        best_text, best_confidence, best_capture = max(results, key=lambda r: r[1])

        # Two captures of one vehicle cannot show two DIFFERENT vehicles. They
        # can easily show two different readings of the same plate -- that is
        # what a degraded frame is, and it is the whole reason best-of-batch
        # exists -- so the question is which of those two this is, and it is
        # answered with numbers the harness measured for these weights rather
        # than with a guess:
        #
        #   * a competitor within the measured SAME-VEHICLE SPREAD of the
        #     winner is another look at the same plate;
        #   * anything else is a second vehicle in the batch, and then the
        #     honest answer is that we do not know which one is at the barrier.
        #
        # Picking the higher score there produces a confident, coherent,
        # in-contract record naming the wrong car, with nothing anywhere to say
        # so. Falling back is loud; that is the point.
        disagreed = False
        for text, confidence, _ in results:
            if text == best_text:
                continue
            if _distance(text, best_text) <= self.same_vehicle_spread:
                continue
            if confidence >= self.threshold or confidence >= best_confidence * 0.5:
                disagreed = True
                log.warning(
                    "a second vehicle appears to be in this batch of %d; answering fallback",
                    len(captures),
                )
                break

        return self._record(
            captures, best_text, best_confidence, best_capture, disagreed, presence
        )

    def _record(
        self,
        captures: Sequence[Capture],
        text: str,
        confidence: float,
        source: Capture | None,
        disagreed: bool,
        presence: Presence,
    ) -> Read:
        identity = Identity(
            plate=text or None,
            # Not invented. The slices that measure these have not been built.
            plate_region=None,
            make=None,
            model=None,
            color=None,
            marks=(),
        )
        # `presence is False` cannot answer, and cannot carry an identity --
        # the contract refuses such a record outright, so this is belt and
        # braces rather than the enforcement.
        answered = (
            bool(text)
            and confidence >= self.threshold
            and not disagreed
            and presence.present is not False
        )
        return Read(
            read_id=new_read_id(),
            # The contract says captured_at is when the FIRST capture was taken.
            # camera_id, though, is the camera the answer actually came from --
            # not the first camera in the list, which is how a plate read off a
            # staff camera used to be stamped with the entry lane's id.
            captured_at=captures[0].captured_at if captures else utc_now(),
            camera_id=(source or (captures[0] if captures else None)).camera_id
            if (source or captures)
            else "unknown",
            identity=identity,
            confidence=confidence,
            engine=self._engine,
            threshold_applied=self.threshold,
            outcome=ANSWER if answered else FALLBACK,
            captures_seen=len(captures),
            presence=presence.present,
            presence_confidence=presence.confidence,
        )


def _no_presence() -> Presence:
    return Presence.unmeasured("no presence detector is configured")


def _distance(a: str, b: str) -> int:
    """Levenshtein. Same implementation the harness measures the spread with."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalise(text: str) -> str:
    """Compare registrations, not layout: case and spacing are not the answer."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _resolve_threshold(weights: Path, digest: str | None, explicit: float | None) -> float:
    """The operating point for THESE weights, or a refusal to start.

    A module constant stamped onto every record regardless of which weights were
    loaded is not a measurement, however carefully it was measured once. Running
    the default smoke model and stamping 0.99 beside a comment claiming that
    number was measured for it is exactly the shape of a silent wrong answer --
    it happens to be safe today only because that model answers nothing.

    So: an explicit threshold is honoured and reported as applied, because it
    IS applied. Otherwise the operating point must have been MEASURED for these
    exact weights, by the harness, and recorded beside them. If it has not been,
    this refuses to construct rather than inventing a number.
    """
    if explicit is not None:
        return explicit
    measured = load_operating_point(weights)
    if measured is None:
        raise UnmeasuredWeights(
            f"no measured operating point for {weights}. It is not a number to "
            "guess: run\n"
            f"    python scripts/eval_plates.py --weights {weights} "
            "--write-operating-point\n"
            "or pass an explicit threshold if you know what you are choosing."
        )
    if measured.get("weights_id") != digest:
        raise UnmeasuredWeights(
            f"the operating point beside {weights} was measured for "
            f"{measured.get('weights_id')!r}, but the weights on disk are {digest!r}. "
            "Refusing to apply an operating point measured against a different model."
        )
    return float(measured["threshold"])


class UnmeasuredWeights(RuntimeError):
    """Raised rather than inventing an operating point for unmeasured weights."""


def operating_point_path(weights: Path) -> Path:
    return Path(weights).with_suffix(".operating-point.json")


def load_operating_point(weights: Path) -> dict | None:
    path = operating_point_path(weights)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_operating_point(
    weights: Path,
    threshold: float,
    evidence: dict,
    same_vehicle_spread: float | None = None,
    noise_confidence_ceiling: float | None = None,
) -> Path:
    """Record what the harness measured, bound to the weights it measured it on."""
    path = operating_point_path(weights)
    path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "same_vehicle_spread": same_vehicle_spread,
                "noise_confidence_ceiling": noise_confidence_ceiling,
                "weights_id": weights_id(weights),
                "measured_at": utc_now(),
                "evidence": evidence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def weights_id(weights: Path) -> str | None:
    """A digest of the weights actually on disk.

    Short, and prefixed with the algorithm so that a future change of digest is
    not silently indistinguishable from the old one.
    """
    path = Path(weights)
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("openparking-vehicle-id")
    except PackageNotFoundError:  # running from a source tree, not installed
        return "0.0.0+source"


def _decode(capture: Capture):
    import cv2
    import numpy as np

    buffer = np.frombuffer(capture.image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        # Never the plate, never the bytes. Nothing in this package logs an
        # identity.
        log.warning("capture from %s could not be decoded", capture.camera_id)
    return image
