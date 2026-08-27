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
import logging
from collections.abc import Sequence
from pathlib import Path

from .contract import ANSWER, FALLBACK, Capture, Engine, Identity, Read, new_read_id, utc_now
from .plates.recognizer import DEFAULT_WEIGHTS, PlateRecognizer

log = logging.getLogger(__name__)

ENGINE_NAME = "openparking-vehicle-id/plates"

#: MEASURED, not chosen. scripts/eval_plates.py, 10-rung ladder, 1500 plates:
#: this is the cheapest threshold whose silent-wrong rate falls below 1%
#: (0.87% wrong-but-answered, 30.9% sent to fallback).
#:
#: It is far above a naive 0.85 because this recogniser is accurate AND
#: overconfident -- its mean confidence barely moves across the ladder while
#: accuracy falls. Anything consuming this engine gets the value in every
#: record's `threshold_applied` rather than having to know it.
RECOMMENDED_CONFIDENCE_THRESHOLD = 0.99


class PlateEngine:
    """Plate-only Vehicle ID. Real weights, measured confidence, no guessing."""

    def __init__(
        self,
        weights: Path = DEFAULT_WEIGHTS,
        device: str = "cpu",
        threshold: float = RECOMMENDED_CONFIDENCE_THRESHOLD,
        version: str = "",
    ) -> None:
        self._recognizer = PlateRecognizer(weights, device=device)
        self.threshold = threshold
        self._engine = Engine(
            name=ENGINE_NAME,
            version=version or _package_version(),
            weights_id=weights_id(weights),
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    def read(self, captures: Sequence[Capture]) -> Read:
        """Identify from one or more captures of the same vehicle.

        Always returns a record. There is no path through this method that
        raises instead of answering, because a consumer at a barrier needs an
        outcome, and "the engine threw" is not one a lane can act on.
        """
        best_text, best_confidence = "", 0.0
        camera_id = captures[0].camera_id if captures else ""
        captured_at = captures[0].captured_at if captures else utc_now()

        # Best of the batch. Several captures exist precisely so that one bad
        # moment -- a wiper, a headlight, a bump -- does not decide.
        for capture in captures:
            image = _decode(capture)
            if image is None:
                continue
            text, confidence = self._recognizer.read(image)
            if text and confidence > best_confidence:
                best_text, best_confidence = text, confidence

        identity = Identity(
            plate=best_text or None,
            # Not invented. The slices that measure these have not been built.
            plate_region=None,
            make=None,
            model=None,
            color=None,
            marks=(),
        )
        answered = bool(best_text) and best_confidence >= self.threshold
        return Read(
            read_id=new_read_id(),
            captured_at=captured_at,
            camera_id=camera_id,
            identity=identity,
            confidence=best_confidence,
            engine=self._engine,
            threshold_applied=self.threshold,
            outcome=ANSWER if answered else FALLBACK,
        )


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
