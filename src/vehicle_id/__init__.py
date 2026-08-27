"""Open Parking AI — Vehicle ID.

A system that identifies a vehicle from camera frames and says how sure it is.
It replaces an LPR unit: it runs on its own, on the same device or LAN as
whatever consumes it, and it identifies with the internet down.

Everything a consumer needs is in `contract`. The engine, the service, the push
delivery and the CLI are implementations of that contract, and Open Parking AI's
own lane controller is a client of it through the same interface a third party
uses -- there is no in-process path reserved for us.

    from vehicle_id import Capture, PlateEngine

    engine = PlateEngine()
    read = engine.read([Capture.now(image_bytes, camera_id="lane-1")])
    if read.is_answer:
        ...
"""

from .contract import (
    ANSWER,
    FALLBACK,
    OUTCOMES,
    SCHEMA_VERSION,
    Capture,
    Engine,
    Identity,
    Read,
)

__all__ = [
    "ANSWER",
    "FALLBACK",
    "OUTCOMES",
    "SCHEMA_VERSION",
    "Capture",
    "Engine",
    "Identity",
    "PlateEngine",
    "RECOMMENDED_CONFIDENCE_THRESHOLD",
    "Read",
]


def __getattr__(name: str):
    # The engine pulls in torch and OpenCV. Importing this package must stay
    # cheap and dependency-free so that `contract` -- the part a consumer
    # integrates against -- can be imported by anything, including a consumer
    # that never runs the engine itself.
    if name in ("PlateEngine", "RECOMMENDED_CONFIDENCE_THRESHOLD"):
        from . import engine as _engine

        return getattr(_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
