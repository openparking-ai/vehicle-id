"""The local service. One HTTP contract, and everything is a client of it.

There is no in-process shortcut reserved for Open Parking AI. Our own lane
controller talks to this service over exactly the interface a third party uses,
which is the only way to find out that the interface is inadequate before an
integrator does.

**Local, always.** No cloud, no remote API, no hosted anything on the
identification path -- the engine has to identify with the internet down, so it
binds to loopback by default and is meant to run on the same device or the same
LAN as whatever consumes it.

Four routes, and each exists because a different kind of consumer needs it:

    POST /v1/reads          submit captures, get the READ back   (synchronous)
    GET  /v1/reads/last     the most recent read                 (pull)
    GET  /v1/reads?since=N  everything after a cursor            (pull, catch-up)
    GET  /v1/health         engine, version, weights, threshold

Push is the fourth shape and is not a route here: configure a URL and this
service POSTs each read to it as it happens (see `push.py`).

Written on `http.server` rather than a framework on purpose. This runs in a gate
housing on a Jetson; a dependency-free service is one fewer thing to keep
patched, and nothing here needs more than the standard library provides.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from collections import deque
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .contract import SCHEMA_VERSION, Capture, Read, utc_now

log = logging.getLogger(__name__)

#: How many recent reads the pull routes can still serve. A consumer that falls
#: further behind than this has a bigger problem than a cursor, and push exists
#: precisely so it does not have to poll.
DEFAULT_HISTORY = 256

MAX_BODY_BYTES = 32 * 1024 * 1024


class ReadStore:
    """Recent reads, in order, addressable by a monotonic cursor.

    In memory by design: this is a catch-up window for a consumer that blinked,
    not a record of anything. The durable copy of a read belongs to whoever
    consumes it -- and to the push queue until they have it.
    """

    def __init__(self, history: int = DEFAULT_HISTORY) -> None:
        self._reads: deque[tuple[int, Read]] = deque(maxlen=history)
        self._cursor = 0
        self._lock = threading.Lock()

    def add(self, read: Read) -> int:
        with self._lock:
            self._cursor += 1
            self._reads.append((self._cursor, read))
            return self._cursor

    def last(self) -> tuple[int, Read] | None:
        with self._lock:
            return self._reads[-1] if self._reads else None

    def since(self, cursor: int) -> list[tuple[int, Read]]:
        with self._lock:
            return [(seq, read) for seq, read in self._reads if seq > cursor]

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor


class VehicleIdService:
    """The engine, a read store and an optional pusher, wired together."""

    def __init__(self, engine, pusher=None, history: int = DEFAULT_HISTORY) -> None:
        self.engine = engine
        self.pusher = pusher
        self.store = ReadStore(history)

    def identify(self, captures: Sequence[Capture]) -> tuple[int, Read]:
        read = self.engine.read(captures)
        seq = self.store.add(read)
        if self.pusher is not None:
            try:
                self.pusher.submit(read)
            except Exception:
                # Push is best-effort at THIS point only because the pusher has
                # already written the read to its durable queue. A failure here
                # must never stop the synchronous caller getting its answer --
                # there is a car at the barrier.
                log.exception("push submission failed; the read is queued")
        return seq, read

    def health(self) -> dict:
        engine = self.engine.engine
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "engine": {
                "name": engine.name,
                "version": engine.version,
                "weights_id": engine.weights_id,
            },
            "threshold_applied": self.engine.threshold,
            "cursor": self.store.cursor,
            "time": utc_now(),
        }


class _Handler(BaseHTTPRequestHandler):
    service: VehicleIdService

    server_version = "openparking-vehicle-id"
    sys_version = ""

    def log_message(self, fmt: str, *args) -> None:
        # The default handler logs the request line to stderr. A request line
        # here can carry a cursor but never an identity; even so, routing it
        # through logging rather than stderr keeps a deployment's log policy in
        # one place.
        log.debug("%s - %s", self.address_string(), fmt % args)

    # --- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (http.server's spelling)
        url = urlparse(self.path)
        if url.path == "/v1/health":
            return self._json(200, self.service.health())

        if url.path == "/v1/reads/last":
            latest = self.service.store.last()
            if latest is None:
                return self._json(404, {"error": "no reads yet"})
            seq, read = latest
            return self._json(200, {"cursor": seq, "read": read.to_dict()})

        if url.path == "/v1/reads":
            raw = parse_qs(url.query).get("since", ["0"])[0]
            try:
                since = int(raw)
            except ValueError:
                return self._json(400, {"error": f"since must be an integer, got {raw!r}"})
            items = self.service.store.since(since)
            return self._json(
                200,
                {
                    "cursor": self.service.store.cursor,
                    "reads": [{"cursor": seq, "read": read.to_dict()} for seq, read in items],
                },
            )

        return self._json(404, {"error": "no such route"})

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/v1/reads":
            return self._json(404, {"error": "no such route"})

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json(400, {"error": "empty body"})
        if length > MAX_BODY_BYTES:
            return self._json(413, {"error": "body too large"})
        body = self.rfile.read(length)

        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        try:
            captures = _captures_from(body, content_type, self.headers.get("X-Camera-Id"))
        except ValueError as exc:
            # A malformed request is a 400 and stays a 400. It is the caller's
            # to fix, and a consumer's retry loop must be able to tell "you sent
            # nonsense" from "try again later".
            return self._json(400, {"error": str(exc)})

        try:
            seq, read = self.service.identify(captures)
        except Exception:
            # The engine is not allowed to take the connection down with it. A
            # dropped connection reads as "the service is gone" to a consumer,
            # which is a different problem with a different remedy.
            log.exception("identification failed")
            return self._json(500, {"error": "identification failed"})
        return self._json(200, {"cursor": seq, "read": read.to_dict()})

    # --- plumbing --------------------------------------------------------

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _captures_from(body: bytes, content_type: str, camera_header: str | None) -> list[Capture]:
    """Two accepted request shapes, because two kinds of caller exist.

    JSON, for a caller sending several frames of one vehicle -- which is what
    the engine wants, and what a lane sends. Raw image bytes, for the caller
    replacing an LPR unit that has exactly one frame and no wish to base64 it.
    """
    if content_type == "application/json":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"body is not valid JSON: {exc}") from exc
        # Valid JSON is not a valid request. A list, a string and a null are all
        # well-formed JSON, and every one of them used to reach `.get` and kill
        # the handler mid-response -- which a caller cannot tell apart from the
        # service being down, so its retry loop backs off forever instead of
        # fixing the request.
        if not isinstance(payload, dict):
            raise ValueError(f"body must be a JSON object, got {type(payload).__name__}")
        camera_id = payload.get("camera_id") or camera_header or "unknown"
        entries = payload.get("captures")
        if not isinstance(entries, list) or not entries:
            raise ValueError("captures must be a non-empty list")
        captures = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"each capture must be an object, got {type(entry).__name__}")
            encoded = entry.get("image_b64")
            if not encoded or not isinstance(encoded, str):
                raise ValueError("each capture needs image_b64, as a string")
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError(f"image_b64 is not valid base64: {exc}") from exc
            captures.append(
                Capture(
                    image_bytes=image_bytes,
                    captured_at=entry.get("captured_at") or utc_now(),
                    camera_id=entry.get("camera_id") or camera_id,
                )
            )
        return captures

    if content_type.startswith("image/") or content_type == "application/octet-stream":
        return [Capture.now(body, camera_header or "unknown")]

    raise ValueError(
        f"unsupported Content-Type {content_type!r}; send application/json or image/*"
    )


def make_server(service: VehicleIdService, host: str = "127.0.0.1", port: int = 8088):
    """Bound to loopback by default. Exposing it is a deployment decision.

    D7 says the identification path is local. Defaulting to 0.0.0.0 would make
    reaching across a network the easy accident rather than the deliberate act.
    """
    handler = type("_BoundHandler", (_Handler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)
