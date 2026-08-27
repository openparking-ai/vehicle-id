"""The local service, over real HTTP.

The requests below go through a socket, not through a handler called directly.
Our own lane controller reaches the engine this way and so does a third party,
so the thing under test has to be the thing they both use.

No engine here: a stub stands in for it, because what these tests are about is
the CONTRACT surface -- routes, shapes, cursors, refusals -- and wiring torch
into them would only make them slow and skippable.
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request

import pytest

from vehicle_id.contract import ANSWER, FALLBACK, Engine, Identity, Read, utc_now
from vehicle_id.service import VehicleIdService, make_server


class StubEngine:
    """Answers for a capture whose bytes contain b"plate", falls back otherwise."""

    threshold = 0.99

    def __init__(self) -> None:
        self.engine = Engine(name="stub", version="0.0.1", weights_id="sha256:stub")
        self.seen: list[int] = []

    def read(self, captures):
        self.seen.append(len(captures))
        found = any(b"plate" in c.image_bytes for c in captures)
        return Read(
            read_id=f"r{len(self.seen)}",
            captured_at=captures[0].captured_at if captures else utc_now(),
            camera_id=captures[0].camera_id if captures else "",
            identity=Identity(plate="ABC123" if found else None),
            confidence=0.995 if found else 0.10,
            engine=self.engine,
            threshold_applied=self.threshold,
            outcome=ANSWER if found else FALLBACK,
        )


@pytest.fixture
def server():
    engine = StubEngine()
    service = VehicleIdService(engine)
    httpd = make_server(service, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    yield f"http://{host}:{port}", engine, service
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def get(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def post(url: str, body: bytes, content_type: str, headers: dict | None = None):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": content_type, **(headers or {})}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def json_body(*images, camera_id="lane-1"):
    return json.dumps(
        {
            "camera_id": camera_id,
            "captures": [{"image_b64": base64.b64encode(i).decode()} for i in images],
        }
    ).encode()


# --- the synchronous route, which is what a lane needs --------------------

def test_submitting_captures_returns_the_read_synchronously(server):
    base, _, _ = server
    status, payload = post(f"{base}/v1/reads", json_body(b"a plate here"), "application/json")
    assert status == 200
    read = Read.from_dict(payload["read"])
    assert read.is_answer
    assert read.identity.plate == "ABC123"
    assert read.camera_id == "lane-1"


def test_several_captures_of_one_vehicle_reach_the_engine_together(server):
    base, engine, _ = server
    post(f"{base}/v1/reads", json_body(b"blur", b"a plate here", b"blur"), "application/json")
    assert engine.seen == [3], "the engine must see one vehicle's captures as one read"


def test_a_raw_image_body_is_accepted_for_a_caller_replacing_an_lpr(server):
    base, _, _ = server
    status, payload = post(
        f"{base}/v1/reads", b"a plate here", "image/jpeg", {"X-Camera-Id": "gate-2"}
    )
    assert status == 200
    assert payload["read"]["camera_id"] == "gate-2"


def test_a_fallback_comes_back_as_a_200_not_an_error(server):
    # The property the whole product rests on: not confident is an ANSWER.
    base, _, _ = server
    status, payload = post(f"{base}/v1/reads", json_body(b"nothing here"), "application/json")
    assert status == 200
    assert payload["read"]["outcome"] == FALLBACK


def test_a_malformed_request_is_a_400_and_stays_one(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(f"{base}/v1/reads", b"{not json", "application/json")
    assert caught.value.code == 400


def test_an_unsupported_content_type_is_refused(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(f"{base}/v1/reads", b"x", "text/plain")
    assert caught.value.code == 400


# --- the pull routes ------------------------------------------------------

def test_last_is_404_before_anything_has_been_read(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(f"{base}/v1/reads/last")
    assert caught.value.code == 404


def test_last_returns_the_most_recent_read(server):
    base, _, _ = server
    post(f"{base}/v1/reads", json_body(b"a plate here"), "application/json")
    post(f"{base}/v1/reads", json_body(b"nothing"), "application/json")
    status, payload = get(f"{base}/v1/reads/last")
    assert status == 200
    assert payload["read"]["outcome"] == FALLBACK


def test_a_cursor_returns_only_what_came_after_it(server):
    base, _, _ = server
    _, first = post(f"{base}/v1/reads", json_body(b"a plate here"), "application/json")
    post(f"{base}/v1/reads", json_body(b"nothing"), "application/json")
    post(f"{base}/v1/reads", json_body(b"a plate here"), "application/json")

    status, payload = get(f"{base}/v1/reads?since={first['cursor']}")
    assert status == 200
    assert len(payload["reads"]) == 2
    assert payload["reads"][0]["cursor"] == first["cursor"] + 1


def test_a_cursor_at_the_end_returns_nothing_rather_than_repeating(server):
    base, _, _ = server
    _, first = post(f"{base}/v1/reads", json_body(b"a plate here"), "application/json")
    _, payload = get(f"{base}/v1/reads?since={first['cursor']}")
    assert payload["reads"] == []


def test_a_nonsense_cursor_is_a_400(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(f"{base}/v1/reads?since=yesterday")
    assert caught.value.code == 400


def test_health_names_the_engine_the_weights_and_the_operating_point(server):
    base, _, _ = server
    status, payload = get(f"{base}/v1/health")
    assert status == 200
    assert payload["engine"]["name"] == "stub"
    assert payload["engine"]["weights_id"] == "sha256:stub"
    assert payload["threshold_applied"] == 0.99
    assert payload["schema_version"] == 1


def test_an_unknown_route_is_a_404(server):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(f"{base}/v1/whatever")
    assert caught.value.code == 404


# --- push, wired into the service ----------------------------------------

def test_every_read_is_handed_to_the_pusher(server):
    base, engine, service = server

    pushed = []

    class Pusher:
        def submit(self, read):
            pushed.append(read.read_id)

    service.pusher = Pusher()
    post(f"{base}/v1/reads", json_body(b"a plate here"), "application/json")
    assert len(pushed) == 1


def test_a_failing_pusher_never_denies_the_caller_its_answer(server):
    # There is a car at the barrier. Push is not on that path.
    base, _, service = server

    class Broken:
        def submit(self, read):
            raise RuntimeError("queue disk is full")

    service.pusher = Broken()
    status, payload = post(f"{base}/v1/reads", json_body(b"a plate here"), "application/json")
    assert status == 200
    assert payload["read"]["outcome"] == ANSWER
