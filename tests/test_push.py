"""Push delivery: a consumer being down must never lose a read."""

from __future__ import annotations

import json

import pytest

from vehicle_id.contract import ANSWER, Engine, Identity, Read, utc_now
from vehicle_id.push import ReadPusher, Refused


def a_read(read_id: str) -> Read:
    return Read(
        read_id=read_id,
        captured_at=utc_now(),
        camera_id="lane-1",
        identity=Identity(plate="ABC123"),
        confidence=0.995,
        engine=Engine(name="test", version="0.1.0"),
        threshold_applied=0.99,
        outcome=ANSWER,
    )


class Consumer:
    """A stub consumer whose behaviour the test sets. The pusher's real retry
    and refusal logic runs against it -- what is faked is the far end, never the
    logic under test."""

    def __init__(self) -> None:
        self.received: list[dict] = []
        self.mode = "ok"

    def __call__(self, url: str, payload: dict, timeout: float) -> None:
        if self.mode == "down":
            raise ConnectionError("consumer is down")
        if self.mode == "refuse":
            raise Refused(400)
        self.received.append(payload)


@pytest.fixture
def pusher(tmp_path):
    consumer = Consumer()
    p = ReadPusher("http://consumer.invalid/reads", tmp_path / "queue.jsonl", opener=consumer)
    return p, consumer


def test_a_delivered_read_leaves_the_queue(pusher):
    p, consumer = pusher
    p.submit(a_read("r1"))
    assert [r["read_id"] for r in consumer.received] == ["r1"]
    assert len(p.queue) == 0


def test_a_read_is_durable_before_delivery_is_attempted(pusher, tmp_path):
    p, consumer = pusher
    consumer.mode = "down"
    p.submit(a_read("r1"))
    # The process could die right here. The read must still exist on disk.
    lines = (tmp_path / "queue.jsonl").read_text().splitlines()
    assert [json.loads(line)["read_id"] for line in lines] == ["r1"]
    assert p.stats.pending == 1


def test_the_queue_drains_when_the_consumer_comes_back(pusher):
    p, consumer = pusher
    consumer.mode = "down"
    p.submit(a_read("r1"))
    p.submit(a_read("r2"))
    assert consumer.received == []

    consumer.mode = "ok"
    stats = p.flush()
    assert [r["read_id"] for r in consumer.received] == ["r1", "r2"]
    assert stats.pending == 0
    assert len(p.queue) == 0


def test_order_is_preserved_when_delivery_fails_midway(tmp_path):
    consumer = Consumer()
    delivered: list[str] = []

    def flaky(url, payload, timeout):
        if payload["read_id"] == "r2":
            raise ConnectionError("dropped")
        delivered.append(payload["read_id"])

    p = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=flaky)
    for name in ("r1", "r2", "r3"):
        p.queue.append(a_read(name))
    p.flush()

    # r3 waits behind r2 rather than overtaking it. A consumer that receives an
    # exit before the entry it belongs to prices the stay wrong.
    assert delivered == ["r1"]
    assert [r.read_id for r in p.queue.load()] == ["r2", "r3"]
    assert consumer.received == []


def test_a_refusal_is_dropped_and_counted_rather_than_retried_forever(pusher):
    p, consumer = pusher
    consumer.mode = "refuse"
    p.submit(a_read("poison"))
    assert p.stats.refused == 1
    assert len(p.queue) == 0, "poison must not block everything behind it"


def test_a_refusal_does_not_block_the_reads_behind_it(tmp_path):
    def selective(url, payload, timeout):
        if payload["read_id"] == "poison":
            raise Refused(422)

    p = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=selective)
    for name in ("poison", "r2"):
        p.queue.append(a_read(name))
    stats = p.flush()
    assert stats.refused == 1
    assert stats.delivered == 1
    assert len(p.queue) == 0


def test_the_queue_survives_a_restart(tmp_path):
    down = Consumer()
    down.mode = "down"
    first = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=down)
    first.submit(a_read("r1"))

    # A new process, same queue file.
    back = Consumer()
    second = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=back)
    second.flush()
    assert [r["read_id"] for r in back.received] == ["r1"]
