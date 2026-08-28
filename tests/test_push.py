"""Push delivery: a consumer being down must never lose a read."""

from __future__ import annotations

import json
import os
import stat
import time

import pytest

from vehicle_id.contract import ANSWER, Engine, Identity, Read, utc_now
from vehicle_id.push import QUEUE_FILE_MODE, ReadPusher, Refused


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


# --- the guarantee under concurrency --------------------------------------

def test_concurrent_submits_never_lose_a_read(tmp_path):
    """The queue's whole reason to exist, tested the way it actually runs.

    The service is threaded, so several vehicles can be mid-read at once. The
    first version of `flush` rewrote the file from a snapshot taken before its
    deliveries began, which truncated anything appended during them -- reads
    that had already been answered, with nothing on disk to say they existed.
    """
    import threading

    consumer = Consumer()
    consumer.mode = "down"
    p = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=consumer)

    def submit(i: int) -> None:
        p.submit(a_read(f"r{i}"))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    queued = {r.read_id for r in p.queue.load()}
    assert queued == {f"r{i}" for i in range(12)}, (
        f"reads were answered and then lost: {sorted({f'r{i}' for i in range(12)} - queued)}"
    )


def test_a_read_arriving_during_a_slow_delivery_survives_it(tmp_path):
    """The narrow version of the race, made deterministic.

    One read is already queued and its delivery is made slow on purpose. A
    second read is submitted while that delivery is in flight. Rewriting the
    queue from the pre-delivery snapshot would drop the second one.
    """
    import threading

    started = threading.Event()
    release = threading.Event()

    def slow(url, payload, timeout):
        if payload["read_id"] == "slow":
            started.set()
            release.wait(5)

    p = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=slow)
    p.queue.append(a_read("slow"))

    flusher = threading.Thread(target=p.flush)
    flusher.start()
    assert started.wait(5), "the slow delivery never started"

    p.queue.append(a_read("arrived-mid-flight"))
    release.set()
    flusher.join(5)

    assert [r.read_id for r in p.queue.load()] == ["arrived-mid-flight"]


def test_start_delivers_what_a_previous_run_left_behind(tmp_path):
    """Retry must not be coupled to new traffic.

    A lane that stops at midnight with reads outstanding, and a consumer that
    comes back at 01:00, must not wait for the next car.
    """
    down = Consumer()
    down.mode = "down"
    first = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=down)
    first.submit(a_read("left-behind"))
    assert len(first.queue) == 1

    back = Consumer()
    second = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=back)
    second.start()
    try:
        assert [r["read_id"] for r in back.received] == ["left-behind"]
    finally:
        second.stop()


def test_the_retry_timer_delivers_without_any_new_traffic(tmp_path):
    consumer = Consumer()
    consumer.mode = "down"
    p = ReadPusher(
        "http://consumer.invalid/reads", tmp_path / "q.jsonl",
        retry_interval=0.05, opener=consumer,
    )
    p.start()          # what `vehicle-id serve` does
    try:
        p.submit(a_read("waiting"))
        assert consumer.received == []
        consumer.mode = "ok"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not consumer.received:
            time.sleep(0.05)
    finally:
        p.stop()
    assert [r["read_id"] for r in consumer.received] == ["waiting"], (
        "nothing was retried without a new read arriving"
    )


# --- the queue as a hostile file ------------------------------------------

def test_a_torn_line_does_not_stop_the_service_starting(tmp_path):
    """A power cut mid-append leaves a partial line.

    Raising on it took the whole engine down at startup -- the lane dead until
    someone found the bad line at 2am. That is a far worse failure than losing
    the one read that was being written when the power went.
    """
    queue = tmp_path / "q.jsonl"
    queue.write_text(
        json.dumps(a_read("intact").to_dict()) + "\n" + '{"read_id": "torn", "captu',
        encoding="utf-8",
    )
    consumer = Consumer()
    p = ReadPusher("http://consumer.invalid/reads", queue, opener=consumer)
    p.start()
    try:
        assert [r["read_id"] for r in consumer.received] == ["intact"]
        assert p.queue.damaged_count == 1
        assert p.queue.damaged.exists(), "the unreadable line was discarded rather than kept"
    finally:
        p.stop()


def test_a_failed_append_is_reported_as_loss_not_as_a_deferred_delivery(tmp_path):
    """The read may exist nowhere. Saying 'it is queued' over the top of that is
    the difference between a problem someone finds and one nobody does."""
    queue = tmp_path / "q.jsonl"
    queue.write_text("", encoding="utf-8")
    queue.chmod(0o444)
    p = ReadPusher("http://consumer.invalid/reads", queue, opener=Consumer())
    try:
        with pytest.raises(PermissionError):
            p.submit(a_read("doomed"))
        assert p.stats.lost == 1
        assert p.stats.as_dict()["last_error"]
    finally:
        queue.chmod(0o644)


def test_settling_a_duplicate_read_id_removes_one_copy_not_both(tmp_path):
    """The contract says duplicate read_ids are normal -- a re-send, a restored
    backup. Removing by identity took out a copy whose delivery had just failed
    and which was still owed to the consumer."""
    delivered = []
    calls = {"n": 0}

    def once_then_fail(url, payload, timeout):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("down")
        delivered.append(payload["identity"]["plate"])

    p = ReadPusher("http://consumer.invalid/reads", tmp_path / "q.jsonl", opener=once_then_fail)
    first = a_read("dup")
    second = Read.from_dict({**a_read("dup").to_dict(), "identity": {"plate": "CAR-B"}})
    p.queue.append(first)
    p.queue.append(second)

    p.flush()
    assert delivered == ["ABC123"]
    remaining = p.queue.load()
    assert len(remaining) == 1, "both copies were removed; one was still owed"
    assert remaining[0].identity.plate == "CAR-B"


# --- the queue's mode, across the ordinary life of the file -----------------


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def widest_umask():
    """Run with a umask that grants everything.

    The mode this test asserts must come from the code, not from whatever the
    runner's umask happens to be. Under 0o000 every file this module creates
    would be 0o666 unless something narrows it deliberately, so the fixture is
    what gives the assertion its meaning.
    """
    previous = os.umask(0o000)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.guarantee
def test_the_queue_is_owner_only_for_its_whole_life_not_only_when_created(tmp_path, widest_umask):
    """The contract says the queue is protected as a credential is.

    It was protected only until the first successful delivery. `settle()` --
    the ordinary path after every delivery -- writes a sibling at the process
    umask and renames it over the queue, so the mode set when the file was
    created was undone by nothing more unusual than the thing working, and
    never restored: subsequent appends saw a file that already existed.

    Nothing here is torn, damaged or interrupted. This is the queue doing its
    job, and the plates in it were world-readable from the first car onwards.
    """
    queue_path = tmp_path / "push-queue.jsonl"
    consumer = Consumer()
    pusher = ReadPusher("http://consumer.invalid/reads", queue_path, opener=consumer)

    consumer.mode = "down"
    pusher.submit(a_read("first"))
    assert _mode(queue_path) & 0o077 == 0, "a queue holding plates is readable by others"
    assert _mode(queue_path) == QUEUE_FILE_MODE

    pusher.submit(a_read("second"))
    assert _mode(queue_path) & 0o077 == 0

    consumer.mode = "ok"
    pusher.flush()
    assert len(consumer.received) == 2, "the deliveries this test turns on did not happen"
    assert _mode(queue_path) & 0o077 == 0, "delivery widened the queue's mode"
    assert _mode(queue_path) == QUEUE_FILE_MODE

    pusher.submit(a_read("third"))
    assert _mode(queue_path) & 0o077 == 0, "the mode was never restored after the first delivery"
    assert _mode(queue_path) == QUEUE_FILE_MODE


@pytest.mark.guarantee
def test_the_quarantine_file_is_owner_only_too(tmp_path, widest_umask):
    """It holds the same lines the queue does, so it needs the same protection.
    A torn line is quarantined verbatim -- plate and all."""
    queue_path = tmp_path / "push-queue.jsonl"
    pusher = ReadPusher("http://consumer.invalid/reads", queue_path, opener=Consumer())
    pusher.queue.append(a_read("good"))
    with queue_path.open("a", encoding="utf-8") as fh:
        fh.write("{this line was torn by a power cut\n")

    pusher.queue.load()

    damaged = pusher.queue.damaged
    assert damaged.exists(), "nothing was quarantined; this test proved nothing"
    assert "torn by a power cut" in damaged.read_text(encoding="utf-8")
    assert _mode(damaged) & 0o077 == 0, "the quarantine file is readable by others"
    assert _mode(queue_path) & 0o077 == 0


@pytest.mark.guarantee
def test_a_queue_file_that_already_exists_is_narrowed_by_the_next_append(tmp_path, widest_umask):
    """A queue can arrive from somewhere other than this code.

    Restored from a backup, recreated by an operator with a text editor, left
    behind by an older build. The mode was applied only when the file did not
    previously exist, so every one of those cases appended plates to a file
    nobody had narrowed. This is the case that reaches the append path with no
    delivery anywhere near it.
    """
    queue_path = tmp_path / "push-queue.jsonl"
    queue_path.write_text("", encoding="utf-8")
    queue_path.chmod(0o666)
    assert _mode(queue_path) & 0o077 != 0, "the fixture did not start wide; it proves nothing"

    pusher = ReadPusher("http://consumer.invalid/reads", queue_path, opener=Consumer())
    pusher.queue.append(a_read("into-a-file-someone-else-made"))

    assert _mode(queue_path) & 0o077 == 0, "an existing queue file was never narrowed"
    assert _mode(queue_path) == QUEUE_FILE_MODE
