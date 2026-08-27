"""Push delivery: every read reaches the consumer, or waits on disk until it can.

The design rule is the one the lane controller already learned the hard way: a
consumer being down must never lose a read. So a read is written to a durable
queue BEFORE any delivery is attempted, and removed only once the consumer has
acknowledged it. A process killed between those two points re-sends; it does not
forget.

Re-sending means duplicates are normal, not exceptional, so `read_id` is stable
across every attempt and the consumer is expected to deduplicate on it. That is
the same bargain the lane's outbox strikes with the platform, for the same
reason.

A 4xx is not retried. The consumer understood the read and said no; retrying
poison forever blocks everything queued behind it. Refused reads are counted and
logged -- never silently dropped, because a lost read is a car nobody can
account for.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .contract import Read

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0

#: How often a pusher with something outstanding tries again on its own,
#: rather than waiting for the next vehicle to arrive.
DEFAULT_RETRY_INTERVAL = 15.0


@dataclass
class PushStats:
    delivered: int = 0
    refused: int = 0
    pending: int = 0


class ReadQueue:
    """An append-only JSONL file of reads not yet acknowledged.

    Deliberately a file and not a database. It has to survive a power cut in a
    gate housing, and it has to be readable by whoever is standing in front of
    that housing at 2am with no tooling.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, read: Read) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(read.to_dict()) + "\n")
            fh.flush()

    def load(self) -> list[Read]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> list[Read]:
        if not self.path.exists():
            return []
        reads = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                reads.append(Read.from_dict(json.loads(line)))
        return reads

    def settle(self, settled: set[str]) -> None:
        """Remove exactly the reads named, and nothing else.

        Deliberately NOT "rewrite the queue with what is left over", which is
        how this was first written and how it lost reads. A delivery attempt
        takes seconds; another request thread appends during it; rewriting the
        file from a snapshot taken before that append truncates the new read
        off the queue forever -- and it had already been answered 200.

        Re-reading under the lock and removing by identity means anything that
        arrived mid-flight is still there afterwards.
        """
        with self._lock:
            keep = [read for read in self._load_unlocked() if read.read_id not in settled]
            scratch = self.path.with_suffix(self.path.suffix + ".partial")
            with scratch.open("w", encoding="utf-8") as fh:
                for read in keep:
                    fh.write(json.dumps(read.to_dict()) + "\n")
                fh.flush()
            # Written to a sibling then moved, so an interruption leaves either
            # the old queue or the new one and never a half-written file.
            scratch.replace(self.path)

    def __len__(self) -> int:
        return len(self.load())


class ReadPusher:
    """POSTs each read to a configured URL, and keeps the ones it could not."""

    def __init__(
        self,
        url: str,
        queue_path: Path,
        timeout: float = DEFAULT_TIMEOUT,
        retry_interval: float = DEFAULT_RETRY_INTERVAL,
        opener=None,
    ) -> None:
        self.url = url
        self.queue = ReadQueue(queue_path)
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.stats = PushStats()
        # Only one flush runs at a time. Two concurrent flushers would each
        # load the same outstanding read and deliver it twice -- survivable,
        # because read_id is stable and consumers deduplicate on it, but there
        # is no reason to make them do that work.
        self._flushing = threading.Lock()
        self._stop = threading.Event()
        self._retrier: threading.Thread | None = None
        # Injected so the tests exercise the real retry and refusal logic
        # against a stub, rather than testing a mock of the logic itself.
        self._open = opener or _post

    def submit(self, read: Read) -> None:
        """Durably queue a read, then try to deliver everything outstanding."""
        self.queue.append(read)
        self.flush()

    def start(self) -> None:
        """Flush now, and keep retrying on a timer.

        Both halves matter and neither is optional. Without the flush at start,
        reads outstanding when the process died wait for the next vehicle --
        which, for the last car of the night, is the next morning. Without the
        timer, retry is coupled to new traffic, so a consumer that comes back
        during a quiet hour is not noticed until someone drives in.
        """
        self.flush()
        if self._retrier is None:
            self._retrier = threading.Thread(target=self._retry_loop, daemon=True)
            self._retrier.start()

    def stop(self) -> None:
        self._stop.set()

    def _retry_loop(self) -> None:
        while not self._stop.wait(self.retry_interval):
            if self.stats.pending:
                self.flush()

    def flush(self) -> PushStats:
        with self._flushing:
            outstanding = self.queue.load()
            if not outstanding:
                self.stats.pending = 0
                return self.stats

            settled: set[str] = set()
            deferred = 0
            for read in outstanding:
                if deferred:
                    # Order is preserved: once one delivery fails, everything
                    # behind it waits. A consumer that receives an exit before
                    # the entry it belongs to prices the stay wrong.
                    break
                try:
                    self._open(self.url, read.to_dict(), self.timeout)
                except Refused as refusal:
                    self.stats.refused += 1
                    settled.add(read.read_id)
                    log.warning(
                        "consumer refused read %s with %s; dropping it rather than "
                        "blocking the queue behind it",
                        read.read_id,
                        refusal.status,
                    )
                except Exception as exc:  # transport, timeout, consumer down
                    log.info("delivery of read %s deferred: %s", read.read_id, exc)
                    deferred += 1
                else:
                    self.stats.delivered += 1
                    settled.add(read.read_id)

            # Remove exactly what was settled. Anything appended by another
            # thread while the deliveries above were in flight is still on the
            # queue afterwards, which is the whole point.
            self.queue.settle(settled)
            self.stats.pending = len(self.queue)
            return self.stats


class Refused(Exception):
    """The consumer understood and said no. Not retryable."""

    def __init__(self, status: int) -> None:
        super().__init__(f"consumer refused with {status}")
        self.status = status


def _post(url: str, payload: dict, timeout: float) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status >= 400:
                raise Refused(response.status)
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            raise Refused(exc.code) from exc
        raise
