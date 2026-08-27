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
        if not self.path.exists():
            return []
        reads = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                reads.append(Read.from_dict(json.loads(line)))
        return reads

    def replace(self, reads: list[Read]) -> None:
        """Rewrite the queue with what is still outstanding.

        Written to a sibling then moved, so an interruption leaves either the
        old queue or the new one and never a half-written file.
        """
        with self._lock:
            scratch = self.path.with_suffix(self.path.suffix + ".partial")
            with scratch.open("w", encoding="utf-8") as fh:
                for read in reads:
                    fh.write(json.dumps(read.to_dict()) + "\n")
                fh.flush()
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
        opener=None,
    ) -> None:
        self.url = url
        self.queue = ReadQueue(queue_path)
        self.timeout = timeout
        self.stats = PushStats()
        # Injected so the tests exercise the real retry and refusal logic
        # against a stub, rather than testing a mock of the logic itself.
        self._open = opener or _post

    def submit(self, read: Read) -> None:
        """Durably queue a read, then try to deliver everything outstanding."""
        self.queue.append(read)
        self.flush()

    def flush(self) -> PushStats:
        outstanding = self.queue.load()
        if not outstanding:
            self.stats.pending = 0
            return self.stats

        remaining: list[Read] = []
        blocked = False
        for read in outstanding:
            if blocked:
                # Order is preserved: once one delivery fails, everything behind
                # it waits. A consumer that receives an exit before the entry it
                # belongs to is a consumer that prices the stay wrong.
                remaining.append(read)
                continue
            try:
                self._open(self.url, read.to_dict(), self.timeout)
            except Refused as refusal:
                self.stats.refused += 1
                log.warning(
                    "consumer refused read %s with %s; dropping it rather than "
                    "blocking the queue behind it",
                    read.read_id,
                    refusal.status,
                )
            except Exception as exc:  # transport, timeout, consumer down
                log.info("delivery of read %s deferred: %s", read.read_id, exc)
                remaining.append(read)
                blocked = True
            else:
                self.stats.delivered += 1

        self.queue.replace(remaining)
        self.stats.pending = len(remaining)
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
