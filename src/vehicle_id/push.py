"""Push delivery.

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
import os
import stat
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .contract import Read

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0

#: The queue holds plates. It is trusted local state and it is the process's
#: own; nothing else has business reading it.
#:
#: Applied on every write and not only at creation, because the queue file is
#: REPLACED by a sibling on the ordinary post-delivery path -- so a mode set
#: once, when the file was first made, was undone by the first successful
#: delivery and never restored. The siblings carry it too: the quarantine file
#: holds the same lines the queue does, and the scratch file IS the queue one
#: rename later, so it is narrowed before the rename rather than after.
QUEUE_FILE_MODE = 0o600

#: The directory the queue lives in, and it is the half that actually gates the
#: forgery path.
#:
#: The FILE's mode stops a stranger READING the plates in it. It does nothing
#: about writing: a queue in a directory anyone can write to can be replaced,
#: and a line written into it by hand is loaded, built into a `Read` and
#: DELIVERED to the consumer as a genuine one -- no camera, no image, no engine.
#: `_load_unlocked` trusts whatever JSON parses, deliberately, because the
#: alternative is a service that dies at 2am over a torn line; that trust is
#: only sound while the directory is the process's own.
QUEUE_DIR_MODE = 0o700

#: How often a pusher with something outstanding tries again on its own,
#: rather than waiting for the next vehicle to arrive.
DEFAULT_RETRY_INTERVAL = 15.0


class QueueDirectoryUnsafe(Exception):
    """The queue directory is writable by somebody else, or owned by them.

    Raised at construction, which is before the port opens, because a service
    that starts and then delivers forged reads is worse than one that does not
    start. There is no flag that turns this off: the fix is to narrow the
    directory or point the queue somewhere the service owns.
    """


def queue_directory_fault(
    mode: int, owner_uid: int, process_uid: int, *, leaf: bool = True
) -> str | None:
    """Why this directory is not safe to hold a queue, or None if it is.

    Separated from the syscalls deliberately. Handing a directory to another
    user needs root, so the ownership branch cannot be exercised by creating
    one -- and an untested branch in a security check is the branch that turns
    out to have been `if False:` all along. This is the decision; the function
    below is the plumbing around it.

    Ownership is checked FIRST. A directory somebody else owns is theirs to
    chmod back whenever they like, so a narrow mode on it proves nothing.

    `leaf=False` asks the same question of an ANCESTOR, and the rule is weaker
    in exactly two places because it has to be:

    * a component may be owned by ROOT as well as by this process. Every path
      ends at `/`, which is root-owned on every system there is, so requiring
      each component to be the process's own refuses every path in the universe
      -- measured, not assumed. Root can read the queue, chmod it, or attach to
      the process anyway, so trusting root-owned components loses nothing.
    * a component may be READABLE by others -- `/`, `/var` and `/private` are
      0755 everywhere -- but it may not be WRITABLE by them. Writable is the
      whole attack: anything that can write a component can rename the queue
      directory aside, put its own in place, and have hand-written lines
      delivered as genuine reads. Read on an ancestor gives away nothing; the
      leaf's own 0700 is what keeps the plates private.

    ...and a STICKY world-writable ancestor is not writable in the sense that
    matters. `/tmp` is 1777 on every system there is, and the sticky bit is
    exactly the rule this check needs: only the owner of an entry may rename or
    remove it, so a stranger cannot swap the queue directory out. Without this
    the check refuses every path under `/tmp` and tells the operator to run
    `chmod go-w /tmp`, which is worse advice than the check is protection.
    """
    trusted_owners = {process_uid} if leaf else {process_uid, 0}
    if owner_uid not in trusted_owners:
        return (
            f"it is owned by uid {owner_uid}, not by this process (uid {process_uid}). A queue "
            "directory somebody else owns is a queue somebody else can write a read into, and "
            "a hand-written line is delivered as genuine."
        )
    if leaf:
        forbidden = 0o077
    elif mode & stat.S_ISVTX:
        forbidden = 0
    else:
        forbidden = 0o022
    if mode & forbidden:
        wider_than = QUEUE_DIR_MODE if leaf else 0o755
        return (
            f"it is mode {mode:04o}, which is wider than {wider_than:04o}. Anything that "
            "can write this directory can put a plate into a consumer's stream."
        )
    return None


def ensure_queue_directory(directory: Path) -> None:
    """Create the queue directory narrow, and refuse a PATH that is not the
    process's own -- every component of it, not just the last one.

    Created and then chmodded, rather than relying on `mkdir`'s mode:
    `parents=True` creates every INTERMEDIATE directory at the default mode
    whatever `mode` says, and a bare `mkdir()` leaves even the leaf to the
    process umask -- 0755 under the usual 022 and 0777 under a permissive one.
    Each component this creates is chmodded as it is made, so what the service
    builds does not depend on how the service happened to be started.

    THE LEAF WAS THE ONLY THING CHECKED, AND THE LEAF IS NOT THE PATH. A correct
    0700 directory under a parent anyone can write is not safe: the parent lets
    a stranger rename it aside and put their own in its place, and nothing holds
    a directory handle -- every load re-opens by path -- so a RUNNING service
    picks the substitution up on its next read and delivers hand-written lines
    as genuine reads. The check therefore walks to the root.

    A RELATIVE path is refused outright. It resolves against whatever directory
    the service was started in, which nothing checks, nothing records and
    nothing names -- so the security of the queue would depend on the caller's
    shell.
    """
    if not directory.is_absolute():
        raise QueueDirectoryUnsafe(
            f"refusing a relative queue path: the queue directory ({directory}) resolves "
            "against whatever directory this process was started in, which nothing here "
            "checks and nothing records. Pass an absolute --queue."
        )

    directory = directory.resolve()
    if not directory.exists():
        # Made one component at a time so each is narrowed as it appears.
        # `mkdir(parents=True)` leaves every intermediate at the umask, and
        # those intermediates are exactly what the walk below then refuses.
        for component in reversed([d for d in (directory, *directory.parents) if not d.exists()]):
            component.mkdir()
            component.chmod(QUEUE_DIR_MODE)
    elif not directory.is_dir():
        raise QueueDirectoryUnsafe(f"{directory} exists and is not a directory")

    for component in (directory, *directory.parents):
        info = component.stat()
        fault = queue_directory_fault(
            stat.S_IMODE(info.st_mode),
            info.st_uid,
            os.getuid(),
            leaf=component == directory,
        )
        if fault is None:
            continue
        if component == directory:
            raise QueueDirectoryUnsafe(
                f"refusing to use {directory} as a queue directory: {fault} "
                f"Run: chmod {QUEUE_DIR_MODE:04o} {directory}"
            )
        raise QueueDirectoryUnsafe(
            f"refusing to use {directory} as a queue directory: its ancestor {component} is "
            f"not safe -- {fault} Run: chmod go-w {component}"
        )


@dataclass
class PushStats:
    delivered: int = 0
    refused: int = 0
    pending: int = 0
    #: Reads that could not even be written to the queue. Nothing acknowledged
    #: them and nothing holds them; this is the count of records that existed
    #: for the length of one HTTP response.
    lost: int = 0
    damaged: int = 0
    last_error: str | None = None

    def as_dict(self) -> dict:
        return {
            "delivered": self.delivered,
            "refused": self.refused,
            "pending": self.pending,
            "lost": self.lost,
            "damaged": self.damaged,
            "last_error": self.last_error,
        }


class ReadQueue:
    """An append-only JSONL file of reads not yet acknowledged.

    Deliberately a file and not a database. It has to survive a power cut in a
    gate housing, and it has to be readable by whoever is standing in front of
    that housing at 2am with no tooling.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Before anything is written, and before the service opens a port.
        ensure_queue_directory(self.path.parent)
        #: Lines this build could not read, kept rather than dropped.
        self.damaged = self.path.with_suffix(self.path.suffix + ".damaged")
        self._damaged_count = 0
        self._lock = threading.Lock()

    @property
    def damaged_count(self) -> int:
        return self._damaged_count

    def append(self, read: Read) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(read.to_dict()) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.path.chmod(QUEUE_FILE_MODE)

    def load(self) -> list[Read]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> list[Read]:
        if not self.path.exists():
            return []
        reads: list[Read] = []
        damaged: list[str] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                reads.append(Read.from_dict(json.loads(line)))
            except Exception as exc:
                # A power cut mid-append leaves a partial line. Raising here
                # took the whole service down at startup -- the barrier dead
                # until someone found the bad line at 2am -- which is a far
                # worse failure than losing the one read that was being written
                # when the power went. It is quarantined, counted and shouted
                # about, never silently discarded.
                damaged.append(line)
                log.error("unreadable line in %s, quarantined: %s", self.path, exc)
        if damaged:
            # Moved out of the queue as soon as it is found, once: appended to
            # the quarantine file and removed from the queue in the same breath.
            # Leaving it in place meant every subsequent load re-counted the
            # same torn line, and the tally on /v1/health was a function of how
            # often the queue happened to be read.
            with self.damaged.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(damaged) + "\n")
            self.damaged.chmod(QUEUE_FILE_MODE)
            self._damaged_count += len(damaged)
            self._write_unlocked(reads)
        return reads

    def _write_unlocked(self, reads: list[Read]) -> None:
        scratch = self.path.with_suffix(self.path.suffix + ".partial")
        with scratch.open("w", encoding="utf-8") as fh:
            for read in reads:
                fh.write(json.dumps(read.to_dict()) + "\n")
            fh.flush()
        # Narrowed BEFORE the rename. The scratch file becomes the queue, so
        # doing it afterwards would leave a window in which the queue is
        # readable by anyone -- and an interruption inside that window would
        # leave it that way for good.
        scratch.chmod(QUEUE_FILE_MODE)
        # Written to a sibling then moved, so an interruption leaves either the
        # old queue or the new one and never a half-written file.
        scratch.replace(self.path)

    def settle(self, settled: list[str]) -> None:
        """Remove exactly the reads named, and nothing else.

        Deliberately NOT "rewrite the queue with what is left over", which is
        how this was first written and how it lost reads. A delivery attempt
        takes seconds; another request thread appends during it; rewriting the
        file from a snapshot taken before that append truncates the new read
        off the queue forever -- and it had already been answered 200.

        Re-reading under the lock and removing by occurrence means anything that
        arrived mid-flight is still there afterwards.
        """
        with self._lock:
            # Removed by OCCURRENCE, not by identity. The contract says
            # duplicate read_ids are normal -- a re-send, a restored backup --
            # and removing by id took out every copy, including one whose
            # delivery had just failed and which was still owed to the consumer.
            remaining = list(settled)
            keep = []
            for read in self._load_unlocked():
                if read.read_id in remaining:
                    remaining.remove(read.read_id)
                    continue
                keep.append(read)
            self._write_unlocked(keep)

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
        """Durably queue a read, then try to deliver everything outstanding.

        An append that fails is DATA LOSS, not a deferred delivery, and it is
        raised so the caller cannot log "the read is queued" over the top of it
        -- which is what used to happen with a full disk or an unwritable queue:
        a 200, a reassuring log line, a health check still saying ok, and the
        record nowhere on earth.
        """
        try:
            self.queue.append(read)
        except Exception:
            self.stats.lost += 1
            self.stats.last_error = f"could not write {read.read_id} to {self.queue.path}"
            raise
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

            settled: list[str] = []
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
                    self.stats.last_error = f"consumer refused {read.read_id} with {refusal.status}"
                    settled.append(read.read_id)
                    log.warning(
                        "consumer refused read %s with %s; dropping it rather than "
                        "blocking the queue behind it",
                        read.read_id,
                        refusal.status,
                    )
                except Exception as exc:  # transport, timeout, consumer down
                    log.info("delivery of read %s deferred: %s", read.read_id, exc)
                    self.stats.last_error = f"{type(exc).__name__}: {exc}"
                    deferred += 1
                else:
                    self.stats.delivered += 1
                    settled.append(read.read_id)

            # Remove exactly what was settled. Anything appended by another
            # thread while the deliveries above were in flight is still on the
            # queue afterwards, which is the whole point.
            self.queue.settle(settled)
            self.stats.pending = len(self.queue)
            self.stats.damaged = self.queue.damaged_count
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
