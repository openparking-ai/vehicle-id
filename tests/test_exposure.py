"""Two routes to a forged read that need no camera, and the two gates on them.

Both were found by reviewers who asked the only question our own rounds never
did: not "is what this publishes honest" but "what would I attack". They are the
same shape — something that is not a camera puts a record into a consumer's
stream — approached from opposite ends:

  * over the network, because `--host` was one flag from the LAN with nothing
    behind it (S4);
  * over the filesystem, because the queue file's mode says who may READ it and
    nothing said who may WRITE its directory, and a line written into that file
    by hand is loaded and delivered as genuine (S5).

Neither is closed by any amount of camera work, which is why they are here and
not in the presence suite.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from vehicle_id.contract import ANSWER, Engine, Identity, Read, utc_now
from vehicle_id.push import (
    QUEUE_DIR_MODE,
    QueueDirectoryUnsafe,
    ReadQueue,
    ensure_queue_directory,
    queue_directory_fault,
)
from vehicle_id.service import (
    InsecureBind,
    VehicleIdService,
    bearer,
    is_loopback,
    make_server,
)

TOKEN = "a-shared-token-for-this-lane"  # not a credential to anything; test fixture


class StubEngine:
    """The engine seam, answering the same record every time.

    What produced the text is invisible to everything downstream, which is the
    whole point of these two findings: a record is a record whether a camera or
    a stranger caused it.
    """

    engine = Engine(name="stub", version="0.0.0", weights_id="sha256:stub")
    threshold = 0.9

    def read(self, captures):
        return Read(
            read_id="rd_stub",
            captured_at=utc_now(),
            camera_id=captures[0].camera_id,
            identity=Identity(plate="ABC123"),
            confidence=0.99,
            engine=self.engine,
            threshold_applied=self.threshold,
            outcome=ANSWER,
        )


@pytest.fixture
def served():
    """A real server on a real socket. The auth path is HTTP, so the test is."""

    def _serve(token=None, host="127.0.0.1"):
        server = make_server(VehicleIdService(StubEngine()), host=host, port=0, token=token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    servers = []

    def build(token=None, host="127.0.0.1"):
        server, base = _serve(token, host)
        servers.append(server)
        return base

    yield build
    for server in servers:
        server.shutdown()
        server.server_close()


def call(url: str, token: str | None = None, method: str = "GET") -> int:
    data = None
    headers = {}
    if method == "POST":
        image = base64.b64encode(b"\x89PNG\r\n\x1a\n<not from a camera>").decode()
        data = json.dumps(
            {"captures": [{"image_b64": image}], "camera_id": "not-a-camera"}
        ).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


# ---------------------------------------------------------------------------
# S4 — the bind
# ---------------------------------------------------------------------------


@pytest.mark.guarantee
def test_a_non_loopback_bind_with_no_token_is_refused():
    for host in ("0.0.0.0", "", "192.0.2.10", "::", "not-a-name-we-can-prove"):
        with pytest.raises(InsecureBind, match="refusing to bind"):
            make_server(VehicleIdService(StubEngine()), host=host, port=0)


def test_loopback_with_no_token_is_unchanged_which_is_every_deployment():
    server = make_server(VehicleIdService(StubEngine()), host="127.0.0.1", port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_what_counts_as_loopback_and_what_does_not():
    """The control on the check above: it must say yes to something, or the
    refusal is a refusal of everything and proves nothing."""
    assert is_loopback("127.0.0.1")
    assert is_loopback("127.0.0.53")
    assert is_loopback("::1")
    assert is_loopback("localhost")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("")
    assert not is_loopback("192.0.2.10")
    # A hostname cannot be PROVEN loopback here -- it resolves at bind time and
    # can resolve to more than one address -- so it is not treated as one.
    assert not is_loopback("gate-controller.local")


@pytest.mark.guarantee
def test_a_read_route_refuses_a_request_with_no_token(served):
    base = served(token=TOKEN)
    assert call(f"{base}/v1/reads", method="POST") == 401
    assert call(f"{base}/v1/reads/last") == 401
    assert call(f"{base}/v1/reads?since=0") == 401


@pytest.mark.guarantee
def test_a_read_route_refuses_a_wrong_or_malformed_token(served):
    base = served(token=TOKEN)
    for header in (f"Bearer {TOKEN}x", f"Bearer {TOKEN[:-1]}", "Bearer ", "Bearer"):
        request = urllib.request.Request(
            f"{base}/v1/reads/last", headers={"Authorization": header}
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError(f"{header!r} was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, header


@pytest.mark.guarantee
def test_the_right_token_is_accepted(served):
    """The control. A gate that refuses everything is not a gate."""
    base = served(token=TOKEN)
    assert call(f"{base}/v1/reads", TOKEN, method="POST") == 200
    assert call(f"{base}/v1/reads/last", TOKEN) == 200
    assert call(f"{base}/v1/reads?since=0", TOKEN) == 200


@pytest.mark.guarantee
def test_with_no_token_configured_nothing_changes(served):
    base = served(token=None)
    assert call(f"{base}/v1/reads", method="POST") == 200
    assert call(f"{base}/v1/reads/last") == 200


def test_health_stays_open_because_it_carries_no_read(served):
    base = served(token=TOKEN)
    assert call(f"{base}/v1/health") == 200


def test_the_authorization_header_is_parsed_and_not_pattern_matched():
    assert bearer("Bearer abc") == "abc"
    assert bearer("bearer abc") == "abc"
    assert bearer("Bearer  abc  ") == "abc"
    assert bearer(None) is None
    assert bearer("") is None
    assert bearer("Basic abc") is None
    assert bearer("abc") is None
    assert bearer("Bearer") is None


# ---------------------------------------------------------------------------
# S5 — the queue directory
# ---------------------------------------------------------------------------


@pytest.mark.guarantee
def test_an_absent_queue_directory_is_created_narrow(tmp_path):
    directory = tmp_path / "var" / "queue"
    ReadQueue(directory / "push-queue.jsonl")
    assert stat.S_IMODE(directory.stat().st_mode) == QUEUE_DIR_MODE


def test_the_mode_is_set_explicitly_and_not_left_to_the_umask(tmp_path):
    """A bare `mkdir()` takes the default 0777 and lets the umask decide what
    survives -- 0755 under the usual 022, 0777 under a permissive one. The
    chmod is what makes the result a rule rather than a property of how the
    service happened to be started, so the umask is the deciding axis here."""
    previous = os.umask(0o000)
    try:
        directory = tmp_path / "under-a-wide-umask"
        ReadQueue(directory / "push-queue.jsonl")
        assert stat.S_IMODE(directory.stat().st_mode) == QUEUE_DIR_MODE

        # The control: the same umask and a plain mkdir, so the axis is shown
        # to be live rather than this passing because nothing was at stake.
        loose = tmp_path / "plain-mkdir"
        loose.mkdir()
        assert stat.S_IMODE(loose.stat().st_mode) == 0o777
    finally:
        os.umask(previous)


@pytest.mark.guarantee
def test_a_directory_owned_by_somebody_else_is_refused():
    """Handing a directory to another user needs root, so the DECISION is
    tested rather than the syscall. Both sides, because a fault function that
    always faults would pass the refusal half and measure nothing."""
    assert queue_directory_fault(0o700, owner_uid=0, process_uid=501) is not None
    assert "owned by uid 0" in queue_directory_fault(0o700, owner_uid=0, process_uid=501)
    assert queue_directory_fault(0o700, owner_uid=501, process_uid=501) is None
    # Ownership before mode. A directory that is BOTH somebody else's AND wide
    # is refused either way, so what the order decides is which reason the
    # operator is given -- and "chmod it" sends them to fix the wrong thing
    # when the directory is not theirs to chmod.
    assert "owned by uid" in queue_directory_fault(0o777, owner_uid=0, process_uid=1)
    assert "wider than" in queue_directory_fault(0o777, owner_uid=1, process_uid=1)


@pytest.mark.guarantee
def test_a_directory_anyone_can_write_is_refused(tmp_path):
    """The forgery path this closes: a hand-written line in the queue file is
    loaded, built into a Read and delivered to the consumer as genuine."""
    directory = tmp_path / "wide"
    directory.mkdir(mode=0o777)
    directory.chmod(0o777)

    with pytest.raises(QueueDirectoryUnsafe, match="wider than"):
        ReadQueue(directory / "push-queue.jsonl")


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o707, 0o701, 0o750, 0o710, 0o702])
def test_every_bit_outside_the_owner_is_refused(mode, tmp_path):
    directory = tmp_path / f"mode-{mode:04o}"
    directory.mkdir()
    directory.chmod(mode)
    with pytest.raises(QueueDirectoryUnsafe):
        ensure_queue_directory(directory)


@pytest.mark.parametrize("mode", [0o700, 0o600, 0o500, 0o400])
def test_an_owner_only_directory_is_accepted(mode, tmp_path):
    """The control on the parametrised refusal above. Both sides of the
    deciding bits are exercised: a check that refuses every mode would pass
    that test and measure nothing."""
    directory = tmp_path / f"ok-{mode:04o}"
    directory.mkdir()
    directory.chmod(mode)
    ensure_queue_directory(directory)  # must not raise
    directory.chmod(0o700)  # so tmp_path can be cleaned up


@pytest.mark.guarantee
def test_widening_the_directory_under_a_running_queue_is_caught_on_the_next_start(tmp_path):
    """The lifecycle the file-mode fix did not cover. The queue is built, used,
    then the directory is widened -- by a deploy script, a backup restore, an
    operator with a problem -- and the next start refuses instead of carrying
    on delivering out of a directory anyone can write."""
    directory = tmp_path / "var"
    queue = ReadQueue(directory / "push-queue.jsonl")
    queue.append(
        Read(
            read_id="rd_1",
            captured_at=utc_now(),
            camera_id="lane-1",
            identity=Identity(plate="ABC123"),
            confidence=0.99,
            engine=Engine(name="stub", version="0.0.0"),
            threshold_applied=0.9,
            outcome=ANSWER,
        )
    )
    assert len(queue) == 1

    directory.chmod(0o777)
    with pytest.raises(QueueDirectoryUnsafe):
        ReadQueue(directory / "push-queue.jsonl")

    directory.chmod(QUEUE_DIR_MODE)
    restored = ReadQueue(directory / "push-queue.jsonl")
    assert len(restored) == 1, "the reads are still there; only the directory was the problem"


def test_a_path_that_is_not_a_directory_is_refused(tmp_path):
    not_a_directory = tmp_path / "a-file"
    not_a_directory.write_text("")
    with pytest.raises(QueueDirectoryUnsafe, match="not a directory"):
        ensure_queue_directory(not_a_directory)


# --- the leaf is not the path ----------------------------------------------


@pytest.mark.guarantee
def test_the_ancestor_decision_trusts_root_and_refuses_a_writable_component():
    """The decision for an ANCESTOR, both sides of both branches.

    It is weaker than the leaf's in exactly two places, and each is forced.
    Every path ends at `/`, which is root-owned everywhere, so requiring each
    component to belong to this process refuses every path there is. And `/`,
    `/var` and `/private` are 0755 everywhere, so requiring 0700 of each
    component does the same. What an ancestor may not be is WRITABLE by anyone
    else -- that is the bit that lets a stranger rename the queue aside.
    """
    ancestor = dict(process_uid=501, leaf=False)
    assert queue_directory_fault(0o755, owner_uid=0, **ancestor) is None
    assert queue_directory_fault(0o755, owner_uid=501, **ancestor) is None
    assert queue_directory_fault(0o775, owner_uid=501, **ancestor) is not None
    assert queue_directory_fault(0o757, owner_uid=0, **ancestor) is not None
    assert queue_directory_fault(0o755, owner_uid=502, **ancestor) is not None

    # The control on the parameter itself: the same three values that an
    # ancestor may hold are refused for the LEAF, so `leaf=False` is doing
    # something rather than being an argument nothing reads.
    assert queue_directory_fault(0o755, owner_uid=0, process_uid=501) is not None
    assert queue_directory_fault(0o755, owner_uid=501, process_uid=501) is not None


@pytest.mark.guarantee
def test_a_sticky_world_writable_ancestor_is_accepted_and_a_plain_one_is_not():
    """`/tmp` is 1777 on every system there is, and the sticky bit is the rule
    this check actually needs: only the owner of an entry may rename or remove
    it, so a stranger cannot swap the queue directory out. Refusing it would
    reject every path under /tmp and answer with `chmod go-w /tmp`, which is
    worse advice than the check is protection. The pair is the point -- the
    same mode without the bit is still refused."""
    ancestor = dict(owner_uid=0, process_uid=501, leaf=False)
    assert queue_directory_fault(0o1777, **ancestor) is None
    assert queue_directory_fault(0o777, **ancestor) is not None
    # And it buys an ANCESTOR nothing at the leaf: the plates in the queue are
    # readable by anyone the mode lets in, sticky or not.
    assert queue_directory_fault(0o1777, owner_uid=501, process_uid=501) is not None


@pytest.mark.guarantee
def test_a_writable_ancestor_is_refused_under_a_perfectly_narrow_leaf(tmp_path):
    """The hole the leaf check could not see. The queue directory itself is
    0700 and owned by this process, and it is still not safe: anything that can
    write the parent can rename it aside and put its own directory there."""
    spool = tmp_path / "var" / "spool"
    directory = spool / "queue"
    directory.mkdir(parents=True)
    directory.chmod(QUEUE_DIR_MODE)
    spool.chmod(0o777)

    with pytest.raises(QueueDirectoryUnsafe, match="ancestor"):
        ReadQueue(directory / "push-queue.jsonl")

    spool.chmod(0o755)  # so tmp_path can be cleaned up


def test_a_correct_chain_starts(tmp_path):
    """The control. Without it the test above is satisfied by a walk that
    refuses every path in existence -- which is what a literal reading of "the
    leaf's rule, applied to every ancestor" produces, because `/` is root's."""
    directory = tmp_path / "var" / "spool" / "queue"
    queue = ReadQueue(directory / "push-queue.jsonl")

    assert queue.load() == []
    for component in (directory, directory.parent, directory.parent.parent):
        assert stat.S_IMODE(component.stat().st_mode) == QUEUE_DIR_MODE, (
            "an intermediate created by the service was left at the umask"
        )


@pytest.mark.guarantee
def test_a_relative_queue_path_is_refused(tmp_path, monkeypatch):
    """It resolves against whatever directory the service was started in, so
    the security of the queue would be decided by the caller's shell. Refused
    with the reason named rather than resolved silently."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(QueueDirectoryUnsafe, match="relative"):
        ReadQueue(Path("var/push-queue.jsonl"))

    # The control: the same path made absolute is accepted, so what is being
    # refused is the relativity and not the path.
    ReadQueue(tmp_path / "var" / "push-queue.jsonl")


@pytest.mark.guarantee
def test_a_queue_directory_substituted_under_a_writable_parent_is_not_loaded(tmp_path):
    """The forgery the walk closes, run the way it was found.

    The attacker cannot write the queue directory -- it is 0700 -- so they
    rename it aside and put their own in its place, which the parent lets them
    do. Nothing holds a directory handle, so a service that started here would
    read the substitute on its next load. It does not start.

    What this does NOT close, and it is the same residual the widening test
    above records: a parent widened AFTER a queue object exists is caught on
    the next start, not while running.
    """
    spool = tmp_path / "var" / "spool"
    directory = spool / "queue"
    directory.mkdir(parents=True)
    directory.chmod(QUEUE_DIR_MODE)
    spool.chmod(0o777)

    forged = spool / "queue-forged"
    forged.mkdir()
    forged.chmod(QUEUE_DIR_MODE)
    (forged / "push-queue.jsonl").write_text(
        json.dumps(
            {
                "read_id": "FAKE-999",
                "captured_at": utc_now(),
                "camera_id": "lane-1",
                "identity": {"plate": "FAKE99"},
                "confidence": 0.99,
                "engine": {"name": "stub", "version": "0.0.0"},
                "threshold_applied": 0.9,
                "outcome": ANSWER,
            }
        )
        + "\n"
    )
    directory.rename(spool / "queue-real")
    forged.rename(directory)

    with pytest.raises(QueueDirectoryUnsafe, match="ancestor"):
        ReadQueue(directory / "push-queue.jsonl")

    spool.chmod(0o755)
