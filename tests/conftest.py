"""Test configuration, and the rule that a guarantee may not skip silently.

Three tests proving the presence gate was wired into the engine skipped in every
CI run for a fortnight. They skipped for a defensible reason -- they needed
weights better than the ones CI trains -- and the build was green throughout, so
nothing said the guarantee was unproven. The file header even claimed the
opposite: "CI trains a small model first, so the guarantee is enforced there
rather than being a test nobody ever runs."

A test that can quietly not run is not a guarantee. So:

  * mark a test `@pytest.mark.guarantee` when it proves something the README or
    the contract states as a promise;
  * a guarantee that skips FAILS the run, unless its reason is named in
    `VEHICLE_ID_ALLOW_SKIPPED` -- a comma-separated list of substrings matched
    against the skip reason.

The allowance is deliberately a reason rather than a test id: naming
`needs weights` in the job that has no weights is a statement about that job,
and it keeps working when a test is added. Naming ids would rot into a list
nobody reads.
"""

from __future__ import annotations

import os

import pytest

ALLOW_ENV = "VEHICLE_ID_ALLOW_SKIPPED"

#: Modules that carry guarantees and may not silently stop being collected.
#: A collection-time skip cannot tell us which tests inside were marked, so the
#: module is named instead. Keep this in step with the `@guarantee` marks.
GUARANTEE_MODULES = (
    "test_presence.py",
    "test_presence_wiring.py",
    "test_contract.py",
    "test_measured_docs.py",
    "test_plates.py",
    "test_push.py",
)

_skipped: list[tuple[str, str]] = []


def _allowances() -> list[str]:
    raw = os.environ.get(ALLOW_ENV, "")
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _reason(report) -> str:
    if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        return str(report.longrepr[2])
    return str(report.longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.get_closest_marker("guarantee"):
        _skipped.append((item.nodeid, _reason(report)))


@pytest.hookimpl(hookwrapper=True)
def pytest_collectreport(report):
    """The half the first version could not see, and it was the half in use.

    `pytest_runtest_makereport` fires for collected ITEMS. A module that calls
    `pytest.importorskip` at import time never produces items -- it is skipped
    during COLLECTION -- so every guarantee in it vanished without appearing
    here, without a warning, and with the run exiting 0. Both presence modules
    open with `importorskip`, so the mechanism this guard exists for was the one
    it could not observe: torch absent with cv2 present drops eight wiring
    guarantees and the build stays green. L3 verified that against this file.

    A skipped module cannot be asked which of its tests were guarantees, because
    it was never imported. So the rule is coarser and deliberately so: a module
    listed in GUARANTEE_MODULES may not vanish. Naming them is the point -- a
    module that carries a promise is a thing somebody decided, and it should
    take a decision to stop running it.
    """
    outcome = yield
    del outcome
    if report.skipped and any(name in str(report.nodeid) for name in GUARANTEE_MODULES):
        _skipped.append((str(report.nodeid), _reason(report)))


def pytest_sessionfinish(session, exitstatus):
    if not _skipped:
        return
    allowed = _allowances()
    unexplained = [
        (nodeid, reason)
        for nodeid, reason in _skipped
        if not any(token in reason.lower() for token in allowed)
    ]
    if not unexplained:
        return
    lines = "\n".join(f"    {nodeid}\n        {reason}" for nodeid, reason in unexplained)
    session.config.stash  # noqa: B018 - keep the config referenced for clarity
    print(
        "\nGUARANTEE TESTS SKIPPED WITHOUT AN ALLOWANCE:\n"
        f"{lines}\n"
        f"\nA guarantee whose test does not run is not a guarantee. Either make it\n"
        f"run, or state in {ALLOW_ENV} why this job cannot -- e.g.\n"
        f"    {ALLOW_ENV}='needs weights'\n"
    )
    session.exitstatus = 1
