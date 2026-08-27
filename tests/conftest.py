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

_skipped: list[tuple[str, str]] = []


def _allowances() -> list[str]:
    raw = os.environ.get(ALLOW_ENV, "")
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.get_closest_marker("guarantee"):
        reason = ""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = str(report.longrepr[2])
        else:
            reason = str(report.longrepr)
        _skipped.append((item.nodeid, reason))


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
