"""The skip guard's own control. A guard that has never caught anything is not
known to work.

`conftest.py` exists because three presence guarantees skipped in every CI run
for a fortnight while the build stayed green. Its first version hooked
`pytest_runtest_makereport`, which fires only for tests that were COLLECTED --
and both presence modules open with `pytest.importorskip`, which skips at
collection time and produces no items at all. So the guard could see the shape
of failure nobody was having, and not the shape everybody was.

L3 found that by writing a throwaway repro. This is that repro, kept: two
pytest sub-runs in a temporary directory, one per shape, each asserting the
guard actually fails the run.

These are deliberately NOT marked `@guarantee`. They test the guard; marking
them would make the guard's own failure route through the guard.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFTEST = ROOT / "tests" / "conftest.py"


def _run(tmp_path: Path, files: dict[str, str], env_extra: dict | None = None):
    """A pytest run in its own directory, against the real conftest."""
    import os

    suite = tmp_path / "tests"
    suite.mkdir(parents=True, exist_ok=True)
    (suite / "conftest.py").write_text(CONFTEST.read_text(encoding="utf-8"), encoding="utf-8")
    for name, body in files.items():
        (suite / name).write_text(body, encoding="utf-8")

    env = dict(os.environ)
    env.pop("VEHICLE_ID_ALLOW_SKIPPED", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )


HEALTHY = """
import pytest

@pytest.mark.guarantee
def test_something_that_runs():
    assert True
"""

INLINE_SKIP = """
import pytest

@pytest.mark.guarantee
def test_inline():
    pytest.skip("an unnamed reason")
"""

# The shape that was in use and could not be seen. The filename matters: the
# collection-time half of the guard keys on the module being a named guarantee
# module, because a module that was never imported cannot be asked what is in it.
COLLECTION_SKIP = """
import pytest

pytest.importorskip("a_module_that_is_definitely_not_installed")

@pytest.mark.guarantee
def test_never_collected():
    assert True
"""


def test_an_inline_guarantee_skip_fails_the_run(tmp_path):
    """The shape the first version could see. Kept so a rewrite cannot lose it."""
    result = _run(tmp_path, {"test_inline_skip.py": INLINE_SKIP, "test_ok.py": HEALTHY})
    assert result.returncode != 0, result.stdout
    assert "GUARANTEE TESTS SKIPPED WITHOUT AN ALLOWANCE" in result.stdout


def test_a_whole_guarantee_MODULE_vanishing_fails_the_run(tmp_path):
    """L3's repro, and the reason this file exists.

    A healthy module runs beside it deliberately: with only the skipped module
    present pytest exits 5 for "no tests collected", which would make this pass
    for a reason that has nothing to do with the guard. That accident is exactly
    what was protecting the engine job, and it protects nothing when one module
    imports and the other does not.
    """
    result = _run(
        tmp_path,
        {"test_presence.py": COLLECTION_SKIP, "test_ok.py": HEALTHY},
    )
    assert result.returncode != 0, (
        "a guarantee module skipped at collection time did not fail the run:\n"
        + result.stdout
    )
    assert "GUARANTEE TESTS SKIPPED WITHOUT AN ALLOWANCE" in result.stdout
    assert "test_presence.py" in result.stdout


def test_a_named_allowance_still_lets_a_declared_skip_through(tmp_path):
    """The control. A guard that failed every run regardless would pass both
    tests above and stop anyone from declaring a legitimate skip -- which is how
    a guard gets switched off entirely."""
    result = _run(
        tmp_path,
        {"test_presence.py": COLLECTION_SKIP, "test_ok.py": HEALTHY},
        env_extra={"VEHICLE_ID_ALLOW_SKIPPED": "could not import"},
    )
    assert result.returncode == 0, (
        "a declared, named skip was refused; the allowance no longer works:\n"
        + result.stdout
    )


def test_an_ordinary_module_may_still_skip_at_collection(tmp_path):
    """Only named guarantee modules are held to this. An optional-dependency
    test file that skips wholesale is normal and must stay normal."""
    result = _run(
        tmp_path,
        {"test_something_optional.py": COLLECTION_SKIP, "test_ok.py": HEALTHY},
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("module", ["test_presence.py", "test_presence_wiring.py"])
def test_the_modules_that_importorskip_are_named_in_the_guard(module):
    """The guard keys on a list of module names, and a list nobody checks rots.

    Every module that can skip at collection time must be in it, or the hole
    reopens silently for that file.
    """
    from conftest import GUARANTEE_MODULES

    source = (ROOT / "tests" / module).read_text(encoding="utf-8")
    assert "importorskip" in source, f"{module} no longer importorskips; update this test"
    assert module in GUARANTEE_MODULES, (
        f"{module} skips at collection time but is not named in GUARANTEE_MODULES, "
        "so its guarantees can vanish without failing the run"
    )
