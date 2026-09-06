"""Real data never enters a repository. This is the check that enforces it.

Standard library only, and deliberately its own module rather than something
imported from a harness: the two harnesses that need it -- `eval_real_plates.py`
and `eval_fingerprint.py` -- would otherwise have to import each other, and the
plate one pulls torch in at module scope for reasons that have nothing to do
with where a file lives.

`eval_real_plates.py` carries its own copy of this logic and is deliberately NOT
changed to import this module. Its published evidence object records a
`harness_sha256` over that file, and that digest is the reason someone can check
later that the guard which produced a number was the guard as written. Moving
the guard out would silently narrow what the digest covers -- a smaller change
on the diff and a larger one in what is provable. Instead the duplication is
CHECKED: `tests/test_eval_fingerprint.py` asserts the two implementations agree
on the same inputs, so drift between them turns the suite red rather than
leaving two guards that quietly disagree.
"""

from __future__ import annotations

from pathlib import Path


def inside_git_work_tree(path: Path) -> Path | None:
    """The work tree `path` sits in, or None.

    `.git` is a directory in a clone and a FILE in a worktree, so both count.
    The check walks up from the resolved path, and a path that does not exist
    yet -- an `--out` about to be created -- is answered by its parent.
    """
    here = Path(path).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


class PathInsideRepository(RuntimeError):
    """Raised rather than reading or writing real data inside a repository."""


def refuse_repository_paths(**paths: Path) -> None:
    bad = []
    for name, path in paths.items():
        tree = inside_git_work_tree(path)
        if tree is not None:
            bad.append(f"--{name} {path} is inside the git work tree at {tree}")
    if bad:
        raise PathInsideRepository(
            "real photographs, their labels and this measurement's output never "
            "enter a repository:\n  " + "\n  ".join(bad)
        )
