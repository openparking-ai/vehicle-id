"""What the descriptor may be built from, and what it may never be built from.

The permitted list is short and every exclusion has its own reason, so they are
asserted separately rather than as one "no contrib" rule:

  * **SURF** -- US patent live to 2029-04-13. Not a licence question we can
    settle by reading a repository's LICENSE file.
  * **`xfeatures2d`'s VGG, BoostDesc, BEBLID, TEBLID** -- trained descriptors.
    They DOWNLOAD WEIGHTS at runtime, which is the same licensing problem the
    plate recogniser spent a round escaping by generating its own training data:
    a car is not a rendered rectangle, so that escape does not transfer.
  * **pHash.org's `libpHash`** -- GPLv3 with a paid commercial exception. If a
    perceptual hash is ever wanted it is `imagehash` (BSD-2), never that one.

AKAZE, BRISK and KAZE are a different case and are NOT forbidden: they were
measured ABSENT from `cv2` 5.0.0, which is what an unpinned
`opencv-python-headless>=4.9` resolves to and what CI installs. Pinning `<5` to
keep three detectors this round does not need would pin the whole engine to an
old major. Cutting them from the permitted list was the smaller change, and it is
forward-compatible -- so what is asserted here is that nothing in the package
DEPENDS on them, not that they are unavailable.

**The positive control is the point of this file.** On a headless OpenCV build,
`assert not hasattr(cv2, "xfeatures2d")` passes because contrib is not installed
-- it would pass on an empty environment, and it proves nothing about whether the
project would notice contrib arriving. So the real assertion is about the
INSTALLED DISTRIBUTIONS, and the probe that makes it is itself proven able to see
a contrib installation, by planting one on `sys.path` and watching it be found.
"""

from __future__ import annotations

import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

guarantee = pytest.mark.guarantee

#: Every module in `cv2` that this project must never reach for. Each is
#: forbidden for its OWN reason -- a patent, a downloaded weight file, a licence
#: -- and collapsing them into "contrib" would lose why.
FORBIDDEN_CV2_MODULES = ("xfeatures2d",)
FORBIDDEN_CV2_FACTORIES = (
    "SURF_create",
    "VGG_create",
    "BoostDesc_create",
    "BEBLID_create",
    "TEBLID_create",
)

#: Distributions whose presence would make the forbidden modules importable, or
#: which are themselves licence problems.
FORBIDDEN_DISTRIBUTIONS = (
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "phash",
    "pyphash",
)


def installed_distributions() -> set[str]:
    """Every distribution name importlib can see, lowercased and normalised.

    This is the probe the real assertion rests on, and it is a function so that
    the positive control below can exercise the SAME code that the assertion
    uses. A control that exercised a second copy would prove the copy works.
    """
    names = set()
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip().lower().replace("_", "-")
        if name:
            names.add(name)
    return names


@guarantee
def test_the_distribution_probe_can_actually_see_a_contrib_installation(tmp_path):
    """THE POSITIVE CONTROL, and without it every assertion below is worthless.

    On a headless build nothing forbidden is installed, so a test asserting
    "contrib is absent" passes trivially -- it would pass with the probe
    returning an empty set, or `False`, or nothing at all. So: plant a
    distribution's metadata on `sys.path`, and require the probe to find it.

    A real `dist-info` directory with a real `METADATA` file, because that is
    what `importlib.metadata` actually reads. A stub that monkeypatched the
    probe would be testing the stub.
    """
    assert "opencv-contrib-python" not in installed_distributions(), (
        "opencv-contrib-python is installed in this environment. The forbidden "
        "modules are importable, and the assertions below are no longer about "
        "an environment this project would ship."
    )

    planted = tmp_path / "opencv_contrib_python-4.9.0.80.dist-info"
    planted.mkdir()
    (planted / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: opencv-contrib-python\nVersion: 4.9.0.80\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        assert "opencv-contrib-python" in installed_distributions(), (
            "the probe cannot see a contrib installation even when one is on "
            "sys.path, so its silence about the real environment means nothing"
        )
    finally:
        sys.path.remove(str(tmp_path))

    # And it is gone again, so a later test in the same session is not reading
    # a planted answer.
    assert "opencv-contrib-python" not in installed_distributions()


@guarantee
@pytest.mark.parametrize("name", FORBIDDEN_DISTRIBUTIONS)
def test_no_forbidden_distribution_is_installed(name):
    """The assertion the control above makes meaningful."""
    assert name not in installed_distributions(), (
        f"{name} is installed. It is forbidden for a reason this file states, "
        "and its presence makes the descriptor's licence position untrue."
    )


@guarantee
@pytest.mark.parametrize("module", FORBIDDEN_CV2_MODULES)
def test_the_forbidden_cv2_modules_are_not_reachable(module):
    """The second, weaker assertion. It is kept because it is the one that
    would fire if contrib arrived by some route the distribution probe cannot
    see -- a vendored build, a system OpenCV -- but it is NOT the primary
    check, and this file says so rather than letting a trivially-passing
    assertion look like coverage."""
    cv2 = pytest.importorskip("cv2")
    assert not hasattr(cv2, module), f"cv2.{module} is present"


@guarantee
@pytest.mark.parametrize("factory", FORBIDDEN_CV2_FACTORIES)
def test_the_forbidden_detectors_are_not_reachable(factory):
    cv2 = pytest.importorskip("cv2")
    assert not hasattr(cv2, factory), f"cv2.{factory} is present"
    if hasattr(cv2, "xfeatures2d"):
        assert not hasattr(cv2.xfeatures2d, factory)


@guarantee
def test_the_permitted_detectors_are_present():
    """The other control: if SIFT and ORB were absent too, "nothing forbidden is
    reachable" would be satisfied by an OpenCV with nothing in it."""
    cv2 = pytest.importorskip("cv2")
    for permitted in ("SIFT_create", "ORB_create", "compareHist", "Canny"):
        assert hasattr(cv2, permitted), f"cv2.{permitted} is missing"


@guarantee
def test_the_package_declares_headless_opencv_and_not_contrib():
    """The declared dependency, not just the installed one.

    An environment can be clean while `pyproject.toml` asks for contrib on the
    next install. This reads what the project ASKS for.
    """
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    engine = config["project"]["optional-dependencies"]["engine"]
    joined = " ".join(engine).lower()
    assert "opencv-python-headless" in joined
    assert "opencv-contrib" not in joined
    assert "phash" not in joined


@guarantee
def test_the_package_does_not_reach_for_a_detector_that_is_not_there():
    """AKAZE, BRISK and KAZE were measured absent from cv2 5.x. Nothing in the
    package may name them, because an unpinned install resolves to 5.x and the
    failure would be an AttributeError at a barrier rather than at import."""
    absent = ("AKAZE_create", "BRISK_create", "KAZE_create", "img_hash")
    sources = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in sources
        for name in absent
        if name in path.read_text(encoding="utf-8")
    ]
    assert not offenders, "the package names a detector cv2 5.x does not have: " + ", ".join(
        offenders
    )
