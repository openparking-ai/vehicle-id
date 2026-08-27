"""A document may not state a figure that disagrees with the measurement.

The documentation equivalent of a fail-control, and it exists because a typed
number survived review by looking measured: `README.md` was edited from 0.7% to
0.3% with nothing re-measuring it, while `tests/test_plates.py` beside it still
said 0.7%. Both looked like findings. Only one had ever been a measurement, and
the repository could not tell you which.

Every published figure now lives in `docs/measured/presence.json`, written by
`scripts/eval_presence.py`, and is cited in prose as a marked span. Editing the
prose is what this test catches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from measured_figures import DOCUMENTS, cited, figures, load_evidence  # noqa: E402

guarantee = pytest.mark.guarantee


@pytest.fixture(scope="module")
def measured():
    return figures(load_evidence(ROOT))


@guarantee
def test_every_figure_a_document_states_matches_the_measurement(measured):
    wrong = []
    for document in DOCUMENTS:
        text = (ROOT / document).read_text(encoding="utf-8")
        for key, stated in cited(text):
            if key not in measured:
                wrong.append(f"{document}: cites {key!r}, which nothing measures")
            elif stated != measured[key]:
                wrong.append(
                    f"{document}: states {key!r} as {stated!r}, "
                    f"but the measurement is {measured[key]!r}"
                )
    assert not wrong, (
        "a document disagrees with docs/measured/presence.json:\n  "
        + "\n  ".join(wrong)
        + "\n\nRe-measure with `python scripts/eval_presence.py --update-docs`. "
        "Do not edit the number by hand."
    )


@guarantee
def test_the_documents_actually_cite_something(measured):
    """The control. A test that checked zero spans would pass forever, which is
    precisely the failure mode it is here to prevent."""
    total = sum(
        len(cited((ROOT / document).read_text(encoding="utf-8"))) for document in DOCUMENTS
    )
    assert total >= 6, f"only {total} measured figures are cited; the check is nearly empty"


def test_the_check_catches_an_edited_number(measured):
    """The planted control. Edit a figure and the comparison must go red."""
    key, value = next(iter(measured.items()))
    tampered = f"<!--m:{key}-->{value} and a bit<!--/m-->"
    (found_key, stated), = cited(tampered)
    assert found_key == key
    assert stated != measured[key]
