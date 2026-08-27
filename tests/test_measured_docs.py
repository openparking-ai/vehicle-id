"""A document may not state a figure -- or a SENTENCE -- that disagrees with the
measurement, and an evidence file may not publish a boolean nobody can make false.

The documentation equivalent of a fail-control, and both halves exist because
something got past the half before it.

The first half caught a typed number: `README.md` was edited from 0.7% to 0.3%
with nothing re-measuring it, while `tests/test_plates.py` beside it still said
0.7%. Both looked like findings; only one had ever been a measurement.

The second half exists because a CORRECT number turned out to be no defence.
The span stating that the gate answers `false` up to 5% streak coverage was
right, and the hand-written words next to it -- "and stops answering above
that" -- were false: the gate reads an empty lane as OCCUPIED from 10% to 25%,
and the evidence file said so. The true number lent its credibility to the false
sentence in both published documents and nothing could see it. So the sentences
describing measured behaviour are GENERATED too, as blocks, and this file checks
them the same way it checks the numbers.

The third half is about the evidence file itself. It published
`never_refuses_in_weather: true` from `all(x is not False or True ...)`, which
is `True` for every input that has ever existed -- and that flag was the
load-bearing safety property of the whole module. So: every boolean the evidence
publishes must name a control, and every control must actually be false.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


from measured_figures import (  # noqa: E402
    BLOCKS,
    DOCUMENTS,
    blocks,
    cited,
    collect_booleans,
    control_key,
    figures,
    generated,
    load_evidence,
    needs_a_control,
)

guarantee = pytest.mark.guarantee


@pytest.fixture(scope="module")
def evidence():
    return load_evidence(ROOT)


@pytest.fixture(scope="module")
def measured(evidence):
    return figures(evidence)


@pytest.fixture(scope="module")
def rendered(evidence):
    return blocks(evidence)


# --- the numbers ---------------------------------------------------------

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


# --- the sentences -------------------------------------------------------

@guarantee
def test_every_generated_section_matches_what_the_evidence_renders(rendered):
    """The half that would have caught "and stops answering above that".

    A section describing measured behaviour is rendered from the evidence file,
    not written. If a word of one has been edited by hand -- or the measurement
    has moved and nobody re-rendered -- this goes red.
    """
    wrong = []
    for document in DOCUMENTS:
        text = (ROOT / document).read_text(encoding="utf-8")
        for key, stated in generated(text):
            if key not in rendered:
                wrong.append(f"{document}: carries a {key!r} section, which nothing renders")
            elif stated.strip() != rendered[key].strip():
                wrong.append(
                    f"{document}: the {key!r} section does not match what the evidence "
                    "renders; it has been edited by hand or the measurement has moved"
                )
    assert not wrong, (
        "a generated section disagrees with docs/measured/presence.json:\n  "
        + "\n  ".join(wrong)
        + "\n\nRe-render with `python scripts/eval_presence.py --update-docs`. "
        "Do not edit the prose by hand -- that is the whole point of the block."
    )


@guarantee
def test_the_documents_actually_carry_the_generated_sections(rendered):
    """The coverage control for the half above.

    A renderer nothing includes protects nothing, and a document that quietly
    lost its blocks would pass the comparison with an empty list. Every renderer
    that exists must appear in at least one document.
    """
    seen = set()
    for document in DOCUMENTS:
        text = (ROOT / document).read_text(encoding="utf-8")
        seen.update(key for key, _ in generated(text))
    missing = sorted(set(BLOCKS) - seen)
    assert not missing, (
        f"{missing} render measured behaviour that no document carries; either "
        "include the section or delete the renderer"
    )
    assert seen, "no document carries a generated section at all"


def test_the_check_catches_an_edited_sentence(rendered):
    """The planted control for the block comparison, and the one that matters.

    The exact shape of the failure this mechanism exists for: a section with one
    word changed. It must not compare equal.
    """
    for key, body in rendered.items():
        tampered = f"<!--mb:{key}-->\n{body} It also stops answering above that.\n<!--/mb-->"
        (found_key, stated), = generated(tampered)
        assert found_key == key
        assert stated.strip() != rendered[key].strip(), (
            f"the {key} section with a sentence appended compared equal to the "
            "rendered one; the block check cannot see an edit"
        )

    # And an EMPTY block must not silently compare equal to a rendered one --
    # that is the shape the marker regex got wrong the first time, when an empty
    # placeholder let the pattern span two markers and eat the one between them.
    first = next(iter(rendered))
    (found_key, stated), = generated(f"<!--mb:{first}-->\n<!--/mb-->")
    assert found_key == first
    assert stated.strip() != rendered[first].strip()


# --- the evidence file's own booleans ------------------------------------

@guarantee
def test_every_published_boolean_names_a_control(evidence):
    """K5b. A flag with no control is not evidence.

    `never_refuses_in_weather` was published as `true` for two rounds from an
    expression that could not return anything else, and it was the safety
    property the module is sold on. Every boolean now has to name the input that
    makes it false.
    """
    published = collect_booleans(evidence)
    controls = evidence.get("controls", {})
    unproven = sorted(
        path
        for path, values in published.items()
        if needs_a_control(path, values) and control_key(path) not in controls
    )
    assert published, "the evidence file publishes no booleans at all; this check is empty"
    assert not unproven, (
        f"{len(unproven)} published boolean(s) name no control: {unproven}\n"
        "Add one to `controls()` in scripts/eval_presence.py: the input that "
        "makes it false, measured in the same run."
    )


@guarantee
def test_every_control_is_actually_false(evidence):
    """The other half, and the one `never_refuses_in_weather` would have failed.

    Naming a control is not enough; the control has to work. A control that
    comes back true is a control that does not break the thing it claims to
    break, which is the same failure one level up.
    """
    controls = evidence.get("controls", {})
    assert controls, "the evidence file publishes no controls at all"
    still_true = sorted(key for key, c in controls.items() if c["value"] is not False)
    assert not still_true, (
        f"{still_true} did not go false under their own control. A boolean whose "
        "control cannot make it false is not evidence of anything."
    )
    for key, control in controls.items():
        assert control.get("how"), f"the control for {key} does not say what it did"


def test_the_boolean_sweep_finds_a_boolean_nobody_controlled(evidence):
    """The planted control for the two above.

    A collector that missed booleans, or a key mapping that silently matched
    everything, would make both tests pass while proving nothing. Add a boolean
    with no control and the sweep must find it.
    """
    tampered = dict(evidence)
    tampered["a_new_section"] = {"a_flag_nobody_controlled": True}
    published = collect_booleans(tampered)
    key = "a_new_section.a_flag_nobody_controlled"
    assert key in published
    assert needs_a_control(key, published[key]), (
        "a flag published only as true was not held to need a control"
    )
    assert control_key(key) not in evidence["controls"]

    # And the other half of the rule: a quantity that is published BOTH ways
    # carries its own control and must not be demanded a planted one.
    assert not needs_a_control("somewhere.a_flag_seen_both_ways", [True, False]), (
        "a boolean the published data already shows false was still demanded a control"
    )
    assert not needs_a_control("somewhere.a_flag_published_false", [False])


def test_the_key_mapping_answers_a_repeated_boolean_from_one_control(evidence):
    """Every row of the matrix publishes its own `separates`; one control answers
    for the quantity rather than one per row. That collapse must be the only
    thing it does -- a mapping that collapsed everything to one key would make
    the coverage check vacuous."""
    assert control_key("separation.ground texture 1, headlights off.separates") == (
        "separation.separates"
    )
    assert control_key("weather.never_refuses_in_weather") == "weather.never_refuses_in_weather"
    assert control_key("exposure.all_false_across_range") != (
        "weather.never_refuses_in_weather"
    )
