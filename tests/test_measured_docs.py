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
    at,
    blocks,
    cited,
    collect_booleans,
    control_key,
    figures,
    generated,
    load_evidence,
    needs_a_control,
    perturbed,
    render,
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


@guarantee
def test_every_claim_names_the_evidence_it_reads(evidence):
    """X1/Y1. A sentence that reads nothing cannot be checked by anything.

    This is the rule that stops a fixed string reappearing in a template. The
    previous round moved the prose out of the documents and into the renderer
    and called it generated; `_safety_block` still ended with "no scene measured
    produced it for a frame with a vehicle in it" regardless of the number it
    had just computed. A claim with no declared reads is that sentence.
    """
    naked = [
        f"{key}[{index}] ({claim.say.__name__})"
        for key, claims in BLOCKS.items()
        for index, claim in enumerate(claims)
        if not claim.reads
    ]
    assert not naked, (
        f"{naked} state something without reading the evidence. A generated section "
        "may not contain a sentence no value decides."
    )
    unreadable = [
        f"{key}[{index}] reads {path!r}, which the evidence does not have"
        for key, claims in BLOCKS.items()
        for index, claim in enumerate(claims)
        for path in claim.reads
        if not _resolves(evidence, path)
    ]
    assert not unreadable, unreadable


def _resolves(evidence, path) -> bool:
    try:
        at(evidence, path)
    except (KeyError, TypeError):
        return False
    return True


@guarantee
def test_every_claim_changes_when_the_evidence_it_reads_changes(evidence):
    """X1b/Y1. The generic control, and the one that would have caught this.

    For every claim of every block, every path it declares is perturbed in turn
    and the claim must produce different words. This is not "somebody thought to
    plant a contradiction in the three sentences L3 attacked" -- it is every
    sentence against every value it says it depends on.

    L3's own reproductions are three instances of it: `weather.vehicle_refusals`
    3 rendered "3 wrongful refusals ... and no scene measured produced it";
    `vehicle_refused` 5 narrated five refusals away as "an empty lane reads as
    occupied"; `smooth_floor.camera_health` set still rendered "with no camera
    fault raised". All three now fail here.
    """
    deaf = []
    for key, claims in BLOCKS.items():
        for index, claim in enumerate(claims):
            before = claim.say(evidence)
            for path in claim.reads:
                after = claim.say(perturbed(evidence, path))
                if after == before:
                    deaf.append(f"{key}[{index}] ({claim.say.__name__}) ignores {path!r}")
    assert not deaf, (
        "these sentences do not move when the evidence they claim to be derived "
        "from moves:\n  " + "\n  ".join(deaf) + "\n\nA claim that survives its own "
        "measurement changing is a hand-written sentence with a citation stapled to it."
    )


def test_the_claim_control_catches_a_sentence_that_ignores_its_evidence(evidence):
    """The planted control for the control above.

    A perturbation that changed nothing, or a comparison that always reported a
    difference, would make the check pass while proving nothing. So: a claim
    that deliberately ignores its evidence must be caught, and one that reads it
    must not be.
    """
    from measured_figures import Claim

    ignores = Claim(("gate.min_occupancy",), lambda evidence: "a fixed sentence")
    reads = Claim(("gate.min_occupancy",), lambda e: f"the floor is {at(e, 'gate.min_occupancy')}")

    assert ignores.say(evidence) == ignores.say(perturbed(evidence, "gate.min_occupancy")), (
        "the perturbation changed a sentence that reads nothing; it is changing "
        "something other than what it says it changes"
    )
    assert reads.say(evidence) != reads.say(perturbed(evidence, "gate.min_occupancy")), (
        "a claim that does read its evidence was not seen to move; the perturbation "
        "is a no-op and the whole check is vacuous"
    )


@guarantee
def test_a_section_states_only_what_its_claims_say(evidence, rendered):
    """The blocks are the claims and nothing else.

    `render` joins what the claims produce, so a section cannot acquire a
    sentence except by acquiring a claim. Checked rather than asserted, because
    "the renderer only emits claims" is exactly the kind of thing that stays
    true until somebody appends one convenient line.
    """
    for key, claims in BLOCKS.items():
        assert rendered[key] == render(evidence, claims)
        for claim in claims:
            said = claim.say(evidence)
            if said:
                assert said in rendered[key]


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


# --- the seam -------------------------------------------------------------
#
# This lives here rather than in `test_presence_wiring.py`, which needs cv2 and
# skips without it. `vehicle_id.presence` imports cv2 lazily, so the disclosure
# can be checked in the CI job that proves the contract stands alone -- the job
# where "the operator was told" matters most, because it is the one that runs
# everywhere.


@guarantee
def test_every_measured_limitation_is_named_at_the_seam(evidence):
    """X3b/Y4. Derived from the measurement, not from a tuple in this file.

    The previous version of this check iterated a hard-coded 5-tuple of topic
    words while its docstring promised that adding a measured limitation without
    telling the operator would turn it red. It could not: `conflated_reasons`
    was measured, was not named at the seam, and the check passed.

    `eval_presence.py` now publishes `limitations` alongside the measurement --
    each entry naming the word the seam must say -- so a limitation that appears
    in the evidence and not at the seam is a failure here. Deriving from the
    top-level SECTIONS instead would have demanded a seam string for `gate` and
    `exposure`, which are not limitations, and the exclusion set that followed
    would have been the same hard-coded tuple with extra steps.
    """
    from vehicle_id.presence import KNOWN_LIMITS

    limits = evidence["limitations"]
    assert limits, "the evidence names no limitations at all; this check is empty"
    spoken = " ".join(KNOWN_LIMITS).lower()
    silent = [
        f"{limit['topic']} (measured in {limit['measured_in']}, seam must say "
        f"{limit['seam_word']!r})"
        for limit in limits
        if limit["seam_word"].lower() not in spoken
    ]
    assert not silent, (
        "the evidence measures limitations the seam does not name:\n  "
        + "\n  ".join(silent)
        + "\n\nAdd them to `KNOWN_LIMITS` in src/vehicle_id/presence.py. The person "
        "typing `--empty-lane` at 6am reads neither document."
    )


def test_the_seam_check_notices_a_limitation_nobody_disclosed(evidence):
    """The planted control, and the exact shape the old check could not see.

    Add a measured limitation to the evidence and the seam must be found
    wanting. If this passes trivially, the check above is iterating something
    that cannot grow.
    """
    from vehicle_id.presence import KNOWN_LIMITS

    spoken = " ".join(KNOWN_LIMITS).lower()
    invented = {
        "measured_in": "a_section_that_did_not_exist",
        "topic": "something nobody has disclosed",
        "seam_word": "a-word-no-disclosure-contains",
    }
    tampered = [*evidence["limitations"], invented]
    unnamed = [limit for limit in tampered if limit["seam_word"].lower() not in spoken]
    assert unnamed == [invented], (
        "a limitation the seam does not mention was not found; the coverage check "
        "cannot see a limitation being added"
    )


@guarantee
def test_no_disclosure_at_the_seam_could_be_deleted_unnoticed(evidence):
    """The other half, and the one that makes "named at the seam" mean something.

    The check above matches a phrase against the whole concatenated disclosure,
    so a limitation can be satisfied by a string that is about something else.
    It was: "headlight" appeared in the camera-fault caveat, so the headlight
    limitation could have been deleted outright with nothing going red, and the
    check would have been measuring the presence of a word rather than the
    presence of a disclosure.

    So every string at the seam has to be LOAD-BEARING: remove it, and some
    measured limitation must go unnamed. A disclosure nothing depends on is one
    a refactor deletes.
    """
    from vehicle_id.presence import KNOWN_LIMITS

    limits = evidence["limitations"]

    def unnamed(disclosure) -> list[str]:
        spoken = " ".join(disclosure).lower()
        return [limit["seam_word"] for limit in limits if limit["seam_word"].lower() not in spoken]

    assert not unnamed(KNOWN_LIMITS), "as shipped, some measured limitation is unnamed"
    spare = []
    for index, text in enumerate(KNOWN_LIMITS):
        without = tuple(x for i, x in enumerate(KNOWN_LIMITS) if i != index)
        if not unnamed(without):
            spare.append(f"KNOWN_LIMITS[{index}]: {text[:60]}...")
    assert not spare, (
        "these disclosures carry no measured limitation of their own — deleting one "
        "would leave the coverage check green:\n  " + "\n  ".join(spare) + "\n\nEither "
        "the limitation's `seam_word` in eval_presence.py is a phrase generic enough "
        "to match some other disclosure, or this string says nothing the measurement "
        "found."
    )


@guarantee
def test_the_camera_fault_caveat_is_one_definition_at_both_seams():
    """X3a. The conflation is disclosed where the count is emitted.

    `reference_not_recognised` is published under `camera_faults`, and an
    operator reading a count of them is the person who would send a technician.
    The caveat is defined once in `presence.py` so the health endpoint and the
    CLI cannot disclose different things -- the failure the previous round fixed
    for the UNVALIDATED string and left open for this one.
    """
    from vehicle_id.presence import CAMERA_FAULTS_CAVEAT, KNOWN_LIMITS

    assert CAMERA_FAULTS_CAVEAT in KNOWN_LIMITS, (
        "the caveat is not one of the limitations, so the CLI does not print it"
    )
    assert "reference_not_recognised" in CAMERA_FAULTS_CAVEAT
    assert "camera_faults" in CAMERA_FAULTS_CAVEAT
