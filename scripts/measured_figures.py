"""The figures and the SENTENCES the documents are allowed to state, rendered
from the evidence.

One place turns a measurement into the words that appear in a document. There
are two mechanisms, and the second exists because the first was not enough.

**Spans.** `<!--m:key-->value<!--/m-->` inserts one measured VALUE into prose.
`eval_presence.py --update-docs` fills them in from `docs/measured/presence.json`
and `tests/test_measured_docs.py` fails when a span and the evidence disagree.
The rule this enforces: a published figure is produced by a command, not typed.
It exists because one was -- a README figure measured at 0.7% was edited to 0.3%
with nothing re-measuring it, the repository's own test still said 0.7%, and the
number passed review by looking measured.

**Blocks.** `<!--mb:key-->...<!--/mb-->` replaces a whole SECTION, prose and
all. This exists because a correct span turned out to be no defence at all: the
span `weather.answers_up_to` rendered "5% of the frame in streaks" perfectly,
and beside it sat the hand-written words "and stops answering above that", which
were false -- the gate reads an empty lane as OCCUPIED from 10% to 25% coverage
and the evidence file said so. A true number lent its credibility to a false
sentence, in both published documents, and nothing could see it.

**Claims, and this is the third mechanism, because the second was not enough
either.** Generating the sections moved the prose out of the document and into
this file; it did not stop it being hand-written. `_safety_block` computed
`refusals` from the evidence and then followed it with a fixed string -- "no
scene measured produced it for a frame with a vehicle in it" -- so planting
three refusals rendered "3 wrongful refusals ... and no scene measured produced
it", and nothing went red, because the check compares the DOCUMENT to the
RENDERER and both carried the contradiction. That is K2's failure reproduced
inside the mechanism built to prevent it, on the one claim that makes this gate
shippable off by default.

So a block is no longer a function that returns a string. **A block is a
sequence of `Claim`s, and a `Claim` cannot emit a sentence without a value
deciding what the sentence says.** Each one declares the evidence paths it
reads, and `tests/test_measured_docs.py` perturbs every declared path of every
claim and requires the prose to change. A claim that declares no reads is
rejected; a claim whose sentence survives its own evidence changing is rejected.
Neither of those is a convention -- recognising a natural-language assertion
would be a classifier, but requiring that a sentence be a function of a named
value is a check.

Outside these sections the rule is still a MANUAL convention, named as one
rather than disguised as a control: prose makes no behavioural claim of its own,
and L3 checks it. Blocks render at column 0 and are not indented into list items
or blockquotes -- a generated section that had to be re-wrapped by hand would be
hand-written again. When a bullet or a blockquote needs to make a measured
claim, the claim MOVES INTO a block and the bullet keeps a pointer to it.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# The weather scope is the gate's own disclosure text, read rather than
# re-typed. It was hand-written here, hand-written at the seam and hand-written
# into the evidence file, and a round that corrected one of the three left the
# other two standing.
from vehicle_id.presence import STREAK_CONDITION

EVIDENCE = Path("docs/measured/presence.json")
SPAN = re.compile(r"<!--m:([a-z0-9_.]+)-->(.*?)<!--/m-->", re.DOTALL)
# The body may not contain another marker. Without that guard an EMPTY
# placeholder -- `<!--mb:k-->\n<!--/mb-->`, one newline, no body -- cannot match
# on its own, and the pattern instead spans from one marker to the NEXT pair's
# closing tag, silently eating the marker in between. It ate three of six the
# first time this ran.
BLOCK = re.compile(
    r"<!--mb:([a-z0-9_.]+)-->\n?((?:(?!<!--mb:|<!--/mb-->).)*?)\n?<!--/mb-->",
    re.DOTALL,
)

#: Documents that cite measurements. Anything with a span or a block must be
#: listed here, or nothing would check it. `EVAL_DATA.md` is here because it
#: cites the fixture's own lowest reference texture, and a measured figure in a
#: document nothing checks is the original defect in a quieter place.
DOCUMENTS = (Path("README.md"), Path("docs/CONTRACT.md"), Path("docs/EVAL_DATA.md"))


# --- reading and perturbing the evidence ---------------------------------


def at(evidence: dict, path: str):
    """The value at a dotted path. The only way a claim reads the evidence."""
    node = evidence
    for part in path.split("."):
        node = node[part]
    return node


def _something_else(value):
    """A different value of the same shape.

    Used only by the control: the point is to change what a claim reads without
    changing what it can read, so that a sentence which does not move when its
    own evidence moves can be seen not to move.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + " (perturbed)"
    if value is None:
        return 0
    if isinstance(value, list):
        return value[:-1] if len(value) > 1 else [*value, *value]
    if isinstance(value, dict):
        if len(value) > 1:
            dropped = next(iter(value))
            return {k: v for k, v in value.items() if k != dropped}
        return {**value, "perturbed": next(iter(value.values()), 0)}
    return value


def perturbed(evidence: dict, path: str) -> dict:
    """The evidence with exactly one value replaced by a different one."""
    parts = path.split(".")
    out = copy.deepcopy(evidence)
    node = out
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = _something_else(node[parts[-1]])
    return out


@dataclass(frozen=True)
class Claim:
    """One paragraph a document is allowed to state, and what it is derived from.

    `reads` is not documentation. It is the control's handle: every path listed
    is perturbed in turn and `say` must produce different words. A claim with an
    empty `reads` cannot be checked and is rejected outright -- that is the rule
    that stops a fixed string reappearing in a template.

    `say` returns "" when the evidence does not support making the claim at all,
    which is how a section drops a paragraph rather than stating it falsely.
    """

    reads: tuple[str, ...]
    say: Callable[[dict], str]


def _rate(block: dict, reads: int, seeds: int) -> str:
    if block["min_pct"] == block["max_pct"] == 0.0:
        return f"0.0% ({seeds} seeds x {reads} reads, {seeds * reads} in total, none)"
    return (
        f"{block['mean_pct']:.1f}% mean, {block['min_pct']:.1f}-{block['max_pct']:.1f}% "
        f"across {seeds} seeds x {reads} reads"
    )


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


#: How a presence verdict is written, in one place. `str(None).lower()` reads
#: `none`, which is a Python spelling of a value every other surface -- the
#: contract, the tables, the engine's own JSON -- calls `null`. Two spellings of
#: one value is the same defect as two copies of one claim.
VERDICT = {True: "`true`", False: "`false`", None: "`null`"}


def _verdict(present) -> str:
    return VERDICT.get(present, f"`{present}`")


def _cell(verdict: dict) -> str:
    """One scene's verdict as a table cell: the value, and the number behind it."""
    word = _verdict(verdict["present"])
    if verdict["occupancy"] is None:
        return f"{word} —"
    return f"{word} {verdict['occupancy']:.3f}"


def _synthetic(evidence: dict) -> str:
    """How the caveat reads, from the count of real frames ever measured."""
    real = at(evidence, "scenes.real_frames_measured")
    if real:
        return f"MEASURED on {real} real frames"
    return "NOT MEASURED on any real frame (0 have ever been through this gate)"


def figures(evidence: dict) -> dict[str, str]:
    """Every single VALUE a document may state, keyed by the name it cites."""
    out: dict[str, str] = {}

    gate = evidence["gate"]
    out["gate.min_occupancy"] = f"{gate['min_occupancy']:.0%}"
    out["gate.max_occupancy"] = f"{gate['max_occupancy']:.0%}"
    out["gate.min_structural_change"] = str(gate["min_structural_change"])
    out["gate.window"] = str(gate["window"])
    out["gate.min_frame_std"] = str(gate["min_frame_std"])
    out["gate.min_reference_texture"] = str(gate["min_reference_texture"])

    sep = evidence["separation"]
    out["separation.cells"] = str(sum(row["cells"] for row in sep.values()))
    out["separation.refusals"] = str(sum(row["vehicle_refused"] for row in sep.values()))

    floor = evidence["texture_floor"]
    axis = floor["matrix_ground_reference_texture"]
    out["texture_floor.matrix_lowest"] = str(min(axis.values()))
    out["texture_floor.smooth_floor"] = str(floor["smooth_floor_reference_texture"])

    noise_block = evidence.get("noise") or {}
    out["noise.weights"] = noise_block.get("weights_id", "AN UNRECORDED ARTEFACT")

    exposure = evidence["exposure"]
    out["exposure.range"] = (
        f"light level {exposure['lowest_level_still_false']} to "
        f"{exposure['highest_level_still_false']} against a reference captured at "
        f"{exposure['reference_level']}"
    )
    out["exposure.holds"] = (
        "every level tested"
        if exposure["all_false_across_range"]
        else "NOT across the whole range"
    )

    transition = evidence["confidence_transition"]
    out["transition.largest_step"] = f"{transition['largest_confidence_step']:.2f}"

    noise = evidence.get("noise")
    if noise:
        reads, seeds = noise["reads_per_seed"], noise["seeds"]
        out["noise.one_capture"] = _rate(noise["no_gate_1_capture"], reads, seeds)
        out["noise.three_capture"] = _rate(noise["no_gate_3_capture"], reads, seeds)
        out["noise.gated_one_capture"] = _rate(noise["gated_1_capture"], reads, seeds)
        out["noise.gated_three_capture"] = _rate(noise["gated_3_capture"], reads, seeds)
    return out


# --- the generated sections, claim by claim ------------------------------
#
# Every function below takes the whole evidence and returns one paragraph. None
# of them may state anything a value in `Claim.reads` does not decide.


def _rows(evidence: dict):
    sep = evidence["separation"]
    return sorted(sep.items(), key=lambda kv: (kv[1]["texture"], kv[1]["headlight"]))


def _separation_intro(evidence: dict) -> str:
    rows = _rows(evidence)
    total = sum(row["cells"] for _, row in rows)
    return (
        f"**The matrix, unedited.** {total} cells sweeping vehicle/ground contrast "
        "through the exactly-invisible case, ground texture, the vehicle's own "
        "surface grain, and a headlight pool on the floor. Each cell carries both "
        "scenes — the vehicle and the empty lane beside it — because a measure that "
        "answered one way for everything would pass a one-sided sweep perfectly."
    )


def _separation_table(evidence: dict) -> str:
    lines = [
        "| configuration | cells | vehicle seen | vehicle refused | vehicle not measured "
        "| empty called empty | empty read occupied | margin | separates |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, row in _rows(evidence):
        margin = "—" if row["margin"] is None else f"{row['margin']:.3f}"
        lines.append(
            f"| {label} | {row['cells']} | {row['vehicle_seen']} | {row['vehicle_refused']} "
            f"| {row['vehicle_not_measured']} | {row['empty_called_empty']} "
            f"| {row['empty_false_positive']} | {margin} "
            f"| {'yes' if row['separates'] else '**no**'} |"
        )
    return "\n".join(lines)


def _why_a_row_fails(row: dict) -> str:
    """The reason THIS row does not separate, from the row.

    `separates` is false on any of three conditions and the old sentence
    asserted one of them for every failing row, so a row failing because
    VEHICLES WERE REFUSED read as "an empty lane reads as occupied" — the
    safety failure narrated away as a nuisance one.
    """
    reasons = []
    if row["vehicle_refused"]:
        reasons.append(f"{row['vehicle_refused']} of {row['cells']} vehicles were REFUSED")
    if row["empty_false_positive"]:
        reasons.append(
            f"{row['empty_false_positive']} of {row['cells']} empty lanes read as occupied"
        )
    if row["margin"] is None:
        reasons.append("no cell in it produced two comparable occupancies")
    elif row["margin"] <= 0:
        reasons.append(f"the occupancy margin is {row['margin']:.3f}, not positive")
    return " and ".join(reasons)


def _separation_summary(evidence: dict) -> str:
    rows = _rows(evidence)
    working = [row["margin"] for _, row in rows if row["separates"]]
    failing = [(label, row) for label, row in rows if not row["separates"]]
    lines = []
    if working:
        lines.append(
            f"It separates vehicle from empty in {len(working)} of {len(rows)} "
            f"configurations, with a worst-case occupancy margin of {min(working):.2f}."
        )
    else:
        lines.append(
            f"It separates vehicle from empty in NONE of the {len(rows)} configurations."
        )
    for label, row in failing:
        lines.append(f"It does **not** separate in {label}: {_why_a_row_fails(row)}.")
    return "\n".join(lines)


def _separation_admissions(evidence: dict) -> str:
    """What the matrix says about admitting a vehicle — the claim moved here.

    `docs/CONTRACT.md` used to say, in a bullet nothing checked, that the gate
    admits a lane with a vehicle in it at EVERY contrast tested. The evidence
    file said `vehicle_seen: 16` against `cells: 18` in one row. The claim now
    lives where the numbers that decide it live, and the bullet points here.
    """
    rows = _rows(evidence)
    cells = sum(row["cells"] for _, row in rows)
    seen = sum(row["vehicle_seen"] for _, row in rows)
    refused = sum(row["vehicle_refused"] for _, row in rows)
    unmeasured = sum(row["vehicle_not_measured"] for _, row in rows)
    if seen == cells:
        return (
            f"**A vehicle is admitted in every one of the {cells} cells**, including "
            "at the ground's exact luminance, where the vehicle and the ground it "
            "stands on are the same brightness."
        )
    health: dict[str, int] = {}
    for _, row in rows:
        for reason, count in row["vehicle_not_measured_health"].items():
            health[reason] = health.get(reason, 0) + count
    named = ", ".join(f"`{reason}` in {count}" for reason, count in sorted(health.items()))
    refusal = (
        f"It is REFUSED in {refused} — `false` for a frame with a car in it."
        if refused
        else "None of them was refused."
    )
    return (
        f"**A vehicle is admitted in {seen} of the {cells} cells**, including at the "
        "ground's exact luminance, where the vehicle and the ground it stands on are "
        f"the same brightness. {refusal} In the remaining {unmeasured} presence is "
        f"`null` and the gate reports a camera fault: {named}. A cell that is not "
        "measured is not a refusal — the lane falls back to a ticket and a human — "
        "but it is a frame with a car in it that this gate answered nothing about, "
        "and the reason it gave names equipment."
    )


def _texture_headline(evidence: dict) -> str:
    floor = at(evidence, "texture_floor.min_reference_texture")
    return (
        "**Ground with no texture of its own is NOT MEASURED, never `false`.** The "
        "comparison asks whether a window still looks like the same piece of ground, "
        f"so ground carrying nothing to recognise leaves it nothing to work with. "
        f"Below {floor} grey levels of typical local texture the gate declines to "
        "answer."
    )


def _texture_axis(evidence: dict) -> str:
    axis = at(evidence, "texture_floor.matrix_ground_reference_texture")
    reaches = at(evidence, "texture_floor.matrix_axis_can_reach_the_floor")
    listing = ", ".join(f"{k} → {v}" for k, v in sorted(axis.items()))
    if reaches:
        return (
            f"The matrix's own ground reaches under that floor at its lowest setting "
            f"({listing})."
        )
    return (
        f"The matrix's own ground never gets near that floor — its texture axis "
        f"bottoms out at {min(axis.values())} grey levels ({listing}), because the "
        "sensor's own grain is most of it."
    )


def _texture_smooth_floor(evidence: dict) -> str:
    texture = at(evidence, "texture_floor.smooth_floor_reference_texture")
    present = at(evidence, "texture_floor.smooth_floor.present")
    health = at(evidence, "texture_floor.smooth_floor.camera_health")
    reason = at(evidence, "texture_floor.smooth_floor_reason")
    fault = (
        "with no camera fault raised"
        if health is None
        else f"and raises the camera fault `{health}`"
    )
    return (
        "Sealed or painted concrete under a clean sensor is a different scene: it "
        f"measures {texture} grey levels, the gate returns "
        f"{_verdict(present)} {fault}, and it says why — "
        f'"{reason}".'
    )


def _texture_consequence(evidence: dict) -> str:
    texture = at(evidence, "texture_floor.smooth_floor_reference_texture")
    floor = at(evidence, "texture_floor.min_reference_texture")
    real = _synthetic(evidence)
    if texture >= floor:
        return ""
    return (
        f"**This matters more than the number suggests.** {texture} grey levels "
        f"against a {floor} floor is a surface this gate declines to answer on at "
        "all. **Whether that describes a given entry is a property of that entry** — "
        "the operator can photograph its floor and score it by the mapping in "
        "`docs/EVAL_DATA.md`; how many entries look like it is not something this "
        f"project has measured. {real}, so how much texture a real covered entry "
        "carries is an open question, and the remedy if it carries too little is "
        "physical — paint markings, add a textured strip in view."
    )


def _weather_table(evidence: dict) -> str:
    sweep = at(evidence, "weather.sweep")
    lines = [
        "**Weather, measured on three scenes at every coverage** — an empty lane, a "
        "vehicle, and the metal plate the gate exists to refuse. The number beside "
        f"each verdict is the measured occupancy, over {len(sweep)} coverages.",
        "",
        "| streak coverage | empty lane | vehicle | metal plate |",
        "|---|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['coverage']:.0%} | {_cell(row['empty_lane'])} "
            f"| {_cell(row['vehicle'])} | {_cell(row['metal_plate'])} |"
        )
    return "\n".join(lines)


def _weather_bands(evidence: dict) -> str:
    answered = at(evidence, "weather.highest_coverage_still_answered_false")
    occupied = at(evidence, "weather.lowest_coverage_reading_occupied")
    declines = at(evidence, "weather.lowest_coverage_declining_to_answer")
    worst = at(evidence, "weather.highest_confidence_reading_occupied")
    if occupied is None:
        return (
            f"Two bands: `false` up to {_pct(answered)} of the frame in streaks, and "
            f"`null` from {_pct(declines)}."
        )
    return (
        f"**Three bands, not two.** `false` up to {_pct(answered)} of the frame in "
        f"streaks; from {_pct(occupied)} an **empty lane reads as OCCUPIED**, at up "
        f"to {worst:.2f} confidence; from {_pct(declines)} the gate declines to "
        "answer at all."
    )


def _weather_consequence(evidence: dict) -> str:
    occupied = at(evidence, "weather.lowest_coverage_reading_occupied")
    if occupied is None:
        return ""
    return (
        f"The band from {_pct(occupied)} is the one to read. `presence: true` with "
        '`outcome: "fallback"` tells a lane controller that a car is there and could '
        "not be identified, and this contract says refusing it is a bug in your "
        "integration — so in that band a conforming lane issues a ticket and raises "
        "an attendant for a car that is not there."
    )


def _weather_fraud(evidence: dict) -> str:
    admitted = at(evidence, "weather.metal_plate_admitted_from")
    answered = at(evidence, "weather.highest_coverage_still_answered_false")
    if admitted is None:
        return (
            "The metal plate on the loop — the case this gate exists for — is refused "
            "at every coverage measured."
        )
    return (
        f"**And the fraud is admitted with it.** The metal plate on the loop — the "
        f"case this gate exists for — is correctly refused up to {_pct(answered)} "
        f"coverage and then **transacts from {_pct(admitted)}**, on the same streaks. "
        "In that band the gate does not merely lose the ability to say `false`; it "
        "issues the ticket for the exact scene it was built to refuse."
    )


def _weather_scope(evidence: dict) -> str:
    refusals = at(evidence, "weather.vehicle_refusals")
    cells = at(evidence, "weather.vehicle_cells")
    real = _synthetic(evidence)
    return (
        "This is a measured REGRESSION against the intensity measure that preceded "
        "it, which called heavy rain an empty lane correctly. It is recorded rather "
        "than argued away. **Whether it reaches a given entry depends on whether "
        f"{STREAK_CONDITION}.** The operator can see that and this project cannot "
        f"count it. {real}, and no frequency is claimed either way. "
        f"Across the sweep, {refusals} of {cells} vehicle scenes were refused."
    )


def _headlight_table(evidence: dict) -> str:
    sweep = at(evidence, "headlight.sweep")
    units = at(evidence, "headlight.pool_units")
    lines = [
        "**Headlights on the floor.** A covered entry is artificially lit and often "
        "dark, so an approaching car throws its beams into frame before the car "
        "itself arrives — a large change in the scene caused by a vehicle that is "
        f"not yet the vehicle. Measured over {len(sweep)} pools, with and without "
        f"the car that cast them. {units}",
        "",
        "| beam pool, peak x ambient | empty lane (car not yet in frame) | vehicle |",
        "|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| x{1 + row['pool']:g} | {_cell(row['empty_lane'])} | {_cell(row['vehicle'])} |"
        )
    return "\n".join(lines)


def _headlight_boundary(evidence: dict) -> str:
    held = at(evidence, "headlight.highest_pool_still_empty")
    tripped = at(evidence, "headlight.lowest_pool_reading_occupied")
    if tripped is None:
        return (
            f"An empty lane holds at `false` through every pool tested, up to "
            f"x{1 + (held or 0):g} ambient."
        )
    return (
        f"An empty lane holds at `false` up to a pool of x{1 + held:g} ambient and "
        f"reads as OCCUPIED from x{1 + tripped:g} — the beams of a car that has not "
        "arrived open a transaction for it."
    )


def _headlight_scope(evidence: dict) -> str:
    refusals = at(evidence, "headlight.vehicle_refusals")
    cells = at(evidence, "headlight.vehicle_cells")
    model = at(evidence, "headlight.model")
    real = _synthetic(evidence)
    return (
        f"{refusals} of {cells} vehicle scenes were refused. **The model is a "
        f"limitation of these numbers**: {model}. A gloss or wet floor at night is a "
        f"specular scene and this is a matte one. {real}."
    )


def _safety_counts(evidence: dict) -> str:
    rows = _rows(evidence)
    matrix_cells = sum(row["cells"] for _, row in rows)
    matrix_refusals = sum(row["vehicle_refused"] for _, row in rows)
    weather_cells = at(evidence, "weather.vehicle_cells")
    weather_refusals = at(evidence, "weather.vehicle_refusals")
    head_cells = at(evidence, "headlight.vehicle_cells")
    head_refusals = at(evidence, "headlight.vehicle_refusals")
    total = matrix_cells + weather_cells + head_cells
    refusals = matrix_refusals + weather_refusals + head_refusals
    counted = (
        f"{matrix_cells} matrix cells, {weather_cells} weather coverages and "
        f"{head_cells} headlight pools, each measured with a vehicle in the frame"
    )
    if refusals:
        return (
            f"**The safety property does NOT hold.** {refusals} wrongful refusals in "
            f"{total} scenes containing a vehicle: {counted}. `false` is the only "
            f"value that ends a transaction and {refusals} frames with a car in them "
            "produced it. This gate is not shippable in this state."
        )
    return (
        f"**The one thing that holds everywhere measured.** {refusals} wrongful "
        f"refusals in {total} scenes containing a vehicle: {counted}. `false` is the "
        "only value that ends a transaction, and no scene measured produced it for a "
        "frame with a vehicle in it. Where this gate fails it fails to `null` — a "
        "ticket and a human."
    )


def _safety_scope(evidence: dict) -> str:
    rows = _rows(evidence)
    total = (
        sum(row["cells"] for _, row in rows)
        + at(evidence, "weather.vehicle_cells")
        + at(evidence, "headlight.vehicle_cells")
    )
    real = _synthetic(evidence)
    return (
        f"Every one of those {total} scenes is a drawn rectangle on a drawn lane — "
        f"{real}. The claim is that the measure holds across everything that has been "
        "put through it, not that everything has been put through it."
    )


def _conflation_list(evidence: dict) -> str:
    reasons = at(evidence, "conflated_reasons.reason_reported")
    causes = at(evidence, "conflated_reasons.causes")
    named = ", ".join(f"`{r}`" for r in reasons) or "no reason at all"
    lines = [
        f"**One reason covers {len(causes)} unrelated conditions, and this release "
        f"cannot tell them apart.** {named} is reported for all of the following:",
        "",
    ]
    lines += [f"- {cause}" for cause in sorted(causes)]
    return "\n".join(lines)


def _conflation_caveat(evidence: dict) -> str:
    shared = at(evidence, "conflated_reasons.all_report_the_same_reason")
    causes = at(evidence, "conflated_reasons.causes")
    if not shared:
        return (
            f"The {len(causes)} conditions report different reasons, so an operator "
            "can act on the one they are given."
        )
    faults = at(evidence, "conflated_reasons.causes_that_are_equipment_faults")
    # The head of each name, before its colon: the caveat names which conditions
    # the label is right about, and the bullets above carry the elaboration.
    heads = sorted(name.split(":")[0] for name in faults)
    return (
        "It is published under `camera_faults` in `GET /v1/health`. That is right "
        f"for {len(faults)} of the {len(causes)} — {'; '.join(heads)} — and "
        f"wrong for the other {len(causes) - len(faults)}, where nothing is broken. "
        "**Do not read this reason as a confirmed equipment fault** — read it as "
        '"the capture no longer matches the reference, for one of several reasons '
        'this build cannot separate". Separating them needs a measurement this '
        "release does not make, and inventing one would be guessing; naming the "
        "conflation is the honest thing available now."
    )


def _conflation_ordinary_arrival(evidence: dict) -> str:
    """The one that is a product finding rather than a documentation one.

    A vehicle of ordinary size, on low-texture ground under a beam pool, lands
    on the same reason as a knocked camera — so an arriving car is counted under
    `camera_faults` and pages a technician. It is stated here from the matrix
    that measured it, and the frequency at a real entry is stated as unmeasured,
    because it is.

    EVERY affected cell is described, and that is the fix, not a flourish. This
    used to take `affected[0]` and render its coordinates as the whole finding:
    "2 of the 108 cells ... at contrast 2.05 and surface grain 0", where the two
    cells are at surface grain 0 AND 0.02. One cell's coordinates presented as
    describing two.
    """
    rows = _rows(evidence)
    cells = sum(row["cells"] for _, row in rows)
    affected = [
        (label, cell)
        for label, row in rows
        for cell in row["vehicle_not_measured_cells"]
    ]
    if not affected:
        return ""
    real = _synthetic(evidence)
    fractions = sorted({cell["vehicle_frame_fraction"] for _, cell in affected})
    size = (
        f"{fractions[0]:.0%}"
        if len(fractions) == 1
        else f"{fractions[0]:.0%} to {fractions[-1]:.0%}"
    )
    named = ", ".join(
        f"`{health}`" for health in sorted({cell["camera_health"] for _, cell in affected})
    )
    grouped: dict[str, list[dict]] = {}
    for label, cell in affected:
        grouped.setdefault(label, []).append(cell)
    where = "; ".join(
        f"{label} — "
        + ", ".join(
            f"contrast {cell['contrast']:g} / surface grain {cell['surface']:g}"
            for cell in group
        )
        for label, group in grouped.items()
    )
    return (
        f"**One of those conditions is a car arriving.** {len(affected)} of the "
        f"{cells} separation-matrix cells put an ordinary vehicle — {size} of the "
        f"frame, not one filling it — in front of the camera and got {named} back. "
        f"Each of those cells, in full: {where}. The gate counts that under "
        "`camera_faults`, so an arriving car pages a technician about a working "
        f"camera. {real}: how often a real entry lands in one of these "
        "configurations is not known, and these are drawn rectangles. What is known "
        "is that the reason cannot be read as equipment on its own."
    )


#: Every generated section, as the claims it is allowed to make. Order is the
#: order they render in.
BLOCKS: dict[str, tuple[Claim, ...]] = {
    "presence.separation": (
        Claim(("separation",), _separation_intro),
        Claim(("separation",), _separation_table),
        Claim(("separation",), _separation_summary),
        Claim(("separation",), _separation_admissions),
    ),
    "presence.texture": (
        Claim(("texture_floor.min_reference_texture",), _texture_headline),
        Claim(
            (
                "texture_floor.matrix_ground_reference_texture",
                "texture_floor.matrix_axis_can_reach_the_floor",
            ),
            _texture_axis,
        ),
        Claim(
            (
                "texture_floor.smooth_floor_reference_texture",
                "texture_floor.smooth_floor.present",
                "texture_floor.smooth_floor.camera_health",
                "texture_floor.smooth_floor_reason",
            ),
            _texture_smooth_floor,
        ),
        Claim(
            (
                "texture_floor.smooth_floor_reference_texture",
                "texture_floor.min_reference_texture",
                "scenes.real_frames_measured",
            ),
            _texture_consequence,
        ),
    ),
    "presence.weather": (
        Claim(("weather.sweep",), _weather_table),
        Claim(
            (
                "weather.highest_coverage_still_answered_false",
                "weather.lowest_coverage_reading_occupied",
                "weather.lowest_coverage_declining_to_answer",
                "weather.highest_confidence_reading_occupied",
            ),
            _weather_bands,
        ),
        Claim(("weather.lowest_coverage_reading_occupied",), _weather_consequence),
        Claim(
            (
                "weather.metal_plate_admitted_from",
                "weather.highest_coverage_still_answered_false",
            ),
            _weather_fraud,
        ),
        Claim(
            (
                "weather.vehicle_refusals",
                "weather.vehicle_cells",
                "scenes.real_frames_measured",
            ),
            _weather_scope,
        ),
    ),
    "presence.headlight": (
        Claim(("headlight.sweep", "headlight.pool_units"), _headlight_table),
        Claim(
            (
                "headlight.highest_pool_still_empty",
                "headlight.lowest_pool_reading_occupied",
            ),
            _headlight_boundary,
        ),
        Claim(
            (
                "headlight.vehicle_refusals",
                "headlight.vehicle_cells",
                "headlight.model",
                "scenes.real_frames_measured",
            ),
            _headlight_scope,
        ),
    ),
    "presence.safety": (
        Claim(
            (
                "separation",
                "weather.vehicle_cells",
                "weather.vehicle_refusals",
                "headlight.vehicle_cells",
                "headlight.vehicle_refusals",
            ),
            _safety_counts,
        ),
        Claim(
            (
                "separation",
                "weather.vehicle_cells",
                "headlight.vehicle_cells",
                "scenes.real_frames_measured",
            ),
            _safety_scope,
        ),
    ),
    "presence.conflation": (
        Claim(
            ("conflated_reasons.reason_reported", "conflated_reasons.causes"),
            _conflation_list,
        ),
        Claim(
            (
                "conflated_reasons.all_report_the_same_reason",
                "conflated_reasons.causes",
                "conflated_reasons.causes_that_are_equipment_faults",
            ),
            _conflation_caveat,
        ),
        Claim(("separation", "scenes.real_frames_measured"), _conflation_ordinary_arrival),
    ),
}


def render(evidence: dict, claims: tuple[Claim, ...]) -> str:
    """One section: every claim its evidence supports, and no other words."""
    return "\n\n".join(said for claim in claims if (said := claim.say(evidence)))


def blocks(evidence: dict) -> dict[str, str]:
    """Every generated section, rendered from the evidence."""
    return {key: render(evidence, claims) for key, claims in BLOCKS.items()}


# --- the evidence file's own booleans ------------------------------------
#
# These live here rather than in `eval_presence.py` for one practical reason and
# one real one. The practical one: `eval_presence` imports cv2, and the CI job
# that proves the contract stands alone has no cv2 -- a check on the evidence
# file must not drag the engine in behind it. The real one: they are about what
# a published FILE is allowed to claim, which is this module's subject, not
# about measuring anything.


def collect_booleans(node, path: tuple[str, ...] = ()) -> dict:
    """Every boolean the evidence file publishes, keyed by its dotted path.

    Used by the controls check. A boolean added to the evidence without a
    control turns the suite red rather than shipping unproven -- which is the
    whole lesson of `never_refuses_in_weather`, whose expression could not
    evaluate to false and which nobody looked at for two rounds.

    Keys drop the row identity of repeated structures -- every `separates` in
    the matrix is `separation.separates` -- because the control is a statement
    about the QUANTITY, not about one row of it.
    """
    found: dict[str, list] = {}
    if isinstance(node, bool):
        return {".".join(path): [node]}
    children = []
    if isinstance(node, dict):
        children = [
            collect_booleans(value, path + (key,))
            for key, value in node.items()
            if key != "controls"
        ]
    elif isinstance(node, list):
        children = [collect_booleans(item, path) for item in node]
    for child in children:
        for key, values in child.items():
            found.setdefault(key, []).extend(values)
    return found


def needs_a_control(path: str, values: list) -> bool:
    """Whether a published boolean has to name a control.

    The rule, stated so it cannot be argued with case by case: **a boolean
    published as `true` needs a control that makes it false, unless the same
    quantity is also published as `false` somewhere in the same run.**

    Both exemptions are the same exemption. A flag published as `false` is not
    the failure this guards against -- the failure is a `true` nobody can
    falsify. And a quantity that takes both values in the published data carries
    its own control, which is worth more than a planted one: `separates` really
    is false on low-texture ground, and the `present` verdicts in the sweeps
    really do take all three values in the tables a reader can see.
    """
    return all(v is True for v in values)


def control_key(path: str) -> str:
    """The control that answers for a boolean at `path`.

    `separation.<row label>.separates` is answered by `separation.separates`:
    one control per quantity, not one per row.
    """
    parts = path.split(".")
    return f"{parts[0]}.{parts[-1]}"


def load_evidence(root: Path | None = None) -> dict:
    path = (root or Path.cwd()) / EVIDENCE
    return json.loads(path.read_text(encoding="utf-8"))


def cited(text: str) -> list[tuple[str, str]]:
    """The (key, stated value) pairs a document cites in a span."""
    return [(m.group(1), m.group(2)) for m in SPAN.finditer(text)]


def generated(text: str) -> list[tuple[str, str]]:
    """The (key, stated section) pairs a document carries as a block."""
    return [(m.group(1), m.group(2)) for m in BLOCK.finditer(text)]


def rewrite(text: str, values: dict[str, str], rendered: dict[str, str] | None = None) -> str:
    def swap_span(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"the documents cite {key!r}, which nothing measures")
        return f"<!--m:{key}-->{values[key]}<!--/m-->"

    def swap_block(match: re.Match) -> str:
        key = match.group(1)
        if key not in (rendered or {}):
            raise KeyError(f"the documents carry a {key!r} section, which nothing renders")
        return f"<!--mb:{key}-->\n{rendered[key]}\n<!--/mb-->"

    text = BLOCK.sub(swap_block, text)
    return SPAN.sub(swap_span, text)
