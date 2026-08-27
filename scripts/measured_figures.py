"""The figures and the SECTIONS the documents are allowed to state, rendered
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

A check that a document makes no behavioural claim outside a span would have to
recognise a natural-language assertion, which is a classifier and not a check.
So the sentences are generated instead: any section describing MEASURED
BEHAVIOUR is rendered here from the evidence, and the document holds only the
markers. There is then no hand-written prose in those sections to go stale, and
the false sentence could not have been written.

Outside those sections the rule is a MANUAL convention, named as one rather than
disguised as a control: prose makes no behavioural claim of its own, and L3
checks it. Blocks render at column 0 and are not indented into list items or
blockquotes -- a generated section that had to be re-wrapped by hand would be
hand-written again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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
#: listed here, or nothing would check it.
DOCUMENTS = (Path("README.md"), Path("docs/CONTRACT.md"))


def _rate(block: dict, reads: int, seeds: int) -> str:
    if block["min_pct"] == block["max_pct"] == 0.0:
        return f"0.0% ({seeds} seeds x {reads} reads, {seeds * reads} in total, none)"
    return (
        f"{block['mean_pct']:.1f}% mean, {block['min_pct']:.1f}-{block['max_pct']:.1f}% "
        f"across {seeds} seeds x {reads} reads"
    )


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def _cell(verdict: dict) -> str:
    """One scene's verdict as a table cell: the value, and the number behind it."""
    present = verdict["present"]
    word = {True: "`true`", False: "`false`", None: "`null`"}[present]
    if verdict["occupancy"] is None:
        return f"{word} —"
    return f"{word} {verdict['occupancy']:.3f}"


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


# --- the generated sections ----------------------------------------------


def _separation_block(evidence: dict) -> str:
    sep = evidence["separation"]
    rows = sorted(sep.items(), key=lambda kv: (kv[1]["texture"], kv[1]["headlight"]))
    total = sum(row["cells"] for _, row in rows)
    working = [label for label, row in rows if row["separates"]]
    failing = [label for label, row in rows if not row["separates"]]
    margins = [row["margin"] for _, row in rows if row["separates"]]

    lines = [
        f"**The matrix, unedited.** {total} cells sweeping vehicle/ground contrast "
        "through the exactly-invisible case, ground texture, the vehicle's own "
        "surface grain, and a headlight pool on the floor. Each cell carries both "
        "scenes — the vehicle and the empty lane beside it — because a measure that "
        "answered one way for everything would pass a one-sided sweep perfectly.",
        "",
        "| configuration | cells | vehicle seen | vehicle refused | empty called empty "
        "| empty read occupied | margin | separates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, row in rows:
        margin = "—" if row["margin"] is None else f"{row['margin']:.3f}"
        lines.append(
            f"| {label} | {row['cells']} | {row['vehicle_seen']} | {row['vehicle_refused']} "
            f"| {row['empty_called_empty']} | {row['empty_false_positive']} | {margin} "
            f"| {'yes' if row['separates'] else '**no**'} |"
        )
    lines.append("")
    if working:
        lines.append(
            f"It separates vehicle from empty in {len(working)} of {len(rows)} "
            f"configurations, with a worst-case occupancy margin of {min(margins):.2f}."
        )
    if failing:
        lines.append(
            f"It does **not** separate in: {'; '.join(failing)}. "
            "In those rows an empty lane reads as occupied, so the gate gives you "
            "nothing there — including no protection against the metal-plate case it "
            "exists for."
        )
    return "\n".join(lines)


def _texture_block(evidence: dict) -> str:
    floor = evidence["texture_floor"]
    axis = floor["matrix_ground_reference_texture"]
    lowest = min(axis.values())
    return "\n".join(
        [
            "**Ground with no texture of its own is NOT MEASURED, never `false`.** The "
            "comparison asks whether a window still looks like the same piece of "
            f"ground, so ground carrying nothing to recognise leaves it nothing to "
            f"work with. Below {floor['min_reference_texture']} grey levels of typical "
            "local texture the gate declines to answer and says why.",
            "",
            f"The matrix's own ground never gets near that floor — its texture axis "
            f"bottoms out at {lowest} grey levels ("
            + ", ".join(f"{k} → {v}" for k, v in sorted(axis.items()))
            + "), because the sensor's own grain is most of it. Sealed or painted "
            "concrete under a clean sensor is a different scene: it measures "
            f"{floor['smooth_floor_reference_texture']} grey levels and the gate "
            f"returns `{str(floor['smooth_floor']['present']).lower()}`, with no camera "
            "fault raised — nothing is broken, this ground is simply not one this "
            "measure can serve.",
            "",
            "**This matters more than the number suggests.** Most garage entries are "
            "covered, and a covered entry is typically sealed or painted concrete "
            "rather than open asphalt — smoother, less grain, fewer markings. The "
            "failing case may well be the common one. **NOT MEASURED**: no real floor "
            "has been photographed, so how much texture a real covered entry carries "
            "is an open question, and the remedy if it carries too little is physical "
            "— paint markings, add a textured strip in view.",
        ]
    )


def _weather_block(evidence: dict) -> str:
    weather = evidence["weather"]
    sweep = weather["sweep"]
    lines = [
        "**Weather, measured on three scenes at every coverage** — an empty lane, a "
        "vehicle, and the metal plate the gate exists to refuse. The number beside "
        "each verdict is the measured occupancy.",
        "",
        "| streak coverage | empty lane | vehicle | metal plate |",
        "|---|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['coverage']:.0%} | {_cell(row['empty_lane'])} "
            f"| {_cell(row['vehicle'])} | {_cell(row['metal_plate'])} |"
        )
    lines.append("")

    answered = weather["highest_coverage_still_answered_false"]
    occupied = weather["lowest_coverage_reading_occupied"]
    declines = weather["lowest_coverage_declining_to_answer"]
    admitted = weather["metal_plate_admitted_from"]

    if occupied is None:
        lines.append(
            f"Two bands: `false` up to {_pct(answered)} of the frame in streaks, and "
            f"`null` from {_pct(declines)}."
        )
    else:
        lines.append(
            f"**Three bands, not two.** `false` up to {_pct(answered)} of the frame in "
            f"streaks; from {_pct(occupied)} an **empty lane reads as OCCUPIED**, at up "
            f"to {max(r['empty_lane']['confidence'] or 0 for r in sweep):.2f} confidence; "
            f"from {_pct(declines)} the gate declines to answer at all."
        )
        lines.append("")
        lines.append(
            "The middle band is the one to read. `presence: true` with "
            "`outcome: \"fallback\"` tells a lane controller that a car is there and "
            "could not be identified, and this contract says refusing it is a bug in "
            "your integration — so in that band a conforming lane issues a ticket and "
            "raises an attendant for a car that is not there."
        )
    if admitted is not None:
        lines.append("")
        lines.append(
            f"**And the fraud is admitted with it.** The metal plate on the loop — the "
            f"case this gate exists for — is correctly refused up to {_pct(answered)} "
            f"coverage and then **transacts from {_pct(admitted)}**, on the same "
            "streaks. In that band the gate does not merely lose the ability to say "
            "`false`; it issues the ticket for the exact scene it was built to refuse."
        )
    lines.append("")
    lines.append(
        f"This is a measured REGRESSION against the intensity measure that preceded "
        f"it, which called heavy rain an empty lane correctly. It is recorded rather "
        f"than argued away. **It applies to open-air entries.** Most garage entries "
        f"are covered, and rain is not in a covered camera's view — how many are open "
        f"is NOT MEASURED. Across the sweep, {weather['vehicle_refusals']} of "
        f"{weather['vehicle_cells']} vehicle scenes were refused."
    )
    return "\n".join(lines)


def _headlight_block(evidence: dict) -> str:
    head = evidence["headlight"]
    sweep = head["sweep"]
    lines = [
        "**Headlights on the floor.** A covered entry is artificially lit and often "
        "dark, so an approaching car throws its beams into frame before the car "
        "itself arrives — a large change in the scene caused by a vehicle that is not "
        "yet the vehicle. Measured with and without the car that cast the pool.",
        "",
        "| beam pool, peak x ambient | empty lane (car not yet in frame) | vehicle |",
        "|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| x{1 + row['pool']:g} | {_cell(row['empty_lane'])} | {_cell(row['vehicle'])} |"
        )
    lines.append("")
    held = head["highest_pool_still_empty"]
    tripped = head["lowest_pool_reading_occupied"]
    if tripped is None:
        lines.append(
            f"An empty lane holds at `false` through every pool tested, up to "
            f"x{1 + (held or 0):g} ambient."
        )
    else:
        lines.append(
            f"An empty lane holds at `false` up to a pool of x{1 + held:g} ambient and "
            f"reads as OCCUPIED from x{1 + tripped:g} — the beams of a car that has not "
            "arrived open a transaction for it."
        )
    lines.append("")
    lines.append(
        f"{head['vehicle_refusals']} of {head['vehicle_cells']} vehicle scenes were "
        f"refused. **The model is a limitation of these numbers**: {head['model']}. A "
        "gloss or wet floor at night is a specular scene and this is a matte one. "
        "**NOT MEASURED** on any real entry."
    )
    return "\n".join(lines)


def _safety_block(evidence: dict) -> str:
    sep = evidence["separation"]
    weather = evidence["weather"]
    head = evidence["headlight"]
    matrix_cells = sum(row["cells"] for row in sep.values())
    matrix_refusals = sum(row["vehicle_refused"] for row in sep.values())
    total = matrix_cells + weather["vehicle_cells"] + head["vehicle_cells"]
    refusals = matrix_refusals + weather["vehicle_refusals"] + head["vehicle_refusals"]
    return "\n".join(
        [
            f"**The one thing that holds everywhere measured.** {refusals} wrongful "
            f"refusals in {total} scenes containing a vehicle: {matrix_cells} matrix "
            f"cells, {weather['vehicle_cells']} weather coverages and "
            f"{head['vehicle_cells']} headlight pools, each measured with a vehicle in "
            "the frame. `false` is the only value that ends a transaction, and no "
            "scene measured produced it for a frame with a vehicle in it. Where this "
            "gate fails it fails to `null` — a ticket and a human.",
            "",
            "Every one of those scenes is a drawn rectangle on a drawn lane. The claim "
            "is that the measure holds across everything that has been put through it, "
            "not that everything has been put through it.",
        ]
    )


def _conflation_block(evidence: dict) -> str:
    conflation = evidence["conflated_reasons"]
    reason = ", ".join(f"`{r}`" for r in conflation["reason_reported"]) or "one reason"
    lines = [
        f"**One reason covers several unrelated conditions, and this release cannot "
        f"tell them apart.** {reason} is reported for all of the following:",
        "",
    ]
    for cause in sorted(conflation["causes"]):
        lines.append(f"- {cause}")
    lines += [
        "",
        "It is published under `camera_faults` in `GET /v1/health`, and for a moved "
        "camera that is right. For heavy weather it is not: nothing is broken. **Do "
        "not read this reason as a confirmed equipment fault** — read it as \"the "
        "capture no longer matches the reference, for one of several reasons this "
        "build cannot separate\". Separating them needs a measurement this release "
        "does not make, and inventing one would be guessing; naming the conflation is "
        "the honest thing available now.",
    ]
    return "\n".join(lines)


#: Every generated section, by the key the documents mark it with.
BLOCKS = {
    "presence.separation": _separation_block,
    "presence.texture": _texture_block,
    "presence.weather": _weather_block,
    "presence.headlight": _headlight_block,
    "presence.safety": _safety_block,
    "presence.conflation": _conflation_block,
}


def blocks(evidence: dict) -> dict[str, str]:
    """Every generated section, rendered from the evidence."""
    return {key: render(evidence) for key, render in BLOCKS.items()}


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
