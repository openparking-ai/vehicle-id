"""The figures the documents are allowed to state, rendered from the evidence.

One place turns a measurement into the words that appear in a document. The
documents carry `<!--m:key-->...<!--/m-->` spans; `eval_presence.py --update-docs`
fills them in from `docs/measured/presence.json`, and
`tests/test_measured_docs.py` fails when a span and the evidence disagree.

The rule this enforces: **a published figure is produced by a command, not
typed.** It exists because one was typed. A README figure measured at 0.7% was
edited to 0.3% with nothing re-measuring it; the repository's own test still
said 0.7%; and the number passed review by looking measured. Neither number can
be edited by hand now without the test going red.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EVIDENCE = Path("docs/measured/presence.json")
SPAN = re.compile(r"<!--m:([a-z0-9_.]+)-->(.*?)<!--/m-->", re.DOTALL)

#: Documents that cite measurements. Anything with a span must be listed here,
#: or nothing would check it.
DOCUMENTS = (Path("README.md"), Path("docs/CONTRACT.md"))


def _rate(block: dict, reads: int, seeds: int) -> str:
    if block["min_pct"] == block["max_pct"] == 0.0:
        return f"0.0% ({seeds} seeds x {reads} reads, {seeds * reads} in total, none)"
    return (
        f"{block['mean_pct']:.1f}% mean, {block['min_pct']:.1f}-{block['max_pct']:.1f}% "
        f"across {seeds} seeds x {reads} reads"
    )


def figures(evidence: dict) -> dict[str, str]:
    """Every figure a document may state, keyed by the name it cites."""
    out: dict[str, str] = {}

    gate = evidence["gate"]
    out["gate.min_occupancy"] = f"{gate['min_occupancy']:.0%}"
    out["gate.max_occupancy"] = f"{gate['max_occupancy']:.0%}"
    out["gate.min_structural_change"] = str(gate["min_structural_change"])
    out["gate.window"] = str(gate["window"])
    out["gate.min_frame_std"] = str(gate["min_frame_std"])

    # G1/G2. What the matrix gave, stated as the measurement rather than as a
    # claim. `separates` is per ground-texture row and the worst cell in the row
    # decides it, so a row that says it separates has no failing cell in it.
    sep = evidence["separation"]
    working = sorted(t for t, row in sep.items() if row["separates"])
    failing = sorted(t for t, row in sep.items() if not row["separates"])
    margins = [row["margin"] for row in sep.values() if row["separates"]]
    out["separation.textures_that_work"] = ", ".join(working) if working else "none"
    out["separation.textures_that_fail"] = ", ".join(failing) if failing else "none"
    out["separation.margin"] = f"{min(margins):.2f}" if margins else "no separation at all"
    out["separation.cells"] = str(sum(row["cells"] for row in sep.values()))
    out["separation.refusals"] = str(sum(row["vehicle_refused"] for row in sep.values()))

    weather = evidence["weather"]
    highest = weather["highest_coverage_still_answered_false"]
    out["weather.answers_up_to"] = (
        f"{highest:.0%} of the frame in streaks" if highest is not None else "no coverage at all"
    )

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


def load_evidence(root: Path | None = None) -> dict:
    path = (root or Path.cwd()) / EVIDENCE
    return json.loads(path.read_text(encoding="utf-8"))


def cited(text: str) -> list[tuple[str, str]]:
    """The (key, stated value) pairs a document cites."""
    return [(m.group(1), m.group(2)) for m in SPAN.finditer(text)]


def rewrite(text: str, values: dict[str, str]) -> str:
    def swap(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"the documents cite {key!r}, which nothing measures")
        return f"<!--m:{key}-->{values[key]}<!--/m-->"

    return SPAN.sub(swap, text)
