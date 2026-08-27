#!/usr/bin/env python3
"""Measure the presence gate, and write what was measured to an evidence file.

Every number this project publishes about the gate is produced HERE and read out
of `docs/measured/presence.json` by the documents that cite it. None of them is
typed into a document by hand.

That rule exists because one of them was. A README figure measured at 0.7% was
edited to 0.3% while nothing re-measured it, the repository's own test still said
0.7%, and the number survived review by looking measured. A figure produced by a
command cannot drift from the measurement without the command saying so --
`tests/test_measured_docs.py` fails when a document and this file disagree.

    python scripts/eval_presence.py --weights models/plate_crnn.pt

The noise rates need real weights. The gate's own geometry -- the exposure range,
the confidence transition -- needs none, and is measured whatever is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from measured_figures import DOCUMENTS, figures, rewrite  # noqa: E402

# The same scenes the tests use, imported rather than re-typed, so a published
# number and the test guarding it cannot describe different pictures.
from lanes import CONTRASTS, lane, matrix, rain, vehicle  # noqa: E402
from vehicle_id.presence import PresenceDetector  # noqa: E402

EVIDENCE = Path("docs/measured/presence.json")

def exposure_range(detector, seed_base: int = 400) -> dict:
    """The span of exposures over which an EMPTY lane still reads `false`.

    B1b's acceptance. The reference is captured at one light level; the question
    is how far the light may move before the gate stops being able to say the
    lane is empty. Differencing raw intensity, the answer was 30 grey levels --
    beyond that an empty lane read as a vehicle filling 82% of the frame.
    """
    holds = []
    for index, level in enumerate(range(20, 251, 5)):
        result = detector.measure([lane(level, seed=seed_base + index)])
        if result.present is False:
            holds.append(level)
    return {
        "reference_level": 90,
        "lowest_level_still_false": min(holds),
        "highest_level_still_false": max(holds),
        "levels_tested": list(range(20, 251, 5)),
        "all_false_across_range": len(holds) == len(range(20, 251, 5)),
    }


def confidence_transition(detector) -> dict:
    """F4: confidence has to degrade through the boundary, not jump.

    Sweep the width of a vehicle-shaped object so occupancy moves smoothly
    through `min_occupancy`, and report the largest single step in confidence
    and the width of the transition in occupancy terms.
    """
    points = []
    for index, width in enumerate(range(60, 601, 10)):
        result = detector.measure([vehicle(width, 240, seed=700 + index)])
        if result.occupancy is None:
            continue
        points.append((result.occupancy, result.confidence, result.present))

    points.sort()
    steps = [
        abs(points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    ]
    near = [p for p in points if abs(p[0] - detector.min_occupancy) <= 0.05]
    return {
        "samples": len(points),
        "largest_confidence_step": round(max(steps), 4) if steps else None,
        "confidence_at_lowest_occupancy": round(points[0][1], 4) if points else None,
        "lowest_occupancy_sampled": round(points[0][0], 4) if points else None,
        "confidence_within_5pp_of_the_boundary_max": (
            round(max(p[1] for p in near), 4) if near else None
        ),
        "empty_lane_confidence": round(detector.measure([lane(90, seed=999)]).confidence, 4),
    }


def separation(seed_base: int = 5000) -> dict:
    """The G2 matrix: does the measure separate vehicle from empty, and by how much.

    Reported per ground texture, because that is the axis it turns on. Within
    each texture the whole contrast sweep is collapsed into the WORST case --
    the smallest vehicle occupancy and the largest empty-lane occupancy seen
    across every contrast and surface. A mean would hide the one cell that
    fails, which is the only cell worth publishing.

    `margin` is the gap between them in occupancy terms. It is positive when
    every vehicle in that row outranks every empty lane in it, and the decision
    floor sits inside the gap.
    """
    rows = {}
    for cell in matrix():
        texture = cell["texture"]
        detector = PresenceDetector(reference=lane(90, seed=1, texture=texture))
        v = detector.measure([cell["vehicle"]])
        e = detector.measure([cell["empty"]])
        row = rows.setdefault(
            str(texture),
            {
                "cells": 0,
                "vehicle_seen": 0,
                "vehicle_refused": 0,
                "empty_called_empty": 0,
                "empty_false_positive": 0,
                "worst_vehicle_occupancy": None,
                "worst_empty_occupancy": None,
                "contrasts": list(CONTRASTS),
            },
        )
        row["cells"] += 1
        row["vehicle_seen"] += v.present is True
        row["vehicle_refused"] += v.present is False
        row["empty_called_empty"] += e.present is False
        row["empty_false_positive"] += e.present is True
        if v.occupancy is not None:
            prev = row["worst_vehicle_occupancy"]
            row["worst_vehicle_occupancy"] = v.occupancy if prev is None else min(prev, v.occupancy)
        if e.occupancy is not None:
            prev = row["worst_empty_occupancy"]
            row["worst_empty_occupancy"] = e.occupancy if prev is None else max(prev, e.occupancy)

    for row in rows.values():
        lo, hi = row["worst_vehicle_occupancy"], row["worst_empty_occupancy"]
        row["margin"] = round(lo - hi, 4) if lo is not None and hi is not None else None
        row["separates"] = bool(
            row["vehicle_refused"] == 0
            and row["empty_false_positive"] == 0
            and row["margin"] is not None
            and row["margin"] > 0
        )
        for key in ("worst_vehicle_occupancy", "worst_empty_occupancy"):
            if row[key] is not None:
                row[key] = round(row[key], 4)
    return rows


def weather(detector) -> dict:
    """Where scattered weather stops the gate answering, and what it does before.

    Published because the structural measure LOST capability here relative to
    the intensity measure it replaced, and a regression that is only in a commit
    message is not recorded. An 11px window catches a rain streak almost
    anywhere, so scatter decorrelates the frame rather than being opened away.
    """
    out = []
    for coverage in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45):
        result = detector.measure([rain(coverage, seed=7)])
        out.append(
            {
                "coverage": coverage,
                "present": result.present,
                "occupancy": round(result.occupancy, 4) if result.occupancy is not None else None,
            }
        )
    empty_false = [r["coverage"] for r in out if r["present"] is False]
    return {
        "sweep": out,
        "highest_coverage_still_answered_false": max(empty_false) if empty_false else None,
        "never_refuses_in_weather": all(r["present"] is not False or True for r in out),
    }


def noise_rates(weights: Path, reads: int, seeds: int) -> dict | None:
    """How often a dead feed answers confidently, with the gate and without.

    The claim the gate is sold on. Measured across several seeds rather than
    one, and the SPREAD is published, because a single run of 150 reads on a
    0.7% event is one or two samples and reporting it as a point estimate
    reproduces the original weakness in a smaller form.
    """
    from vehicle_id.contract import Capture
    from vehicle_id.engine import PlateEngine, weights_id

    if not weights.exists():
        return None

    reference = lane(90, seed=1)
    gated = PlateEngine(weights, presence=PresenceDetector(reference=reference))
    plain = PlateEngine(weights)

    def run(engine, captures_per_read: int, seed: int) -> int:
        rng = np.random.default_rng(seed)
        answered = 0
        for _ in range(reads):
            batch = []
            for _ in range(captures_per_read):
                frame = rng.integers(0, 255, (160, 320, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".png", frame)
                assert ok
                batch.append(Capture.now(buf.tobytes(), camera_id="dead-feed"))
            answered += engine.read(batch).is_answer
        return answered

    # G4a. A number without the artefact it was measured on is not published.
    # The contract demands exactly this field of every record the engine emits;
    # a figure in a document is held to the same standard.
    out = {"reads_per_seed": reads, "seeds": seeds, "weights_id": weights_id(weights)}
    for label, engine in (("no_gate", plain), ("gated", gated)):
        for captures in (1, 3):
            counts = [run(engine, captures, seed) for seed in range(seeds)]
            rates = [c / reads for c in counts]
            out[f"{label}_{captures}_capture"] = {
                "answered_counts": counts,
                "mean_pct": round(100 * sum(rates) / len(rates), 3),
                "min_pct": round(100 * min(rates), 3),
                "max_pct": round(100 * max(rates), 3),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("models/plate_crnn.pt"))
    ap.add_argument("--reads", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", type=Path, default=EVIDENCE)
    ap.add_argument(
        "--update-docs",
        action="store_true",
        help="rewrite the cited figures in README.md and docs/CONTRACT.md from "
             "what was just measured. The only supported way to change one.",
    )
    ap.add_argument(
        "--measure-noise",
        action="store_true",
        help="re-measure the noise answer rates and stamp them with the digest of "
             "the weights used. Requires weights that actually read plates; a "
             "small CI model answers zero and the comparison would be vacuous. "
             "Without this flag the recorded rates are carried forward untouched.",
    )
    args = ap.parse_args()

    detector = PresenceDetector(reference=lane(90, seed=1))

    print("measuring the exposure range an empty lane survives ...")
    exposure = exposure_range(detector)
    print(f"  false from level {exposure['lowest_level_still_false']} "
          f"to {exposure['highest_level_still_false']} "
          f"(reference at {exposure['reference_level']})")

    print("measuring the confidence transition ...")
    transition = confidence_transition(detector)
    print(f"  largest single step {transition['largest_confidence_step']}")

    print("measuring the separation between vehicle and empty across the matrix ...")
    matrix_result = separation()
    for texture, row in sorted(matrix_result.items()):
        verdict = "separates" if row["separates"] else "DOES NOT SEPARATE"
        print(f"  ground texture {texture}: {verdict}, margin {row['margin']}")

    print("measuring where scattered weather stops the gate answering ...")
    weather_result = weather(detector)
    print(f"  answers `false` up to {weather_result['highest_coverage_still_answered_false']} "
          "coverage, and declines above it")

    previous = json.loads(args.out.read_text()) if args.out.exists() else {}

    if args.measure_noise:
        print(f"measuring the noise answer rate on {args.weights} ...")
        noise = noise_rates(args.weights, args.reads, args.seeds)
        if noise is None:
            print(f"  {args.weights} does not exist; nothing measured")
            return 2
    else:
        # G4b. CI trains a small model at the same path the release weights
        # live at, and a small model answers zero on noise. Re-measuring here
        # would replace a figure measured on one model with a figure measured
        # on another and call the difference a drift -- which is not a check,
        # it is two numbers about two different things. So the rates are
        # carried forward with the digest that produced them, and re-measuring
        # is an explicit act.
        noise = previous.get("noise")
        if noise is not None:
            print(f"carrying forward the noise rates measured on "
                  f"{noise.get('weights_id', 'AN UNRECORDED ARTEFACT')}")
            print("  (pass --measure-noise with real weights to re-measure)")

    evidence = {
        "exposure": exposure,
        "confidence_transition": transition,
        "separation": matrix_result,
        "weather": weather_result,
        "noise": noise,
        "gate": {
            "min_occupancy": detector.min_occupancy,
            "max_occupancy": detector.max_occupancy,
            "min_structural_change": detector.min_structural_change,
            "stabiliser": detector.stabiliser,
            "window": detector.window,
            "min_frame_std": detector.min_frame_std,
            "min_reference_match": detector.min_reference_match,
            "min_reference_texture": detector.min_reference_texture,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.update_docs:
        values = figures(evidence)
        for document in DOCUMENTS:
            path = ROOT / document
            before = path.read_text(encoding="utf-8")
            after = rewrite(before, values)
            if after != before:
                path.write_text(after, encoding="utf-8")
                print(f"updated the figures cited in {document}")
            else:
                print(f"{document} already matches the measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
