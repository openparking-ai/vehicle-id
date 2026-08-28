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

**Every boolean this file publishes carries a CONTROL**, in the `controls`
block: the input that makes it false, measured in the same run. A boolean nobody
has ever seen go red is not evidence, and this file shipped one -- the weather
safety flag was `all(x is not False or True ...)`, which is `True` for every
input that exists. It was the load-bearing safety property of the module. See
`controls()` below, and `tests/test_measured_docs.py` for the check that every
boolean has one and that every control is actually false.
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
from measured_figures import BLOCKS, DOCUMENTS, blocks, figures, rewrite  # noqa: E402

# The same scenes the tests use, imported rather than re-typed, so a published
# number and the test guarding it cannot describe different pictures.
from lanes import (  # noqa: E402
    CONTRASTS,
    HEADLIGHT_LEVELS,
    TEXTURES,
    VEHICLE_SIZE,
    H,
    W,
    lane,
    matrix,
    rain,
    smooth_floor,
    vehicle,
)
from vehicle_id.presence import STREAK_CONDITION, PresenceDetector  # noqa: E402

EVIDENCE = Path("docs/measured/presence.json")

#: The coverages the weather sweep samples, and the beam pools the headlight
#: sweep samples. Named here because the control runs have to sweep the same
#: points as the real run, or the control is measuring something else.
RAIN_COVERAGES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45)

#: The object the metal-plate case is made of: roughly plate-sized against a
#: lane-filling camera, about 1% of the frame. The scene the gate exists for.
PLATE_SIZE = (80, 40)

#: How much of the frame the matrix's vehicle covers. Published beside the cells
#: where that vehicle was not measured, because "an ordinary car" and "a car
#: close enough to fill the frame" land on the same reason and only the number
#: tells them apart.
VEHICLE_FRAME_FRACTION = (VEHICLE_SIZE[0] * VEHICLE_SIZE[1]) / (W * H)

#: Real frames ever put through this gate. Published so that every NOT MEASURED
#: caveat in the documents is DERIVED from it rather than written -- a sentence
#: that says "no real footage" while a number beside it says otherwise is the
#: failure this whole layer exists to make impossible.
REAL_FRAMES_MEASURED = 0


def _verdict(result) -> dict:
    return {
        "present": result.present,
        "occupancy": round(result.occupancy, 4) if result.occupancy is not None else None,
        "confidence": round(result.confidence, 4) if result.confidence is not None else None,
        "camera_health": result.camera_health,
    }


def exposure_range(detector, seed_base: int = 400, levels=range(20, 251, 5)) -> dict:
    """The span of exposures over which an EMPTY lane still reads `false`.

    B1b's acceptance. The reference is captured at one light level; the question
    is how far the light may move before the gate stops being able to say the
    lane is empty. Differencing raw intensity, the answer was 30 grey levels --
    beyond that an empty lane read as a vehicle filling 82% of the frame.

    `levels` is a parameter so that the control run can sweep a range this is
    NOT expected to survive. See `controls()`.
    """
    levels = list(levels)
    holds = []
    for index, level in enumerate(levels):
        result = detector.measure([lane(level, seed=seed_base + index)])
        if result.present is False:
            holds.append(level)
    return {
        "reference_level": 90,
        "lowest_level_still_false": min(holds) if holds else None,
        "highest_level_still_false": max(holds) if holds else None,
        "levels_tested": levels,
        "all_false_across_range": len(holds) == len(levels),
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


def _row_label(texture: float, headlight: float) -> str:
    lit = "headlights off" if headlight == 0 else f"headlight pool x{1 + headlight:g}"
    return f"ground texture {texture:g}, {lit}"


def separation(**detector_kwargs) -> dict:
    """The G2 matrix: does the measure separate vehicle from empty, and by how much.

    Reported per (ground texture, headlight) configuration, because those are the
    axes it turns on. Within each row the whole contrast and surface sweep is
    collapsed into the WORST case -- the smallest vehicle occupancy and the
    largest empty-lane occupancy seen across every cell in the row. A mean would
    hide the one cell that fails, which is the only cell worth publishing.

    Headlights are a row rather than a column deliberately. The empty scene of a
    lit row is a lane with a beam pool on it and NO car in frame -- the second
    before a car arrives -- and a gate that calls that occupied transacts for a
    vehicle that is not there yet. Folding it in with the unlit rows would let
    one hide inside the other.

    `margin` is the gap between them in occupancy terms. It is positive when
    every vehicle in that row outranks every empty lane in it, and the decision
    floor sits inside the gap.

    `detector_kwargs` exists for the control run, which deliberately breaks the
    detector so that `separates` can be seen to go false.
    """
    rows = {}
    for cell in matrix():
        reference = lane(90, seed=1, texture=cell["texture"])
        detector = PresenceDetector(reference=reference, **detector_kwargs)
        v = detector.measure([cell["vehicle"]])
        e = detector.measure([cell["empty"]])
        row = rows.setdefault(
            _row_label(cell["texture"], cell["headlight"]),
            {
                "texture": cell["texture"],
                "headlight": cell["headlight"],
                "cells": 0,
                "vehicle_seen": 0,
                "vehicle_refused": 0,
                "vehicle_not_measured": 0,
                "vehicle_not_measured_health": {},
                "vehicle_not_measured_cells": [],
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
        # Y5. A cell where the vehicle was neither admitted nor refused used to
        # be a hole in the arithmetic: `cells` and `vehicle_seen` disagreed and
        # nothing said why. It is the interesting case -- a frame with a car in
        # it that the gate answered nothing about, and named EQUIPMENT as the
        # reason -- so the reason is recorded, not just the count.
        if v.present is None:
            health = v.camera_health or "no fault reported"
            row["vehicle_not_measured"] += 1
            row["vehicle_not_measured_health"][health] = (
                row["vehicle_not_measured_health"].get(health, 0) + 1
            )
            row["vehicle_not_measured_cells"].append(
                {
                    "contrast": cell["contrast"],
                    "surface": cell["surface"],
                    "vehicle_frame_fraction": round(VEHICLE_FRAME_FRACTION, 4),
                    "camera_health": health,
                    "reason": v.reason,
                }
            )
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


def weather(detector, coverages=RAIN_COVERAGES) -> dict:
    """What weather does to each of the three scenes that matter.

    Published because the structural measure LOST capability here relative to
    the intensity measure it replaced, and a regression that is only in a commit
    message is not recorded. An 11px window catches a rain streak almost
    anywhere, so scatter decorrelates the frame rather than being opened away.

    THREE scenes at every coverage, not one. The previous version measured the
    empty lane alone and the documents then claimed "no wrongful refusal across
    the weather sweep" -- a claim about frames containing a vehicle, made over a
    sweep in which no frame contained one. The vehicle case is now measured, and
    so is the metal plate, because the plate is what the gate exists for and
    "the fraud is admitted in moderate rain" is the operational consequence a
    garage has to be told about.
    """
    sweep = []
    for coverage in coverages:
        empty = detector.measure([rain(coverage, seed=7)])
        car = detector.measure(
            [rain(coverage, seed=7, base=vehicle(*VEHICLE_SIZE, seed=7, contrast=1.0))]
        )
        plate = detector.measure(
            [rain(coverage, seed=7, base=vehicle(*PLATE_SIZE, seed=7, contrast=2.05))]
        )
        sweep.append(
            {
                "coverage": coverage,
                "empty_lane": _verdict(empty),
                "vehicle": _verdict(car),
                "metal_plate": _verdict(plate),
            }
        )

    answered_false = [r["coverage"] for r in sweep if r["empty_lane"]["present"] is False]
    read_occupied = [r["coverage"] for r in sweep if r["empty_lane"]["present"] is True]
    declined = [r["coverage"] for r in sweep if r["empty_lane"]["present"] is None]
    admitted = [r["coverage"] for r in sweep if r["metal_plate"]["present"] is True]
    refusals = sum(1 for r in sweep if r["vehicle"]["present"] is False)
    occupied_confidences = [
        r["empty_lane"]["confidence"]
        for r in sweep
        if r["empty_lane"]["present"] is True and r["empty_lane"]["confidence"] is not None
    ]
    return {
        "sweep": sweep,
        "highest_coverage_still_answered_false": max(answered_false) if answered_false else None,
        "coverages_reading_occupied_with_an_empty_lane": read_occupied,
        "lowest_coverage_reading_occupied": min(read_occupied) if read_occupied else None,
        # The confidence the gate puts behind an empty lane it has called
        # occupied. Published as its own figure rather than recomputed from the
        # table by whatever prints it: a sentence derived from a summary nobody
        # can perturb is a sentence nobody can check.
        "highest_confidence_reading_occupied": (
            max(occupied_confidences) if occupied_confidences else None
        ),
        "lowest_coverage_declining_to_answer": min(declined) if declined else None,
        "metal_plate_admitted_from": min(admitted) if admitted else None,
        "vehicle_cells": len(sweep),
        "vehicle_refusals": refusals,
        # FIXED. This was `all(r["present"] is not False or True for r in out)`,
        # which is True for every input in the universe, and it was the module's
        # load-bearing safety property. It now reads the vehicle scenes -- which
        # did not exist when it was written -- and `controls()` shows it false.
        "never_refuses_in_weather": refusals == 0 and len(sweep) > 0,
    }


def headlights(detector, amounts=HEADLIGHT_LEVELS, ambient: float = 90) -> dict:
    """A beam pool on the floor, with and without the car that cast it.

    M3's axis. A car with its beams on throws them into frame BEFORE the car
    does. That is a large change in the scene caused by a vehicle that is not
    yet the vehicle, and nothing had measured what this gate makes of it.

    The empty-lane row is the interesting one: it is the second before a car
    arrives, and a `true` there is a transaction opened for a vehicle that has
    not got there. The vehicle row is the ordinary night arrival.

    G1: `ambient` is a PARAMETER and every row records the level it was measured
    at, because the boundary is a property of the sweep and not of the gate --
    `lane()` clips to uint8 after the pool multiplies, so saturation and
    therefore the boundary move with the level. It was previously pinned at 90
    inside the body, where nothing published it. `ambient_levels_swept` is
    counted from the rows rather than typed. It is a count of how many levels
    THIS run swept, and nothing more: one caller passes one `ambient`, so it is
    1 by construction, and no document branches on it.
    """
    sweep = []
    for amount in amounts:
        empty = detector.measure([lane(ambient, seed=610, headlight=amount)])
        car = detector.measure(
            [vehicle(*VEHICLE_SIZE, level=ambient, seed=611, contrast=1.0, headlight=amount)]
        )
        sweep.append(
            {
                "pool": amount,
                "ambient": ambient,
                "empty_lane": _verdict(empty),
                "vehicle": _verdict(car),
            }
        )

    held = [r["pool"] for r in sweep if r["empty_lane"]["present"] is False]
    tripped = [r["pool"] for r in sweep if r["empty_lane"]["present"] is True]
    refusals = sum(1 for r in sweep if r["vehicle"]["present"] is False)
    return {
        "sweep": sweep,
        "model": "multiplicative pool on a matte floor; no specular glare, no beam cut-off",
        # X5/G6. `pool` is what the beam ADDS and PEAK is the published
        # convention, so the two differ by one and a table reading "x3" sits
        # beside a stored 2.0. The offset is published as a VALUE rather than
        # as a sentence a reader has to have read: every rendered figure derives
        # from it, and the table carries both columns so the conversion is on
        # the page rather than in someone's memory.
        "peak_offset": 1,
        "ambient_level": ambient,
        "ambient_levels_swept": len({row["ambient"] for row in sweep}),
        "highest_pool_still_empty": max(held) if held else None,
        "lowest_pool_reading_occupied": min(tripped) if tripped else None,
        "vehicle_cells": len(sweep),
        "vehicle_refusals": refusals,
        "never_refuses_under_headlights": refusals == 0 and len(sweep) > 0,
    }


def texture_floor(scenes=None) -> dict:
    """Q3a: the branch that says "this ground carries nothing to recognise".

    It was never once reached. `min_reference_texture` is 1.5 grey levels, and
    the fixture's sensor grain alone is 3.15 at level 90 -- so no value of the
    `texture` axis could get under the floor, and an axis existed that could not
    reach the code path it was there to test. Grain is now its own axis and
    `smooth_floor()` is a real picture, in focus, of ground with nothing on it.

    Both halves are published: the reference texture the matrix's own ground
    measures (all comfortably above the floor, which is the fact that hid this),
    and the smooth floor that reaches it.
    """
    axis = {}
    for texture in TEXTURES:
        detector = PresenceDetector(reference=lane(90, seed=1, texture=texture))
        axis[f"texture {texture:g}"] = round(detector._reference_texture, 3)

    detector = PresenceDetector(reference=smooth_floor())
    measured = detector.measure([smooth_floor()])
    return {
        "min_reference_texture": detector.min_reference_texture,
        "matrix_ground_reference_texture": axis,
        "matrix_axis_can_reach_the_floor": any(
            v < detector.min_reference_texture for v in axis.values()
        ),
        "smooth_floor_reference_texture": round(detector._reference_texture, 3),
        "smooth_floor": _verdict(measured),
        # X1c. The documents said the gate "declines to answer and says why"
        # while nothing recorded the why, so half that sentence had no
        # measurement behind it in either direction. This is the why.
        "smooth_floor_reason": measured.reason,
        "reached": measured.present is None and "texture" in measured.reason,
    }


def conflated_reasons() -> dict:
    """K3/Q1: what `reference_not_recognised` actually means, measured.

    The label is documented as an equipment fault -- "something wrong with the
    camera or the reference" -- and the service publishes it under
    `camera_faults`. Heavy weather lands on it too, and weather is not a fault.

    The fix needs a discriminator this release does not have: the branch is
    reached by a moved camera, a rebuilt scene, a vehicle filling the frame and
    heavy weather alike, and `presence.py` says so in as many words -- "All
    three are indistinguishable from here." Relabelling the branch would trade a
    false page for a MISSING one, on the knocked camera the contract advertises.
    So the release DISCLOSES the conflation instead of guessing, and this
    measures the disclosure so the document cannot drift from it.

    X4. The fourth cause is the one that makes this a product finding rather
    than a documentation one, and it came out of the matrix that was already
    being measured: an ORDINARY vehicle -- 43.75% of the frame, not one filling
    it -- on low-texture ground, under a beam pool, lands on the same reason.
    An arriving car is counted under `camera_faults`. It is measured here as
    the matrix builds it, reference and all, so that the scene in the document
    is the scene in the sweep.
    """
    from vehicle_id.plates.generator import PlateGenerator

    plain = lane(90, seed=1)
    arrival = next(
        cell
        for cell in matrix()
        if cell["texture"] == 0.25
        and cell["headlight"] == 2.0
        and cell["contrast"] == 2.05
        and cell["surface"] == 0.0
    )
    # (reference the gate was configured with, capture put through it)
    # The name carries its own summary before the colon. The caveat below names
    # the conditions the label is RIGHT about and there is no room there for the
    # whole elaboration, so the head of the name is what it quotes -- a rule,
    # rather than a second wording of the same thing kept in step by hand.
    knocked = (
        "a capture that is not a view of this lane: a camera knocked out of "
        "alignment, or a scene rebuilt overnight"
    )
    causes = {
        "a vehicle close enough to fill the frame": (plain, vehicle(636, 356, seed=13)),
        "heavy weather": (plain, rain(0.45, seed=7)),
        knocked: (plain, PlateGenerator(seed=11).sample(degradation=0).image),
        "an ordinary vehicle arriving on low-texture ground under a beam pool": (
            lane(90, seed=1, texture=arrival["texture"]),
            arrival["vehicle"],
        ),
    }
    seen = {
        name: PresenceDetector(reference=reference).measure([scene]).camera_health
        for name, (reference, scene) in causes.items()
    }
    return {
        "causes": seen,
        # Which of them the label is actually RIGHT about. Published rather than
        # decided by whatever renders the document: the disclosure's whole point
        # is that the reason is correct for one of these and wrong for the rest,
        # and a renderer choosing which on its own would be a hand-written claim
        # in the one section that exists to have none.
        "causes_that_are_equipment_faults": [knocked],
        "all_report_the_same_reason": len(set(seen.values())) == 1,
        "reason_reported": sorted(set(v for v in seen.values() if v)),
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


# --- the controls --------------------------------------------------------
#
# K5b. The fail-control rule, applied to an evidence file rather than to a test.
# It exists because this file published `never_refuses_in_weather: true` from an
# expression that could not evaluate to anything else, and that boolean was the
# module's load-bearing safety property. A flag nobody has ever seen go red is
# not evidence of anything, and neither is one that CANNOT.
#
# Every boolean the evidence file publishes appears here with the input that
# makes it false, MEASURED in the same run rather than argued. Two of them come
# free -- the low-texture rows really do fail to separate, and the matrix's own
# ground really cannot reach the texture floor -- and a control that occurs in
# the published data is worth more than a planted one. The rest are planted.
#
# `tests/test_measured_docs.py` fails if a published boolean has no control, or
# if any control is not false.


def controls(detector) -> dict:
    """The input that makes each published boolean false, measured."""
    out = {}

    # Break the light, not the gate: sweep exposures the fit cannot reconcile
    # with a reference captured at 90. A gain under 0.2 is not this lane.
    out["exposure.all_false_across_range"] = {
        "how": "the same sweep extended to light levels 1-11, which the "
               "illumination fit cannot reconcile with a reference captured at 90",
        "value": exposure_range(detector, seed_base=900, levels=range(1, 12, 2))[
            "all_false_across_range"
        ],
    }

    # A detector that cannot see a change at all. Nothing registers as changed,
    # so no vehicle is seen and no row can separate.
    broken = separation(min_structural_change=0.999)
    out["separation.separates"] = {
        "how": "the same matrix with min_structural_change at 0.999, so no window "
               "can register as changed",
        "value": any(row["separates"] for row in broken.values()),
        "also_occurs_unplanted": "the ground texture 0.25 rows separate=false in the "
                                 "published run; a real failure beats a planted one",
    }

    # A floor so high that a vehicle occupying 43% of the frame is called an
    # empty lane. That is the wrongful refusal the flag exists to detect, and it
    # controls both the summary flag AND the raw verdicts it summarises: under
    # this detector a vehicle in rain reads `false`, in the same table, so the
    # published "the vehicle scenes never read false" is a statement the run can
    # be seen to contradict when it is untrue.
    refusing = PresenceDetector(reference=lane(90, seed=1), min_occupancy=0.85)
    broken_weather = weather(refusing)
    broken_headlight = headlights(refusing)
    how_refusing = (
        "the same sweep with min_occupancy at 0.85, so a vehicle reads as an empty lane"
    )
    out["weather.never_refuses_in_weather"] = {
        "how": how_refusing,
        "value": broken_weather["never_refuses_in_weather"],
    }
    out["headlight.never_refuses_under_headlights"] = {
        "how": how_refusing,
        "value": broken_headlight["never_refuses_under_headlights"],
    }
    # The raw verdicts underneath those flags. A verdict published only ever as
    # `true` is a claim like any other, and this is the input that makes it
    # `false` -- recorded as the verdict itself, measured, not as a promise.
    out["weather.present"] = {
        "how": how_refusing + "; the verdict for the vehicle scene at the lowest coverage",
        "value": broken_weather["sweep"][0]["vehicle"]["present"],
    }
    out["headlight.present"] = {
        "how": how_refusing + "; the verdict for the vehicle scene at the lowest pool",
        "value": broken_headlight["sweep"][0]["vehicle"]["present"],
    }

    # Conditions that do NOT share a reason, so the flag saying three of them do
    # can be seen to go false. A dead feed and a knocked view are named
    # differently, and that is the distinction the conflation flag is about.
    from lanes import flat as _flat_scene

    # X5. The label was wrong: this scene is the frame-filling vehicle, which
    # the measured section names "a vehicle close enough to fill the frame". A
    # control that mislabels its own input is a small version of the disease
    # this block exists for.
    distinct = {
        "a dead camera": detector.measure([_flat_scene(0)]).camera_health,
        "a vehicle close enough to fill the frame": detector.measure(
            [vehicle(636, 356, seed=13)]
        ).camera_health,
    }
    out["conflated_reasons.all_report_the_same_reason"] = {
        "how": "a dead camera beside a frame-filling object; these are named "
               "differently, so the flag can be seen to distinguish them",
        "value": len(set(distinct.values())) == 1,
        "reasons_seen": distinct,
    }

    # The one that was true for two rounds without anybody noticing: the matrix
    # axis cannot reach the texture floor, and it is published beside the scene
    # that can.
    #
    # X5. This used to publish `matrix_axis_can_reach_the_floor` -- a DIFFERENT
    # predicate that happens to be false -- rather than re-running `reached` on
    # the matrix ground. A control has to be the same measurement under a
    # different input, or it proves something about a quantity nobody published.
    smoothest = lane(90, seed=1, texture=min(TEXTURES))
    on_matrix_ground = PresenceDetector(reference=smoothest).measure([smoothest])
    out["texture_floor.reached"] = {
        "how": "the same predicate on the matrix's own ground at its smoothest "
               "setting; sensor grain alone keeps it above the floor, so the "
               "untextured branch is never taken",
        "value": on_matrix_ground.present is None and "texture" in on_matrix_ground.reason,
        "reason_seen": on_matrix_ground.reason,
    }
    return out


#: What the gate is MEASURED not to be able to do, produced from the measurement
#: rather than listed by hand.
#:
#: Y4. `tests/test_measured_docs.py` requires every entry's `seam_word` to appear
#: in `presence.KNOWN_LIMITS`, so a limitation cannot be measured without the
#: operator switching the gate on being told about it. The check used to iterate
#: a hard-coded 5-tuple inside the test and its docstring promised that adding a
#: measured limitation would turn it red; it could not. Deriving the list from
#: the top-level SECTIONS of this file instead would demand a seam string for
#: `gate` and `exposure`, which are not limitations, and the implementer would
#: add an exclusion set -- the same hard-coded tuple with extra steps. So the
#: measurement says which of its results are limitations, here, beside the code
#: that measured them.
def limitations(sep: dict, weather_result: dict, headlight_result: dict,
                floor: dict, conflation: dict) -> list[dict]:
    """Every measured thing this gate cannot do, and the words the seam must say.

    `seam_word` is a PHRASE, not a keyword, and deliberately one that only the
    disclosure covering this limitation contains. A bare word matches the whole
    concatenated disclosure, so "headlight" was satisfied by the camera-fault
    caveat mentioning a headlight pool and the headlight limitation could have
    been deleted with nothing going red -- a check measuring the presence of a
    word rather than the presence of a disclosure.
    """
    found = []
    if any(not row["separates"] for row in sep.values()):
        found.append({
            "measured_in": "separation",
            "topic": "ground smooth enough that vehicle and empty lane do not separate",
            "seam_word": "smooth ground",
        })
    if weather_result["lowest_coverage_reading_occupied"] is not None:
        found.append({
            "measured_in": "weather",
            "topic": f"an empty lane reads as occupied wherever {STREAK_CONDITION}",
            # The phrase only this disclosure contains, and it moved with the
            # wording: "moderate rain" named a cause the measurement never
            # isolated. Coverage of the frame by streaks is the axis.
            "seam_word": "bright streaks",
        })
    if weather_result["metal_plate_admitted_from"] is not None:
        found.append({
            "measured_in": "weather",
            "topic": "the metal plate on the loop transacts in that band",
            "seam_word": "metal plate",
        })
    if headlight_result["lowest_pool_reading_occupied"] is not None:
        found.append({
            "measured_in": "headlight",
            "topic": "a beam pool reads as occupied before the car that cast it arrives",
            "seam_word": "headlight pool on the floor",
        })
    if floor["reached"]:
        found.append({
            "measured_in": "texture_floor",
            "topic": "ground under the texture floor is declined rather than answered",
            "seam_word": "it fails to null",
        })
    if conflation["all_report_the_same_reason"]:
        found.append({
            "measured_in": "conflated_reasons",
            "topic": "one reason covers several unrelated conditions",
            "seam_word": "reference_not_recognised",
        })
    if any(row["vehicle_not_measured"] for row in sep.values()):
        found.append({
            "measured_in": "separation",
            "topic": "an ordinary arrival is reported as a camera fault",
            "seam_word": "camera_faults",
        })
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("models/plate_crnn.pt"))
    ap.add_argument("--reads", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", type=Path, default=EVIDENCE)
    ap.add_argument(
        "--update-docs",
        action="store_true",
        help="rewrite the cited figures AND the generated sections in README.md "
             "and docs/CONTRACT.md from what was just measured. The only "
             "supported way to change one.",
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
    for label, row in sorted(matrix_result.items()):
        verdict = "separates" if row["separates"] else "DOES NOT SEPARATE"
        print(f"  {label}: {verdict}, margin {row['margin']}")

    print("measuring what weather does to an empty lane, a vehicle and a metal plate ...")
    weather_result = weather(detector)
    print(f"  empty lane answers `false` up to "
          f"{weather_result['highest_coverage_still_answered_false']}, reads OCCUPIED from "
          f"{weather_result['lowest_coverage_reading_occupied']}, declines from "
          f"{weather_result['lowest_coverage_declining_to_answer']}")
    print(f"  the metal plate is admitted from {weather_result['metal_plate_admitted_from']}; "
          f"vehicles refused: {weather_result['vehicle_refusals']}")

    print("measuring a headlight pool on the floor, with and without the car ...")
    headlight_result = headlights(detector)
    offset = headlight_result["peak_offset"]
    held = headlight_result["highest_pool_still_empty"] or 0
    tripped = headlight_result["lowest_pool_reading_occupied"] or 0
    print(f"  empty lane holds to a pool of x{offset + held:g}, "
          f"reads occupied from x{offset + tripped:g}, "
          f"at ambient level {headlight_result['ambient_level']:g} "
          f"({headlight_result['ambient_levels_swept']} level swept); "
          f"vehicles refused: {headlight_result['vehicle_refusals']}")

    print("measuring the ground the structural comparison cannot serve ...")
    floor_result = texture_floor()
    bottom = min(floor_result["matrix_ground_reference_texture"].values())
    print(f"  the matrix axis bottoms out at {bottom} grey levels, above the "
          f"{floor_result['min_reference_texture']} floor; a smooth floor measures "
          f"{floor_result['smooth_floor_reference_texture']} and is NOT MEASURED")

    print("measuring which conditions share the reference_not_recognised reason ...")
    conflation = conflated_reasons()
    shared = "ONE shared reason" if conflation["all_report_the_same_reason"] else "distinct"
    print(f"  {len(conflation['causes'])} unrelated conditions, {shared}")

    limits = limitations(
        matrix_result, weather_result, headlight_result, floor_result, conflation
    )
    print(f"  {len(limits)} measured limitations, each of which the seam must name: "
          + ", ".join(sorted(limit["seam_word"] for limit in limits)))

    print("measuring the control that makes each published boolean false ...")
    control_result = controls(detector)
    for key, control in sorted(control_result.items()):
        state = "false, as required" if control["value"] is False else "STILL TRUE"
        print(f"  {key}: {state}")

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
        "headlight": headlight_result,
        "texture_floor": floor_result,
        "conflated_reasons": conflation,
        "limitations": limits,
        # Every NOT MEASURED caveat in the documents is derived from this, so
        # that "no real footage has been through this gate" is a value rather
        # than a sentence somebody remembered to keep true.
        "scenes": {"real_frames_measured": REAL_FRAMES_MEASURED},
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
        "controls": control_result,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.update_docs:
        values = figures(evidence)
        rendered = blocks(evidence)
        for document in DOCUMENTS:
            path = ROOT / document
            before = path.read_text(encoding="utf-8")
            after = rewrite(before, values, rendered)
            if after != before:
                path.write_text(after, encoding="utf-8")
                print(f"updated the figures and generated sections in {document}")
            else:
                print(f"{document} already matches the measurement")
        missing = [key for key in BLOCKS if key not in rendered]
        if missing:
            print(f"WARNING: nothing rendered for {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
