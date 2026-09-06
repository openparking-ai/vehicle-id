#!/usr/bin/env python3
"""The harness. Every accuracy number Open Parking AI quotes comes from here.

    python scripts/eval_plates.py

Reports, side by side, the general-OCR baseline and our own recogniser, across
the whole degradation ladder, plus the per-execution-path timings and the
confidence calibration V2 needs.

Two rules this file exists to enforce (V3):

  * No accuracy claim exists outside this output. The knowhow repo quotes these
    numbers or none.
  * Numbers that cannot honestly be produced are printed as NOT MEASURABLE,
    with the reason. Full-identity accuracy is one of those today.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from vehicle_id.engine import weights_id
from vehicle_id.plates.dataset import EVAL_SEED
from vehicle_id.plates.generator import PlateGenerator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

LADDER = list(range(0, 10))

#: What this harness writes so the documents can cite it. `docs/measured/` is
#: the only place a published plate figure may come from; `models/` is
#: gitignored, so the operating-point sidecar cannot serve -- no file in the
#: repository recorded which checkpoint produced the published plate figures,
#: and this is that file.
PLATES_EVIDENCE = Path("docs/measured/plates.json")

#: The threshold the README contrasts the measured operating point against.
#: Named here so the document's "at a naive 0.85" and the measured 0.85 row are
#: one number rather than two that agree today.
NAIVE_THRESHOLD = 0.85


def normalise(text: str) -> str:
    """Compare registrations, not layout: case and gaps are not the answer."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def evaluate(reader, samples) -> dict:
    exact = 0
    char_err = 0
    char_total = 0
    confidences = []
    wrong_confident = 0
    for s in samples:
        got, conf = reader.read(s.image)
        want = normalise(s.text)
        got_n = normalise(got)
        confidences.append(conf)
        if got_n == want:
            exact += 1
        else:
            if conf >= 0.85:
                wrong_confident += 1
        char_total += len(want)
        char_err += _levenshtein(got_n, want)
    n = len(samples)
    return {
        "n": n,
        "exact": exact,
        "exact_pct": 100.0 * exact / n,
        "cer_pct": 100.0 * char_err / max(char_total, 1),
        "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
        "wrong_and_confident": wrong_confident,
    }


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def same_vehicle_spread(reader, sets) -> float:
    """How far apart two readings of the SAME plate get, at the 99.5th centile.

    This is the number the engine needs to tell "a degraded second look at the
    car in front of me" from "a second car in the frame". It is measured, not
    chosen: the same plate is read at every rung of the ladder and every pair of
    successful readings is compared, so the answer comes from this model's own
    behaviour on this generator rather than from somebody's intuition about how
    wrong OCR usually is.
    """
    distances = []
    per_plate: dict[str, list[str]] = {}
    for samples in sets.values():
        for sample in samples:
            got, _ = reader.read(sample.image)
            if got:
                per_plate.setdefault(normalise(sample.text), []).append(normalise(got))
    for readings in per_plate.values():
        for i, a in enumerate(readings):
            for b in readings[i + 1:]:
                distances.append(_levenshtein(a, b))
    if not distances:
        return 0.0
    distances.sort()
    index = min(len(distances) - 1, int(0.995 * len(distances)))
    return float(distances[index])


def noise_ceiling(reader, count: int = 200) -> float:
    """The highest confidence this model gives to an image with no plate in it.

    A competitor reading below this is indistinguishable from the engine
    reading shapes out of noise, and must not be allowed to send a good read to
    fallback.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(count):
        image = rng.integers(0, 255, (160, 320, 3), dtype=np.uint8)
        _, confidence = reader.read(image)
        best = max(best, confidence)
    return best


#: The thresholds the operating point is chosen from. A list, so the chosen
#: point is reproducible and comparable between runs -- and note that its TOP is
#: a real limit: a model needing more than 0.995 has no candidate here at all,
#: which is a refusal for a reason that has nothing to do with the model.
CANDIDATE_THRESHOLDS = (0.0, 0.50, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995)

#: A silent wrong answer bills a stranger's car to somebody else; a fallback
#: costs an operator a glance. Published, and the reason the chooser exists.
MAX_SILENT_WRONG_PCT = 1.0


class NoOperatingPoint(Exception):
    """No candidate threshold satisfies every constraint on these weights.

    Carries the numbers that conflict. Raised rather than returned as a None
    that a caller can forget to check, and rather than quietly writing the
    cheapest candidate that meets ONE of the three -- which is what this file
    used to do, and it wrote 0.995 for a model whose pristine plates answer only
    at 0.99 and whose noise ceiling sits at 0.9004.
    """

    def __init__(self, message: str, conflicts: dict) -> None:
        super().__init__(message)
        self.conflicts = conflicts


def choose_operating_point(
    rows,
    *,
    clean_plate_confidence: float,
    noise_confidence_ceiling: float,
    max_silent_wrong_pct: float = MAX_SILENT_WRONG_PCT,
) -> dict:
    """The operating point, or a refusal naming the numbers that conflict.

    `rows` is one `(threshold, silent_wrong_pct, fallback_pct)` per candidate,
    cheapest first. Three constraints, all three MEASURED for these weights, and
    each one is a published guarantee rather than a preference:

      * **silent-wrong under the bar.** The original, and until this round the
        only one. `silent_wrong_pct < max_silent_wrong_pct`.

      * **pristine plates are answered.** `threshold <= clean_plate_confidence`,
        the median confidence of correctly-read undegraded plates. A point above
        it sends the typical clean plate to fallback, which is the guarantee two
        of this suite's tests assert directly. The chooser had NO such
        constraint: silent-wrong falls monotonically as the threshold rises, so
        left alone the rule always prefers the strictest candidate, and the
        strictest candidate is the one most likely to refuse clean plates.

      * **the gate's noise measurement stays possible.**
        `threshold <= noise_confidence_ceiling`. The ceiling is the highest
        confidence this model gives an image with no plate in it. Below the
        threshold, no noise frame can clear the operating point, so the ungated
        control in `test_the_presence_gate_moves_the_noise_measurement` answers
        zero by construction and the gate's ACCURACY becomes unmeasurable. The
        same function measured this number eleven lines further down and did not
        consult it.

    Unsure is a first-class answer here, exactly as it is at the barrier. When
    the constraints cannot all be met, the correct output is a refusal that
    names the numbers -- not a threshold that satisfies whichever one was
    checked first.
    """
    qualifying = []
    for threshold, silent, fallback in rows:
        failed = []
        if not silent < max_silent_wrong_pct:
            failed.append("silent_wrong")
        if threshold > clean_plate_confidence:
            failed.append("clean_plate")
        if threshold > noise_confidence_ceiling:
            failed.append("noise_ceiling")
        if not failed:
            qualifying.append((threshold, silent, fallback))

    if qualifying:
        threshold, silent, fallback = qualifying[0]
        return {
            "threshold": threshold,
            "silent_wrong_pct": silent,
            "fallback_pct": fallback,
            "clean_plate_confidence": clean_plate_confidence,
            "noise_confidence_ceiling": noise_confidence_ceiling,
        }

    # The refusal, and it names every number a reader would need to check it.
    meeting_silent = [t for t, s, _ in rows if s < max_silent_wrong_pct]
    ceiling = min(clean_plate_confidence, noise_confidence_ceiling)
    binding = (
        "clean-plate confidence"
        if clean_plate_confidence <= noise_confidence_ceiling
        else "noise confidence ceiling"
    )
    lines = [
        "no candidate threshold satisfies every constraint on these weights:",
        f"    silent-wrong < {max_silent_wrong_pct:.2f}%  needs threshold >= "
        + (f"{meeting_silent[0]:.3f}" if meeting_silent else "MORE THAN ANY CANDIDATE"),
        f"    pristine plates answered  needs threshold <= {clean_plate_confidence:.4f}"
        "  (median confidence of correctly-read undegraded plates)",
        f"    gate noise measurable     needs threshold <= {noise_confidence_ceiling:.4f}"
        "  (measured noise confidence ceiling)",
    ]
    if meeting_silent:
        lines.append(
            f"    the binding pair: {meeting_silent[0]:.3f} against {ceiling:.4f} "
            f"({binding}). No threshold sits between them."
        )
    else:
        lines.append(
            f"    no candidate up to {max(t for t, _, _ in rows):.3f} reaches the "
            "silent-wrong bar at all; the candidate list itself is the limit."
        )
    raise NoOperatingPoint(
        "\n".join(lines),
        {
            "max_silent_wrong_pct": max_silent_wrong_pct,
            "lowest_threshold_meeting_silent_wrong": meeting_silent[0] if meeting_silent else None,
            "clean_plate_confidence": clean_plate_confidence,
            "noise_confidence_ceiling": noise_confidence_ceiling,
            "highest_candidate": max(t for t, _, _ in rows) if rows else None,
        },
    )


def median_clean_confidence(confidences: list[float]) -> float:
    """Median confidence over the pristine plates this model reads CORRECTLY.

    Correctly-read only: a wrong read's confidence says nothing about whether
    the operating point admits a good one. Median, because that is the statistic
    the suite's own `clean_confidence` fixture uses -- here on the ladder's rung
    0, which is larger and reproducible from `EVAL_SEED`.
    """
    if not confidences:
        return 0.0
    ordered = sorted(confidences)
    return ordered[len(ordered) // 2]


def timing(reader, samples, label: str) -> float:
    reader.read(samples[0].image)
    times = []
    for s in samples[:40]:
        t = time.perf_counter()
        reader.read(s.image)
        times.append((time.perf_counter() - t) * 1000)
    median = statistics.median(times)
    print(f"    {label:34} {median:6.2f} ms/plate  ({1000 / median:6.1f} plates/sec)")
    return median


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-rung", type=int, default=200)
    ap.add_argument("--weights", type=Path, default=Path("models/plate_crnn.pt"))
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument(
        "--write-operating-point",
        action="store_true",
        help="record the measured operating point beside the weights, so the "
             "engine can apply a number that was measured for THESE weights",
    )
    ap.add_argument(
        "--update-docs",
        action="store_true",
        help="write docs/measured/plates.json and rewrite the figures the "
             "documents cite from it. A published figure is PRODUCED by a "
             "command, not typed: README:196-197 published 0.87%% / 30.9%% "
             "against a measurement of 0.80%% / 30.6%% and nothing could see it.",
    )
    args = ap.parse_args()

    print("=" * 78)
    print(" Open Parking AI — plate recogniser evaluation")
    print(" reproduce with:  python scripts/eval_plates.py")
    print("=" * 78)

    # The eval set is re-derived from a seed, never stored, never committed.
    sets = {
        rung: PlateGenerator(seed=EVAL_SEED + rung).batch(args.per_rung, degradation=rung)
        for rung in LADDER
    }
    print(f"\n eval set: {args.per_rung} synthetic plates per rung, "
          f"{len(LADDER)} rungs, seed {EVAL_SEED} (reproducible, not stored)")

    readers = {}
    from vehicle_id.plates.recognizer import PlateRecognizer

    readers["ours (CRNN+CTC, trained on synthetic)"] = PlateRecognizer(args.weights)
    if not args.skip_baseline:
        from vehicle_id.baseline import RapidOcrBaseline

        readers["baseline (RapidOCR PP-OCRv3)"] = RapidOcrBaseline()

    results = {}
    for name, reader in readers.items():
        print(f"\n {name}")
        cols = ("rung", "exact %", "CER %", "mean conf", "wrong>=.85")
        print(f"   {cols[0]:>4}  {cols[1]:>8}  {cols[2]:>7}  {cols[3]:>9}  {cols[4]:>11}")
        rows = {}
        for rung in LADDER:
            r = evaluate(reader, sets[rung])
            rows[rung] = r
            print(f"   {rung:>4}  {r['exact_pct']:>8.1f}  {r['cer_pct']:>7.2f}  "
                  f"{r['mean_confidence']:>9.3f}  {r['wrong_and_confident']:>12d}")
        results[name] = rows

    print("\n timing, per execution path (D4: labelled, never assumed)")
    ours = readers["ours (CRNN+CTC, trained on synthetic)"]
    timing(ours, sets[0], "ours — CPU")
    if torch.backends.mps.is_available():
        timing(PlateRecognizer(args.weights, device="mps"), sets[0], "ours — MPS")
    if not args.skip_baseline:
        timing(readers["baseline (RapidOCR PP-OCRv3)"], sets[0], "baseline — CPU (onnxruntime)")
    print("    CoreML: measured SLOWER than CPU on this Mac (8-partition split); not used.")

    # V2 calibration: where does the threshold have to sit?
    print("\n confidence calibration (V2)")
    print("   A raw score is not a threshold. Our recogniser is ACCURATE and")
    print("   OVERCONFIDENT: mean confidence barely moves across the ladder while")
    print("   accuracy falls. So the operating point has to be measured, not chosen.")
    plate_evidence = None
    for name, reader in readers.items():
        print(f"\n   {name}")
        print(f"     {'threshold':>9}  {'answers':>8}  {'of those wrong':>15}  {'-> fallback':>11}")
        pairs = []
        clean_correct = []
        for rung in LADDER:
            for sample in sets[rung]:
                got, conf = reader.read(sample.image)
                ok = normalise(got) == normalise(sample.text)
                pairs.append((conf, ok))
                # Rung 0 is the pristine set. Kept here rather than re-read, so
                # the clean-plate constraint costs nothing and is measured on
                # exactly the samples the table above reports.
                if rung == 0 and ok:
                    clean_correct.append(conf)
        total = len(pairs)
        rows = []
        for threshold in CANDIDATE_THRESHOLDS:
            answered = [ok for conf, ok in pairs if conf >= threshold]
            wrong = sum(1 for ok in answered if not ok)
            silent = 100.0 * wrong / total
            fallback = 100.0 * (total - len(answered)) / total
            print(f"     {threshold:>9.3f}  {len(answered):>8}  {wrong:>7} ({silent:>5.2f}%)  "
                  f"{fallback:>10.1f}%")
            rows.append((threshold, silent, fallback))

        # The two constraints the chooser used to ignore, MEASURED for these
        # weights before it runs rather than eleven lines after it.
        clean = median_clean_confidence(clean_correct)
        ceiling = noise_ceiling(reader)
        print(f"     clean-plate confidence (rung 0, correct reads, median): {clean:.4f}")
        print(f"     noise confidence ceiling: {ceiling:.4f}")

        try:
            point = choose_operating_point(
                rows, clean_plate_confidence=clean, noise_confidence_ceiling=ceiling
            )
        except NoOperatingPoint as refusal:
            point = None
            print(f"     -> REFUSED: {refusal}")
        else:
            print(f"     -> operating point {point['threshold']:.3f}: "
                  f"silent-wrong {point['silent_wrong_pct']:.2f}%, "
                  f"fallback {point['fallback_pct']:.1f}%")

        if name.startswith("ours"):
            plate_evidence = {
                "weights_id": weights_id(args.weights),
                "per_rung": args.per_rung,
                "rungs": len(LADDER),
                "eval_seed": EVAL_SEED,
                "max_silent_wrong_pct": MAX_SILENT_WRONG_PCT,
                "clean_plate_confidence": clean,
                "noise_confidence_ceiling": ceiling,
                "candidates": [
                    {"threshold": t, "silent_wrong_pct": s, "fallback_pct": f}
                    for t, s, f in rows
                ],
                "operating_point": point,
                "naive_threshold": NAIVE_THRESHOLD,
            }

        if args.write_operating_point and name.startswith("ours"):
            if point is None:
                # Refusing to write one is the honest outcome: these weights have
                # no operating point that meets every bar, and the engine must
                # not be handed a number that pretends otherwise. The engine then
                # refuses to start, which is the correct end of this path -- a
                # model with no admissible operating point is not one to run a
                # barrier on.
                print("     -> NOT written: no threshold on this ladder qualifies.")
            else:
                from vehicle_id.engine import write_operating_point

                spread = same_vehicle_spread(reader, sets)
                print(f"     -> same-vehicle reading spread (p99.5): {spread:.0f} characters")
                written = write_operating_point(
                    args.weights,
                    point["threshold"],
                    {
                        "silent_wrong_pct": point["silent_wrong_pct"],
                        "fallback_pct": point["fallback_pct"],
                        "clean_plate_confidence": clean,
                        "max_silent_wrong_pct": MAX_SILENT_WRONG_PCT,
                        "per_rung": args.per_rung,
                        "rungs": len(LADDER),
                        "eval_seed": EVAL_SEED,
                    },
                    same_vehicle_spread=spread,
                    noise_confidence_ceiling=ceiling,
                )
                print(f"     -> wrote {written}")

    print("\n NOT MEASURABLE, and why (V3 requires saying so rather than omitting it)")
    print("   full-identity accuracy   : requires bench ground truth pairing plate with")
    print("                              make/model/colour/appearance. No adequately")
    print("                              licensed public set exists (see docs/EVAL_DATA.md).")
    print("   real-plate accuracy      : requires the physical bench. Synthetic fonts are")
    print("                              not embossing typefaces; this number is NOT a")
    print("                              prediction of real-world accuracy.")
    print("   false-match rate         : belongs with fusion (V2), after re-ID lands.")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\n wrote {args.json_out}")

    if args.update_docs:
        # `plate_evidence` is bound by the "ours" arm above, which is always in
        # `readers`. Asserted rather than assumed, because a silent `None` here
        # would write an empty evidence file over a good one.
        assert plate_evidence is not None, "no measurement for our own recogniser"
        PLATES_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        PLATES_EVIDENCE.write_text(
            json.dumps(plate_evidence, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n wrote {PLATES_EVIDENCE}")

        from measured_figures import DOCUMENTS, blocks, figures, load_evidence, rewrite

        evidence = load_evidence(ROOT)
        values, rendered = figures(evidence), blocks(evidence)
        for document in DOCUMENTS:
            path = ROOT / document
            before = path.read_text(encoding="utf-8")
            after = rewrite(before, values, rendered)
            if after != before:
                path.write_text(after, encoding="utf-8")
                print(f" updated the cited figures in {document}")
            else:
                print(f" {document} already matches the measurement")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
