#!/usr/bin/env python3
"""Does a car photographed twice match ITSELF, and not match other cars?

    python scripts/eval_fingerprint.py --photos DIR --index FILE --out FILE
    python scripts/eval_fingerprint.py --write-synthetic DIR [--pairs 8]

That is the whole question, and this harness is the whole answer to it. Nothing
downstream of a match -- a schema, a three-way identity, a live exit lane -- is
worth building until the number this produces exists.

The input is PAIRS: the same vehicle photographed twice from the same position,
minutes to hours apart. Same position both times, because a different angle
measures a harder problem than the product has and produces a pessimistic number
that means nothing. Eighteen photographs of eighteen DIFFERENT cars, which is
what the previous round collected, cannot measure matching at all.

  * **No real data reaches a repository.** The photographs and the index live
    outside every git work tree and this refuses to run if any path it is handed
    resolves inside one. What it writes is counts, and the writer REFUSES to
    write any string the OPERATOR used to label a photograph -- a pair id, a file
    name -- so a photograph cannot be identified from the evidence file however
    it was named. Path-shaped strings are refused separately, by shape, because
    nothing this harness publishes is a location. Directory names are NOT
    treated as labels; see `input_tokens` for why that was tried and reverted.

  * **THREE outcomes per arm, never two.** A frame with too little texture
    yields no keypoints and there is nothing to match. "Nothing matched" and
    "there was nothing to match" are different answers: collapsing them counts
    an unmeasurable same-car pair as a MISS and an unmeasurable different-car
    pair as a correct reject, which flatters the exact number this exists to
    produce. Every count is stated over the measurable subset with the
    unmeasurable count beside it.

  * **Distances first, counts second.** The five terms sit on five scales, one
    of them unbounded, and one of them is a similarity that has been converted.
    A single threshold cannot serve them and three chosen thresholds would make
    the comparison an artefact of the choosing. So the PRIMARY result is the
    distance distributions and their separation, threshold-free: the AUC is the
    probability that a same-car distance is below a different-car one, and it
    needs no operating point to exist.

  * **The threshold is not fitted on the pairs the counts are reported over.**
    The pairs are split by id into a FIT half and a REPORT half. The operating
    point is chosen on the first and the count table is produced on the second.
    Comparisons that straddle the halves are used in neither, and how many were
    dropped is published.

And one number that decides a later round on its own: the TIE RATE. For every
same-car pair, how many OTHER images in the set fall within the same distance --
how often the answer is "two or more candidates are equally close" rather than
"one is clearly right". Whether a tie-breaking stage is needed at all is
measured here and nowhere else. It is reported as a count against the set size,
never as a rate for a garage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from outside_repositories import (  # noqa: E402
    PathInsideRepository,  # noqa: F401 - re-exported for the tests
    refuse_repository_paths,
)

from vehicle_id.fingerprint import (  # noqa: E402
    DESCRIPTOR_VERSION,
    KINDS,
    ORB,
    TERM_NAMES,
    TERMS,
    NoOperatingPoint,
    choose_operating_point,
    compare,
    compute,
    rates_at,
)

#: The two classes of comparison, named once so the output and the code cannot
#: drift. A comparison is between two IMAGES: same-car when both come from one
#: pair, different-car when they come from two.
SAME_CAR = "same_car"
DIFFERENT_CAR = "different_car"
CLASSES = (SAME_CAR, DIFFERENT_CAR)

#: The three outcomes. UNMEASURABLE is not a kind of failure to match; it is the
#: absence of a measurement, and it is counted in its own cell everywhere.
MATCHED = "matched"
NOT_MATCHED = "not_matched"
UNMEASURABLE = "unmeasurable"
OUTCOMES = (MATCHED, NOT_MATCHED, UNMEASURABLE)

#: The halves the pairs are split into. Deterministic, by sorted position, so a
#: re-run on the same index produces the same split and the same threshold.
FIT = "fit"
REPORT = "report"

#: The confidence level every published bound is stated at.
BOUND_ALPHA = 0.05


# --- the guard: nothing from the input may reach the output ---------------


class InputLeakedIntoOutput(RuntimeError):
    """Raised rather than writing a photograph's name into a file."""


#: A separator with a non-space character against it. `a/b` and `/tmp/x` match;
#: the ratio `matches / comparisons` in a published scale does not.
_PATH_SHAPED = re.compile(r"\S[/\\]|[/\\]\S")


def path_shaped(value: str) -> bool:
    return _PATH_SHAPED.search(value) is not None


def input_tokens(index: dict) -> set[str]:
    """Every string the OPERATOR used to label a photograph.

    Pair ids, and file names with and without their extension. DERIVED from the
    index that was actually read rather than from a pattern, because an operator
    naming a pair after a registration is exactly the case a pattern would be
    written too late for.

    The photograph DIRECTORY's path components are deliberately not in here, and
    that is a correction rather than an omission. Including them was tried and
    turned the writer red on its own first real run: a directory called `pairs`
    made every key containing the word "pairs" a leak, and a two-character path
    component matched inside the word "acquires". A guard that has to be
    loosened by hand the first time it fires is not a guard. What it was
    protecting against is covered anyway -- nothing here copies a path into the
    object, and `path_shaped` refuses one that ever arrives.

    False positives are the SAFE direction and are left in: an operator who
    names a pair `auc` will get a refusal rather than a file, and a refusal is
    diagnosable.
    """
    tokens: set[str] = set()
    for pair_id, pair in (index.get("pairs") or {}).items():
        tokens.add(str(pair_id))
        for side in ("a", "b"):
            name = (pair.get(side) or {}).get("file")
            if name:
                tokens.add(str(name))
                tokens.add(Path(str(name)).stem)
    return {t for t in tokens if len(t) >= 2}


def leaks(node, tokens: set[str], path: str = "$") -> list[str]:
    """Every string in the object that came from outside the harness.

    Recursive, and keys are checked as well as values -- a key is as published
    as a value, and the per-arm blocks are sub-objects. Two rules:

      * a PATH-SHAPED string is refused outright. Nothing this harness
        publishes is a location. Path-shaped means a `/` or `\\` with a
        non-space character against it -- not any slash at all, because the
        scale of one of the published terms is a ratio and reads
        `matches / comparisons`. A rule that refused that would have to be
        loosened by hand the first time it fired, and a guard loosened under
        pressure is the one that is not there when it matters.
      * a string containing any input token, case-insensitively, is refused.
        Containment rather than equality, because "p07.png", "p07" and
        "photo of p07" are the same leak.

    Derived from the input that was actually read rather than from a pattern for
    a registration or a filename. A pattern has to be written before the thing it
    catches exists; this cannot be, because the tokens ARE the operator's own
    names for the photographs.
    """
    lowered = {t.lower() for t in tokens if t}
    found: list[str] = []

    def check(where: str, value: str) -> None:
        if path_shaped(value):
            found.append(f"{where}={value!r} is path-shaped")
            return
        low = value.lower()
        for token in lowered:
            if token in low:
                found.append(f"{where}={value!r} contains the input token {token!r}")
                return

    def walk(item, where: str) -> None:
        if isinstance(item, str):
            check(where, item)
        elif isinstance(item, dict):
            for key, value in item.items():
                if isinstance(key, str):
                    check(f"{where}.<key>", key)
                walk(value, f"{where}.{key}")
        elif isinstance(item, (list, tuple)):
            for i, entry in enumerate(item):
                walk(entry, f"{where}[{i}]")

    walk(node, path)
    return found


def write_output(obj: dict, path: Path, tokens: set[str]) -> Path:
    """Write the evidence object, or refuse to.

    The refusal is the point, and it is the half of the no-real-data rule that
    lives outside the repository: `check-no-real-data.js` scans TRACKED files
    and can never see this one.
    """
    offenders = leaks(obj, tokens)
    if offenders:
        raise InputLeakedIntoOutput(
            f"refusing to write: the object carries {len(offenders)} string(s) "
            "sourced from the input at " + "; ".join(offenders)
        )
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return path


# --- arithmetic ----------------------------------------------------------


def quantile(values: list[float], q: float) -> float | None:
    """Linear interpolation between order statistics. None on an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def auc(same: list[float], different: list[float]) -> float | None:
    """P(a same-car distance < a different-car one), ties counted as a half.

    The threshold-free separation, and the PRIMARY result of this harness. 1.0
    means every same-car comparison is closer than every different-car one; 0.5
    means the descriptor carries no information about identity at all. It exists
    without an operating point, which is what makes it the number worth
    reporting when there are twenty pairs to compute it from.
    """
    if not same or not different:
        return None
    ordered = sorted(different)
    wins = 0.0
    for value in same:
        # The different-car distances this same-car one beats are the ones ABOVE
        # it, not below it. Written the other way round first, and the synthetic
        # run caught it immediately: a perfectly separating arm reported 0.0000.
        # A 0.5 and a 0.0 are both "wrong" in a way a reader might rationalise;
        # 0.0 on synthetic data built to separate is not.
        at_or_below = _bisect_right(ordered, value)
        equal = at_or_below - _bisect_left(ordered, value)
        wins += (len(ordered) - at_or_below) + 0.5 * equal
    return wins / (len(same) * len(different))


def _bisect_left(ordered: list[float], value: float) -> int:
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if ordered[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_right(ordered: list[float], value: float) -> int:
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if ordered[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def overlap(same: list[float], different: list[float]) -> float | None:
    """The fraction of different-car distances at or below the same-car maximum.

    A companion to the AUC that says something the AUC cannot: whether the two
    distributions touch AT ALL. An AUC of 0.999 with an overlap of 0.02 is a
    descriptor with a clean threshold and a few bad pairs; the same AUC with an
    overlap of 0.4 is one where no threshold separates the tails.
    """
    if not same or not different:
        return None
    ceiling = max(same)
    return sum(1 for d in different if d <= ceiling) / len(different)


def binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, alpha: float = BOUND_ALPHA) -> float | None:
    """The 95% one-sided upper bound on a rate, given `k` events in `n` trials.

    EXACT, by bisection on the binomial CDF, rather than the rule of three --
    which is an approximation valid only at k=0 and only for large n, and n here
    is small on purpose. This is the number that stops a table of zeros being
    read as a guarantee: zero misses in twenty same-car comparisons bounds the
    miss rate at about 15%, not at 0.
    """
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def distribution(values: list[float], unmeasurable: int) -> dict:
    """One class's distances on one arm, and how many could not be measured.

    The unmeasurable count sits INSIDE the block rather than beside it, so a
    reader cannot quote the quantiles without also having been handed the number
    of comparisons they were not computed over.
    """
    return {
        "measurable": len(values),
        "unmeasurable": unmeasurable,
        "min": min(values) if values else None,
        "p05": quantile(values, 0.05),
        "median": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


# --- the run -------------------------------------------------------------


def split_of(pair_ids: list[str]) -> dict[str, str]:
    """Which half each pair belongs to. Deterministic and alternating.

    Alternating over the SORTED ids rather than random, so the split is a
    property of the index and not of when the harness was run: two runs of the
    same photographs choose the same threshold on the same pairs, and a receipt
    can be checked.
    """
    return {pid: (FIT if i % 2 == 0 else REPORT) for i, pid in enumerate(sorted(pair_ids))}


def load_images(photos: Path, index: dict) -> tuple[dict[str, dict[str, np.ndarray]], int, list]:
    """Every pair's two images, cropped if the index says so.

    Returns the images keyed by pair id and side, how many pairs were excluded,
    and the reasons -- which are the operator's words and therefore never leave
    this function.
    """
    images: dict[str, dict[str, np.ndarray]] = {}
    excluded = 0
    reasons = []
    for pair_id, pair in sorted((index.get("pairs") or {}).items()):
        if pair.get("excluded"):
            excluded += 1
            reasons.append((pair_id, pair["excluded"]))
            continue
        sides = {}
        for side in ("a", "b"):
            entry = pair[side]
            image = cv2.imread(str(Path(photos) / entry["file"]))
            if image is None:
                raise RuntimeError(f"{pair_id}/{side}: the photograph could not be read")
            rect = entry.get("rect")
            if rect:
                x, y, w, h = rect
                image = image[y:y + h, x:x + w]
            sides[side] = image
        images[pair_id] = sides
    return images, excluded, reasons


def describe_all(images: dict, kind: str) -> dict[tuple[str, str], object]:
    return {
        (pair_id, side): compute(image, kind)
        for pair_id, sides in images.items()
        for side, image in sides.items()
    }


def all_comparisons(descriptors: dict) -> list[tuple[tuple, tuple, str]]:
    """Every unordered pair of images, labelled same-car or different-car.

    With twenty pairs this is 780 comparisons: 40 images choose 2. Twenty of
    them are same-car and 760 are different-car, and that asymmetry is why the
    two bounds this harness publishes are so different from each other.
    """
    keys = sorted(descriptors)
    out = []
    for left, right in combinations(keys, 2):
        klass = SAME_CAR if left[0] == right[0] else DIFFERENT_CAR
        out.append((left, right, klass))
    return out


def measure(descriptors: dict, comparisons: list) -> dict:
    """Every comparison, every term. `{(left, right): Comparison}`."""
    return {
        (left, right): compare(descriptors[left], descriptors[right])
        for left, right, _ in comparisons
    }


def ties_for(term: str, descriptors: dict, measured: dict, pair_ids: list[str]) -> dict:
    """A5.1: how often two or more candidates are equally close.

    For each same-car pair, the distance between its own two images is the
    answer the system would have to find; every OTHER image in the set is a
    candidate that could be at least as close. This counts them.

    A pair whose own comparison is UNMEASURABLE on this term is excluded and
    counted separately -- there is no distance for a competitor to be nearer
    than. So is a candidate the term could not measure against, and that count
    is published too, because a tie count computed over a shrinking candidate
    pool is not comparable between arms.

    Reported as counts against the pool size. It is not a rate for a garage: a
    tie rate over forty photographs says nothing about a car park with four
    hundred cars in it, and stating it as a percentage would invite exactly that
    reading.
    """
    def distance(left, right):
        key = (left, right) if (left, right) in measured else (right, left)
        return measured[key][term]

    histogram: dict[int, int] = {}
    excluded = 0
    unmeasurable_candidates = 0
    pool = 0
    for pair_id in pair_ids:
        own = distance((pair_id, "a"), (pair_id, "b"))
        if not own.measurable:
            excluded += 1
            continue
        ties = 0
        candidates = 0
        for other_id, side in sorted(descriptors):
            if other_id == pair_id:
                continue
            candidates += 1
            competitor = distance((pair_id, "a"), (other_id, side))
            if not competitor.measurable:
                unmeasurable_candidates += 1
                continue
            if competitor.value <= own.value:
                ties += 1
        pool = max(pool, candidates)
        histogram[ties] = histogram.get(ties, 0) + 1
    measured_pairs = sum(histogram.values())
    return {
        "same_car_pairs_measured": measured_pairs,
        "same_car_pairs_excluded_unmeasurable": excluded,
        "candidate_pool_per_pair": pool,
        "unmeasurable_candidate_comparisons": unmeasurable_candidates,
        "pairs_with_no_closer_or_equal_candidate": histogram.get(0, 0),
        "pairs_with_at_least_one": measured_pairs - histogram.get(0, 0),
        "histogram": {str(k): v for k, v in sorted(histogram.items())},
    }


def arm(
    term: str,
    measured: dict,
    comparisons: list,
    split: dict[str, str],
    descriptors: dict,
    pair_ids: list[str],
    *,
    max_false_match_rate: float,
    max_miss_rate: float,
) -> dict:
    """One term's whole answer: distributions, separation, counts, bounds, ties."""
    spec = next(t for t in TERMS if t.name == term)

    values: dict[str, list[float]] = {SAME_CAR: [], DIFFERENT_CAR: []}
    unmeasurable: dict[str, int] = {SAME_CAR: 0, DIFFERENT_CAR: 0}
    halves: dict[str, dict[str, list[float]]] = {
        FIT: {SAME_CAR: [], DIFFERENT_CAR: []},
        REPORT: {SAME_CAR: [], DIFFERENT_CAR: []},
    }
    half_unmeasurable: dict[str, dict[str, int]] = {
        FIT: {SAME_CAR: 0, DIFFERENT_CAR: 0},
        REPORT: {SAME_CAR: 0, DIFFERENT_CAR: 0},
    }
    straddling = 0

    for left, right, klass in comparisons:
        distance = measured[(left, right)][term]
        if distance.measurable:
            values[klass].append(distance.value)
        else:
            unmeasurable[klass] += 1
        left_half, right_half = split[left[0]], split[right[0]]
        if left_half != right_half:
            straddling += 1
            continue
        if distance.measurable:
            halves[left_half][klass].append(distance.value)
        else:
            half_unmeasurable[left_half][klass] += 1

    out = {
        "scale": {
            "units": spec.units,
            "lower_bound": spec.lower_bound,
            "upper_bound": spec.upper_bound,
            "lower_is_closer": True,
            "note": spec.note,
        },
        "distances": {
            klass: distribution(values[klass], unmeasurable[klass]) for klass in CLASSES
        },
        "separation": {
            "auc": auc(values[SAME_CAR], values[DIFFERENT_CAR]),
            "overlap": overlap(values[SAME_CAR], values[DIFFERENT_CAR]),
            "over": {klass: len(values[klass]) for klass in CLASSES},
            "note": (
                "Threshold-free, and the primary result. The AUC is the "
                "probability that a same-car distance is below a different-car "
                "one; 0.5 is no information at all. The overlap is the fraction "
                "of different-car distances at or below the largest same-car one."
            ),
        },
        "ties": ties_for(term, descriptors, measured, pair_ids),
    }

    fit, report = halves[FIT], halves[REPORT]
    try:
        chosen = choose_operating_point(
            fit[SAME_CAR],
            fit[DIFFERENT_CAR],
            max_false_match_rate=max_false_match_rate,
            max_miss_rate=max_miss_rate,
        )
    except NoOperatingPoint as refusal:
        out["operating_point"] = {
            "refused": True,
            "why": str(refusal),
            "fitted_on": {
                klass: {
                    "measurable": len(fit[klass]),
                    "unmeasurable": half_unmeasurable[FIT][klass],
                }
                for klass in CLASSES
            },
        }
        out["counts"] = {
            "available": False,
            "why": "no operating point was chosen, so there is nothing to count at",
        }
        out["straddling_comparisons_used_in_neither_half"] = straddling
        return out

    threshold = chosen["threshold"]
    miss_rate, false_match_rate = rates_at(threshold, report[SAME_CAR], report[DIFFERENT_CAR])
    misses = round(miss_rate * len(report[SAME_CAR]))
    false_matches = round(false_match_rate * len(report[DIFFERENT_CAR]))
    out["operating_point"] = {
        "refused": False,
        "threshold": threshold,
        "fitted_on_half": FIT,
        "constraints": {
            "max_false_match_rate": max_false_match_rate,
            "max_miss_rate": max_miss_rate,
        },
        "fit_miss_rate": chosen["miss_rate"],
        "fit_false_match_rate": chosen["false_match_rate"],
        "fitted_on": {
            klass: {
                "measurable": len(fit[klass]),
                "unmeasurable": half_unmeasurable[FIT][klass],
            }
            for klass in CLASSES
        },
    }
    out["counts"] = {
        "available": True,
        "reported_on_half": REPORT,
        SAME_CAR: {
            MATCHED: len(report[SAME_CAR]) - misses,
            NOT_MATCHED: misses,
            UNMEASURABLE: half_unmeasurable[REPORT][SAME_CAR],
        },
        DIFFERENT_CAR: {
            MATCHED: false_matches,
            NOT_MATCHED: len(report[DIFFERENT_CAR]) - false_matches,
            UNMEASURABLE: half_unmeasurable[REPORT][DIFFERENT_CAR],
        },
        "note": (
            "Secondary. Counted at one operating point, chosen on the OTHER "
            "half of the pairs, over the measurable subset only."
        ),
    }
    out["bounds_95"] = {
        "miss_rate": {
            # NOT MEASURED, not zero. `rates_at` answers 0.0 for an empty class
            # because a rate over nothing has no other arithmetic to give, and
            # that 0.0 used to be published here beside `over: 0` -- the best
            # possible value, manufactured, in the field a reader quotes. Same
            # shape as `distribution` above and as `TermDistance`: the number is
            # null exactly when there was nothing to compute it over.
            "observed": miss_rate if report[SAME_CAR] else None,
            "upper": clopper_pearson_upper(misses, len(report[SAME_CAR])),
            "over": len(report[SAME_CAR]),
        },
        "false_match_rate": {
            "observed": false_match_rate if report[DIFFERENT_CAR] else None,
            "upper": clopper_pearson_upper(false_matches, len(report[DIFFERENT_CAR])),
            "over": len(report[DIFFERENT_CAR]),
        },
        "note": (
            "Clopper-Pearson, one-sided, 95%. Zero events does not mean a rate "
            "of zero: it means the rate is no worse than the bound, and with "
            "this many comparisons the bound is what the decision has to be "
            "made on. An `observed` of null means the report half held nothing "
            "to measure -- `over` is 0 and no rate was computed. It is NOT a "
            "measured rate of zero, and neither is the bound beside it, which "
            "is null for the same reason."
        ),
    }
    out["straddling_comparisons_used_in_neither_half"] = straddling
    return out


def build_output(
    *,
    arms: dict,
    kind: str,
    n_pairs: int,
    n_images: int,
    excluded_pairs: int,
    comparison_counts: dict,
    split_counts: dict,
    synthetic: bool,
    descriptor_version: int,
    cv2_version: str,
    python_version: str,
    package_version: str,
    harness_digest: str,
    guard_digest: str,
    timestamp: str,
) -> dict:
    """The evidence object. A pure function: values in, object out.

    The real run and the tests both call this, so the object's SHAPE cannot
    drift between what is measured and what is checked.
    """
    return {
        "question": (
            "does a vehicle photographed twice at a fixed position match itself, "
            "and not match other vehicles"
        ),
        "synthetic": synthetic,
        "is_a_measurement_of_real_vehicles": not synthetic,
        "descriptor_kind": kind,
        "descriptor_version": descriptor_version,
        "pairs": n_pairs,
        "images": n_images,
        "excluded_pairs": excluded_pairs,
        "comparisons": comparison_counts,
        "split": split_counts,
        "arms": arms,
        "bound_alpha": BOUND_ALPHA,
        "cv2": cv2_version,
        "python": python_version,
        "package_version": package_version,
        "harness_sha256": harness_digest,
        "guard_sha256": guard_digest,
        "measured_at": timestamp,
        "reading_this": (
            "The separation block is the answer. The counts are one operating "
            "point's view of it and the bounds beside them are what the counts "
            "can actually support. An arm whose operating point was REFUSED has "
            "no counts, and that refusal is a result, not an error."
        ),
    }


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- synthetic pairs, so the harness is provable before the photographs ----


def synthetic_vehicle(seed: int, flat: bool = False) -> np.ndarray:
    """One invented vehicle. Not a photograph and not a measurement.

    Deterministic in `seed`, so the pair set a test builds and the pair set CI
    builds are the same one. `flat` makes a low-texture body -- a plain van
    filling the crop, a night arrival -- which is what forces the UNMEASURABLE
    outcome to be exercised rather than assumed reachable.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=3)
    image = np.full((240, 360, 3), base, np.uint8)
    if flat:
        return image
    for _ in range(28):
        x, y = int(rng.integers(0, 320)), int(rng.integers(0, 200))
        w, h = int(rng.integers(12, 60)), int(rng.integers(8, 40))
        colour = [int(c) for c in rng.integers(0, 256, size=3)]
        if rng.random() < 0.5:
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, -1)
        else:
            cv2.circle(image, (x + w // 2, y + h // 2), max(3, h // 2), colour, -1)
    return image


def second_view(image: np.ndarray, seed: int) -> np.ndarray:
    """The same vehicle, photographed again: shifted, relit, and a little noisier."""
    rng = np.random.default_rng(seed + 10_000)
    dx, dy = int(rng.integers(-6, 7)), int(rng.integers(-4, 5))
    shifted = cv2.warpAffine(
        image, np.float32([[1, 0, dx], [0, 1, dy]]), (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )
    relit = cv2.convertScaleAbs(shifted, alpha=float(rng.uniform(0.9, 1.1)),
                                beta=float(rng.integers(-12, 13)))
    noise = rng.normal(0, 3, relit.shape)
    return np.clip(relit.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def write_synthetic(directory: Path, pairs: int, flat: int) -> Path:
    """A synthetic pair set on disk, plus the index that points at it.

    The index carries `"synthetic": true`, the evidence object repeats it, and
    every number a synthetic run produces is invented. It exists so that the
    harness, the three outcomes, the refusal and the writer are all proven to
    RUN before a single real photograph exists -- and so that nothing measured
    this way can be mistaken later for the answer.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index: dict = {"synthetic": True, "pairs": {}}
    for i in range(pairs):
        pid = f"s{i:03d}"
        vehicle = synthetic_vehicle(seed=1000 + i, flat=i < flat)
        first, second = vehicle, second_view(vehicle, seed=1000 + i)
        names = {}
        for side, image in (("a", first), ("b", second)):
            name = f"{pid}{side}.png"
            cv2.imwrite(str(directory / name), image)
            names[side] = {"file": name}
        index["pairs"][pid] = names
    path = directory / "index.json"
    path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return path


# --- main ----------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos", type=Path)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--kind", choices=KINDS, default=ORB)
    #: Both constraints are REQUIRED to have a value and neither has a product
    #: default. The number a barrier would open on is not a default in a
    #: harness; these are the constraints this RUN was asked to satisfy, and the
    #: refusal names them when nothing does.
    ap.add_argument("--max-false-match-rate", type=float, default=0.01)
    ap.add_argument("--max-miss-rate", type=float, default=0.20)
    ap.add_argument("--write-synthetic", type=Path)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--flat", type=int, default=1)
    args = ap.parse_args(argv)

    if args.write_synthetic is not None:
        refuse_repository_paths(write_synthetic=args.write_synthetic)
        path = write_synthetic(args.write_synthetic, args.pairs, args.flat)
        print(f"  wrote a SYNTHETIC pair set ({args.pairs} pairs, {args.flat} flat) at {path}")
        print("  its numbers are invented and are not a measurement of anything")
        return 0

    missing = [n for n in ("photos", "index", "out") if getattr(args, n) is None]
    if missing:
        ap.error("--" + ", --".join(missing) + " are required unless --write-synthetic is given")

    refuse_repository_paths(photos=args.photos, index=args.index, out=args.out)
    index = json.loads(args.index.read_text(encoding="utf-8"))
    tokens = input_tokens(index)

    images, excluded, reasons = load_images(args.photos, index)
    for pair_id, why in reasons:
        # Terminal only. Both halves of this line came from the operator.
        print(f"  {pair_id}: EXCLUDED ({why})")
    pair_ids = sorted(images)
    if len(pair_ids) < 2:
        raise SystemExit("at least two pairs are needed; one pair has no different-car comparison")

    descriptors = describe_all(images, args.kind)
    comparisons = all_comparisons(descriptors)
    measured = measure(descriptors, comparisons)
    split = split_of(pair_ids)

    counts = {klass: sum(1 for *_, k in comparisons if k == klass) for klass in CLASSES}
    counts["total"] = len(comparisons)

    arms = {
        term: arm(
            term, measured, comparisons, split, descriptors, pair_ids,
            max_false_match_rate=args.max_false_match_rate,
            max_miss_rate=args.max_miss_rate,
        )
        for term in TERM_NAMES
    }

    fit_pairs = [p for p in pair_ids if split[p] == FIT]
    report_pairs = [p for p in pair_ids if split[p] == REPORT]
    assert not (set(fit_pairs) & set(report_pairs)), "the halves are not disjoint"

    obj = build_output(
        arms=arms,
        kind=args.kind,
        n_pairs=len(pair_ids),
        n_images=len(descriptors),
        excluded_pairs=excluded,
        comparison_counts=counts,
        split_counts={
            "fit_pairs": len(fit_pairs),
            "report_pairs": len(report_pairs),
            "disjoint": True,
            "note": (
                "The operating point is fitted on the fit half and the counts "
                "are reported on the report half. Comparisons whose two images "
                "come from different halves are used in neither."
            ),
        },
        synthetic=bool(index.get("synthetic")),
        descriptor_version=DESCRIPTOR_VERSION,
        cv2_version=cv2.__version__,
        python_version=sys.version.split()[0],
        package_version=_package_version(),
        harness_digest=digest(Path(__file__)),
        guard_digest=digest(SCRIPTS / "outside_repositories.py"),
        timestamp=datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
    )
    write_output(obj, args.out, tokens)

    print(f"  kind={args.kind}  pairs={len(pair_ids)}  images={len(descriptors)}  "
          f"comparisons={counts['total']} ({counts[SAME_CAR]} same, "
          f"{counts[DIFFERENT_CAR]} different)")
    for term in TERM_NAMES:
        block = arms[term]
        sep = block["separation"]
        same = block["distances"][SAME_CAR]
        different = block["distances"][DIFFERENT_CAR]
        print(f"  {term:<22} auc={_num(sep['auc'])} overlap={_num(sep['overlap'])}  "
              f"same n={same['measurable']}/-{same['unmeasurable']} med={_num(same['median'])}  "
              f"diff n={different['measurable']}/-{different['unmeasurable']} "
              f"med={_num(different['median'])}")
        if block["operating_point"].get("refused"):
            print("      operating point: REFUSED")
    if obj["synthetic"]:
        print("\n  SYNTHETIC. These numbers are invented and measure nothing.")
    print(f"  wrote {args.out}")
    return 0


def _num(value) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("openparking-vehicle-id")
    except PackageNotFoundError:
        return "0.0.0+source"


if __name__ == "__main__":
    raise SystemExit(main())
