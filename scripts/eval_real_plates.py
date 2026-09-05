#!/usr/bin/env python3
"""Read PHOTOGRAPHED plates through the engine, and count what came back.

    python scripts/eval_real_plates.py \
        --photos DIR --labels FILE --weights FILE --out FILE [--as-is | --letterbox]

Every accuracy figure this project publishes was measured on plates the
generator drew. This harness measures the other thing: real photographs of real
cars, cropped to the plate by hand, read through the same path the service uses.
It produces a COUNT on a small set, never a percentage and never a published
figure -- what it writes is an evidence object, outside every repository.

Three rules this file exists to enforce.

  * **No real data reaches a repository.** The photographs, the crops and the
    registrations live outside every git work tree, and this refuses to run if
    any path it is handed resolves inside one. What it writes is counts; the
    writer REFUSES any string shaped like a registration, anywhere in the
    object, and that refusal is proven able to fire before this ever runs for
    real.
  * **The crop's aspect ratio is a measurement condition, not an incidental.**
    `to_tensor` resizes to a fixed IMG_W x IMG_H with no aspect preservation,
    and a generated plate is PLATE_W x PLATE_H while a photographed one is
    whatever the camera saw. So both conditions are measured and neither is
    "the" number: `--as-is` is what the service does today, `--letterbox` pads
    to the training aspect. The pair is what separates "fails on fonts" from
    "fails on shape".
  * **The threshold belongs to the weights.** ANSWER and FALLBACK are counted
    separately from right and wrong, because a read can be exactly right and
    still fall back under the operating point measured for these weights --
    which is expected to be the common case here. Collapsing the two is the
    cell that would eat the answer.

No plate, no filename, no path and no place appears in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vehicle_id.contract import ANSWER, Capture
from vehicle_id.engine import PlateEngine, normalise
from vehicle_id.plates.generator import PLATE_H, PLATE_W

#: The alphanumeric lengths of the registration patterns the layout uses,
#: established for this round and recorded in the receipt. The refusal window
#: below is DERIVED from these -- it is not a number chosen in advance, because
#: a guard whose length window predates the pattern misses the strings it
#: exists to catch.
PATTERN_ALNUM_LENGTHS = (7,)

#: The crop conditions, named once so the output and the CLI cannot drift.
AS_IS = "as-is"
LETTERBOX = "letterbox"

#: Tilt is a judgement, recorded per photograph by eye. These are its values.
TILTS = ("none", "slight", "marked")


# --- the guard -----------------------------------------------------------


def registration_window(lengths=PATTERN_ALNUM_LENGTHS) -> tuple[int, int]:
    """The length window a registration-shaped string can occupy.

    Derived from every pattern the layout establishes, widened by one at each
    end so a neighbouring length is caught too. Never typed.
    """
    return min(lengths) - 1, max(lengths) + 1


def registration_shaped(value: str, window: tuple[int, int] | None = None) -> bool:
    """Whether a string looks like a registration.

    Spaces and dashes are stripped first, because a plate is written with both
    and neither is part of the identity. What remains must be alphanumeric, must
    carry BOTH a letter and a digit -- which is what keeps version strings,
    condition names and the assertion prose out of the net -- and must fall in
    the derived length window.
    """
    lo, hi = window or registration_window()
    stripped = value.replace(" ", "").replace("-", "")
    if not stripped.isalnum() or not stripped.isascii():
        return False
    if not (any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped)):
        return False
    return lo <= len(stripped) <= hi


def find_registration_shaped(node, window=None, path="") -> list[str]:
    """Every registration-shaped string in the object, keys included.

    The walk is recursive because the output nests -- the per-tilt counts are
    sub-objects -- and a guard that checked only the top level would pass the
    one structure it is least likely to be safe on. Keys are checked too: a key
    is as published as a value.
    """
    found = []
    if isinstance(node, str):
        if registration_shaped(node, window):
            found.append(f"{path}={node!r}")
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and registration_shaped(key, window):
                found.append(f"{path}.<key>={key!r}")
            found.extend(find_registration_shaped(value, window, f"{path}.{key}"))
    elif isinstance(node, (list, tuple)):
        for i, item in enumerate(node):
            found.extend(find_registration_shaped(item, window, f"{path}[{i}]"))
    return found


class RegistrationInOutput(RuntimeError):
    """Raised rather than writing a registration into a file."""


def write_output(obj: dict, path: Path, window=None) -> Path:
    """Write the evidence object, or refuse to.

    The refusal is the point. Every other guard in this round protects the
    repository; this one protects the file itself, wherever it is written.
    """
    offenders = find_registration_shaped(obj, window)
    if offenders:
        raise RegistrationInOutput(
            "refusing to write: the object carries "
            f"{len(offenders)} registration-shaped string(s) at {', '.join(offenders)}"
        )
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return path


# --- paths ---------------------------------------------------------------


def inside_git_work_tree(path: Path) -> Path | None:
    """The work tree `path` sits in, or None.

    `.git` is a directory in a clone and a FILE in a worktree, so both count.
    The check walks up from the resolved path, and a path that does not exist
    yet -- an `--out` about to be created -- is answered by its parent.
    """
    here = path.resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


class PathInsideRepository(RuntimeError):
    """Raised rather than reading or writing real data inside a repository."""


def refuse_repository_paths(**paths: Path) -> None:
    bad = []
    for name, path in paths.items():
        tree = inside_git_work_tree(path)
        if tree is not None:
            bad.append(f"--{name} {path} is inside the git work tree at {tree}")
    if bad:
        raise PathInsideRepository(
            "real photographs, their labels and this measurement's output never "
            "enter a repository:\n  " + "\n  ".join(bad)
        )


# --- the crop ------------------------------------------------------------


def border_colour(crop: np.ndarray, ring: int = 2) -> np.ndarray:
    """The median colour of the crop's outermost `ring` pixels.

    Stated rather than assumed, because on a tilted plate that ring is road and
    bumper rather than plate, and it is what the padding is made of.
    """
    top = crop[:ring].reshape(-1, crop.shape[2])
    bottom = crop[-ring:].reshape(-1, crop.shape[2])
    left = crop[:, :ring].reshape(-1, crop.shape[2])
    right = crop[:, -ring:].reshape(-1, crop.shape[2])
    ringpx = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(ringpx, axis=0).astype(np.uint8)


def training_aspect() -> float:
    """The aspect every generated plate has, read from the generator.

    Imported rather than written down: a second copy of this number would make
    "padded to the training aspect" true only by coincidence.
    """
    return PLATE_W / PLATE_H


def letterbox_to_training_aspect(crop: np.ndarray) -> np.ndarray:
    """Pad WHICHEVER axis is short until the crop matches the training aspect.

    Both directions happen. An axis-aligned box around a tilted plate is taller
    than the plate itself, and past enough tilt it is already narrower than the
    training aspect -- so the pad goes on the width, not the height.
    """
    h, w = crop.shape[:2]
    target = training_aspect()
    colour = border_colour(crop)
    if w / h > target:                       # too wide: pad the height
        new_h = int(round(w / target))
        pad = new_h - h
        top, bottom, left, right = pad // 2, pad - pad // 2, 0, 0
    else:                                    # too tall: pad the width
        new_w = int(round(h * target))
        pad = new_w - w
        top, bottom, left, right = 0, 0, pad // 2, pad - pad // 2
    return cv2.copyMakeBorder(
        crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[int(c) for c in colour]
    )


def as_capture(image: np.ndarray, camera_id: str) -> Capture:
    """PNG, never JPEG: no second lossy step between the photograph and the number."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not PNG-encode the crop")
    return Capture.now(buf.tobytes(), camera_id=camera_id)


# --- the object ----------------------------------------------------------


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_output(
    *,
    buckets: dict,
    per_tilt: dict,
    mean_conf_exact: float | None,
    mean_conf_wrong: float | None,
    n: int,
    excluded: int,
    weights_id: str | None,
    threshold: float,
    condition: str,
    python_version: str,
    torch_version: str,
    package_version: str,
    script_digest: str,
    window: tuple[int, int],
    timestamp: str,
) -> dict:
    """The evidence object. A pure function: values in, object out.

    The real run and the tests both call this, so the object's SHAPE cannot
    drift between what is measured and what is checked -- while the tests stay
    free to supply representative values for fields a run derives from a
    checkpoint that a test deliberately never loads.
    """
    return {
        "n": n,
        "excluded": excluded,
        "counts": dict(buckets),
        "exact": buckets["exact_answer"] + buckets["exact_fallback"],
        "wrong": buckets["wrong_answer"] + buckets["wrong_fallback"],
        "no_text": buckets["no_text"],
        "mean_confidence_exact": mean_conf_exact,
        "mean_confidence_wrong": mean_conf_wrong,
        "per_tilt": per_tilt,
        "weights_id": weights_id,
        "threshold": threshold,
        "crop_condition": condition,
        "registration_window": list(window),
        "python": python_version,
        "torch": torch_version,
        "package_version": package_version,
        "harness_sha256": script_digest,
        "measured_at": timestamp,
        "capture": {
            "value": "hand-held phone camera",
            "assertion": True,
            "asserted_by": "Gokhan",
        },
        "crops": {
            "value": "manual rectangles, iterated by eye at full resolution",
            "assertion": True,
            "asserted_by": "CC",
        },
        "tilt": {
            "value": "coarse tilt classified by eye: none / slight / marked",
            "assertion": True,
            "asserted_by": "CC",
        },
    }


def empty_buckets() -> dict:
    return {
        "exact_answer": 0,
        "exact_fallback": 0,
        "wrong_answer": 0,
        "wrong_fallback": 0,
        "no_text": 0,
    }


def classify(read, want: str) -> str:
    """Which bucket a `Read` and its label fall into.

    `identity.plate` is None or text and `outcome` is ANSWER or FALLBACK
    INDEPENDENTLY, so the two questions are asked separately and answered in one
    label. A read with no text is `no_text` whatever its outcome -- there is no
    text to be right or wrong about.
    """
    got = read.identity.plate
    if not got:
        return "no_text"
    answered = read.outcome == ANSWER
    correct = normalise(got) == normalise(want)
    if correct:
        return "exact_answer" if answered else "exact_fallback"
    return "wrong_answer" if answered else "wrong_fallback"


# --- the run -------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--as-is", dest="condition", action="store_const", const=AS_IS)
    mode.add_argument("--letterbox", dest="condition", action="store_const", const=LETTERBOX)
    ap.set_defaults(condition=AS_IS)
    args = ap.parse_args()

    refuse_repository_paths(photos=args.photos, labels=args.labels, out=args.out)

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    photos = labels["photos"]

    engine = PlateEngine(args.weights)
    window = registration_window(tuple(labels.get("pattern_alnum_lengths", PATTERN_ALNUM_LENGTHS)))

    buckets = empty_buckets()
    per_tilt = {t: {"n": 0, "exact": 0, "wrong": 0, "no_text": 0} for t in TILTS}
    conf_exact: list[float] = []
    conf_wrong: list[float] = []
    excluded = 0

    print(f"  condition: {args.condition}    weights: {engine.engine.weights_id}"
          f"    threshold: {engine.threshold}")
    print(f"  {'id':>4}  {'read':<12} {'conf':>7}  {'outcome':<8} {'verdict':<12} tilt")
    for pid in sorted(photos):
        entry = photos[pid]
        if entry.get("excluded"):
            excluded += 1
            print(f"  {pid:>4}  {'-':<12} {'-':>7}  {'-':<8} {'EXCLUDED':<12} "
                  f"{entry['excluded']}")
            continue
        img = cv2.imread(str(args.photos / entry["file"]))
        if img is None:
            raise RuntimeError(f"{pid}: could not read the photograph")
        x, y, w, h = entry["rect"]
        crop = img[y:y + h, x:x + w]
        if args.condition == LETTERBOX:
            crop = letterbox_to_training_aspect(crop)
        read = engine.read([as_capture(crop, camera_id=pid)])
        bucket = classify(read, entry["registration"])
        buckets[bucket] += 1
        tilt = entry.get("tilt", "none")
        per_tilt[tilt]["n"] += 1
        if bucket.startswith("exact"):
            per_tilt[tilt]["exact"] += 1
            conf_exact.append(read.confidence)
        elif bucket.startswith("wrong"):
            per_tilt[tilt]["wrong"] += 1
            conf_wrong.append(read.confidence)
        else:
            per_tilt[tilt]["no_text"] += 1
        # Terminal only. This line carries a real registration whenever the read
        # is right, so it goes into no file and no receipt.
        print(f"  {pid:>4}  {read.identity.plate or '-':<12} {read.confidence:>7.4f}  "
              f"{read.outcome:<8} {bucket:<12} {tilt}")

    n = sum(buckets.values())
    obj = build_output(
        buckets=buckets,
        per_tilt=per_tilt,
        mean_conf_exact=statistics.mean(conf_exact) if conf_exact else None,
        mean_conf_wrong=statistics.mean(conf_wrong) if conf_wrong else None,
        n=n,
        excluded=excluded,
        weights_id=engine.engine.weights_id,
        threshold=engine.threshold,
        condition=args.condition,
        python_version=sys.version.split()[0],
        torch_version=__import__("torch").__version__,
        package_version=engine.engine.version,
        script_digest=script_sha256(),
        window=window,
        timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    )
    write_output(obj, args.out, window)
    print(f"\n  n={n} excluded={excluded}  "
          + "  ".join(f"{k}={v}" for k, v in buckets.items()))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
