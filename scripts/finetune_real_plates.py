#!/usr/bin/env python3
"""Fine-tune the plate reader on PHOTOGRAPHED plates, then let the harness read them again.

    python scripts/finetune_real_plates.py \
        --photos DIR --labels FILE --init WEIGHTS --out WEIGHTS \
        --train-ids p01,p02,… [--exclude-ids p13,p14,…] [--steps N]

Every checkpoint this project has published was trained on plates the generator
drew. The measurement that motivated this script read 0 of 18 photographed
plates with two such checkpoints — which says synthetic-only training reads
nothing real, and says nothing at all about training that includes real plates.
This runs that second experiment on the only real plates there are.

Three rules it enforces, and the first is new here:

  * **A checkpoint trained on real plates IS real data.** It encodes the
    registrations it was fitted to. So `--out` is refused inside any git work
    tree exactly as `--photos` and `--labels` are; `models/` being gitignored is
    not the point and would not be enough.
  * **A model may not be tested on a plate it trained on.** `--exclude-ids`
    names the ids a fold will later be tested against, and any overlap with
    `--train-ids` refuses to start. The caller states both; this refuses to be
    the thing that quietly decides.
  * **A label that cannot be encoded is not a label.** `model.encode` silently
    DROPS any character outside `CHARS`, which would train the wrong target
    without a word, so every label character is checked against `charset()`
    before the first step.

Nothing about a registration reaches stdout: the run prints the ids it trained
on and their count, never their text.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# The guards are IMPORTED, not copied: two copies of a refusal drift, and the
# one that matters is always the copy nobody re-read.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_real_plates import (  # noqa: E402
    letterbox_to_training_aspect,
    refuse_repository_paths,
)

from vehicle_id.engine import normalise, weights_id  # noqa: E402
from vehicle_id.plates.dataset import (  # noqa: E402
    TRAIN_SEED,
    SyntheticPlates,
    collate,
    to_tensor,
)
from vehicle_id.plates.generator import PLATE_H, PLATE_W, degrade  # noqa: E402
from vehicle_id.plates.model import BLANK, PlateNet, encode  # noqa: E402
from vehicle_id.plates.templates import charset  # noqa: E402

#: The synthetic loader draws degradation from this range; real samples are
#: augmented over the same one so the two halves of a batch are comparable.
DEGRADE_RANGE = (0, 6)

#: 18 photographs would otherwise become 18 memorised pixel arrays.
ROTATION_DEG = 4.0
SCALE_JITTER = 0.06
TRANSLATE_PX = 4


class OverlappingIds(RuntimeError):
    """Raised rather than training on a plate the caller will test against."""


class UnencodableLabel(RuntimeError):
    """Raised rather than training towards a target `encode` would silently trim."""


def refuse_overlap(train_ids: list[str], exclude_ids: list[str]) -> None:
    """A fold may not train on what it will be tested on.

    Checked before anything is loaded, and checked on the ids the caller states
    rather than on anything this script infers -- inference is how a hold-out
    quietly stops being one.
    """
    overlap = sorted(set(train_ids) & set(exclude_ids))
    if overlap:
        raise OverlappingIds(
            "these ids are in --train-ids AND in --exclude-ids, so the model "
            f"would be tested on a plate it trained on: {', '.join(overlap)}"
        )


def refuse_unencodable(labels: dict[str, str]) -> None:
    """Every label character must survive `encode`."""
    alphabet = set(charset())
    bad = sorted({pid for pid, text in labels.items() if set(text) - alphabet})
    if bad:
        raise UnencodableLabel(
            f"{len(bad)} label(s) contain a character outside charset() and "
            "`encode` would drop it silently, training the wrong target: "
            f"{', '.join(bad)}"
        )


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """A small affine, then the generator's own degradation ladder.

    The affine is what stops eighteen pictures becoming eighteen memorised pixel
    arrays; the ladder is reused rather than reinvented so a real sample and a
    synthetic one are degraded by the same code.
    """
    angle = rng.uniform(-ROTATION_DEG, ROTATION_DEG)
    scale = 1.0 + rng.uniform(-SCALE_JITTER, SCALE_JITTER)
    matrix = cv2.getRotationMatrix2D((PLATE_W / 2, PLATE_H / 2), angle, scale)
    matrix[0, 2] += rng.uniform(-TRANSLATE_PX, TRANSLATE_PX)
    matrix[1, 2] += rng.uniform(-TRANSLATE_PX, TRANSLATE_PX)
    out = cv2.warpAffine(image, matrix, (PLATE_W, PLATE_H), borderMode=cv2.BORDER_REPLICATE)
    return degrade(out, rng.randint(*DEGRADE_RANGE), rng)


def real_bases(photos: Path, labels: dict, train_ids: list[str]) -> list[tuple[np.ndarray, str]]:
    """One base image per (id, crop condition), sized as the generator sizes plates.

    BOTH conditions the harness measures are produced, so the fine-tune sees the
    geometry it will be tested under either way. Everything is resized to
    PLATE_W x PLATE_H because `degrade` is built for exactly that -- its
    perspective warp is sized to those constants.
    """
    out = []
    for pid in train_ids:
        entry = labels["photos"][pid]
        if "rect" not in entry:
            raise KeyError(f"{pid} has no rect; it is excluded and cannot be trained on")
        image = cv2.imread(str(photos / entry["file"]))
        if image is None:
            raise RuntimeError(f"{pid}: could not read the photograph")
        x, y, w, h = entry["rect"]
        crop = image[y:y + h, x:x + w]
        text = normalise(entry["registration"])
        for variant in (crop, letterbox_to_training_aspect(crop)):
            out.append((cv2.resize(variant, (PLATE_W, PLATE_H),
                                   interpolation=cv2.INTER_AREA), text))
    return out


def finetune(
    bases: list[tuple[np.ndarray, str]],
    init: Path,
    out: Path,
    steps: int,
    real_per_batch: int,
    synthetic_per_batch: int,
    device: torch.device,
    train_ids: list[str],
    seed: int = 0,
) -> dict:
    """`train.py`'s loop, with an init and a mixed batch.

    The synthetic half is not decoration: without it the checkpoint drifts off
    the ladder its operating point is measured on, and that operating point is
    what the engine refuses to start without.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = PlateNet().to(device)
    blob = torch.load(init, map_location=device)
    model.load_state_dict(blob["state_dict"])

    synthetic = SyntheticPlates(size=steps * synthetic_per_batch // 4 + 1024, seed=TRAIN_SEED)
    # `collate` is train.py's: CTC targets are variable length and the default
    # collate would try to stack them.
    loader = DataLoader(
        synthetic, batch_size=synthetic_per_batch, shuffle=True,
        collate_fn=collate, drop_last=True,
    )

    loss_fn = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)

    model.train()
    step, started, running = 0, time.perf_counter(), 0.0
    while step < steps:
        for s_images, s_targets, s_lengths, _ in loader:
            if step >= steps:
                break
            images = [s_images]
            targets = [s_targets]
            lengths = [int(n) for n in s_lengths]
            for _ in range(real_per_batch):
                base, text = rng.choice(bases)
                images.append(to_tensor(augment(base, rng)).unsqueeze(0))
                ids = encode(text)
                targets.append(torch.tensor(ids, dtype=torch.long))
                lengths.append(len(ids))
            batch = torch.cat(images).to(device)
            target = torch.cat(targets)
            length = torch.tensor(lengths, dtype=torch.long)

            logits = model(batch)
            logp = logits.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((batch.size(0),), logits.size(1), dtype=torch.long)
            # aten::_ctc_loss has no MPS kernel; same explicit CPU hop train.py makes.
            loss = loss_fn(logp.cpu(), target, input_lengths, length)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

            running += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"  step {step:5d}/{steps}  loss {running / 100:.4f}")
                running = 0.0

    elapsed = time.perf_counter() - started
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "steps": steps,
            "init_weights_id": weights_id(init),
            "trained_on_ids": list(train_ids),
        },
        out,
    )
    return {"steps": steps, "seconds": elapsed, "device": device.type}


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-ids", required=True,
                    help="comma-separated ids this run may train on; nothing else is loaded")
    ap.add_argument("--exclude-ids", default="",
                    help="comma-separated ids this checkpoint will be TESTED on; overlap refuses")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--real-per-batch", type=int, default=16)
    ap.add_argument("--synthetic-per-batch", type=int, default=48)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    train_ids = [i.strip() for i in args.train_ids.split(",") if i.strip()]
    exclude_ids = [i.strip() for i in args.exclude_ids.split(",") if i.strip()]

    # Refusals first, before a single photograph is opened.
    refuse_overlap(train_ids, exclude_ids)
    refuse_repository_paths(photos=args.photos, labels=args.labels, out=args.out)

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    texts = {
        pid: normalise(labels["photos"][pid]["registration"])
        for pid in train_ids
        if "registration" in labels["photos"].get(pid, {})
    }
    missing = [pid for pid in train_ids if pid not in texts]
    if missing:
        raise KeyError(f"no usable label for: {', '.join(missing)}")
    refuse_unencodable(texts)

    print(f"  training on {len(train_ids)} id(s): {', '.join(train_ids)}")
    if exclude_ids:
        print(f"  held out (never loaded): {', '.join(exclude_ids)}")
    bases = real_bases(args.photos, labels, train_ids)
    print(f"  {len(bases)} real base images ({len(train_ids)} ids x 2 crop conditions)")

    stats = finetune(
        bases=bases,
        init=args.init,
        out=args.out,
        steps=args.steps,
        real_per_batch=args.real_per_batch,
        synthetic_per_batch=args.synthetic_per_batch,
        device=pick_device(args.device),
        train_ids=train_ids,
    )
    print(f"\n  fine-tuned {stats['steps']} steps in {stats['seconds']:.1f}s on {stats['device']}")
    print(f"  init {weights_id(args.init)} -> out {weights_id(args.out)}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
