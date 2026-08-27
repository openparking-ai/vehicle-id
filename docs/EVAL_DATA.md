# Evaluation data, and why most of it cannot be used

Every accuracy number Open Parking AI publishes comes from
`scripts/eval_plates.py`. This page is the audit of what that harness is allowed
to run on, and why the answer is mostly "nothing public".

## The rule

A dataset is usable only if its licence permits **commercial use**. "Free for
research" is not free for us. And nothing is committed to this repository: eval
data is downloaded locally by a script, or generated.

## What was checked

| Dataset | Licence | Usable | Note |
|---|---|---|---|
| Roboflow / Open Images — License Plates (US/EU) | images CC BY 2.0, annotations CC BY 4.0 | **yes**, with attribution | Only **350 images**, bounding boxes only — no plate text. A real-image sanity check, never a benchmark. |
| CCPD | **MIT** | legally yes | Chinese plates. Tells us nothing about Florida accuracy. |
| UFPR-ALPR | academic research only, non-commercial | **no** | Explicitly excludes commercial use. |
| VeRi-776 | non-commercial, email request | **no** | Also: "due to the privacy issue we will not provide the license plates in the future". |
| Stanford Cars | ImageNet-style, research | **no** | Still available, still not commercially licensed. |

Sources are linked from `scripts/fetch_eval_data.py` and the receipt.

**There is no adequately licensed public set for US plate READING.** That is a
finding, not a failure — and it is why the recogniser is trained on generated
plates instead.

## What we use instead

`vehicle_id.plates.generator` produces both the training and the
evaluation sets. The eval set is re-derived from a **seed**, so it is
reproducible without being stored, and there is nothing to accidentally commit.

Two consequences worth stating plainly:

**The training data has no licence question**, because we made it. The weights
that come out are Open Parking AI's own asset, under our own terms, with no
upstream able to withdraw them.

**Synthetic accuracy is not real accuracy.** The generator uses OpenCV's
built-in fonts; real plates use embossing typefaces we neither have nor could
redistribute. That is the largest domain gap, and it is why the harness prints
real-plate accuracy as NOT MEASURABLE rather than quoting the synthetic number
and hoping. The real number comes from the physical bench, from our own
vehicles, kept local, never committed.

## Scoring a photograph of a real floor, defined before any exists

The presence gate does not separate a vehicle from an empty lane on ground that
carries no texture of its own, and it says so with a number: below
`DEFAULT_MIN_REFERENCE_TEXTURE` grey levels of typical local texture it declines
to answer at all. Every figure published about that lives on a `texture` axis of
a synthetic fixture, and the open question — the module's central one, now that
most garage entries are known to be covered and therefore likely sealed or
painted concrete — is what a REAL floor measures on the same scale.

**The mapping is defined here, before the first photograph is taken, and
deliberately so.** A photograph scored after the fact, against a scale chosen
after seeing it, is not a measurement. The quantity is:

> the **median local standard deviation**, in grey levels, over an 11x11 window
> of the greyscale image — which is exactly `_typical_local_texture(grey, 11)`
> in `src/vehicle_id/presence.py`, the same function the detector runs on a
> reference view when one is configured.

```python
import cv2, sys
sys.path.insert(0, "src")
from vehicle_id.presence import DEFAULT_MIN_REFERENCE_TEXTURE, _typical_local_texture

grey = cv2.cvtColor(cv2.imread("floor.jpg"), cv2.COLOR_BGR2GRAY)
texture = _typical_local_texture(grey, 11)
print(f"{texture:.2f} grey levels; the gate declines below {DEFAULT_MIN_REFERENCE_TEXTURE}")
```

Read the result against `matrix_ground_reference_texture` in
`docs/measured/presence.json`, which records what the fixture's own ground
measures at each value of its axis. Two caveats that are part of the definition
rather than notes on it:

- **It is a property of the photograph, not of the floor.** Exposure, focus,
  resolution, distance and JPEG compression all move it. The number is only
  comparable between images taken the way a lane camera takes them: the whole
  entry in frame, in focus, at the light the lane actually has.
- **It says whether the gate can ANSWER, not whether it will answer well.** The
  floor is the point below which the gate returns `null` — measured at
  <!--m:gate.min_reference_texture-->1.5<!--/m--> grey levels. Sitting above it
  is necessary and not sufficient: the fixture's own 0.25-texture row measures
  <!--m:texture_floor.matrix_lowest-->3.821<!--/m--> grey levels, comfortably
  above the floor, and still fails to separate a vehicle from an empty lane at
  all. The scene that does reach the floor measures
  <!--m:texture_floor.smooth_floor-->0.67<!--/m-->.

Photographs of real floors are **never committed to this repository**, the same
rule as every other piece of real data here. The number they produce is.

## Attribution

If the Roboflow/Open Images set is used, CC BY requires attribution: Open Images
Dataset (Google LLC), curated by Roboflow. `scripts/fetch_eval_data.py` writes
the notice alongside the download.

---

Built by 72 Knots Method by 72Knots.ai
