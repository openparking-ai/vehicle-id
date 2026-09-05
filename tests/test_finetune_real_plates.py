"""The fine-tune script's two refusals, and its sample builder.

Fine-tuning on photographed plates puts two new ways to get a wrong number
within reach, and neither is visible in the output once it has happened:

  * **testing on a plate the model trained on.** A hold-out that quietly stops
    being one reports memorisation as reading. `refuse_overlap` is checked on
    the ids the CALLER states, before anything is loaded, because inference
    about which ids overlap is exactly where a fold loses its meaning.
  * **training towards a target that was silently trimmed.** `model.encode`
    drops any character outside `CHARS` without a word, so a label carrying one
    would train the model to emit something shorter than the plate and nothing
    anywhere would say so.

No registration-shaped literal appears in this file. The fixtures use characters
that are in the charset but do not form a registration-length alphanumeric
string, so the repository's own plate-string guard has nothing to catch here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")
pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from finetune_real_plates import (  # noqa: E402
    OverlappingIds,
    UnencodableLabel,
    augment,
    real_bases,
    refuse_overlap,
    refuse_unencodable,
)
from vehicle_id.plates.generator import PLATE_H, PLATE_W  # noqa: E402
from vehicle_id.plates.templates import charset  # noqa: E402

guarantee = pytest.mark.guarantee


# --- the hold-out refusal ------------------------------------------------


@guarantee
def test_an_overlapping_fold_refuses_to_start():
    """The fold guard. Its whole job is to make a hold-out stay one."""
    with pytest.raises(OverlappingIds) as caught:
        refuse_overlap(["p01", "p02", "p03"], ["p03", "p04"])
    assert "p03" in str(caught.value)


@guarantee
def test_a_disjoint_fold_is_allowed():
    """The control for the control: without it the refusal could be unconditional."""
    refuse_overlap(["p01", "p02"], ["p03", "p04"])


@guarantee
def test_the_overlap_check_does_not_depend_on_order_or_duplicates():
    with pytest.raises(OverlappingIds):
        refuse_overlap(["p09", "p09", "p01"], ["p01"])


# --- the encodable-label refusal -----------------------------------------


@guarantee
def test_a_label_with_an_uncodable_character_refuses_to_start():
    """`encode` would drop it silently and train a shorter target."""
    outside = next(chr(c) for c in range(ord("a"), ord("z") + 1) if chr(c) not in charset())
    with pytest.raises(UnencodableLabel) as caught:
        refuse_unencodable({"p01": "AB" + outside})
    assert "p01" in str(caught.value)


@guarantee
def test_labels_entirely_inside_the_charset_are_allowed():
    alphabet = charset()
    assert refuse_unencodable({"p01": alphabet[0] + alphabet[1], "p02": alphabet[-1]}) is None


# --- the sample builder --------------------------------------------------


def fixture_photo(tmp_path: Path, w: int = 900, h: int = 240) -> tuple[Path, dict]:
    """A synthetic stand-in for a photograph. No real image, no registration."""
    image = np.full((h * 2, w * 2, 3), 180, np.uint8)
    cv2.rectangle(image, (100, 100), (100 + w, 100 + h), (240, 240, 240), -1)
    # Structure inside the plate area. A UNIFORM fill letterboxes to the same
    # pixels it started as -- the padding colour is the median of its own border
    # ring -- so a flat fixture would make the two conditions indistinguishable
    # and the check below vacuous.
    for i in range(6):
        cx = 100 + int(w * (0.12 + 0.13 * i))
        cv2.rectangle(image, (cx, 100 + h // 4), (cx + w // 22, 100 + 3 * h // 4),
                      (30, 30, 30), -1)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), image)
    alphabet = charset()
    labels = {
        "photos": {
            "p01": {
                "file": path.name,
                "rect": [100, 100, w, h],
                "registration": alphabet[0] + alphabet[1],
            }
        }
    }
    return path, labels


@guarantee
def test_every_real_sample_is_sized_the_way_degrade_expects(tmp_path):
    """`degrade`'s perspective warp is sized to PLATE_W x PLATE_H literally.

    Handed anything else it would warp to the wrong corners, so the builder
    resizes first and this asserts it did.
    """
    path, labels = fixture_photo(tmp_path)
    bases = real_bases(path.parent, labels, ["p01"])
    assert len(bases) == 2, "both crop conditions are built for every id"
    for image, _ in bases:
        assert image.shape[:2] == (PLATE_H, PLATE_W)
        assert image.dtype == np.uint8


@guarantee
def test_the_two_crop_conditions_are_actually_different(tmp_path):
    """Building both and getting one twice would silently halve the real data."""
    path, labels = fixture_photo(tmp_path)
    (as_is, _), (letterboxed, _) = real_bases(path.parent, labels, ["p01"])
    assert not np.array_equal(as_is, letterboxed)


@guarantee
def test_a_sample_whose_id_has_no_rect_is_refused(tmp_path):
    """An excluded photograph has no crop, and must not be trained on by accident."""
    path, labels = fixture_photo(tmp_path)
    labels["photos"]["p99"] = {"file": path.name, "excluded": "no ground truth"}
    with pytest.raises(KeyError):
        real_bases(path.parent, labels, ["p99"])


@guarantee
def test_augmentation_moves_the_image_and_keeps_its_shape():
    """Eighteen photographs must not become eighteen memorised pixel arrays."""
    import random

    base = np.full((PLATE_H, PLATE_W, 3), 200, np.uint8)
    cv2.rectangle(base, (40, 40), (280, 120), (20, 20, 20), 3)
    a = augment(base, random.Random(1))
    b = augment(base, random.Random(2))
    assert a.shape == base.shape and b.shape == base.shape
    assert not np.array_equal(a, b), "two draws produced identical images"
    assert not np.array_equal(a, base), "augmentation was a no-op"
