"""Is a VEHICLE there? Asked before anything is read, and answered separately.

This is deliberately not "is a plate there". They are different questions and
conflating them breaks the product in both directions: a car with a filthy,
damaged or missing front plate is a legitimate entry and must be admitted, and
a steel plate held over the loop is not a car and must receive nothing. A gate
keyed on plate presence refuses honest customers AND still cannot tell the
fraud from a dirty plate.

**No model, no weights, no dataset.** That is a licensing decision before it is
a technical one, and it is the same one the recogniser forced. Every
general-purpose detector worth using is COCO-trained, and COCO's images are not
the consortium's to license -- the annotations are CC BY 4.0 but the images are
Flickr's, individually, with users accepting full responsibility. torchvision's
own documentation says the same thing about its weights in as many words: they
"may have their own licenses ... derived from the dataset used for training",
and it is the user's job to work out whether they may use them. That is exactly
the shape of trap that disqualified Ultralytics and OpenALPR before a line was
written. And the escape hatch the recogniser used -- generate the training data
ourselves -- is not available here: a plate is a rendered rectangle with a
parameterisable font and layout, which is why that generator works. A car is
not, and no adequately licensed real vehicle imagery exists (see
docs/EVAL_DATA.md, where Stanford Cars and VeRi-776 are both excluded).

So the question is answered the way a fixed camera pointed at a fixed piece of
tarmac makes possible: by measuring how much of the scene changed. A vehicle
occupies a large, contiguous part of the frame. A person holding a metal plate
over an inductive loop does not, and neither does rain, a bird or a shadow.

**Three states, not two.** `True`, `False`, and `None` for NOT MEASURED. Without
a reference view of the empty lane there is nothing to compare against, and the
contract's first rule applies to this field like any other: a value that was not
measured is null. A lane that cannot measure presence behaves exactly as it did
before this existed -- it does not start refusing customers on the strength of a
number nobody computed.

What is measurable now and what is not is stated plainly in the README: this
gate can be shown to reject sensor noise, and its behaviour on real vehicles at
a real lane is NOT MEASURED until the physical bench exists.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Fraction of the frame a vehicle is expected to occupy at an entry lane.
#: NOT a measured number -- it cannot be, without lane footage -- and it is
#: therefore configurable and stated as an assumption rather than buried. A
#: camera aimed at an entry lane sees a car fill much of the frame; the value
#: below is deliberately low so that the gate errs towards admitting.
DEFAULT_MIN_OCCUPANCY = 0.15

#: How different a pixel must be from the reference to count as changed. Also
#: an assumption, also configurable.
DEFAULT_PIXEL_DELTA = 30


@dataclass(frozen=True, slots=True)
class Presence:
    """What was measured about whether a vehicle is there.

    `present` is None when it could not be measured at all. `occupancy` is the
    fraction of the frame that differs from the reference, and it is reported
    whether the answer is yes or no -- a lane operator debugging a gate that
    refuses everything needs the number, not just the verdict.
    """

    present: bool | None
    confidence: float | None = None
    occupancy: float | None = None
    reason: str = ""

    @staticmethod
    def unmeasured(reason: str) -> Presence:
        return Presence(present=None, confidence=None, occupancy=None, reason=reason)


class PresenceDetector:
    """Compares captures against a reference view of the empty lane."""

    def __init__(
        self,
        reference: object | None = None,
        min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
        pixel_delta: int = DEFAULT_PIXEL_DELTA,
    ) -> None:
        self.min_occupancy = min_occupancy
        self.pixel_delta = pixel_delta
        self._reference = None
        if reference is not None:
            self.set_reference(reference)

    def set_reference(self, image) -> None:
        """Set the empty-lane view everything is compared against."""
        self._reference = _grey(image)

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    def measure(self, images: Sequence) -> Presence:
        """The largest occupancy across the captures decides.

        Largest, not mean: the captures are a burst around one moment and a
        vehicle may only be fully in frame for part of it. A gate that averaged
        would refuse a car that arrived halfway through the burst.
        """
        if self._reference is None:
            # Not a refusal and not an admission. Nobody measured it.
            return Presence.unmeasured("no reference view of the empty lane is configured")
        usable = [_grey(image) for image in images if image is not None]
        if not usable:
            return Presence.unmeasured("no capture could be decoded")

        occupancies = [self._occupancy(image) for image in usable]
        occupancy = max(occupancies)
        present = occupancy >= self.min_occupancy
        # Distance from the threshold, scaled so that a scene which is entirely
        # unchanged or entirely changed reads as certain. It is a measure of how
        # far from the decision boundary this is -- not a probability, and the
        # contract does not claim it is one.
        span = max(self.min_occupancy, 1.0 - self.min_occupancy)
        confidence = min(1.0, abs(occupancy - self.min_occupancy) / span)
        return Presence(
            present=present,
            confidence=confidence,
            occupancy=occupancy,
            reason=(
                f"{occupancy:.1%} of the frame differs from the empty lane, "
                f"threshold {self.min_occupancy:.1%}"
            ),
        )

    def _occupancy(self, image) -> float:
        import cv2
        import numpy as np

        reference = self._reference
        if image.shape != reference.shape:
            image = cv2.resize(image, (reference.shape[1], reference.shape[0]))
        difference = cv2.absdiff(image, reference)
        changed = difference >= self.pixel_delta
        if not changed.any():
            return 0.0

        # Contiguity is what separates a vehicle from noise, rain or a bird.
        # Scattered pixels covering 40% of the frame are not a car; one blob
        # covering 20% is. Only the largest connected region counts.
        mask = (changed.astype("uint8")) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), "uint8"))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return 0.0
        largest = max(stats[1:, cv2.CC_STAT_AREA])
        return float(largest) / float(mask.size)


def _grey(image):
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array
