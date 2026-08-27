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
tarmac makes possible: by measuring how much of the scene STOPPED LOOKING LIKE
THE SAME GROUND, in one contiguous region. A vehicle covers a large part of the
frame. A person holding a metal plate over an inductive loop does not, and
neither does a bird or a shadow.

**Structure, not brightness, and that distinction cost a round.** The measure
used to be the magnitude of an intensity difference, and it was blind to a
vehicle the same brightness as the ground it stood on: at a reference level of
90, a car occupying 43.75% of the frame -- nearly three times the occupancy
floor -- read `present=False` at 0.97 confidence for any body shade between
roughly 70 and 120. That is an ordinary grey car on grey asphalt, `False` is the
value that ends a transaction, and moving the threshold only moves the band.
What is measured now is the contrast and structure terms of SSIM with the
LUMINANCE term dropped, per window: a car at the tarmac's own luminance still
occludes the tarmac's stone texture, its markings and its drain.

That change did not come free, and the cost is measured rather than argued away:
see `docs/measured/presence.json`. It does not separate vehicle from empty on
near-featureless ground, and heavy weather stops it answering where the old
measure answered correctly. Both fail to `None`. No case measured produces a
wrongful `False`, which is the property that was worth buying.

**Three states, not two.** `True`, `False`, and `None` for NOT MEASURED. Without
a reference view of the empty lane there is nothing to compare against, and the
contract's first rule applies to this field like any other: a value that was not
measured is null. A lane that cannot measure presence behaves exactly as it did
before this existed -- it does not start refusing customers on the strength of a
number nobody computed.

## The three ways this refuses to answer, and why none of them is `False`

The first version of this module measured raw intensity difference against the
reference and called any large contiguous change a vehicle. That is not the same
question, and it was wrong on every frame where the whole scene moved at once: a
dead camera (flat black), a blown exposure (flat white), dusk against a daylight
reference and sun on tarmac all reported `presence=true`, three of them at
confidence 1.0, with nothing in frame. None of them opened a barrier -- the
recogniser's measured operating point held -- but the gate was asserting a
positive measurement of something it had not measured.

Each of those is now a distinct refusal, and every one of them resolves to
`None`, never to `False`:

  * **The frame carries no information** -- flat black, flat white, a covered
    lens. Near-zero variance. A dead camera is an equipment fault, not a
    vehicle, and it is not "no vehicle" either: nobody can see the lane.
  * **The lane no longer looks like the reference** -- so little of the
    reference view is still recognisable that there is nothing left to compare
    against. A camera that was knocked, a scene rebuilt overnight, or a vehicle
    so close it fills the frame. All three are indistinguishable from here, and
    the honest answer is that this measurement does not apply.
  * **The change fills the frame** -- the ceiling. Occupancy approaching the
    whole frame is evidence the scene changed, not that a car is present.

`False` means "the lane is visible, its ground is still recognisable as the
ground the reference describes, and nothing is standing on it". Note the middle
clause: it is what the measurement establishes, and it is weaker than "there is
nothing there". That is the only claim that ends a transaction before it starts,
so it is the only one this module will make from a measurement it actually has.
Everything else is `None`, which puts the lane back to the behaviour it had
before this stage existed: a ticket and a human.

**Illumination is cancelled, not measured.** The comparison fits a global gain
and offset between the reference and the capture before comparing, so the
same lane at a different exposure differs by nothing. The structural terms are
already invariant to a global gain; the fit is kept because it is also the guard
that decides whether a capture is a view of THIS LANE at all. What survives it is
a change in the scene rather than a change in the light.

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

#: The ceiling. Above this the change is not a vehicle-shaped part of the scene,
#: it IS the scene, and the honest answer is NOT MEASURED rather than either
#: verdict. Also an assumption, also configurable.
#:
#: It deliberately does not return `False`. An upper bound that refused would
#: turn a van, a truck or a bus -- which legitimately fill an entry lane view --
#: into "no vehicle", and at the lane that means no ticket, no session and no
#: vend for a customer who is really there.
DEFAULT_MAX_OCCUPANCY = 0.90

#: How much of a window's STRUCTURAL agreement with the reference must be lost
#: before it counts as changed, 0 to 1. Also an assumption, also configurable.
#:
#: This replaced a grey-level distance, and the replacement is a change of
#: measure rather than a retuned constant. Deciding "changed" on the magnitude
#: of an intensity difference is blind to an object at the same luminance as the
#: ground it sits on: at a reference level of 90 a vehicle occupying 43.75% of
#: the frame -- nearly three times the occupancy floor -- read `present=False`
#: at 0.97 confidence for any body shade between roughly 70 and 120, which is a
#: perfectly ordinary grey car on grey asphalt. Moving the threshold moves that
#: band; it does not remove it, because the quantity being measured is the wrong
#: one.
#:
#: What is measured instead is whether the window still looks like the same
#: PIECE OF GROUND: the contrast and structure terms of SSIM, with the luminance
#: term dropped precisely because absolute brightness is the part that must not
#: decide. A car at the tarmac's own mean luminance still occludes the tarmac's
#: stone texture, its markings and its drain, and that is what this sees.
DEFAULT_MIN_STRUCTURAL_CHANGE = 0.5

#: The stabiliser in the SSIM terms, as a fraction of the reference's OWN
#: typical local texture. SSIM's textbook constants are a fraction of the 8-bit
#: range, which is calibrated for photographs; on a lane whose ground varies by
#: a few grey levels they swamp the measurement and report every window as
#: agreeing. Scaling to the lane means "half its structural agreement" means the
#: same thing on coarse chip seal and on new blacktop.
DEFAULT_STABILISER = 0.5

#: The side of the local window, in pixels. Large enough that the correlation
#: inside it is estimated from enough samples to mean something, small enough
#: that a vehicle covers many of them.
DEFAULT_WINDOW = 11

#: Below this typical local texture, in grey levels, the reference view carries
#: nothing for a structural measure to compare against -- a blank wall, a lens
#: pointed at the sky. That is NOT MEASURED, and never `false`: a measure that
#: cannot see the ground cannot report that the ground is empty.
DEFAULT_MIN_REFERENCE_TEXTURE = 1.5

#: Below this standard deviation a frame carries no information: flat black,
#: flat white, a lens someone taped over. Grey levels. A real lane view is well
#: above it -- tarmac alone has markings, kerbs and drains -- and a sensor
#: producing nothing but its own read noise is well below it.
DEFAULT_MIN_FRAME_STD = 4.0

#: The other way a frame carries no information: plenty of variance, none of it
#: structured. Neighbouring pixels of any real photograph agree; neighbouring
#: samples of sensor noise do not. Measured as the correlation between a frame
#: and itself shifted one pixel, and no model or dataset is involved in the
#: distinction. Measured on the scenes in `tests/lanes.py`:
#:
#:     lane view          0.92      plate crop        0.90
#:     lane + vehicle     0.98      heavy rain (45%)  0.57
#:     salt noise (45%)   0.02      dead feed         0.0005
#:
#: The threshold sits between the two clusters rather than beside either: rain
#: is the lowest thing that must survive and salt noise the highest that must
#: not, and 0.25 is roughly the middle of that gap in log terms. Weather has to
#: clear it comfortably, because a lane that falls back all afternoon whenever
#: it rains is a lane nobody keeps.
#:
#: It matters that this is separate from "does it match the reference". A tight
#: plate crop does not match a lane reference either, and a caller replacing an
#: LPR unit sends nothing but tight plate crops. That caller must keep getting
#: reads; a dead feed must not.
DEFAULT_MIN_STRUCTURE = 0.25

#: How much of the reference view must still be recognisable for the comparison
#: to mean anything. The TOTAL area still matching the reference after
#: illumination is cancelled -- not the largest contiguous patch of it, because
#: heavy rain or snow fragments a perfectly recognisable lane into confetti
#: while leaving most of it matching.
DEFAULT_MIN_REFERENCE_MATCH = 0.10

#: The gain the illumination fit is allowed to reach before the capture is
#: judged not to be a view of this lane at all.
GAIN_LIMITS = (0.2, 5.0)

#: Named equipment faults, reported rather than folded into the verdict. These
#: are the reasons a lane operator has to be able to act on: every one of them
#: is something wrong with the camera or the reference, not something about a
#: vehicle.
NO_SIGNAL = "no_signal"
REFERENCE_NOT_RECOGNISED = "reference_not_recognised"
SCENE_CHANGED = "scene_changed"


@dataclass(frozen=True, slots=True)
class Presence:
    """What was measured about whether a vehicle is there.

    `present` is None when it could not be measured at all. `occupancy` is the
    fraction of the frame that differs from the reference, and it is reported
    whether the answer is yes or no -- a lane operator debugging a gate that
    refuses everything needs the number, not just the verdict.

    `camera_health` is set when the reason presence could not be measured is an
    equipment fault rather than an absent configuration. It is the difference
    between "nobody set a reference view up" and "the camera is dead", and only
    the second one is something to go and fix.
    """

    present: bool | None
    confidence: float | None = None
    occupancy: float | None = None
    reason: str = ""
    camera_health: str | None = None

    @staticmethod
    def unmeasured(reason: str, camera_health: str | None = None) -> Presence:
        return Presence(
            present=None,
            confidence=None,
            occupancy=None,
            reason=reason,
            camera_health=camera_health,
        )


class PresenceDetector:
    """Compares captures against a reference view of the empty lane."""

    def __init__(
        self,
        reference: object | None = None,
        min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
        max_occupancy: float = DEFAULT_MAX_OCCUPANCY,
        min_frame_std: float = DEFAULT_MIN_FRAME_STD,
        min_reference_match: float = DEFAULT_MIN_REFERENCE_MATCH,
        min_structure: float = DEFAULT_MIN_STRUCTURE,
        min_structural_change: float = DEFAULT_MIN_STRUCTURAL_CHANGE,
        stabiliser: float = DEFAULT_STABILISER,
        window: int = DEFAULT_WINDOW,
        min_reference_texture: float = DEFAULT_MIN_REFERENCE_TEXTURE,
    ) -> None:
        if not 0.0 < min_occupancy < max_occupancy <= 1.0:
            raise ValueError(
                "need 0 < min_occupancy < max_occupancy <= 1, got "
                f"{min_occupancy} and {max_occupancy}"
            )
        self.min_occupancy = min_occupancy
        self.max_occupancy = max_occupancy
        self.min_frame_std = min_frame_std
        self.min_reference_match = min_reference_match
        self.min_structure = min_structure
        self.min_structural_change = min_structural_change
        self.stabiliser = stabiliser
        self.window = window
        self.min_reference_texture = min_reference_texture
        self._reference = None
        #: The reference's own typical local texture, measured once when it is
        #: set. Everything the structural comparison does is scaled by it.
        self._reference_texture = 0.0
        if reference is not None:
            self.set_reference(reference)

    def set_reference(self, image) -> None:
        """Set the empty-lane view everything is compared against.

        The reference's typical local texture is measured here, once, because
        every later comparison is expressed as a fraction of it.
        """
        self._reference = _grey(image)
        self._reference_texture = _typical_local_texture(self._reference, self.window)

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

        # B1a, and it runs FIRST. A frame carrying no information must never be
        # able to produce a verdict of either kind, and the order matters in
        # both directions: the illumination fit below reads a flat frame as
        # matching the reference perfectly once gain is cancelled, which would
        # make a dead camera report a confident `False`. See the planted control
        # in tests/test_presence.py.
        if self._blank(self._reference):
            return Presence.unmeasured(
                "the configured reference view carries no detail to compare against",
                camera_health=REFERENCE_NOT_RECOGNISED,
            )
        # A reference the structural measure cannot work with. Not an equipment
        # fault and not a missing configuration -- a real view of a piece of
        # ground that carries no texture to recognise later. `camera_health`
        # stays unset for exactly that reason: there is nothing to go and fix,
        # the measure simply does not apply to this lane. NOT MEASURED, never
        # `false`: a measure that cannot see the ground cannot report that the
        # ground is empty.
        if self._reference_texture < self.min_reference_texture:
            return Presence.unmeasured(
                f"the reference view's local texture is {self._reference_texture:.2f} "
                f"grey levels, below {self.min_reference_texture}; this ground carries "
                "nothing for a structural comparison to recognise"
            )

        lit = [image for image in usable if not self._blank(image)]
        if not lit:
            return Presence.unmeasured(
                "every capture is a blank frame; the camera is not producing a picture",
                camera_health=NO_SIGNAL,
            )

        measurements = [self._occupancy(image) for image in lit]
        # A capture the illumination fit could not reconcile with the reference
        # is not evidence either way, so it does not get to vote.
        comparable = [m for m in measurements if m is not None]
        if not comparable:
            return Presence.unmeasured(
                "no capture could be reconciled with the reference view of the lane",
                camera_health=REFERENCE_NOT_RECOGNISED,
            )

        occupancy, matched = max(comparable, key=lambda m: m[0])

        # So little of the reference view is left that there is nothing to
        # compare against: a camera that was knocked, a scene rebuilt, or a
        # vehicle close enough to fill the frame. Indistinguishable from here,
        # and NOT MEASURED is the only honest answer. Never `False`.
        if matched < self.min_reference_match:
            return Presence.unmeasured(
                f"only {matched:.1%} of the reference view is still recognisable, "
                f"below {self.min_reference_match:.1%}; this capture is not a view "
                "of the lane the reference describes",
                camera_health=REFERENCE_NOT_RECOGNISED,
            )

        # The ceiling. Occupancy approaching the whole frame is evidence the
        # scene changed, not that a car is present -- and a bounded band that
        # answered `False` up here would refuse the vans and trucks that
        # legitimately fill an entry lane view.
        if occupancy >= self.max_occupancy:
            return Presence.unmeasured(
                f"{occupancy:.1%} of the frame differs from the empty lane, at or "
                f"above the {self.max_occupancy:.1%} ceiling: the scene changed "
                "rather than something arriving in it",
                camera_health=SCENE_CHANGED,
            )

        present = occupancy >= self.min_occupancy
        return Presence(
            present=present,
            confidence=self._confidence(occupancy),
            occupancy=occupancy,
            reason=(
                f"{occupancy:.1%} of the frame differs from the empty lane, "
                f"threshold {self.min_occupancy:.1%}"
            ),
        )

    def _blank(self, grey) -> bool:
        """A frame carrying no information, in either of the two ways.

        Flat -- black, white, a taped lens -- has no variance. A dead feed has
        plenty and no structure: neighbouring samples of sensor noise do not
        agree, where neighbouring pixels of any real photograph do.

        Both are equipment faults and neither can contain a vehicle, so neither
        may produce a verdict. What this deliberately does NOT catch is a real
        picture that simply is not this lane -- a plate crop, a camera that was
        knocked. Those are NOT MEASURED too, but they are still pictures, and
        the engine goes on reading them exactly as it did before this stage
        existed.
        """
        return _flat(grey, self.min_frame_std) or _unstructured(grey, self.min_structure)

    def _confidence(self, occupancy: float) -> float:
        """How far the measurement sat from the decision boundary, 0 to 1.

        Each side is scaled by its OWN distance to its extreme, so that a scene
        which is entirely unchanged and one that fills the frame both read as
        certain. The first version divided both sides by the wider of the two,
        which capped a `False` verdict at min_occupancy/(1 - min_occupancy) --
        0.176 at the default -- so an entirely empty lane reported 17.6%
        confident that it was empty while its own docstring said such a scene
        should read as certain. The arithmetic was what was wrong; the sentence
        describing it was right, and now holds.

        Continuous through the boundary: both branches reach 0 there.
        """
        if occupancy >= self.min_occupancy:
            span = self.max_occupancy - self.min_occupancy
            return min(1.0, (occupancy - self.min_occupancy) / span)
        return min(1.0, (self.min_occupancy - occupancy) / self.min_occupancy)

    def _occupancy(self, image) -> tuple[float, float] | None:
        """Changed fraction and still-matching fraction, or None if unusable.

        Returns the largest connected CHANGED region, and the total fraction of
        the frame still matching the reference. The second is what says whether
        the comparison meant anything: if almost nothing still matches, the
        capture is not a view of this lane.

        "Changed" is STRUCTURAL, not photometric. See
        `DEFAULT_MIN_STRUCTURAL_CHANGE`: a window counts as changed when it has
        stopped looking like the same piece of ground, not when it has got
        brighter or darker. That is the whole point -- a grey car on grey
        asphalt differs from the tarmac by almost nothing in intensity and by
        everything in structure.
        """
        import cv2

        reference = self._reference
        if image.shape != reference.shape:
            image = cv2.resize(image, (reference.shape[1], reference.shape[0]))

        # B1b, and it still runs. The structural terms below are invariant to a
        # global gain on their own, but the illumination fit is also the guard
        # that decides whether this capture is a view of THIS LANE at all -- a
        # tight plate crop and a knocked camera are caught by its gain limits,
        # and both must stay NOT MEASURED rather than becoming a verdict.
        fitted = _match_illumination(image, reference)
        if fitted is None:
            return None

        changed = _structurally_changed(
            fitted,
            reference,
            window=self.window,
            floor=self.stabiliser * self._reference_texture,
            threshold=self.min_structural_change,
        )

        # Contiguity is what separates a vehicle from noise, rain or a bird.
        # Scattered windows covering 40% of the frame are not a car; one blob
        # covering 20% is. Only the largest connected region counts.
        return (
            _largest_region(changed),
            float((~changed).sum()) / float(changed.size),
        )


def _local_moments(a, b, window: int):
    """Per-window means, variances and covariance of two frames.

    Box filters rather than an explicit sliding window: the same arithmetic,
    and it runs in milliseconds on a lane-sized frame.
    """
    import cv2

    size = (window, window)
    mu_a = cv2.boxFilter(a, -1, size, normalize=True)
    mu_b = cv2.boxFilter(b, -1, size, normalize=True)
    var_a = cv2.boxFilter(a * a, -1, size, normalize=True) - mu_a * mu_a
    var_b = cv2.boxFilter(b * b, -1, size, normalize=True) - mu_b * mu_b
    cov = cv2.boxFilter(a * b, -1, size, normalize=True) - mu_a * mu_b
    return var_a, var_b, cov


def _typical_local_texture(grey, window: int) -> float:
    """The median local standard deviation of a frame, in grey levels.

    The median rather than the mean: a lane view is mostly tarmac with a few
    high-contrast features -- painted lines, a kerb, a drain -- and the mean
    would report the features while the question being asked is what the
    ORDINARY window of this lane looks like.
    """
    import numpy as np

    values = grey.astype("float32")
    var, _, _ = _local_moments(values, values, window)
    return float(np.median(np.sqrt(np.maximum(var, 0.0))))


def _structurally_changed(image, reference, *, window: int, floor: float, threshold: float):
    """Which windows have stopped looking like the same piece of ground.

    The contrast and structure terms of SSIM, with the LUMINANCE term dropped.
    Dropping it is the entire point: absolute brightness is the quantity that
    must not decide, because an object at the ground's own luminance is a grey
    car on grey asphalt and it has to be seen.

        contrast  = (2*sd_r*sd_c + k) / (var_r + var_c + k)
        structure = (cov + k/2)      / (sd_r*sd_c + k/2)

    Both are 1 for a window that is unchanged and fall towards 0 -- or below it,
    for an anti-correlated window -- as it stops matching. A window is changed
    when their product has fallen by more than `threshold`.

    `k` is scaled to the reference's own typical texture rather than to the
    8-bit range. SSIM's textbook constant is calibrated for photographs; on
    ground that varies by a few grey levels it swamps everything and reports
    universal agreement, which would reproduce the blind spot this replaced.
    """
    import numpy as np

    a = reference.astype("float32")
    b = image.astype("float32")
    var_r, var_c, cov = _local_moments(a, b, window)
    var_r = np.maximum(var_r, 0.0)
    var_c = np.maximum(var_c, 0.0)
    sd_r, sd_c = np.sqrt(var_r), np.sqrt(var_c)

    k = max(float(floor) ** 2, 1e-6)
    contrast = (2.0 * sd_r * sd_c + k) / (var_r + var_c + k)
    structure = (cov + k / 2.0) / (sd_r * sd_c + k / 2.0)
    similarity = contrast * structure
    return (1.0 - similarity) >= threshold


def _largest_region(mask) -> float:
    """The largest connected region of `mask`, as a fraction of the frame."""
    import cv2
    import numpy as np

    if not mask.any():
        return 0.0
    binary = mask.astype("uint8") * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), "uint8"))
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0.0
    return float(max(stats[1:, cv2.CC_STAT_AREA])) / float(binary.size)


def _flat(grey, min_std: float) -> bool:
    """A frame with no information in it: flat black, flat white, a taped lens."""
    import numpy as np

    return bool(np.std(grey.astype("float32")) < min_std)


def _unstructured(grey, min_correlation: float) -> bool:
    """A frame that is all variance and no picture: a dead or disconnected feed.

    The correlation between the frame and itself shifted one pixel across. Any
    real scene is smooth at that scale and scores near 1; uniform sensor noise
    is independent sample to sample and scores near 0.
    """
    import numpy as np

    values = grey.astype("float32")
    if values.shape[1] < 2:
        return False
    left, right = values[:, :-1].ravel(), values[:, 1:].ravel()
    if np.std(left) < 1e-6 or np.std(right) < 1e-6:
        return False  # flat; _flat is the check that owns that case
    correlation = float(np.corrcoef(left, right)[0, 1])
    return not np.isfinite(correlation) or correlation < min_correlation


def _match_illumination(image, reference):
    """`image` rescaled so its light matches the reference's, or None.

    Fits one gain and one offset over the whole frame, refitting on the closest
    60% so that a vehicle in the picture does not drag the fit onto itself. What
    survives is a change in the SCENE; a change in the LIGHT cancels.

    None when no plausible fit exists -- the capture is then not a view of this
    lane, which is NOT MEASURED rather than either verdict.
    """
    import numpy as np

    # A fixed stride rather than a random sample: the same capture must produce
    # the same answer every time it is measured.
    x = reference.astype("float64").ravel()[::7]
    y = image.astype("float64").ravel()[::7]
    keep = np.ones(x.size, dtype=bool)
    gain = offset = 0.0
    for _ in range(3):
        xs, ys = x[keep], y[keep]
        if xs.size < 32 or np.std(xs) < 1e-6:
            return None
        gain, offset = np.polyfit(xs, ys, 1)
        residual = np.abs(gain * x + offset - y)
        keep = residual <= max(float(np.quantile(residual, 0.6)), 1e-9)

    low, high = GAIN_LIMITS
    if not np.isfinite(gain) or not np.isfinite(offset) or not low <= gain <= high:
        return None
    # Undo the fitted light on the CAPTURE, so the comparison happens in the
    # reference's own exposure and `pixel_delta` keeps meaning grey levels of
    # the reference.
    corrected = (image.astype("float32") - offset) / gain
    return np.clip(corrected, 0, 255).astype("uint8")


def _grey(image):
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array
