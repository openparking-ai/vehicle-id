"""The scenes the presence gate is measured against.

One definition, imported by both the tests and `scripts/eval_presence.py`, so
that a published number and the test that guards it cannot end up describing
different pictures.

Four things about these fixtures are load-bearing, and each of them was missing
in the round that made it necessary.

**Contrast is an axis, not a constant.** `vehicle()` used to compute its
object's brightness as a fixed `level * 2.05`, so every one of its call sites
sampled the same point on the only quantity that decides `false`. A gate keyed
on intensity distance is blind to an object at the same luminance as the ground,
and a fixture that cannot express one cannot find that out. `contrast` is now a
parameter with no default hiding inside the body, and `matrix()` sweeps it
through 1.0 -- the exactly-invisible case -- from both sides.

**Tarmac texture is STATIONARY; sensor grain is NOT.** Real asphalt has a fixed
stone texture that is in the reference view and in every capture of it, pixel
for pixel. Sensor grain is a fresh draw every frame. The old `lane()` drew both
from `seed`, so the only fine detail in the picture was per-frame noise, and a
plain patch of tarmac carried nothing a structural measure could recognise
across two frames. That is a property of the fixture, not of tarmac, and it
would have made any local-structure measure look worse than it is. The stone
texture is now drawn from a fixed generator and `seed` moves only the grain.

**Grain is its OWN axis, decoupled from texture.** It was not, and the
consequence was that an axis could not reach the code path it existed to test.
`GRAIN` is 3.5% of the light level -- 3.15 grey levels at level 90 -- and the
detector's `min_reference_texture` floor is 1.5. With grain welded on, the
measured reference texture never fell below 3.28 however far `texture` was
wound down (0.10 -> 3.28, 0.25 -> 3.82), so the fixture could not express ground
smoother than its own sensor noise and the NOT-MEASURED-on-untextured-ground
branch was never once exercised by the matrix. `grain=` is now a parameter that
can go to zero. `smooth_floor()` below is the planted control that reaches the
branch.

**Weather and headlights take a SCENE, not just a lane.** `rain()` used to build
its own empty lane and had no way to put a car in the picture, so the safety
claim "no wrongful refusal in weather" was published over a sweep in which no
frame ever contained a vehicle. Both now accept a `base` scene, so the vehicle
case is measured rather than assumed.

`texture=` scales the stationary component, so "new smooth blacktop" and "coarse
chip seal" are both expressible and both get measured.
"""

from __future__ import annotations

import cv2
import numpy as np

H, W = 360, 640

#: The stationary stone texture of the tarmac, in grey levels of contrast at
#: `texture=1.0`. Asphalt aggregate is coarse; a lane camera sees it.
AGGREGATE = 9.0

#: Per-frame sensor grain, as a fraction of the light level. Not stationary --
#: a fresh draw every capture, which is what makes it noise rather than scene.
#:
#: A DEFAULT, not a constant: it is a parameter of every scene below precisely
#: so that it can be wound to zero. At the default it is 3.15 grey levels at
#: level 90, which is above the detector's 1.5 texture floor all on its own --
#: so a fixture that could not switch it off could not express ground the
#: measure is unable to serve, and could not reach the branch that says so.
GRAIN = 0.035

#: Where a car's beams land on the floor of a covered entry, as a fraction of
#: the frame. Centred low and ahead: the camera looks down the lane, and the
#: pool arrives before the car does.
BEAM_CENTRE = (0.50, 0.72)
BEAM_SPREAD = (0.30, 0.28)


def _aggregate(texture: float):
    """The tarmac's own stone texture. Fixed: it is in the reference and in
    every capture of it, because it is the ground rather than the sensor."""
    rng = np.random.default_rng(20260827)
    fine = rng.normal(0, 1.0, (H, W)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 0.8)
    fine /= max(float(np.std(fine)), 1e-6)
    return fine * AGGREGATE * texture


def _beam(amount: float):
    """A headlight pool as a MULTIPLIER on the scene, not an addition to it.

    This is the modelling decision the axis stands on, so it is stated rather
    than buried. A headlight adds illumination to a matte floor, and the light
    that comes back is `(ambient + beam) * albedo`. The floor's own texture is
    therefore still in the picture, scaled up -- which is exactly why a pool is
    not automatically a vehicle to a structural measure, and why the question is
    worth asking at all. Adding a constant instead would model a glowing
    rectangle lying on the ground, which is not what a headlight is.

    What this does NOT model, stated so nobody reads more into the numbers than
    is in them: specular glare off a wet or polished floor, the beam's own
    cut-off line, the vehicle's lit front face, and colour. A gloss floor at
    night is a specular scene and this is a matte one. NOT MEASURED.
    """
    cx, cy = BEAM_CENTRE[0] * W, BEAM_CENTRE[1] * H
    sx, sy = BEAM_SPREAD[0] * W, BEAM_SPREAD[1] * H
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    fall = np.exp(-(((x - cx) ** 2) / (2 * sx * sx) + ((y - cy) ** 2) / (2 * sy * sy)))
    return 1.0 + amount * fall


def lane(
    level: float = 90,
    seed: int = 1,
    texture: float = 1.0,
    grain: float = GRAIN,
    headlight: float = 0.0,
):
    """An entry-lane view: tarmac, two painted edge lines, a drain, a kerb.

    Structure, not just a tinted rectangle. A reference that is flat grey with a
    little noise is a degenerate case any difference lights up -- and a
    near-degenerate one is what made the first version of this gate look like it
    worked, because raw intensity differencing has nothing else to go on.

    `level` is the light. The same lane at a different `level` is the same
    scene under a different exposure, which is what a lane looks like at dusk.

    `texture` scales the stationary stone texture of the tarmac. It is an axis
    because how much fine detail the ground carries is exactly what decides
    whether a structural measure has anything to work with, and a garage with
    new blacktop has less of it than one with chip seal.

    `grain` is the per-frame sensor noise, and it is a SEPARATE axis from
    `texture` because the two are separate things: one is the ground and one is
    the camera. Wound to zero, `texture` can express a floor the measure cannot
    serve; welded on, it could not.

    `headlight` is the peak of a beam pool on the floor, as a multiple of the
    ambient light there. A covered entry is artificially lit and often dark, and
    an approaching car throws its beams into frame BEFORE the car itself
    arrives -- a large scene change caused by a vehicle that is not yet the
    vehicle. See `_beam` for what the model does and does not include.
    """
    rng = np.random.default_rng(seed)
    image = np.full((H, W), 1.0, np.float32) * level
    image += np.linspace(-8, 8, H, dtype=np.float32)[:, None]
    image += _aggregate(texture) * (level / 90.0)
    cv2.rectangle(image, (40, 0), (52, H), level * 2.0, -1)
    cv2.rectangle(image, (W - 52, 0), (W - 40, H), level * 2.0, -1)
    cv2.circle(image, (320, 300), 26, level * 0.45, -1)
    cv2.rectangle(image, (0, 0), (W, 26), level * 0.7, -1)
    if headlight > 0:
        # Applied to the SCENE, before the sensor's own noise: the beam lights
        # the ground, it does not light the sensor.
        image *= _beam(headlight)
    if grain > 0:
        image += rng.normal(0, level * grain, (H, W)).astype(np.float32)
    return cv2.merge([np.clip(image, 0, 255).astype(np.uint8)] * 3)


def vehicle(
    width: int,
    height: int,
    level: float = 90,
    seed: int = 1,
    contrast: float = 2.05,
    texture: float = 1.0,
    surface: float = 0.0,
    grain: float = GRAIN,
    headlight: float = 0.0,
):
    """The empty lane with one solid object of a given size on it.

    `contrast` is the object's brightness as a RATIO of the light: 2.05 is a
    pale car against dark tarmac, 0.5 a dark one, and **1.0 is an object that
    reflects exactly as much light as the ground it sits on**. That last case is
    invisible to anything comparing intensity magnitude, it is a perfectly
    ordinary grey car on grey asphalt, and it is the case that has to be
    measured rather than assumed away.

    `surface` is the object's OWN grain, as a fraction of its level -- a car
    body is not mathematically flat. Zero is the hardest case for a measure that
    looks at texture, so it is the default: a featureless panel occluding the
    ground, carrying no help of its own.

    `headlight` lights the GROUND and not the object, which is the right way
    round: a car's own beams fall in front of it and do not illuminate its front
    face. The object is drawn after the pool for exactly that reason.

    The object has no internal detail -- no glass, no wheel arches, no shadow.
    A real vehicle has all three and is therefore EASIER than this. Every number
    measured on these scenes is a lower bound on a real one, and none of them is
    a substitute for footage.
    """
    scene = lane(level, seed, texture=texture, grain=grain, headlight=headlight).copy()
    x0, y0 = (W - width) // 2, (H - height) // 2
    shade = float(np.clip(level * contrast, 0, 255))
    cv2.rectangle(scene, (x0, y0), (x0 + width, y0 + height), (shade,) * 3, -1)
    if surface > 0:
        rng = np.random.default_rng(seed + 991)
        patch = scene[y0 : y0 + height, x0 : x0 + width, 0].astype(np.float32)
        patch += rng.normal(0, max(shade, 1.0) * surface, patch.shape).astype(np.float32)
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        scene[y0 : y0 + height, x0 : x0 + width] = cv2.merge([patch] * 3)
    return scene


def smooth_floor(level: float = 120, seed: int = 3):
    """Sealed or painted concrete under a clean sensor: ground with nothing on
    it for a structural measure to recognise.

    The planted control for the texture floor, and the reason `grain` had to
    become an axis. `lane(texture=0.1)` still measures 3.28 grey levels of local
    texture because the sensor's own grain is 3.15 of them, so no value of
    `texture` alone could ever fall under the detector's 1.5 floor. This scene
    can, because it carries almost no grain -- and a smooth sealed floor under a
    good sensor really is that scene.

    Per M2: this is ground a covered entry can have, and the honest answer for
    it is NOT MEASURED. How often it occurs is a second measurement nobody has
    made, and none is claimed.
    """
    plain = np.full((H, W), float(level), np.float32)
    plain += np.linspace(-6, 6, H, dtype=np.float32)[:, None]
    cv2.rectangle(plain, (0, 0), (W, 20), level * 0.85, -1)
    plain += np.random.default_rng(seed).normal(0, 0.6, (H, W)).astype(np.float32)
    return cv2.merge([np.clip(plain, 0, 255).astype(np.uint8)] * 3)


def flat(level: int):
    """A frame carrying no information: a dead sensor, or a taped-over lens."""
    return np.full((H, W, 3), level, np.uint8)


def sensor_noise(seed: int = 5):
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def rain(coverage: float = 0.45, seed: int = 7, texture: float = 1.0, base=None):
    """Heavy rain or snow: a large fraction of the frame changes, scattered.

    Streaks rather than salt. Weather is made of drops that occupy several
    pixels each and fall in lines, so a rained-on frame is still a PICTURE --
    neighbouring pixels still agree. Randomising every other pixel instead
    produces something no camera has ever emitted, and it lands in the
    dead-feed case rather than the weather case, which is a different test.

    `base` is the scene the weather falls on, and it defaults to the empty lane.
    It exists because the published claim "no wrongful refusal in weather" was
    measured over a sweep in which no frame ever contained a vehicle: `rain()`
    built its own empty lane and there was no way to pass one in. Pass a
    `vehicle()` scene and the claim is measured instead of asserted.
    """
    scene = (lane(90, seed=seed, texture=texture) if base is None else base).copy()
    rng = np.random.default_rng(seed + 1)
    # Drawn until the fraction of the frame actually covered reaches the target,
    # rather than until a guessed number of streaks have been drawn. Streaks
    # overlap, so the two are not the same number and only one of them is what
    # the test says it is testing.
    wet = np.zeros((H, W), bool)
    while wet.mean() < coverage:
        for _ in range(200):
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H - 12))
            end = (x + int(rng.integers(-2, 3)), y + 12)
            cv2.line(scene, (x, y), end, (245,) * 3, 2)
            cv2.line(wet.view(np.uint8), (x, y), end, 1, 2)
    return scene


def dead_sensor(fraction: float = 0.45, seed: int = 7):
    """Salt noise over the lane: a sensor failing, not weather."""
    scene = lane(90, seed=seed).copy()
    mask = np.random.default_rng(seed + 1).random((H, W)) < fraction
    scene[mask] = 255
    return scene


# --- the matrix ----------------------------------------------------------
#
# G2b: the axes are read off the DECISION, not imagined. Every quantity
# `PresenceDetector.measure` branches on, and the scene property that moves it:
#
#   decision branch                     scene axis that moves it
#   ---------------------------------   ---------------------------------------
#   _flat / _unstructured               frame information content  -> `flat`,
#                                       `sensor_noise`, `dead_sensor`
#   illumination fit / GAIN_LIMITS      light level                -> `level`
#   is this a view of this lane         view match                 -> plate crop
#   reference texture vs the floor      GROUND SMOOTHNESS          -> `grain`
#                                       plus `texture` -> `smooth_floor`
#   per-window structural change        OBJECT/GROUND CONTRAST     -> `contrast`
#                                       GROUND TEXTURE ENERGY      -> `texture`
#                                       OBJECT SURFACE GRAIN       -> `surface`
#                                       LOCAL ILLUMINATION         -> `headlight`
#   largest connected changed region    object size                -> `width`,
#                                       `height`; scatter -> `rain`
#   matched vs min_reference_match      how much still matches     -> big objects
#   occupancy vs min/max_occupancy      object size                -> `width`
#   which capture in a burst decides    burst composition          -> lists
#
# `contrast`, `texture` and `surface` are the three the old fixture could not
# express at all. `headlight` and `grain` are the two the round after that
# could not: one because a covered entry is lit artificially and nothing
# modelled it, one because it was welded to `texture` and so an axis could not
# reach the branch it existed to test.

#: Swept through 1.0 from both sides. 1.0 is the object at exactly the ground's
#: luminance; 0.78 and 1.22 sit inside the +/-30-grey-level band that the
#: intensity measure was blind to at a reference level of 90.
CONTRASTS = (0.35, 0.55, 0.78, 0.9, 1.0, 1.1, 1.22, 1.5, 2.05)

#: New blacktop through coarse chip seal. 0.25 is a garage whose ground gives a
#: structural measure very little to work with. Ground smoother than this is
#: `smooth_floor()`, because no value here can reach it -- see `GRAIN`.
TEXTURES = (0.25, 1.0, 2.0)

#: A featureless panel, and a body with its own grain.
SURFACES = (0.0, 0.02)

#: Beams off, and a pool at twice the ambient light on the floor. The sweep that
#: finds where a pool stops being tolerated is `headlight_sweep()`; this axis is
#: here so that every other cell of the matrix is measured with the lights on as
#: well as off, which is the state a covered entry is actually in.
HEADLIGHTS = (0.0, 2.0)

#: Comfortably above the 15% floor and below the 90% ceiling, so that the
#: verdict turns on the MEASURE rather than on the object being the wrong size.
VEHICLE_SIZE = (420, 240)

#: Where a beam pool stops being tolerated. Swept as its own sweep rather than
#: as a matrix axis for the same reason weather is: the interesting output is a
#: BOUNDARY, and a boundary needs samples along it, not two points.
HEADLIGHT_LEVELS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def matrix():
    """Every (contrast, texture, surface, headlight) cell, with the empty lane beside it.

    Each cell carries both scenes deliberately. A measure that answered `true`
    everywhere would pass a vehicle-only sweep perfectly, and a measure that
    answered `false` everywhere would pass an empty-lane-only sweep perfectly.
    Only the pair says anything.

    The empty scene of a headlight cell is a lane with a beam pool on it and NO
    car in frame. That is not an idle scene: it is the second before a car
    arrives, and a gate that calls it occupied transacts for a vehicle that is
    not there yet. It belongs beside the vehicle case, not in a footnote.
    """
    cells = []
    index = 0
    for texture in TEXTURES:
        for headlight in HEADLIGHTS:
            for contrast in CONTRASTS:
                for surface in SURFACES:
                    index += 1
                    cells.append(
                        {
                            "contrast": contrast,
                            "texture": texture,
                            "surface": surface,
                            "headlight": headlight,
                            "vehicle": vehicle(
                                *VEHICLE_SIZE,
                                seed=1000 + index,
                                contrast=contrast,
                                texture=texture,
                                surface=surface,
                                headlight=headlight,
                            ),
                            "empty": lane(
                                90,
                                seed=2000 + index,
                                texture=texture,
                                headlight=headlight,
                            ),
                        }
                    )
    return cells
