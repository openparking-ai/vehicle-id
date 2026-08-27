"""The scenes the presence gate is measured against.

One definition, imported by both the tests and `scripts/eval_presence.py`, so
that a published number and the test that guards it cannot end up describing
different pictures.

Two things about these fixtures are load-bearing, and round 2 was cracked
because the first of them was missing.

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
GRAIN = 0.035


def _aggregate(texture: float):
    """The tarmac's own stone texture. Fixed: it is in the reference and in
    every capture of it, because it is the ground rather than the sensor."""
    rng = np.random.default_rng(20260827)
    fine = rng.normal(0, 1.0, (H, W)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 0.8)
    fine /= max(float(np.std(fine)), 1e-6)
    return fine * AGGREGATE * texture


def lane(level: float = 90, seed: int = 1, texture: float = 1.0):
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
    """
    rng = np.random.default_rng(seed)
    image = np.full((H, W), 1.0, np.float32) * level
    image += np.linspace(-8, 8, H, dtype=np.float32)[:, None]
    image += _aggregate(texture) * (level / 90.0)
    cv2.rectangle(image, (40, 0), (52, H), level * 2.0, -1)
    cv2.rectangle(image, (W - 52, 0), (W - 40, H), level * 2.0, -1)
    cv2.circle(image, (320, 300), 26, level * 0.45, -1)
    cv2.rectangle(image, (0, 0), (W, 26), level * 0.7, -1)
    image += rng.normal(0, level * GRAIN, (H, W)).astype(np.float32)
    return cv2.merge([np.clip(image, 0, 255).astype(np.uint8)] * 3)


def vehicle(
    width: int,
    height: int,
    level: float = 90,
    seed: int = 1,
    contrast: float = 2.05,
    texture: float = 1.0,
    surface: float = 0.0,
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

    The object has no internal detail -- no glass, no wheel arches, no shadow.
    A real vehicle has all three and is therefore EASIER than this. Every number
    measured on these scenes is a lower bound on a real one, and none of them is
    a substitute for footage.
    """
    scene = lane(level, seed, texture=texture).copy()
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


def flat(level: int):
    """A frame carrying no information: a dead sensor, or a taped-over lens."""
    return np.full((H, W, 3), level, np.uint8)


def sensor_noise(seed: int = 5):
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def rain(coverage: float = 0.45, seed: int = 7, texture: float = 1.0):
    """Heavy rain or snow: a large fraction of the frame changes, scattered.

    Streaks rather than salt. Weather is made of drops that occupy several
    pixels each and fall in lines, so a rained-on frame is still a PICTURE --
    neighbouring pixels still agree. Randomising every other pixel instead
    produces something no camera has ever emitted, and it lands in the
    dead-feed case rather than the weather case, which is a different test.
    """
    base = lane(90, seed=seed, texture=texture)
    scene = base.copy()
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
#   per-window structural change        OBJECT/GROUND CONTRAST     -> `contrast`
#                                       GROUND TEXTURE ENERGY      -> `texture`
#                                       OBJECT SURFACE GRAIN       -> `surface`
#   largest connected changed region    object size                -> `width`,
#                                       `height`; scatter -> `rain`
#   matched vs min_reference_match      how much still matches     -> big objects
#   occupancy vs min/max_occupancy      object size                -> `width`
#   which capture in a burst decides    burst composition          -> lists
#
# `contrast`, `texture` and `surface` are the three the old fixture could not
# express at all. They are the ones the measure now turns on.

#: Swept through 1.0 from both sides. 1.0 is the object at exactly the ground's
#: luminance; 0.78 and 1.22 sit inside the +/-30-grey-level band that the
#: intensity measure was blind to at a reference level of 90.
CONTRASTS = (0.35, 0.55, 0.78, 0.9, 1.0, 1.1, 1.22, 1.5, 2.05)

#: New blacktop through coarse chip seal. 0.25 is a garage whose ground gives a
#: structural measure very little to work with.
TEXTURES = (0.25, 1.0, 2.0)

#: A featureless panel, and a body with its own grain.
SURFACES = (0.0, 0.02)

#: Comfortably above the 15% floor and below the 90% ceiling, so that the
#: verdict turns on the MEASURE rather than on the object being the wrong size.
VEHICLE_SIZE = (420, 240)


def matrix():
    """Every (contrast, texture, surface) cell, with the empty lane beside it.

    Each cell carries both scenes deliberately. A measure that answered `true`
    everywhere would pass a vehicle-only sweep perfectly, and a measure that
    answered `false` everywhere would pass an empty-lane-only sweep perfectly.
    Only the pair says anything.
    """
    cells = []
    index = 0
    for texture in TEXTURES:
        for contrast in CONTRASTS:
            for surface in SURFACES:
                index += 1
                cells.append(
                    {
                        "contrast": contrast,
                        "texture": texture,
                        "surface": surface,
                        "vehicle": vehicle(
                            *VEHICLE_SIZE,
                            seed=1000 + index,
                            contrast=contrast,
                            texture=texture,
                            surface=surface,
                        ),
                        "empty": lane(90, seed=2000 + index, texture=texture),
                    }
                )
    return cells
