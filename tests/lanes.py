"""The scenes the presence gate is measured against.

One definition, imported by both the tests and `scripts/eval_presence.py`, so
that a published number and the test that guards it cannot end up describing
different pictures.
"""

from __future__ import annotations

import cv2
import numpy as np

H, W = 360, 640


def lane(level: float = 90, seed: int = 1):
    """An entry-lane view: tarmac, two painted edge lines, a drain, a kerb.

    Structure, not just a tinted rectangle. A reference that is flat grey with a
    little noise is a degenerate case any difference lights up -- and a
    near-degenerate one is what made the first version of this gate look like it
    worked, because raw intensity differencing has nothing else to go on.

    `level` is the light. The same lane at a different `level` is the same
    scene under a different exposure, which is what a lane looks like at dusk.
    """
    rng = np.random.default_rng(seed)
    image = np.full((H, W), 1.0, np.float32) * level
    image += np.linspace(-8, 8, H, dtype=np.float32)[:, None]
    cv2.rectangle(image, (40, 0), (52, H), level * 2.0, -1)
    cv2.rectangle(image, (W - 52, 0), (W - 40, H), level * 2.0, -1)
    cv2.circle(image, (320, 300), 26, level * 0.45, -1)
    cv2.rectangle(image, (0, 0), (W, 26), level * 0.7, -1)
    image += rng.normal(0, level * 0.035, (H, W)).astype(np.float32)
    return cv2.merge([np.clip(image, 0, 255).astype(np.uint8)] * 3)


def vehicle(width: int, height: int, level: float = 90, seed: int = 1):
    """The empty lane with one solid object of a given size on it.

    The object's brightness is a RATIO of the light, not a constant. A fixed
    grey makes the object vanish into the tarmac at high exposures, which is a
    property of the fixture rather than of the gate -- an object genuinely the
    same brightness as the ground it sits on is invisible to anything that
    compares intensity, and a test built that way measures the fixture.
    """
    scene = lane(level, seed).copy()
    x0, y0 = (W - width) // 2, (H - height) // 2
    shade = float(min(255.0, level * 2.05))
    cv2.rectangle(scene, (x0, y0), (x0 + width, y0 + height), (shade,) * 3, -1)
    return scene


def flat(level: int):
    """A frame carrying no information: a dead sensor, or a taped-over lens."""
    return np.full((H, W, 3), level, np.uint8)


def sensor_noise(seed: int = 5):
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def rain(coverage: float = 0.45, seed: int = 7):
    """Heavy rain or snow: a large fraction of the frame changes, scattered.

    Streaks rather than salt. Weather is made of drops that occupy several
    pixels each and fall in lines, so a rained-on frame is still a PICTURE --
    neighbouring pixels still agree. Randomising every other pixel instead
    produces something no camera has ever emitted, and it lands in the
    dead-feed case rather than the weather case, which is a different test.
    """
    base = lane(90, seed=seed)
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
