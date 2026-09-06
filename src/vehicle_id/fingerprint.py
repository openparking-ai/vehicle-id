"""The appearance descriptor, and a matcher that publishes a DISTANCE.

A plate is one component of a vehicle's identity, not the identity. A garage may
run on plate alone, on appearance alone, on a QR code alone, or on any
combination -- and plate recording is mandatory nowhere. This module produces the
appearance component: an opaque, versioned, compact record computed from ONE
capture, and a comparison between two of them.

Four rules, and each one is enforced here rather than documented and hoped for.

  * **It is CLASSICAL, and the permitted list is short.** SIFT (patent expired
    2020-03-06, in OpenCV's main `features2d`), ORB, `cv2.compareHist` and Canny.
    No trained model, no downloaded weights, no dataset -- so there is no licence
    on the descriptor and nothing to fetch at runtime. What is forbidden, and
    why, is asserted in `tests/test_fingerprint_dependencies.py`: SURF (patent
    live to 2029-04-13), `xfeatures2d`'s trained descriptors (VGG, BoostDesc,
    BEBLID, TEBLID -- they download weights), and pHash.org's `libpHash` (GPLv3
    with a paid commercial exception). AKAZE, BRISK and KAZE are not forbidden;
    they are simply ABSENT from `cv2` 5.x, which is what `opencv-python-headless`
    resolves to, so nothing here may depend on them.

  * **It is not an image.** What it carries is a bounded set of keypoint
    descriptors, a colour histogram and a coarse edge grid. A photograph cannot
    be reconstructed from it, and nothing in this package writes one to disk.

  * **It carries the version of the code that produced it, and two descriptors
    of different versions REFUSE to compare.** They do not compare "carefully",
    or fall back to whichever terms both understand. The version is in the
    string's prefix, so the refusal happens before anything is decoded.

  * **`compare` publishes a distance per term, never a verdict.** No boolean
    crosses this module without the number that justified it. Where a threshold
    is wanted, `choose_operating_point` takes its constraints explicitly and
    REFUSES, naming the conflict, when no threshold satisfies them.

And one thing that is a measurement rather than a convenience: a comparison has
THREE outcomes, not two. A frame with too little texture yields no keypoints, and
"there was nothing to match" is a different answer from "nothing matched". Every
term reports `measurable` separately from its value, and a term that could not be
measured carries `None` and a reason. Collapsing the two would count an
unmeasurable same-car pair as a miss and an unmeasurable different-car pair as a
correct reject -- flattering the exact number this exists to produce.
"""

from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass

import cv2
import numpy as np

#: The DESCRIPTOR's own version, and it is not `contract.SCHEMA_VERSION`.
#:
#: These two version each other's business deliberately: adding this field to
#: the record is an ADDITIVE contract change and does not move the schema
#: version, while a change to how the bytes below are computed makes every
#: stored descriptor incomparable and must move THIS one. A single number could
#: not say both things.
DESCRIPTOR_VERSION = 1

#: The string's prefix. Present so a consumer can refuse a descriptor it does not
#: understand without decoding it -- and so a descriptor is recognisable as one
#: in a log or a database column.
DESCRIPTOR_PREFIX = "opvid-fp"

ORB = "orb"
SIFT = "sift"
KINDS = (ORB, SIFT)

#: Keypoints kept, per kind. ORB's descriptor is 32 bytes and SIFT's is 128, so
#: the caps differ to keep the encoded record within the same order of size:
#: ORB ~8 KiB of keypoints, SIFT ~16 KiB, before compression and base64.
MAX_KEYPOINTS = {ORB: 256, SIFT: 128}

#: Below this many keypoints on EITHER side, the structure term is UNMEASURABLE
#: rather than distant. A plain van filling the crop, a night arrival, a wet
#: windscreen: there is nothing to match, which is not the same as not matching.
#: It is also the floor that makes `knnMatch(k=2)` well defined.
MEASURABLE_MIN_KEYPOINTS = 8

#: The colour histogram's bins, in HSV. Hue gets the most because it is the axis
#: that survives a lighting change; value gets the fewest because it is the axis
#: that does not.
HIST_BINS = (12, 8, 4)

#: The coarse geometry term: edge density over a GRID_SIDE x GRID_SIDE grid.
#: Coarse on purpose -- this is meant to say "the mass is arranged like this",
#: not to localise anything.
GRID_SIDE = 8

#: Every image is resized so its longer side is this, preserving aspect. Bounds
#: the cost and stops the descriptor depending on how many pixels the camera
#: happened to have.
WORK_LONG_SIDE = 512

#: Lowe's ratio. A match is kept when the best neighbour is clearly better than
#: the second best; without it, every keypoint matches something.
LOWE_RATIO = 0.75

_MAGIC = b"OPFP"
_HEADER = "!4sBBHHBBBB"
_HEADER_SIZE = struct.calcsize(_HEADER)
_KIND_CODES = {ORB: 1, SIFT: 2}
_CODE_KINDS = {code: kind for kind, code in _KIND_CODES.items()}


# --- the terms -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Term:
    """One published distance, and the scale it is on.

    The scale is part of the answer. Three of these are bounded and one is not,
    and one of them is a NATIVE similarity that has been converted -- a reader
    handed five numbers with no scale beside them cannot tell which. So the
    table is data, the harness reads it rather than restating it, and nothing
    downstream has to know that `cv2.HISTCMP_INTERSECT` counts the other way.
    """

    name: str
    units: str
    lower_bound: float
    upper_bound: float | None
    note: str = ""


#: Every published term is a DISTANCE: lower means more alike, always. That is
#: asserted, not just written down -- see `test_every_published_term_is_a_distance`.
TERMS: tuple[Term, ...] = (
    Term(
        name="structure",
        units="1 - (matches both ways) / (2 * min(keypoints_a, keypoints_b))",
        lower_bound=0.0,
        upper_bound=1.0,
        note=(
            "Symmetric by construction: matched both directions and averaged, "
            "because a ratio test in one direction is not the same in the other. "
            "The raw match distance -- Hamming 0-256 for ORB, L2 for SIFT -- is "
            "NOT published: it is on a different scale per kind, and a table "
            "mixing the two would be an artefact of the detector."
        ),
    ),
    Term(
        name="colour_bhattacharyya",
        units="cv2.HISTCMP_BHATTACHARYYA over the L1-normalised HSV histogram",
        lower_bound=0.0,
        upper_bound=1.0,
    ),
    Term(
        name="colour_chisqr",
        units="cv2.HISTCMP_CHISQR_ALT over the L1-normalised HSV histogram",
        lower_bound=0.0,
        upper_bound=None,
        note=(
            "UNBOUNDED above. It shares no scale with any other term here. "
            "The ALT form, and that is a correctness fix rather than a "
            "preference: plain HISTCMP_CHISQR divides by the FIRST histogram "
            "only, so d(a, b) and d(b, a) are different numbers -- measured at "
            "126.84 and 0.99 on one synthetic pair. A distance between two "
            "vehicles that depends on which one you called `a` makes an exit "
            "lookup depend on the order the candidates came out of the "
            "database. The ALT form divides by their sum and is symmetric."
        ),
    ),
    Term(
        name="colour_intersection",
        units="1 - cv2.HISTCMP_INTERSECT over the L1-normalised HSV histogram",
        lower_bound=0.0,
        upper_bound=1.0,
        note=(
            "CONVERTED. Intersection is a similarity -- higher is closer -- and "
            "publishing it beside four distances is how a comparison acquires "
            "two opposite signs and nobody notices."
        ),
    ),
    Term(
        name="geometry",
        units="mean |difference| of the Canny edge-density grid, per cell",
        lower_bound=0.0,
        upper_bound=1.0,
    ),
)

TERM_NAMES = tuple(term.name for term in TERMS)


@dataclass(frozen=True, slots=True)
class TermDistance:
    """One term's answer, and whether there was an answer to give.

    `value` is `None` exactly when `measurable` is false, and then `reason` says
    what was missing. A caller that reads `value or 0.0` has turned "could not
    measure" into "identical", which is the worst of the three outcomes to
    invent.
    """

    term: str
    value: float | None
    measurable: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.measurable and self.value is None:
            raise ValueError(f"{self.term}: measurable with no value")
        if not self.measurable and self.value is not None:
            raise ValueError(f"{self.term}: unmeasurable but carries {self.value!r}")
        if not self.measurable and not self.reason:
            raise ValueError(f"{self.term}: unmeasurable with no reason given")


@dataclass(frozen=True, slots=True)
class Comparison:
    """What `compare` returns: one `TermDistance` per term, and no verdict."""

    kind: str
    version: int
    terms: tuple[TermDistance, ...]

    def __getitem__(self, name: str) -> TermDistance:
        for term in self.terms:
            if term.term == name:
                return term
        raise KeyError(name)

    @property
    def measurable_terms(self) -> tuple[str, ...]:
        return tuple(t.term for t in self.terms if t.measurable)


# --- the descriptor ------------------------------------------------------


class IncomparableDescriptors(ValueError):
    """Raised rather than comparing two descriptors that do not compare."""


class MalformedDescriptor(ValueError):
    """Raised rather than half-reading a descriptor string."""


@dataclass(frozen=True, slots=True)
class Descriptor:
    """The decoded record. Opaque to everything but this module.

    Held as arrays rather than as the string because comparing is what it is
    for; `text` is the form that travels and is stored. The two are kept exactly
    in step by `compute`, which encodes and then decodes its own payload, so the
    descriptor compared in memory is byte-for-byte the descriptor written down.
    Without that, quantisation would make a stored descriptor score differently
    from the one that produced it, and only in the third decimal place.
    """

    version: int
    kind: str
    keypoints: np.ndarray  # (n, dim) uint8; n may be 0
    histogram: np.ndarray  # (bins,) float32, L1-normalised
    grid: np.ndarray  # (GRID_SIDE, GRID_SIDE) uint8
    text: str

    @property
    def keypoint_count(self) -> int:
        return int(self.keypoints.shape[0])


def descriptor_version_of(text: str) -> int:
    """The version in the prefix, WITHOUT decoding anything else.

    Deliberately cheap and deliberately first: a consumer holding a descriptor
    from a future build must be able to refuse it without trusting a single byte
    of the payload it does not understand.
    """
    if not isinstance(text, str):
        raise MalformedDescriptor(f"a descriptor is a string, got {type(text).__name__}")
    prefix, _, _ = text.partition(":")
    name, _, version = prefix.partition("/")
    if name != DESCRIPTOR_PREFIX or not version.isdigit():
        raise MalformedDescriptor(f"not a {DESCRIPTOR_PREFIX} descriptor: {prefix!r}")
    return int(version)


def _to_working_image(image: np.ndarray) -> np.ndarray:
    if image is None or getattr(image, "size", 0) == 0:
        raise MalformedDescriptor("no image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > WORK_LONG_SIDE:
        scale = WORK_LONG_SIDE / longest
        # AREA on the way down: it averages rather than samples, so the colour
        # histogram of a big frame and of the same frame at half the resolution
        # are the same histogram.
        image = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    return image


def _detector(kind: str):
    if kind == ORB:
        return cv2.ORB_create(nfeatures=MAX_KEYPOINTS[ORB])
    if kind == SIFT:
        return cv2.SIFT_create(nfeatures=MAX_KEYPOINTS[SIFT])
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def _keypoints(image: np.ndarray, kind: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = _detector(kind).detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((0, 32 if kind == ORB else 128), np.uint8)
    kept = descriptors[: MAX_KEYPOINTS[kind]]
    if kind == SIFT:
        # SIFT's descriptor is float32 over roughly [0, 255] by construction.
        # Quantising to uint8 is what keeps the record compact, and it is stated
        # rather than hidden: the comparison happens on the quantised values, on
        # both sides, always.
        kept = np.clip(np.rint(kept), 0, 255)
    return np.ascontiguousarray(kept.astype(np.uint8))


def _histogram(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256])
    flat = hist.flatten().astype(np.float32)
    total = float(flat.sum())
    if total <= 0:
        return flat
    return flat / total


def _edge_grid(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholds from the image's own median rather than from two constants: the
    # same car at noon and at dusk is the case this whole module exists for, and
    # fixed thresholds turn a lighting change into a geometry change.
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, max(upper, lower + 1))
    h, w = edges.shape
    grid = np.zeros((GRID_SIDE, GRID_SIDE), np.uint8)
    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE):
            y0, y1 = row * h // GRID_SIDE, max((row + 1) * h // GRID_SIDE, row * h // GRID_SIDE + 1)
            x0, x1 = col * w // GRID_SIDE, max((col + 1) * w // GRID_SIDE, col * w // GRID_SIDE + 1)
            cell = edges[y0:y1, x0:x1]
            density = float(cell.mean()) / 255.0 if cell.size else 0.0
            grid[row, col] = int(round(min(1.0, density) * 255))
    return grid


def encode(kind: str, keypoints: np.ndarray, histogram: np.ndarray, grid: np.ndarray) -> str:
    """Pack, compress, base64url. The inverse of `decode`, and tested as such."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    n, dim = (keypoints.shape if keypoints.size else (0, 32 if kind == ORB else 128))
    quantised = np.rint(np.clip(histogram, 0.0, 1.0) * 65535).astype(">u2")
    header = struct.pack(
        _HEADER, _MAGIC, DESCRIPTOR_VERSION, _KIND_CODES[kind], n, dim, *HIST_BINS, GRID_SIDE
    )
    blob = header + keypoints.astype(np.uint8).tobytes() + quantised.tobytes() + grid.tobytes()
    payload = base64.urlsafe_b64encode(zlib.compress(blob, 6)).decode("ascii").rstrip("=")
    return f"{DESCRIPTOR_PREFIX}/{DESCRIPTOR_VERSION}:{payload}"


def decode(text: str) -> Descriptor:
    """A descriptor, or a refusal. Never a partial read.

    The version is checked from the prefix FIRST, so a descriptor from a build
    that packed different bytes is refused before those bytes are touched.
    """
    version = descriptor_version_of(text)
    if version != DESCRIPTOR_VERSION:
        raise IncomparableDescriptors(
            f"descriptor version {version} was produced by a different build; "
            f"this one understands {DESCRIPTOR_VERSION}. Refusing to guess what "
            "its bytes mean."
        )
    _, _, payload = text.partition(":")
    try:
        blob = zlib.decompress(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception as exc:  # noqa: BLE001 - any decode failure is one refusal
        raise MalformedDescriptor(f"descriptor payload does not decode: {exc}") from exc
    if len(blob) < _HEADER_SIZE:
        raise MalformedDescriptor("descriptor payload is shorter than its header")
    magic, ver, code, n, dim, hb, sb, vb, grid_side = struct.unpack(
        _HEADER, blob[:_HEADER_SIZE]
    )
    if magic != _MAGIC:
        raise MalformedDescriptor(f"descriptor payload is not {_MAGIC!r}")
    if ver != version:
        raise MalformedDescriptor(
            f"the prefix says version {version} and the payload says {ver}"
        )
    if code not in _CODE_KINDS:
        raise MalformedDescriptor(f"unknown descriptor kind code {code}")
    bins = hb * sb * vb
    want = _HEADER_SIZE + n * dim + bins * 2 + grid_side * grid_side
    if len(blob) != want:
        raise MalformedDescriptor(f"descriptor payload is {len(blob)} bytes, expected {want}")
    at = _HEADER_SIZE
    keypoints = np.frombuffer(blob, np.uint8, count=n * dim, offset=at).reshape(n, dim).copy()
    at += n * dim
    quantised = np.frombuffer(blob, ">u2", count=bins, offset=at).astype(np.float32) / 65535.0
    at += bins * 2
    grid = np.frombuffer(blob, np.uint8, count=grid_side * grid_side, offset=at)
    grid = grid.reshape(grid_side, grid_side).copy()
    total = float(quantised.sum())
    histogram = quantised / total if total > 0 else quantised
    return Descriptor(
        version=version,
        kind=_CODE_KINDS[code],
        keypoints=keypoints,
        histogram=np.ascontiguousarray(histogram, np.float32),
        grid=grid,
        text=text,
    )


def compute(image: np.ndarray, kind: str = ORB) -> Descriptor:
    """One capture in, one descriptor out.

    Encodes and then decodes its own payload rather than returning what it just
    built. That round trip is the point: the histogram is quantised on the way
    into the string, so a descriptor compared in memory and the same descriptor
    read back out of a database would otherwise differ in the third decimal
    place -- a discrepancy that would show up as unexplained drift in a match
    score months later and be blamed on the camera.
    """
    working = _to_working_image(image)
    text = encode(kind, _keypoints(working, kind), _histogram(working), _edge_grid(working))
    return decode(text)


def compute_text(image: np.ndarray, kind: str = ORB) -> str:
    """The string form, which is what the record carries."""
    return compute(image, kind).text


class DescriptorComputer:
    """The engine's injection point, shaped like the presence detector's.

    Optional there, and absent by default, so a build that has not been asked
    for a descriptor emits `identity.descriptor = null` -- NOT MEASURED, the
    same answer every other unmeasured component gives.
    """

    def __init__(self, kind: str = ORB) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind

    def compute(self, image: np.ndarray) -> str:
        return compute_text(image, self.kind)


# --- the comparison ------------------------------------------------------


def _as_descriptor(value) -> Descriptor:
    return value if isinstance(value, Descriptor) else decode(value)


def _structure_distance(a: Descriptor, b: Descriptor) -> TermDistance:
    n_a, n_b = a.keypoint_count, b.keypoint_count
    if n_a < MEASURABLE_MIN_KEYPOINTS or n_b < MEASURABLE_MIN_KEYPOINTS:
        return TermDistance(
            term="structure",
            value=None,
            measurable=False,
            reason=(
                f"too few keypoints to match: {n_a} and {n_b}, minimum "
                f"{MEASURABLE_MIN_KEYPOINTS}. There was nothing to match, which "
                "is not the same as nothing matching."
            ),
        )
    norm = cv2.NORM_HAMMING if a.kind == ORB else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    left = a.keypoints
    right = b.keypoints if a.kind == ORB else b.keypoints.astype(np.float32)
    if a.kind == SIFT:
        left = left.astype(np.float32)

    def good(query, train) -> int:
        kept = 0
        for pair in matcher.knnMatch(query, train, k=2):
            if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance:
                kept += 1
        return kept

    matched = good(left, right) + good(right, left)
    fraction = matched / (2 * min(n_a, n_b))
    return TermDistance(term="structure", value=float(min(1.0, max(0.0, 1.0 - fraction))),
                        measurable=True)


def _colour_distances(a: Descriptor, b: Descriptor) -> list[TermDistance]:
    ha, hb = a.histogram, b.histogram
    if float(ha.sum()) <= 0 or float(hb.sum()) <= 0:
        reason = "an empty colour histogram: the crop carried no pixels to bin"
        return [
            TermDistance(term=name, value=None, measurable=False, reason=reason)
            for name in ("colour_bhattacharyya", "colour_chisqr", "colour_intersection")
        ]
    return [
        TermDistance(
            term="colour_bhattacharyya",
            value=float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)),
            measurable=True,
        ),
        TermDistance(
            term="colour_chisqr",
            # ALT, not plain: plain chi-square divides by the first histogram
            # and is therefore asymmetric. See the term's note in TERMS.
            value=float(cv2.compareHist(ha, hb, cv2.HISTCMP_CHISQR_ALT)),
            measurable=True,
        ),
        TermDistance(
            term="colour_intersection",
            # Converted, once, here. Intersection of two L1-normalised
            # histograms is in [0, 1] and counts the OTHER way.
            value=float(1.0 - cv2.compareHist(ha, hb, cv2.HISTCMP_INTERSECT)),
            measurable=True,
        ),
    ]


def _geometry_distance(a: Descriptor, b: Descriptor) -> TermDistance:
    if a.grid.shape != b.grid.shape:
        return TermDistance(
            term="geometry",
            value=None,
            measurable=False,
            reason=f"grids differ in shape: {a.grid.shape} and {b.grid.shape}",
        )
    diff = np.abs(a.grid.astype(np.float32) - b.grid.astype(np.float32)) / 255.0
    return TermDistance(term="geometry", value=float(diff.mean()), measurable=True)


def compare(a, b) -> Comparison:
    """Two descriptors in, one distance per term out. NEVER a verdict.

    Accepts either the decoded record or the string. Refuses -- rather than
    returning a large number -- when the two do not compare at all:

      * different DESCRIPTOR VERSIONS. Two builds that pack different bytes
        produce numbers that share a name and nothing else.
      * different KINDS. An ORB structure distance and a SIFT one are computed
        from different features under different metrics; putting them on the
        same axis is how a table becomes an artefact of the detector.
    """
    left, right = _as_descriptor(a), _as_descriptor(b)
    if left.version != right.version:
        raise IncomparableDescriptors(
            f"descriptor versions {left.version} and {right.version} do not "
            "compare. The bytes mean different things."
        )
    if left.kind != right.kind:
        raise IncomparableDescriptors(
            f"descriptor kinds {left.kind!r} and {right.kind!r} do not compare. "
            "Their structure terms are different measurements under different "
            "metrics."
        )
    terms = [_structure_distance(left, right), *_colour_distances(left, right),
             _geometry_distance(left, right)]
    by_name = {t.term: t for t in terms}
    ordered = tuple(by_name[name] for name in TERM_NAMES)
    return Comparison(kind=left.kind, version=left.version, terms=ordered)


# --- the operating point -------------------------------------------------


class NoOperatingPoint(RuntimeError):
    """Raised rather than choosing a threshold that satisfies only one constraint.

    The same rule the plate recogniser's chooser follows, and for the same
    reason: unsure is a first-class answer here exactly as it is at the barrier.
    """


def _candidate_thresholds(same: list[float], different: list[float]) -> list[float]:
    """Midpoints between adjacent observed distances, plus the two extremes.

    DERIVED from the data rather than a fixed list, because the five terms are
    on five different scales -- one of them unbounded -- and a list of constants
    would be meaningful for at most one of them.
    """
    values = sorted(set(same) | set(different))
    if not values:
        return []
    candidates = [values[0] - 1e-9]
    candidates.extend((lo + hi) / 2.0 for lo, hi in zip(values, values[1:], strict=False))
    candidates.append(values[-1] + 1e-9)
    return candidates


def rates_at(threshold: float, same: list[float], different: list[float]) -> tuple[float, float]:
    """(miss rate, false-match rate) at `threshold`, over the MEASURABLE values only.

    "Matched" is `distance <= threshold`. A miss is a same-car comparison that
    did not match; a false match is a different-car comparison that did.
    """
    misses = sum(1 for d in same if d > threshold)
    false_matches = sum(1 for d in different if d <= threshold)
    miss_rate = misses / len(same) if same else 0.0
    false_match_rate = false_matches / len(different) if different else 0.0
    return miss_rate, false_match_rate


def choose_operating_point(
    same: list[float],
    different: list[float],
    *,
    max_false_match_rate: float,
    max_miss_rate: float,
) -> dict:
    """A threshold that satisfies BOTH constraints, or a refusal naming the conflict.

    Both constraints are supplied by the caller and neither has a default. A
    default here would be a product decision made in a helper function, and the
    number a barrier opens on is not that.

    The search is over candidates derived from the data. Because "matched" is
    `distance <= threshold`, the false-match rate is non-decreasing in the
    threshold and the miss rate is non-increasing, so the best available miss
    rate is the one at the LARGEST threshold still satisfying the false-match
    constraint. If that is still too many misses, no threshold satisfies both --
    and the refusal says so from both sides, giving the best each constraint can
    do alone, because "it refused" without the numbers is not diagnosable.
    """
    if not same or not different:
        raise NoOperatingPoint(
            "nothing to choose from: "
            f"{len(same)} measurable same-car and {len(different)} measurable "
            "different-car distances. A threshold fitted on one class is not a "
            "threshold."
        )
    candidates = _candidate_thresholds(same, different)
    scored = [(t, *rates_at(t, same, different)) for t in candidates]

    admissible = [row for row in scored if row[2] <= max_false_match_rate]
    if admissible:
        threshold, miss, false_match = max(admissible, key=lambda row: row[0])
        if miss <= max_miss_rate:
            return {
                "threshold": threshold,
                "miss_rate": miss,
                "false_match_rate": false_match,
                "max_miss_rate": max_miss_rate,
                "max_false_match_rate": max_false_match_rate,
                "fitted_on": {"same_car": len(same), "different_car": len(different)},
            }
        best_miss_under_fm = miss
    else:
        best_miss_under_fm = None

    by_miss = [row for row in scored if row[1] <= max_miss_rate]
    best_fm_under_miss = min((row[2] for row in by_miss), default=None)

    raise NoOperatingPoint(
        "no threshold satisfies both constraints.\n"
        f"  keeping false matches at or below {max_false_match_rate:.4f} allows a "
        f"miss rate no better than {_fmt(best_miss_under_fm)} "
        f"(wanted {max_miss_rate:.4f})\n"
        f"  keeping misses at or below {max_miss_rate:.4f} costs a false-match rate "
        f"of at least {_fmt(best_fm_under_miss)} "
        f"(wanted {max_false_match_rate:.4f})\n"
        f"  measured over {len(same)} same-car and {len(different)} different-car "
        "distances"
    )


def _fmt(value: float | None) -> str:
    return "no threshold at all" if value is None else f"{value:.4f}"
