"""Plate layout templates, the highest-volume one first.

These are APPROXIMATIONS of real layouts, not reproductions. They exist to give
the generator a realistic distribution of character counts, groupings and
furniture (state name, county strip, slogan) so a recogniser trained on them is
not learning a single rigid shape.

Two honest limitations, recorded here rather than discovered later:

  * The FONTS are wrong. Real plates use specific embossing typefaces which
    we neither have nor could redistribute. The generator uses OpenCV's built-in
    Hershey fonts. This is the single largest domain gap between synthetic and
    real plates, and it is why real-plate accuracy stays unmeasured until the
    physical bench exists.
  * The COLOURS and furniture are stylised. Enough to vary the background, not
    enough to be a facsimile of any state's plate.

No real plate, real vehicle or real registration appears anywhere here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 'L' = letter, 'N' = digit, ' ' = a visible gap
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O omitted: most states skip them
DIGITS = "0123456789"


@dataclass(frozen=True, slots=True)
class PlateTemplate:
    state: str
    patterns: tuple[str, ...]
    top_text: str = ""
    bottom_text: str = ""
    #: BGR, because OpenCV
    background: tuple[int, int, int] = (245, 245, 245)
    ink: tuple[int, int, int] = (35, 35, 45)
    #: Relative frequency when sampling. Florida is weighted up deliberately.
    weight: float = 1.0
    #: The letters THIS layout uses, when it uses fewer than all of them. A
    #: layout whose registrations can only contain a restricted set would
    #: otherwise be trained on registrations it cannot have -- so the generator
    #: draws from this when it is set, and from `LETTERS` when it is not. The
    #: model's charset is the UNION, so a template adding a letter is the only
    #: thing that moves the class count.
    letters: str = ""
    #: Width in px of a plain coloured band down the left edge, carrying no
    #: text. The registration is centred in what is left. 0 means no band.
    band: int = 0
    #: BGR, because OpenCV.
    band_colour: tuple[int, int, int] = (140, 60, 20)


TEMPLATES: tuple[PlateTemplate, ...] = (
    PlateTemplate(
        state="FL",
        patterns=("LLL LNN", "LNN NLL", "NNL LNN"),
        top_text="FLORIDA",
        bottom_text="MYFLORIDA.COM",
        background=(250, 250, 250),
        ink=(60, 45, 40),
        weight=6.0,
    ),
    PlateTemplate(
        state="GA",
        patterns=("LLL NNNN", "LLLL NNN"),
        top_text="GEORGIA",
        background=(248, 246, 240),
        weight=1.0,
    ),
    PlateTemplate(
        state="NY",
        patterns=("LLL NNNN",),
        top_text="NEW YORK",
        background=(252, 248, 232),
        ink=(40, 55, 95),
        weight=1.0,
    ),
    PlateTemplate(
        state="TX",
        patterns=("LLL NNNN", "NNN LLLL"),
        top_text="TEXAS",
        background=(250, 250, 250),
        weight=1.0,
    ),
    PlateTemplate(
        state="CA",
        patterns=("NLLL NNN",),
        top_text="CALIFORNIA",
        background=(252, 252, 248),
        ink=(30, 45, 90),
        weight=1.0,
    ),
    # A layout with a plain band down the left edge and a restricted letter
    # set. It carries no top or bottom text: there is nothing it needs to say.
    #
    # The band is 6 px, and that number is DERIVED rather than chosen. The
    # widest registration this pattern can draw is "MMM 0000" -- M is the widest
    # letter these fonts have -- which renders 298 px at the top of the scale
    # range. Centring keeps 8 px each side, so the band can be at most
    # PLATE_W - 298 - 16 = 6 px. It is thin, and thin is the honest answer: the
    # alternative is narrowing the scale range for this template alone, and the
    # scale variation is what stands in for the font variation the generator
    # cannot model -- narrowing it on the one layout being measured against real
    # photographs would bias that measurement.
    PlateTemplate(
        state="BAND3L4N",
        patterns=("LLL NNNN",),
        letters="ABEZHIKMNOPTYX",
        background=(250, 250, 250),
        ink=(35, 35, 40),
        weight=1.0,
        band=6,
    ),
)


def charset() -> str:
    """Every character a plate can contain, plus CTC blank handled separately.

    The UNION of the module's letters and every template's own, sorted so the
    class count is a property of the templates rather than of their order. A
    template that introduces a letter grows this, and growing this changes
    `PlateNet`'s class count -- which is a full retrain from seed and a new
    `weights_id`, never a silent change.
    """
    letters = set(LETTERS)
    for template in TEMPLATES:
        letters |= set(template.letters)
    return "".join(sorted(letters)) + DIGITS


@dataclass(frozen=True, slots=True)
class GeneratedPlate:
    text: str
    state: str
    template: PlateTemplate = field(repr=False)
