"""The operating-point chooser: what it admits, and what it must refuse.

The chooser picks the number the engine applies to every read. Until this round
it picked on exactly one criterion -- the cheapest candidate threshold whose
pooled silent-wrong rate falls under 1% -- with nothing tying it to whether the
model still answers a clean plate, and nothing tying it to the noise ceiling the
same function measured eleven lines further down.

That is not a quirk of one checkpoint. Silent-wrong falls monotonically as the
threshold rises, so an unconstrained "cheapest under the bar" rule always drifts
towards the strictest candidate, and the strictest candidate is the one most
likely to refuse the plates the product exists to read. It happened once, and it
would happen again on the next retrain that shifts the confidence distribution
down.

So the fixtures below are not hypotheticals. `c825c957` is the retrained
checkpoint of this round, with its measured numbers, and the chooser must REFUSE
it: answering its pristine plates needs a threshold at or below 0.9947, keeping
the gate's noise measurement possible needs 0.9004 or below, and the under-1%
rule needs 0.995 or above. No threshold sits in that gap, and a refusal naming
the numbers is the correct output. `0de21983` is the reference checkpoint, whose
0.99 the chooser must still choose -- without that side, "refuses" and "refuses
everything" are the same test.

Needs no weights, no torch and no model: the chooser is a pure function over a
table of measured rows, which is what makes both answers reachable here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_plates import (  # noqa: E402
    CANDIDATE_THRESHOLDS,
    MAX_SILENT_WRONG_PCT,
    NoOperatingPoint,
    choose_operating_point,
    median_clean_confidence,
)

guarantee = pytest.mark.guarantee


#: The reference checkpoint `0de21983`, PRODUCED rather than typed:
#:
#:     python scripts/eval_plates.py --weights models/plate_crnn_ref_20260826.pt \
#:         --skip-baseline
#:
#: silent-wrong % and fallback % per candidate threshold, 200 plates x 10 rungs,
#: seed 9999991. The 0.990 row is the operating point this checkpoint carries in
#: its sidecar, and the 0.850 row is the 4.45% the README publishes.
REFERENCE_ROWS = [
    (0.000, 4.80, 0.0),
    (0.500, 4.80, 0.0),
    (0.800, 4.60, 0.2),
    (0.850, 4.45, 0.3),
    (0.900, 3.35, 1.9),
    (0.950, 2.10, 11.1),
    (0.970, 1.30, 18.2),
    (0.980, 1.25, 23.1),
    (0.990, 0.80, 30.6),
    (0.995, 0.45, 37.2),
]
REFERENCE_CLEAN = 0.9982
REFERENCE_CEILING = 0.9998

#: The retrained checkpoint `c825c957` of this round. Its three numbers are the
#: conflict, measured during review and recorded here because the checkpoint
#: itself no longer loads: it was trained at 36 classes, and the layout template
#: that grew the charset to 36 came out with this round.
RETRAINED_ROWS = [
    (0.980, 1.35, 50.0),
    (0.990, 1.00, 58.0),
    (0.995, 0.40, 66.2),
]
RETRAINED_CLEAN = 0.9947
RETRAINED_CEILING = 0.9004


# --- the accept side ------------------------------------------------------


@guarantee
def test_the_reference_checkpoints_own_point_is_still_chosen():
    """The control for every refusal below. A chooser that refuses everything
    would pass all of them and ship nothing."""
    point = choose_operating_point(
        REFERENCE_ROWS,
        clean_plate_confidence=REFERENCE_CLEAN,
        noise_confidence_ceiling=REFERENCE_CEILING,
    )
    assert point["threshold"] == 0.990
    assert point["silent_wrong_pct"] == 0.80
    assert point["fallback_pct"] == 30.6


@guarantee
def test_the_cheapest_qualifying_threshold_wins_not_the_strictest():
    """A fallback costs an operator a glance, so the point is the cheapest
    candidate that satisfies every constraint -- not the safest-looking one."""
    point = choose_operating_point(
        REFERENCE_ROWS,
        clean_plate_confidence=REFERENCE_CLEAN,
        noise_confidence_ceiling=REFERENCE_CEILING,
    )
    assert point["threshold"] == 0.990, "0.995 also clears every bar and must NOT win"


# --- the refusal side -----------------------------------------------------


@guarantee
def test_the_retrained_checkpoint_is_refused_rather_than_given_0_995():
    """The defect, reproduced. Under the old rule this returned 0.995, the
    engine applied it, and pristine plates stopped being answered."""
    with pytest.raises(NoOperatingPoint) as raised:
        choose_operating_point(
            RETRAINED_ROWS,
            clean_plate_confidence=RETRAINED_CLEAN,
            noise_confidence_ceiling=RETRAINED_CEILING,
        )

    message = str(raised.value)
    # It must NAME the numbers that conflict. A refusal a reader cannot check is
    # not better than a wrong answer.
    assert "0.995" in message, "the threshold the silent-wrong rule demands is not named"
    assert "0.9947" in message, "the clean-plate confidence is not named"
    assert "0.9004" in message, "the noise confidence ceiling is not named"

    conflicts = raised.value.conflicts
    assert conflicts["lowest_threshold_meeting_silent_wrong"] == 0.995
    assert conflicts["clean_plate_confidence"] == RETRAINED_CLEAN
    assert conflicts["noise_confidence_ceiling"] == RETRAINED_CEILING


@guarantee
@pytest.mark.parametrize(
    "clean,ceiling,expected,which",
    [
        (0.9947, 0.9004, None, "both ceilings bind: the round's own case"),
        (0.9947, 0.9998, None, "clean-plate alone binds"),
        (0.9998, 0.9004, None, "noise ceiling alone binds"),
        (0.9998, 0.9998, 0.995, "neither binds: 0.995 is admissible after all"),
    ],
)
def test_each_constraint_can_refuse_on_its_own(clean, ceiling, expected, which):
    """One constraint at a time. Without the fourth row, a chooser that ignored
    one of the two new inputs entirely would still pass the three above."""
    if expected is None:
        with pytest.raises(NoOperatingPoint):
            choose_operating_point(
                RETRAINED_ROWS, clean_plate_confidence=clean, noise_confidence_ceiling=ceiling
            )
    else:
        point = choose_operating_point(
            RETRAINED_ROWS, clean_plate_confidence=clean, noise_confidence_ceiling=ceiling
        )
        assert point["threshold"] == expected, which


@guarantee
@pytest.mark.parametrize(
    "clean,admissible,which",
    [
        (0.9899, False, "just below 0.990: the cheapest candidate is above it"),
        (0.9900, True, "exactly at it: equal is admissible"),
        (0.9901, True, "just above it"),
    ],
)
def test_the_clean_plate_constraint_is_tight_at_its_boundary(clean, admissible, which):
    """Fixtures on both sides of the constant, and on it. `<` and `<=` are a
    different chooser, and only the equal case tells them apart."""
    rows = [(0.990, 0.80, 30.6), (0.995, 0.40, 55.1)]
    if admissible:
        point = choose_operating_point(
            rows, clean_plate_confidence=clean, noise_confidence_ceiling=0.9998
        )
        assert point["threshold"] == 0.990, which
    else:
        with pytest.raises(NoOperatingPoint):
            choose_operating_point(
                rows, clean_plate_confidence=clean, noise_confidence_ceiling=0.9998
            )


@guarantee
def test_a_model_needing_more_than_the_top_candidate_is_told_so():
    """0.995 is the top of the list, so "nothing qualifies" can mean the model
    or it can mean the list. The refusal has to say which."""
    rows = [(t, 5.0, 10.0) for t in CANDIDATE_THRESHOLDS]
    with pytest.raises(NoOperatingPoint) as raised:
        choose_operating_point(
            rows, clean_plate_confidence=0.9998, noise_confidence_ceiling=0.9998
        )
    assert "candidate list itself is the limit" in str(raised.value)
    assert raised.value.conflicts["lowest_threshold_meeting_silent_wrong"] is None
    assert raised.value.conflicts["highest_candidate"] == max(CANDIDATE_THRESHOLDS)


@guarantee
def test_the_silent_wrong_bar_is_strict_and_one_percent_exactly_does_not_pass():
    """The retrained checkpoint's 0.990 row is 1.00%, which is why 0.995 was
    reached at all. `<` rather than `<=`, fixtured on both sides."""
    assert MAX_SILENT_WRONG_PCT == 1.0
    with pytest.raises(NoOperatingPoint):
        choose_operating_point(
            [(0.990, 1.00, 58.0)], clean_plate_confidence=0.9998, noise_confidence_ceiling=0.9998
        )
    point = choose_operating_point(
        [(0.990, 0.99, 58.0)], clean_plate_confidence=0.9998, noise_confidence_ceiling=0.9998
    )
    assert point["threshold"] == 0.990


# --- the statistic the clean-plate constraint is built on -----------------


@guarantee
def test_the_clean_plate_confidence_is_the_median_and_an_empty_set_is_zero():
    """Zero on an empty set, so a model that reads NO clean plate correctly is
    refused rather than handed an unconstrained choice."""
    assert median_clean_confidence([0.1, 0.9, 0.5]) == 0.5
    assert median_clean_confidence([]) == 0.0
    with pytest.raises(NoOperatingPoint):
        choose_operating_point(
            REFERENCE_ROWS,
            clean_plate_confidence=median_clean_confidence([]),
            noise_confidence_ceiling=REFERENCE_CEILING,
        )
