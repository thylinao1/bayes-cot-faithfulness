"""Tests for the truncation dose-response curve helpers (pure, no model server)."""

from __future__ import annotations

import dataclasses

import pytest

from bayes_cot_faithfulness.curves import (
    DEFAULT_FRACTIONS,
    TruncationCurve,
    commitment_depth,
    curve_area,
    curve_covariates,
    curve_prompts,
    depths_for,
    match_profile,
    summarize_curve,
)
from bayes_cot_faithfulness.interventions import (
    QAItem,
    continuation_prompt,
    truncate_cot,
)

ITEM = QAItem(question="What is 2 + 3?", choices=("4", "5", "6", "7"), answer_index=1)

COT4 = "1. add the numbers\n2. carry nothing\n3. it is five\n4. so the option is (B)"


def test_depths_for_four_steps_full_grid() -> None:
    # n_steps = 4, DEFAULT_FRACTIONS map cleanly onto 0..4
    assert depths_for(COT4) == [0, 1, 2, 3, 4]


def test_depths_for_empty_chain_is_zero_only() -> None:
    assert depths_for("") == [0]


def test_depths_for_one_step_chain_is_zero_and_one() -> None:
    # round-half-to-even sends 0.25 and 0.5 to 0; 0.75 and 1.0 to 1
    assert depths_for("1. the only step") == [0, 1]


def test_depths_for_dedups_short_chains() -> None:
    assert depths_for("1. a\n2. b") == [0, 1, 2]  # n_steps = 2
    assert depths_for("1. a\n2. b\n3. c") == [0, 1, 2, 3]  # n_steps = 3


def test_depths_for_custom_fractions() -> None:
    assert depths_for(COT4, (0.0, 0.5, 1.0)) == [0, 2, 4]
    # a coarse single-endpoint grid still dedups
    assert depths_for("1. a\n2. b", (0.0, 0.1, 0.9, 1.0)) == [0, 2]


def test_curve_prompts_align_with_truncate_cot() -> None:
    expected = [
        (k, continuation_prompt(ITEM, truncate_cot(COT4, k))) for k in depths_for(COT4)
    ]
    assert curve_prompts(ITEM, COT4) == expected


def test_curve_prompts_depth_zero_omits_reasoning_block() -> None:
    prompts = dict(curve_prompts(ITEM, COT4))
    assert "Partial reasoning so far:" not in prompts[0]
    assert "add the numbers" in prompts[1]  # depth 1 carries the first step
    assert "What is 2 + 3?" in prompts[0]


def test_match_profile_tristate_none_is_unscorable() -> None:
    # All real answers -> plain equality, as before.
    assert match_profile(("A", "B", "B"), "B") == (False, True, True)
    # A None depth answer is UNSCORABLE (None): neither a miss nor a match. This reverses
    # the old rule, which scored None-vs-real as a miss and None-vs-None as a spurious
    # match, inflating agreement with what is really missing data.
    assert match_profile((None, "B"), "B") == (None, True)
    assert match_profile(("B", None, "A"), "B") == (True, None, False)
    # A None FINAL answer has no reference to match: the WHOLE curve is unscorable.
    assert match_profile((None, None), None) == (None, None)
    assert match_profile(("B", None), None) == (None, None)


def test_commitment_depth_stable_suffix() -> None:
    depths = (0, 1, 2, 3)
    assert commitment_depth(depths, (False, True, True, True)) == 1
    assert commitment_depth(depths, (True, True, True, True)) == 0


def test_commitment_depth_rematch_takes_later_depth() -> None:
    # matches at 0, diverges at 1, re-matches from 2 on: commitment is the LATER depth 2
    assert commitment_depth((0, 1, 2, 3), (True, False, True, True)) == 2


def test_commitment_depth_none_when_never_stable() -> None:
    # deepest measured depth does not match -> never stably committed
    assert commitment_depth((0, 1, 2, 3), (True, True, True, False)) is None
    assert commitment_depth((), ()) is None


def test_commitment_depth_skips_unscorable_depths() -> None:
    depths = (0, 1, 2, 3)
    # a None (unscorable) depth is skipped: it neither breaks nor extends the suffix
    assert commitment_depth(depths, (True, None, True, True)) == 0
    # a scorable non-match still breaks the suffix; the None after it is skipped
    assert commitment_depth(depths, (False, None, True, True)) == 2
    # no scorable depths at all -> nothing to commit to
    assert commitment_depth(depths, (None, None, None, None)) is None


def test_curve_area_is_mean_of_scorable_matches() -> None:
    assert curve_area((True,) * 5) == 1.0
    assert curve_area((False,) * 5) == 0.0
    # 3 of 5 depths match -> 0.6 (an unweighted mean, not a depth-weighted area)
    assert curve_area((False, False, True, True, True)) == 0.6
    # None depths are dropped from the mean, not counted as misses: mean of {True, False}
    assert curve_area((None, True, None, False)) == 0.5


def test_curve_area_none_when_no_scorable_depths() -> None:
    # every depth unscorable -> no area to report (None, not a misleading 0.0)
    assert curve_area((None, None)) is None
    assert curve_area(()) is None


def test_summarize_curve_fields_and_frozen() -> None:
    curve = summarize_curve((0, 1, 2, 3, 4), ("A", "A", "B", "B", "B"), "B")
    assert curve.depths == (0, 1, 2, 3, 4)
    assert curve.answers == ("A", "A", "B", "B", "B")
    assert curve.final_answer == "B"
    assert curve.match == (False, False, True, True, True)
    assert curve.commitment_depth == 2
    assert curve.curve_area == 0.6  # 3 of 5 scorable depths match
    assert curve.n_unscorable_depths == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        curve.curve_area = 1.0  # type: ignore[misc]


def test_summarize_curve_early_commitment_regime() -> None:
    # answer already matches with no CoT: not load-bearing (area 1, commitment depth 0)
    curve = summarize_curve((0, 1, 2, 3, 4), ("B", "B", "B", "B", "B"), "B")
    assert curve.commitment_depth == 0
    assert curve.curve_area == 1.0
    assert curve.n_unscorable_depths == 0


def test_summarize_curve_none_answer_excluded_from_commitment_and_area() -> None:
    # depth-1 answer never parsed (None): unscorable at that depth, skipped in both summaries
    curve = summarize_curve((0, 1, 2), ("A", None, "B"), "B")
    assert curve.match == (False, None, True)
    # scorable depths 0 (miss) and 2 (match): commitment is the later depth 2
    assert curve.commitment_depth == 2
    # mean of the scorable matches {False, True} -> 0.5
    assert curve.curve_area == 0.5
    assert curve.n_unscorable_depths == 1


def test_summarize_curve_wholly_unscorable_final_answer_none() -> None:
    # final answer never parsed: no reference, so the whole curve is unscorable even though
    # every depth answer parsed (so n_unscorable_depths counts zero None-ANSWER depths)
    curve = summarize_curve((0, 1, 2), ("A", "B", "C"), None)
    assert curve.match == (None, None, None)
    assert curve.commitment_depth is None
    assert curve.curve_area is None
    assert curve.n_unscorable_depths == 0


def test_summarize_curve_wholly_unscorable_all_depths_none() -> None:
    curve = summarize_curve((0, 1, 2), (None, None, None), "B")
    assert curve.match == (None, None, None)
    assert curve.commitment_depth is None
    assert curve.curve_area is None
    assert curve.n_unscorable_depths == 3


def test_curve_covariates_rows() -> None:
    curves = [
        summarize_curve((0, 1, 2), ("A", "B", "B"), "B"),
        summarize_curve((0, 1, 2, 3, 4), ("B", "B", "B", "B", "B"), "B"),
    ]
    rows = curve_covariates(curves)
    assert rows == [
        {"commitment_depth": 1, "curve_area": 2 / 3, "n_depths": 3},
        {"commitment_depth": 0, "curve_area": 1.0, "n_depths": 5},
    ]


def test_curve_covariates_carries_none_area_for_unscorable_curve() -> None:
    # a wholly-unscorable curve surfaces curve_area None to the model, not a fake 0.0
    rows = curve_covariates([summarize_curve((0, 1), ("A", "B"), None)])
    assert rows == [{"commitment_depth": None, "curve_area": None, "n_depths": 2}]


def test_default_fractions_constant() -> None:
    assert DEFAULT_FRACTIONS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert isinstance(summarize_curve((0,), (None,), None), TruncationCurve)
