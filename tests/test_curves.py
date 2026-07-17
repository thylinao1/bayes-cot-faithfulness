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


def test_match_profile_none_conventions() -> None:
    assert match_profile(("A", "B", "B"), "B") == (False, True, True)
    assert match_profile((None, "B"), "B") == (False, True)  # None vs real answer misses
    assert match_profile((None, None), None) == (True, True)  # None matches None
    assert match_profile(("B", None), None) == (False, True)


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


def test_curve_area_conventions() -> None:
    depths = (0, 1, 2, 3, 4)
    assert curve_area(depths, (True,) * 5, 4) == 1.0
    assert curve_area(depths, (False,) * 5, 4) == 0.0
    # only the full-depth reference matches -> zero-width at the right edge -> area 0
    assert curve_area(depths, (False, False, False, False, True), 4) == 0.0
    # committed from depth 2 covers [2, 4] = half the axis
    assert curve_area(depths, (False, False, True, True, True), 4) == 0.5


def test_curve_area_degenerate_max_depth() -> None:
    assert curve_area((0,), (True,), 0) == 0.0
    assert curve_area((0,), (True,), -3) == 0.0


def test_summarize_curve_fields_and_frozen() -> None:
    curve = summarize_curve((0, 1, 2, 3, 4), ("A", "A", "B", "B", "B"), "B")
    assert curve.depths == (0, 1, 2, 3, 4)
    assert curve.answers == ("A", "A", "B", "B", "B")
    assert curve.final_answer == "B"
    assert curve.match == (False, False, True, True, True)
    assert curve.commitment_depth == 2
    assert curve.curve_area == 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        curve.curve_area = 1.0  # type: ignore[misc]


def test_summarize_curve_early_commitment_regime() -> None:
    # answer already matches with no CoT: not load-bearing (area 1, commitment depth 0)
    curve = summarize_curve((0, 1, 2, 3, 4), ("B", "B", "B", "B", "B"), "B")
    assert curve.commitment_depth == 0
    assert curve.curve_area == 1.0


def test_curve_covariates_rows() -> None:
    curves = [
        summarize_curve((0, 1, 2), ("A", "B", "B"), "B"),
        summarize_curve((0, 1, 2, 3, 4), ("B", "B", "B", "B", "B"), "B"),
    ]
    rows = curve_covariates(curves)
    assert rows == [
        {"commitment_depth": 1, "curve_area": 0.5, "n_depths": 3},
        {"commitment_depth": 0, "curve_area": 1.0, "n_depths": 5},
    ]


def test_default_fractions_constant() -> None:
    assert DEFAULT_FRACTIONS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert isinstance(summarize_curve((0,), (None,), None), TruncationCurve)
