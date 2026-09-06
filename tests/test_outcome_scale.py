"""Offline tests for the CONTRACT intervention_level / outcome_scale record fields."""

from __future__ import annotations

import math

import pytest

from bayes_cot_faithfulness.outcome_scale import (
    INTERVENTION_LEVELS,
    OUTCOME_SCALES,
    OutcomeScaleError,
    assert_records_scaled,
    check_outcome_scale,
    letter_logprob_fields,
)


def test_the_contract_vocabularies_are_exactly_what_the_contract_names():
    assert set(INTERVENTION_LEVELS) == {"logit", "text"}
    assert set(OUTCOME_SCALES) == {
        "raw",
        "renormalized_over_letters",
        "logprob_margin",
        "binary_follow",
    }


def test_text_level_accepts_binary_follow():
    check_outcome_scale("text", "binary_follow")


@pytest.mark.parametrize(
    "scale", ["raw", "renormalized_over_letters", "logprob_margin"]
)
def test_logit_level_accepts_the_three_logprob_scales(scale):
    check_outcome_scale("logit", scale)


def test_missing_level_is_refused():
    with pytest.raises(OutcomeScaleError, match="intervention_level"):
        check_outcome_scale(None, "binary_follow")


def test_unknown_scale_is_refused():
    with pytest.raises(OutcomeScaleError, match="outcome_scale"):
        check_outcome_scale("text", "accuracy")


def test_a_scale_the_level_cannot_carry_is_refused():
    """A text arm reporting a logprob margin would be pooled into the row estimand."""
    with pytest.raises(OutcomeScaleError, match="not reportable"):
        check_outcome_scale("text", "logprob_margin")
    with pytest.raises(OutcomeScaleError, match="not reportable"):
        check_outcome_scale("logit", "binary_follow")


def test_assert_records_scaled_returns_the_denominator():
    recs = [{"intervention_level": "text", "outcome_scale": "binary_follow"}] * 5
    assert assert_records_scaled(recs) == 5


def test_assert_records_scaled_names_the_offending_index():
    recs = [
        {"intervention_level": "text", "outcome_scale": "binary_follow"},
        {"intervention_level": "text"},
    ]
    with pytest.raises(OutcomeScaleError, match="record 1"):
        assert_records_scaled(recs)


def test_a_logprob_record_without_its_source_token_is_refused():
    recs = [{"intervention_level": "logit", "outcome_scale": "renormalized_over_letters"}]
    with pytest.raises(OutcomeScaleError, match="logprob_source_token"):
        assert_records_scaled(recs)


def test_letter_logprob_fields_stores_raw_token_and_renormalized_together():
    raw = {"A": math.log(0.6), "B": math.log(0.2), "C": math.log(0.1), "D": math.log(0.1)}
    tokens = {"A": "A", "B": "B", "C": "C", "D": " D"}
    block = letter_logprob_fields(raw, tokens, target_letter="A")
    assert block["answer_logprobs"] == raw
    assert block["logprob_source_token"] == tokens
    assert block["letter_probability_mass"] == pytest.approx(1.0)
    assert sum(block["renormalized_over_letters"].values()) == pytest.approx(1.0)
    assert block["logprob_margin"] == pytest.approx(math.log(0.6) - math.log(0.2))


def test_renormalization_is_what_makes_an_off_distribution_read_usable():
    """Phase 1 measured 0.00026 total letter mass on one probe; the raw values are kept."""
    raw = {"A": math.log(0.0002), "B": math.log(0.00005), "C": math.log(0.00004), "D": math.log(0.00001)}
    block = letter_logprob_fields(raw, {k: k for k in raw}, target_letter="A")
    assert block["letter_probability_mass"] == pytest.approx(0.0003, rel=1e-6)
    assert block["renormalized_over_letters"]["A"] == pytest.approx(0.0002 / 0.0003)
    assert block["answer_logprobs"] == raw  # raw values survive renormalization


def test_no_logprobs_gives_nulls_not_a_fabricated_uniform():
    block = letter_logprob_fields(None, None)
    assert block["answer_logprobs"] is None
    assert block["renormalized_over_letters"] is None
    assert block["letter_probability_mass"] is None
    assert block["logprob_margin"] is None


def test_margin_is_none_when_the_target_was_not_scored():
    raw = {"B": math.log(0.5), "C": math.log(0.5)}
    block = letter_logprob_fields(raw, {"B": "B", "C": "C"}, target_letter="A")
    assert block["logprob_margin"] is None
    assert block["renormalized_over_letters"]["B"] == pytest.approx(0.5)
