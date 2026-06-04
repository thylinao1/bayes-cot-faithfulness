"""Tests for the statistical guardrails: sample-ratio, attrition, power, CIs.

These guard the experiment arms the same way SRM and power checks guard an A/B
test: a differential dropout across arms biases the mediation effect, and a 0%
follow rate on a tiny clean-correct set says nothing without an upper confidence
bound. Everything here is closed-form and offline.
"""

from __future__ import annotations

import pytest

from bayes_cot_faithfulness.guardrails import (
    attrition_balance,
    minimum_detectable_rate,
    proportion_ci_upper,
    rule_of_three_upper,
    srm_test,
)


# --------------------------------------------------------------------------- #
# srm_test
# --------------------------------------------------------------------------- #
def test_srm_balanced_counts_not_flagged() -> None:
    # Arrange: two arms at the expected 50/50 split, with ordinary noise.
    counts = (505, 495)

    # Act
    result = srm_test(counts)

    # Assert
    assert result.p_value > 0.05
    assert not result.flagged
    assert result.dof == 1


def test_srm_skewed_counts_flagged() -> None:
    # Arrange: a 50/50 design that actually delivered 900/100 is a gross mismatch.
    counts = (900, 100)

    # Act
    result = srm_test(counts)

    # Assert
    assert result.p_value < 1e-3
    assert result.flagged


def test_srm_dof_is_k_minus_one() -> None:
    # The classic trap: two arms is dof 1, not dof 2.
    assert srm_test((500, 500)).dof == 1
    assert srm_test((300, 300, 300)).dof == 2


def test_srm_expected_is_equal_split_by_default() -> None:
    # Arrange / Act
    result = srm_test((10, 30))

    # Assert: 40 observed total splits evenly across two arms.
    assert result.expected == (20.0, 20.0)


def test_srm_custom_expected_ratios() -> None:
    # Arrange: a deliberate 3:1 design met exactly should not flag.
    result = srm_test((750, 250), expected_ratios=(3.0, 1.0))

    # Assert
    assert result.expected == (750.0, 250.0)
    assert not result.flagged
    assert result.p_value == pytest.approx(1.0, abs=1e-9)


def test_srm_low_expected_flag_set_for_tiny_cells() -> None:
    # Arrange: expected cell of 2 is below the chi-square rule of thumb (>= 5).
    result = srm_test((3, 1))

    # Assert: still computed, but warned.
    assert result.low_expected
    assert result.expected == (2.0, 2.0)


def test_srm_large_balanced_cells_not_low_expected() -> None:
    assert not srm_test((500, 500)).low_expected


# --------------------------------------------------------------------------- #
# attrition_balance
# --------------------------------------------------------------------------- #
def test_attrition_equal_survival_not_flagged() -> None:
    # Arrange: both arms keep 80% of what entered.
    entered = {"clean": 100, "hinted": 100}
    survived = {"clean": 80, "hinted": 80}

    # Act
    result = attrition_balance(entered, survived)

    # Assert
    assert not result.flagged
    assert result.p_value > 0.05
    assert result.survival_rates["clean"] == pytest.approx(0.8)
    assert result.survival_rates["hinted"] == pytest.approx(0.8)


def test_attrition_lopsided_dropout_flagged() -> None:
    # Arrange: clean arm keeps almost everything, hinted arm loses half.
    entered = {"clean": 500, "hinted": 500}
    survived = {"clean": 480, "hinted": 250}

    # Act
    result = attrition_balance(entered, survived)

    # Assert
    assert result.flagged
    assert result.p_value < 1e-3


def test_attrition_zero_dropout_is_clean() -> None:
    # Arrange: nobody dropped in either arm.
    entered = {"clean": 50, "hinted": 50}
    survived = {"clean": 50, "hinted": 50}

    # Act
    result = attrition_balance(entered, survived)

    # Assert: degenerate table -> no imbalance, p == 1.0.
    assert result.p_value == 1.0
    assert not result.flagged
    assert result.survival_rates["clean"] == 1.0


def test_attrition_three_arms() -> None:
    entered = {"a": 100, "b": 100, "c": 100}
    survived = {"a": 90, "b": 88, "c": 91}
    result = attrition_balance(entered, survived)
    assert not result.flagged
    assert set(result.survival_rates) == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# proportion_ci_upper (Clopper-Pearson exact)
# --------------------------------------------------------------------------- #
def test_ci_upper_zero_of_eleven() -> None:
    # 0/11 has a wide one-sided 95% upper bound near 24%.
    x = proportion_ci_upper(0, 11)
    assert 0.20 < x < 0.30


def test_ci_upper_zero_of_twentytwo() -> None:
    # 0/22 tightens to roughly 12.7%.
    x = proportion_ci_upper(0, 22)
    assert 0.10 < x < 0.16


def test_ci_upper_all_successes_is_one() -> None:
    assert proportion_ci_upper(11, 11) == 1.0
    assert proportion_ci_upper(5, 5) == 1.0


def test_ci_upper_zero_n_is_one() -> None:
    assert proportion_ci_upper(0, 0) == 1.0


def test_ci_upper_shrinks_with_n() -> None:
    assert proportion_ci_upper(0, 100) < proportion_ci_upper(0, 11)


# --------------------------------------------------------------------------- #
# rule_of_three_upper
# --------------------------------------------------------------------------- #
def test_rule_of_three() -> None:
    assert rule_of_three_upper(11) == pytest.approx(3.0 / 11.0)
    assert rule_of_three_upper(300) == pytest.approx(0.01)


def test_rule_of_three_zero_n_is_one() -> None:
    assert rule_of_three_upper(0) == 1.0


def test_rule_of_three_approximates_clopper_pearson() -> None:
    # The 3/n shortcut should track the one-sided exact bound at k = 0, converging as
    # n grows (3/n sits a touch above the exact bound at small n).
    for n in (60, 100, 300):
        approx = rule_of_three_upper(n)
        exact = proportion_ci_upper(0, n)
        assert abs(approx - exact) < 0.01


# --------------------------------------------------------------------------- #
# minimum_detectable_rate
# --------------------------------------------------------------------------- #
def test_mdr_decreases_with_n() -> None:
    # More trials -> a smaller true rate becomes detectable.
    small_n = minimum_detectable_rate(11)
    large_n = minimum_detectable_rate(100)
    assert large_n < small_n


def test_mdr_is_a_probability() -> None:
    for n in (11, 22, 50, 100):
        mdr = minimum_detectable_rate(n)
        assert 0.0 <= mdr <= 1.0


def test_mdr_monotone_over_several_sizes() -> None:
    rates = [minimum_detectable_rate(n) for n in (11, 25, 50, 100, 200)]
    assert rates == sorted(rates, reverse=True)


def test_mdr_higher_power_needs_higher_rate() -> None:
    # Demanding more power at fixed n cannot lower the detectable rate.
    low = minimum_detectable_rate(50, power=0.8)
    high = minimum_detectable_rate(50, power=0.95)
    assert high >= low
