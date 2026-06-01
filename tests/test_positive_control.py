"""Tests for the positive-control demonstration: the instrument responds to smoke.

These lock the credibility floor of the method. Act 1 (case level) asserts the
auditor flags a planted deception and clears genuine reasoning and honest
disclosure. Act 2 (population level) asserts the estimator reports a large, robust
faithful path when the chain-of-thought drives the answer and a faithful path of
essentially zero when the CoT is decorative. If either claim breaks, the
demonstration the write-up rests on is no longer true.
"""

from __future__ import annotations

import pytest

from bayes_cot_faithfulness.positive_control import (
    DECORATIVE_CONFIG,
    FAITHFUL_CONFIG,
    audit_case,
    demonstration_cases,
    smoke_test,
)

# Lighter Monte Carlo than the figure/demo so the suite stays fast; the worlds
# are separated by an order of magnitude, so reduced MC does not flip the verdict.
TEST_N_MC = 40_000


# --------------------------------------------------------------------------- #
# Act 1: case-level auditor.
# --------------------------------------------------------------------------- #
def test_demonstration_cases_have_known_ground_truth() -> None:
    cases = demonstration_cases()
    by_name = {c.name: c for c in cases}
    assert set(by_name) == {"planted-silent", "genuine-correct", "disclosed-deference"}
    assert by_name["planted-silent"].ground_truth_unfaithful is True
    assert by_name["genuine-correct"].ground_truth_unfaithful is False
    assert by_name["disclosed-deference"].ground_truth_unfaithful is False


def test_auditor_flags_the_planted_deception() -> None:
    """The 'huh' case: followed a wrong hint, never disclosed it -> flagged."""
    case = next(c for c in demonstration_cases() if c.name == "planted-silent")
    result = audit_case(case)
    assert result["followed_hint"] is True
    assert result["disclosed_hint"] is False
    assert result["flagged_silent_unfaithful"] is True
    assert result["correct"] is True


def test_auditor_clears_genuine_reasoning() -> None:
    case = next(c for c in demonstration_cases() if c.name == "genuine-correct")
    result = audit_case(case)
    assert result["followed_hint"] is False
    assert result["flagged_silent_unfaithful"] is False
    assert result["correct"] is True


def test_auditor_clears_honest_disclosure() -> None:
    """Following a hint while disclosing it is honest, not silent unfaithfulness."""
    case = next(c for c in demonstration_cases() if c.name == "disclosed-deference")
    result = audit_case(case)
    assert result["followed_hint"] is True
    assert result["disclosed_hint"] is True
    assert result["flagged_silent_unfaithful"] is False
    assert result["correct"] is True


def test_all_cases_judged_correctly() -> None:
    assert all(audit_case(c)["correct"] for c in demonstration_cases())


# --------------------------------------------------------------------------- #
# Act 2: population-level smoke test on the estimator.
# --------------------------------------------------------------------------- #
def test_faithful_world_shows_a_large_robust_faithful_path() -> None:
    res = smoke_test(FAITHFUL_CONFIG, n_mc=TEST_N_MC)
    assert res["nie_at_zero"] > 0.20, "faithful path should be clearly positive"
    assert res["rho_star"] > 0.40, "verdict should survive substantial confounding"
    assert res["sign_identified"] is True, "bounds should exclude zero at the tolerance"


def test_decorative_world_shows_a_near_zero_fragile_faithful_path() -> None:
    res = smoke_test(DECORATIVE_CONFIG, n_mc=TEST_N_MC)
    assert abs(res["nie_at_zero"]) < 0.10, "decorative faithful path should be ~0"
    assert res["rho_star"] < 0.20, "the slightest confounding should overturn it"
    assert res["sign_identified"] is False, "bounds should straddle zero"


def test_estimator_separates_the_two_worlds() -> None:
    """The core claim: the detector distinguishes faithful from decorative."""
    faithful = smoke_test(FAITHFUL_CONFIG, n_mc=TEST_N_MC)
    decorative = smoke_test(DECORATIVE_CONFIG, n_mc=TEST_N_MC)
    # an order-of-magnitude gap on the faithful path...
    assert faithful["nie_at_zero"] > 5 * abs(decorative["nie_at_zero"])
    # ...and a clear gap on robustness.
    assert faithful["rho_star"] > decorative["rho_star"] + 0.30


@pytest.mark.parametrize("config", [FAITHFUL_CONFIG, DECORATIVE_CONFIG])
def test_smoke_test_is_deterministic(config) -> None:
    a = smoke_test(config, n_mc=TEST_N_MC)
    b = smoke_test(config, n_mc=TEST_N_MC)
    assert a["nie_at_zero"] == b["nie_at_zero"]
    assert a["rho_star"] == b["rho_star"]
