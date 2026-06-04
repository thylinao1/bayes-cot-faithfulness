"""Tests for the prior-sensitivity sweep.

Two layers, matching the repo's fast/slow split:

* Fast unit tests run a tiny, few-draw sweep so they fit in the normal suite.
  They check the table shape, the baseline bookkeeping, finiteness/sanity of the
  headline, and that the headline is robust to the *family* of the pooling-scale
  prior (the core robustness claim).
* The real-size fit lives behind ``@pytest.mark.slow``: it confirms the headline
  also recovers the true population slope and stays robust at a sample size where
  the estimate is not dominated by small-sample noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.hierarchical import (
    HierarchicalCoTConfig,
    simulate_hierarchical_cot,
)
from bayes_cot_faithfulness.prior_sensitivity import (
    PriorSpec,
    default_prior_specs,
    pooling_family_specs,
    prior_sensitivity_sweep,
)

# Tiny, deterministic fit so the unit tests stay in the fast suite.
_FAST_FIT = dict(n_samples=150, n_tune=150, n_chains=2, random_seed=0)


def _tiny_data(n_groups: int = 3, seed: int = 11):
    cfg = HierarchicalCoTConfig(
        n_groups=n_groups, mu_beta=1.3, min_per_group=20, max_per_group=35, rng_seed=seed
    )
    group, X, M, Y, truth = simulate_hierarchical_cot(cfg)
    return group, X, M, Y, truth, cfg


@pytest.fixture(scope="module")
def family_sweep():
    """One tiny pooling-family sweep, shared across the structural assertions.

    These tests interrogate the same immutable result from different angles, so
    refitting per test would only burn time; the sweep is deterministic, so the
    shared object is stable. Tests that need a different configuration (custom
    parameter, error paths) fit their own.
    """
    group, X, M, Y, *_ = _tiny_data()
    return prior_sensitivity_sweep(group, X, M, Y, specs=pooling_family_specs(), **_FAST_FIT)


def test_spec_helpers_are_one_at_a_time() -> None:
    # Arrange / Act
    family = pooling_family_specs()
    full = default_prior_specs()

    # Assert: family sweep changes only tau_beta and keeps mu_beta_sd fixed;
    # the full sweep is the family sweep plus the two coefficient-sd rows.
    baseline = family[0]
    assert all(s.mu_beta_sd == baseline.mu_beta_sd for s in family)
    assert {s.tau_beta_family for s in family} == {"halfnormal", "exponential", "halfstudentt"}
    assert len(full) == len(family) + 2
    assert full[:len(family)] == family


def test_sweep_returns_one_row_per_spec(family_sweep) -> None:
    # Arrange
    specs = pooling_family_specs()

    # Act
    res = family_sweep

    # Assert
    assert len(res.rows) == len(specs)
    assert [r.label for r in res.rows] == [s.label for s in specs]


def test_baseline_row_has_zero_shift_and_is_flagged(family_sweep) -> None:
    # Act
    res = family_sweep

    # Assert: exactly one baseline, it is the first row, and its shift is zero.
    baselines = [r for r in res.rows if r.is_baseline]
    assert len(baselines) == 1
    assert res.rows[0].is_baseline
    assert res.rows[0].shift_vs_baseline == 0.0
    assert res.baseline_label == res.rows[0].label


def test_headline_is_finite_and_in_sane_range(family_sweep) -> None:
    # Act
    res = family_sweep

    # Assert: every estimate and HDI bound is finite, the HDI brackets the mean,
    # and the slope sits in a wide-but-bounded plausible band.
    for r in res.rows:
        assert np.isfinite(r.estimate)
        assert np.isfinite(r.hdi_low)
        assert np.isfinite(r.hdi_high)
        assert r.hdi_low <= r.estimate <= r.hdi_high
        assert -2.0 < r.estimate < 6.0


def test_headline_robust_to_pooling_prior_family(family_sweep) -> None:
    """The faithfulness headline barely moves when only the pooling-scale prior
    family changes. This is the robustness payoff of the sweep."""
    # Assert: loose threshold (observed shift is ~0.02; this guards against a
    # regression that lets the prior family drive the estimate).
    assert family_sweep.max_abs_shift < 0.15


def test_max_abs_shift_matches_row_shifts(family_sweep) -> None:
    # Arrange
    res = family_sweep

    # Act
    expected = max(abs(r.shift_vs_baseline) for r in res.rows if not r.is_baseline)

    # Assert
    assert res.max_abs_shift == pytest.approx(expected)


def test_custom_parameter_can_be_swept() -> None:
    """The headline parameter is configurable; tau_beta is a valid alternative."""
    # Arrange
    group, X, M, Y, *_ = _tiny_data()

    # Act
    res = prior_sensitivity_sweep(
        group, X, M, Y, specs=pooling_family_specs(), parameter="tau_beta", **_FAST_FIT
    )

    # Assert: tau_beta is a positive scale, so every estimate is non-negative.
    assert res.parameter == "tau_beta"
    assert all(r.estimate >= 0.0 for r in res.rows)


def test_unknown_family_raises() -> None:
    # Arrange
    group, X, M, Y, *_ = _tiny_data()
    bad = (PriorSpec(label="bogus", tau_beta_family="cauchy"),)

    # Act / Assert
    with pytest.raises(ValueError):
        prior_sensitivity_sweep(group, X, M, Y, specs=bad, **_FAST_FIT)


def test_empty_specs_raises() -> None:
    group, X, M, Y, *_ = _tiny_data()
    with pytest.raises(ValueError):
        prior_sensitivity_sweep(group, X, M, Y, specs=(), **_FAST_FIT)


def test_validation_rejects_noncontiguous_groups() -> None:
    # Arrange: group index jumps from 0 to 2.
    group = np.array([0, 2, 2])
    X = np.array([0, 1, 0])
    M = np.array([0.1, 0.2, 0.3])
    Y = np.array([0, 1, 1])

    # Act / Assert
    with pytest.raises(ValueError):
        prior_sensitivity_sweep(group, X, M, Y, specs=pooling_family_specs(), **_FAST_FIT)


def test_validation_rejects_mismatched_lengths() -> None:
    group = np.array([0, 1])
    X = np.array([0, 1])
    M = np.array([0.1, 0.2, 0.3])
    Y = np.array([0, 1])
    with pytest.raises(ValueError):
        prior_sensitivity_sweep(group, X, M, Y, specs=pooling_family_specs(), **_FAST_FIT)


@pytest.mark.slow
def test_full_sweep_recovers_truth_and_is_family_robust() -> None:
    """At a realistic sample size the headline sits near the true population slope
    and the pooling-family swap moves it only a little, while the full table
    (including the coefficient-sd probe) stays finite and sign-correct."""
    # Arrange
    cfg = HierarchicalCoTConfig(
        n_groups=10, mu_beta=1.2, tau_beta=0.4,
        min_per_group=60, max_per_group=120, rng_seed=21,
    )
    group, X, M, Y, truth = simulate_hierarchical_cot(cfg)

    # Act: a heavier, still-deterministic fit over the full default sweep.
    res = prior_sensitivity_sweep(
        group, X, M, Y, specs=default_prior_specs(),
        n_samples=700, n_tune=700, n_chains=2, random_seed=0,
    )
    family_only = prior_sensitivity_sweep(
        group, X, M, Y, specs=pooling_family_specs(),
        n_samples=700, n_tune=700, n_chains=2, random_seed=0,
    )

    # Assert: baseline headline recovers the true population slope; the faithful
    # path is clearly positive under every prior; the family swap is tight.
    baseline = res.rows[0]
    assert baseline.estimate == pytest.approx(cfg.mu_beta, abs=0.35)
    assert all(r.estimate > 0.3 for r in res.rows)
    assert all(np.isfinite([r.hdi_low, r.hdi_high]).all() for r in res.rows)
    assert family_only.max_abs_shift < 0.15
