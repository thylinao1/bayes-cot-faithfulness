"""Tests for natural-effect computations."""

from __future__ import annotations

import numpy as np

from bayes_cot_faithfulness.effects import (
    monte_carlo_true_effects,
    posterior_natural_effects,
)
from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig


def test_total_effect_is_nde_plus_nie():
    config = SyntheticCoTConfig(
        alpha_direct=0.4,
        beta_mediated=1.0,
        gamma_xm=0.7,
        sigma_m=0.5,
    )
    nde, nie, te = monte_carlo_true_effects(config, n_mc=200_000, rng_seed=11)
    assert abs(te - (nde + nie)) < 1e-9


def test_fully_mediated_means_nde_is_zero():
    """If alpha=0 (no direct path), NDE should be ~ 0."""
    config = SyntheticCoTConfig(alpha_direct=0.0, beta_mediated=1.0, gamma_xm=1.0)
    nde, _, _ = monte_carlo_true_effects(config, n_mc=500_000, rng_seed=3)
    assert abs(nde) < 0.005


def test_no_mediation_means_nie_is_zero():
    """If gamma=0 (X does not move M), NIE should be ~ 0."""
    config = SyntheticCoTConfig(alpha_direct=1.0, beta_mediated=1.0, gamma_xm=0.0)
    _, nie, _ = monte_carlo_true_effects(config, n_mc=500_000, rng_seed=4)
    assert abs(nie) < 0.005


def test_posterior_effects_recover_truth_with_oracle_samples():
    """If we feed posterior_natural_effects the true coefficients (zero
    posterior variance), it should reproduce monte_carlo_true_effects exactly.
    """
    config = SyntheticCoTConfig(
        alpha_direct=0.3,
        beta_mediated=1.5,
        gamma_xm=0.8,
        sigma_m=0.5,
    )
    true_nde, true_nie, true_te = monte_carlo_true_effects(config, n_mc=400_000, rng_seed=0)

    n_draws = 200
    alpha_samples = np.full(n_draws, config.alpha_direct)
    beta_samples = np.full(n_draws, config.beta_mediated)
    gamma_samples = np.full(n_draws, config.gamma_xm)
    sigma_samples = np.full(n_draws, config.sigma_m)

    pe = posterior_natural_effects(
        alpha_samples, beta_samples, gamma_samples, sigma_samples,
        n_mc_per_draw=2_000, rng_seed=5,
    )

    # With oracle parameters and enough MC, the posterior point estimates
    # should sit right on top of the analytical truth.
    assert abs(pe.nde_mean - true_nde) < 0.01
    assert abs(pe.nie_mean - true_nie) < 0.01
    assert abs(pe.te_mean - true_te) < 0.01


def test_contains_returns_correct_booleans():
    """A trivial contains-check on a known credible interval."""
    n_draws = 100
    alpha_samples = np.full(n_draws, 0.0)
    beta_samples = np.full(n_draws, 0.0)
    gamma_samples = np.full(n_draws, 0.0)
    sigma_samples = np.full(n_draws, 1.0)

    pe = posterior_natural_effects(
        alpha_samples, beta_samples, gamma_samples, sigma_samples,
        n_mc_per_draw=1_000, rng_seed=99,
    )

    # Truth set to 0 should be contained; truth set to 100 should not.
    inside = pe.contains(true_nde=0.0, true_nie=0.0, true_te=0.0)
    outside = pe.contains(true_nde=100.0, true_nie=100.0, true_te=100.0)
    assert inside == {"nde": True, "nie": True, "te": True}
    assert outside == {"nde": False, "nie": False, "te": False}
