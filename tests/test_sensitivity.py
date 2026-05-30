"""Tests for the sequential-ignorability sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.sensitivity import (
    ConfoundedCoTConfig,
    fit_probit_mediation_map,
    probit_natural_effects,
    robustness_interval,
    sensitivity_sweep,
    simulate_confounded_cot,
)


def test_simulate_shapes_and_binary() -> None:
    cfg = ConfoundedCoTConfig(n_prompts=500, rng_seed=1)
    X, M, Y = simulate_confounded_cot(cfg)
    assert X.shape == M.shape == Y.shape == (500,)
    assert set(np.unique(X)).issubset({0, 1})
    assert set(np.unique(Y)).issubset({0, 1})
    assert M.dtype == float


def test_rho_zero_gives_uncorrelated_errors() -> None:
    cfg = ConfoundedCoTConfig(n_prompts=20_000, rho_confound=0.0, rng_seed=3)
    X, M, _ = simulate_confounded_cot(cfg)
    resid_m = M - cfg.gamma_xm * X
    assert resid_m.std() == pytest.approx(cfg.sigma_m, rel=0.05)


def test_positive_rho_induces_residual_correlation() -> None:
    cfg = ConfoundedCoTConfig(n_prompts=40_000, rho_confound=0.6, rng_seed=5)
    X, M, Y = simulate_confounded_cot(cfg)
    resid_m = M - cfg.gamma_xm * X
    hi = resid_m > np.median(resid_m)
    assert Y[hi].mean() > Y[~hi].mean()


def test_probit_natural_effects_te_is_nde_plus_nie() -> None:
    nde, nie, te = probit_natural_effects(
        alpha=0.3, beta=1.0, gamma=0.8, sigma_m=0.5, rho=0.0, n_mc=50_000, rng_seed=0
    )
    assert te == pytest.approx(nde + nie, abs=1e-9)


def test_fully_mediated_world_has_near_zero_nde() -> None:
    nde, nie, te = probit_natural_effects(
        alpha=0.0, beta=1.2, gamma=0.9, sigma_m=0.5, rho=0.0, n_mc=100_000, rng_seed=0
    )
    assert abs(nde) < 0.01
    assert nie > 0.05


def test_rho_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        probit_natural_effects(0.3, 1.0, 0.8, 0.5, rho=0.999)
    with pytest.raises(ValueError):
        simulate_confounded_cot(ConfoundedCoTConfig(rho_confound=0.99))


def test_fit_validates_inputs() -> None:
    X = np.array([0, 1, 2])
    M = np.array([0.1, 0.2, 0.3])
    Y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        fit_probit_mediation_map(X, M, Y, rho=0.0)


def test_sweep_returns_one_point_per_grid_value() -> None:
    cfg = ConfoundedCoTConfig(n_prompts=800, rng_seed=2)
    X, M, Y = simulate_confounded_cot(cfg)
    grid = np.array([-0.3, 0.0, 0.3])
    sweep = sensitivity_sweep(X, M, Y, rho_grid=grid, n_mc=20_000)
    assert len(sweep) == 3
    assert [p.rho for p in sweep] == [-0.3, 0.0, 0.3]


def test_robustness_interval_detects_sign_stability() -> None:
    cfg = ConfoundedCoTConfig(n_prompts=1500, beta_mediated=1.2, rng_seed=4)
    X, M, Y = simulate_confounded_cot(cfg)
    sweep = sensitivity_sweep(
        X, M, Y, rho_grid=np.round(np.arange(-0.4, 0.61, 0.2), 4), n_mc=30_000
    )
    band = robustness_interval(sweep, key="nie")
    assert band is not None
    lo, hi = band
    assert lo <= 0.0 <= hi


@pytest.mark.slow
def test_ignorability_assumption_biases_nie_but_true_rho_recovers() -> None:
    """The headline result: assuming rho=0 under real confounding is biased,
    and the sensitivity analysis at the true rho recovers the truth."""
    cfg = ConfoundedCoTConfig(
        n_prompts=6000, alpha_direct=0.3, beta_mediated=1.0,
        gamma_xm=0.8, sigma_m=0.5, rho_confound=0.5, rng_seed=7,
    )
    X, M, Y = simulate_confounded_cot(cfg)

    true_nde, true_nie, _ = probit_natural_effects(
        cfg.alpha_direct, cfg.beta_mediated, cfg.gamma_xm, cfg.sigma_m,
        cfg.rho_confound, n_mc=300_000, rng_seed=1,
    )

    a0, b0, g0, s0 = fit_probit_mediation_map(X, M, Y, 0.0)
    _, nie_ign, _ = probit_natural_effects(a0, b0, g0, s0, 0.0, n_mc=300_000, rng_seed=2)

    aT, bT, gT, sT = fit_probit_mediation_map(X, M, Y, cfg.rho_confound)
    _, nie_true, _ = probit_natural_effects(aT, bT, gT, sT, cfg.rho_confound, n_mc=300_000, rng_seed=3)

    bias_ignorability = abs(nie_ign - true_nie)
    err_true_rho = abs(nie_true - true_nie)

    assert err_true_rho < 0.02
    assert bias_ignorability > 0.03
    assert err_true_rho < bias_ignorability
    assert bT == pytest.approx(cfg.beta_mediated, abs=0.1)
