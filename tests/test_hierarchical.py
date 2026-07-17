"""Tests for the partially-pooled hierarchical mediation model."""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.hierarchical import (
    HierarchicalCoTConfig,
    build_hierarchical_model,
    fit_hierarchical_mediation,
    no_pool_beta,
    simulate_hierarchical_cot,
    simulate_hierarchical_cot_ext,
)


def test_simulate_shapes_and_groups() -> None:
    cfg = HierarchicalCoTConfig(n_groups=5, rng_seed=1)
    group, X, M, Y, truth = simulate_hierarchical_cot(cfg)
    n = len(group)
    assert len(X) == len(M) == len(Y) == n
    assert set(np.unique(group)) == set(range(5))
    assert truth["beta_g"].shape == (5,)
    assert set(np.unique(X)).issubset({0, 1})
    assert set(np.unique(Y)).issubset({0, 1})


def test_group_counts_vary() -> None:
    cfg = HierarchicalCoTConfig(n_groups=8, rng_seed=2)
    _, _, _, _, truth = simulate_hierarchical_cot(cfg)
    counts = truth["n_per_group"]
    assert counts.min() >= cfg.min_per_group
    assert counts.max() <= cfg.max_per_group
    assert counts.min() < counts.max()


def test_true_betas_spread_around_mu() -> None:
    cfg = HierarchicalCoTConfig(n_groups=40, tau_beta=0.5, rng_seed=3)
    _, _, _, _, truth = simulate_hierarchical_cot(cfg)
    assert truth["beta_g"].mean() == pytest.approx(cfg.mu_beta, abs=0.25)
    assert truth["beta_g"].std() == pytest.approx(cfg.tau_beta, abs=0.25)


def test_no_pool_beta_recovers_sign() -> None:
    cfg = HierarchicalCoTConfig(n_groups=6, mu_beta=1.3, rng_seed=4)
    group, X, M, Y, truth = simulate_hierarchical_cot(cfg)
    betas = no_pool_beta(group, X, M, Y)
    finite = betas[np.isfinite(betas)]
    assert finite.size >= 4
    assert np.median(finite) > 0


def test_validate_rejects_noncontiguous_groups() -> None:
    group = np.array([0, 2, 2])
    X = np.array([0, 1, 0])
    M = np.array([0.1, 0.2, 0.3])
    Y = np.array([0, 1, 1])
    with pytest.raises(ValueError):
        no_pool_beta(group, X, M, Y)
    with pytest.raises(ValueError):
        fit_hierarchical_mediation(group, X, M, Y)


@pytest.mark.slow
def test_hierarchical_recovers_mu_beta_and_shrinks() -> None:
    """Partial pooling recovers the population slope and produces better per-group
    estimates than no-pooling when groups are small and noisy.

    Sampling is pinned to one core for determinism (chains run in-process)."""
    cfg = HierarchicalCoTConfig(
        n_groups=12, mu_beta=1.1, tau_beta=0.4,
        min_per_group=40, max_per_group=90, rng_seed=11,
    )
    group, X, M, Y, truth = simulate_hierarchical_cot(cfg)

    trace = fit_hierarchical_mediation(
        group, X, M, Y, n_samples=800, n_tune=800, n_chains=2,
        random_seed=0, cores=1, progressbar=False,
    )
    post = trace.posterior

    mu_beta = post["mu_beta"].values.flatten()
    lo, hi = np.quantile(mu_beta, [0.025, 0.975])
    assert lo <= cfg.mu_beta <= hi

    pooled_beta_g = post["beta_g"].values.reshape(-1, cfg.n_groups).mean(axis=0)
    nopool = no_pool_beta(group, X, M, Y)
    true_beta_g = truth["beta_g"]
    finite = np.isfinite(nopool)
    assert finite.sum() >= 8

    pooled_rmse = np.sqrt(np.mean((pooled_beta_g[finite] - true_beta_g[finite]) ** 2))
    nopool_rmse = np.sqrt(np.mean((nopool[finite] - true_beta_g[finite]) ** 2))
    assert pooled_rmse < nopool_rmse


# --- Phase-2 extensions: T12 covariates and A2 hint-type grouping factor ---

# The v1 model's variables, so a no-extension build can be checked to add nothing.
_V1_NAMED_VARS = {
    "mu_alpha", "mu_beta", "mu_gamma",
    "tau_alpha", "tau_beta", "tau_gamma", "sigma_m",
    "z_alpha", "z_beta", "z_gamma",
    "alpha_g", "beta_g", "gamma_g",
    "M_obs", "Y_obs",
}


def _small_valid_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group = np.array([0, 0, 0, 1, 1, 1])
    X = np.array([0, 1, 0, 1, 0, 1])
    M = np.array([0.1, -0.2, 0.3, 0.0, 0.4, -0.1])
    Y = np.array([0, 1, 0, 1, 1, 0])
    return group, X, M, Y


def test_validate_rejects_3d_covariates() -> None:
    group, X, M, Y = _small_valid_arrays()
    covariates = np.zeros((len(group), 1, 1))  # 3-D, not (n_obs, k)
    with pytest.raises(ValueError):
        build_hierarchical_model(group, X, M, Y, covariates=covariates)


def test_validate_rejects_covariate_wrong_nobs() -> None:
    group, X, M, Y = _small_valid_arrays()
    covariates = np.zeros((len(group) + 2, 1))  # wrong number of rows
    with pytest.raises(ValueError):
        build_hierarchical_model(group, X, M, Y, covariates=covariates)


def test_validate_rejects_nonfinite_covariates() -> None:
    group, X, M, Y = _small_valid_arrays()
    covariates = np.zeros((len(group), 1))
    covariates[0, 0] = np.nan
    with pytest.raises(ValueError):
        build_hierarchical_model(group, X, M, Y, covariates=covariates)


def test_validate_rejects_covariate_names_length_mismatch() -> None:
    group, X, M, Y = _small_valid_arrays()
    covariates = np.zeros((len(group), 2))  # k == 2
    with pytest.raises(ValueError):
        build_hierarchical_model(
            group, X, M, Y, covariates=covariates, covariate_names=["only_one"]
        )


def test_validate_rejects_noncontiguous_hint_type() -> None:
    group, X, M, Y = _small_valid_arrays()
    hint_type = np.array([0, 2, 2, 0, 2, 0])  # skips 1
    with pytest.raises(ValueError):
        build_hierarchical_model(group, X, M, Y, hint_type=hint_type)


def test_build_model_baseline_matches_v1_named_vars() -> None:
    cfg = HierarchicalCoTConfig(n_groups=4, rng_seed=7)
    group, X, M, Y, _ = simulate_hierarchical_cot(cfg)
    model = build_hierarchical_model(group, X, M, Y)
    assert set(model.named_vars) == _V1_NAMED_VARS
    for absent in ("delta", "alpha_h", "beta_h"):
        assert absent not in model.named_vars


def test_build_model_covariates_add_delta_only() -> None:
    cfg = HierarchicalCoTConfig(n_groups=4, rng_seed=7)
    group, X, M, Y, _, cov, _ = simulate_hierarchical_cot_ext(cfg)
    model = build_hierarchical_model(
        group, X, M, Y, covariates=cov, covariate_names=["clue_need"]
    )
    assert "delta" in model.named_vars
    for absent in ("alpha_h", "beta_h"):
        assert absent not in model.named_vars


def test_build_model_hint_type_adds_deviations_only() -> None:
    cfg = HierarchicalCoTConfig(n_groups=4, rng_seed=7)
    group, X, M, Y, hint_type, _, _ = simulate_hierarchical_cot_ext(cfg)
    model = build_hierarchical_model(group, X, M, Y, hint_type=hint_type)
    for present in ("alpha_h", "beta_h", "tau_alpha_h", "tau_beta_h"):
        assert present in model.named_vars
    assert "delta" not in model.named_vars


def test_build_model_full_extension_has_all_new_vars() -> None:
    cfg = HierarchicalCoTConfig(n_groups=4, rng_seed=7)
    group, X, M, Y, hint_type, cov, _ = simulate_hierarchical_cot_ext(cfg)
    model = build_hierarchical_model(
        group, X, M, Y, covariates=cov, covariate_names=["clue_need"], hint_type=hint_type
    )
    added = set(model.named_vars) - _V1_NAMED_VARS
    assert {"delta", "alpha_h", "beta_h", "z_alpha_h", "z_beta_h"} <= added


def test_simulate_ext_shapes_dtypes_and_contiguity() -> None:
    cfg = HierarchicalCoTConfig(n_groups=6, rng_seed=5)
    group, X, M, Y, hint_type, cov, truth = simulate_hierarchical_cot_ext(
        cfg, n_hint_types=3
    )
    n = len(group)
    assert len(X) == len(M) == len(Y) == len(hint_type) == n
    assert cov.shape == (n, 1)
    assert group.dtype == int and hint_type.dtype == int
    assert X.dtype == int and Y.dtype == int and M.dtype == float and cov.dtype == float
    assert set(np.unique(hint_type)) == set(range(3))  # contiguous from 0
    assert np.all(np.isfinite(cov))
    assert cov.mean() == pytest.approx(0.0, abs=1e-9)
    assert cov.std() == pytest.approx(1.0, abs=1e-9)
    assert truth["beta_h"].shape == (3,)
    assert float(truth["covariate_effect"]) == pytest.approx(0.8)


def test_simulate_ext_is_deterministic_under_seed() -> None:
    cfg = HierarchicalCoTConfig(n_groups=5, rng_seed=9)
    a = simulate_hierarchical_cot_ext(cfg, n_hint_types=4, covariate_effect=1.1)
    b = simulate_hierarchical_cot_ext(cfg, n_hint_types=4, covariate_effect=1.1)
    for arr_a, arr_b in zip(a[:6], b[:6]):
        assert np.array_equal(arr_a, arr_b)
    assert np.array_equal(a[6]["beta_h"], b[6]["beta_h"])


@pytest.mark.slow
def test_hierarchical_recovers_delta_sign() -> None:
    """A clearly positive covariate effect is recovered with a positive delta posterior.

    Small but real fit (modest draws, one core for determinism); the assertion is only
    on the sign so it stays minutes-cheap."""
    cfg = HierarchicalCoTConfig(
        n_groups=8, min_per_group=40, max_per_group=90, rng_seed=21,
    )
    group, X, M, Y, hint_type, covariates, truth = simulate_hierarchical_cot_ext(
        cfg, n_hint_types=3, covariate_effect=1.2, hint_beta_spread=0.5,
    )
    assert float(truth["covariate_effect"]) > 0

    trace = fit_hierarchical_mediation(
        group, X, M, Y, n_samples=200, n_tune=200, n_chains=2,
        random_seed=0, cores=1, progressbar=False,
        covariates=covariates, covariate_names=["clue_need"], hint_type=hint_type,
    )
    delta = trace.posterior["delta"].values.flatten()
    assert delta.mean() > 0
