"""Tests for the partially-pooled hierarchical mediation model."""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.hierarchical import (
    HierarchicalCoTConfig,
    fit_hierarchical_mediation,
    no_pool_beta,
    simulate_hierarchical_cot,
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
