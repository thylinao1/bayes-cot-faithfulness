"""Tests for the synthetic CoT trace generator."""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig, _sigmoid, simulate_cot_trace


def test_shapes_match_config():
    config = SyntheticCoTConfig(n_prompts=200, rng_seed=0)
    X, M, Y = simulate_cot_trace(config)
    assert X.shape == (200,)
    assert M.shape == (200,)
    assert Y.shape == (200,)


def test_X_and_Y_are_binary():
    config = SyntheticCoTConfig(n_prompts=500, rng_seed=1)
    X, _, Y = simulate_cot_trace(config)
    assert set(np.unique(X)).issubset({0, 1})
    assert set(np.unique(Y)).issubset({0, 1})


def test_reproducible_with_seed():
    config = SyntheticCoTConfig(n_prompts=100, rng_seed=42)
    X1, M1, Y1 = simulate_cot_trace(config)
    X2, M2, Y2 = simulate_cot_trace(config)
    assert np.array_equal(X1, X2)
    assert np.allclose(M1, M2)
    assert np.array_equal(Y1, Y2)


def test_different_seeds_differ():
    c1 = SyntheticCoTConfig(n_prompts=100, rng_seed=1)
    c2 = SyntheticCoTConfig(n_prompts=100, rng_seed=2)
    _, M1, _ = simulate_cot_trace(c1)
    _, M2, _ = simulate_cot_trace(c2)
    assert not np.allclose(M1, M2)


def test_gamma_shifts_mediator():
    """Larger gamma should produce a larger gap between E[M|X=1] and E[M|X=0]."""
    c_small = SyntheticCoTConfig(n_prompts=5_000, gamma_xm=0.1, rng_seed=0)
    c_large = SyntheticCoTConfig(n_prompts=5_000, gamma_xm=2.0, rng_seed=0)
    X1, M1, _ = simulate_cot_trace(c_small)
    X2, M2, _ = simulate_cot_trace(c_large)
    gap_small = M1[X1 == 1].mean() - M1[X1 == 0].mean()
    gap_large = M2[X2 == 1].mean() - M2[X2 == 0].mean()
    assert gap_large > gap_small


def test_beta_zero_means_M_does_not_affect_Y():
    """When beta=0, P(Y=1) should not depend on M."""
    config = SyntheticCoTConfig(
        n_prompts=10_000,
        alpha_direct=0.0,
        beta_mediated=0.0,
        gamma_xm=2.0,
        rng_seed=7,
    )
    _, _, Y = simulate_cot_trace(config)
    # With alpha=beta=0, P(Y=1) is exactly 0.5 regardless of X, M.
    assert abs(Y.mean() - 0.5) < 0.03


@pytest.mark.parametrize("z,expected", [(0.0, 0.5), (10.0, 0.999955), (-10.0, 0.000045)])
def test_sigmoid_endpoints(z, expected):
    assert abs(_sigmoid(np.array([z]))[0] - expected) < 1e-4
