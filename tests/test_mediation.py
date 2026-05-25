"""Tests for the PyMC mediation estimator.

This module is gated behind a fast/slow flag because PyMC sampling is the
expensive step. The fast tests verify input validation; the slow test does
a small end-to-end recovery check.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.mediation import extract_parameter_samples, fit_mediation_model
from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig, simulate_cot_trace


def test_fit_rejects_mismatched_lengths():
    X = np.array([0, 1, 0])
    M = np.array([0.1, 0.2])
    Y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="same length"):
        fit_mediation_model(X, M, Y)


def test_fit_rejects_non_binary_X():
    X = np.array([0, 1, 2, 1])
    M = np.array([0.1, 0.2, 0.3, 0.4])
    Y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="binary"):
        fit_mediation_model(X, M, Y)


def test_fit_rejects_non_binary_Y():
    X = np.array([0, 1, 0, 1])
    M = np.array([0.1, 0.2, 0.3, 0.4])
    Y = np.array([0, 1, 2, 1])
    with pytest.raises(ValueError, match="binary"):
        fit_mediation_model(X, M, Y)


@pytest.mark.slow
def test_recovers_coefficients_on_synthetic_data():
    """End-to-end smoke test: on synthetic data with known coefficients,
    the posterior means should land within 3 posterior-std-devs of truth.

    Marked slow because PyMC sampling takes ~30s.
    """
    config = SyntheticCoTConfig(
        n_prompts=600,
        alpha_direct=0.4,
        beta_mediated=1.2,
        gamma_xm=0.7,
        sigma_m=0.4,
        rng_seed=2025,
    )
    X, M, Y = simulate_cot_trace(config)
    trace = fit_mediation_model(
        X, M, Y, n_samples=600, n_tune=600, n_chains=2, progressbar=False, random_seed=0,
    )
    alpha, beta, gamma, sigma_m = extract_parameter_samples(trace)

    def within_3std(mean: float, samples: np.ndarray, truth: float) -> bool:
        return abs(mean - truth) <= 3.0 * samples.std()

    assert within_3std(alpha.mean(), alpha, config.alpha_direct)
    assert within_3std(beta.mean(), beta, config.beta_mediated)
    assert within_3std(gamma.mean(), gamma, config.gamma_xm)
    assert within_3std(sigma_m.mean(), sigma_m, config.sigma_m)
