"""Bayesian mediation model in PyMC.

The model jointly fits:

    Mediator equation:   M | X ~ Normal(gamma * X, sigma_m)
    Outcome equation:    Y | X, M ~ Bernoulli(sigmoid(alpha * X + beta * M))

with weakly informative priors. Posterior samples over (alpha, beta, gamma,
sigma_m) are then converted into a posterior over the natural direct and
natural indirect effects on the probability scale by Monte Carlo integration
(see ``effects.posterior_natural_effects``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import arviz as az


def fit_mediation_model(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_samples: int = 1500,
    n_tune: int = 1500,
    n_chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 0,
    progressbar: bool = True,
) -> "az.InferenceData":
    """Fit a Bayesian mediation model and return posterior samples.

    Parameters
    ----------
    X : ndarray of shape (n,), int
        Binary prompt feature.
    M : ndarray of shape (n,), float
        Continuous CoT-as-mediator.
    Y : ndarray of shape (n,), int
        Binary answer.

    Returns
    -------
    arviz.InferenceData
        Trace with posterior samples for alpha, beta, gamma, sigma_m.
    """
    import pymc as pm

    if len(X) != len(M) or len(M) != len(Y):
        raise ValueError("X, M, Y must all have the same length.")
    if X.ndim != 1:
        raise ValueError("X must be 1-D.")
    if not set(np.unique(X)).issubset({0, 1}):
        raise ValueError("X must be binary {0, 1}.")
    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be binary {0, 1}.")

    with pm.Model() as _model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=2.0)
        gamma = pm.Normal("gamma", mu=0.0, sigma=1.5)
        sigma_m = pm.HalfNormal("sigma_m", sigma=1.0)

        mu_m = gamma * X
        pm.Normal("M_obs", mu=mu_m, sigma=sigma_m, observed=M)

        logit_y = alpha * X + beta * M
        pm.Bernoulli("Y_obs", logit_p=logit_y, observed=Y)

        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    return trace


def extract_parameter_samples(
    trace: "az.InferenceData",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten posterior samples for (alpha, beta, gamma, sigma_m)."""
    posterior = trace.posterior
    alpha = posterior["alpha"].values.flatten()
    beta = posterior["beta"].values.flatten()
    gamma = posterior["gamma"].values.flatten()
    sigma_m = posterior["sigma_m"].values.flatten()
    return alpha, beta, gamma, sigma_m
