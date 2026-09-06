"""Bayesian mediation model in PyMC.

The model jointly fits:

    Mediator equation:   M | X ~ Normal(mu_m + gamma * X, sigma_m)
    Outcome equation:    Y | X, M ~ Bernoulli(sigmoid(alpha0 + alpha * X + beta * M))

with weakly informative priors. Posterior samples over (alpha, beta, gamma,
sigma_m, mu_m, alpha0) are then converted into a posterior over the natural
direct and natural indirect effects on the probability scale by Monte Carlo
integration (see ``effects.posterior_natural_effects``).

``mu_m`` and ``alpha0`` are the baseline intercepts added in the 2026-09-07
estimator repair and are on by default. Without them the model asserts
``E[M | X=0] = 0`` and a clean-arm answer rate of 0.5, and on a mediator with a
natural baseline level (a CoT step count, a truncation depth) it reads that
level as a treatment shift and reports mediation where there is none. Pass
``intercepts=False`` only to reproduce a pre-repair fit.
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
    intercepts: bool = True,
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
    intercepts : bool
        ``True`` (default) adds the mediator baseline ``mu_m`` and the outcome
        baseline ``alpha0``. ``False`` restores the pre-2026-09-07 model, whose
        graph and variable names are unchanged.

    Returns
    -------
    arviz.InferenceData
        Trace with posterior samples for alpha, beta, gamma, sigma_m and, when
        ``intercepts`` is on, mu_m and alpha0.
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

        mean_m = gamma * X
        logit_y = alpha * X + beta * M
        if intercepts:
            # Weakly informative and scale-aware: the mediator baseline prior is
            # centred on the observed control-arm level rather than on zero, so a
            # step count in the tens is not fought by the prior.
            mu_m = pm.Normal("mu_m", mu=float(np.mean(M)), sigma=float(_prior_scale(M)))
            alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.5)
            mean_m = mu_m + mean_m
            logit_y = alpha0 + logit_y
        pm.Normal("M_obs", mu=mean_m, sigma=sigma_m, observed=M)
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


def _prior_scale(M: np.ndarray) -> float:
    """Prior sd for the mediator baseline: the mediator's own spread, floored."""
    return float(max(np.std(M), 1.0))


def extract_parameter_samples(
    trace: "az.InferenceData",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten posterior samples for (alpha, beta, gamma, sigma_m).

    Deliberately unchanged by the intercept repair, so every existing four-way
    unpack keeps working. Use ``extract_intercept_samples`` for the baselines
    and pass both into ``effects.posterior_natural_effects``.
    """
    posterior = trace.posterior
    alpha = posterior["alpha"].values.flatten()
    beta = posterior["beta"].values.flatten()
    gamma = posterior["gamma"].values.flatten()
    sigma_m = posterior["sigma_m"].values.flatten()
    return alpha, beta, gamma, sigma_m


def extract_intercept_samples(
    trace: az.InferenceData,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten posterior samples for the baselines ``(mu_m, alpha0)``.

    A trace from a fit run with ``intercepts=False`` has no such variables; this
    returns exact zeros of the right length in that case, which is precisely the
    constraint that fit imposed.
    """
    posterior = trace.posterior
    n_draws = posterior["alpha"].values.size
    if "mu_m" in posterior:
        mu_m = posterior["mu_m"].values.flatten()
        alpha0 = posterior["alpha0"].values.flatten()
        return mu_m, alpha0
    return np.zeros(n_draws), np.zeros(n_draws)
