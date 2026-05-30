"""Hierarchical (partially pooled) mediation across prompts.

A single faithfulness number for a model is the wrong unit. Faithfulness varies
by prompt, by task type, and by seed, and some prompts have far fewer usable
traces than others. Estimating each prompt in isolation overfits the small ones;
pooling everything into one number hides real heterogeneity. The standard fix is
partial pooling: a hierarchical model where each prompt gets its own coefficients
drawn from a population distribution, so sparse prompts borrow strength from the
rest.

Model (logit outcome, shared mediator noise)::

    M_ig | X_ig ~ Normal(gamma_g * X_ig, sigma_m)
    Y_ig | X_ig, M_ig ~ Bernoulli(sigmoid(alpha_g * X_ig + beta_g * M_ig))

    alpha_g ~ Normal(mu_alpha, tau_alpha)     # population spread of the direct path
    beta_g  ~ Normal(mu_beta,  tau_beta)      # population spread of the faithful path
    gamma_g ~ Normal(mu_gamma, tau_gamma)

The population mean ``mu_beta`` is the model-level faithfulness slope with the
right uncertainty; ``tau_beta`` says how much prompts disagree. A non-centred
parameterisation keeps the geometry friendly for NUTS.

The natural extension is a full covariance on (alpha_g, beta_g, gamma_g) via
``pm.LKJCholeskyCov``; the independent-tau version here is the validated v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import arviz as az


@dataclass(frozen=True)
class HierarchicalCoTConfig:
    """Synthetic process with prompt-level heterogeneity in the causal paths."""

    n_groups: int = 8
    mu_alpha: float = 0.2
    mu_beta: float = 1.3
    mu_gamma: float = 0.8
    tau_alpha: float = 0.25
    tau_beta: float = 0.5
    tau_gamma: float = 0.2
    sigma_m: float = 0.5
    min_per_group: int = 30
    max_per_group: int = 250
    rng_seed: int = 42


def simulate_hierarchical_cot(
    config: HierarchicalCoTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Draw traces from several prompts with varying counts and true slopes.

    Returns
    -------
    group, X, M, Y : ndarrays of shape (n_total,)
        ``group`` is the integer prompt index for each observation.
    truth : dict
        True per-group (alpha_g, beta_g, gamma_g) and the hyper-means, for
        scoring recovery and shrinkage.
    """
    rng = np.random.default_rng(config.rng_seed)
    g = config.n_groups

    alpha_g = rng.normal(config.mu_alpha, config.tau_alpha, size=g)
    beta_g = rng.normal(config.mu_beta, config.tau_beta, size=g)
    gamma_g = rng.normal(config.mu_gamma, config.tau_gamma, size=g)
    n_per = rng.integers(config.min_per_group, config.max_per_group + 1, size=g)

    groups, Xs, Ms, Ys = [], [], [], []
    for gi in range(g):
        n = int(n_per[gi])
        X = rng.binomial(1, 0.5, size=n)
        M = rng.normal(gamma_g[gi] * X, config.sigma_m)
        logit = alpha_g[gi] * X + beta_g[gi] * M
        Y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
        groups.append(np.full(n, gi))
        Xs.append(X)
        Ms.append(M)
        Ys.append(Y)

    truth = {
        "alpha_g": alpha_g,
        "beta_g": beta_g,
        "gamma_g": gamma_g,
        "n_per_group": n_per,
        "mu_alpha": np.array(config.mu_alpha),
        "mu_beta": np.array(config.mu_beta),
        "mu_gamma": np.array(config.mu_gamma),
    }
    return (
        np.concatenate(groups).astype(int),
        np.concatenate(Xs).astype(int),
        np.concatenate(Ms).astype(float),
        np.concatenate(Ys).astype(int),
        truth,
    )


def fit_hierarchical_mediation(
    group: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_samples: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 0,
    cores: int | None = None,
    progressbar: bool = True,
) -> "az.InferenceData":
    """Fit the partially-pooled mediation model with a non-centred parameterisation.

    Pass ``cores=1`` for fully deterministic, in-process sampling (chains run
    sequentially); the default lets PyMC parallelise across cores.
    """
    import pymc as pm

    _validate(group, X, M, Y)
    n_groups = int(group.max()) + 1

    with pm.Model() as _model:
        mu_alpha = pm.Normal("mu_alpha", 0.0, 1.5)
        mu_beta = pm.Normal("mu_beta", 0.0, 2.0)
        mu_gamma = pm.Normal("mu_gamma", 0.0, 1.5)
        tau_alpha = pm.HalfNormal("tau_alpha", 1.0)
        tau_beta = pm.HalfNormal("tau_beta", 1.0)
        tau_gamma = pm.HalfNormal("tau_gamma", 1.0)
        sigma_m = pm.HalfNormal("sigma_m", 1.0)

        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, shape=n_groups)
        z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=n_groups)
        z_gamma = pm.Normal("z_gamma", 0.0, 1.0, shape=n_groups)

        alpha_g = pm.Deterministic("alpha_g", mu_alpha + tau_alpha * z_alpha)
        beta_g = pm.Deterministic("beta_g", mu_beta + tau_beta * z_beta)
        gamma_g = pm.Deterministic("gamma_g", mu_gamma + tau_gamma * z_gamma)

        pm.Normal("M_obs", mu=gamma_g[group] * X, sigma=sigma_m, observed=M)
        logit_y = alpha_g[group] * X + beta_g[group] * M
        pm.Bernoulli("Y_obs", logit_p=logit_y, observed=Y)

        sample_kwargs = dict(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )
        if cores is not None:
            sample_kwargs["cores"] = cores
        trace = pm.sample(**sample_kwargs)
    return trace


def no_pool_beta(
    group: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Per-group no-pooling logistic-slope estimate of beta (for shrinkage plots).

    Returns one beta per group (NaN if the group is degenerate, e.g. all one class).
    """
    _validate(group, X, M, Y)
    n_groups = int(group.max()) + 1
    betas = np.full(n_groups, np.nan)
    for gi in range(n_groups):
        mask = group == gi
        yg = Y[mask]
        if len(np.unique(yg)) < 2:
            continue
        design = np.column_stack([np.ones(mask.sum()), X[mask], M[mask]])
        betas[gi] = _logistic_fit(design, yg)[2]
    return betas


def _logistic_fit(design: np.ndarray, y: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Newton-Raphson logistic regression. Returns coefficient vector."""
    beta = np.zeros(design.shape[1])
    for _ in range(n_iter):
        eta = design @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = design.T @ (y - p)
        hess = design.T @ (design * w[:, None])
        try:
            step = np.linalg.solve(hess + 1e-6 * np.eye(design.shape[1]), grad)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _validate(group: np.ndarray, X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> None:
    if not (len(group) == len(X) == len(M) == len(Y)):
        raise ValueError("group, X, M, Y must all have the same length.")
    if group.min() != 0 or set(np.unique(group)) != set(range(int(group.max()) + 1)):
        raise ValueError("group must be contiguous integers starting at 0.")
    if not set(np.unique(X)).issubset({0, 1}):
        raise ValueError("X must be binary {0, 1}.")
    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be binary {0, 1}.")
