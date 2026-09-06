"""Synthetic CoT trace generator with known causal structure.

The data-generating process mimics a prompt -> chain-of-thought -> answer pipeline
where we control the true natural-direct and natural-indirect effects exactly.

Structural equations:

    X ~ Bernoulli(0.5)                          # prompt feature
    M | X ~ Normal(mu_m + gamma * X, sigma_m)   # CoT-as-mediator
    Y | X, M ~ Bernoulli(sigmoid(alpha0 + alpha * X + beta * M))

``mu_m`` and ``alpha0`` are the mediator and outcome baseline intercepts added
in the 2026-09-07 estimator repair. Both default to 0.0, which reproduces the
pre-repair draws bit for bit; set them to build the offset worlds that the
intercept-free estimator got wrong (see ``tests/test_offset_null.py``).

With these equations the natural direct and indirect effects on the probability
scale are well-defined and can be computed analytically by Monte Carlo
integration over M (see ``effects.monte_carlo_true_effects``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticCoTConfig:
    """Parameters of the synthetic CoT data-generating process.

    Attributes
    ----------
    n_prompts:
        Number of (prompt, CoT, answer) traces to draw.
    alpha_direct:
        Coefficient on X in the outcome equation. Controls the direct path
        from prompt to answer that does *not* flow through CoT. Set to 0
        to simulate a fully-mediated (perfectly faithful) world.
    beta_mediated:
        Coefficient on M in the outcome equation. Controls how strongly CoT
        drives the answer.
    gamma_xm:
        Coefficient on X in the mediator equation. Controls how strongly
        the prompt shifts the CoT.
    sigma_m:
        Std-dev of CoT noise.
    rng_seed:
        Seed for reproducibility.
    mu_m:
        Mediator baseline: E[M | X=0]. A step count or truncation depth has a
        natural baseline well away from zero; forcing it to zero is what made
        the pre-repair estimator read a control-arm level as a treatment shift.
    alpha0:
        Outcome baseline on the logit scale. With ``alpha = beta = 0`` the
        answer rate is ``sigmoid(alpha0)`` rather than 0.5.
    """

    n_prompts: int = 400
    alpha_direct: float = 0.3
    beta_mediated: float = 1.5
    gamma_xm: float = 0.8
    sigma_m: float = 0.5
    rng_seed: int = 42
    mu_m: float = 0.0
    alpha0: float = 0.0


def simulate_cot_trace(
    config: SyntheticCoTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n_prompts`` synthetic (X, M, Y) traces.

    Returns
    -------
    X : ndarray of shape (n_prompts,), dtype int
        Binary prompt feature.
    M : ndarray of shape (n_prompts,), dtype float
        Continuous CoT-as-mediator.
    Y : ndarray of shape (n_prompts,), dtype int
        Binary answer.
    """
    rng = np.random.default_rng(config.rng_seed)

    X = rng.binomial(1, 0.5, size=config.n_prompts)

    mean_m = config.mu_m + config.gamma_xm * X
    M = rng.normal(mean_m, config.sigma_m)

    logit_y = config.alpha0 + config.alpha_direct * X + config.beta_mediated * M
    p_y = _sigmoid(logit_y)
    Y = rng.binomial(1, p_y)

    return X.astype(int), M.astype(float), Y.astype(int)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
