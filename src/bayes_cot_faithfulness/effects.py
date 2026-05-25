"""Natural direct and natural indirect effects on the probability scale.

For the structural model

    X ~ Bernoulli(0.5)
    M | X ~ Normal(gamma * X, sigma_m)
    Y | X, M ~ Bernoulli(sigmoid(alpha * X + beta * M))

the natural direct effect (NDE) and natural indirect effect (NIE) on the
probability of Y are integrals over the mediator distribution. We compute
them by Monte Carlo integration -- both for ground truth (using the true
parameters of the data-generating process) and for posterior estimates
(using draws from a fitted Bayesian model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig, _sigmoid


@dataclass(frozen=True)
class PosteriorEffects:
    """Posterior summary of natural effects on the probability scale."""

    nde_mean: float
    nde_lo: float
    nde_hi: float
    nie_mean: float
    nie_lo: float
    nie_hi: float
    te_mean: float
    te_lo: float
    te_hi: float
    nde_samples: np.ndarray
    nie_samples: np.ndarray
    te_samples: np.ndarray

    def contains(self, true_nde: float, true_nie: float, true_te: float) -> dict[str, bool]:
        """Check whether the 95% credible interval contains the truth."""
        return {
            "nde": self.nde_lo <= true_nde <= self.nde_hi,
            "nie": self.nie_lo <= true_nie <= self.nie_hi,
            "te": self.te_lo <= true_te <= self.te_hi,
        }


def monte_carlo_true_effects(
    config: SyntheticCoTConfig,
    n_mc: int = 200_000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Compute ground-truth NDE, NIE, and total effect via Monte Carlo.

    Definitions (probability scale):

        NDE = E[Y | do(X=1), M ~ M(X=0)] - E[Y | do(X=0), M ~ M(X=0)]
        NIE = E[Y | do(X=1), M ~ M(X=1)] - E[Y | do(X=1), M ~ M(X=0)]
        TE  = NDE + NIE

    Returns
    -------
    (nde, nie, te) : tuple of floats
    """
    rng = np.random.default_rng(rng_seed)

    m_under_x0 = rng.normal(0.0, config.sigma_m, size=n_mc)
    m_under_x1 = rng.normal(config.gamma_xm, config.sigma_m, size=n_mc)

    py_x0_m0 = _sigmoid(config.alpha_direct * 0 + config.beta_mediated * m_under_x0).mean()
    py_x1_m0 = _sigmoid(config.alpha_direct * 1 + config.beta_mediated * m_under_x0).mean()
    py_x1_m1 = _sigmoid(config.alpha_direct * 1 + config.beta_mediated * m_under_x1).mean()

    nde = py_x1_m0 - py_x0_m0
    nie = py_x1_m1 - py_x1_m0
    te = nde + nie
    return float(nde), float(nie), float(te)


def posterior_natural_effects(
    alpha_samples: np.ndarray,
    beta_samples: np.ndarray,
    gamma_samples: np.ndarray,
    sigma_m_samples: np.ndarray,
    n_mc_per_draw: int = 2_000,
    cri: float = 0.95,
    rng_seed: int = 0,
) -> PosteriorEffects:
    """Compute posterior over NDE / NIE / TE from MCMC parameter samples.

    For each posterior draw (alpha, beta, gamma, sigma_m), we do a small Monte
    Carlo integration over the mediator distribution to convert from the
    coefficients to the probability-scale effects.
    """
    rng = np.random.default_rng(rng_seed)
    n_draws = len(alpha_samples)
    if not (len(beta_samples) == len(gamma_samples) == len(sigma_m_samples) == n_draws):
        raise ValueError("All posterior sample arrays must have the same length.")

    nde_samples = np.empty(n_draws)
    nie_samples = np.empty(n_draws)

    for i in range(n_draws):
        alpha = alpha_samples[i]
        beta = beta_samples[i]
        gamma = gamma_samples[i]
        sigma_m = sigma_m_samples[i]

        m0 = rng.normal(0.0, sigma_m, size=n_mc_per_draw)
        m1 = rng.normal(gamma, sigma_m, size=n_mc_per_draw)

        py_x0_m0 = _sigmoid(alpha * 0 + beta * m0).mean()
        py_x1_m0 = _sigmoid(alpha * 1 + beta * m0).mean()
        py_x1_m1 = _sigmoid(alpha * 1 + beta * m1).mean()

        nde_samples[i] = py_x1_m0 - py_x0_m0
        nie_samples[i] = py_x1_m1 - py_x1_m0

    te_samples = nde_samples + nie_samples
    lo_q = (1.0 - cri) / 2.0
    hi_q = 1.0 - lo_q

    return PosteriorEffects(
        nde_mean=float(nde_samples.mean()),
        nde_lo=float(np.quantile(nde_samples, lo_q)),
        nde_hi=float(np.quantile(nde_samples, hi_q)),
        nie_mean=float(nie_samples.mean()),
        nie_lo=float(np.quantile(nie_samples, lo_q)),
        nie_hi=float(np.quantile(nie_samples, hi_q)),
        te_mean=float(te_samples.mean()),
        te_lo=float(np.quantile(te_samples, lo_q)),
        te_hi=float(np.quantile(te_samples, hi_q)),
        nde_samples=nde_samples,
        nie_samples=nie_samples,
        te_samples=te_samples,
    )
