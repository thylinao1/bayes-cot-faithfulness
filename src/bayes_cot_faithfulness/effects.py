"""Natural direct and natural indirect effects on the probability scale.

For the structural model

    X ~ Bernoulli(0.5)
    M | X ~ Normal(mu_m + gamma * X, sigma_m)
    Y | X, M ~ Bernoulli(sigmoid(alpha0 + alpha * X + beta * M))

the natural direct effect (NDE) and natural indirect effect (NIE) on the
probability of Y are integrals over the mediator distribution. We compute
them by Monte Carlo integration, both for ground truth (using the true
parameters of the data-generating process) and for posterior estimates
(using draws from a fitted Bayesian model).

``mu_m`` and ``alpha0`` are the baseline intercepts added in the 2026-09-07
estimator repair. Omit them and the computation is the pre-repair one, which is
correct only when the mediator really is centred at zero in the control arm and
the clean-arm answer rate really is 0.5.

The structural model above is the logistic one used by ``synthetic.py``. The
estimator of record is the probit one (``sensitivity``, ``closed_form``,
``mediation.fit_mediation_model``), so ``posterior_natural_effects`` takes a
``link`` and defaults to ``"probit"``; ``monte_carlo_true_effects`` stays
logistic because the generator whose config it reads is logistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form_samples
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

    m_under_x0 = rng.normal(config.mu_m, config.sigma_m, size=n_mc)
    m_under_x1 = rng.normal(config.mu_m + config.gamma_xm, config.sigma_m, size=n_mc)

    a0 = config.alpha0
    py_x0_m0 = _sigmoid(a0 + config.alpha_direct * 0 + config.beta_mediated * m_under_x0).mean()
    py_x1_m0 = _sigmoid(a0 + config.alpha_direct * 1 + config.beta_mediated * m_under_x0).mean()
    py_x1_m1 = _sigmoid(a0 + config.alpha_direct * 1 + config.beta_mediated * m_under_x1).mean()

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
    mu_m_samples: np.ndarray | None = None,
    alpha0_samples: np.ndarray | None = None,
    link: str = "probit",
) -> PosteriorEffects:
    """Compute posterior over NDE / NIE / TE from MCMC parameter samples.

    ``link`` must be the link the draws were fitted under, and it is ``"probit"``
    by default because that is the link of the pre-registered estimand and of
    ``mediation.fit_mediation_model``. Prefer ``mediation.natural_effects_from_trace``,
    which reads the link off the trace instead of trusting a call site.

    Under ``"probit"`` the mediator integral is elementary, so the conversion is
    the exact closed form in ``closed_form`` (the same function the maximum-likelihood
    path uses, which is what keeps one definition of the estimand in the package)
    and ``n_mc_per_draw`` is ignored. Under ``"logit"`` there is no elementary
    integral, so each draw gets a small Monte Carlo integration over the mediator
    distribution; that branch is for the logistic generator in ``synthetic.py``.

    ``mu_m_samples`` and ``alpha0_samples`` are the baseline-intercept draws from
    a model fitted with ``intercepts=True``; pass them whenever the fit had
    intercepts, or the effects are integrated against the wrong mediator
    distribution and the wrong clean-arm baseline. Omitting them (the default)
    treats both intercepts as exactly zero, which is the pre-2026-09-07
    behaviour.
    """
    if link not in ("probit", "logit"):
        raise ValueError(f"link must be 'probit' or 'logit', got {link!r}.")
    rng = np.random.default_rng(rng_seed)
    n_draws = len(alpha_samples)
    if not (len(beta_samples) == len(gamma_samples) == len(sigma_m_samples) == n_draws):
        raise ValueError("All posterior sample arrays must have the same length.")
    mu_m_samples = np.zeros(n_draws) if mu_m_samples is None else np.asarray(mu_m_samples)
    alpha0_samples = (
        np.zeros(n_draws) if alpha0_samples is None else np.asarray(alpha0_samples)
    )
    if len(mu_m_samples) != n_draws or len(alpha0_samples) != n_draws:
        raise ValueError("Intercept sample arrays must match the other draws in length.")

    if link == "probit":
        nde_samples, nie_samples, _ = probit_natural_effects_closed_form_samples(
            np.asarray(alpha_samples, dtype=float),
            np.asarray(beta_samples, dtype=float),
            np.asarray(gamma_samples, dtype=float),
            np.asarray(sigma_m_samples, dtype=float),
            rho=0.0,
            mu_m=mu_m_samples,
            alpha0=alpha0_samples,
        )
        return _summarise(nde_samples, nie_samples, cri)

    nde_samples = np.empty(n_draws)
    nie_samples = np.empty(n_draws)

    for i in range(n_draws):
        alpha = alpha_samples[i]
        beta = beta_samples[i]
        gamma = gamma_samples[i]
        sigma_m = sigma_m_samples[i]
        mu_m = mu_m_samples[i]
        alpha0 = alpha0_samples[i]

        m0 = rng.normal(mu_m, sigma_m, size=n_mc_per_draw)
        m1 = rng.normal(mu_m + gamma, sigma_m, size=n_mc_per_draw)

        py_x0_m0 = _sigmoid(alpha0 + alpha * 0 + beta * m0).mean()
        py_x1_m0 = _sigmoid(alpha0 + alpha * 1 + beta * m0).mean()
        py_x1_m1 = _sigmoid(alpha0 + alpha * 1 + beta * m1).mean()

        nde_samples[i] = py_x1_m0 - py_x0_m0
        nie_samples[i] = py_x1_m1 - py_x1_m0

    return _summarise(nde_samples, nie_samples, cri)


def _summarise(nde_samples: np.ndarray, nie_samples: np.ndarray, cri: float) -> PosteriorEffects:
    """Means, equal-tailed credible intervals and the draws themselves."""
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
