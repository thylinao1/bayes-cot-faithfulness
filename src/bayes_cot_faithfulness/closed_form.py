"""Closed-form natural effects for the probit SCM, as an independent cross-check.

The natural effects in this project are computed by Monte Carlo integration over
the mediator (see ``sensitivity.probit_natural_effects`` and
``effects.posterior_natural_effects``). Monte Carlo is general but easy to get
subtly wrong: an off-by-one in which mediator distribution is paired with which
``do(X)`` arm silently biases the NDE/NIE. The cheapest insurance against that
class of bug is a second estimator that shares no code path and is derived by
hand, then a test that the two agree.

For the probit structural causal model used in the sensitivity analysis the
natural effects have an *exact* closed form, so the cross-check is tight (the
only disagreement with Monte Carlo is sampling noise that shrinks like
1/sqrt(n_mc)).

Probit SCM (see ``sensitivity``)::

    X ~ Bernoulli(0.5)
    (eps_M, eps_Y) ~ Normal2(0, [[sigma_m^2, rho*sigma_m], [rho*sigma_m, 1]])
    M = mu_m + gamma * X + eps_M
    Y = 1[alpha0 + alpha * X + beta * M + eps_Y > 0]

``mu_m`` and ``alpha0`` are the mediator and outcome baseline intercepts added
in the 2026-09-07 estimator repair. Both default to 0.0, which is the exact
pre-repair specification, so every historical call keeps its meaning.

Derivation of the closed form
-----------------------------
A cross-world potential outcome fixes the direct-path treatment to ``x_prime``
and draws the mediator from its distribution under ``x``:

    M(x) = mu_m + gamma * x + eps_M
    Y(x_prime, M(x)) = 1[ alpha0 + alpha*x_prime + beta*M(x) + eps_Y > 0 ]
                     = 1[ (alpha0 + beta*mu_m) + alpha*x_prime + beta*gamma*x
                          + (beta*eps_M + eps_Y) > 0 ]

The bracket is a fixed offset plus the linear combination ``beta*eps_M + eps_Y``
of a mean-zero bivariate normal, so it is itself univariate normal with

    mean = alpha0 + beta*mu_m + alpha*x_prime + beta*gamma*x
    var  = beta^2 * sigma_m^2 + 2*beta*rho*sigma_m + 1     (= Var(beta*eps_M + eps_Y))

Hence, with ``Phi`` the standard-normal CDF,

    P(Y(x_prime, M(x)) = 1)
        = Phi( (alpha0 + beta*mu_m + alpha*x_prime + beta*gamma*x) / sqrt(var) )

The intercepts enter only through the shared offset ``alpha0 + beta*mu_m``, so
``var`` is unchanged and the NIE still vanishes exactly when ``beta*gamma = 0``.

and the probability-scale natural effects follow directly:

    NDE = P(1, 0) - P(0, 0)
    NIE = P(1, 1) - P(1, 0)
    TE  = NDE + NIE = P(1, 1) - P(0, 0)

This is exact for every ``rho`` in the admissible band; there is no
approximation step. The logistic (sigmoid) DGP in ``synthetic.py`` has no such
elementary closed form. The logit-normal integral is not elementary, and the
usual probit approximation to the logistic carries an error floor around 1e-2
that does not shrink with more Monte Carlo draws, which would make it a weaker
correctness net than the Monte Carlo it is meant to check. So the closed form
here is deliberately scoped to the probit SCM only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Mirror the numerical guard used by the probit sensitivity module so the two
# estimators accept exactly the same admissible range of rho.
_RHO_ABS_MAX = 0.95


def probit_natural_effects_closed_form(
    alpha: float,
    beta: float,
    gamma: float,
    sigma_m: float,
    rho: float,
    mu_m: float = 0.0,
    alpha0: float = 0.0,
) -> tuple[float, float, float]:
    """Exact probability-scale ``(NDE, NIE, TE)`` for the probit SCM.

    Independent, deterministic counterpart to
    ``sensitivity.probit_natural_effects`` (which integrates by Monte Carlo).
    The two agree up to Monte Carlo error, so a test that pins them together
    catches mediator/treatment pairing bugs in either estimator.

    Parameters
    ----------
    alpha, beta, gamma, sigma_m, rho:
        The probit structural parameters. ``rho`` is the residual correlation
        between the mediator and outcome errors (``rho = 0`` is sequential
        ignorability).
    mu_m, alpha0:
        Mediator and outcome baseline intercepts. Both default to 0.0, the
        pre-2026-09-07 specification, so existing five-argument calls are
        unchanged.

    Returns
    -------
    (nde, nie, te) : tuple of floats
    """
    if abs(rho) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")
    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive.")

    var = beta**2 * sigma_m**2 + 2.0 * beta * rho * sigma_m + 1.0
    # Var(beta*eps_M + eps_Y) is positive for any admissible covariance, but
    # guard against a non-positive value from extreme inputs before sqrt.
    if var <= 0.0:  # pragma: no cover - unreachable for |rho| <= 0.95
        raise ValueError("Degenerate latent variance; check (beta, sigma_m, rho).")

    nde, nie, te = _effects(alpha, beta, gamma, sigma_m, rho, mu_m, alpha0)
    return float(nde), float(nie), float(te)


def _effects(alpha, beta, gamma, sigma_m, rho, mu_m, alpha0):
    """The closed form itself, on scalars or on numpy arrays of equal shape.

    Kept as one function so the scalar cross-check and the posterior converter in
    ``effects.posterior_natural_effects`` cannot drift apart: there is exactly one
    place in this repository where the probit natural effects are written down.
    """
    var = beta**2 * sigma_m**2 + 2.0 * beta * rho * sigma_m + 1.0
    sd = np.sqrt(var)
    offset = alpha0 + beta * mu_m

    py_x0_m0 = norm.cdf(offset / sd)
    py_x1_m0 = norm.cdf((offset + alpha) / sd)
    py_x1_m1 = norm.cdf((offset + alpha + beta * gamma) / sd)

    nde = py_x1_m0 - py_x0_m0
    nie = py_x1_m1 - py_x1_m0
    return nde, nie, nde + nie


def probit_natural_effects_closed_form_samples(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma_m: np.ndarray,
    rho: float = 0.0,
    mu_m: np.ndarray | float = 0.0,
    alpha0: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ``(NDE, NIE, TE)``: the same closed form over a whole posterior.

    Draw-by-draw Monte Carlo integration over the mediator is not needed for the
    probit model, because the integral is elementary. Feeding the posterior through
    the same function the MAP path uses is also what keeps the two paths on one
    definition of the estimand.

    Parameters
    ----------
    alpha, beta, gamma, sigma_m:
        Posterior draws, all of the same length.
    rho:
        Assumed residual correlation, a scalar (``0.0`` is sequential ignorability).
    mu_m, alpha0:
        Intercept draws, or ``0.0`` for a fit that forced them to zero.

    Returns
    -------
    (nde, nie, te) : tuple of ndarray
    """
    if abs(float(rho)) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    sigma_m = np.asarray(sigma_m, dtype=float)
    if not (alpha.shape == beta.shape == gamma.shape == sigma_m.shape):
        raise ValueError("alpha, beta, gamma and sigma_m draws must have the same shape.")
    if np.any(sigma_m <= 0.0):
        raise ValueError("sigma_m must be positive.")
    nde, nie, te = _effects(
        alpha, beta, gamma, sigma_m, float(rho), np.asarray(mu_m, dtype=float),
        np.asarray(alpha0, dtype=float),
    )
    return np.asarray(nde), np.asarray(nie), np.asarray(te)
