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
    M = gamma * X + eps_M
    Y = 1[alpha * X + beta * M + eps_Y > 0]

Derivation of the closed form
-----------------------------
A cross-world potential outcome fixes the direct-path treatment to ``x_prime``
and draws the mediator from its distribution under ``x``:

    M(x) = gamma * x + eps_M
    Y(x_prime, M(x)) = 1[ alpha*x_prime + beta*M(x) + eps_Y > 0 ]
                     = 1[ alpha*x_prime + beta*gamma*x + (beta*eps_M + eps_Y) > 0 ]

The bracket is a fixed offset plus the linear combination ``beta*eps_M + eps_Y``
of a mean-zero bivariate normal, so it is itself univariate normal with

    mean = alpha*x_prime + beta*gamma*x
    var  = beta^2 * sigma_m^2 + 2*beta*rho*sigma_m + 1     (= Var(beta*eps_M + eps_Y))

Hence, with ``Phi`` the standard-normal CDF,

    P(Y(x_prime, M(x)) = 1) = Phi( (alpha*x_prime + beta*gamma*x) / sqrt(var) )

and the probability-scale natural effects follow directly:

    NDE = P(1, 0) - P(0, 0)
    NIE = P(1, 1) - P(1, 0)
    TE  = NDE + NIE = P(1, 1) - P(0, 0)

This is exact for every ``rho`` in the admissible band; there is no
approximation step. The logistic (sigmoid) DGP in ``synthetic.py`` has no such
elementary closed form -- the logit-normal integral is not elementary, and the
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
    sd = np.sqrt(var)

    def p(x_prime: int, x: int) -> float:
        return float(norm.cdf((alpha * x_prime + beta * gamma * x) / sd))

    py_x0_m0 = p(0, 0)
    py_x1_m0 = p(1, 0)
    py_x1_m1 = p(1, 1)

    nde = py_x1_m0 - py_x0_m0
    nie = py_x1_m1 - py_x1_m0
    te = nde + nie
    return float(nde), float(nie), float(te)
