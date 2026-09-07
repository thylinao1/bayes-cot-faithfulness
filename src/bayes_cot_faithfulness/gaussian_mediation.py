"""Continuous-outcome (Gaussian) mediation, for an outcome that varies in BOTH arms.

Why this module exists
----------------------
The probit path in ``sensitivity`` is the estimator of record for the text-level
outcome ``binary_follow``. On the frozen outcome population that outcome is 0 on
every clean row by construction (element 1's clean-correct restriction plus the
A1 planted-wrong-option taxonomy), so the clean arm carries zero outcome
variance and the NDE/NIE split rests on the probit link extrapolating into a
region the clean arm never visits. Only the TE is checked directly, by the
randomized arm difference. That is Amendment A4.6(b) and ``docs/WAVE1-FITS.md``
sections 2 and 7, measured: clean-arm outcome variance exactly 0.0000 in all
three wave-1 cells.

The logit-level outcome of element 0 and section 9.1 - the log-probability
margin of the planted option on the letter distribution renormalized over the
allowed answer set - is continuous and varies in the clean arm. On that scale
the outcome equation is a linear regression and the two natural effects are
regression coefficients:

    M = mu_m + gamma * X + eps_M
    Y = alpha0 + alpha * X + beta * M + eps_Y
    (eps_M, eps_Y) ~ Normal2(0, [[sigma_m^2,             rho*sigma_m*sigma_y],
                                 [rho*sigma_m*sigma_y,   sigma_y^2          ]])

    NDE = alpha
    NIE = beta * gamma
    TE  = alpha + beta * gamma

There is no link to extrapolate through: ``alpha`` is the within-mediator slope
of Y on X and is estimated from data wherever the two arms overlap in M. This
module is the continuous counterpart of ``sensitivity`` + ``closed_form``, with
the SAME rho parameterisation, the same admissible band, the same intercept
discipline and the same optimizer-failure semantics. It does not touch the
probit path, which stays the estimator of record for ``binary_follow``.

Scale, stated once
------------------
The effects are in the outcome's own units (nats, for a logprob margin). The
pre-registered 0.15 load-bearing threshold of element 1 section 2.5 is on the
PROBABILITY scale and does NOT transfer to nats. Nothing here defines a verdict
threshold; a caller that wants one has to state it, and a published one needs an
amendment. ``standardised_effects`` is provided so a caller can express the
effects in clean-arm outcome standard deviations, which is scale-free, but it is
a reporting convenience and not a pre-registered estimand.

The rho parameterisation
------------------------
Conditioning on the observed M (so ``eps_M = M - mu_m - gamma*X`` is known),

    Y | X, M ~ Normal( alpha0 + alpha*X + beta*M
                       + (rho*sigma_y/sigma_m) * (M - mu_m - gamma*X),
                       sigma_y^2 * (1 - rho^2) )

Write the observed least-squares regression of Y on (1, X, M) as
``A0 + A*X + B*M`` with residual standard deviation ``S``. Matching term by term,
with ``k = rho * sigma_y / sigma_m``:

    B  = beta  + k                 =>  beta(rho)   = B  - k
    A  = alpha - k*gamma           =>  alpha(rho)  = A  + k*gamma
    A0 = alpha0 - k*mu_m           =>  alpha0(rho) = A0 + k*mu_m
    S^2 = sigma_y^2 * (1 - rho^2)  =>  sigma_y(rho) = S / sqrt(1 - rho^2)
    k(rho) = rho * S / (sigma_m * sqrt(1 - rho^2))

which is the probit module's mapping with the outcome error scale freed (the
probit fixes it at 1). Two consequences worth naming, both pinned by tests:

* ``TE(rho) = alpha(rho) + beta(rho)*gamma = A + B*gamma`` for every rho. On this
  scale the total effect is rho-INVARIANT and equals the randomized arm
  difference exactly. rho moves the split and nothing else.
* The NIE crosses zero when ``beta(rho) = 0``, at
  ``rho*_point = |B| * sigma_m / sqrt(S^2 + B^2 * sigma_m^2)``, which is the
  probit formula ``|B| sigma_m / sqrt(1 + B^2 sigma_m^2)`` with ``S`` in place of
  the probit's fixed 1. It contains neither ``alpha`` nor either intercept, so
  the invariance section 8.2 of the pre-registration states carries over
  unchanged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from bayes_cot_faithfulness.sensitivity import OptimizerFailure, OptimizerWarning

# Same admissible band as the probit path, for the same reason: outside it the
# 1/sqrt(1-rho^2) reparameterisation is numerically unstable.
_RHO_ABS_MAX = 0.95
_MIN_SIGMA = 1e-8


class GaussianMediationFit(NamedTuple):
    """Fitted linear-linear mediation coefficients at one assumed ``rho``.

    ``sigma_y`` is the extra parameter the continuous outcome has and the probit
    does not: the probit identifies the outcome error scale only up to a
    normalisation and fixes it at 1.
    """

    alpha: float
    beta: float
    gamma: float
    sigma_m: float
    mu_m: float = 0.0
    alpha0: float = 0.0
    sigma_y: float = 1.0
    converged: bool = True


@dataclass(frozen=True)
class GaussianSensitivityPoint:
    """Natural effects on the outcome's own scale under one assumed ``rho``."""

    rho: float
    alpha: float
    beta: float
    gamma: float
    sigma_m: float
    sigma_y: float
    mu_m: float
    alpha0: float
    nde: float
    nie: float
    te: float


def gaussian_natural_effects(
    alpha: float,
    beta: float,
    gamma: float,
    sigma_m: float = 1.0,
    rho: float = 0.0,
    mu_m: float = 0.0,
    alpha0: float = 0.0,
) -> tuple[float, float, float]:
    """Exact ``(NDE, NIE, TE)`` on the outcome's own scale for the linear-linear SCM.

    The cross-world potential outcome is
    ``Y(x', M(x)) = alpha0 + alpha*x' + beta*(mu_m + gamma*x + eps_M) + eps_Y``,
    whose expectation is linear, so

        NDE = E[Y(1, M(0))] - E[Y(0, M(0))] = alpha
        NIE = E[Y(1, M(1))] - E[Y(1, M(0))] = beta * gamma
        TE  = alpha + beta * gamma

    ``sigma_m``, ``rho``, ``mu_m`` and ``alpha0`` do not enter. They are accepted
    so this is a drop-in for ``closed_form.probit_natural_effects_closed_form``
    and so the rho band is validated on the same argument, and their absence from
    the answer is the substantive difference between the two scales: on the
    probability scale the effects depend on where the link is evaluated, and here
    they do not.
    """
    if abs(float(rho)) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")
    if float(sigma_m) <= 0.0:
        raise ValueError("sigma_m must be positive.")
    nde = float(alpha)
    nie = float(beta) * float(gamma)
    return nde, nie, nde + nie


def gaussian_natural_effects_samples(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ``(NDE, NIE, TE)`` over a whole posterior or bootstrap."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if not (alpha.shape == beta.shape == gamma.shape):
        raise ValueError("alpha, beta and gamma draws must have the same shape.")
    nie = beta * gamma
    return alpha, nie, alpha + nie


def standardised_effects(
    nde: float, nie: float, te: float, clean_arm_sd: float
) -> tuple[float, float, float]:
    """The three effects in clean-arm outcome standard deviations.

    A reporting convenience for a scale whose natural unit is nats, so that a
    magnitude can be compared across models. It is NOT a pre-registered estimand
    and it is not a threshold: the 0.15 of element 1 section 2.5 is on the
    probability scale and does not transfer here.
    """
    if not np.isfinite(clean_arm_sd) or clean_arm_sd <= 0.0:
        raise ValueError("clean_arm_sd must be finite and positive.")
    return float(nde / clean_arm_sd), float(nie / clean_arm_sd), float(te / clean_arm_sd)


def _validate_xmy(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> None:
    """Same shape and support checks the probit path makes, minus the binary-Y one."""
    if len(X) != len(M) or len(M) != len(Y):
        raise ValueError("X, M, Y must all have the same length.")
    if np.ndim(X) != 1:
        raise ValueError("X must be 1-D.")
    if not set(np.unique(np.asarray(X))).issubset({0, 1}):
        raise ValueError("X must be binary {0, 1}.")
    if not np.all(np.isfinite(np.asarray(M, dtype=float))):
        raise ValueError("M must be finite.")
    if not np.all(np.isfinite(np.asarray(Y, dtype=float))):
        raise ValueError("Y must be finite.")


def _ols_fit(X: np.ndarray, M: np.ndarray, Y: np.ndarray, intercepts: bool):
    """The observed-data regressions: M on X, then Y on (1, X, M).

    Both are ordinary least squares. That is not an approximation: the joint
    log-likelihood factorises as ``f(M | X) f(Y | M, X)``, and for free
    ``(alpha0, alpha, beta)`` the conditional mean
    ``alpha0 + alpha*X + beta*M + k*(M - mu_m - gamma*X)`` reparameterises to a
    free linear function of ``(1, X, M)``. So the mediator equation's MLE does
    not move with the assumed rho, exactly as in the probit case, and the Y
    equation's profile MLE is the plain regression of Y on (1, X, M).
    """
    x = np.asarray(X, dtype=float)
    m = np.asarray(M, dtype=float)
    y = np.asarray(Y, dtype=float)
    n = len(x)

    if intercepts:
        design_m = np.column_stack([np.ones(n), x])
        design_y = np.column_stack([np.ones(n), x, m])
    else:
        design_m = x.reshape(-1, 1)
        design_y = np.column_stack([x, m])

    coef_m, *_ = np.linalg.lstsq(design_m, m, rcond=None)
    resid_m = m - design_m @ coef_m
    sigma_m = float(max(np.sqrt(np.mean(resid_m**2)), _MIN_SIGMA))

    coef_y, *_ = np.linalg.lstsq(design_y, y, rcond=None)
    resid_y = y - design_y @ coef_y
    s_y = float(max(np.sqrt(np.mean(resid_y**2)), _MIN_SIGMA))

    if intercepts:
        mu_m, gamma = float(coef_m[0]), float(coef_m[1])
        a0, a, b = float(coef_y[0]), float(coef_y[1]), float(coef_y[2])
    else:
        mu_m, gamma = 0.0, float(coef_m[0])
        a0, a, b = 0.0, float(coef_y[0]), float(coef_y[1])
    return mu_m, gamma, sigma_m, a0, a, b, s_y


def fit_gaussian_mediation_closed_form(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho: float = 0.0,
    intercepts: bool = True,
) -> GaussianMediationFit:
    """Exact maximum-likelihood fit at fixed ``rho``, in closed form.

    Two least-squares solves plus the rho mapping in the module docstring. It is
    the independent cross-check on ``fit_gaussian_mediation_map`` for the same
    reason ``closed_form`` is the cross-check on the probit Monte Carlo: two
    estimators that share no code path, pinned together by a test.
    """
    if abs(float(rho)) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")
    _validate_xmy(X, M, Y)
    mu_m, gamma, sigma_m, a0, a, b, s_y = _ols_fit(X, M, Y, intercepts)

    rho = float(rho)
    root = np.sqrt(1.0 - rho**2)
    sigma_y = s_y / root
    k = rho * sigma_y / sigma_m
    return GaussianMediationFit(
        alpha=float(a + k * gamma),
        beta=float(b - k),
        gamma=float(gamma),
        sigma_m=float(sigma_m),
        mu_m=float(mu_m) if intercepts else 0.0,
        alpha0=float(a0 + k * mu_m) if intercepts else 0.0,
        sigma_y=float(sigma_y),
        converged=True,
    )


def _joint_negloglik(
    params: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho: float,
) -> float:
    """Negative log-likelihood of the linear-linear mediation model at fixed ``rho``.

    Dispatches on the length of ``params`` the way the probit module does:

    * 5 entries ``(alpha, beta, gamma, log_sigma_m, log_sigma_y)`` - the
      intercept-free specification, kept only so the same offset-null contrast
      the probit path documents can be run on this scale.
    * 7 entries ``(alpha, beta, gamma, log_sigma_m, log_sigma_y, mu_m, alpha0)``
      - the specification with both intercepts, which is the default.
    """
    if len(params) == 5:
        alpha, beta, gamma, log_sigma_m, log_sigma_y = params
        mu_m, alpha0 = 0.0, 0.0
    else:
        alpha, beta, gamma, log_sigma_m, log_sigma_y, mu_m, alpha0 = params
    sigma_m = np.exp(log_sigma_m)
    sigma_y = np.exp(log_sigma_y)

    resid_m = M - mu_m - gamma * X
    ll_m = -0.5 * np.sum((resid_m / sigma_m) ** 2) - len(M) * (
        np.log(sigma_m) + 0.5 * np.log(2.0 * np.pi)
    )

    cond_sd = sigma_y * np.sqrt(1.0 - rho**2)
    mean_y = alpha0 + alpha * X + beta * M + (rho * sigma_y / sigma_m) * resid_m
    resid_y = Y - mean_y
    ll_y = -0.5 * np.sum((resid_y / cond_sd) ** 2) - len(Y) * (
        np.log(cond_sd) + 0.5 * np.log(2.0 * np.pi)
    )
    return float(-(ll_m + ll_y))


def fit_gaussian_mediation_map(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho: float = 0.0,
    intercepts: bool = True,
    strict: bool = False,
) -> GaussianMediationFit:
    """Maximum-a-posteriori (here, MLE) fit of the linear-linear model at fixed ``rho``.

    The continuous-outcome counterpart of ``sensitivity.fit_probit_mediation_map``,
    with the same arguments and the same failure semantics: ``strict=True`` raises
    ``OptimizerFailure``, the default warns with ``OptimizerWarning`` and returns
    the unconverged parameters with ``converged=False``.

    ``intercepts=True`` (the default) fits the mediator baseline ``mu_m`` and the
    outcome baseline ``alpha0``. Dropping them is the same defect the 2026-09-07
    repair fixed on the probit path: a mediator with a control-arm level and an
    outcome with a control-arm level are both read as treatment shifts.
    ``tests/test_gaussian_mediation.py`` runs the reviewer's count-offset and
    Gaussian-offset nulls on this fit.

    The optimiser is started at the closed-form solution, so it is a check on the
    closed form rather than a race against it; the two agree to optimiser
    tolerance and a test pins that.
    """
    if abs(float(rho)) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")
    _validate_xmy(X, M, Y)

    x = np.asarray(X, dtype=float)
    m = np.asarray(M, dtype=float)
    y = np.asarray(Y, dtype=float)

    start = fit_gaussian_mediation_closed_form(x, m, y, float(rho), intercepts=intercepts)
    if intercepts:
        init = np.array(
            [
                start.alpha,
                start.beta,
                start.gamma,
                np.log(start.sigma_m),
                np.log(start.sigma_y),
                start.mu_m,
                start.alpha0,
            ]
        )
    else:
        init = np.array(
            [start.alpha, start.beta, start.gamma, np.log(start.sigma_m), np.log(start.sigma_y)]
        )

    result = minimize(_joint_negloglik, init, args=(x, m, y, float(rho)), method="L-BFGS-B")
    if not result.success:
        message = (
            f"gaussian mediation fit did not converge at rho={float(rho):+.4f} "
            f"(intercepts={intercepts}): {result.message}"
        )
        if strict:
            raise OptimizerFailure(message)
        warnings.warn(message, OptimizerWarning, stacklevel=2)

    if intercepts:
        alpha, beta, gamma, log_sigma_m, log_sigma_y, mu_m, alpha0 = result.x
    else:
        alpha, beta, gamma, log_sigma_m, log_sigma_y = result.x
        mu_m, alpha0 = 0.0, 0.0
    return GaussianMediationFit(
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        sigma_m=float(np.exp(log_sigma_m)),
        mu_m=float(mu_m),
        alpha0=float(alpha0),
        sigma_y=float(np.exp(log_sigma_y)),
        converged=bool(result.success),
    )


def gaussian_effects_curve(
    fit: GaussianMediationFit,
    rho_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(NDE, NIE, TE)`` across ``rho_grid`` from ONE ``rho = 0`` fit.

    The vectorised analogue of ``experiments/mechanism_battery.effects_curve``,
    using the mapping in the module docstring. ``fit`` must be the rho = 0 fit, so
    that ``(A0, A, B, S)`` are read straight off it.
    """
    rho = np.asarray(rho_grid, dtype=float)
    if np.any(np.abs(rho) > _RHO_ABS_MAX):
        raise ValueError(f"every |rho| must be <= {_RHO_ABS_MAX}.")
    root = np.sqrt(1.0 - rho**2)
    k = rho * (fit.sigma_y / root) / fit.sigma_m
    alpha_r = fit.alpha + k * fit.gamma
    beta_r = fit.beta - k
    nie = beta_r * fit.gamma
    return alpha_r, nie, alpha_r + nie


def gaussian_sensitivity_sweep(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho_grid: np.ndarray | None = None,
    intercepts: bool = True,
) -> list[GaussianSensitivityPoint]:
    """Refit at every grid point and convert to effects: the slow, honest sweep.

    Kept as the cross-check on ``gaussian_effects_curve`` exactly as
    ``sensitivity.sensitivity_sweep`` is the cross-check on the battery's
    vectorised curve.
    """
    if rho_grid is None:
        rho_grid = np.round(np.arange(-0.6, 0.81, 0.1), 4)
    points: list[GaussianSensitivityPoint] = []
    for rho in np.asarray(rho_grid, dtype=float):
        fit = fit_gaussian_mediation_closed_form(X, M, Y, float(rho), intercepts=intercepts)
        nde, nie, te = gaussian_natural_effects(fit.alpha, fit.beta, fit.gamma)
        points.append(
            GaussianSensitivityPoint(
                rho=float(rho),
                alpha=fit.alpha,
                beta=fit.beta,
                gamma=fit.gamma,
                sigma_m=fit.sigma_m,
                sigma_y=fit.sigma_y,
                mu_m=fit.mu_m,
                alpha0=fit.alpha0,
                nde=nde,
                nie=nie,
                te=te,
            )
        )
    return points


def gaussian_rho_star_point(beta: float, sigma_m: float, sigma_y: float) -> float:
    """The point-estimate zero crossing of the NIE on the continuous scale.

    ``|B| * sigma_m / sqrt(sigma_y^2 + B^2 * sigma_m^2)``, unsigned; the crossing
    sits on the side of ``sign(B)``, since ``beta(rho) = B - k`` vanishes where
    ``rho/sqrt(1-rho^2) = B*sigma_m/S``. It is the probit module's
    ``|B| sigma_m / sqrt(1 + B^2
    sigma_m^2)`` with the outcome error scale freed, and like it, it contains
    neither the direct coefficient nor either intercept: section 8.2's invariance
    holds on this scale too.

    ``beta`` and ``sigma_y`` here are the rho = 0 fit's values, that is, the
    observed regression slope ``B`` and residual sd ``S``.
    """
    if sigma_m <= 0.0 or sigma_y <= 0.0:
        raise ValueError("sigma_m and sigma_y must be positive.")
    c = abs(float(beta)) * float(sigma_m) / float(sigma_y)
    return float(c / np.sqrt(1.0 + c * c))
