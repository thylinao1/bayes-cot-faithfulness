"""Sensitivity analysis for the sequential-ignorability assumption.

Every causal-mediation estimate of chain-of-thought faithfulness rests on an
assumption that nobody can verify from data: that there is no unmeasured
confounder sitting between the CoT (mediator ``M``) and the answer (outcome
``Y``) once the prompt ``X`` is fixed. This is assumption (A3), sequential
ignorability. In an LLM the natural CoT and the answer both depend on the same
hidden activations, so (A3) is almost certainly violated to some degree.

This module quantifies how badly that matters. Following Imai, Keele and
Tingley (2010), we introduce a single sensitivity parameter

    rho = Corr(eps_M, eps_Y)

the residual correlation between the mediator error and the outcome error.
``rho = 0`` is exactly sequential ignorability. We then re-estimate the natural
effects across a grid of assumed ``rho`` and report the range over which the
faithfulness conclusion survives.

To make ``rho`` enter cleanly we use a probit outcome (a latent-Gaussian
specification), which is the standard move in the Imai-Keele-Tingley framework.
The data-generating process is::

    X ~ Bernoulli(0.5)
    (eps_M, eps_Y) ~ Normal2(0, [[sigma_m^2, rho*sigma_m], [rho*sigma_m, 1]])
    M = gamma * X + eps_M
    Y = 1[alpha * X + beta * M + eps_Y > 0]

Conditioning on the observed ``M`` (so ``eps_M = M - gamma*X`` is known) gives an
exact observed-data likelihood for any fixed ``rho``::

    P(Y=1 | X, M) = Phi( (alpha*X + beta*M + (rho/sigma_m)*(M - gamma*X))
                          / sqrt(1 - rho^2) )

At ``rho = 0`` this is ordinary probit mediation. At the true ``rho`` it
recovers the de-confounded structural coefficients, hence the true effects.
That recovery is what ``notebooks/03_sensitivity_analysis.py`` validates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm

# Outside this band the latent-Gaussian reparameterisation 1/sqrt(1-rho^2)
# becomes numerically unstable. The honest scientific range is well inside it.
_RHO_ABS_MAX = 0.95


@dataclass(frozen=True)
class ConfoundedCoTConfig:
    """Synthetic CoT process with a controllable M-Y confounder.

    Identical to ``SyntheticCoTConfig`` but with a probit outcome and a true
    residual correlation ``rho_confound`` between the mediator and outcome
    errors. ``rho_confound = 0`` restores sequential ignorability.
    """

    n_prompts: int = 600
    alpha_direct: float = 0.3
    beta_mediated: float = 1.0
    gamma_xm: float = 0.8
    sigma_m: float = 0.5
    rho_confound: float = 0.5
    rng_seed: int = 42


@dataclass(frozen=True)
class SensitivityPoint:
    """Natural effects estimated under one assumed value of ``rho``."""

    rho: float
    alpha: float
    beta: float
    gamma: float
    sigma_m: float
    nde: float
    nie: float
    te: float


def simulate_confounded_cot(
    config: ConfoundedCoTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw probit CoT traces with a known M-Y confounder.

    Returns
    -------
    X, M, Y : ndarrays of shape (n_prompts,)
    """
    if abs(config.rho_confound) > _RHO_ABS_MAX:
        raise ValueError(f"|rho_confound| must be <= {_RHO_ABS_MAX}.")

    rng = np.random.default_rng(config.rng_seed)
    n = config.n_prompts
    sm = config.sigma_m

    X = rng.binomial(1, 0.5, size=n)

    cov = np.array([[sm**2, config.rho_confound * sm], [config.rho_confound * sm, 1.0]])
    eps = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
    eps_m, eps_y = eps[:, 0], eps[:, 1]

    M = config.gamma_xm * X + eps_m
    latent_y = config.alpha_direct * X + config.beta_mediated * M + eps_y
    Y = (latent_y > 0.0).astype(int)

    return X.astype(int), M.astype(float), Y


def probit_natural_effects(
    alpha: float,
    beta: float,
    gamma: float,
    sigma_m: float,
    rho: float,
    n_mc: int = 200_000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Natural effects on the probability scale for the probit SCM.

    The cross-world potential outcomes are evaluated by Monte Carlo over the
    joint error draw ``(eps_M, eps_Y) ~ Normal2(0, Sigma(sigma_m, rho))``. This
    same function computes ground truth (true params, true rho) and any
    estimate (fitted params, assumed rho), which is what makes the recovery
    check in the sensitivity sweep meaningful.
    """
    if abs(rho) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")

    rng = np.random.default_rng(rng_seed)
    cov = np.array([[sigma_m**2, rho * sigma_m], [rho * sigma_m, 1.0]])
    eps = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_mc)
    eps_m, eps_y = eps[:, 0], eps[:, 1]

    m_under_x0 = gamma * 0 + eps_m
    m_under_x1 = gamma * 1 + eps_m

    y_x0_m0 = (alpha * 0 + beta * m_under_x0 + eps_y > 0.0).mean()
    y_x1_m0 = (alpha * 1 + beta * m_under_x0 + eps_y > 0.0).mean()
    y_x1_m1 = (alpha * 1 + beta * m_under_x1 + eps_y > 0.0).mean()

    nde = y_x1_m0 - y_x0_m0
    nie = y_x1_m1 - y_x1_m0
    te = nde + nie
    return float(nde), float(nie), float(te)


def _joint_negloglik(
    params: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho: float,
) -> float:
    """Negative log-likelihood of the probit mediation model at fixed ``rho``."""
    alpha, beta, gamma, log_sigma_m = params
    sigma_m = np.exp(log_sigma_m)

    # Mediator equation: M | X ~ Normal(gamma*X, sigma_m).
    resid_m = M - gamma * X
    ll_m = norm.logpdf(resid_m, loc=0.0, scale=sigma_m).sum()

    # Outcome equation: probit with the confounding offset folded in.
    scale = np.sqrt(1.0 - rho**2)
    eta = (alpha * X + beta * M + (rho / sigma_m) * resid_m) / scale
    # Stable log-likelihood via logcdf / logsf.
    ll_y = np.where(Y == 1, norm.logcdf(eta), norm.logsf(eta)).sum()

    return -(ll_m + ll_y)


def fit_probit_mediation_map(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho: float,
) -> tuple[float, float, float, float]:
    """Maximum-a-posteriori (here, MLE) fit of the probit model at fixed ``rho``.

    Fast deterministic optimisation, suitable for a dense ``rho`` grid and for
    tests. Returns ``(alpha, beta, gamma, sigma_m)``.
    """
    if abs(rho) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")
    _validate_xmy(X, M, Y)

    # Initialise from naive least squares on the mediator and zeros elsewhere.
    gamma0 = float(np.cov(X, M, bias=True)[0, 1] / max(np.var(X), 1e-6))
    resid0 = M - gamma0 * X
    sigma0 = float(max(resid0.std(), 1e-2))
    init = np.array([0.0, 0.0, gamma0, np.log(sigma0)])

    result = minimize(
        _joint_negloglik,
        init,
        args=(X, M, Y, rho),
        method="L-BFGS-B",
    )
    alpha, beta, gamma, log_sigma_m = result.x
    return float(alpha), float(beta), float(gamma), float(np.exp(log_sigma_m))


def sensitivity_sweep(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho_grid: np.ndarray | None = None,
    n_mc: int = 100_000,
    rng_seed: int = 0,
) -> list[SensitivityPoint]:
    """Estimate the natural effects across a grid of assumed ``rho``.

    For each ``rho`` we fit the de-confounded coefficients and convert them to
    probability-scale natural effects. The resulting curve is the sensitivity
    analysis: how the faithfulness verdict (sign and size of the NIE) moves as
    the unmeasured-confounding assumption is relaxed.
    """
    if rho_grid is None:
        rho_grid = np.round(np.arange(-0.6, 0.81, 0.1), 4)

    points: list[SensitivityPoint] = []
    for rho in rho_grid:
        alpha, beta, gamma, sigma_m = fit_probit_mediation_map(X, M, Y, float(rho))
        nde, nie, te = probit_natural_effects(
            alpha, beta, gamma, sigma_m, float(rho), n_mc=n_mc, rng_seed=rng_seed
        )
        points.append(
            SensitivityPoint(
                rho=float(rho),
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                sigma_m=sigma_m,
                nde=nde,
                nie=nie,
                te=te,
            )
        )
    return points


def robustness_interval(
    sweep: list[SensitivityPoint],
    key: str = "nie",
) -> tuple[float, float] | None:
    """Largest contiguous ``rho`` range over which ``key`` keeps its sign at rho=0.

    Returns ``(rho_lo, rho_hi)`` of the sign-preserving band that contains the
    ``rho = 0`` estimate, or ``None`` if the effect is not sign-stable there.
    This is the headline robustness statement: "the indirect (faithful) path
    stays positive for all rho in [rho_lo, rho_hi]."
    """
    if not sweep:
        return None
    ordered = sorted(sweep, key=lambda p: p.rho)
    rhos = [p.rho for p in ordered]
    vals = [getattr(p, key) for p in ordered]

    # Locate the grid point nearest rho = 0 as the anchor.
    anchor = min(range(len(rhos)), key=lambda i: abs(rhos[i]))
    sign = np.sign(vals[anchor])
    if sign == 0:
        return None

    lo = hi = anchor
    while lo - 1 >= 0 and np.sign(vals[lo - 1]) == sign:
        lo -= 1
    while hi + 1 < len(vals) and np.sign(vals[hi + 1]) == sign:
        hi += 1
    return float(rhos[lo]), float(rhos[hi])


@dataclass(frozen=True)
class BreakdownFrontier:
    """How much unmeasured M-Y confounding it takes to overturn an effect's sign.

    This is the sensitivity-analysis analogue of VanderWeele's E-value for a
    mediation effect: starting from the sequential-ignorability estimate at
    ``rho = 0``, it reports the residual correlation ``rho`` at which the chosen
    natural effect (by default the NIE, the faithful path) crosses zero. The
    headline number is ``robustness``: the smallest ``|rho|`` that flips the
    verdict. Larger means the conclusion survives more confounding.
    """

    key: str
    effect_at_zero: float
    rho_star_pos: float | None
    rho_star_neg: float | None
    robustness: float
    rho_max: float
    survives_full_range: bool


def breakdown_frontier(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    key: str = "nie",
    n_mc: int = 200_000,
    rng_seed: int = 0,
    rho_max: float = 0.95,
) -> BreakdownFrontier:
    """Smallest residual confounding ``|rho|`` that overturns the sign of an effect.

    Starting from the sequential-ignorability estimate (``rho = 0``), find the
    nearest ``rho`` on each side at which the chosen natural effect (``"nde"``,
    ``"nie"`` or ``"te"``) crosses zero, by refitting the probit model at each
    candidate ``rho`` and converting to natural effects. Monte Carlo uses a fixed
    seed so the effect is a deterministic, smooth function of ``rho`` (common random
    numbers), which keeps the root-find stable.

    The headline output is ``robustness``: the minimum ``|rho|`` that flips the
    verdict, an E-value-style sensitivity scalar a faithfulness claim can be
    reported with. ``rho_star_pos`` / ``rho_star_neg`` are the signed crossings (or
    ``None`` when the sign never flips on that side within the searched range).
    """
    if key not in {"nde", "nie", "te"}:
        raise ValueError("key must be one of 'nde', 'nie', 'te'.")
    _validate_xmy(X, M, Y)
    rho_max = min(float(rho_max), _RHO_ABS_MAX - 1e-3)

    def effect_at(rho: float) -> float:
        alpha, beta, gamma, sigma_m = fit_probit_mediation_map(X, M, Y, float(rho))
        nde, nie, te = probit_natural_effects(
            alpha, beta, gamma, sigma_m, float(rho), n_mc=n_mc, rng_seed=rng_seed
        )
        return {"nde": nde, "nie": nie, "te": te}[key]

    e0 = effect_at(0.0)
    sign0 = np.sign(e0)

    def crossing(bound: float) -> float | None:
        if sign0 == 0:
            return None
        if np.sign(effect_at(bound)) == sign0:
            return None  # sign holds across this side; no breakdown in range
        lo, hi = (0.0, bound) if bound > 0 else (bound, 0.0)
        return float(brentq(effect_at, lo, hi, xtol=1e-4, rtol=1e-6, maxiter=100))

    rho_star_pos = crossing(rho_max)
    rho_star_neg = crossing(-rho_max)

    magnitudes = [abs(r) for r in (rho_star_pos, rho_star_neg) if r is not None]
    survives_full_range = not magnitudes
    robustness = min(magnitudes) if magnitudes else rho_max

    return BreakdownFrontier(
        key=key,
        effect_at_zero=float(e0),
        rho_star_pos=rho_star_pos,
        rho_star_neg=rho_star_neg,
        robustness=float(robustness),
        rho_max=float(rho_max),
        survives_full_range=survives_full_range,
    )


def _validate_xmy(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> None:
    if len(X) != len(M) or len(M) != len(Y):
        raise ValueError("X, M, Y must all have the same length.")
    if X.ndim != 1:
        raise ValueError("X must be 1-D.")
    if not set(np.unique(X)).issubset({0, 1}):
        raise ValueError("X must be binary {0, 1}.")
    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be binary {0, 1}.")
