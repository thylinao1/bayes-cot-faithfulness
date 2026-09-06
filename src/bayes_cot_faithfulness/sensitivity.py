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
    M = mu_m + gamma * X + eps_M
    Y = 1[alpha0 + alpha * X + beta * M + eps_Y > 0]

Baseline intercepts (2026-09-07 repair)
---------------------------------------
``mu_m`` and ``alpha0`` are the mediator and outcome **baseline intercepts**.
Until 2026-09-07 both were forced to zero, which is a substantive and usually
false modelling claim: it asserts ``E[M | X=0] = 0`` and a clean-arm outcome
baseline of ``Phi(0) = 0.5``. On any mediator with a natural baseline level -
a CoT step count, a truncation depth, a token budget - forcing ``E[M|X=0]=0``
makes the estimator read the *level* of the control arm as a treatment shift,
and it reports a large mediated effect on data with no causal effect at all.
The independent review of 6 September 2026 (section 3.3) demonstrated exactly
this: ``M`` an X-independent positive count, ``Y`` an independent coin, true
NDE = NIE = TE = 0, and the intercept-free fit reported NIE 0.2131, TE 0.2697.
``tests/test_offset_null.py`` runs that null on every commit.

``intercepts=True`` (the default everywhere in this module) estimates ``mu_m``
and ``alpha0`` from the data. ``intercepts=False`` restores the old
intercept-free specification bit for bit, so historical numbers computed before
the repair stay reproducible; it is pinned by
``tests/test_offset_null.py::test_legacy_fit_is_bit_for_bit_reproducible``.

Observed-data likelihood
------------------------
Conditioning on the observed ``M`` (so ``eps_M = M - mu_m - gamma*X`` is known)
gives an exact observed-data likelihood for any fixed ``rho``::

    P(Y=1 | X, M) = Phi( (alpha0 + alpha*X + beta*M
                          + (rho/sigma_m)*(M - mu_m - gamma*X))
                          / sqrt(1 - rho^2) )

At ``rho = 0`` this is ordinary probit mediation. At the true ``rho`` it
recovers the de-confounded structural coefficients, hence the true effects.
That recovery is what ``notebooks/03_sensitivity_analysis.py`` validates.

The rho reparameterisation, with intercepts
-------------------------------------------
The mediator equation is a plain linear regression of ``M`` on ``X``, so
``(mu_m, gamma, sigma_m)`` are identified by the ``M`` marginal alone and do
**not** move with the assumed ``rho``. All of the ``rho`` dependence lives in
the outcome equation. Write the observed probit regression of ``Y`` on
``(1, X, M)`` as ``eta = A0 + A*X + B*M``. Matching it term by term against the
displayed likelihood gives::

    B  = (beta + rho/sigma_m) / sqrt(1 - rho^2)
    A  = (alpha - (rho/sigma_m)*gamma) / sqrt(1 - rho^2)
    A0 = (alpha0 - (rho/sigma_m)*mu_m) / sqrt(1 - rho^2)

and therefore, holding the observed fit ``(A0, A, B)`` fixed, the structural
coefficients implied by an assumed ``rho`` are::

    beta(rho)   = B * sqrt(1 - rho^2) - rho/sigma_m
    alpha(rho)  = A * sqrt(1 - rho^2) + (rho/sigma_m) * gamma
    alpha0(rho) = A0 * sqrt(1 - rho^2) + (rho/sigma_m) * mu_m

The intercepts change ``alpha0(rho)`` and nothing else in the mapping: the
``beta(rho)`` line, and hence the breakdown frontier below, has exactly the
form it had before the repair. ``(A0, A, B)`` are the ``rho = 0`` fit, so
``beta(0) = B``. ``tests/test_rho_star_semantics.py`` checks this
reparameterisation against a direct refit and against the closed form.

Identification note (see docs/phase2_design_notes.md section 1): in this
project's experiments the cue X is randomized by protocol, so the treatment
half of sequential ignorability holds by construction and ``rho`` prices only
the mediator-outcome half. The reported ``rho*`` is therefore interpretable
purely as mediator-outcome confounding tolerance, with no X-side residue.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm

# Outside this band the latent-Gaussian reparameterisation 1/sqrt(1-rho^2)
# becomes numerically unstable. The honest scientific range is well inside it.
_RHO_ABS_MAX = 0.95


class OptimizerFailure(RuntimeError):
    """The probit fit's optimizer reported failure.

    Raised by ``fit_probit_mediation_map(..., strict=True)``. With the default
    ``strict=False`` the same condition is reported as an ``OptimizerWarning``
    and the (unconverged) parameters are returned, because a sensitivity sweep
    that dies on one grid point is less useful than one that flags it.
    """


class OptimizerWarning(RuntimeWarning):
    """The probit fit's optimizer reported failure; results may be unconverged."""


@dataclass(frozen=True)
class ConfoundedCoTConfig:
    """Synthetic CoT process with a controllable M-Y confounder.

    Identical to ``SyntheticCoTConfig`` but with a probit outcome and a true
    residual correlation ``rho_confound`` between the mediator and outcome
    errors. ``rho_confound = 0`` restores sequential ignorability.

    ``mu_m`` and ``alpha0`` are the mediator and outcome baseline intercepts.
    Both default to 0.0, which reproduces the pre-2026-09-07 generator exactly;
    set them to simulate the offset world that broke the intercept-free
    estimator (see the module docstring and ``tests/test_offset_null.py``).
    """

    n_prompts: int = 600
    alpha_direct: float = 0.3
    beta_mediated: float = 1.0
    gamma_xm: float = 0.8
    sigma_m: float = 0.5
    rho_confound: float = 0.5
    rng_seed: int = 42
    mu_m: float = 0.0
    alpha0: float = 0.0


class ProbitMediationFit(NamedTuple):
    """Fitted probit mediation coefficients at one assumed ``rho``.

    The first four fields are the pre-repair return value in its original
    order, so ``fit.alpha, fit.beta, fit.gamma, fit.sigma_m`` reads exactly as
    the old 4-tuple did. ``mu_m`` and ``alpha0`` are the baseline intercepts,
    identically 0.0 when the fit was run with ``intercepts=False``.
    ``converged`` mirrors the optimizer's ``result.success``.
    """

    alpha: float
    beta: float
    gamma: float
    sigma_m: float
    mu_m: float = 0.0
    alpha0: float = 0.0
    converged: bool = True


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
    mu_m: float = 0.0
    alpha0: float = 0.0


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

    # Adding the default 0.0 intercepts is exact in IEEE 754, so a config that
    # leaves them at 0.0 reproduces the pre-repair draws bit for bit.
    M = config.mu_m + config.gamma_xm * X + eps_m
    latent_y = config.alpha0 + config.alpha_direct * X + config.beta_mediated * M + eps_y
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
    *,
    mu_m: float = 0.0,
    alpha0: float = 0.0,
) -> tuple[float, float, float]:
    """Natural effects on the probability scale for the probit SCM.

    The cross-world potential outcomes are evaluated by Monte Carlo over the
    joint error draw ``(eps_M, eps_Y) ~ Normal2(0, Sigma(sigma_m, rho))``. This
    same function computes ground truth (true params, true rho) and any
    estimate (fitted params, assumed rho), which is what makes the recovery
    check in the sensitivity sweep meaningful.

    ``mu_m`` and ``alpha0`` are the mediator and outcome baseline intercepts;
    both default to 0.0, the pre-2026-09-07 specification. They are keyword-only
    so that every existing positional call site keeps its meaning.
    """
    if abs(rho) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")

    rng = np.random.default_rng(rng_seed)
    cov = np.array([[sigma_m**2, rho * sigma_m], [rho * sigma_m, 1.0]])
    eps = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_mc)
    eps_m, eps_y = eps[:, 0], eps[:, 1]

    m_under_x0 = mu_m + gamma * 0 + eps_m
    m_under_x1 = mu_m + gamma * 1 + eps_m

    y_x0_m0 = (alpha0 + alpha * 0 + beta * m_under_x0 + eps_y > 0.0).mean()
    y_x1_m0 = (alpha0 + alpha * 1 + beta * m_under_x0 + eps_y > 0.0).mean()
    y_x1_m1 = (alpha0 + alpha * 1 + beta * m_under_x1 + eps_y > 0.0).mean()

    nde = y_x1_m0 - y_x0_m0
    nie = y_x1_m1 - y_x1_m0
    te = nde + nie
    return float(nde), float(nie), float(te)


def natural_effects_from_fit(
    fit: ProbitMediationFit,
    rho: float,
    n_mc: int = 200_000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """``probit_natural_effects`` applied to a whole fit, intercepts included.

    One place where a fit is converted to effects, so no call site can forget to
    pass the intercepts through. That omission is exactly the failure mode the
    2026-09-07 repair is about.
    """
    return probit_natural_effects(
        fit.alpha,
        fit.beta,
        fit.gamma,
        fit.sigma_m,
        float(rho),
        n_mc=n_mc,
        rng_seed=rng_seed,
        mu_m=fit.mu_m,
        alpha0=fit.alpha0,
    )


def _joint_negloglik(
    params: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho: float,
) -> float:
    """Negative log-likelihood of the probit mediation model at fixed ``rho``.

    Accepts either parameterisation, dispatching on the length of ``params``:

    * 4 entries ``(alpha, beta, gamma, log_sigma_m)`` - the legacy,
      intercept-free model. This branch is byte-identical to the pre-repair
      function so ``intercepts=False`` reproduces historical fits exactly.
    * 6 entries ``(alpha, beta, gamma, log_sigma_m, mu_m, alpha0)`` - the
      repaired model with mediator and outcome baseline intercepts.
    """
    if len(params) == 4:
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

    alpha, beta, gamma, log_sigma_m, mu_m, alpha0 = params
    sigma_m = np.exp(log_sigma_m)

    # Mediator equation: M | X ~ Normal(mu_m + gamma*X, sigma_m).
    resid_m = M - mu_m - gamma * X
    ll_m = norm.logpdf(resid_m, loc=0.0, scale=sigma_m).sum()

    # Outcome equation: probit with the confounding offset folded in.
    scale = np.sqrt(1.0 - rho**2)
    eta = (alpha0 + alpha * X + beta * M + (rho / sigma_m) * resid_m) / scale
    ll_y = np.where(Y == 1, norm.logcdf(eta), norm.logsf(eta)).sum()

    return -(ll_m + ll_y)


def fit_probit_mediation_map(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho: float,
    intercepts: bool = True,
    strict: bool = False,
) -> ProbitMediationFit:
    """Maximum-a-posteriori (here, MLE) fit of the probit model at fixed ``rho``.

    Fast deterministic optimisation, suitable for a dense ``rho`` grid and for
    tests.

    Parameters
    ----------
    intercepts:
        ``True`` (default) fits the repaired model with a mediator baseline
        ``mu_m`` and an outcome baseline ``alpha0``. ``False`` restores the
        pre-2026-09-07 model that forced both to zero; use it only to reproduce
        historical numbers, never for a new scientific estimate. The
        intercept-free model reports a large mediated effect on data with no
        causal effect whenever the mediator has a nonzero control-arm level
        (see the module docstring and ``tests/test_offset_null.py``).
    strict:
        ``True`` raises ``OptimizerFailure`` when the optimizer reports failure.
        The default warns with ``OptimizerWarning`` and returns the unconverged
        parameters, with ``fit.converged`` set to ``False``.

    Returns
    -------
    ProbitMediationFit
        ``(alpha, beta, gamma, sigma_m, mu_m, alpha0, converged)``. The first
        four fields are the pre-repair return value, in order.
    """
    if abs(rho) > _RHO_ABS_MAX:
        raise ValueError(f"|rho| must be <= {_RHO_ABS_MAX}.")
    _validate_xmy(X, M, Y)

    if intercepts:
        init = _init_with_intercepts(X, M, Y)
    else:
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
    if not result.success:
        message = (
            f"probit mediation fit did not converge at rho={float(rho):+.4f} "
            f"(intercepts={intercepts}): {result.message}"
        )
        if strict:
            raise OptimizerFailure(message)
        warnings.warn(message, OptimizerWarning, stacklevel=2)

    if intercepts:
        alpha, beta, gamma, log_sigma_m, mu_m, alpha0 = result.x
    else:
        alpha, beta, gamma, log_sigma_m = result.x
        mu_m, alpha0 = 0.0, 0.0
    return ProbitMediationFit(
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        sigma_m=float(np.exp(log_sigma_m)),
        mu_m=float(mu_m),
        alpha0=float(alpha0),
        converged=bool(result.success),
    )


def _init_with_intercepts(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Starting point for the intercept model: arm means for M, marginal probit for Y.

    Starting ``mu_m`` at the control-arm mean of ``M`` and ``alpha0`` at
    ``Phi^-1(mean(Y))`` matters on real mediators: a CoT step count sits far
    from zero, and an origin start leaves L-BFGS-B climbing out of a region
    where the probit is saturated.
    """
    x0 = X == 0
    mu0 = float(M[x0].mean()) if x0.any() else float(M.mean())
    x1 = X == 1
    mu1 = float(M[x1].mean()) if x1.any() else mu0
    gamma0 = mu1 - mu0
    resid0 = M - mu0 - gamma0 * X
    sigma0 = float(max(resid0.std(), 1e-2))
    p_y = float(np.clip(Y.mean(), 1e-3, 1.0 - 1e-3))
    alpha0_init = float(norm.ppf(p_y))
    return np.array([0.0, 0.0, gamma0, np.log(sigma0), mu0, alpha0_init])


def sensitivity_sweep(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho_grid: np.ndarray | None = None,
    n_mc: int = 100_000,
    rng_seed: int = 0,
    intercepts: bool = True,
) -> list[SensitivityPoint]:
    """Estimate the natural effects across a grid of assumed ``rho``.

    For each ``rho`` we fit the de-confounded coefficients and convert them to
    probability-scale natural effects. The resulting curve is the sensitivity
    analysis: how the faithfulness verdict (sign and size of the NIE) moves as
    the unmeasured-confounding assumption is relaxed.

    ``intercepts=False`` reproduces the pre-2026-09-07 sweep.
    """
    if rho_grid is None:
        rho_grid = np.round(np.arange(-0.6, 0.81, 0.1), 4)

    points: list[SensitivityPoint] = []
    for rho in rho_grid:
        fit = fit_probit_mediation_map(X, M, Y, float(rho), intercepts=intercepts)
        nde, nie, te = natural_effects_from_fit(
            fit, float(rho), n_mc=n_mc, rng_seed=rng_seed
        )
        points.append(
            SensitivityPoint(
                rho=float(rho),
                alpha=fit.alpha,
                beta=fit.beta,
                gamma=fit.gamma,
                sigma_m=fit.sigma_m,
                nde=nde,
                nie=nie,
                te=te,
                mu_m=fit.mu_m,
                alpha0=fit.alpha0,
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

    ``unresolved`` is ``True`` when there was no supported effect at ``rho = 0``
    to be robust *about* (see ``breakdown_frontier``). In that case the frontier
    fields carry no verdict: ``rho_star_pos``/``rho_star_neg`` are ``None`` and
    ``robustness`` is NaN.
    """

    key: str
    effect_at_zero: float
    rho_star_pos: float | None
    rho_star_neg: float | None
    robustness: float
    rho_max: float
    survives_full_range: bool
    unresolved: bool = False


def breakdown_frontier(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    key: str = "nie",
    n_mc: int = 200_000,
    rng_seed: int = 0,
    rho_max: float = 0.95,
    intercepts: bool = True,
    min_effect: float = 0.0,
) -> BreakdownFrontier:
    """Smallest residual confounding ``|rho|`` that overturns the sign of an effect.

    Starting from the sequential-ignorability estimate (``rho = 0``), find the
    nearest ``rho`` on each side at which the chosen natural effect (``"nde"``,
    ``"nie"`` or ``"te"``) crosses zero, by refitting the probit model at each
    candidate ``rho`` and converting to natural effects. Monte Carlo uses a fixed
    seed so the effect is a deterministic, smooth function of ``rho`` (common random
    numbers), which keeps the root-find stable.

    What rho* is, and what it is not
    --------------------------------
    ``rho*`` is **the robustness of the sign of the M-to-Y coefficient**, and
    nothing more. It is not a measure of direct bypass, not a severity scale for
    a hidden trigger, and not a ranking of how trustworthy a model's reasoning
    is. For the NIE the reason is exact: with ``gamma != 0`` the NIE vanishes if
    and only if ``beta(rho) = 0``, and by the reparameterisation in the module
    docstring ``beta(rho) = B*sqrt(1-rho^2) - rho/sigma_m`` with ``B`` the
    ``rho = 0`` fitted mediator coefficient. Its zero crossing is

        rho* = |B| * sigma_m / sqrt(1 + B^2 * sigma_m^2)

    which contains neither ``alpha`` nor either intercept. Increase the direct
    bypass ``alpha`` and the mediated share collapses while ``rho*`` does not
    move at all: the review of 6 September 2026 (section 3.4) tabulates
    NIE/TE falling from 100% to 1.6% as ``alpha`` goes 0 to 3, with ``rho*``
    fixed at 0.624695 throughout. ``tests/test_rho_star_semantics.py`` pins that
    invariance so nobody can quietly reinterpret ``rho*`` as a bypass measure.
    Report NDE, NIE, TE and their uncertainty beside ``rho*``, never ``rho*``
    alone.

    A large ``rho*`` is also compatible with zero true mediation: for the pure
    shared-cause process ``M = X + U``, ``Y = 1[X + U + E > 0]`` there is no
    M-to-Y arrow at all, yet the compatible zero crossing is about 0.707. The
    sensitivity analysis is correctly exposing that alternative, not endorsing
    the mediated reading.

    Unresolved verdicts
    -------------------
    A frontier only means something if there is a supported effect at ``rho = 0``
    to defend. When ``|effect_at_zero| <= min_effect`` there is none, and the
    old code rewarded that absence: no crossing was found on either side, so it
    returned ``survives_full_range=True`` and the maximum possible robustness.
    Such a result is now returned with ``unresolved=True``, both crossings
    ``None``, ``survives_full_range=False`` and ``robustness=NaN``. ``min_effect``
    defaults to 0.0, so only an exactly-zero effect is unresolved unless the
    caller sets a practical threshold; pass the smallest effect worth defending
    to get the honest verdict.

    ``intercepts=False`` reproduces the pre-2026-09-07 frontier, including the
    baseline-offset defect. The public real-model ``rho*`` values (0.708 to
    0.800) were computed that way; see ``docs/ESTIMATOR-REPAIR-2026-09-07.md``.
    """
    if key not in {"nde", "nie", "te"}:
        raise ValueError("key must be one of 'nde', 'nie', 'te'.")
    if min_effect < 0.0:
        raise ValueError("min_effect must be non-negative.")
    _validate_xmy(X, M, Y)
    rho_max = min(float(rho_max), _RHO_ABS_MAX - 1e-3)

    def effect_at(rho: float) -> float:
        fit = fit_probit_mediation_map(X, M, Y, float(rho), intercepts=intercepts)
        nde, nie, te = natural_effects_from_fit(
            fit, float(rho), n_mc=n_mc, rng_seed=rng_seed
        )
        return {"nde": nde, "nie": nie, "te": te}[key]

    e0 = effect_at(0.0)
    sign0 = np.sign(e0)

    if abs(e0) <= min_effect:
        # No supported effect at rho = 0: there is no conclusion to be robust
        # about, so report UNRESOLVED rather than a frontier.
        return BreakdownFrontier(
            key=key,
            effect_at_zero=float(e0),
            rho_star_pos=None,
            rho_star_neg=None,
            robustness=float("nan"),
            rho_max=float(rho_max),
            survives_full_range=False,
            unresolved=True,
        )

    def crossing(bound: float) -> float | None:
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
        unresolved=False,
    )


@dataclass(frozen=True)
class PartialIDBounds:
    """Worst-case bounds on a natural effect over a plausible confounding range.

    The residual M-Y correlation ``rho`` is unknown but, by assumption, bounded:
    ``|rho| <= rho_bar``. The effect is then not point-identified, but it provably
    lies in ``[lower, upper]`` (the min and max over that range). If the interval
    excludes zero the effect is **sign-identified**: the faithfulness verdict holds
    for every level of confounding you are willing to entertain, with no appeal to
    an unverifiable ``rho = 0`` assumption. This is the honest object to report
    alongside a point estimate.
    """

    key: str
    rho_bar: float
    lower: float
    upper: float
    rho_at_lower: float
    rho_at_upper: float
    sign_identified: bool


def partial_identification_bounds(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    rho_bar: float = 0.5,
    key: str = "nie",
    n_grid: int = 41,
    n_mc: int = 150_000,
    rng_seed: int = 0,
    intercepts: bool = True,
) -> PartialIDBounds:
    """Sharp bounds on a natural effect when ``rho`` is only known to satisfy ``|rho| <= rho_bar``.

    Evaluates the effect on a grid of ``rho`` in ``[-rho_bar, rho_bar]`` (refit + natural
    effects at each, common random numbers for a smooth curve) and returns the min and
    max as the identified interval. Use it to make a confounding-agnostic statement:
    "for any residual correlation up to ``rho_bar``, the faithful path is in
    ``[lower, upper]``" and whether that interval keeps its sign.

    ``intercepts=False`` reproduces the pre-2026-09-07 bounds.
    """
    if key not in {"nde", "nie", "te"}:
        raise ValueError("key must be one of 'nde', 'nie', 'te'.")
    _validate_xmy(X, M, Y)
    rho_bar = min(abs(float(rho_bar)), _RHO_ABS_MAX - 1e-3)
    grid = np.linspace(-rho_bar, rho_bar, n_grid)

    vals = np.empty(n_grid)
    for i, rho in enumerate(grid):
        fit = fit_probit_mediation_map(X, M, Y, float(rho), intercepts=intercepts)
        nde, nie, te = natural_effects_from_fit(
            fit, float(rho), n_mc=n_mc, rng_seed=rng_seed
        )
        vals[i] = {"nde": nde, "nie": nie, "te": te}[key]

    i_lo = int(np.argmin(vals))
    i_hi = int(np.argmax(vals))
    lower, upper = float(vals[i_lo]), float(vals[i_hi])
    sign_identified = lower > 0.0 or upper < 0.0
    return PartialIDBounds(
        key=key,
        rho_bar=float(rho_bar),
        lower=lower,
        upper=upper,
        rho_at_lower=float(grid[i_lo]),
        rho_at_upper=float(grid[i_hi]),
        sign_identified=sign_identified,
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
