"""Bayesian mediation model in PyMC.

The model jointly fits:

    Mediator equation:   M | X ~ Normal(mu_m + gamma * X, sigma_m)
    Outcome equation:    Y | X, M ~ Bernoulli(Phi(alpha0 + alpha * X + beta * M))

with weakly informative priors. Posterior samples over (alpha, beta, gamma,
sigma_m, mu_m, alpha0) are then converted into a posterior over the natural
direct and natural indirect effects on the probability scale (see
``natural_effects_from_trace``, which is the only conversion that reads the link
off the trace instead of assuming one).

``mu_m`` and ``alpha0`` are the baseline intercepts added in the 2026-09-07
estimator repair and are on by default. Without them the model asserts
``E[M | X=0] = 0`` and a clean-arm answer rate of 0.5, and on a mediator with a
natural baseline level (a CoT step count, a truncation depth) it reads that
level as a treatment shift and reports mediation where there is none. Pass
``intercepts=False`` only to reproduce a pre-repair fit.

The link (2026-09-07, W4c)
--------------------------
The outcome link is **probit** by default, which is the link the pre-registered
estimand, ``sensitivity.fit_probit_mediation_map``, ``sensitivity.probit_natural_effects``
and ``closed_form.probit_natural_effects_closed_form`` are all written on. Until
this change this model used a logistic link while every other path in the package
used a probit one, so the posterior draws were pushed through probit natural-effect
formulas in the analysis and compared against probit truths in the mechanism
battery. ``link="logit"`` keeps the logistic model available for the logistic
data generator in ``synthetic.py``; it is not the estimator of record.

The priors (2026-09-07, W4c)
----------------------------
Every prior is stated relative to the mediator's own spread, so the same model
applies whether the mediator is counted in reasoning steps, tokens or thousands
of tokens. The mediator is centred inside the outcome equation, so the free
outcome intercept is the index at the *mean* mediator rather than at M = 0. With
fixed-scale outcome priors (``alpha0 ~ Normal(0, 1.5)``, ``beta ~ Normal(0, 2)``)
a mediator with a baseline near six forces an intercept many prior standard
deviations from zero, and the posterior shrinks the intercept and the mediator
coefficient together: NIE coverage in the CPU mechanism battery was 7/20 in the
rationalization family and 12/20 in the opposing-effects family, against 93/100
for the MAP bootstrap on the same data. See docs/ESTIMATOR-PRIORS-2026-09-07.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bayes_cot_faithfulness.effects import PosteriorEffects, posterior_natural_effects

if TYPE_CHECKING:  # pragma: no cover
    import arviz as az

# Prior widths, all stated on the index scale of whichever link is in use, so a
# change of mediator unit cannot change what the prior says.
INDEX_PRIOR_SD = 1.5  # direct effect and the centred outcome intercept
MEDIATOR_INDEX_PRIOR_SD = 2.0  # implied index contribution of one sd of M
GAMMA_PRIOR_SD_IN_M_SD = 1.5  # treatment shift, in mediator standard deviations
LINKS = ("probit", "logit")
LINK_ATTR = "outcome_link"


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
    link: str = "probit",
) -> az.InferenceData:
    """Fit a Bayesian mediation model and return posterior samples.

    Parameters
    ----------
    X : ndarray of shape (n,), int
        Binary prompt feature.
    M : ndarray of shape (n,), float
        Continuous CoT-as-mediator. Must not be constant: every prior below is
        stated relative to ``sd(M)``.
    Y : ndarray of shape (n,), int
        Binary answer.
    intercepts : bool
        ``True`` (default) adds the mediator baseline ``mu_m`` and the outcome
        baseline ``alpha0``. ``False`` restores the pre-2026-09-07 model, whose
        graph, variable names and fixed-scale priors are unchanged; with
        ``link="logit"`` that combination reproduces historical fits exactly.
    link : {"probit", "logit"}
        Outcome link. ``"probit"`` (default) is the link of the pre-registered
        estimand and of every other estimator path in this package.

    Priors, with the reason for each
    --------------------------------
    Write ``mbar = mean(M)`` and ``s = sd(M)``. With ``intercepts=True`` the
    outcome index is ``alpha0_centered + alpha*X + beta*(M - mbar)`` and

    * ``mu_m ~ Normal(mbar, s)`` - the mediator baseline sits where the control
      arm sits, so a step count in the tens is not fought by the prior.
    * ``gamma ~ Normal(0, 1.5*s)`` - a treatment shift of a few mediator standard
      deviations, which is the same statement in any mediator unit.
    * ``sigma_m ~ HalfNormal(s)`` - likewise for the mediator's own spread.
    * ``beta ~ Normal(0, 2/s)`` - so the implied index contribution of one
      standard deviation of the mediator, ``beta*s``, is ``Normal(0, 2)`` for
      every mediator unit. A fixed ``Normal(0, 2)`` on ``beta`` says something
      different for a mediator measured in steps than for one measured in tokens.
    * ``alpha0_centered ~ Normal(0, 1.5)`` - the index at the mean mediator in
      the clean arm, that is, a clean-arm answer rate anywhere in roughly
      [0.07, 0.93] at one prior sd. The reported ``alpha0`` is the deterministic
      un-centred value ``alpha0_centered - beta*mbar``, so every downstream
      consumer (``extract_intercept_samples``, the natural-effect converters)
      sees the same parameterisation it saw before.
    * ``alpha ~ Normal(0, 1.5)`` - a direct effect on the index scale.

    Centring is a reparameterisation of the same model, so the estimand is
    unchanged: the natural effects depend on the intercepts only through
    ``alpha0 + beta*mu_m``, which centring leaves alone. What changes is where
    the prior mass sits. ``tests/test_scale_aware_priors.py`` pins both
    invariances (shift and scale) on data with a large baseline.

    Returns
    -------
    arviz.InferenceData
        Trace with posterior samples for alpha, beta, gamma, sigma_m and, when
        ``intercepts`` is on, mu_m and alpha0. ``trace.attrs["outcome_link"]``
        records the link, so a converter never has to guess it.
    """
    import pymc as pm
    import pytensor.tensor as pt

    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}.")
    if len(X) != len(M) or len(M) != len(Y):
        raise ValueError("X, M, Y must all have the same length.")
    if X.ndim != 1:
        raise ValueError("X must be 1-D.")
    if not set(np.unique(X)).issubset({0, 1}):
        raise ValueError("X must be binary {0, 1}.")
    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be binary {0, 1}.")

    M = np.asarray(M, dtype=float)
    m_bar = float(np.mean(M))
    m_sd = _mediator_scale(M)

    with pm.Model() as _model:
        if intercepts:
            alpha = pm.Normal("alpha", mu=0.0, sigma=INDEX_PRIOR_SD)
            beta = pm.Normal("beta", mu=0.0, sigma=MEDIATOR_INDEX_PRIOR_SD / m_sd)
            gamma = pm.Normal("gamma", mu=0.0, sigma=GAMMA_PRIOR_SD_IN_M_SD * m_sd)
            sigma_m = pm.HalfNormal("sigma_m", sigma=m_sd)
            mu_m = pm.Normal("mu_m", mu=m_bar, sigma=m_sd)
            alpha0_c = pm.Normal("alpha0_centered", mu=0.0, sigma=INDEX_PRIOR_SD)
            # What downstream reads: the intercept of the un-centred equation.
            pm.Deterministic("alpha0", alpha0_c - beta * m_bar)
            mean_m = mu_m + gamma * X
            index = alpha0_c + alpha * X + beta * (M - m_bar)
        else:
            # The pre-2026-09-07 model, unchanged: fixed-scale priors, no
            # intercepts, mediator uncentred. Kept so historical numbers stay
            # reproducible from this repository, never for a new estimate.
            alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
            beta = pm.Normal("beta", mu=0.0, sigma=2.0)
            gamma = pm.Normal("gamma", mu=0.0, sigma=1.5)
            sigma_m = pm.HalfNormal("sigma_m", sigma=1.0)
            mean_m = gamma * X
            index = alpha * X + beta * M

        pm.Normal("M_obs", mu=mean_m, sigma=sigma_m, observed=M)
        if link == "logit":
            pm.Bernoulli("Y_obs", logit_p=index, observed=Y)
        else:
            y_obs = np.asarray(Y, dtype=float)
            pm.Potential(
                "Y_obs_logp",
                pt.sum(
                    y_obs * _log_std_normal_cdf(index)
                    + (1.0 - y_obs) * _log_std_normal_cdf(-index)
                ),
            )

        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    trace.attrs[LINK_ATTR] = link
    trace.posterior.attrs[LINK_ATTR] = link
    return trace


def _log_std_normal_cdf(eta):
    """``log Phi(eta)`` for a pytensor expression, finite and differentiable everywhere.

    ``log(0.5 * erfc(-eta/sqrt(2)))`` underflows to ``-inf`` in the left tail, which
    would hand NUTS an infinite log-density during tuning. ``erfc(z) = erfcx(z) *
    exp(-z^2)`` moves the exponential out of the floating-point range and into the
    exponent, so ``log erfc(z) = log erfcx(z) - z^2`` is exact for large positive
    ``z``. Each branch is fed a clamped argument, so the branch that is *not*
    selected still evaluates to a finite number and cannot poison the gradient of
    ``pt.switch`` with a NaN.
    """
    import pytensor.tensor as pt

    z = -eta / np.sqrt(2.0)
    z_pos = pt.maximum(z, 0.0)
    z_neg = pt.minimum(z, 0.0)
    log_erfc = pt.switch(
        pt.gt(z, 0.0),
        pt.log(pt.erfcx(z_pos)) - z_pos**2,
        pt.log(pt.erfc(z_neg)),
    )
    return np.log(0.5) + log_erfc


def _mediator_scale(M: np.ndarray) -> float:
    """The mediator's own spread, which every scale-aware prior is stated against."""
    sd = float(np.std(M))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(
            "M must not be constant: the mediator priors are stated relative to sd(M)."
        )
    return sd


def extract_parameter_samples(
    trace: az.InferenceData,
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

    ``alpha0`` is the intercept of the *un-centred* outcome equation, which is what
    every natural-effect converter expects; the model samples the centred version
    and records this one as a deterministic, so the centring is invisible here.

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


def trace_link(trace: az.InferenceData) -> str:
    """The outcome link the trace was fitted with, read off the trace itself.

    Raises rather than guessing: a trace with no stamp predates the 2026-09-07 link
    audit, and the caller has to say which link it means.
    """
    for attrs in (getattr(trace, "attrs", {}) or {}, getattr(trace.posterior, "attrs", {}) or {}):
        if LINK_ATTR in attrs:
            return str(attrs[LINK_ATTR])
    raise ValueError(
        "This trace carries no outcome_link attribute (it predates the 2026-09-07 link "
        "audit). Pass link= explicitly rather than assuming one."
    )


def natural_effects_from_trace(
    trace: az.InferenceData,
    link: str | None = None,
    n_mc_per_draw: int = 2_000,
    cri: float = 0.95,
    rng_seed: int = 0,
) -> PosteriorEffects:
    """Convert a fitted trace to natural effects, on the link the trace was fitted with.

    The single place a trace becomes an effect, for the same reason
    ``sensitivity.natural_effects_from_fit`` exists: with two links and two
    parameterisations in the package, any call site that re-assembles the
    conversion by hand is one edit away from pushing logistic draws through a
    probit formula. That mismatch is exactly what the 2026-09-07 link audit found.
    """
    alpha, beta, gamma, sigma_m = extract_parameter_samples(trace)
    mu_m, alpha0 = extract_intercept_samples(trace)
    return posterior_natural_effects(
        alpha,
        beta,
        gamma,
        sigma_m,
        n_mc_per_draw=n_mc_per_draw,
        cri=cri,
        rng_seed=rng_seed,
        mu_m_samples=mu_m,
        alpha0_samples=alpha0,
        link=link if link is not None else trace_link(trace),
    )
