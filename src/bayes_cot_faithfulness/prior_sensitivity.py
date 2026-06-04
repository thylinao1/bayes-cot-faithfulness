"""Prior-sensitivity sweep for the hierarchical mediation model.

A Bayesian faithfulness number is only as trustworthy as its priors. The
hierarchical model in ``hierarchical.py`` puts a half-prior on the pooling scale
``tau_beta`` (how much prompts disagree about the faithful path) and a normal
prior on the population slope ``mu_beta`` (the headline faithfulness number). If
the headline moves a lot when those priors are swapped for equally defensible
alternatives, the conclusion is being driven by the prior rather than the data,
and that has to be reported.

This module runs the standard discipline: vary ONE hyperprior at a time, refit,
and tabulate how far the headline estimate (default ``mu_beta``) and its 94 percent
HDI move relative to a fixed baseline. Two axes are supported, each a one-at-a-time
change off the baseline:

* the pooling-scale prior family on ``tau_beta``: HalfNormal vs Exponential vs
  HalfStudentT, matched to a comparable prior mean so the comparison isolates the
  shape of the prior (tail weight) rather than smuggling in a different scale;
* the coefficient-prior standard deviation on ``mu_beta``: a tighter or wider
  normal, to check the headline is not sensitive to how informative that prior is.

The model refit here mirrors ``hierarchical.fit_hierarchical_mediation`` exactly
(same non-centred parameterisation, same likelihood) and only substitutes the one
targeted prior, so any movement in the headline is attributable to that prior and
nothing else. Fits are deliberately kept small and deterministic (fixed seed,
single core) so the sweep is fast and testable; scale up draws for a publication
run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default HDI mass. 94 percent is the ArviZ default and avoids the false
# precision people read into a round 95.
_HDI_PROB = 0.94


@dataclass(frozen=True)
class PriorSpec:
    """One prior configuration to refit under.

    Exactly one hyperprior differs from the baseline. ``tau_beta_family`` selects
    the pooling-scale prior shape; ``tau_beta_scale`` is its scale parameter
    (sigma for the half-normal and half-Student-t, mean for the exponential, all
    tuned to a comparable prior mean by ``_build_model``). ``mu_beta_sd`` is the
    standard deviation of the normal prior on the population slope. Leaving a
    field at its baseline value keeps that prior fixed.

    Attributes
    ----------
    label:
        Short human-readable name for the table row.
    tau_beta_family:
        One of ``"halfnormal"``, ``"exponential"``, ``"halfstudentt"``.
    tau_beta_scale:
        Scale of the pooling-scale prior (see above).
    tau_beta_nu:
        Degrees of freedom for the half-Student-t family (ignored otherwise).
    mu_beta_sd:
        Standard deviation of the ``mu_beta`` normal prior.
    """

    label: str
    tau_beta_family: str = "halfnormal"
    tau_beta_scale: float = 1.0
    tau_beta_nu: float = 4.0
    mu_beta_sd: float = 2.0


@dataclass(frozen=True)
class PriorSensitivityRow:
    """Headline estimate under one prior, with shift relative to the baseline."""

    label: str
    parameter: str
    estimate: float
    hdi_low: float
    hdi_high: float
    is_baseline: bool
    shift_vs_baseline: float


@dataclass(frozen=True)
class PriorSensitivityResult:
    """Full prior-sensitivity table plus the headline robustness summary."""

    parameter: str
    baseline_label: str
    rows: tuple[PriorSensitivityRow, ...] = field(default_factory=tuple)

    @property
    def max_abs_shift(self) -> float:
        """Largest absolute shift of the headline across all non-baseline priors."""
        shifts = [abs(r.shift_vs_baseline) for r in self.rows if not r.is_baseline]
        return max(shifts) if shifts else 0.0


def pooling_family_specs() -> tuple[PriorSpec, ...]:
    """The pooling-scale prior-family sweep, matched to a comparable prior mean.

    This is the core robustness exhibit: it varies only the *shape* of the prior
    on ``tau_beta`` (its tail weight) while holding the prior mean near 0.8, so any
    movement in the headline is attributable to the family choice and nothing
    else. On synthetic data the headline should barely move across these, because
    a half-normal, an exponential and a heavy-tailed half-Student-t with the same
    bulk imply nearly the same amount of partial pooling.
    """
    return (
        PriorSpec(label="baseline: HalfNormal(1.0)"),
        PriorSpec(label="Exponential(mean=0.8)", tau_beta_family="exponential", tau_beta_scale=0.8),
        PriorSpec(
            label="HalfStudentT(nu=4, 0.8)",
            tau_beta_family="halfstudentt",
            tau_beta_scale=0.8,
            tau_beta_nu=4.0,
        ),
    )


def default_prior_specs() -> tuple[PriorSpec, ...]:
    """A small, defensible sweep: a baseline plus one-at-a-time prior changes.

    The pooling-scale families are matched to a comparable prior mean (~0.8) so
    the contrast is the prior's tail weight, not its scale:

    * HalfNormal(sigma=1.0)            mean ~ 0.80   (baseline)
    * Exponential(mean=0.8)            mean = 0.80
    * HalfStudentT(nu=4, sigma=0.8)    mean ~ 0.80, heavier tail

    The coefficient-prior rows widen and tighten the ``mu_beta`` normal sd. Unlike
    the pooling-family rows, a tight coefficient prior is *meant* to move a small
    sample's headline (that is what informativeness does), so read those two rows
    as a sensitivity probe rather than a robustness guarantee.
    """
    return (
        *pooling_family_specs(),
        PriorSpec(label="mu_beta sd=1.0 (tight)", mu_beta_sd=1.0),
        PriorSpec(label="mu_beta sd=3.0 (wide)", mu_beta_sd=3.0),
    )


def _build_model(
    spec: PriorSpec,
    group: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
):
    """Build the hierarchical mediation model with one prior swapped per ``spec``.

    Identical to ``hierarchical.fit_hierarchical_mediation`` except for the
    ``tau_beta`` prior family/scale and the ``mu_beta`` prior sd.
    """
    import pymc as pm

    n_groups = int(group.max()) + 1
    model = pm.Model()
    with model:
        mu_alpha = pm.Normal("mu_alpha", 0.0, 1.5)
        mu_beta = pm.Normal("mu_beta", 0.0, spec.mu_beta_sd)
        mu_gamma = pm.Normal("mu_gamma", 0.0, 1.5)
        tau_alpha = pm.HalfNormal("tau_alpha", 1.0)
        tau_beta = _tau_beta_prior(spec)
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
    return model


def _tau_beta_prior(spec: PriorSpec):
    """Construct the pooling-scale prior named by ``spec`` (called inside a model)."""
    import pymc as pm

    family = spec.tau_beta_family.lower()
    if family == "halfnormal":
        return pm.HalfNormal("tau_beta", spec.tau_beta_scale)
    if family == "exponential":
        # Exponential mean is 1/lam; expose the mean as the scale for parity.
        if spec.tau_beta_scale <= 0.0:
            raise ValueError("tau_beta_scale (exponential mean) must be positive.")
        return pm.Exponential("tau_beta", 1.0 / spec.tau_beta_scale)
    if family == "halfstudentt":
        return pm.HalfStudentT("tau_beta", nu=spec.tau_beta_nu, sigma=spec.tau_beta_scale)
    raise ValueError(
        f"Unknown tau_beta_family {spec.tau_beta_family!r}; "
        "expected 'halfnormal', 'exponential' or 'halfstudentt'."
    )


def _fit_headline(
    spec: PriorSpec,
    group: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    parameter: str,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    target_accept: float,
    random_seed: int,
) -> tuple[float, float, float]:
    """Refit under ``spec`` and return ``(posterior_mean, hdi_low, hdi_high)``."""
    import arviz as az
    import pymc as pm

    model = _build_model(spec, group, X, M, Y)
    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            cores=1,  # deterministic, in-process sampling for a reproducible sweep
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )

    draws = trace.posterior[parameter].values.flatten()
    hdi = np.asarray(az.hdi(draws, hdi_prob=_HDI_PROB)).ravel()
    return float(draws.mean()), float(hdi[0]), float(hdi[1])


def prior_sensitivity_sweep(
    group: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    specs: tuple[PriorSpec, ...] | None = None,
    parameter: str = "mu_beta",
    n_samples: int = 400,
    n_tune: int = 400,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 0,
) -> PriorSensitivityResult:
    """Refit the hierarchical model under each prior spec and tabulate the headline.

    The first spec is treated as the baseline; every other row's
    ``shift_vs_baseline`` is its headline estimate minus the baseline estimate.
    A small ``max_abs_shift`` is the robustness statement: the faithfulness
    headline barely moves under reasonable changes of prior.

    Parameters
    ----------
    group, X, M, Y:
        Hierarchical traces, as produced by ``simulate_hierarchical_cot``.
    specs:
        Prior configurations to sweep. Defaults to ``default_prior_specs()``.
    parameter:
        Which posterior scalar is the headline. Defaults to ``"mu_beta"`` (the
        population faithfulness slope).
    n_samples, n_tune, n_chains, target_accept, random_seed:
        Sampler controls. Defaults are intentionally small for a fast, testable
        sweep; raise ``n_samples`` / ``n_tune`` for a production run.

    Returns
    -------
    PriorSensitivityResult
        One row per spec plus the ``max_abs_shift`` summary.
    """
    _validate(group, X, M, Y)
    if specs is None:
        specs = default_prior_specs()
    if len(specs) == 0:
        raise ValueError("specs must contain at least one prior specification.")

    baseline = specs[0]
    baseline_mean, baseline_lo, baseline_hi = _fit_headline(
        baseline, group, X, M, Y, parameter,
        n_samples, n_tune, n_chains, target_accept, random_seed,
    )

    rows = [
        PriorSensitivityRow(
            label=baseline.label,
            parameter=parameter,
            estimate=baseline_mean,
            hdi_low=baseline_lo,
            hdi_high=baseline_hi,
            is_baseline=True,
            shift_vs_baseline=0.0,
        )
    ]

    for spec in specs[1:]:
        mean, lo, hi = _fit_headline(
            spec, group, X, M, Y, parameter,
            n_samples, n_tune, n_chains, target_accept, random_seed,
        )
        rows.append(
            PriorSensitivityRow(
                label=spec.label,
                parameter=parameter,
                estimate=mean,
                hdi_low=lo,
                hdi_high=hi,
                is_baseline=False,
                shift_vs_baseline=mean - baseline_mean,
            )
        )

    return PriorSensitivityResult(
        parameter=parameter,
        baseline_label=baseline.label,
        rows=tuple(rows),
    )


def _validate(group: np.ndarray, X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> None:
    if not (len(group) == len(X) == len(M) == len(Y)):
        raise ValueError("group, X, M, Y must all have the same length.")
    if group.min() != 0 or set(np.unique(group)) != set(range(int(group.max()) + 1)):
        raise ValueError("group must be contiguous integers starting at 0.")
    if not set(np.unique(X)).issubset({0, 1}):
        raise ValueError("X must be binary {0, 1}.")
    if not set(np.unique(Y)).issubset({0, 1}):
        raise ValueError("Y must be binary {0, 1}.")
