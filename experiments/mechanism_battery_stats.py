"""Aggregation, binomial intervals, cross-checks and the PyMC subset for the CPU battery.

Split out of ``experiments/mechanism_battery.py`` so each file stays readable. Nothing
here imports the battery module: the aggregation functions take the mechanism and the
per-dataset results as plain objects and read their attributes, so the two files have no
import cycle and either can be loaded by file path on its own.

Everything reported from here carries its denominator, because the battery's whole
purpose is to say how often the instrument is right, not that it is.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as beta_dist

from bayes_cot_faithfulness.mediation import (
    extract_intercept_samples,
    extract_parameter_samples,
    fit_mediation_model,
    natural_effects_from_trace,
)
from bayes_cot_faithfulness.sensitivity import fit_probit_mediation_map

CRI = 0.95


def clopper_pearson_interval(k: int, n: int, conf: float = CRI) -> tuple[float, float]:
    """Two-sided exact binomial interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        return 0.0, 1.0
    if k < 0 or k > n:
        raise ValueError("k must lie in [0, n].")
    tail = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - tail, k + 1, n - k))
    return lo, hi


# --------------------------------------------------------------------------- #
# Aggregation over the seeded datasets of one condition
# --------------------------------------------------------------------------- #
def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _effect_block(est: np.ndarray, lo: np.ndarray, hi: np.ndarray, truth: float) -> dict:
    """Bias with its Monte Carlo standard error, plus coverage with a binomial interval."""
    n = len(est)
    bias = est - truth
    covered = int(np.sum((lo <= truth) & (truth <= hi)))
    cp_lo, cp_hi = clopper_pearson_interval(covered, n)
    return {
        "truth": float(truth),
        "estimate_mean": float(est.mean()),
        "bias_mean": float(bias.mean()),
        "bias_mcse": float(bias.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "rmse": float(math.sqrt(float(np.mean(bias**2)))),
        "coverage_k": covered,
        "coverage_n": n,
        "coverage": covered / n,
        "coverage_ci_lo": cp_lo,
        "coverage_ci_hi": cp_hi,
        "mean_interval_width": float(np.mean(hi - lo)),
    }


def aggregate(mech, rows, truth_mc: tuple[float, float, float]) -> dict:
    """Everything the report prints for one condition at one sample size, with denominators."""
    n = len(rows)
    est = {k: np.array([getattr(r, k) for r in rows]) for k in ("nde", "nie", "te")}
    lo = {k: np.array([getattr(r, f"{k}_lo") for r in rows]) for k in ("nde", "nie", "te")}
    hi = {k: np.array([getattr(r, f"{k}_hi") for r in rows]) for k in ("nde", "nie", "te")}

    load_bearing = int(sum(r.load_bearing_at_zero for r in rows))
    lb_lo, lb_hi = clopper_pearson_interval(load_bearing, n)
    truth_nie_zero = abs(mech.analytic_truth[1]) < 1e-12

    decisions = [r.rho_star_decision for r in rows if r.rho_star_decision is not None]
    no_crossing = int(sum(r.decision_no_crossing_in_range for r in rows))
    rs = np.array([r.rho_star_point for r in rows])

    # NIE/TE is descriptive and is not printed for any dataset whose TE interval
    # includes zero, so the share is computed on exactly that subset.
    reportable = [r for r in rows if (r.te_lo > 0.0 or r.te_hi < 0.0) and abs(r.te) > 1e-9]
    te_excl_zero = len(reportable)
    shares = np.array([r.nie / r.te for r in reportable])

    return {
        "condition": mech.key,
        "family": mech.family,
        "label": mech.label,
        "n_rows": rows[0].n_rows,
        "n_datasets": n,
        "truth_analytic": {
            "nde": mech.analytic_truth[0],
            "nie": mech.analytic_truth[1],
            "te": mech.analytic_truth[2],
        },
        "truth_monte_carlo": {"nde": truth_mc[0], "nie": truth_mc[1], "te": truth_mc[2]},
        "truth_max_abs_difference": float(
            max(abs(a - b) for a, b in zip(mech.analytic_truth, truth_mc))
        ),
        "effects": {
            k: _effect_block(est[k], lo[k], hi[k], mech.analytic_truth[i])
            for i, k in enumerate(("nde", "nie", "te"))
        },
        "verdict": {
            "true_nie_is_zero": truth_nie_zero,
            "load_bearing_k": load_bearing,
            "load_bearing_n": n,
            "load_bearing_rate": load_bearing / n,
            "load_bearing_ci_lo": lb_lo,
            "load_bearing_ci_hi": lb_hi,
            "false_robust_verdict_rate": (load_bearing / n) if truth_nie_zero else None,
            "unresolved_k": n - load_bearing,
            "mean_prob_nie_above_threshold": float(
                np.mean([r.prob_nie_above_threshold_at_zero for r in rows])
            ),
        },
        "rho_star_point": {
            **_summary(rs),
            "in_range_k": int(sum(r.rho_star_point_in_range for r in rows)),
            "in_range_n": n,
            "mean_interval_lo": float(np.mean([r.rho_star_point_lo for r in rows])),
            "mean_interval_hi": float(np.mean([r.rho_star_point_hi for r in rows])),
        },
        "rho_star_decision": {
            "unresolved_k": n - load_bearing,
            "crossing_k": len(decisions),
            "no_crossing_in_range_k": no_crossing,
            "n": n,
            "median_crossing": float(np.median(decisions)) if decisions else None,
            "min_crossing": float(np.min(decisions)) if decisions else None,
            "max_crossing": float(np.max(decisions)) if decisions else None,
        },
        "mediated_share": {
            "mean": float(shares.mean()) if shares.size else None,
            "n": int(shares.size),
            "te_interval_excludes_zero_k": te_excl_zero,
            "reportable": te_excl_zero == n,
        },
        "diagnostics": {
            "non_converged_k": int(sum(not r.converged for r in rows)),
            "bootstrap_redraws_total": int(sum(r.degenerate_bootstrap_redraws for r in rows)),
            "mean_answer_rate": float(np.mean([r.answer_rate for r in rows])),
            "mean_mediator_mean": float(np.mean([r.mediator_mean for r in rows])),
            "mean_corr_m_y": float(np.mean([r.corr_m_y for r in rows])),
            "mean_fitted_beta": float(np.mean([r.fitted_beta for r in rows])),
            "mean_fitted_gamma": float(np.mean([r.fitted_gamma for r in rows])),
        },
    }


# --------------------------------------------------------------------------- #
# The PyMC subset: the repaired posterior instead of the bootstrap
# --------------------------------------------------------------------------- #
def pymc_subset(
    mech,
    n_rows: int,
    seeds: list[int],
    map_rows: list,
    n_samples: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 4,
) -> dict:
    """Re-run a subset of one condition's datasets through the repaired PyMC posterior.

    The battery's headline intervals are bootstrap intervals around the MAP fit, because
    a full posterior on 2,200 datasets does not fit on the machine this runs on. This
    function re-does a named subset with ``fit_mediation_model(..., intercepts=True)`` and
    the posterior natural effects, so the report can say how far the cheap interval is
    from the expensive one instead of assuming they agree.

    One difference is deliberate and is reported, not hidden: the PyMC model's outcome
    equation is logistic while every generator here has a Gaussian latent (probit) outcome
    and the MAP estimator is probit. The probability-scale natural effects are close under
    either link once the coefficients are refitted, and the comparison below measures how
    close on these datasets rather than asserting it.

    ``map_rows`` are the already-computed MAP results for the same seeds, in the same
    order, so the two paths are compared on identical data.
    """
    import arviz as az

    truth = mech.analytic_truth
    per_dataset = []
    for seed, map_row in zip(seeds, map_rows):
        x, m, y = mech.draw(n_rows, seed)
        trace = fit_mediation_model(
            x, m, y, n_samples=n_samples, n_tune=n_tune, n_chains=n_chains,
            random_seed=int(seed % 2**31), progressbar=False, intercepts=True,
        )
        eff = natural_effects_from_trace(trace)
        summary = az.summary(trace, var_names=["alpha", "beta", "gamma", "sigma_m"])
        per_dataset.append(
            {
                "seed": int(seed),
                "max_r_hat": float(summary["r_hat"].max()),
                "divergences": int(trace.sample_stats["diverging"].values.sum()),
                "nde": eff.nde_mean, "nde_lo": eff.nde_lo, "nde_hi": eff.nde_hi,
                "nie": eff.nie_mean, "nie_lo": eff.nie_lo, "nie_hi": eff.nie_hi,
                "te": eff.te_mean, "te_lo": eff.te_lo, "te_hi": eff.te_hi,
                "map_nde": map_row.nde, "map_nie": map_row.nie, "map_te": map_row.te,
                "map_nie_lo": map_row.nie_lo, "map_nie_hi": map_row.nie_hi,
            }
        )

    out = {
        "condition": mech.key,
        "family": mech.family,
        "n_rows": n_rows,
        "n_datasets": len(per_dataset),
        "truth": {"nde": truth[0], "nie": truth[1], "te": truth[2]},
        "max_r_hat": max(d["max_r_hat"] for d in per_dataset),
        "total_divergences": sum(d["divergences"] for d in per_dataset),
        "per_dataset": per_dataset,
    }
    for i, key in enumerate(("nde", "nie", "te")):
        est = np.array([d[key] for d in per_dataset])
        lo = np.array([d[f"{key}_lo"] for d in per_dataset])
        hi = np.array([d[f"{key}_hi"] for d in per_dataset])
        out[key] = _effect_block(est, lo, hi, truth[i])
    map_nie = np.array([d["map_nie"] for d in per_dataset])
    gap = np.abs(np.array([d["nie"] for d in per_dataset]) - map_nie)
    out["posterior_minus_map_nie"] = {
        "mean_abs_difference": float(np.mean(gap)),
        "max_abs_difference": float(np.max(gap)),
        "mean_posterior_width": float(
            np.mean([d["nie_hi"] - d["nie_lo"] for d in per_dataset])
        ),
        "mean_bootstrap_width": float(
            np.mean([d["map_nie_hi"] - d["map_nie_lo"] for d in per_dataset])
        ),
    }
    return out


# The standard probit-to-logit coefficient scale factor: a probit slope b corresponds to a
# logistic slope of about 1.702 b for the same fitted probabilities.
LOGIT_SCALE = 1.702


def prior_scale_probe(mechs, n_rows: int, dataset_index: int, dataset_seed) -> list[dict]:
    """Why the posterior and the MAP path disagree on some families, on one dataset each.

    The PyMC model's outcome priors are fixed-scale (``alpha0 ~ Normal(0, 1.5)``,
    ``beta ~ Normal(0, 2)``, as written in ``mediation.fit_mediation_model``), while the
    mediator baseline got a scale-aware prior in the 2026-09-07 repair. On a mediator with
    a large baseline level that also drives the answer, the implied outcome intercept sits
    many prior standard deviations from zero, so the posterior shrinks it and the mediator
    coefficient together. This probe records, for one seeded dataset per mechanism, the
    posterior for those two parameters beside the probit MAP fit converted to the logit
    scale, so the report can name the mechanism instead of describing a gap.
    """
    out = []
    for mech in mechs:
        x, m, y = mech.draw(n_rows, dataset_seed(dataset_index, n_rows))
        trace = fit_mediation_model(
            x, m, y, n_samples=1000, n_tune=1000, n_chains=4,
            random_seed=dataset_index + 1, progressbar=False, intercepts=True,
        )
        _, beta, _, _ = extract_parameter_samples(trace)
        mu_m, alpha0 = extract_intercept_samples(trace)
        fit = fit_probit_mediation_map(x, m, y, rho=0.0, intercepts=True)
        out.append(
            {
                "condition": mech.key,
                "n_rows": n_rows,
                "dataset_index": dataset_index,
                "posterior_beta_mean": float(beta.mean()),
                "posterior_beta_lo": float(np.quantile(beta, 0.025)),
                "posterior_beta_hi": float(np.quantile(beta, 0.975)),
                "posterior_alpha0_mean": float(alpha0.mean()),
                "posterior_alpha0_lo": float(np.quantile(alpha0, 0.025)),
                "posterior_alpha0_hi": float(np.quantile(alpha0, 0.975)),
                "posterior_mu_m_mean": float(mu_m.mean()),
                "map_beta": float(fit.beta),
                "map_alpha0": float(fit.alpha0),
                "map_mu_m": float(fit.mu_m),
                "map_beta_on_logit_scale": float(fit.beta * LOGIT_SCALE),
                "map_alpha0_on_logit_scale": float(fit.alpha0 * LOGIT_SCALE),
            }
        )
    return out
