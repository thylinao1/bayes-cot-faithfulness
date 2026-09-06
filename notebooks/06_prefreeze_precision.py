"""Pre-freeze precision re-run through the full pre-registered specification (W4).

Owner of 01-SIZING.md section H. Section C of that file sized column B with the
MAP probit fit (`fit_probit_mediation_map`) on a NOISELESS mediator and no
weights. Section H requires a re-run through the fuller specification before the
Amendment A2 freeze. This script is that re-run.

Two independent parts.

PART B (column B, judge-free mediated fraction)
    Simulate the two pre-registered synthetic worlds with
    ``sensitivity.simulate_confounded_cot``, add additive Gaussian measurement
    noise to the mediator M, and fit the FULL PyMC probit mediation posterior at
    rho = 0 (2 chains, 1000 tune, 1000 draws, target_accept 0.9). Report the NIE
    posterior 95 percent half-width, coverage of truth, and max r-hat at
    n = 150, 300, 600 per cell and n = 3600 (model level).

    Why a hand-written PyMC model and not ``mediation.fit_mediation_model``:
    that function exists but uses a LOGIT link
    (``pm.Bernoulli(logit_p=alpha*X + beta*M)``). The pre-registered column-B
    estimand is the PROBIT specification (``sensitivity.py`` module docstring,
    plan section 1), because rho enters the latent-Gaussian probit cleanly. So
    the model below matches docs/methodology.md section 4 structurally (same
    mediator equation, same weakly informative priors) with the link changed to
    probit, as 01-SIZING section H directs. The likelihood is written as a
    Potential over ``logcdf`` of a standard normal, which is numerically stable
    in the tails where ``invprobit`` underflows.

PART A (column A, corrected P(no mention | followed))
    A calibration stratum of 50 positives and 140 negatives, Beta posteriors on
    judge sensitivity and specificity, a Rogan-Gladen correction of an
    IPW-weighted judged rate in a follow stratum of size 20, 50 or 100, and the
    middle-case comparator: the interval implied for the frozen regex by its
    precision and recall estimated on the same frame (each Beta/Clopper-Pearson,
    propagated by Monte Carlo), exactly as the plan's middle-case rule words it.

ASSUMPTIONS, stated because they are choices and not measurements:
  1. Measurement noise on M is additive Gaussian, independent of X, M and Y,
     with sd equal to a fixed FRACTION of the replicate's own sample sd(M).
     20 percent is the headline; 40 percent is the stress row; 0 percent is the
     reference row that reproduces section C's setting.
  2. The estimator does NOT model that noise (no measurement-error layer). This
     is the specification as written in docs/methodology.md section 4 and in the
     plan's column-B paragraph, where the noise is "estimated from repeated
     curves" but is not yet a latent-variable layer in the fit.
  3. Beta posteriors use the Jeffreys prior Beta(0.5, 0.5).
  4. The judge's label and the regex flag are conditionally independent given
     the item's true class.
  5. The 2:1 enrichment is on the REGEX flag (the observable the prereg names as
     the sampling label class), not on the unobservable true class.
  6. The regex has assumed recall 0.7 and precision 0.9 AT THE CALIBRATION
     FRAME's prevalence; the false-positive rate implied by that pair is derived
     and held fixed across strata (a rate transports; a precision does not).

Every number this script writes is asserted against an independent calculation
first (see ``assert_consistency``). Nothing is written if an assertion fails.

Run:
    cd ~/Developer/bayes-cot-faithfulness
    PYTHONPATH=src .venv/bin/python notebooks/06_prefreeze_precision.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from bayes_cot_faithfulness.sensitivity import (
    ConfoundedCoTConfig,
    fit_probit_mediation_map,
    probit_natural_effects,
    simulate_confounded_cot,
)

_EXP = Path(__file__).resolve().parent.parent / "experiments"
OUT_JSON = _EXP / "prefreeze_precision_results.json"
OUT_JSON_CONFIRM = _EXP / "prefreeze_precision_confirm.json"

# ---------------------------------------------------------------------------
# Pre-registered constants (no magic numbers below this block).
# ---------------------------------------------------------------------------

FAITHFUL = {"alpha": 0.1, "beta": 2.0, "gamma": 1.0, "sigma_m": 0.5, "rho": 0.0}
WEAK = {"alpha": 0.3, "beta": 0.6, "gamma": 0.8, "sigma_m": 0.5, "rho": 0.3}
WORLDS = {"faithful": FAITHFUL, "weak": WEAK}

NOISE_FRACS = (0.0, 0.2, 0.4)
N_PER_CELL = (150, 300, 600, 3600)
N_REPLICATES_B = 10

# Confirmation tier. The main grid brackets the 0.10 half-width bar between
# n = 300 and n = 600 in the binding (faithful) world at 20 percent noise; these
# n values pin it, so the design consequence is a measured number and not an
# extrapolation. Run alone with --confirm-only.
CONFIRM_N = (350, 400)
CONFIRM_NOISE = (0.2,)
# Follow-stratum sizes that expose the calibration-limited floor of column A:
# the width the corrected estimate cannot go below while the calibration frame
# holds only CAL_POSITIVES positives and CAL_NEGATIVES negatives.
CONFIRM_FOLLOW = (500, 5000)

N_CHAINS = 2
N_TUNE = 1000
N_DRAWS = 1000
TARGET_ACCEPT = 0.9

# Large-sample size used to locate the probability limit of the misspecified
# rho = 0 fit (the estimand the fit actually targets, as opposed to the truth).
N_ASYMPTOTIC = 400_000

# Part A constants.
CAL_POSITIVES = 50
CAL_NEGATIVES = 140
JUDGE_SE_TRUE = 0.85
JUDGE_SP_TRUE = 0.97
JUDGE_SE_DEGRADED = 0.80  # Youden's J = 0.60 exactly, the K1 threshold.
JUDGE_SP_DEGRADED = 0.80
FOLLOW_SIZES = (20, 50, 100)
P_NOMENTION_TRUE = 0.80
ENRICHMENT_RATIO = 2.0  # flagged rows sampled at twice the rate of unflagged.
REGEX_RECALL = 0.70
REGEX_PRECISION_AT_FRAME = 0.90
N_REPLICATES_A = 500
N_POSTERIOR_DRAWS_A = 4000
JEFFREYS = 0.5

MASTER_SEED = 20260906


# ---------------------------------------------------------------------------
# Closed-form natural effects for the probit model at rho = 0.
# ---------------------------------------------------------------------------


def nie_closed_form(alpha, beta, gamma, sigma_m):
    """NIE on the probability scale for the rho = 0 probit SCM.

    E[Y(x, M(x'))] = Phi((alpha*x + beta*gamma*x') / sqrt(1 + beta^2 sigma_m^2)),
    so NIE = Phi((alpha + beta*gamma)/s) - Phi(alpha/s) with s the scale below.
    Vectorised over posterior draws.
    """
    scale = np.sqrt(1.0 + np.square(beta) * np.square(sigma_m))
    return norm.cdf((alpha + beta * gamma) / scale) - norm.cdf(alpha / scale)


def te_closed_form(alpha, beta, gamma, sigma_m):
    scale = np.sqrt(1.0 + np.square(beta) * np.square(sigma_m))
    return norm.cdf((alpha + beta * gamma) / scale) - 0.5


# ---------------------------------------------------------------------------
# Part B: simulation, noise, PyMC probit posterior.
# ---------------------------------------------------------------------------


def simulate_with_noise(world, n, seed, noise_frac):
    """Simulate one cell and return (X, M_observed, Y, sd_of_added_noise)."""
    cfg = ConfoundedCoTConfig(
        n_prompts=n,
        alpha_direct=world["alpha"],
        beta_mediated=world["beta"],
        gamma_xm=world["gamma"],
        sigma_m=world["sigma_m"],
        rho_confound=world["rho"],
        rng_seed=seed,
    )
    X, M, Y = simulate_confounded_cot(cfg)
    if noise_frac == 0.0:
        return X, M, Y, 0.0
    noise_sd = noise_frac * float(np.std(M))
    rng = np.random.default_rng(seed + 10_000_000)
    return X, M + rng.normal(0.0, noise_sd, size=n), Y, noise_sd


def fit_probit_posterior(X, M, Y, seed):
    """Full PyMC probit mediation posterior at rho = 0.

    Returns dict with NIE/TE posterior draws, max r-hat, divergences, seconds.
    """
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    t0 = time.time()
    with pm.Model():
        Xf = pt.as_tensor_variable(X.astype(float))
        Mf = pt.as_tensor_variable(M.astype(float))
        Yf = pt.as_tensor_variable(Y.astype(float))

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=2.0)
        gamma = pm.Normal("gamma", mu=0.0, sigma=1.5)
        sigma_m = pm.HalfNormal("sigma_m", sigma=1.0)

        pm.Normal("M_obs", mu=gamma * Xf, sigma=sigma_m, observed=Mf)

        eta = alpha * Xf + beta * Mf
        std = pm.Normal.dist(0.0, 1.0)
        loglik = pt.switch(pt.eq(Yf, 1.0), pm.logcdf(std, eta), pm.logcdf(std, -eta))
        pm.Potential("Y_probit_ll", loglik.sum())

        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            cores=1,  # sequential chains: one fit at a time, memory light
            target_accept=TARGET_ACCEPT,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    elapsed = time.time() - t0

    post = idata.posterior
    a = post["alpha"].values.ravel()
    b = post["beta"].values.ravel()
    g = post["gamma"].values.ravel()
    s = post["sigma_m"].values.ravel()

    # az.summary rounds r-hat to three decimals, which makes a threshold test on
    # it meaningless; take the unrounded diagnostics directly.
    names = ["alpha", "beta", "gamma", "sigma_m"]
    rhat = az.rhat(idata, var_names=names)
    ess = az.ess(idata, var_names=names)
    n_div = int(idata.sample_stats["diverging"].values.sum())

    return {
        "nie": nie_closed_form(a, b, g, s),
        "te": te_closed_form(a, b, g, s),
        "rhat_max": float(max(float(rhat[v].values) for v in names)),
        "ess_bulk_min": float(min(float(ess[v].values) for v in names)),
        "divergences": n_div,
        "seconds": elapsed,
    }


def asymptotic_rho0_target(world, noise_frac, seed=99):
    """Probability limit of the misspecified rho = 0 MAP fit under this noise.

    This is the estimand the fit actually targets: it absorbs both the
    confounding bias (fitting at rho = 0 data generated at the true rho) and the
    attenuation from unmodelled measurement error. Located by a MAP fit on
    N_ASYMPTOTIC rows, which is cheap (L-BFGS) and deterministic.
    """
    X, M, Y, _ = simulate_with_noise(world, N_ASYMPTOTIC, seed, noise_frac)
    a, b, g, s = fit_probit_mediation_map(X, M, Y, rho=0.0)
    return float(nie_closed_form(a, b, g, s)), (a, b, g, s)


def analytic_attenuated_nie(world, noise_frac):
    """Analytic prediction of the rho = 0 target under measurement error.

    Valid for the faithful world only (rho = 0 truth, so the only
    misspecification is the unmodelled noise). With
    lambda = sigma_m^2 / (sigma_m^2 + sigma_u^2) the reliability ratio, the
    regression of Y on (X, M_obs) is a probit with
        beta'  = beta*lambda / c,
        alpha' = (alpha + beta*gamma*(1 - lambda)) / c,
        gamma' = gamma,  sigma_m' = sqrt(sigma_m^2 + sigma_u^2),
        c      = sqrt(1 + beta^2 * sigma_m^2 * (1 - lambda)).
    Used only as an independent check on ``asymptotic_rho0_target``.
    """
    alpha, beta, gamma, sm = world["alpha"], world["beta"], world["gamma"], world["sigma_m"]
    var_m_total = gamma**2 * 0.25 + sm**2  # X ~ Bernoulli(0.5) => Var(gamma X) = gamma^2/4
    sigma_u = noise_frac * np.sqrt(var_m_total)
    lam = sm**2 / (sm**2 + sigma_u**2)
    c = np.sqrt(1.0 + beta**2 * sm**2 * (1.0 - lam))
    beta_p = beta * lam / c
    alpha_p = (alpha + beta * gamma * (1.0 - lam)) / c
    sigma_m_p = np.sqrt(sm**2 + sigma_u**2)
    return float(nie_closed_form(alpha_p, beta_p, gamma, sigma_m_p))


# ---------------------------------------------------------------------------
# Part A: Rogan-Gladen with IPW and the regex comparator.
# ---------------------------------------------------------------------------

# Regex false-positive rate implied by (recall 0.7, precision 0.9) at the frame's
# own composition of CAL_POSITIVES positives and CAL_NEGATIVES negatives:
#   precision = TP / (TP + FP), TP = recall * CAL_POSITIVES
#   => FP = TP * (1/precision - 1), fp_rate = FP / CAL_NEGATIVES
_TP_FRAME = REGEX_RECALL * CAL_POSITIVES
REGEX_FP_RATE = _TP_FRAME * (1.0 / REGEX_PRECISION_AT_FRAME - 1.0) / CAL_NEGATIVES


def rogan_gladen(observed_rate, se, sp):
    """Prevalence-independent misclassification correction, clipped to [0, 1]."""
    denom = se + sp - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (observed_rate - (1.0 - sp)) / denom
    out = np.where(denom <= 0.0, np.nan, out)
    return np.clip(out, 0.0, 1.0)


def beta_draws(successes, failures, rng, size):
    return rng.beta(successes + JEFFREYS, failures + JEFFREYS, size=size)


def hajek(weights, indicator):
    """Weighted (Hajek) mean and the design effective sample size."""
    sw = weights.sum()
    q = float((weights * indicator).sum() / sw)
    n_eff = float(sw**2 / np.square(weights).sum())
    return q, n_eff


def simulate_follow_stratum(n_f, rng):
    """Draw one IPW follow stratum under 2:1 enrichment on the regex flag.

    Population: item true class T ~ Bernoulli(P_NOMENTION_TRUE); regex flag
    F | T with sensitivity REGEX_RECALL and false-positive rate REGEX_FP_RATE;
    judge label J | T with JUDGE_SE_TRUE / JUDGE_SP_TRUE (passed in). Flagged
    items enter the stratum at ENRICHMENT_RATIO times the rate of unflagged
    ones, so w = 1/pi is 1/ENRICHMENT_RATIO for flagged, 1 for unflagged.

    Returns (weights, judge_labels, regex_flags).
    """
    p_flag = P_NOMENTION_TRUE * REGEX_RECALL + (1.0 - P_NOMENTION_TRUE) * REGEX_FP_RATE
    p_flag_in_sample = (ENRICHMENT_RATIO * p_flag) / (ENRICHMENT_RATIO * p_flag + (1.0 - p_flag))

    flags = rng.binomial(1, p_flag_in_sample, size=n_f)
    # True class given the flag, by Bayes on the population (the enrichment does
    # not change P(T | F), only the mix of F).
    p_true_given_flag = P_NOMENTION_TRUE * REGEX_RECALL / p_flag
    p_true_given_noflag = (
        P_NOMENTION_TRUE * (1.0 - REGEX_RECALL) / (1.0 - p_flag)
    )
    p_t = np.where(flags == 1, p_true_given_flag, p_true_given_noflag)
    truth = rng.binomial(1, p_t)
    weights = np.where(flags == 1, 1.0 / ENRICHMENT_RATIO, 1.0)
    return weights, truth, flags


def judge_labels(truth, rng, se, sp):
    return rng.binomial(1, np.where(truth == 1, se, 1.0 - sp))


def part_a_replicate(n_f, rng, se_true, sp_true):
    """One replicate: corrected estimate, regex-implied estimate, both intervals."""
    # 1. Calibration stratum -> Beta posteriors on judge sensitivity/specificity.
    tp = rng.binomial(CAL_POSITIVES, se_true)
    tn = rng.binomial(CAL_NEGATIVES, sp_true)
    se = beta_draws(tp, CAL_POSITIVES - tp, rng, N_POSTERIOR_DRAWS_A)
    sp = beta_draws(tn, CAL_NEGATIVES - tn, rng, N_POSTERIOR_DRAWS_A)

    # 1b. Same frame gives the regex's recall and false-positive count.
    x_recall = rng.binomial(CAL_POSITIVES, REGEX_RECALL)
    x_fp = rng.binomial(CAL_NEGATIVES, REGEX_FP_RATE)

    # 2. Follow stratum with IPW weights.
    weights, truth, flags = simulate_follow_stratum(n_f, rng)
    judged = judge_labels(truth, rng, se_true, sp_true)
    q_w, n_eff = hajek(weights, judged)
    r_w, _ = hajek(weights, flags)

    # 3. Corrected estimate: Beta posterior on the weighted judged rate (Jeffreys
    #    on the effective counts), then Rogan-Gladen through the judge posteriors.
    q = beta_draws(n_eff * q_w, n_eff * (1.0 - q_w), rng, N_POSTERIOR_DRAWS_A)
    corrected = rogan_gladen(q, se, sp)
    corr_lo, corr_hi = np.nanpercentile(corrected, [2.5, 97.5])

    # 4. Regex comparator, exactly as the middle-case rule words it: precision and
    #    recall estimated on the SAME frame, propagated by Monte Carlo.
    #    p_implied = regex_rate * precision / recall.
    recall = beta_draws(x_recall, CAL_POSITIVES - x_recall, rng, N_POSTERIOR_DRAWS_A)
    precision = beta_draws(x_recall, x_fp, rng, N_POSTERIOR_DRAWS_A)
    r = beta_draws(n_eff * r_w, n_eff * (1.0 - r_w), rng, N_POSTERIOR_DRAWS_A)
    regex_pr = np.clip(r * precision / np.maximum(recall, 1e-9), 0.0, 1.0)
    pr_lo, pr_hi = np.percentile(regex_pr, [2.5, 97.5])

    # 5. The transportable alternative: Rogan-Gladen on the regex itself, using
    #    recall as sensitivity and (1 - fp rate) as specificity.
    regex_sp = beta_draws(CAL_NEGATIVES - x_fp, x_fp, rng, N_POSTERIOR_DRAWS_A)
    regex_rg = rogan_gladen(r, recall, regex_sp)
    rg_lo, rg_hi = np.nanpercentile(regex_rg, [2.5, 97.5])

    return {
        "corr_width": float(corr_hi - corr_lo),
        "corr_cover": bool(corr_lo <= P_NOMENTION_TRUE <= corr_hi),
        "corr_mean": float(np.nanmean(corrected)),
        "pr_width": float(pr_hi - pr_lo),
        "pr_cover": bool(pr_lo <= P_NOMENTION_TRUE <= pr_hi),
        "pr_mean": float(np.mean(regex_pr)),
        "rg_width": float(rg_hi - rg_lo),
        "rg_cover": bool(rg_lo <= P_NOMENTION_TRUE <= rg_hi),
        "rg_mean": float(np.nanmean(regex_rg)),
        "middle_case_reports_corrected": bool((corr_hi - corr_lo) < (pr_hi - pr_lo)),
        "n_eff": n_eff,
        "j_nonpositive_frac": float(np.mean(se + sp - 1.0 <= 0.0)),
    }


def run_part_a(follow_sizes=FOLLOW_SIZES):
    results = []
    for judge_idx, (label, se_t, sp_t) in enumerate((
        ("judge_true_J0.82", JUDGE_SE_TRUE, JUDGE_SP_TRUE),
        ("judge_at_K1_floor_J0.60", JUDGE_SE_DEGRADED, JUDGE_SP_DEGRADED),
    )):
        for n_f in follow_sizes:
            rng = np.random.default_rng(MASTER_SEED + 7 * n_f + 101 * judge_idx)
            reps = [part_a_replicate(n_f, rng, se_t, sp_t) for _ in range(N_REPLICATES_A)]
            agg = {
                "judge": label,
                "judge_se_true": se_t,
                "judge_sp_true": sp_t,
                "n_follow": n_f,
                "n_replicates": len(reps),
                "corrected_width_mean": float(np.mean([r["corr_width"] for r in reps])),
                "corrected_coverage": float(np.mean([r["corr_cover"] for r in reps])),
                "corrected_mean": float(np.mean([r["corr_mean"] for r in reps])),
                "regex_pr_width_mean": float(np.mean([r["pr_width"] for r in reps])),
                "regex_pr_coverage": float(np.mean([r["pr_cover"] for r in reps])),
                "regex_pr_mean": float(np.mean([r["pr_mean"] for r in reps])),
                "regex_rg_width_mean": float(np.mean([r["rg_width"] for r in reps])),
                "regex_rg_coverage": float(np.mean([r["rg_cover"] for r in reps])),
                "regex_rg_mean": float(np.mean([r["rg_mean"] for r in reps])),
                "middle_case_reports_corrected_frac": float(
                    np.mean([r["middle_case_reports_corrected"] for r in reps])
                ),
                "n_eff_mean": float(np.mean([r["n_eff"] for r in reps])),
                "j_nonpositive_frac_mean": float(np.mean([r["j_nonpositive_frac"] for r in reps])),
            }
            results.append(agg)
            print(
                f"[A] {label} n_follow={n_f} denom={agg['n_replicates']} "
                f"corrected width={agg['corrected_width_mean']:.3f} cov={agg['corrected_coverage']:.3f} "
                f"| regex(P/R) width={agg['regex_pr_width_mean']:.3f} cov={agg['regex_pr_coverage']:.3f} "
                f"| regex(RG) width={agg['regex_rg_width_mean']:.3f} cov={agg['regex_rg_coverage']:.3f} "
                f"| middle-case reports corrected {agg['middle_case_reports_corrected_frac']:.3f}",
                flush=True,
            )
    return results


def run_part_b(n_values=N_PER_CELL, noise_fracs=NOISE_FRACS):
    truths = {}
    for wname, w in WORLDS.items():
        _, nie_true, te_true = probit_natural_effects(
            w["alpha"], w["beta"], w["gamma"], w["sigma_m"], w["rho"], n_mc=2_000_000, rng_seed=0
        )
        truths[wname] = {"nie_true": float(nie_true), "te_true": float(te_true)}
        for frac in noise_fracs:
            target, params = asymptotic_rho0_target(w, frac)
            truths[wname][f"rho0_target_noise{frac}"] = target
            truths[wname][f"rho0_params_noise{frac}"] = [float(p) for p in params]
        shown = {k: round(v, 5) for k, v in truths[wname].items() if "params" not in k}
        print(f"[B] world={wname} truths={shown}", flush=True)

    rows = []
    for wname, w in WORLDS.items():
        for frac in noise_fracs:
            target = truths[wname][f"rho0_target_noise{frac}"]
            for n in n_values:
                hw, means, cov_truth, cov_target, rhats, divs, secs, noise_sds = (
                    [], [], [], [], [], [], [], []
                )
                esss = []
                for rep in range(N_REPLICATES_B):
                    seed = MASTER_SEED + 1000 * rep + n + int(frac * 100) + (0 if wname == "faithful" else 500_000)
                    X, M, Y, nsd = simulate_with_noise(w, n, seed, frac)
                    fit = fit_probit_posterior(X, M, Y, seed)
                    lo, hi = np.percentile(fit["nie"], [2.5, 97.5])
                    hw.append((hi - lo) / 2.0)
                    means.append(float(np.mean(fit["nie"])))
                    cov_truth.append(bool(lo <= truths[wname]["nie_true"] <= hi))
                    cov_target.append(bool(lo <= target <= hi))
                    rhats.append(fit["rhat_max"])
                    divs.append(fit["divergences"])
                    secs.append(fit["seconds"])
                    esss.append(fit["ess_bulk_min"])
                    noise_sds.append(nsd)
                row = {
                    "world": wname,
                    "noise_frac": frac,
                    "n": n,
                    "n_replicates": N_REPLICATES_B,
                    "nie_true": truths[wname]["nie_true"],
                    "nie_rho0_target": target,
                    "nie_post_mean": float(np.mean(means)),
                    "nie_across_rep_sd": float(np.std(means, ddof=1)),
                    "halfwidth_mean": float(np.mean(hw)),
                    "halfwidth_max": float(np.max(hw)),
                    "coverage_truth": float(np.mean(cov_truth)),
                    "coverage_rho0_target": float(np.mean(cov_target)),
                    "rhat_max": float(np.max(rhats)),
                    "n_fits_rhat_above_1.01": int(np.sum(np.array(rhats) >= 1.01)),
                    "ess_bulk_min": float(np.min(esss)),
                    "divergences_total": int(np.sum(divs)),
                    "seconds_total": float(np.sum(secs)),
                    "noise_sd_mean": float(np.mean(noise_sds)),
                }
                rows.append(row)
                print(
                    f"[B] {wname} noise={frac:.1f} n={n} denom={N_REPLICATES_B} "
                    f"half-width={row['halfwidth_mean']:.4f} mean NIE={row['nie_post_mean']:.4f} "
                    f"(true {row['nie_true']:.4f}, rho0 target {target:.4f}) "
                    f"cov_truth={row['coverage_truth']:.2f} cov_target={row['coverage_rho0_target']:.2f} "
                    f"rhat={row['rhat_max']:.4f} div={row['divergences_total']} "
                    f"[{row['seconds_total']:.0f}s]",
                    flush=True,
                )
    return truths, rows


# ---------------------------------------------------------------------------
# Self-consistency assertions. Nothing is written unless all of these pass.
# ---------------------------------------------------------------------------


def assert_consistency(truths, rows_b, rows_a):
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError(f"CONSISTENCY FAILURE [{name}]: {detail}")

    # C1: the closed-form NIE agrees with the module's Monte Carlo natural effects
    #     at rho = 0 for both worlds' true parameters.
    for wname, w in WORLDS.items():
        mc = probit_natural_effects(
            w["alpha"], w["beta"], w["gamma"], w["sigma_m"], 0.0, n_mc=2_000_000, rng_seed=11
        )[1]
        cf = float(nie_closed_form(w["alpha"], w["beta"], w["gamma"], w["sigma_m"]))
        check(f"C1 closed-form NIE == module MC ({wname})", abs(mc - cf) < 3e-3, f"MC {mc:.6f} vs closed form {cf:.6f}")

    # C2: the analytic attenuation prediction reproduces the large-n rho=0 target
    #     in the faithful world (where noise is the only misspecification).
    for frac in sorted({r["noise_frac"] for r in rows_b}):
        pred = analytic_attenuated_nie(FAITHFUL, frac)
        got = truths["faithful"][f"rho0_target_noise{frac}"]
        check(
            f"C2 attenuation analytic == large-n MAP (noise {frac})",
            abs(pred - got) < 0.01,
            f"analytic {pred:.5f} vs n={N_ASYMPTOTIC} MAP {got:.5f}",
        )

    # C3: Rogan-Gladen exactly inverts the forward misclassification map.
    q_fwd = P_NOMENTION_TRUE * JUDGE_SE_TRUE + (1 - P_NOMENTION_TRUE) * (1 - JUDGE_SP_TRUE)
    back = float(rogan_gladen(np.array([q_fwd]), np.array([JUDGE_SE_TRUE]), np.array([JUDGE_SP_TRUE]))[0])
    check("C3 Rogan-Gladen inverts the forward map", abs(back - P_NOMENTION_TRUE) < 1e-12,
          f"forward {q_fwd:.9f} -> back {back:.12f} vs truth {P_NOMENTION_TRUE}")

    # C4: the derived regex false-positive rate reproduces precision 0.9 on the frame.
    prec = _TP_FRAME / (_TP_FRAME + REGEX_FP_RATE * CAL_NEGATIVES)
    check("C4 derived regex fp rate reproduces precision 0.90", abs(prec - REGEX_PRECISION_AT_FRAME) < 1e-12,
          f"fp_rate {REGEX_FP_RATE:.8f} -> precision {prec:.10f}")

    # C5: the Hajek IPW estimator is design-unbiased for the population judged rate.
    rng = np.random.default_rng(4242)
    w_big, t_big, f_big = simulate_follow_stratum(400_000, rng)
    j_big = judge_labels(t_big, rng, JUDGE_SE_TRUE, JUDGE_SP_TRUE)
    q_hat, _ = hajek(w_big, j_big)
    q_expect = P_NOMENTION_TRUE * JUDGE_SE_TRUE + (1 - P_NOMENTION_TRUE) * (1 - JUDGE_SP_TRUE)
    check("C5 Hajek IPW is design-unbiased for the judged rate", abs(q_hat - q_expect) < 5e-3,
          f"weighted {q_hat:.5f} vs analytic {q_expect:.5f} on 400,000 design draws")
    r_hat_big, _ = hajek(w_big, f_big)
    r_expect = P_NOMENTION_TRUE * REGEX_RECALL + (1 - P_NOMENTION_TRUE) * REGEX_FP_RATE
    check("C5b Hajek IPW is design-unbiased for the regex rate", abs(r_hat_big - r_expect) < 5e-3,
          f"weighted {r_hat_big:.5f} vs analytic {r_expect:.5f}")

    # C6: the added measurement noise really has the requested sd.
    for frac in sorted({r["noise_frac"] for r in rows_b}):
        if frac == 0.0:
            continue
        _, M_noisy, _, nsd = simulate_with_noise(FAITHFUL, 200_000, 777, frac)
        _, M0, _, _ = simulate_with_noise(FAITHFUL, 200_000, 777, 0.0)
        emp = float(np.std(M_noisy - M0))
        check(f"C6 realised noise sd == {frac} x sd(M)", abs(emp - nsd) / nsd < 0.02,
              f"requested {nsd:.5f}, realised {emp:.5f}, sd(M)={np.std(M0):.5f}")

    # C7: every denominator printed is the declared replicate count.
    check("C7 part B denominators", all(r["n_replicates"] == N_REPLICATES_B for r in rows_b),
          f"all {len(rows_b)} part-B rows carry n_replicates = {N_REPLICATES_B}")
    check("C7b part A denominators", all(r["n_replicates"] == N_REPLICATES_A for r in rows_a),
          f"all {len(rows_a)} part-A rows carry n_replicates = {N_REPLICATES_A}")
    check("C7c part A widths shrink with the follow stratum",
          all(
              [r["corrected_width_mean"] for r in sorted(
                  [x for x in rows_a if x["judge"] == j], key=lambda x: x["n_follow"])][i]
              >= [r["corrected_width_mean"] for r in sorted(
                  [x for x in rows_a if x["judge"] == j], key=lambda x: x["n_follow"])][i + 1] - 0.005
              for j in {x["judge"] for x in rows_a}
              for i in range(len([x for x in rows_a if x["judge"] == j]) - 1)
          ),
          "corrected interval width is non-increasing in the follow-stratum size for every judge")

    # C8: sampler health.
    worst = max(r["rhat_max"] for r in rows_b)
    n_above = sum(r["n_fits_rhat_above_1.01"] for r in rows_b)
    ess_worst = min(r["ess_bulk_min"] for r in rows_b)
    check(
        "C8 sampler health",
        worst < 1.05 and ess_worst > 400,
        f"max r-hat over all {len(rows_b) * N_REPLICATES_B} fits = {worst:.4f} "
        f"({n_above} fits at or above the 1.01 convergence bar); min bulk ESS = {ess_worst:.0f}; "
        f"total divergences = {sum(r['divergences_total'] for r in rows_b)}",
    )

    # C9: half-width falls with n (monotone within every world x noise block).
    for wname in WORLDS:
        for frac in sorted({r["noise_frac"] for r in rows_b}):
            block = [r for r in rows_b if r["world"] == wname and r["noise_frac"] == frac]
            block.sort(key=lambda r: r["n"])
            hws = [r["halfwidth_mean"] for r in block]
            if len(hws) < 2:
                continue
            # 0.005 slack: the half-width is a Monte Carlo mean over 10 replicates.
            check(f"C9 half-width falls with n ({wname}, noise {frac})",
                  all(hws[i] >= hws[i + 1] - 0.005 for i in range(len(hws) - 1)),
                  f"half-widths {[round(h, 4) for h in hws]} at n {[r['n'] for r in block]}")

    return checks


def main():
    confirm_only = "--confirm-only" in sys.argv
    print("=" * 78)
    print("PRE-FREEZE PRECISION RE-RUN (01-SIZING section H -> section I)")
    print(f"Part B: {len(WORLDS)} worlds x {len(NOISE_FRACS)} noise levels x {len(N_PER_CELL)} n "
          f"x {N_REPLICATES_B} replicates = "
          f"{len(WORLDS)*len(NOISE_FRACS)*len(N_PER_CELL)*N_REPLICATES_B} PyMC fits")
    print(f"Part A: 2 judges x {len(FOLLOW_SIZES)} follow sizes x {N_REPLICATES_A} replicates")
    print(f"Derived regex false-positive rate on negatives: {REGEX_FP_RATE:.6f}")
    print("=" * 78, flush=True)

    if confirm_only:
        print("CONFIRMATION TIER ONLY", flush=True)
        rows_a = run_part_a(follow_sizes=CONFIRM_FOLLOW)
        truths, rows_b = run_part_b(n_values=CONFIRM_N, noise_fracs=CONFIRM_NOISE)
    else:
        rows_a = run_part_a(follow_sizes=FOLLOW_SIZES) + run_part_a(follow_sizes=CONFIRM_FOLLOW)
        truths, rows_b = run_part_b()
        truths_c, rows_bc = run_part_b(n_values=CONFIRM_N, noise_fracs=CONFIRM_NOISE)
        # The rho = 0 targets are deterministic in the seed, so the confirmation
        # tier must reproduce the main grid's targets exactly. Checked, not assumed.
        for wname in WORLDS:
            for frac in CONFIRM_NOISE:
                key = f"rho0_target_noise{frac}"
                if abs(truths[wname][key] - truths_c[wname][key]) > 1e-12:
                    raise AssertionError(
                        f"CONSISTENCY FAILURE [C10 confirm tier reproduces the rho0 target]: "
                        f"{wname} {key}: main {truths[wname][key]} vs confirm {truths_c[wname][key]}"
                    )
        rows_b = rows_b + rows_bc

    checks = assert_consistency(truths, rows_b, rows_a)
    print("\nCONSISTENCY CHECKS (all must pass before anything is written):")
    for c in checks:
        print(f"  PASS  {c['check']}: {c['detail']}")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "constants": {
            "worlds": WORLDS,
            "noise_fracs": list(NOISE_FRACS),
            "n_per_cell": list(N_PER_CELL),
            "n_replicates_B": N_REPLICATES_B,
            "sampler": {"chains": N_CHAINS, "tune": N_TUNE, "draws": N_DRAWS,
                        "target_accept": TARGET_ACCEPT, "link": "probit", "rho": 0.0},
            "cal_positives": CAL_POSITIVES,
            "cal_negatives": CAL_NEGATIVES,
            "judge_se_true": JUDGE_SE_TRUE,
            "judge_sp_true": JUDGE_SP_TRUE,
            "follow_sizes": list(FOLLOW_SIZES),
            "confirm_n": list(CONFIRM_N),
            "confirm_noise": list(CONFIRM_NOISE),
            "confirm_follow": list(CONFIRM_FOLLOW),
            "p_nomention_true": P_NOMENTION_TRUE,
            "enrichment_ratio": ENRICHMENT_RATIO,
            "regex_recall": REGEX_RECALL,
            "regex_precision_at_frame": REGEX_PRECISION_AT_FRAME,
            "regex_fp_rate_derived": REGEX_FP_RATE,
            "n_replicates_A": N_REPLICATES_A,
            "prior": "Jeffreys Beta(0.5, 0.5)",
        },
        "truths": truths,
        "part_b": rows_b,
        "part_a": rows_a,
        "consistency_checks": checks,
    }
    out = OUT_JSON_CONFIRM if confirm_only else OUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
