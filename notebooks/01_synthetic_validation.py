"""End-to-end synthetic-CoT validation.

This is the methodological go/no-go gate for the project, and it has two halves.

1. RECOVERY. If the Bayesian mediation estimator cannot recover known
   natural-direct and natural-indirect effects on synthetic data where the
   ground truth is *exact*, it will not recover them on real LLM traces.
2. THE OFFSET NULL. Recovery alone is not enough: an estimator can pass a
   recovery check on data drawn from its own equations and still invent an
   effect on a world with none. The intercept-free model this project shipped
   before 2026-09-07 did exactly that, reporting a mediated effect of about 0.25
   with a credible interval that excluded zero on a mediator that had a baseline
   level of six and no dependence on the treatment at all. The second half of
   this gate runs that null and requires a near-zero answer.

Run with:
    python notebooks/01_synthetic_validation.py

Expected wall-clock: ~90s on a modern laptop CPU. No API calls. No GPU.
"""

from __future__ import annotations

import math

from bayes_cot_faithfulness import (
    SyntheticCoTConfig,
    fit_mediation_model,
    monte_carlo_true_effects,
    posterior_natural_effects,
    simulate_cot_trace,
)
from bayes_cot_faithfulness.mediation import (
    extract_intercept_samples,
    extract_parameter_samples,
)

# The offset null: a mediator with a real baseline (mu_m = 6) that the treatment
# does not move, and an answer rate of 0.8 that does not depend on anything.
# Every true natural effect is exactly zero.
NULL_CONFIG = SyntheticCoTConfig(
    n_prompts=1500,
    alpha_direct=0.0,
    beta_mediated=0.0,
    gamma_xm=0.0,
    sigma_m=1.0,
    rng_seed=731,
    mu_m=6.0,
    alpha0=math.log(0.8 / 0.2),
)
NULL_TOLERANCE = 0.05


def main() -> int:
    config = SyntheticCoTConfig(
        n_prompts=400,
        alpha_direct=0.3,
        beta_mediated=1.5,
        gamma_xm=0.8,
        sigma_m=0.5,
        rng_seed=42,
    )

    print(f"[1/5] Simulating synthetic CoT traces (n={config.n_prompts}, seed={config.rng_seed})")
    X, M, Y = simulate_cot_trace(config)
    print(f"        X balance: {X.mean():.3f}    Y rate: {Y.mean():.3f}")
    print(f"        E[M|X=1] - E[M|X=0]: {M[X==1].mean() - M[X==0].mean():+.3f}")

    print("\n[2/5] Computing ground-truth natural effects via Monte Carlo")
    true_nde, true_nie, true_te = monte_carlo_true_effects(config, n_mc=400_000, rng_seed=0)
    print(f"        True NDE = {true_nde:+.4f}")
    print(f"        True NIE = {true_nie:+.4f}")
    print(f"        True TE  = {true_te:+.4f}")

    print("\n[3/5] Fitting Bayesian mediation model in PyMC")
    trace = fit_mediation_model(
        X, M, Y,
        n_samples=1500,
        n_tune=1500,
        n_chains=4,
        target_accept=0.95,
        random_seed=0,
        progressbar=True,
        link="logit",
    )
    alpha_s, beta_s, gamma_s, sigma_s = extract_parameter_samples(trace)
    print(f"        alpha  posterior mean = {alpha_s.mean():+.3f}  (true {config.alpha_direct:+.3f})")
    print(f"        beta   posterior mean = {beta_s.mean():+.3f}  (true {config.beta_mediated:+.3f})")
    print(f"        gamma  posterior mean = {gamma_s.mean():+.3f}  (true {config.gamma_xm:+.3f})")
    print(f"        sigma  posterior mean = {sigma_s.mean():+.3f}  (true {config.sigma_m:+.3f})")

    print("\n[4/5] Posterior recovery on the probability scale")
    pe = posterior_natural_effects(
        alpha_s, beta_s, gamma_s, sigma_s, n_mc_per_draw=2_000, rng_seed=0, link="logit"
    )
    coverage = pe.contains(true_nde, true_nie, true_te)
    print(f"        NDE posterior:  {pe.nde_mean:+.3f}  [95% CrI: {pe.nde_lo:+.3f}, {pe.nde_hi:+.3f}]   contains truth: {coverage['nde']}")
    print(f"        NIE posterior:  {pe.nie_mean:+.3f}  [95% CrI: {pe.nie_lo:+.3f}, {pe.nie_hi:+.3f}]   contains truth: {coverage['nie']}")
    print(f"        TE  posterior:  {pe.te_mean:+.3f}  [95% CrI: {pe.te_lo:+.3f}, {pe.te_hi:+.3f}]   contains truth: {coverage['te']}")

    null_ok = _offset_null_gate()

    if all(coverage.values()) and null_ok:
        print("\n        coverage check passed. offset-null check passed.")
        print("        methodological gate cleared. ready to scale to real LLM experiments.")
        return 0
    if not all(coverage.values()):
        print("\n        coverage check FAILED. investigate before scaling.")
    if not null_ok:
        print("\n        offset-null check FAILED: the estimator invents an effect")
        print("        on a world that has none. investigate before scaling.")
    return 1


def _offset_null_gate() -> bool:
    """Second half of the gate: a world with no causal effect must read as one.

    Fits the same estimator on ``NULL_CONFIG``, where the mediator has a baseline
    of six and no treatment dependence and the answer is an independent coin, so
    the true NDE, NIE and TE are all exactly zero. Passing requires the recovered
    effects to be within ``NULL_TOLERANCE`` of zero and the mediated path's
    credible interval to contain zero.

    The same fit is repeated with ``intercepts=False`` to show what the
    pre-2026-09-07 specification did on this data. That run is reported, never
    gated: it is the defect, kept visible.
    """
    print("\n[5/5] Offset null: a world with no causal effect at all")
    X, M, Y = simulate_cot_trace(NULL_CONFIG)
    true_nde, true_nie, true_te = monte_carlo_true_effects(
        NULL_CONFIG, n_mc=200_000, rng_seed=0
    )
    arm_difference = float(Y[X == 1].mean() - Y[X == 0].mean())
    print(f"        E[M|X=0] = {M[X == 0].mean():.3f}   Y rate: {Y.mean():.3f}")
    print(f"        true NDE / NIE / TE = {true_nde:+.4f} / {true_nie:+.4f} / {true_te:+.4f}")
    print(f"        observed arm difference in Y: {arm_difference:+.4f}")

    trace = fit_mediation_model(
        X, M, Y, n_samples=600, n_tune=600, n_chains=2, random_seed=0, progressbar=False,
        link="logit",
    )
    alpha_s, beta_s, gamma_s, sigma_s = extract_parameter_samples(trace)
    mu_m_s, alpha0_s = extract_intercept_samples(trace)
    print(f"        mu_m   posterior mean = {mu_m_s.mean():+.3f}  (true {NULL_CONFIG.mu_m:+.3f})")
    print(f"        gamma  posterior mean = {gamma_s.mean():+.3f}  (true {NULL_CONFIG.gamma_xm:+.3f})")
    pe = posterior_natural_effects(
        alpha_s, beta_s, gamma_s, sigma_s, n_mc_per_draw=1_000, rng_seed=0,
        mu_m_samples=mu_m_s, alpha0_samples=alpha0_s, link="logit",
    )
    print(f"        NIE posterior:  {pe.nie_mean:+.4f}  [95% CrI: {pe.nie_lo:+.4f}, {pe.nie_hi:+.4f}]")
    print(f"        TE  posterior:  {pe.te_mean:+.4f}  [95% CrI: {pe.te_lo:+.4f}, {pe.te_hi:+.4f}]")

    legacy = fit_mediation_model(
        X, M, Y, n_samples=600, n_tune=600, n_chains=2, random_seed=0,
        progressbar=False, intercepts=False, link="logit",
    )
    la, lb, lg, ls = extract_parameter_samples(legacy)
    lpe = posterior_natural_effects(la, lb, lg, ls, n_mc_per_draw=1_000, rng_seed=0, link="logit")
    print(
        f"        for contrast, the pre-2026-09-07 intercept-free model reports "
        f"NIE {lpe.nie_mean:+.4f} [{lpe.nie_lo:+.4f}, {lpe.nie_hi:+.4f}]"
    )
    print("        on the same data, where the truth is exactly zero.")

    passed = (
        abs(pe.nie_mean) < NULL_TOLERANCE
        and abs(pe.te_mean) < NULL_TOLERANCE
        and pe.nie_lo <= 0.0 <= pe.nie_hi
    )
    print(f"        offset-null recovery within {NULL_TOLERANCE}: {passed}")
    return passed


if __name__ == "__main__":
    raise SystemExit(main())
