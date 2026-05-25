"""End-to-end synthetic-CoT validation.

This is the methodological go/no-go gate for the project. If the Bayesian
mediation estimator cannot recover known natural-direct and natural-indirect
effects on synthetic data where the ground truth is *exact*, it will not
recover them on real LLM traces.

Run with:
    python notebooks/01_synthetic_validation.py

Expected wall-clock: ~60s on a modern laptop CPU. No API calls. No GPU.
"""

from __future__ import annotations

from bayes_cot_faithfulness import (
    SyntheticCoTConfig,
    fit_mediation_model,
    monte_carlo_true_effects,
    posterior_natural_effects,
    simulate_cot_trace,
)
from bayes_cot_faithfulness.mediation import extract_parameter_samples


def main() -> int:
    config = SyntheticCoTConfig(
        n_prompts=400,
        alpha_direct=0.3,
        beta_mediated=1.5,
        gamma_xm=0.8,
        sigma_m=0.5,
        rng_seed=42,
    )

    print(f"[1/4] Simulating synthetic CoT traces (n={config.n_prompts}, seed={config.rng_seed})")
    X, M, Y = simulate_cot_trace(config)
    print(f"        X balance: {X.mean():.3f}    Y rate: {Y.mean():.3f}")
    print(f"        E[M|X=1] - E[M|X=0]: {M[X==1].mean() - M[X==0].mean():+.3f}")

    print("\n[2/4] Computing ground-truth natural effects via Monte Carlo")
    true_nde, true_nie, true_te = monte_carlo_true_effects(config, n_mc=400_000, rng_seed=0)
    print(f"        True NDE = {true_nde:+.4f}")
    print(f"        True NIE = {true_nie:+.4f}")
    print(f"        True TE  = {true_te:+.4f}")

    print("\n[3/4] Fitting Bayesian mediation model in PyMC")
    trace = fit_mediation_model(
        X, M, Y,
        n_samples=1500,
        n_tune=1500,
        n_chains=4,
        target_accept=0.95,
        random_seed=0,
        progressbar=True,
    )
    alpha_s, beta_s, gamma_s, sigma_s = extract_parameter_samples(trace)
    print(f"        alpha  posterior mean = {alpha_s.mean():+.3f}  (true {config.alpha_direct:+.3f})")
    print(f"        beta   posterior mean = {beta_s.mean():+.3f}  (true {config.beta_mediated:+.3f})")
    print(f"        gamma  posterior mean = {gamma_s.mean():+.3f}  (true {config.gamma_xm:+.3f})")
    print(f"        sigma  posterior mean = {sigma_s.mean():+.3f}  (true {config.sigma_m:+.3f})")

    print("\n[4/4] Posterior recovery on the probability scale")
    pe = posterior_natural_effects(alpha_s, beta_s, gamma_s, sigma_s, n_mc_per_draw=2_000, rng_seed=0)
    coverage = pe.contains(true_nde, true_nie, true_te)
    print(f"        NDE posterior:  {pe.nde_mean:+.3f}  [95% CrI: {pe.nde_lo:+.3f}, {pe.nde_hi:+.3f}]   contains truth: {coverage['nde']}")
    print(f"        NIE posterior:  {pe.nie_mean:+.3f}  [95% CrI: {pe.nie_lo:+.3f}, {pe.nie_hi:+.3f}]   contains truth: {coverage['nie']}")
    print(f"        TE  posterior:  {pe.te_mean:+.3f}  [95% CrI: {pe.te_lo:+.3f}, {pe.te_hi:+.3f}]   contains truth: {coverage['te']}")

    if all(coverage.values()):
        print("\n        coverage check passed.")
        print("        methodological gate cleared. ready to scale to real LLM experiments.")
        return 0
    else:
        print("\n        coverage check FAILED. investigate before scaling.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
