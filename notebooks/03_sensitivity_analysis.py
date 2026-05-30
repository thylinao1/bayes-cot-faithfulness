"""Sensitivity analysis for the sequential-ignorability assumption (A3).

Run end-to-end on a laptop CPU in a few seconds, no API calls, no GPU::

    python notebooks/03_sensitivity_analysis.py

The story this script tells:

1. We simulate CoT traces where the mediator (CoT) and the answer share an
   unmeasured confounder. The true residual correlation is rho = 0.5, so
   sequential ignorability is violated by design.
2. An analyst who assumes ignorability (rho = 0) reads off a biased natural
   indirect effect. The bias does not go away with more data.
3. The sensitivity sweep traces the natural effects across a grid of assumed
   rho. At the true rho the estimator recovers the truth, and the curve tells
   us the range of confounding over which the faithfulness verdict survives.
"""

from __future__ import annotations

import numpy as np

from bayes_cot_faithfulness.sensitivity import (
    ConfoundedCoTConfig,
    fit_probit_mediation_map,
    probit_natural_effects,
    robustness_interval,
    sensitivity_sweep,
    simulate_confounded_cot,
)

RHO_TRUE = 0.5
RHO_GRID = np.round(np.arange(-0.6, 0.81, 0.1), 4)


def main() -> None:
    cfg = ConfoundedCoTConfig(
        n_prompts=4000,
        alpha_direct=0.3,
        beta_mediated=1.0,
        gamma_xm=0.8,
        sigma_m=0.5,
        rho_confound=RHO_TRUE,
        rng_seed=7,
    )

    print(f"[1/4] Simulating confounded CoT traces (n={cfg.n_prompts}, true rho={RHO_TRUE})")
    X, M, Y = simulate_confounded_cot(cfg)
    print(f"        X balance: {X.mean():.3f}    Y rate: {Y.mean():.3f}")

    print("\n[2/4] Ground-truth natural effects (true params, true rho)")
    true_nde, true_nie, true_te = probit_natural_effects(
        cfg.alpha_direct, cfg.beta_mediated, cfg.gamma_xm, cfg.sigma_m,
        cfg.rho_confound, n_mc=300_000, rng_seed=1,
    )
    print(f"        True NDE = {true_nde:+.4f}")
    print(f"        True NIE = {true_nie:+.4f}   (the faithful path)")
    print(f"        True TE  = {true_te:+.4f}")

    print("\n[3/4] What an analyst sees if they assume sequential ignorability (rho=0)")
    a0, b0, g0, s0 = fit_probit_mediation_map(X, M, Y, 0.0)
    nde0, nie0, _ = probit_natural_effects(a0, b0, g0, s0, 0.0, n_mc=300_000, rng_seed=2)
    print(f"        NIE assuming rho=0:  {nie0:+.4f}   (true {true_nie:+.4f})")
    print(f"        bias from ignoring confounding: {nie0 - true_nie:+.4f}")

    print("\n[4/4] Sensitivity sweep across assumed rho")
    sweep = sensitivity_sweep(X, M, Y, rho_grid=RHO_GRID, n_mc=120_000)
    print("        rho     NDE       NIE       TE")
    for p in sweep:
        marker = "  <- true rho" if abs(p.rho - RHO_TRUE) < 1e-6 else ""
        print(f"        {p.rho:+.2f}   {p.nde:+.4f}   {p.nie:+.4f}   {p.te:+.4f}{marker}")

    band = robustness_interval(sweep, key="nie")
    if band is not None:
        lo, hi = band
        print(
            f"\n        Robustness: the indirect (faithful) path stays positive "
            f"for all assumed rho in [{lo:+.2f}, {hi:+.2f}]."
        )

    nearest = min(sweep, key=lambda p: abs(p.rho - RHO_TRUE))
    print(
        f"        At assumed rho={nearest.rho:+.2f} the NIE estimate is "
        f"{nearest.nie:+.4f} vs true {true_nie:+.4f} "
        f"(recovery error {abs(nearest.nie - true_nie):.4f})."
    )
    print("\n        Takeaway: under positive M-Y confounding, assuming ignorability")
    print("        overstates the faithful path; correcting for it shrinks the estimate.")
    print("        The sensitivity curve makes the hidden assumption auditable.")


if __name__ == "__main__":
    main()
