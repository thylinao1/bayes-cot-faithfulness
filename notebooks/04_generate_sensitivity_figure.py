"""Generate the sensitivity-analysis figure and self-check the computation.

This script doubles as a correctness gate: it asserts that the sweep is
internally consistent (its rho=0 point equals a direct rho=0 computation), that
the natural indirect effect is monotone in assumed rho, and that the estimator
recovers the truth at the true rho. It exits nonzero if any check fails, so the
figure can never silently encode a broken sweep.

Run::

    python notebooks/04_generate_sensitivity_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
N_MC = 150_000
SWEEP_SEED = 0


def main() -> None:
    cfg = ConfoundedCoTConfig(
        n_prompts=4000, alpha_direct=0.3, beta_mediated=1.0,
        gamma_xm=0.8, sigma_m=0.5, rho_confound=RHO_TRUE, rng_seed=7,
    )
    X, M, Y = simulate_confounded_cot(cfg)

    true_nde, true_nie, _ = probit_natural_effects(
        cfg.alpha_direct, cfg.beta_mediated, cfg.gamma_xm, cfg.sigma_m,
        cfg.rho_confound, n_mc=300_000, rng_seed=1,
    )

    sweep = sensitivity_sweep(X, M, Y, rho_grid=RHO_GRID, n_mc=N_MC, rng_seed=SWEEP_SEED)
    rhos = np.array([p.rho for p in sweep])
    nde = np.array([p.nde for p in sweep])
    nie = np.array([p.nie for p in sweep])
    te = np.array([p.te for p in sweep])

    # --- correctness gate -------------------------------------------------
    # (1) the sweep's rho=0 point reproduces a direct rho=0 computation.
    a0, b0, g0, s0 = fit_probit_mediation_map(X, M, Y, 0.0)
    nde0_direct, nie0_direct, _ = probit_natural_effects(
        a0, b0, g0, s0, 0.0, n_mc=N_MC, rng_seed=SWEEP_SEED
    )
    i0 = int(np.argmin(np.abs(rhos - 0.0)))
    assert abs(nie[i0] - nie0_direct) < 1e-9, (
        f"sweep rho=0 NIE {nie[i0]} != direct {nie0_direct}"
    )

    # (2) the faithful-path estimate falls monotonically as we assume more
    #     positive M-Y confounding: a larger assumed rho explains away more of
    #     the observed mediator-outcome association, shrinking the causal path.
    assert np.all(np.diff(nie) < 1e-6), "NIE should be monotone decreasing in rho"

    # (3) at the true rho the sweep lands far closer to the truth than the
    #     ignorability estimate does -- recovery beats the naive bias by >4x.
    i_true = int(np.argmin(np.abs(rhos - RHO_TRUE)))
    recovery_err = abs(nie[i_true] - true_nie)
    ignorability_bias = nie[i0] - true_nie
    assert recovery_err < 0.25 * abs(ignorability_bias), (
        f"recovery {recovery_err:.4f} not much smaller than bias {ignorability_bias:.4f}"
    )

    # (4) under positive M-Y confounding, assuming ignorability OVERSTATES the
    #     faithful path: the naive estimate sits above the truth.
    assert ignorability_bias > 0.05, "expected upward bias under ignorability"

    band = robustness_interval(sweep, key="nie")

    # --- figure -----------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    ax.axhline(0.0, color="#999", lw=0.8, zorder=1)
    ax.plot(rhos, te, color="#94a3b8", lw=1.6, ls=":", label="Total effect", zorder=2)
    ax.plot(rhos, nde, color="#ef4444", lw=2.2, marker="o", ms=4,
            label="Natural direct effect (decorative path)", zorder=3)
    ax.plot(rhos, nie, color="#2563eb", lw=2.2, marker="o", ms=4,
            label="Natural indirect effect (faithful path)", zorder=4)

    ax.axhline(true_nie, color="#2563eb", lw=1.0, ls="--", alpha=0.5)
    ax.axhline(true_nde, color="#ef4444", lw=1.0, ls="--", alpha=0.5)

    ax.axvline(0.0, color="#111", lw=1.0, alpha=0.7)
    ax.annotate("assumes\nignorability", xy=(0.0, nie[i0]), xytext=(-0.02, nie[i0] - 0.085),
                ha="right", fontsize=9, color="#111")
    ax.scatter([RHO_TRUE], [nie[i_true]], s=120, facecolor="none",
               edgecolor="#16a34a", lw=2.2, zorder=6)
    ax.annotate("true confounding\nrecovers truth", xy=(RHO_TRUE, nie[i_true]),
                xytext=(RHO_TRUE + 0.02, nie[i_true] + 0.05),
                fontsize=9, color="#16a34a")

    ax.set_xlabel(r"assumed sensitivity parameter  $\rho = \mathrm{Corr}(\varepsilon_M, \varepsilon_Y)$", labelpad=10)
    ax.set_ylabel("effect on P(answer) scale")
    ax.set_title(
        "Sensitivity of CoT faithfulness to the sequential-ignorability assumption",
        fontsize=12.5, pad=10,
    )
    subtitle = "dashed lines: ground truth   |   black line: the assumption every prior causal-CoT estimate makes"
    if band is not None:
        subtitle += f"\nindirect (faithful) path stays positive for rho in [{band[0]:+.1f}, {band[1]:+.1f}]"
    ax.text(0.5, -0.24, subtitle, transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sensitivity_curve.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"OK wrote {out_path}")
    print(f"OK all correctness checks passed (recovery err {recovery_err:.4f}, "
          f"ignorability overstates NIE by {ignorability_bias:+.4f})")


if __name__ == "__main__":
    main()
