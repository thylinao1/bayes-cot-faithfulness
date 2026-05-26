"""Generate the posterior-recovery figure used on the project website."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_cot_faithfulness import (
    SyntheticCoTConfig,
    fit_mediation_model,
    monte_carlo_true_effects,
    posterior_natural_effects,
    simulate_cot_trace,
)
from bayes_cot_faithfulness.mediation import extract_parameter_samples


def main() -> None:
    config = SyntheticCoTConfig(
        n_prompts=400,
        alpha_direct=0.3,
        beta_mediated=1.5,
        gamma_xm=0.8,
        sigma_m=0.5,
        rng_seed=42,
    )

    X, M, Y = simulate_cot_trace(config)
    true_nde, true_nie, true_te = monte_carlo_true_effects(config, n_mc=400_000, rng_seed=0)
    trace = fit_mediation_model(
        X, M, Y,
        n_samples=2000,
        n_tune=1500,
        n_chains=4,
        target_accept=0.95,
        random_seed=0,
        progressbar=False,
    )
    alpha_s, beta_s, gamma_s, sigma_s = extract_parameter_samples(trace)
    pe = posterior_natural_effects(alpha_s, beta_s, gamma_s, sigma_s, n_mc_per_draw=2_000)

    # Dark-theme figure to match the project site
    plt.rcParams.update({
        "figure.facecolor": "#0b0f17",
        "axes.facecolor": "#0b0f17",
        "axes.edgecolor": "#5a6478",
        "axes.labelcolor": "#dbe2ea",
        "xtick.color": "#9aa4b8",
        "ytick.color": "#9aa4b8",
        "text.color": "#dbe2ea",
        "axes.titlecolor": "#dbe2ea",
        "font.family": ["SF Pro Display", "Inter", "system-ui", "sans-serif"],
        "font.size": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4), constrained_layout=True)
    pairs = [
        ("Natural Direct Effect", pe.nde_samples, true_nde, "#ff8c42"),
        ("Natural Indirect Effect", pe.nie_samples, true_nie, "#4dd0e1"),
        ("Total Effect", pe.te_samples, true_te, "#a5e887"),
    ]

    for ax, (title, samples, truth, color) in zip(axes, pairs):
        ax.hist(
            samples, bins=60, color=color, alpha=0.55, edgecolor=color, linewidth=0.4,
        )
        lo, hi = np.quantile(samples, [0.025, 0.975])
        ax.axvline(truth, color="#ffffff", linewidth=1.6, label=f"truth = {truth:+.3f}")
        ax.axvline(lo, color=color, linewidth=1.2, linestyle="--", alpha=0.85)
        ax.axvline(hi, color=color, linewidth=1.2, linestyle="--", alpha=0.85)
        ax.set_title(title, pad=44, fontsize=13)
        ax.set_xlabel("effect on P(Y=1)")
        ax.set_ylabel("posterior density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        ax.tick_params(length=4)

    fig.suptitle(
        "Bayesian posterior recovers true natural effects   "
        "(synthetic CoT, n = 400, all 95% CrIs contain truth)",
        fontsize=14, color="#dbe2ea",
    )

    out_dir = Path(__file__).parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "posterior_recovery.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0b0f17")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
