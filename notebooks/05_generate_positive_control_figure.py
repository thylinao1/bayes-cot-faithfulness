"""Generate the positive-control figure and self-check the computation.

This is the figure the write-up leads with: proof that the instrument responds
when unfaithfulness is present. It plots the faithful path (NIE) as a function of
the assumed confounding rho for two synthetic worlds with the same total effect
but opposite mechanisms.

- FAITHFUL world (cyan): the answer flows through the chain-of-thought. The
  faithful path sits well above zero and only a large amount of hidden confounding
  (rho* ~ 0.69) overturns it.
- DECORATIVE world (orange): the answer comes from a shortcut that bypasses the
  stated reasoning. The faithful path hugs zero and the smallest confounding
  (rho* ~ 0.05) overturns it. That curve is what a non-result looks like.

Like notebook 04, this doubles as a correctness gate: it asserts the two worlds
are genuinely separated before writing the figure, and exits nonzero otherwise.

Run::

    python notebooks/05_generate_positive_control_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figstyle import PALETTE, glow_line, style_legend, use_house_style

from bayes_cot_faithfulness.positive_control import (
    DECORATIVE_CONFIG,
    DEFAULT_RHO_BAR,
    FAITHFUL_CONFIG,
    audit_case,
    demonstration_cases,
    smoke_test,
)
from bayes_cot_faithfulness.sensitivity import sensitivity_sweep, simulate_confounded_cot

RHO_GRID = np.round(np.arange(-0.9, 0.91, 0.05), 4)
N_MC = 100_000
SWEEP_SEED = 0


def _nie_curve(config) -> np.ndarray:
    sweep = sensitivity_sweep(
        *simulate_confounded_cot(config), rho_grid=RHO_GRID, n_mc=N_MC, rng_seed=SWEEP_SEED
    )
    return np.array([p.nie for p in sweep])


def main() -> None:
    rhos = RHO_GRID
    nie_faithful = _nie_curve(FAITHFUL_CONFIG)
    nie_decorative = _nie_curve(DECORATIVE_CONFIG)

    faithful = smoke_test(FAITHFUL_CONFIG, n_mc=N_MC)
    decorative = smoke_test(DECORATIVE_CONFIG, n_mc=N_MC)

    # --- correctness gate: the two worlds must be genuinely separated --------
    assert faithful["nie_at_zero"] > 0.10, "faithful world should show a clear faithful path"
    assert abs(decorative["nie_at_zero"]) < 0.10, "decorative world's faithful path should be ~0"
    assert faithful["rho_star"] > decorative["rho_star"] + 0.3, "robustness not separated"
    assert faithful["sign_identified"] and not decorative["sign_identified"], (
        "partial-ID bounds should exclude zero only for the faithful world"
    )
    # the case-level auditor must also get all three transcripts right
    cases = [audit_case(c) for c in demonstration_cases()]
    assert all(c["correct"] for c in cases), "case-level auditor mislabelled a transcript"

    # --- figure (dark house style, matches the project site) -----------------
    use_house_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=PALETTE["border"], lw=0.7, alpha=0.55)
    ax.set_axisbelow(True)

    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.18, 0.98)

    ax.axhline(0.0, color=PALETTE["edge"], lw=0.9, zorder=1)
    glow_line(ax, rhos, nie_faithful, PALETTE["cyan"], lw=2.6, zorder=4,
              label="faithful CoT: answer flows through the reasoning")
    glow_line(ax, rhos, nie_decorative, PALETTE["orange"], lw=2.6, zorder=3,
              label="decorative CoT: answer takes a shortcut")

    # the breakdown rho* crossings: where each world's faithful path hits zero.
    for res, color, label_dy in ((faithful, PALETTE["cyan"], 0.07),
                                  (decorative, PALETTE["orange"], -0.085)):
        r = res["rho_star_pos"]
        if r is not None:
            ax.scatter([r], [0.0], s=130, facecolor="none", edgecolor=PALETTE["green"],
                       lw=2.4, zorder=6)
            ax.annotate(rf"$\rho^*\!=\!{res['rho_star']:.2f}$", xy=(r, 0.0),
                        xytext=(r, label_dy), ha="center", fontsize=9.5,
                        color=color, zorder=7)

    ax.axvline(0.0, color=PALETTE["text_2"], lw=1.0, alpha=0.65)
    ax.annotate("assumes\nignorability", xy=(0.0, 0.86), xytext=(0.06, 0.86),
                ha="left", va="center", fontsize=9, color=PALETTE["text_2"])

    ax.set_xlabel(
        r"assumed sensitivity parameter  $\rho = \mathrm{Corr}(\varepsilon_M, \varepsilon_Y)$",
        labelpad=10,
    )
    ax.set_ylabel("estimated faithful path  (NIE, on P(answer) scale)")
    ax.set_title(
        "Positive control: the estimator tells a faithful CoT from a decorative one",
        fontsize=12.5, pad=10,
    )
    # compact partial-ID readout, inside the axes where the curves leave room.
    readout = (
        rf"bounds for $|\rho|\leq{DEFAULT_RHO_BAR:.1f}$:"
        "\n"
        rf"faithful $[{faithful['nie_lower']:+.2f},{faithful['nie_upper']:+.2f}]$ excludes 0"
        "\n"
        rf"decorative $[{decorative['nie_lower']:+.2f},{decorative['nie_upper']:+.2f}]$ does not"
    )
    ax.text(0.975, 0.965, readout, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=PALETTE["text_2"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["panel"],
                      edgecolor=PALETTE["border"], lw=0.8))
    ax.text(0.5, -0.22,
            "green ring = breakdown rho*: the hidden M-Y confounding it takes to overturn the verdict",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=PALETTE["text_3"])
    style_legend(ax, loc="lower left", fontsize=9)

    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "positive_control.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"OK wrote {out_path}")
    print(f"OK faithful  NIE@0 {faithful['nie_at_zero']:+.3f}  rho* {faithful['rho_star']:.3f}  "
          f"bounds [{faithful['nie_lower']:+.3f}, {faithful['nie_upper']:+.3f}]")
    print(f"OK decorative NIE@0 {decorative['nie_at_zero']:+.3f}  rho* {decorative['rho_star']:.3f}  "
          f"bounds [{decorative['nie_lower']:+.3f}, {decorative['nie_upper']:+.3f}]")
    print(f"OK case-level auditor: {sum(c['correct'] for c in cases)}/{len(cases)} transcripts correct")


if __name__ == "__main__":
    main()
