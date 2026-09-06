"""The label-versus-trace frontier figure (01-SIZING section J.2).

Split out of ``07_contrast_precision.py``. The generator asserts every claim the
figure makes BEFORE it writes anything, and raises rather than drawing a figure
whose caption would not be true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402
from contrast_sizing import FRONTIER_REF_LABELS, FRONTIER_REF_TRACES  # noqa: E402

OUT_FIG = Path(__file__).resolve().parent.parent / "figures" / "label_vs_trace_frontier.png"


def assert_and_write_frontier(frontier, path=OUT_FIG):
    """Assert the label-side floor exists, then draw the figure. Asserts first.

    The claim being tested is not "the curve flattens". It is that the asymptote
    the trace axis approaches is a function of the LABEL budget alone, so buying
    more traces cannot cross it and buying more labels moves it.
    """
    ts = frontier["trace_sweep"]
    ls = frontier["label_sweep"]
    tr_w = np.array([p["contrast_hw"] for p in ts])
    tr_se = np.array([p["contrast_hw_se"] for p in ts])
    tr_x = np.array([p["n_traces"] for p in ts])
    lb_w = np.array([p["contrast_hw"] for p in ls])
    lb_f = np.array([p["contrast_floor"] for p in ls])
    lb_x = np.array([p["n_labels"] for p in ls])
    ref = [p for p in ls if p["n_labels"] == FRONTIER_REF_LABELS][0]
    floor = ref["contrast_floor"]
    n_rep = ts[0]["n_replicates"]

    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    first_decade = tr_w[0] - tr_w[2]     # 300 -> 3,000 traces.
    last_decade = tr_w[-3] - tr_w[-1]    # 10,000 -> 50,000 traces.
    step_tol = 3.0 * np.sqrt(tr_se[:-1] ** 2 + tr_se[1:] ** 2)

    chk("F1 more traces do narrow the interval, at first",
        first_decade > 0.05,
        f"width {tr_w[0]:.4f} at {tr_x[0]} traces -> {tr_w[2]:.4f} at {tr_x[2]}, a fall "
        f"of {first_decade:.4f} (denominator {n_rep} replicates per point)")
    chk("F2 the trace curve is monotone within Monte Carlo error",
        bool(np.all(np.diff(tr_w) < step_tol)),
        f"successive changes {np.round(np.diff(tr_w), 4).tolist()} against three-sigma "
        f"tolerances {np.round(step_tol, 4).tolist()}")
    chk("F3 the last decade of traces buys under a tenth of the first",
        last_decade < 0.1 * first_decade,
        f"{tr_x[-3]} -> {tr_x[-1]} traces moves the width {last_decade:.4f}, against "
        f"{first_decade:.4f} for {tr_x[0]} -> {tr_x[2]}")
    chk("F4 the trace axis lands on the calibration-only floor",
        abs(tr_w[-1] - floor) < 0.015,
        f"width {tr_w[-1]:.4f} at {tr_x[-1]} traces against the floor {floor:.4f} at "
        f"the same {FRONTIER_REF_LABELS} labels, gap {tr_w[-1] - floor:+.4f}")
    chk("F5 THE LABEL-SIDE FLOOR EXISTS: the floor moves with labels and only with labels",
        (lb_f[0] - lb_f[-1]) > 0.02 and lb_f[0] > lb_f[-1],
        f"the calibration-only floor is {lb_f[0]:.4f} at {lb_x[0]} labels and "
        f"{lb_f[-1]:.4f} at {lb_x[-1]}, a fall of {lb_f[0] - lb_f[-1]:.4f} that no "
        f"number of traces reproduces (the trace axis is stuck at {tr_w[-1]:.4f})")
    chk("F6 labels beat traces at the flat end of the trace axis",
        (lb_w[0] - lb_w[-1]) > 3.0 * last_decade,
        f"labels {lb_x[0]} -> {lb_x[-1]} moves the width {lb_w[0] - lb_w[-1]:.4f}, more "
        f"than three times the {last_decade:.4f} the last trace decade buys")
    failed = [c["name"] for c in checks if not c["ok"]]
    if failed:
        raise AssertionError(f"frontier figure NOT written; failed checks: {failed}")

    figstyle.use_house_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.10, right=0.97)
    ax.set_xscale("log")
    ax.set_xlabel(f"traces per cell (bottom axis; human labels fixed at {FRONTIER_REF_LABELS})")
    ax.set_ylabel("95% interval half-width, cross-family contrast")
    ax.set_ylim(0.0, max(float(tr_w.max()), float(lb_w.max())) * 1.30)
    ax.grid(True, which="both", alpha=0.35, lw=0.6)

    figstyle.glow_line(ax, tr_x, tr_w, figstyle.PALETTE["orange"], marker="o", ms=5,
                       label="more traces (labels fixed)")
    ax.axhline(floor, color=figstyle.PALETTE["green"], ls="--", lw=1.4, alpha=0.9)
    ax.annotate(
        f"label-side floor {floor:.3f}\n(traces cannot cross it)",
        xy=(tr_x[2], floor), xytext=(tr_x[1], floor + 0.055),
        color=figstyle.PALETTE["green"], fontsize=10,
        arrowprops=dict(arrowstyle="->", color=figstyle.PALETTE["green"], lw=1.1),
    )

    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(lb_x[0] * 0.85, lb_x[-1] * 1.18)
    top.set_xlabel(f"human labels (top axis; traces fixed at {FRONTIER_REF_TRACES})",
                   color=figstyle.PALETTE["cyan"])
    top.tick_params(axis="x", colors=figstyle.PALETTE["cyan"])
    top.set_xticks(lb_x)
    top.set_xticklabels([str(v) for v in lb_x])
    top.minorticks_off()
    figstyle.glow_line(top, lb_x, lb_w, figstyle.PALETTE["cyan"], marker="s", ms=5,
                       label="more human labels (traces fixed)")
    top.plot(lb_x, lb_f, color=figstyle.PALETTE["violet"], ls=":", lw=1.6, marker="^",
             ms=4, alpha=0.95)

    handles = [
        plt.Line2D([], [], color=figstyle.PALETTE["orange"], marker="o", lw=2.4,
                   label="more traces (labels fixed at %d)" % FRONTIER_REF_LABELS),
        plt.Line2D([], [], color=figstyle.PALETTE["cyan"], marker="s", lw=2.4,
                   label="more human labels (traces fixed at %d)" % FRONTIER_REF_TRACES),
        plt.Line2D([], [], color=figstyle.PALETTE["green"], ls="--", lw=1.4,
                   label="floor at %d labels (traces cannot cross)" % FRONTIER_REF_LABELS),
        plt.Line2D([], [], color=figstyle.PALETTE["violet"], ls=":", marker="^", lw=1.6,
                   label="floor as a function of labels"),
    ]
    legend = ax.legend(handles=handles, framealpha=0.0, loc="upper right", fontsize=9)
    for t in legend.get_texts():
        t.set_color(figstyle.PALETTE["text_2"])
    fig.suptitle("Free compute does not buy precision past the label-side floor",
                 y=0.975, fontsize=14, color=figstyle.PALETTE["text"])
    ax.set_title(
        f"corrected P(no mention | followed), cross-family contrast, "
        f"{n_rep} replicates per point",
        pad=46, fontsize=10.5, color=figstyle.PALETTE["text_2"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=figstyle.PALETTE["bg"])
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return checks
