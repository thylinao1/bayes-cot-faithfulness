"""Contrast precision, the label-versus-trace frontier, and the ranking rule (W4).

Owner of 01-SIZING.md section J (REVIEW-CHANGE-LIST PF-11). Section I sized ONE
model's corrected P(no mention | followed) against ONE calibration stratum with
ONE judge, unclustered and unpooled. Nothing in 01-SIZING sized a cross-model
CONTRAST, while the plan promises 18 ranked rows. This script sizes the
contrast, the frontier that bounds it, and the partial-ordering rule.

THE DESIGN SIMULATED (01-SIZING section F, CONTRACT.md panel rule)

  Calibration frame: five strata (Qwen, Llama, Gemma, gpt-oss, OLMo), each of
  200 rows = 50 Q1-positive hinted rows, 100 Q1-negative hinted rows, 40
  clean-arm rows (negatives, specificity only) and 10 simple-random rows drawn
  at the natural Q1 prevalence. 1,000 human labels in total at the design point.

  Panel: the deployed label is the majority of a 3-judge panel. A random 20
  percent of every stratum's rows carries a fourth vote; that vote identifies
  per-judge error and is NOT used for the deployed aggregate (the aggregate is
  always scored on the deployed 3-panel, so all 50 positives per stratum
  calibrate the deployed object). A 4-panel row's 2-2 tie resolves to an
  independent tiebreaker.

  Clustering, both levels, as PF-12 (iii) requires:
    vote level   votes inside one item share an item difficulty effect
                 (tetrachoric correlation RHO_VOTE), so a panel is worth less
                 than three independent votes and per-judge counts carry a
                 design effect 1 + (m - 1) * ICC.
    item level   items arrive in base-problem clusters of CLUSTER_SIZE (the same
                 base question under two cue arms), true class correlated within
                 cluster at RHO_ITEM, so the follow-stratum rate carries a design
                 effect 1 + (c - 1) * ICC estimated per replicate by one-way
                 ANOVA and divided into the Hajek effective sample size.

  Enrichment: rows enter the follow stratum with RECORDED probabilities, flagged
  rows at ENRICHMENT_RATIO times the rate of unflagged ones; every estimate is
  the Hajek ratio estimator with w = 1 / pi.

  Pooling: sensitivity and specificity are partially pooled across the five
  strata by a normal-normal hierarchical model on the logit scale with a
  stratum random effect (half-Cauchy prior on tau, exact grid-plus-conjugate
  posterior sampler, validated against the no-pooling Beta limit in check C3).
  The four models whose family has no calibration stratum (Mistral x2, Phi, GLM)
  draw theta_new from the posterior PREDICTIVE, which is the plan's "coverage 14
  of 18" made quantitative.

WHAT IS REPORTED

  J.1  corrected-rate contrast half-width, same family and different families,
       at follow-stratum sizes 20, 50, 100, 300 and at the follow strata implied
       by sweeps of 300 and 5,000 traces, beside the calibration-only floor
       (the width when the follow-stratum rate is known exactly).
  J.2  the label-versus-trace frontier as figures/label_vs_trace_frontier.png,
       with an embedded assert that the label-side floor exists.
  J.3  the ranking simulation: 18 models on a pre-registered spread, the
       fraction of the 153 pairwise orderings that reach posterior probability
       RANK_THRESHOLD, and the accuracy of the ones that do.

ASSUMPTIONS, stated because they are choices and not measurements:
  1. JUDGE_SE / JUDGE_SP are PER-JUDGE truths. The deployed aggregate's
     sensitivity and specificity are DERIVED from them by Gauss-Hermite
     integration over the shared item effect and printed beside them; they are
     better than a single judge and worse than three independent votes.
  2. RHO_VOTE and RHO_ITEM (both 0.30) are placeholders. Nothing here measures
     them; Phase 1 does, and the tables move if they move.
  3. CLUSTER_SIZE 2: the same base problem appears under two cue arms.
  4. A 2-2 tie resolves by an independent tiebreaker of accuracy TIE_ACCURACY
     0.5, the uninformative (conservative) reading of the coherence gate, which
     is not itself calibrated yet.
  5. The regex flag driving the enrichment has recall REGEX_RECALL and the
     false-positive rate implied by REGEX_PRECISION_AT_FRAME, carried over from
     notebook 06 unchanged so section J nests section I.
  6. Judge label and regex flag are conditionally independent given true class.
  7. Beta and logit-normal posteriors use the Jeffreys prior Beta(0.5, 0.5).

Every number this script writes is asserted against an independent calculation
first (see ``assert_consistency``). Nothing is written if an assertion fails,
and the figure generator asserts before it writes.

Run:
    cd ~/Developer/bcf-sizing
    PYTHONPATH=src .venv/bin/python notebooks/07_contrast_precision.py
    PYTHONPATH=src .venv/bin/python notebooks/07_contrast_precision.py --smoke
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402
from contrast_sizing import *  # noqa: E402,F403
from contrast_sizing import (  # noqa: E402
    _clustered_bernoulli,
    _draw_panel_labels,
    _hw,
    _icc_anova,
)
import contrast_sizing  # noqa: E402
from contrast_figure import OUT_FIG, assert_and_write_frontier  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = _ROOT / "experiments" / "contrast_precision_results.json"

# ---------------------------------------------------------------------------
# 4. J.1 contrast half-width.
# ---------------------------------------------------------------------------

SAME_FAMILY_STRATUM = (0, 0)   # two Qwen models share a calibration stratum.
DIFF_FAMILY_STRATUM = (0, 2)   # Qwen against Gemma.
JUDGE_SETTINGS = (
    ("judge_assumed", JUDGE_SE, JUDGE_SP, 0),
    ("judge_stress", JUDGE_SE_STRESS, JUDGE_SP_STRESS, 100_003),
)


def run_j1(judge_se, judge_sp, follow_sizes, label, seed_offset, n_replicates=N_REPLICATES):
    se_agg, sp_agg = aggregate_error(judge_se, judge_sp)
    q_pop_a = population_judged_rate(P_NOMENTION_A, se_agg, sp_agg)
    q_pop_b = population_judged_rate(P_NOMENTION_B, se_agg, sp_agg)
    truth_gap = P_NOMENTION_A - P_NOMENTION_B
    rows = []
    for n_f in follow_sizes:
        rng = np.random.default_rng(MASTER_SEED + 13 * n_f + seed_offset)
        acc = {k: [] for k in (
            "same_hw", "diff_hw", "single_hw", "same_cov", "diff_cov", "single_cov",
            "same_mean", "diff_mean", "single_mean",
            "same_post_sd", "diff_post_sd",
            "floor_same", "floor_diff", "floor_single",
            "n_eff", "vote_deff", "item_deff", "j_agg_hw",
        )}
        for _ in range(n_replicates):
            frame = simulate_calibration_frame(rng, judge_se, judge_sp)
            se_d = hierarchical_draws(frame["x_se"], frame["n_se"], rng)
            sp_d = hierarchical_draws(frame["x_sp"], frame["n_sp"], rng)

            qa, na = simulate_follow_stratum(n_f, rng, P_NOMENTION_A, judge_se, judge_sp)
            qb, nb = simulate_follow_stratum(n_f, rng, P_NOMENTION_B, judge_se, judge_sp)
            s0, s1 = SAME_FAMILY_STRATUM
            d0, d1 = DIFF_FAMILY_STRATUM

            a_same = corrected_draws(qa, na, se_d[:, s0], sp_d[:, s0], rng)
            b_same = corrected_draws(qb, nb, se_d[:, s1], sp_d[:, s1], rng)
            a_diff = corrected_draws(qa, na, se_d[:, d0], sp_d[:, d0], rng)
            b_diff = corrected_draws(qb, nb, se_d[:, d1], sp_d[:, d1], rng)

            for key, draws, target in (
                ("same", a_same - b_same, truth_gap),
                ("diff", a_diff - b_diff, truth_gap),
                ("single", a_same, P_NOMENTION_A),
            ):
                lo, hi = np.nanpercentile(draws, [2.5, 97.5])
                acc[f"{key}_hw"].append(float((hi - lo) / 2.0))
                acc[f"{key}_cov"].append(bool(lo <= target <= hi))
                acc[f"{key}_mean"].append(float(np.nanmean(draws)))
                if key != "single":
                    acc[f"{key}_post_sd"].append(float(np.nanstd(draws)))

            # Calibration-only floor: the follow-stratum rate known exactly.
            acc["floor_same"].append(_hw(
                rogan_gladen(q_pop_a, se_d[:, s0], sp_d[:, s0])
                - rogan_gladen(q_pop_b, se_d[:, s1], sp_d[:, s1])))
            acc["floor_diff"].append(_hw(
                rogan_gladen(q_pop_a, se_d[:, d0], sp_d[:, d0])
                - rogan_gladen(q_pop_b, se_d[:, d1], sp_d[:, d1])))
            acc["floor_single"].append(_hw(rogan_gladen(q_pop_a, se_d[:, s0], sp_d[:, s0])))
            acc["j_agg_hw"].append(_hw(se_d[:, s0] + sp_d[:, s0] - 1.0))
            acc["n_eff"].append(na)
            acc["vote_deff"].append(frame["vote_deff"])
            acc["item_deff"].append(frame["item_deff"])

        row = {"judge": label, "n_follow": int(n_f), "n_replicates": n_replicates,
               "judge_se_per_judge": judge_se, "judge_sp_per_judge": judge_sp,
               "se_aggregate": se_agg, "sp_aggregate": sp_agg,
               "j_aggregate": se_agg + sp_agg - 1.0, "true_contrast": truth_gap}
        # Across-replicate spread of the point estimate, beside the mean posterior
        # sd. If the first exceeds the second the posterior is too narrow, which is
        # a measurement and not an inference about why.
        row["same_mean_sd_across_replicates"] = float(np.std(acc["same_mean"], ddof=1))
        row["diff_mean_sd_across_replicates"] = float(np.std(acc["diff_mean"], ddof=1))
        row.update({k: float(np.mean(v)) for k, v in acc.items()})
        rows.append(row)
        print(
            f"[J.1] {label} n_follow={n_f} denom={n_replicates} "
            f"same-family hw={row['same_hw']:.4f} (floor {row['floor_same']:.4f}) "
            f"diff-family hw={row['diff_hw']:.4f} (floor {row['floor_diff']:.4f}) "
            f"single hw={row['single_hw']:.4f} (floor {row['floor_single']:.4f}) "
            f"cov same/diff={row['same_cov']:.3f}/{row['diff_cov']:.3f}",
            flush=True,
        )
    return rows


def run_j1b(judge_se, judge_sp, label, seed_offset, n_replicates=N_REPLICATES):
    """PF-12 (iii): the weighted Youden J interval with and without clustering.

    Item level: the aggregate J estimated from item-level panel labels, with and
    without the within-class item design effect.
    Vote level: the per-judge J estimated from all votes cast, with and without
    the intra-item vote design effect. Ignoring the vote clustering treats three
    correlated votes on one item as three independent observations.
    """
    rng = np.random.default_rng(MASTER_SEED + 555 + seed_offset)
    se_agg, sp_agg = aggregate_error(judge_se, judge_sp)
    acc = {k: [] for k in ("agg_corrected", "agg_naive", "vote_corrected", "vote_naive",
                           "item_deff", "vote_deff", "item_icc", "vote_icc")}
    for _ in range(n_replicates):
        f = simulate_calibration_frame(rng, judge_se, judge_sp)
        for key, xs, ns, xp, npn in (
            ("agg_corrected", f["x_se"], f["n_se"], f["x_sp"], f["n_sp"]),
            ("agg_naive", f["x_se_raw"], f["n_se_raw"], f["x_sp_raw"], f["n_sp_raw"]),
            ("vote_corrected", f["vote_x_se"] / f["vote_deff"], f["vote_n_se"] / f["vote_deff"],
             f["vote_x_sp"] / f["vote_deff"], f["vote_n_sp"] / f["vote_deff"]),
            ("vote_naive", f["vote_x_se"], f["vote_n_se"], f["vote_x_sp"], f["vote_n_sp"]),
        ):
            se_d = hierarchical_draws(xs, ns, rng)
            sp_d = hierarchical_draws(xp, npn, rng)
            acc[key].append(_hw(se_d[:, 0] + sp_d[:, 0] - 1.0))
        for k in ("item_deff", "vote_deff", "item_icc", "vote_icc"):
            acc[k].append(f[k])
    out = {"judge": label, "n_replicates": n_replicates,
           "se_aggregate": se_agg, "sp_aggregate": sp_agg,
           "j_aggregate": se_agg + sp_agg - 1.0,
           "j_per_judge": judge_se + judge_sp - 1.0}
    out.update({k: float(np.mean(v)) for k, v in acc.items()})
    print(
        f"[J.1b] {label} denom={n_replicates} aggregate-J half-width "
        f"{out['agg_corrected']:.4f} clustered vs {out['agg_naive']:.4f} naive "
        f"(item deff {out['item_deff']:.3f}); per-judge-J half-width "
        f"{out['vote_corrected']:.4f} clustered vs {out['vote_naive']:.4f} naive "
        f"(vote deff {out['vote_deff']:.3f})",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# 5. J.2 label-versus-trace frontier.
# ---------------------------------------------------------------------------


def frontier_point(n_labels, n_traces, judge_se, judge_sp, rng, n_replicates):
    """Mean half-width of the single-model rate and the cross-family contrast."""
    scale = n_labels / DESIGN_LABELS
    n_f = max(4, int(round(n_traces * FOLLOW_RATE)))
    se_agg, sp_agg = aggregate_error(judge_se, judge_sp)
    q_pop_a = population_judged_rate(P_NOMENTION_A, se_agg, sp_agg)
    q_pop_b = population_judged_rate(P_NOMENTION_B, se_agg, sp_agg)
    single, contrast, f_single, f_contrast = [], [], [], []
    for _ in range(n_replicates):
        frame = simulate_calibration_frame(rng, judge_se, judge_sp, scale)
        se_d = hierarchical_draws(frame["x_se"], frame["n_se"], rng)
        sp_d = hierarchical_draws(frame["x_sp"], frame["n_sp"], rng)
        d0, d1 = DIFF_FAMILY_STRATUM
        qa, na = simulate_follow_stratum(n_f, rng, P_NOMENTION_A, judge_se, judge_sp)
        qb, nb = simulate_follow_stratum(n_f, rng, P_NOMENTION_B, judge_se, judge_sp)
        a = corrected_draws(qa, na, se_d[:, d0], sp_d[:, d0], rng)
        b = corrected_draws(qb, nb, se_d[:, d1], sp_d[:, d1], rng)
        single.append(_hw(a))
        contrast.append(_hw(a - b))
        f_single.append(_hw(rogan_gladen(q_pop_a, se_d[:, d0], sp_d[:, d0])))
        f_contrast.append(_hw(
            rogan_gladen(q_pop_a, se_d[:, d0], sp_d[:, d0])
            - rogan_gladen(q_pop_b, se_d[:, d1], sp_d[:, d1])
        ))
    return {
        "n_labels": int(n_labels), "n_traces": int(n_traces), "n_follow": int(n_f),
        "labels_per_stratum": int(round(CAL_ROWS_PER_STRATUM * scale)),
        "n_replicates": n_replicates,
        "single_hw": float(np.mean(single)), "contrast_hw": float(np.mean(contrast)),
        "single_floor": float(np.mean(f_single)),
        "contrast_floor": float(np.mean(f_contrast)),
        "contrast_hw_se": float(np.std(contrast, ddof=1) / np.sqrt(n_replicates)),
        "contrast_floor_se": float(np.std(f_contrast, ddof=1) / np.sqrt(n_replicates)),
    }


def run_j2(judge_se, judge_sp, n_replicates=N_REPLICATES_FRONTIER):
    rng = np.random.default_rng(MASTER_SEED + 991)
    trace_sweep, label_sweep = [], []
    for n_traces in FRONTIER_TRACES:
        pt = frontier_point(FRONTIER_REF_LABELS, n_traces, judge_se, judge_sp, rng, n_replicates)
        trace_sweep.append(pt)
        print(f"[J.2 traces] labels={pt['n_labels']} traces={pt['n_traces']} "
              f"follow={pt['n_follow']} denom={n_replicates} "
              f"single={pt['single_hw']:.4f} contrast={pt['contrast_hw']:.4f} "
              f"floor={pt['contrast_floor']:.4f}", flush=True)
    for n_labels in FRONTIER_LABELS:
        pt = frontier_point(n_labels, FRONTIER_REF_TRACES, judge_se, judge_sp, rng, n_replicates)
        label_sweep.append(pt)
        print(f"[J.2 labels] labels={pt['n_labels']} traces={pt['n_traces']} "
              f"denom={n_replicates} single={pt['single_hw']:.4f} "
              f"contrast={pt['contrast_hw']:.4f} floor={pt['contrast_floor']:.4f}", flush=True)
    return {"trace_sweep": trace_sweep, "label_sweep": label_sweep}


# ---------------------------------------------------------------------------
# 6. J.3 ranking simulation.
# ---------------------------------------------------------------------------


# Fixed pre-registered permutation that assigns the spread's rank positions to
# the roster, so a model's family is not a function of its true rate (families
# arrive as contiguous blocks, and a block-ordered assignment would make every
# same-family pair adjacent and unresolvable by construction).
RANK_PERMUTATION_STRIDE = 7


def _model_strata():
    strata = []
    for s_idx, k in enumerate(FAMILY_SIZES_CALIBRATED):
        strata.extend([s_idx] * k)
    strata.extend([N_STRATA + i for i in range(N_UNCALIBRATED)])
    return np.array(strata)


def run_ranking(judge_se, judge_sp, spread, n_follow, label, seed_offset,
                n_replicates=N_REPLICATES_RANK):
    lo, hi = spread
    grid = np.linspace(lo, hi, N_MODELS)
    order = (np.arange(N_MODELS) * RANK_PERMUTATION_STRIDE) % N_MODELS
    true_rates = grid[order]
    strata = _model_strata()
    assert strata.size == N_MODELS
    calibrated = strata < N_STRATA

    iu, ju = np.triu_indices(N_MODELS, k=1)
    n_pairs = int(iu.size)
    same_family = np.array([strata[i] == strata[j] and strata[i] < N_STRATA
                            for i, j in zip(iu, ju)])
    both_calibrated = calibrated[iu] & calibrated[ju]
    any_uncalibrated = ~both_calibrated
    truth_gt = true_rates[iu] > true_rates[ju]
    true_gap = np.abs(true_rates[iu] - true_rates[ju])

    rng = np.random.default_rng(MASTER_SEED + 7717 + n_follow + seed_offset)
    resolved_count = np.zeros(n_pairs)
    correct_count = np.zeros(n_pairs)
    for _ in range(n_replicates):
        frame = simulate_calibration_frame(rng, judge_se, judge_sp)
        se_d = hierarchical_draws(frame["x_se"], frame["n_se"], rng, n_new=N_UNCALIBRATED)
        sp_d = hierarchical_draws(frame["x_sp"], frame["n_sp"], rng, n_new=N_UNCALIBRATED)
        draws = np.empty((N_DRAWS, N_MODELS))
        for m in range(N_MODELS):
            q, ne = simulate_follow_stratum(n_follow, rng, true_rates[m], judge_se, judge_sp)
            draws[:, m] = corrected_draws(q, ne, se_d[:, strata[m]], sp_d[:, strata[m]], rng)
        p_gt = np.nanmean(draws[:, iu] > draws[:, ju], axis=0)
        resolved = np.maximum(p_gt, 1.0 - p_gt) >= RANK_THRESHOLD
        correct = resolved & ((p_gt >= 0.5) == truth_gt)
        resolved_count += resolved
        correct_count += correct

    denom = n_pairs * n_replicates
    n_res = float(resolved_count.sum())
    n_cor = float(correct_count.sum())

    def _subset(mask):
        d = int(mask.sum()) * n_replicates
        r = float(resolved_count[mask].sum())
        c = float(correct_count[mask].sum())
        return {"denominator": d, "n_resolved": int(r),
                "resolved_fraction": (r / d) if d else float("nan"),
                "accuracy_of_resolved": (c / r) if r else float("nan")}

    # Resolution as a function of the true gap: the sizing question a reader has.
    gap_bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]
    by_gap = []
    for g_lo, g_hi in gap_bins:
        mask = (true_gap >= g_lo) & (true_gap < g_hi)
        if not mask.any():
            continue
        entry = {"gap_low": g_lo, "gap_high": g_hi, "n_pairs": int(mask.sum())}
        entry.update(_subset(mask))
        by_gap.append(entry)

    out = {
        "judge": label, "spread": list(spread), "n_follow": int(n_follow),
        "n_models": N_MODELS, "n_pairs": n_pairs, "n_replicates": n_replicates,
        "denominator_pair_decisions": int(denom),
        "adjacent_gap_on_the_spread": float(grid[1] - grid[0]),
        "resolved_fraction": n_res / denom, "n_resolved": int(n_res),
        "accuracy_of_resolved": (n_cor / n_res) if n_res else float("nan"),
        "n_resolved_correct": int(n_cor),
        "same_family": _subset(same_family),
        "cross_family": _subset(~same_family),
        "both_calibrated": _subset(both_calibrated),
        "any_uncalibrated": _subset(any_uncalibrated),
        "by_true_gap": by_gap,
    }
    print(
        f"[J.3] {label} spread={spread} n_follow={n_follow} denom={denom} pair decisions "
        f"({n_pairs} pairs x {n_replicates} replicates): resolved "
        f"{out['resolved_fraction']:.4f} ({int(n_res)}/{denom}), accuracy of resolved "
        f"{out['accuracy_of_resolved']:.4f} ({int(n_cor)}/{int(n_res)}); "
        f"both-calibrated {out['both_calibrated']['resolved_fraction']:.4f} vs "
        f"has-uncalibrated {out['any_uncalibrated']['resolved_fraction']:.4f}",
        flush=True,
    )
    for e in by_gap:
        print(f"      gap [{e['gap_low']:.2f}, {e['gap_high']:.2f}): "
              f"{e['n_pairs']} pairs, denom {e['denominator']}, resolved "
              f"{e['resolved_fraction']:.4f}, accuracy {e['accuracy_of_resolved']:.4f}",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# 7. Consistency checks. Nothing is written unless every one passes.
# ---------------------------------------------------------------------------


def assert_consistency(j1_main, j1_stress, j1b, frontier, ranking, n_rep):
    checks = []

    def chk(name, ok, detail, skipped=False):
        checks.append({"name": name, "ok": bool(ok) or skipped,
                       "skipped": bool(skipped), "detail": detail})
        tag = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        print(f"  [{tag}] {name}: {detail}", flush=True)

    # C1. Rogan-Gladen inverts the population judged rate exactly.
    se_agg, sp_agg = aggregate_error(JUDGE_SE, JUDGE_SP)
    q = population_judged_rate(P_NOMENTION_A, se_agg, sp_agg)
    back = float(rogan_gladen(np.array([q]), np.array([se_agg]), np.array([sp_agg]))[0])
    chk("C1 Rogan-Gladen inverts the population rate",
        abs(back - P_NOMENTION_A) < 1e-9,
        f"aggregate se={se_agg:.6f} sp={sp_agg:.6f}; RG({q:.6f}) = {back:.10f} "
        f"vs truth {P_NOMENTION_A}")

    # C2. The Gauss-Hermite aggregate matches a brute-force Monte Carlo panel.
    rng = np.random.default_rng(4242)
    n_mc = 400_000
    lab1, _ = _draw_panel_labels(np.ones(n_mc, dtype=np.int8), rng, JUDGE_SE, JUDGE_SP,
                                PANEL_DEPLOYED)
    lab0, _ = _draw_panel_labels(np.zeros(n_mc, dtype=np.int8), rng, JUDGE_SE, JUDGE_SP,
                                PANEL_DEPLOYED)
    se_mc, sp_mc = float(lab1.mean()), float(1.0 - lab0.mean())
    chk("C2 Gauss-Hermite aggregate matches Monte Carlo",
        abs(se_mc - se_agg) < 0.004 and abs(sp_mc - sp_agg) < 0.004,
        f"se {se_agg:.5f} vs {se_mc:.5f}, sp {sp_agg:.5f} vs {sp_mc:.5f} "
        f"(denominator {n_mc} simulated items per class)")

    # C3. In the no-pooling limit (tau forced large) the hierarchical sampler
    #     reproduces the Beta posterior it approximates, up to the second-order
    #     bias of the logit-normal approximation, which is predicted analytically
    #     here so its size is on the record rather than absorbed by a tolerance.
    rng = np.random.default_rng(99)
    x = np.array([42.0, 43.0, 41.0, 44.0, 40.0])
    n = np.full(5, 50.0)
    saved = contrast_sizing.TAU_GRID
    n_d = 200_000
    try:
        contrast_sizing.TAU_GRID = np.array([TAU_NO_POOLING])
        no_pool = hierarchical_draws(x, n, rng, n_draws=n_d)
    finally:
        contrast_sizing.TAU_GRID = saved
    beta_ref = rng.beta(x[0] + JEFFREYS, n[0] - x[0] + JEFFREYS, size=n_d)
    y0 = float(logit((x[0] + JEFFREYS) / (n[0] + 2.0 * JEFFREYS)))
    v0 = 1.0 / (x[0] + JEFFREYS) + 1.0 / (n[0] - x[0] + JEFFREYS)
    p0 = float(expit(y0))
    predicted = p0 + 0.5 * v0 * p0 * (1.0 - p0) * (1.0 - 2.0 * p0)
    chk("C3 the no-pooling sampler matches the Beta posterior up to its predicted bias",
        abs(float(no_pool[:, 0].mean()) - predicted) < 0.002
        and abs(float(no_pool[:, 0].std()) - float(beta_ref.std())) < 0.004,
        f"no-pooling mean {no_pool[:, 0].mean():.4f} against the delta-method "
        f"prediction {predicted:.4f}; exact Beta mean {beta_ref.mean():.4f}, so the "
        f"logit-normal approximation displaces the mean by "
        f"{predicted - float(beta_ref.mean()):+.4f} at {int(x[0])} of {int(n[0])}. "
        f"sd {no_pool[:, 0].std():.4f} vs exact {beta_ref.std():.4f} "
        f"(denominator {n_d} draws each)")

    # C4. Pooling shrinks: the fitted tau grid gives a narrower stratum interval.
    pooled = hierarchical_draws(x, n, rng, n_draws=n_d)
    chk("C4 partial pooling narrows the stratum interval",
        float(pooled[:, 0].std()) < float(no_pool[:, 0].std()),
        f"pooled sd {pooled[:, 0].std():.5f} < no-pooling sd {no_pool[:, 0].std():.5f} "
        f"(denominator {n_d} draws each; five strata of 50)")

    # C5. The clustered Bernoulli preserves its marginal and hits the Pearson ICC
    #     implied by its tetrachoric correlation (which is NOT RHO_ITEM itself).
    rng = np.random.default_rng(7)
    p_target = 0.3
    v, _ = _clustered_bernoulli(np.full(400_000, p_target), rng)
    icc = _icc_anova(v.astype(float))
    expected_icc = tetrachoric_to_pearson(p_target, RHO_ITEM)
    chk("C5 clustered Bernoulli keeps its marginal and its implied ICC",
        abs(v.mean() - p_target) < 0.004 and abs(icc - expected_icc) < 0.015,
        f"mean {v.mean():.5f} vs {p_target}; recovered ICC {icc:.4f} against the "
        f"{expected_icc:.4f} implied by a tetrachoric {RHO_ITEM} at p={p_target} "
        f"(denominator {v.size} items)")

    # C6/C7/C8/C9: the J.1 grid, evaluated on the sorted follow sizes.
    order = np.argsort([r["n_follow"] for r in j1_main])
    same = [j1_main[i]["same_hw"] for i in order]
    diff = [j1_main[i]["diff_hw"] for i in order]
    floors = [j1_main[i]["floor_diff"] for i in order]
    sizes = [j1_main[i]["n_follow"] for i in order]
    tol = 0.004
    chk("C6 contrast width is monotone in the follow stratum",
        bool(np.all(np.diff(same) < tol)) and bool(np.all(np.diff(diff) < tol)),
        f"follow sizes {sizes}; same-family {np.round(same, 4).tolist()}, "
        f"different-family {np.round(diff, 4).tolist()} "
        f"(denominator {j1_main[0]['n_replicates']} replicates per cell)")
    chk("C7 the different-family contrast never beats its calibration-only floor",
        all(d >= f - tol for d, f in zip(diff, floors)),
        f"widths {np.round(diff, 4).tolist()} against floors "
        f"{np.round(floors, 4).tolist()}")
    chk("C8 sharing a calibration stratum narrows the contrast",
        all(a <= b + 1e-9 for a, b in zip(same, diff)),
        f"same-family {np.round(same, 4).tolist()} <= different-family "
        f"{np.round(diff, 4).tolist()} at every follow size")

    # C9. Calibration is checked where the design lives. The plan's follow
    #     stratum is 20 to 300 items (01-SIZING section G: at least 20, about 98
    #     at 300 clean-correct items per cell); the 1,625 row exists to locate
    #     the floor, and its departure is reported by C16 rather than hidden.
    design = [r for r in j1_main + j1_stress if r["n_follow"] <= max(FOLLOW_SIZES)]
    covs = [r["diff_cov"] for r in design] + [r["same_cov"] for r in design]
    band = 3.0 * np.sqrt(0.95 * 0.05 / n_rep)
    chk("C9 contrast coverage is near nominal across the design region",
        min(covs) > 0.95 - band and max(covs) < 0.95 + band,
        f"coverage over the {len(design)} cells at follow sizes "
        f"{sorted({r['n_follow'] for r in design})} runs {min(covs):.3f} to "
        f"{max(covs):.3f}, inside the three-sigma window "
        f"[{0.95 - band:.3f}, {0.95 + band:.3f}] at {n_rep} replicates per cell")

    # C16. The departure at the largest follow stratum, reported with the
    #     decomposition that measures it. The assert covers only what is
    #     observed; the detail string carries the diagnostic so section J can
    #     state the cause from measurement rather than from a story.
    big = sorted(j1_main, key=lambda r: r["n_follow"])[-1]
    truth_gap = P_NOMENTION_A - P_NOMENTION_B
    bias_same = big["same_mean"] - truth_gap
    sd_across = big["same_mean_sd_across_replicates"]
    sd_post = big["same_post_sd"]
    # A coverage gap of about 0.06 cannot be seen at a handful of replicates: the
    # standard error of a coverage estimate at nominal 0.95 is 0.049 at 20
    # replicates and 0.010 at 500. Below the threshold the check is recorded as
    # not evaluated rather than passed.
    chk("C16 past the design region the same-family interval stops covering at nominal",
        big["same_cov"] < big["diff_cov"]
        and 0.85 < big["same_cov"] < 0.99 and 0.85 < big["diff_cov"] < 0.99,
        (f"NOT EVALUATED at {n_rep} replicates (needs at least {C16_MIN_REPLICATES}; "
         f"the coverage standard error is "
         f"{np.sqrt(0.95 * 0.05 / n_rep):.3f} here against the ~0.06 gap being tested)")
        if n_rep < C16_MIN_REPLICATES else
        f"at {big['n_follow']} followed items, far past the 20 to "
        f"{max(FOLLOW_SIZES)} the design uses, the same-family contrast covers "
        f"{big['same_cov']:.3f} and the different-family one {big['diff_cov']:.3f} "
        f"against nominal 0.95 (denominator {big['n_replicates']} replicates). "
        f"Decomposition: the posterior mean is {big['same_mean']:.4f} against a truth "
        f"of {truth_gap}, a displacement of {bias_same:+.4f}; the mean posterior sd is "
        f"{sd_post:.4f} while the across-replicate sd of that mean is {sd_across:.4f}, "
        f"a ratio of {sd_across / sd_post:.3f}. Both together, not either alone, are "
        f"the size of the shortfall; the interval at this width is approaching the "
        f"{big['floor_same']:.4f} same-family floor, where the pooled logit-normal "
        f"approximation's own displacement (C3) is no longer negligible beside it",
        skipped=n_rep < C16_MIN_REPLICATES)

    # C10. The stress judge is worse than the assumed judge everywhere.
    pair = sorted(zip(j1_main, j1_stress), key=lambda t: t[0]["n_follow"])
    for a, b in pair:
        assert a["n_follow"] == b["n_follow"]
    chk("C10 the stress judge widens every contrast",
        all(b["diff_hw"] > a["diff_hw"] for a, b in pair),
        f"assumed {[round(a['diff_hw'], 4) for a, _ in pair]} vs stress "
        f"{[round(b['diff_hw'], 4) for _, b in pair]}")

    # C11. Ignoring the clustering makes the J interval look narrower than it is.
    chk("C11 clustering widens the J interval at both levels",
        all(r["agg_corrected"] >= r["agg_naive"] - 1e-9
            and r["vote_corrected"] > r["vote_naive"] for r in j1b),
        "; ".join(
            f"{r['judge']}: aggregate J half-width {r['agg_corrected']:.4f} clustered vs "
            f"{r['agg_naive']:.4f} naive, per-judge J {r['vote_corrected']:.4f} vs "
            f"{r['vote_naive']:.4f} (denominator {r['n_replicates']} replicates)"
            for r in j1b))

    # C12. The frontier's two axes cross at the design point.
    crossing = [pt for pt in frontier["label_sweep"] if pt["n_labels"] == FRONTIER_REF_LABELS]
    chk("C12 the two frontier axes cross at the design point",
        len(crossing) == 1 and crossing[0]["n_traces"] == FRONTIER_REF_TRACES,
        f"label sweep contains labels={FRONTIER_REF_LABELS} at "
        f"traces={FRONTIER_REF_TRACES}")

    # C13/C14. Ranking behaviour.
    by_key = {(r["judge"], r["n_follow"], tuple(r["spread"])): r for r in ranking}
    k100 = by_key[("judge_assumed", RANK_FOLLOW_SIZES[0], RANK_SPREAD_PRIMARY)]
    k300 = by_key[("judge_assumed", RANK_FOLLOW_SIZES[1], RANK_SPREAD_PRIMARY)]
    kstr = by_key[("judge_assumed", RANK_FOLLOW_SIZES[0], RANK_SPREAD_STRESS)]
    chk("C13 ranking resolution rises with the follow stratum and falls with a narrow spread",
        k300["resolved_fraction"] > k100["resolved_fraction"]
        and kstr["resolved_fraction"] < k100["resolved_fraction"],
        f"{k100['resolved_fraction']:.4f} at n_follow {k100['n_follow']} -> "
        f"{k300['resolved_fraction']:.4f} at {k300['n_follow']} on spread "
        f"{RANK_SPREAD_PRIMARY}; {kstr['resolved_fraction']:.4f} on the narrower "
        f"{RANK_SPREAD_STRESS} (denominators {k100['denominator_pair_decisions']}, "
        f"{k300['denominator_pair_decisions']}, {kstr['denominator_pair_decisions']} "
        f"pair decisions)")
    chk("C14 the 0.90 rule's declared orderings keep their nominal error rate",
        all(r["accuracy_of_resolved"] >= RANK_THRESHOLD for r in ranking),
        "accuracy of resolved pairs against the rule's nominal "
        f"{RANK_THRESHOLD} floor: "
        + ", ".join(f"{r['judge']}/n_follow {r['n_follow']}/spread {tuple(r['spread'])} = "
                    f"{r['accuracy_of_resolved']:.4f} on {r['n_resolved']} resolved"
                    for r in ranking))

    # C15. A model with no calibration stratum of its own resolves less often.
    chk("C15 an uncalibrated family's pairs resolve less often",
        all(r["any_uncalibrated"]["resolved_fraction"]
            <= r["both_calibrated"]["resolved_fraction"] + 0.01 for r in ranking),
        "; ".join(
            f"{r['judge']}/{r['n_follow']}/{tuple(r['spread'])}: both calibrated "
            f"{r['both_calibrated']['resolved_fraction']:.4f} "
            f"(denominator {r['both_calibrated']['denominator']}) vs has-uncalibrated "
            f"{r['any_uncalibrated']['resolved_fraction']:.4f} "
            f"(denominator {r['any_uncalibrated']['denominator']})" for r in ranking))

    failed = [c["name"] for c in checks if not c["ok"]]
    if failed:
        raise AssertionError(f"consistency checks FAILED, nothing written: {failed}")
    n_skip = sum(c["skipped"] for c in checks)
    print(f"  {len(checks) - n_skip} consistency checks passed, {n_skip} not evaluated, "
          f"0 failed.", flush=True)
    return checks


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------


def main():
    if "--figure-only" in sys.argv:
        # Redraw from the banked run. The asserts still run before the write.
        banked = json.loads(OUT_JSON.read_text())
        assert_and_write_frontier(banked["j2_frontier"])
        return
    smoke = "--smoke" in sys.argv
    n_rep = 20 if smoke else N_REPLICATES
    n_rep_f = 15 if smoke else N_REPLICATES_FRONTIER
    n_rep_r = 10 if smoke else N_REPLICATES_RANK
    sweep_follow = [max(4, int(round(t * FOLLOW_RATE))) for t in SWEEP_TRACES]
    follow_sizes = tuple(sorted(set(list(FOLLOW_SIZES) + sweep_follow)))
    t0 = time.time()

    print(f"J.1 contrast half-width; follow sizes {follow_sizes}; the sweep sizes "
          f"{SWEEP_TRACES} traces map to {sweep_follow} followed items at the banked "
          f"{FOLLOW_RATE} follow rate")
    (m_label, m_se, m_sp, m_off), (s_label, s_se, s_sp, s_off) = JUDGE_SETTINGS
    j1_main = run_j1(m_se, m_sp, follow_sizes, m_label, m_off, n_rep)
    j1_stress = run_j1(s_se, s_sp, follow_sizes, s_label, s_off, n_rep)

    print("\nJ.1b the weighted J interval with and without the two clustering levels")
    j1b = [run_j1b(se, sp, lab, off, n_rep) for lab, se, sp, off in JUDGE_SETTINGS]

    print("\nJ.2 label-versus-trace frontier")
    frontier = run_j2(m_se, m_sp, n_rep_f)

    print("\nJ.3 ranking simulation")
    ranking = []
    for lab, se, sp, off in JUDGE_SETTINGS:
        for spread in (RANK_SPREAD_PRIMARY, RANK_SPREAD_STRESS):
            for n_f in RANK_FOLLOW_SIZES:
                if lab == s_label and (spread != RANK_SPREAD_PRIMARY
                                       or n_f != RANK_FOLLOW_SIZES[0]):
                    continue
                ranking.append(run_ranking(se, sp, spread, n_f, lab, off, n_rep_r))

    print("\nConsistency checks")
    checks = assert_consistency(j1_main, j1_stress, j1b, frontier, ranking, n_rep)

    print("\nFrontier figure checks")
    fig_checks = assert_and_write_frontier(frontier)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "smoke": smoke,
        "elapsed_seconds": round(time.time() - t0, 1),
        "constants": {
            "cal_rows_per_stratum": CAL_ROWS_PER_STRATUM, "n_strata": N_STRATA,
            "cal_positives": CAL_POSITIVES, "cal_neg_hinted": CAL_NEG_HINTED,
            "cal_neg_clean": CAL_NEG_CLEAN, "cal_srs": CAL_SRS,
            "design_labels": DESIGN_LABELS, "natural_prevalence": NATURAL_PREVALENCE,
            "panel_deployed": PANEL_DEPLOYED, "panel_all_judges": PANEL_ALL_JUDGES,
            "frac_all_judges": FRAC_ALL_JUDGES, "tie_accuracy": TIE_ACCURACY,
            "rho_vote": RHO_VOTE, "rho_item": RHO_ITEM, "cluster_size": CLUSTER_SIZE,
            "enrichment_ratio": ENRICHMENT_RATIO, "follow_rate": FOLLOW_RATE,
            "regex_recall": REGEX_RECALL, "regex_fp_rate": REGEX_FP_RATE,
            "p_nomention_a": P_NOMENTION_A, "p_nomention_b": P_NOMENTION_B,
            "judge_se": JUDGE_SE, "judge_sp": JUDGE_SP,
            "judge_se_stress": JUDGE_SE_STRESS, "judge_sp_stress": JUDGE_SP_STRESS,
            "follow_sizes": list(follow_sizes), "sweep_traces": list(SWEEP_TRACES),
            "sweep_follow": sweep_follow,
            "rank_spread_primary": list(RANK_SPREAD_PRIMARY),
            "rank_spread_stress": list(RANK_SPREAD_STRESS),
            "rank_threshold": RANK_THRESHOLD,
            "rank_permutation_stride": RANK_PERMUTATION_STRIDE,
            "family_sizes_calibrated": list(FAMILY_SIZES_CALIBRATED),
            "n_uncalibrated": N_UNCALIBRATED,
            "n_draws": N_DRAWS, "master_seed": MASTER_SEED,
        },
        "j1_contrast": j1_main + j1_stress,
        "j1b_clustering": j1b,
        "j2_frontier": frontier,
        "j3_ranking": ranking,
        "consistency_checks": checks,
        "figure_checks": fig_checks,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_JSON} ({payload['elapsed_seconds']} s)")


if __name__ == "__main__":
    main()
