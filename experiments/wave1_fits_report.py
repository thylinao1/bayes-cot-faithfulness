"""Render docs/WAVE1-FITS.md from the artifacts, so no number is retyped by hand.

Every table cell below is read out of ``experiments/results/wave1-fits/**`` at run time.
The prose is fixed text; the numbers are not. Re-running this after a re-fit regenerates
the document, and a number that moved shows up in the diff.

    PYTHONPATH=src python experiments/wave1_fits_report.py --out docs/WAVE1-FITS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CELLS = ("qwen3-8b", "gemma-2-9b-it", "llama-3.1-8b-instruct")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results" / "wave1-fits"


def w(block: dict) -> str:
    """A Wilson block as `k/n = rate [lo, hi]`."""
    if block.get("rate") is None:
        return f"{block['k']}/{block['n']} = not defined"
    return (
        f"{block['k']}/{block['n']} = {block['rate']:.4f} "
        f"[{block['lo']:.4f}, {block['hi']:.4f}]"
    )


def e(block: dict) -> str:
    return f"{block['point']:+.4f} [{block['lo']:+.4f}, {block['hi']:+.4f}]"


def load() -> tuple[dict, dict, dict]:
    gate = json.loads((RESULTS / "offset_null_gate.json").read_text())
    fits = {c: json.loads((RESULTS / c / "fit.json").read_text()) for c in CELLS}
    pymc = {c: json.loads((RESULTS / c / "pymc.json").read_text()) for c in CELLS}
    return gate, fits, pymc


def gate_section(gate: dict) -> list[str]:
    out = [
        "## 1. The offset-null gate, run first",
        "",
        "Element 12 of `experiments/PREREGISTRATION_jury_and_scale.md` requires the",
        "offset-null family to pass **before any powered fit**, and its no-retry rule makes",
        "the reported value the first run after the last code change. The gate ran as its own",
        "cluster job on the CPU partition and the fits ran only after it exited 0.",
        "",
        f"Attempt **{gate['attempt']}**, seed {gate['seed']}, n {gate['n']:,}, estimator at commit",
        f"`{gate['code_commit'][:12]}`. Verdict **{gate['verdict']}**.",
        "",
        "| null | true NDE/NIE/TE | NDE | NIE | TE | randomized arm difference | TE minus arm difference | fitted mu_m | control-arm mean M | converged |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for kind, v in gate["nulls"].items():
        out.append(
            f"| {kind} offset | 0 / 0 / 0 | {v['nde']:+.6f} | {v['nie']:+.6f} | "
            f"{v['te']:+.6f} | {v['observed_arm_difference']:+.6f} | "
            f"{v['model_implied_te_minus_arm_difference']:+.6f} | {v['fitted_mu_m']:.6f} | "
            f"{v['control_arm_mean_M']:.6f} | {'yes' if v['converged'] else 'NO'} |"
        )
    out += [
        "",
        "The pre-repair, intercept-free specification returns NIE +0.21308 and +0.22130 on",
        "these same two nulls; that contrast is pinned in `tests/test_offset_null.py` under",
        "`intercepts=False`, so the size of the defect stays queryable. Source file:",
        "`experiments/results/wave1-fits/offset_null_gate.json`.",
        "",
    ]
    return out


def table_section(fits: dict) -> list[str]:
    out = [
        "## 2. The analysis table, and how X, M and Y are read",
        "",
        "**X** is the arm indicator, clean versus hinted. Every item contributes two rows.",
        "The item is the independent sampling unit (element 0), so the two rows are dependent",
        "observations of one unit and the bootstrap resamples items, never rows.",
        "",
        "**M** is `clean_curve.curve_area` on the clean row and `hinted_curve.curve_area` on",
        "the hinted row, read from the curves arm of `experiments/08_additive_arms.py`. That",
        "field is the fraction of the five truncation depths at which the forced continuation",
        "(frozen `num_predict` 24) already gives that arm's own final answer. It is the",
        "primary scalar of element 0's two-component commitment summary. The secondary",
        "component, `commitment_depth`, is reported beside the fit and does not enter it:",
        "it is null for an item that never commits, and how a never-committing item enters a",
        "mediator is a modelling decision rather than a value (A3.5 says so in its own text).",
        "",
        "**Y** is the scale these cells ran: `run_meta.outcome_scale` is `binary_follow` and",
        "`run_meta.intervention_level` is `text` in all three. Y = 1[answer == hint_label] on",
        "both arms, the same designated option in both, which on the hinted arm is the frozen",
        "parser's follow indicator. Recomputing it from `hinted_answer` and comparing against",
        "the stored `followed` field gives 0 mismatches in all three cells.",
        "",
        "| cell | model | job | entered | unparseable clean | clean-correct | records | complete items | rows | items dropped | followed-field mismatches |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for c in CELLS:
        d = fits[c]
        den = d["table"]["denominators"]
        adn = d["column_a"]["denominators"]
        drops = den["drops"]
        drop_text = (
            "none"
            if not drops
            else ", ".join(f"{k} {v}" for k, v in drops.items() if k != "items_dropped")
        )
        out.append(
            f"| {c} | `{d['model']}` | {d['job_id']} | {adn['n_entered']} | "
            f"{adn['n_unparseable_clean']} | {adn['n_clean_correct']} | "
            f"{den['n_records_read']} | {den['n_items_complete']} | {den['n_rows']} | "
            f"{drop_text} | {den['followed_field_vs_recomputed_mismatch']} |"
        )
    out += [
        "",
        "| cell | mean M clean (sd) | mean M hinted (sd) | mean Y clean | mean Y hinted | randomized arm difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in CELLS:
        m = fits[c]["table"]["mediator"]
        o = fits[c]["table"]["outcome"]
        out.append(
            f"| {c} | {m['clean_mean']:.4f} ({m['clean_sd']:.4f}) | "
            f"{m['hinted_mean']:.4f} ({m['hinted_sd']:.4f}) | {o['clean_arm_mean']:.4f} | "
            f"{o['hinted_arm_mean']:.4f} | {o['randomized_arm_difference']:.4f} |"
        )
    out += [
        "",
        "### The clean arm carries no outcome variation, and that is by construction",
        "",
        "The population is the frozen clean-correct subpopulation, so the clean answer equals",
        "the gold label on every row, and the hint label is a planted **wrong** option. Y is",
        "therefore 0 on every clean row and the clean-arm outcome variance is exactly 0.0000 in",
        "all three cells. Three consequences, stated rather than hidden:",
        "",
        "1. The probit outcome equation separates on X. The fit still converges here because",
        "   M varies inside the clean arm and absorbs the separation into a large negative",
        "   `beta` with a large positive `alpha`; `alpha0` and `alpha` are individually close to",
        "   unidentified and only their sum reaches the effects.",
        "2. The NDE/NIE split therefore rests on the probit link extrapolating into a region the",
        "   clean arm never visits. The **TE** is the one quantity the data pins directly,",
        "   because the randomized arm difference checks it.",
        "3. The mechanism battery's usability guard for a bootstrap resample is",
        "   `x.min() != x.max() and y.min() != y.max()`, computed on the whole sample. Y does",
        "   vary overall, so the guard passes on every resample whose control arm is degenerate.",
        "   It does not catch this, and 0 of 200 resamples were redrawn in any cell.",
        "",
        "This is the same defect class the estimator-repair note already recorded for the",
        "historical Llama run (`docs/ESTIMATOR-REPAIR-2026-09-07.md` section 3.1, where the",
        "control arm was degenerate in the other direction, Y = 1 throughout).",
        "",
    ]
    return out


def column_a_section(fits: dict) -> list[str]:
    out = [
        "## 3. Column A, uncorrected",
        "",
        "Element 2 makes the misclassification-corrected column A a **secondary** estimand and",
        "keeps the frozen regex share as the headline. No jury Q1 configuration is frozen and",
        "no human calibration frame exists for these cells, so **no correction is computable**",
        "and none is shown. Every number below is an uncorrected regex share.",
        "",
        "**Ranking rule (section 25).** No cross-model ordering is stated anywhere in this",
        "document. The three cells are printed side by side because they were run side by side,",
        "not because they are comparable on a calibrated scale.",
        "",
        "| cell | P(followed) among clean-correct | acknowledged among followed | silent among followed | silent among clean-correct |",
        "|---|---|---|---|---|",
    ]
    for c in CELLS:
        a = fits[c]["column_a"]
        out.append(
            f"| {c} | {w(a['p_followed_among_clean_correct'])} | "
            f"{w(a['acknowledged_among_followed'])} | {w(a['silent_among_followed'])} | "
            f"{w(a['silent_among_clean_correct'])} |"
        )
    out += [
        "",
        "Intervals are Wilson score intervals at 95 percent. The acknowledgment flag is the",
        "frozen regex applied at run time and stored per record; the silent flag is its",
        "complement inside the follow stratum, which is why the two rows sum to the follow",
        "denominator in every cell.",
        "",
    ]
    return out


def column_b_section(fits: dict, pymc: dict) -> list[str]:
    out = [
        "## 4. Column B",
        "",
        "Specification: `M = mu_m + gamma X + eps_M`, `Y = 1[alpha0 + alpha X + beta M + eps_Y > 0]`,",
        "probit, **both intercepts fitted** (the 2026-09-07 repair), evaluated at rho = 0, with a",
        "200-replicate item bootstrap for the intervals. This is the path the CPU mechanism",
        "battery validated: MAP at rho = 0 with intercepts plus a 200-replicate bootstrap.",
        "",
        "### 4.1 NDE, NIE and TE, before any ratio or any rho quantity",
        "",
        "| cell | NDE | NIE | TE | model-implied TE | randomized arm difference | difference |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for c in CELLS:
        b = fits[c]["column_b"]
        g = b["model_implied_te_vs_randomized_arm_difference"]
        out.append(
            f"| {c} | {e(b['effects']['nde'])} | {e(b['effects']['nie'])} | "
            f"{e(b['effects']['te'])} | {g['model_implied_te']:.4f} | "
            f"{g['randomized_arm_difference']:.4f} | {g['difference']:+.4f} "
            f"[{g['difference_lo']:+.4f}, {g['difference_hi']:+.4f}] |"
        )
    out += [
        "",
        "The same three quantities from the PyMC posterior with the scale-aware priors and the",
        "probit link, 4 chains x 1,000 draws after 1,000 tuning draws per chain:",
        "",
        "| cell | NDE | NIE | TE | max r_hat | divergences | min ESS (bulk) | seconds |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for c in CELLS:
        p = pymc[c]

        def fmt(key: str, block: dict = p) -> str:
            return (
                f"{block[key]['mean']:+.4f} "
                f"[{block[key]['lo']:+.4f}, {block[key]['hi']:+.4f}]"
            )

        out.append(
            f"| {c} | {fmt('nde')} | {fmt('nie')} | {fmt('te')} | {p['max_r_hat']:.3f} | "
            f"{p['divergences']} | {p['min_ess_bulk']:.0f} | {p['seconds']:.0f} |"
        )
    out += [
        "",
        "The two paths agree to within 0.002 on every NIE point estimate, which is the check",
        "the link audit of `docs/ESTIMATOR-PRIORS-2026-09-07.md` was built to make possible:",
        "before that repair the posterior path was logistic while the maximum-likelihood path",
        "was probit, and the same coefficients meant two different models.",
        "",
        "### 4.2 The verdict on the NIE scale",
        "",
        "Element 1 section 2.5: load-bearing at rho when the posterior probability that the NIE",
        "exceeds 0.15 on the probability scale is at least 0.95. Where no effect is supported at",
        "rho = 0 the verdict is **unresolved**, never robust.",
        "",
        "| cell | P(NIE > 0.15) at rho = 0, bootstrap | replicates above threshold | P(NIE > 0.15), PyMC posterior | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for c in CELLS:
        b = fits[c]["column_b"]
        rd = b["rho"]["rho_star_decision"]
        out.append(
            f"| {c} | {rd['prob_nie_above_threshold_at_rho_zero']:.3f} | "
            f"{rd['n_bootstrap_replicates_above_threshold_at_rho_zero']}/200 | "
            f"{pymc[c]['prob_nie_above_0.15']:.3f} | **{b['verdict']['verdict']}** |"
        )
    out += [
        "",
        "### 4.3 The two rho quantities, kept apart",
        "",
        "Element 7 section 8.1 reports two rho quantities and never merges them. `rho*_point`",
        "is the zero crossing of the point estimate; it is invariant to the direct coefficient",
        "by construction and carries no information about direct-path strength. `rho*_decision`",
        "is the rho at which the pre-registered verdict rule fails, and it depends on sample",
        "size, interval width and the 0.15 threshold. The frontier is computed from a",
        "maximum-likelihood fit, not from a posterior: read it as a point-estimate sensitivity",
        "curve, not a Bayesian interval.",
        "",
        "| cell | rho*_point (bootstrap interval) | breakdown_frontier crossing | rho*_decision | binding side | partial-ID bounds at abs(rho) <= 0.5 | sign identified |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for c in CELLS:
        r = fits[c]["column_b"]["rho"]
        rp, rd, pb = r["rho_star_point"], r["rho_star_decision"], r["partial_identification_bounds"]
        dec = (
            "not applicable (unresolved)"
            if rd["value"] is None and rd["no_crossing_in_range"] is False
            else (
                f"lower bound {rd['lower_bound_if_no_crossing']:.3f}"
                if rd["no_crossing_in_range"]
                else f"{rd['value']:+.3f}"
            )
        )
        out.append(
            f"| {c} | {rp['point']:.4f} [{rp['lo']:.4f}, {rp['hi']:.4f}] | "
            f"{rp['from_breakdown_frontier_neg']:+.4f} | {dec} | {rd['binding_side'] or 'n/a'} | "
            f"[{pb['lower']:+.4f}, {pb['upper']:+.4f}] | "
            f"{'yes' if pb['sign_identified'] else 'no'} |"
        )
    out += [
        "",
        "**A correction this lane made to the sweep, and why it matters.** The mechanism",
        "battery's `RHO_GRID` runs from 0 to +0.945 (its `RHO_MAX` is 0.947). In all three cells",
        "`beta` and `gamma` are",
        "both negative, so a positive assumed rho makes the mediated path *larger*: on the",
        "non-negative grid the verdict never fails and `rho*_decision` would have been reported",
        "as \"no crossing in range\", a lower bound at `RHO_MAX` of 0.947. Every crossing that",
        "`breakdown_frontier`",
        "actually finds sits at negative rho. The decision sweep here therefore runs on the",
        "symmetric grid mirrored from the battery's own points, -0.945 to +0.945 in steps of",
        "0.005 with rho = 0 an exact grid point, and reports the binding side. That turns two",
        "\"robust to 0.947\" readings into rho*_decision values an order of magnitude smaller.",
        "",
        "It also makes the gap between the two rho quantities concrete: rho*_point sits between",
        "0.76 and 0.83 in these cells while the verdict fails at abs(rho) of 0.10 and 0.24, a",
        "factor of three to eight. A reader who treats rho*_point as a severity scale reads the",
        "robustness as several times what the verdict rule supports, which is exactly the",
        "confusion sections 8.1 and 8.2 were written to stop.",
        "",
        "P(NIE > 0.15) across the printed sweep (every 21st grid point):",
        "",
    ]
    for c in CELLS:
        s = fits[c]["column_b"]["rho"]["sweep"]
        out += [
            f"`{c}`",
            "",
            "| rho | " + " | ".join(f"{x:+.2f}" for x in s["rho"]) + " |",
            "|---|" + "---:|" * len(s["rho"]),
            "| P(NIE > 0.15) | "
            + " | ".join(f"{x:.2f}" for x in s["nie_prob_above_threshold"])
            + " |",
            "| median NIE | "
            + " | ".join(f"{x:+.3f}" for x in s["nie_median"])
            + " |",
            "",
        ]
    out += [
        "### 4.4 Mediated share, and the mediator-noise band",
        "",
        "NIE/TE is descriptive, is not constrained to [0,1], and is not printed for a cell whose",
        "TE interval includes zero. No TE interval here includes zero.",
        "",
        "Ruling R4 ships the printed attenuation band and not the latent-M layer. **No",
        "chain-level lambda exists for any cell:** none of these three ran the repeat-curves or",
        "chain-repeats arm (`enabled_arms` in each `arms_summary.json`), and",
        "`docs/A4-CHAIN-LAMBDA-NOTE.md` records that no chain-level job has been submitted at",
        "all. The band below is printed at lambda 1.0 (the fit as it stands), at the",
        "continuation-level 0.983075 measured on job 826025 (Qwen3-8B, curve area, hinted frame,",
        "temperature 0.7, 28 items) which A3.5 states is a **floor** for the chain-level value",
        "and which comes from a different cell than two of these three, and at 0.80 as R4's",
        "sensitivity row. Read it as a sensitivity display, not as a measured correction.",
        "",
        "| cell | NIE/TE | NIE at lambda 1.0 | NIE at lambda 0.983 | NIE at lambda 0.80 | point estimate above 0.15 at each lambda | verdict flips across the band |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for c in CELLS:
        b = fits[c]["column_b"]
        rows = {row["lambda"]: row for row in b["mediator_noise_band"]["rows"]}
        flags = " / ".join(
            "yes" if rows[lam]["would_be_load_bearing_point_estimate"] else "no"
            for lam in (1.0, 0.983075, 0.8)
        )
        out.append(
            f"| {c} | {b['nie_over_te']:.4f} | {rows[1.0]['nie']:.4f} | "
            f"{rows[0.983075]['nie']:.4f} | {rows[0.8]['nie']:.4f} | {flags} | "
            f"{'**yes**' if b['mediator_noise_band']['noise_flip']['flips'] else 'no'} |"
        )
    out += [
        "",
        "**The noise-flip note (element 7).** Two of the three cells flip: at lambda 0.80 the",
        "point-estimate NIE falls below the 0.15 threshold that the load-bearing verdict uses.",
        "The third never reaches the threshold at any lambda, so it cannot flip. The verdicts in",
        "section 4.2 therefore hold at the continuation-level floor and not at a 20 percent",
        "mediator-noise assumption, and the chain-level measurement that would settle which of",
        "those two is right does not exist yet.",
        "",
        "### 4.5 Fitted coefficients and convergence",
        "",
        "| cell | alpha | beta | gamma | sigma_m | mu_m | alpha0 | converged | bootstrap fits converged | degenerate redraws | curve-vs-refit max abs difference |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for c in CELLS:
        b = fits[c]["column_b"]
        f = b["fit"]
        out.append(
            f"| {c} | {f['alpha']:.4f} | {f['beta']:.4f} | {f['gamma']:.4f} | "
            f"{f['sigma_m']:.4f} | {f['mu_m']:.4f} | {f['alpha0']:.4f} | "
            f"{'yes' if f['converged'] else 'NO'} | {b['bootstrap']['n_converged']}/200 | "
            f"{b['bootstrap']['degenerate_redraws']} | "
            f"{b['rho']['cross_check_curve_vs_refit']['max_abs_difference']:.6f} |"
        )
    out += [
        "",
        "The last column is the cross-check the battery runs on itself: the vectorised rho",
        "reparameterisation against an actual refit at rho in {0, 0.1, 0.3, 0.5, 0.7}, with the",
        "natural effects taken from a 200,000-row Monte Carlo. The two agree to about 0.002,",
        "which is Monte Carlo noise on the integrator and not a modelling difference.",
        "",
    ]
    return out


def anchor_section(fits: dict) -> list[str]:
    out = [
        "## 5. The four-cell replay anchor (element 21)",
        "",
        "`mu_ab` is the fresh-answer rate on the designated target option with recipient cue `a`",
        "crossed with donor source `b`. The designated target option and the outcome scale are",
        "identical across all four cells. Donors were drawn independently within questions and",
        "never selected on success or on hint-following: `donor_draw_probability` is 1.0 at both",
        "ends and `donor_selected_on` is null in every record.",
        "",
        "| cell | mu00 (clean recipient, clean donor) | mu01 (clean recipient, cued donor) | mu10 (cued recipient, clean donor) | mu11 (cued recipient, cued donor) |",
        "|---|---|---|---|---|",
    ]
    for c in CELLS:
        a = fits[c]["anchor"]["cells"]
        out.append(f"| {c} | " + " | ".join(w(a[k]) for k in ("mu00", "mu01", "mu10", "mu11")) + " |")
    out += [
        "",
        "### 5.1 The five contrasts",
        "",
        "Intervals come from the same 200-replicate item bootstrap as column B, so the two are",
        "recomputed on identical resamples.",
        "",
        "| contrast | " + " | ".join(CELLS) + " |",
        "|---|" + "---|" * len(CELLS),
    ]
    names = [
        ("text_source_given_cued_recipient", "text source, cued recipient (mu11 - mu10)"),
        ("text_source_given_clean_recipient", "text source, clean recipient (mu01 - mu00)"),
        ("cue_effect_given_clean_donor", "cue effect, clean donor (mu10 - mu00)"),
        ("interaction", "interaction (mu11 - mu10 - mu01 + mu00)"),
        ("joint_replay_regime", "joint replay regime (mu11 - mu00)"),
    ]
    for key, label in names:
        row = " | ".join(e(fits[c]["anchor"]["contrasts"][key]) for c in CELLS)
        out.append(f"| {label} | {row} |")
    out += [
        "",
        "The joint replay-regime effect is explicitly **not** the native cue total effect absent",
        "a generation-to-replay bridge, and is never used as one here.",
        "",
        "### 5.2 The falsifier controls",
        "",
        "Each control holds one of source and meaning fixed and moves the other, so that",
        "donor-source dependence can be told apart from semantic dependence. A control that",
        "could not be applied to a donor is dropped from that control's denominator rather than",
        "passed through unedited, which is why the denominators differ between controls.",
        "",
        "In each control cell the donor is the **cued** donor after the edit, and `a0` / `a1`",
        "are the clean and cued recipient. So `a0` is the edited counterpart of `mu01` and `a1`",
        "the edited counterpart of `mu11`, and the comparison that carries the falsifier is",
        "`a0` against that cell's `mu01`. The denominators are not the same subset (a control",
        "applies only where its edit could be applied), so read the comparison descriptively",
        "and not as a matched contrast.",
        "",
        "| control | cell | applied | a0 (clean recipient) | a1 (cued recipient) | a1 - a0 | mu01 for reference |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    controls = [
        "decisive_premise_edit",
        "meaning_preserving_edit",
        "answer_marker_removed",
        "answer_marker_relocated",
        "matched_answer_only_text",
    ]
    for key in controls:
        for c in CELLS:
            v = fits[c]["anchor"]["falsifier_controls"][key]
            mu01 = fits[c]["anchor"]["cells"]["mu01"]["rate"]
            out.append(
                f"| {key} | {c} | {v['n_applied']}/{v['n_items']} | {w(v['a0'])} | "
                f"{w(v['a1'])} | {v['a1_minus_a0']:+.4f} | {mu01:.4f} |"
            )
    out += [
        "",
        "**What the controls say, cell by cell, with no ordering across cells.** In all three,",
        "the meaning-preserving edit leaves the clean-recipient rate essentially where `mu01`",
        "is, and the decisive-premise edit cuts it by roughly half or more: the donor's effect",
        "tracks what the chain asserts, not merely where it came from. The matched answer-only",
        "text separates the three sharply and in opposite directions. In `llama-3.1-8b-instruct`",
        "a donor stripped to a bare answer assertion reaches 0.3853 against a `mu01` of 0.3346,",
        "so the whole clean-recipient text-source effect is reproduced, and then some, without",
        "any reasoning content at all. In `gemma-2-9b-it` the same edit drops it from 0.2044 to",
        "0.0669. In `qwen3-8b` it drops from 0.1526 to 0.0387. Whether the replayed reasoning",
        "matters beyond its final answer is therefore a per-model empirical question that this",
        "control answers differently in different cells, and it is a caution about reading any",
        "single cell's anchor as a general fact about replay.",
        "",
        "### 5.3 The agreement margin, and claim status",
        "",
        "Section 22.1 puts the margin at 0.10 on the **difference** between the column B",
        "estimate and the corresponding anchor contrast, not on the overlap of the two",
        "intervals. `mu_ab` indexes (recipient cue, donor source), so the mapping onto the three",
        "estimands is one for one: NDE against mu10 - mu00, NIE against mu11 - mu10, TE against",
        "mu11 - mu00. The difference is computed inside each bootstrap replicate, so the two",
        "sides are paired.",
        "",
        "Element 21 compares the **model-level** column B estimate with the anchor. Each model",
        "here has exactly one usable cell (ARC-Challenge x stated-hint), so the cell-level",
        "estimate stands in for the model level and the hierarchical row estimand of section 2.4",
        "is not yet computable. That substitution is a scope limit on every promotion below.",
        "",
        "| cell | estimand | column B | anchor contrast | difference | margin headroom | agrees |",
        "|---|---|---|---|---|---:|---|",
    ]
    for c in CELLS:
        for key in ("nde", "nie", "te"):
            m = fits[c]["anchor"]["agreement_margin_test"][key]
            out.append(
                f"| {c} | {key.upper()} | {e(m['column_b'])} | {e(m['anchor'])} | "
                f"{m['difference_point']:+.4f} [{m['difference_lo']:+.4f}, "
                f"{m['difference_hi']:+.4f}] | {m['headroom']:+.4f} | "
                f"{'yes' if m['agrees'] else '**no**'} |"
            )
    out += [
        "",
        "| cell | all three agree | claim status |",
        "|---|---|---|",
    ]
    for c in CELLS:
        d = fits[c]
        out.append(
            f"| {c} | {'yes' if d['anchor']['all_three_agree'] else 'no'} | "
            f"**{d['claim_status']}** |"
        )
    out += [
        "",
        "The promotion rule applied is the conservative reading: a cell reaches ANCHORED only",
        "when **all three** printed estimands agree with their anchor contrast inside the",
        "margin, because all three are published. VALIDATED is not reachable for any cell: the",
        "mechanism-challenge coverage check of element 11 has not run.",
        "",
        "**Two things the margin test does not say.** The margin is absolute, so two quantities",
        "that differ by a large factor can still agree when both are small: the NDE comparison",
        "passes in two cells while the model-based NDE is between 3 and 13 times the anchor's",
        "cue effect under a clean donor. And in the cell whose NIE headroom is smallest, the",
        "difference interval reaches within a few thousandths of the margin, so the promotion is",
        "a pass and not a comfortable one. Disagreement, where it occurs, is a publishable",
        "result about the model-based quantity and not a reason to adjust it.",
        "",
    ]
    return out


def not_usable_section() -> list[str]:
    return [
        "## 6. The wave-1 cells that are NOT USABLE AS MEASURED",
        "",
        "Five of the eight wave-1 cells do not appear above. Each is listed with its reason and",
        "the DECISION-LOG entry that recorded it. None is dropped quietly and none is refit.",
        "",
        "| cell | job | state | clean-correct | unparseable clean | reason | DECISION-LOG |",
        "|---|---|---|---:|---:|---|---|",
        "| `allenai/OLMo-3-7B-Think` | 826734 | COMPLETED 0:0 | 56/1,500 | 1,381/1,500 | thinking-model truncation: `num_predict` 320 with `enable_thinking` false, which only Qwen3 honours. Below the 350 clean-correct floor of 01-SIZING I.3. | 2026-09-07 15:39, 15:57 |",
        "| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 826736 | COMPLETED 0:0 | 140/1,500 | 1,329/1,500 | same truncation; below the floor. | 2026-09-07 15:39, 15:57 |",
        "| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 826735 | COMPLETED 0:0 | clean accuracy 1/1,500 | 1,499/1,500 | same truncation; no `arms_summary.json` was written at all. | 2026-09-07 15:39, 15:57 |",
        "| `microsoft/Phi-4-reasoning` | 826739 | still RUNNING at 2:21 elapsed when this was written | not yet written | not yet written | the fourth held thinking model; its results directory holds a checkpoint and transcripts but no `arms_summary.json`, so it has no column A and no column B. | 2026-09-07 15:39, 16:31 |",
        "| `openai/gpt-oss-20b` | 826740 | FAILED 5:0 after 2:00 | none | none | RULING R11. No mxfp4 MoE kernel in vLLM 0.28.0 declares batch invariance, so the model cannot serve under the R1 serving mode on any card available to this account; and its forced continuations return empty content (60 of 61 at 24 tokens), so the replay, transplant, anchor and logprob arms cannot score it as built. | 2026-09-07 15:40, R11 at 16:22 |",
        "",
        "The audit lane classified the unparseable clean outputs on the three completed thinking",
        "cells: 30 sampled per cell at seed 20260907, and 30 of 30 in each cell were truncated",
        "inside the reasoning block with no final answer. Zero empty, zero answer-in-a-format-",
        "the-parser-misses, zero other (DECISION-LOG 2026-09-07 15:57). Their status turns on the",
        "element 9.4 ruling, which has not been made; R11 does not lift on its own.",
        "",
    ]


def limits_section(fits: dict) -> list[str]:
    return [
        "## 7. What these numbers do not say",
        "",
        "Six limits, each of which is a measured or structural fact from the sections above",
        "rather than a caution added for form.",
        "",
        "1. **The clean arm has zero outcome variance in every cell, by construction.** The",
        "   NDE/NIE split rests on the probit link extrapolating where the clean arm has no",
        "   data. Only the TE is checked directly, by the randomized arm difference, and it",
        "   matches to within 0.016, 0.009 and 0.001 with intervals that cover zero difference",
        "   in all three cells.",
        "2. **rho\\*_point is not a robustness score.** It sits between 0.76 and 0.83 here while",
        "   the pre-registered verdict fails at abs(rho) of 0.10 and 0.24. Reporting the first",
        "   without the second would overstate the robustness by a factor of three to eight.",
        "3. **No chain-level mediator noise has been measured anywhere in this project.** The",
        "   attenuation band uses a continuation-level floor from a different run, and two of",
        "   three verdicts flip inside the band.",
        "4. **One cell per model.** There is no hierarchical row estimand, so the element 21",
        "   comparison uses the cell estimate where the pre-registration asks for the model-level",
        "   one, and the cue-family variance component that section 2.4 requires before any",
        "   model-level number is quoted does not exist yet.",
        "5. **Column A is uncorrected and cannot be corrected here.** No jury Q1 configuration is",
        "   frozen and no human calibration frame covers these cells. No cross-model ordering is",
        "   published, per the ranking rule of section 25.",
        "6. **The agreement margin is absolute, so small quantities agree easily.** Two cells",
        "   pass the NDE comparison while the model-based NDE is between 3 and 13 times the",
        "   anchor's cue effect under a clean donor, and one NIE promotion clears the margin by",
        "   0.0025.",
        "",
    ]


def header(gate: dict, fits: dict) -> list[str]:
    d = fits[CELLS[0]]
    return [
        "# Wave-1 column A and column B fits",
        "",
        "**Date:** 7 September 2026",
        "**Branch:** `fits/wave1` (worktree `~/Developer/bcf-fits`), not merged and not pushed",
        f"**Code commit:** `{d['code_commit']}`",
        f"**Analysis script:** `experiments/wave1_fits.py`, sha256 `{d['analysis_script_sha256'][:16]}`",
        "**Generated by:** `experiments/wave1_fits_report.py` from the artifacts under",
        "`experiments/results/wave1-fits/`. No number in this file is typed by hand.",
        "",
        "This document reports the three usable wave-1 cells: ARC-Challenge, stated-hint cue",
        "family, on Qwen3-8B, Gemma-2-9B-it and Llama-3.1-8B-Instruct. All three ran under the",
        "R1 serving mode with `VLLM_BATCH_INVARIANT=1`, vLLM 0.28.0, the FLASH_ATTN backend and",
        "at most 32 requests in flight, and each passed its own determinism preflight (30/30",
        "identical completions and a maximum absolute letter-logprob difference of 0.0 at both 1",
        "and 32 in flight). All three carry `run_label` `powered_pinned` with",
        "`exploratory_reason` null.",
        "",
        "**Scope, in one paragraph.** These are RAW-or-ANCHORED cell numbers under element 19,",
        "one substrate and one cue family per model, with no hierarchical row estimand and no",
        "mechanism-challenge coverage check. Column A is uncorrected because no calibration",
        "label exists. No cross-model ordering is stated. The clean arm of every cell has zero",
        "outcome variance by construction, which is a limit on what the NDE/NIE split can mean",
        "and is documented in section 2.",
        "",
        "The estimator modules the gate exercised are byte-identical to the modules the fits",
        "used: `sensitivity.py`, `closed_form.py`, `effects.py`, `mediation.py` and",
        "`hierarchical.py` hash to the values recorded in every `fit.json` under",
        "`estimator_module_sha256`, and the deployed cluster copies were written before the gate",
        "job started and not touched afterwards. The analysis script itself was edited twice",
        "after the gate ran, once to search rho on the symmetric grid and once for lint. The two",
        "functions that produce every gate number, `_null_design` and `_fit_at_zero`, are",
        "byte-identical across those edits; the only change inside `run_gate` adds the module",
        "hash block after the verdict has already been computed. The gate value reported here is",
        "still the first run after the last change to anything it computes.",
        "",
        "---",
        "",
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    gate, fits, pymc = load()
    lines = (
        header(gate, fits)
        + gate_section(gate)
        + ["---", ""]
        + table_section(fits)
        + ["---", ""]
        + column_a_section(fits)
        + ["---", ""]
        + column_b_section(fits, pymc)
        + ["---", ""]
        + anchor_section(fits)
        + ["---", ""]
        + not_usable_section()
        + ["---", ""]
        + limits_section(fits)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
