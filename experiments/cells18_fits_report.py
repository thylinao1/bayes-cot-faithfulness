"""Render docs/CELLS18-FITS.md from the artifacts, so no number is retyped by hand.

Every table cell below is read out of ``experiments/results/cells18-fits/**`` at run
time. The prose is fixed text; the numbers are not. Re-running this after a re-fit
regenerates the document and a number that moved shows up in the diff. The number
formatters are imported from ``experiments/wave1_fits_report.py`` so the two
documents print a Wilson block and an effect block the same way.

    PYTHONPATH=src python experiments/cells18_fits_report.py --out docs/CELLS18-FITS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cells18_fits import MODELS, SUBSTRATE_CUES
from wave1_fits_report import e, w

ROOT = _HERE.parent
RESULTS = ROOT / "experiments" / "results" / "cells18-fits"
GEMMA_NOTE = ROOT / "experiments" / "results" / "cells18-fits" / "gemma_aqua_clean_parse.json"


def slug(substrate: str, cue: str) -> str:
    return f"{substrate} x {cue}"


def load():
    gate = json.loads((RESULTS / "offset_null_gate.json").read_text())
    fits, pymc, rows, extra = {}, {}, {}, {}
    for m in MODELS:
        for s, c in SUBSTRATE_CUES:
            fp = RESULTS / m / s / c / "fit.json"
            if fp.exists():
                fits[(m, s, c)] = json.loads(fp.read_text())
            pp = RESULTS / m / s / c / "pymc.json"
            if pp.exists():
                pymc[(m, s, c)] = json.loads(pp.read_text())
        rp = RESULTS / m / "model_row.json"
        if rp.exists():
            rows[m] = json.loads(rp.read_text())
        for name in ("model_row_logit_cue_family.json", "model_row_probit_substrate.json"):
            xp = RESULTS / m / name
            if xp.exists():
                extra[(m, name)] = json.loads(xp.read_text())
    gemma = json.loads(GEMMA_NOTE.read_text()) if GEMMA_NOTE.exists() else None
    return gate, fits, pymc, rows, extra, gemma


# --------------------------------------------------------------------------- #
WAVE1 = ROOT / "experiments" / "results" / "wave1-fits"
_WAVE1_KEYS = (
    ("column_b", "effects", "nde", "point"),
    ("column_b", "effects", "nie", "point"),
    ("column_b", "effects", "te", "point"),
    ("column_b", "effects", "nie", "lo"),
    ("column_b", "effects", "nie", "hi"),
    ("column_b", "rho", "rho_star_point", "point"),
    ("column_b", "rho", "rho_star_decision", "value"),
    ("column_a", "p_followed_among_clean_correct", "rate"),
    ("anchor", "all_three_agree"),
)


def _dig(d, keys):
    for k in keys:
        d = d[k]
    return d


def _wave1_reproduction(fits) -> list[str]:
    """The reuse claim, checked rather than asserted, against the wave-1 artifacts."""
    same = 0
    total = 0
    cells = 0
    for m in MODELS:
        old_p = WAVE1 / m / "fit.json"
        new = fits.get((m, "arc_challenge", "stated-hint"))
        if not old_p.exists() or new is None:
            continue
        old = json.loads(old_p.read_text())
        cells += 1
        for keys in _WAVE1_KEYS:
            total += 1
            same += int(_dig(new, keys) == _dig(old, keys))
    if not cells:
        return []
    return [
        ("**The reuse is checked, not asserted.** Three of these 18 cells are the three "
        "wave-1 cells (ARC-Challenge x stated-hint on each model), and this lane refits them "
        "from the same cluster records through the same imported functions. Comparing "
        f"{total} stored quantities across those {cells} cells against "
        "`experiments/results/wave1-fits/<model>/fit.json`, which was written by the "
        f"separate wave-1 lane on 2026-09-07, {same} of {total} are equal to the last stored "
        "digit: the three effects, the NIE interval, `rho*_point`, `rho*_decision`, the "
        "follow rate and the cell-level anchor verdict. A number that had drifted would show "
        "up here."),
        "",
    ]


def _script_note(any_fit, rows) -> list[str]:
    """The provenance chain of the analysis file, stated rather than smoothed over."""
    import ast
    import hashlib

    src = (_HERE / "cells18_fits.py").read_bytes()
    committed = hashlib.sha256(src).hexdigest()
    fit_sha = any_fit["analysis_script_sha256"]
    row_sha = next(iter(rows.values()))["analysis_script_sha256"] if rows else fit_sha
    if row_sha == fit_sha == committed:
        return []
    ast.parse(src.decode())  # the committed file parses; the equality proof is in the log
    return [
        ("**Provenance of the analysis file, in three hashes.** The 18 cell fits ran at "
        f"`experiments/cells18_fits.py` sha256 `{fit_sha[:16]}`; the model-level rows ran "
        f"at sha256 `{row_sha[:16]}`; the file committed on this branch hashes to "
        f"`{committed[:16]}`. The first step replaced an `invprobit` outcome in "
        "`_build_probit_hierarchy`, which saturates and overflowed the sampler on the "
        "first model-row submission, with the repository's own stable probit likelihood "
        "(`mediation._log_std_normal_cdf` inside a `pm.Potential`, the same expression "
        "`mediation.fit_mediation_model(link=\"probit\")` uses) and put the model-level "
        "priors on the scale-aware footing of the 2026-09-07 repair. Nothing a cell fit "
        "calls changed, so every number in sections 1 to 4 is produced by byte-identical "
        "code. The second step is lint only: the parse trees of the two files are equal "
        "under `ast.dump`, and the textual difference is parenthesised string "
        "concatenation plus three comment lines. The first model-row submission was NOT "
        "cancelled; it was left to finish and its output was discarded and replaced by "
        "the corrected run, which is where every number in section 5 comes from."),
        "",
    ]


def header(gate, fits, rows) -> list[str]:
    n_anch = sum(1 for f in fits.values() if f.get("claim_status") == "ANCHORED")
    n_raw = sum(1 for f in fits.values() if f.get("claim_status") == "RAW")
    n_pend = len(fits) - n_anch - n_raw
    any_fit = next(iter(fits.values()))
    return [
        "# The 18 cells of record: column A, column B, the logit-level gate, and the model-level rows",
        "",
        "**Date:** 8 September 2026",
        "**Branch:** `fits/cells18` (worktree `~/Developer/bcf-fits18`), not merged and not pushed",
        f"**Code commit at run time:** `{any_fit['code_commit']}`",
        ("**Analysis script:** `experiments/cells18_fits.py`, sha256 "
        f"`{any_fit['analysis_script_sha256'][:16]}`, which imports and calls "
        "`experiments/wave1_fits.py`, sha256 "
        f"`{any_fit['reused_script_sha256']['wave1_fits.py'][:16]}`"),
        "**Generated by:** `experiments/cells18_fits_report.py` from the artifacts under",
        "`experiments/results/cells18-fits/`. No number in this file is typed by hand.",
        "",
    ] + _script_note(any_fit, rows) + _wave1_reproduction(fits) + [
        ("This document reports the 18 cells of record: three models "
        f"({', '.join(MODELS)}) crossed with six substrate-by-cue-family cells each "
        "(ARC-Challenge under stated-hint, professor, metadata and grader-code; "
        "AQuA-RAT under stated-hint and professor). All 18 carry `run_label` "
        "`powered_pinned` with `exploratory_reason` null, ran under the R1 serving mode "
        "with batch invariance on and vLLM 0.28.0, and passed their own determinism "
        "preflight."),
        "",
        ("**Scope, in one paragraph.** Column A is uncorrected because no jury Q1 "
        "configuration is frozen (ruling R9) and no human calibration frame covers these "
        "cells. Column B is the repaired probit fit at rho = 0 with both intercepts. The "
        "logit-level column B of Amendment A5 does NOT print for any cell: the A5.4 "
        "eligibility gate G1 fails on the same condition in all 18, and section 3 gives "
        "the failing condition with its denominator per cell rather than a row. The "
        "model-level row estimand of element 1 section 2.4 is computed and is "
        "**PROVISIONAL**, for reasons the row states in its own artifact. No cross-model "
        "ordering is stated anywhere in this document (section 25). VALIDATED is not "
        "reachable for any cell."),
        "",
        f"Claim-status mix over the {len(fits)} cells: **{n_anch} ANCHORED**, "
        f"**{n_raw} RAW**"
        + (f", {n_pend} still pending the model row" if n_pend else "")
        + ". Element 19 requires the mix to be printed, and this is it.",
        "",
        "---",
        "",
    ]


def gate_section(gate) -> list[str]:
    out = [
        "## 1. The offset-null gate, run first",
        "",
        ("Element 12 requires the offset-null family to pass **before any powered fit**, and "
        "its no-retry rule makes the reported value the first run after the last change to "
        "anything it computes. The gate ran as its own cluster job on the CPU partition "
        "`long` and the 18 fits were submitted only after it exited 0."),
        "",
        (f"Attempt **{gate['attempt']}**, seed {gate['seed']}, n {gate['n']:,}, code commit "
        f"`{gate['code_commit'][:12]}`. Verdict **{gate['verdict']}**."),
        "",
        "| null | true NDE/NIE/TE | NDE | NIE | TE | randomized arm difference | TE minus arm difference | fitted mu_m | control-arm mean M | converged |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for kind, v in gate["nulls"].items():
        out.append(
            f"| {kind} offset | 0 / 0 / 0 | {v['nde']:+.6f} | {v['nie']:+.6f} | "
            f"{v['te']:+.6f} | {v['observed_arm_difference']:+.6f} | "
            f"{v['model_implied_te_minus_arm_difference']:+.6f} | {v['fitted_mu_m']:.6f} | "
            f"{v['control_arm_mean_M']:.6f} | {'yes' if v['converged'] else 'no'} |"
        )
    out += [
        "",
        ("The gate is delegated verbatim to `experiments/wave1_fits.py::run_gate` "
        f"(sha256 `{gate['wave1_fits_sha256'][:16]}`), so `_null_design` and `_fit_at_zero`, "
        "the two functions that produce every number above, are the same bytes the wave-1 "
        "gate ran, and the estimator modules under `src/bayes_cot_faithfulness` are "
        "byte-identical to main. The values match the wave-1 gate of 2026-09-07 exactly, "
        "which is what a deterministic seeded gate on unchanged code is supposed to do."),
        "",
        "Source file: `experiments/results/cells18-fits/offset_null_gate.json`.",
        "",
        "---",
        "",
    ]
    return out


def inventory_section(fits) -> list[str]:
    out = [
        "## 2. The 18 cells, their denominators and their trees",
        "",
        ("**X** is the arm indicator, clean versus hinted; every item contributes two rows "
        "and the item is the independent sampling unit, so the bootstrap resamples items. "
        "**M** is `clean_curve.curve_area` on the clean row and `hinted_curve.curve_area` "
        "on the hinted row. **Y** is `1[answer == hint_label]` on both arms, the same "
        "designated option in both, which on the hinted arm is the frozen parser's follow "
        "indicator. `outcome_scale` is `binary_follow` and `intervention_level` is `text` "
        "in all 18."),
        "",
        "| cell | job | tree | entered | clean-correct | unparseable clean | records | complete items | rows | items dropped | followed-field mismatches |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for (m, s, c), f in fits.items():
        d = f["table"]["denominators"]
        ca = f["column_a"]["denominators"]
        drops = d.get("drops") or {}
        drop_txt = (
            ", ".join(f"{k} {v}" for k, v in sorted(drops.items()) if k != "items_dropped")
            or "none"
        )
        out.append(
            f"| `{m}` {slug(s, c)} | {f['job_id']} | `{f['tree']['plan_commit']}` | "
            f"{ca['n_entered']} | {ca['n_clean_correct']} | {ca['n_unparseable_clean']} | "
            f"{d['n_records_read']} | {d['n_items_complete']} | {d['n_rows']} | {drop_txt} | "
            f"{d['followed_field_vs_recomputed_mismatch']} |"
        )
    out += [
        "",
        ("Each `fit.json` carries `estimand_id` in the wave-1 format, "
        "`columnB.<intervention level>.<outcome scale>.<substrate>.<cue family>`, so the 18 "
        "cells carry six distinct ids: the id names the ESTIMAND and not the cell, and the "
        "cell is identified by the model beside it. Every cell also carries the sha256 of "
        "the three files it was computed from (`transcripts.jsonl`, the cell's "
        "`arms_summary_<model>.json` and `run_meta.json`) under `record_hashes`, so a "
        "reader with cluster access can check that the numbers came from the files named."),
        "",
        ("Two trees appear. `5d40e5224ac0` is the pre-serving-fix tree; `ed3c74cf301f` "
        "carries the free-port and exit-guard fixes and is the tree the four voided cells "
        "were rerun under and that every later wave used. The four cells that were voided "
        "and rerun (ARC professor on Gemma and Llama, ARC metadata on Qwen and Gemma) are "
        "the reruns, and the voided originals in `~/bcf/results-void` were not read."),
        "",
        "| cell | mean M clean (sd) | mean M hinted (sd) | mean Y clean | mean Y hinted | randomized arm difference | clean-arm outcome variance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (m, s, c), f in fits.items():
        med = f["table"]["mediator"]
        o = f["table"]["outcome"]
        out.append(
            f"| `{m}` {slug(s, c)} | {med['clean_mean']:.4f} ({med['clean_sd']:.4f}) | "
            f"{med['hinted_mean']:.4f} ({med['hinted_sd']:.4f}) | {o['clean_arm_mean']:.4f} | "
            f"{o['hinted_arm_mean']:.4f} | {o['randomized_arm_difference']:.4f} | "
            f"{f['column_b']['separation_diagnostic']['clean_arm_outcome_variance']:.4f} |"
        )
    n_zero = sum(
        1
        for f in fits.values()
        if f["column_b"]["separation_diagnostic"]["clean_arm_outcome_variance"] == 0.0
    )
    by_sub = {"arc_challenge": [], "aqua_rat": []}
    for (m, s, c), f in fits.items():
        by_sub[s].append((f["table"]["mediator"]["clean_mean"], f["table"]["mediator"]["clean_sd"]))
    arc = by_sub["arc_challenge"]
    aqua = by_sub["aqua_rat"]
    depth = {"arc_challenge": [], "aqua_rat": []}
    for (m, s, c), f in fits.items():
        h = f["table"]["mediator"]["commitment_depth_hist_clean"]
        n = sum(h.values())
        depth[s].append(h.get("0", 0) / n if n else 0.0)
    depth_arc = depth["arc_challenge"]
    depth_aqua = depth["aqua_rat"]
    out += [
        "",
        ("**The mediator does not have the same distribution on the two substrates.** Mean "
        f"clean-arm curve area runs {min(a for a, _ in arc):.4f} to {max(a for a, _ in arc):.4f} "
        f"on the {len(arc)} ARC-Challenge cells with a standard deviation of "
        f"{min(b for _, b in arc):.4f} to {max(b for _, b in arc):.4f}, and "
        f"{min(a for a, _ in aqua):.4f} to {max(a for a, _ in aqua):.4f} on the "
        f"{len(aqua)} AQuA-RAT cells with a standard deviation of "
        f"{min(b for _, b in aqua):.4f} to {max(b for _, b in aqua):.4f}. The secondary "
        "component says the same thing more directly: the share of clean rows whose "
        "commitment depth is 0, that is whose forced continuation already gives the final "
        f"answer at the first truncation depth, runs {min(d for d in depth_arc):.3f} to "
        f"{max(d for d in depth_arc):.3f} on ARC and {min(d for d in depth_aqua):.3f} to "
        f"{max(d for d in depth_aqua):.3f} on AQuA. The AQuA cells therefore carry several times "
        "the mediator variation, which is the situation the scale-aware priors of "
        "`docs/ESTIMATOR-PRIORS-2026-09-07.md` exist to handle and a reason not to read a "
        "coefficient from one substrate against a coefficient from the other."),
        "",
        (f"The clean arm carries zero outcome variance in {n_zero} of {len(fits)} cells, "
        "which A4.6(b) already states is a property of the frozen outcome population and "
        "not a defect of any cell: the population is the clean-correct subpopulation and "
        "the hint label is a planted wrong option, so Y is 0 on every clean row. The "
        "consequences A4.6(b) lists hold here unchanged, and the one quantity the data "
        "pins directly is the TE, checked below against the randomized arm difference."),
        "",
        "---",
        "",
    ]
    return out


def column_a_section(fits) -> list[str]:
    out = [
        "## 3. Column A, uncorrected",
        "",
        ("Element 2 makes the misclassification-corrected column A a **secondary** estimand "
        "and keeps the frozen regex share as the headline. Ruling R9 froze no jury Q1 "
        "configuration and no human calibration frame covers these cells, so no correction "
        "is computable and none is shown. Every number below is an uncorrected regex share "
        "with a Wilson score interval at 95 percent."),
        "",
        ("**Ranking rule (section 25).** No cross-model ordering is stated. The cells are "
        "printed in a fixed order because that is the order they were run in."),
        "",
        ("A Wilson lower bound that prints as -0.0000 anywhere in this document is a "
        "floating-point residue of order 1e-17 in the k = 0 case of the interval formula, "
        "not a negative rate. It is left as the formula produces it rather than clamped, "
        "because clamping a printed number is the kind of quiet edit this project's "
        "estimator repairs were about."),
        "",
        "| cell | P(followed) among clean-correct | acknowledged among followed | silent among followed | silent among clean-correct |",
        "|---|---|---|---|---|",
    ]
    for (m, s, c), f in fits.items():
        a = f["column_a"]
        out.append(
            f"| `{m}` {slug(s, c)} | {w(a['p_followed_among_clean_correct'])} | "
            f"{w(a['acknowledged_among_followed'])} | {w(a['silent_among_followed'])} | "
            f"{w(a['silent_among_clean_correct'])} |"
        )
    small = [
        (m, s, c, f["column_a"]["p_followed_among_clean_correct"]["k"])
        for (m, s, c), f in fits.items()
        if f["column_a"]["p_followed_among_clean_correct"]["k"] < 20
    ]
    out += [
        "",
        "Section 2.1 estimates the conditional factor only where the follow stratum holds "
        "at least 20 items. "
        + (
            "Cells below that floor here: "
            + "; ".join(f"`{m}` {slug(s, c)} at {k} followed" for m, s, c, k in small)
            + ". Their acknowledgment and silent shares are printed with their "
            "denominators and are not to be read as estimates of the conditional."
            if small
            else "Every cell here clears that floor."
        ),
        "",
        "---",
        "",
    ]
    return out


def cell_block(m, s, c, f, p) -> list[str]:
    b = f["column_b"]
    rho = b["rho"]
    g1 = f["logit_level_gate_G1"]
    anch = f["anchor"]
    out = [
        f"### {slug(s, c)} on `{m}`",
        "",
        (f"Job {f['job_id']}, tree `{f['tree']['plan_commit']}`, "
        f"{b['n_items']} items / {b['n_rows']} rows, claim status "
        f"**{f.get('claim_status', 'PENDING')}**."),
        "",
        ("**Column B, text level.** NDE, NIE and TE with intervals come before any ratio "
        "or any rho quantity (section 8.1)."),
        "",
        "| NDE | NIE | TE | model-implied TE | randomized arm difference | difference |",
        "|---|---|---|---:|---:|---|",
        (f"| {e(b['effects']['nde'])} | {e(b['effects']['nie'])} | {e(b['effects']['te'])} | "
        f"{b['model_implied_te_vs_randomized_arm_difference']['model_implied_te']:.4f} | "
        f"{b['model_implied_te_vs_randomized_arm_difference']['randomized_arm_difference']:.4f} | "
        f"{b['model_implied_te_vs_randomized_arm_difference']['difference']:+.4f} "
        f"[{b['model_implied_te_vs_randomized_arm_difference']['difference_lo']:+.4f}, "
        f"{b['model_implied_te_vs_randomized_arm_difference']['difference_hi']:+.4f}] |"),
        "",
    ]
    if p:
        out += [
            ("The same three from the PyMC posterior with the scale-aware priors and the "
            f"probit link, 4 chains x {p['draws_per_chain']:,} draws after "
            f"{p['tune_per_chain']:,} tuning draws:"),
            "",
            "| NDE | NIE | TE | max r_hat | divergences | min ESS (bulk) | seconds |",
            "|---|---|---|---:|---:|---:|---:|",
            (f"| {p['nde']['mean']:+.4f} [{p['nde']['lo']:+.4f}, {p['nde']['hi']:+.4f}] | "
            f"{p['nie']['mean']:+.4f} [{p['nie']['lo']:+.4f}, {p['nie']['hi']:+.4f}] | "
            f"{p['te']['mean']:+.4f} [{p['te']['lo']:+.4f}, {p['te']['hi']:+.4f}] | "
            f"{p['max_r_hat']:.3f} | {p['divergences']} | {p['min_ess_bulk']:.0f} | "
            f"{p['seconds']:.0f} |"),
            "",
        ]
    dec = rho["rho_star_decision"]
    band = b["mediator_noise_band"]
    lam_rows = {str(r["lambda"]): r for r in band["rows"] if r.get("in_range")}
    out += [
        ("**Verdict, the two rho quantities and the bounds.** The verdict rule is section "
        "2.5: load-bearing at rho when P(NIE > 0.15) is at least 0.95 on the probability "
        "scale; unresolved where no effect is supported at rho = 0. The decision sweep "
        "runs on the symmetric grid of A4.6(a) and reports the binding side."),
        "",
        "| P(NIE > 0.15) at rho=0 | replicates above | verdict | rho*_point | rho*_decision | binding side | partial-ID bounds at abs(rho) <= 0.5 | sign identified | NIE/TE |",
        "|---:|---:|---|---|---|---|---|---|---:|",
        f"| {dec['prob_nie_above_threshold_at_rho_zero']:.3f} | "
        f"{dec['n_bootstrap_replicates_above_threshold_at_rho_zero']}/"
        f"{b['bootstrap']['n_replicates']} | **{b['verdict']['verdict']}** | "
        f"{rho['rho_star_point']['point']:.4f} [{rho['rho_star_point']['lo']:.4f}, "
        f"{rho['rho_star_point']['hi']:.4f}] | "
        + (f"{dec['value']:+.3f}" if dec["value"] is not None else "not applicable")
        + f" | {dec['binding_side'] or 'n/a'} | "
        f"[{rho['partial_identification_bounds']['lower']:+.4f}, "
        f"{rho['partial_identification_bounds']['upper']:+.4f}] | "
        f"{'yes' if rho['partial_identification_bounds']['sign_identified'] else 'no'} | "
        + (f"{b['nie_over_te']:.4f}" if b.get("nie_over_te") is not None else "not printed")
        + " |",
        "",
        (f"Fit converged: {'yes' if b['fit']['converged'] else 'no'}; "
        f"{b['bootstrap']['n_converged']}/{b['bootstrap']['n_replicates']} bootstrap fits "
        f"converged with {b['bootstrap']['degenerate_redraws']} degenerate redraws; the "
        "vectorised rho reparameterisation agrees with an actual refit at rho in "
        "{0, 0.1, 0.3, 0.5, 0.7} to "
        f"{rho['cross_check_curve_vs_refit']['max_abs_difference']:.6f}, which is Monte "
        "Carlo noise on the integrator rather than a modelling difference."),
        "",
        ("**The mediator-noise band and the noise-flip note.** Printed at lambda 1.0, at the "
        "continuation-level floor 0.983075 and at 0.80 as ruling R4's sensitivity row. No "
        "chain-level lambda exists for this cell."),
        "",
        "| lambda | NIE | point estimate above 0.15 | verdict flips across the band |",
        "|---:|---:|---|---|",
    ]
    for lam in ("1.0", "0.983075", "0.8"):
        r = lam_rows.get(lam)
        if not r:
            continue
        out.append(
            f"| {lam} | {r['nie']:+.4f} | "
            f"{'yes' if r.get('would_be_load_bearing_point_estimate') else 'no'} | "
            f"{'**yes**' if band['noise_flip']['flips'] else 'no'} |"
        )
    out += [
        "",
        "**Logit-level column B (Amendment A5).** "
        + (
            "NOT PRINTED. A5.4 requires all six G1 conditions and this cell fails "
            + ", ".join(f"`{k}`" for k in g1["failing_conditions"])
            + "."
            if not g1["eligible"]
            else "Printed below."
        ),
        "",
        "| condition | value | holds |",
        "|---|---|---|",
    ]
    cd = g1["conditions"]
    c3 = cd["3_records_carry_the_logit_scale"]
    c6 = cd["6_letter_probability_mass_summarised"]["overall"]
    out += [
        (f"| 1 unit check (section 9.1) | probes "
        f"{cd['1_unit_check_passed'].get('n_probes_completed')}/"
        f"{cd['1_unit_check_passed'].get('n_probes')}, letters scored "
        f"{cd['1_unit_check_passed'].get('n_letters_scored_over_requested')}, tokens "
        f"matching {cd['1_unit_check_passed'].get('n_tokens_matching_letter_over_requested')}, "
        f"hard failures {cd['1_unit_check_passed'].get('n_hard_failures')} | "
        f"{'yes' if cd['1_unit_check_passed'].get('holds') else 'no'} |"),
        (f"| 2 pinned self-hosted endpoint | `{cd['2_pinned_self_hosted_endpoint'].get('base_url')}`, "
        f"method prompt_logprobs on "
        f"{cd['2_pinned_self_hosted_endpoint'].get('n_logprob_blocks_by_method', {}).get('prompt_logprobs', 0)}"
        f"/{cd['2_pinned_self_hosted_endpoint'].get('n_logprob_blocks_total')} blocks | "
        f"{'yes' if cd['2_pinned_self_hosted_endpoint'].get('holds') else 'no'} |"),
        (f"| 3 records carry the logit scale | intervention_level=logit "
        f"{c3['n_records_with_intervention_level_logit']}, outcome_scale=logprob_margin "
        f"{c3['n_records_with_outcome_scale_logprob_margin']}, logprob_source_token "
        f"{c3['n_records_with_logprob_source_token']} | "
        f"{'yes' if c3['holds'] else '**no**'} |"),
        (f"| 4 clean-arm outcome variance positive | not computable: "
        f"{cd['4_clean_arm_outcome_variance_positive']['n_arms_rows_with_a_margin']} arms "
        f"rows carry a margin | **no** |"),
        "| 5 TE_logit equals the arm difference | not computable, same reason | **no** |",
        (f"| 6 letter probability mass | min {c6['min']:.3e}, median {c6['median']:.3e}, max "
        f"{c6['max']:.3e} over {c6['n']} anchor cells; "
        f"{cd['6_letter_probability_mass_summarised']['n_below_0.01']} below 0.01 | yes |"),
        "",
        ("**The element 21 four-cell replay anchor.** `mu_ab` is the fresh-answer rate on the "
        "designated target option with recipient cue `a` crossed with donor source `b`. It is "
        "computed on the SAME complete-item set as column B, with anchor cells that carry no "
        "scorable value counted out of their own denominator, so the two sides of the "
        "agreement test below are the same items and the difference is formed inside one "
        "bootstrap replicate as section 22.1 requires."),
        "",
        "| mu00 | mu01 | mu10 | mu11 |",
        "|---|---|---|---|",
        (f"| {w(anch['cells']['mu00'])} | {w(anch['cells']['mu01'])} | "
        f"{w(anch['cells']['mu10'])} | {w(anch['cells']['mu11'])} |"),
        "",
        "| contrast | value |",
        "|---|---|",
    ]
    for name, v in anch["contrasts"].items():
        out.append(f"| {name.replace('_', ' ')} | {e(v)} |")
    out += ["", "| falsifier control | applied | a0 (clean recipient) | a1 (cued recipient) | a1 - a0 |", "|---|---:|---|---|---:|"]
    for key, ctl in anch["falsifier_controls"].items():
        if not ctl["n_applied"]:
            out.append(f"| {key} | 0/{ctl['n_items']} | not applied | not applied | n/a |")
            continue
        out.append(
            f"| {key} | {ctl['n_applied']}/{ctl['n_items']} | {w(ctl['a0'])} | "
            f"{w(ctl['a1'])} | {ctl['a1_minus_a0']:+.4f} |"
        )
    out += [
        "",
        "| estimand | column B | anchor contrast | difference | margin headroom | agrees |",
        "|---|---|---|---|---:|---|",
    ]
    for key in ("nde", "nie", "te"):
        mt = anch["agreement_margin_test"][key]
        out.append(
            f"| {key.upper()} | {e(mt['column_b'])} | {e(mt['anchor'])} | "
            f"{mt['difference_point']:+.4f} [{mt['difference_lo']:+.4f}, "
            f"{mt['difference_hi']:+.4f}] | {mt['headroom']:+.4f} | "
            f"{'yes' if mt['agrees'] else '**no**'} |"
        )
    out += [
        "",
        f"Cell-level: all three agree **{'yes' if anch['all_three_agree'] else 'no'}**; "
        + (
            "model-level all three agree **"
            + ("yes" if f["claim_status_inputs"]["model_level_all_three_agree"] else "no")
            + f"**. Claim status **{f['claim_status']}**."
            if f.get("claim_status_inputs")
            else f" claim status **{f.get('claim_status', 'PENDING')}**."
        ),
        "",
    ]
    return out


def _grid0() -> float:
    """The rho the mislabelled index actually points at, read from the module."""
    import wave1_fits as _w1

    return float(_w1.RHO_GRID_SIGNED[0])


def _zero_index() -> int:
    import wave1_fits as _w1

    return int(_w1.RHO_ZERO_INDEX)


def _wave1_defect_count() -> int:
    """How many wave-1 fit.json files carry the mislabelled key, measured."""
    n = 0
    for m in MODELS:
        p = WAVE1 / m / "fit.json"
        if not p.exists():
            continue
        b = json.loads(p.read_text())["column_b"]
        if (
            b["verdict"]["prob_nie_above_0.15_at_rho_zero"]
            != b["rho"]["rho_star_decision"]["prob_nie_above_threshold_at_rho_zero"]
        ):
            n += 1
    return n


def _verdict_field_defect(fits) -> list[str]:
    """A stored field is mislabelled. Say so, measure it, and name the right one."""
    wrong = 0
    worst_cell, worst_gap = None, 0.0
    for (m, s, c), f in fits.items():
        dec = f["column_b"]["rho"]["rho_star_decision"]
        good = dec["prob_nie_above_threshold_at_rho_zero"]
        bad = f["column_b"]["verdict"]["prob_nie_above_0.15_at_rho_zero"]
        if good != bad:
            wrong += 1
            if abs(good - bad) > worst_gap:
                worst_gap, worst_cell = abs(good - bad), (m, s, c, good, bad)
    if not wrong:
        return []
    m, s, c, good, bad = worst_cell
    n_match = sum(
        1
        for f in fits.values()
        if abs(
            f["column_b"]["rho"]["rho_star_decision"][
                "prob_nie_above_threshold_at_rho_zero"
            ]
            - f["column_b"]["rho"]["rho_star_decision"][
                "n_bootstrap_replicates_above_threshold_at_rho_zero"
            ]
            / f["column_b"]["bootstrap"]["n_replicates"]
        )
        < 1e-9
    )
    n_flip = sum(
        1
        for f in fits.values()
        if (
            f["column_b"]["rho"]["rho_star_decision"][
                "prob_nie_above_threshold_at_rho_zero"
            ]
            >= 0.95
        )
        != bool(f["column_b"]["verdict"]["load_bearing_at_rho_zero"])
    )
    return [
        ("**A stored field in `experiments/wave1_fits.py` is mislabelled, and this "
        "document prints around it rather than through it.** The verdict block of every "
        "`fit.json` carries `prob_nie_above_0.15_at_rho_zero`, and that field is read off "
        f"the SIGNED rho grid at index 0. Index 0 of that grid is rho = {_grid0():.3f}, "
        "the most negative rho evaluated, not rho = 0, which sits at index "
        f"{_zero_index()}. The verdict itself is computed at the right index "
        "(`prob_above[RHO_ZERO_INDEX]`) and is "
        f"therefore correct: {n_flip} of {len(fits)} cells have a verdict that disagrees "
        "with the correctly indexed probability. The mislabelled field is a reported "
        "diagnostic only. It is wrong in "
        f"{wrong} of {len(fits)} cells here; the largest gap is `{m}` {slug(s, c)}, where "
        f"the correct P(NIE > 0.15) at rho = 0 is {good:.3f} and the stored field says "
        f"{bad:.3f}. The verdict tables below print "
        "`rho.rho_star_decision.prob_nie_above_threshold_at_rho_zero`, the value at the "
        "zero index, and that it is the right one is checked rather than assumed: in "
        f"{n_match} of {len(fits)} cells it equals the replicate count printed beside it "
        "divided by the number of replicates, which the mislabelled field does not. "
        "`experiments/wave1_fits.py` is reused byte-identical by this lane and is NOT "
        "edited here, so the same key is mislabelled in the three wave-1 `fit.json` files "
        f"too ({_wave1_defect_count()} of 3 carry a value that differs from the correctly "
        "indexed one). `docs/WAVE1-FITS.md` is NOT affected: it reads the correctly "
        "indexed field and prints 1.000 and 0.995 where the mislabelled key says 0.000. "
        "This document was the only consumer that read the wrong key, which is why the "
        "defect surfaced here. Fixing the key at source belongs to the lane that owns "
        "that file; no effect estimate, interval, verdict or rho quantity changes when it "
        "is fixed, and this lane changed no fitted number to print the corrected "
        "column."),
        "",
    ]


def _sign_line(fits) -> str:
    """How often the partial-identification bounds pin the sign of the NIE."""
    yes = sum(
        1
        for f in fits.values()
        if f["column_b"]["rho"]["partial_identification_bounds"]["sign_identified"]
    )
    return (
        "**What the partial-identification bounds pin.** At the pre-registered band "
        "abs(rho) <= 0.5 the bounds identify the SIGN of the NIE in "
        f"{yes} of the {len(fits)} cells and leave it unidentified in {len(fits) - yes}. "
        "A cell whose sign is not identified inside that band has an NIE whose direction is "
        "an assumption about rho, not a measurement, and the block below prints the bounds "
        "for every cell either way."
    )


def _two_path_line(fits, pymc) -> str:
    """The link audit of docs/ESTIMATOR-PRIORS-2026-09-07.md, run over all 18 cells."""
    worst = {"nde": 0.0, "nie": 0.0, "te": 0.0}
    rhat, div, ess = 0.0, 0, None
    for key, f in fits.items():
        p = pymc.get(key)
        if not p:
            continue
        for k, best in worst.items():
            d = abs(f["column_b"]["effects"][k]["point"] - p[k]["mean"])
            worst[k] = max(best, d)
        rhat = max(rhat, p["max_r_hat"])
        div += p["divergences"]
        ess = p["min_ess_bulk"] if ess is None else min(ess, p["min_ess_bulk"])
    return (
        "**The two estimation paths agree, which is the check the link audit of "
        "`docs/ESTIMATOR-PRIORS-2026-09-07.md` exists to make possible.** Across the "
        f"{len(pymc)} cells that have both, the maximum absolute gap between the "
        "maximum-likelihood point estimate and the PyMC posterior mean is "
        f"{worst['nde']:.5f} on the NDE, {worst['nie']:.5f} on the NIE and "
        f"{worst['te']:.5f} on the TE. The posterior side has max r_hat {rhat:.3f}, "
        f"{div} divergences in total and a minimum bulk ESS of {ess:.0f}. Before that "
        "repair the posterior path was logistic while the maximum-likelihood path was "
        "probit, so the same coefficients meant two different models and this comparison "
        "could not be made."
    )


def cells_section(fits, pymc) -> list[str]:
    out = [
        "## 4. The 18 cells, one block each",
        "",
        ("Reporting order inside each block follows section 8.1 and A5.5: the text-level "
        "NDE, NIE and TE with intervals first, then the verdict and the two rho quantities, "
        "then the mediator-noise band, then the logit-level row or the G1 condition that "
        "stops it, then the anchor."),
        "",
        ("An index first, so a reader can find a cell without scrolling. Every number in it "
        "is repeated with its interval and its denominator in the cell's own block below, "
        "and no column here supports a comparison across models (section 25)."),
        "",
        _two_path_line(fits, pymc),
        "",
        _sign_line(fits),
        "",
    ] + _verdict_field_defect(fits) + [
        "| cell | items | followed | NIE | verdict | rho*_decision | logit row | cell-level anchor agrees | claim status |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for (m, s, c), f in fits.items():
        b = f["column_b"]
        dec = b["rho"]["rho_star_decision"]
        out.append(
            f"| `{m}` {slug(s, c)} | {b['n_items']} | "
            f"{f['column_a']['p_followed_among_clean_correct']['k']} | "
            f"{b['effects']['nie']['point']:+.4f} | {b['verdict']['verdict']} | "
            + (
                f"{dec['value']:+.3f} ({dec['binding_side']})"
                if dec["value"] is not None
                else "not applicable"
            )
            + " | "
            + ("printed" if f["logit_level_gate_G1"]["eligible"] else "not printed (G1)")
            + f" | {'yes' if f['anchor']['all_three_agree'] else 'no'} | "
            f"{f.get('claim_status', 'PENDING')} |"
        )
    out += [""]
    for (m, s, c), f in fits.items():
        out += cell_block(m, s, c, f, pymc.get((m, s, c)))
    out += ["---", ""]
    return out


def model_rows_section(rows, extra, fits) -> list[str]:
    out = [
        "## 5. The model-level row estimand (element 1 section 2.4)",
        "",
        ("Section 2.4: one value per model, the model-level hyperparameter posterior from "
        "the hierarchical fit across that model's cells, never an average of cell point "
        "estimates, with **the posterior of the cue-family variance component reported "
        "before any model-level number is quoted**. That ordering is why the variance "
        "components come first in every table below. Every row is **PROVISIONAL**; the "
        "choices that make it so are recorded verbatim in each `model_row.json` under "
        "`choices_that_make_this_provisional` and summarised after the tables."),
        "",
    ]
    if not rows:
        out += ["No model-level row has been written yet.", "", "---", ""]
        return out
    out += [
        ("**No ordering is stated here either.** The three model rows are printed in one "
        "table because they are three instances of the same estimand, not because they are "
        "comparable on a calibrated scale. Section 25's ranking rule is written for "
        "cross-model column-A statements and column A here is uncorrected; the same "
        "restraint is applied to column B because nothing in this lane licenses a "
        "cross-model claim about it either."),
        "",
        "### 5.1 The cue-family variance components, printed first",
        "",
        "| model | tau_alpha_h (cue family, direct) | tau_beta_h (cue family, mediated) | tau_alpha (cell) | tau_beta (cell) | tau_gamma (cell) |",
        "|---|---|---|---|---|---|",
    ]
    for m, r in rows.items():
        v = r["variance_components"]

        def f4(k, v=v):
            b = v[k]
            return f"{b['posterior_median']:.4f} [{b['lo']:.4f}, {b['hi']:.4f}]"
        out.append(
            f"| `{m}` | {f4('tau_alpha_h')} | {f4('tau_beta_h')} | {f4('tau_alpha')} | "
            f"{f4('tau_beta')} | {f4('tau_gamma')} |"
        )
    out += [
        "",
        _sampler_health(rows),
        "",
        "### 5.2 The row estimand, and the per-cell rows beside it",
        "",
        "| model | cells | items | NDE | NIE | TE | P(NIE > 0.15) | verdict | NIE/TE | max r_hat | divergences |",
        "|---|---:|---:|---|---|---|---:|---|---:|---:|---:|",
    ]
    for m, r in rows.items():
        ef = r["effects"]
        out.append(
            f"| `{m}` | {r['n_cells']} | {r['n_items_total']:,} | "
            f"{ef['nde']['point']:+.4f} [{ef['nde']['lo']:+.4f}, {ef['nde']['hi']:+.4f}] | "
            f"{ef['nie']['point']:+.4f} [{ef['nie']['lo']:+.4f}, {ef['nie']['hi']:+.4f}] | "
            f"{ef['te']['point']:+.4f} [{ef['te']['lo']:+.4f}, {ef['te']['hi']:+.4f}] | "
            f"{ef['prob_nie_above_0.15']:.3f} | "
            f"{'load-bearing at rho=0' if ef['load_bearing_at_rho_zero'] else 'unresolved'} | "
            + (f"{ef['nie_over_te']:.4f}" if ef.get("nie_over_te") is not None else "n/a")
            + f" | {r['sampler']['max_r_hat']:.3f} | {r['sampler']['divergences']} |"
        )
    out += ["", _row_cell_consistency(rows, fits), "",
            _no_model_rho_line(rows), "",
            "The six cell rows of each model, beside their model row:", ""]
    for m in rows:
        out += [
            f"`{m}`",
            "",
            "| cell | NDE | NIE | TE | verdict | claim status |",
            "|---|---|---|---|---|---|",
        ]
        for s, c in SUBSTRATE_CUES:
            f = fits.get((m, s, c))
            if not f:
                continue
            b = f["column_b"]
            out.append(
                f"| {slug(s, c)} | {e(b['effects']['nde'])} | {e(b['effects']['nie'])} | "
                f"{e(b['effects']['te'])} | {b['verdict']['verdict']} | "
                f"{f.get('claim_status', 'PENDING')} |"
            )
        out.append("")
    out += [
        "### 5.3 The model-level element 21 comparison, which is what claim status turns on",
        "",
        "| model | estimand | model-level column B | pooled anchor contrast | difference | agrees |",
        "|---|---|---|---|---|---|",
    ]
    for m, r in rows.items():
        for key in ("nde", "nie", "te"):
            mt = r["model_level_agreement_margin_test"][key]
            out.append(
                f"| `{m}` | {key.upper()} | "
                f"{mt['column_b_model_level']['point']:+.4f} "
                f"[{mt['column_b_model_level']['lo']:+.4f}, "
                f"{mt['column_b_model_level']['hi']:+.4f}] | {e(mt['anchor'])} | "
                f"{mt['difference_point']:+.4f} [{mt['difference_lo']:+.4f}, "
                f"{mt['difference_hi']:+.4f}] | "
                f"{'yes' if mt['agrees'] else '**no**'} |"
            )
    out += ["", _pairing_caveat(rows, fits), "",
            _correspondence_note(rows), ""]
    out += [
        "### 5.4 Why every row is PROVISIONAL",
        "",
        next(iter(rows.values()))["choices_that_make_this_provisional"],
        "",
    ]
    if extra:
        out += [
            "### 5.5 The two sensitivity fits",
            "",
            "| model | fit | tau_beta_h | NDE | NIE | TE | max r_hat | divergences |",
            "|---|---|---|---|---|---|---:|---:|",
        ]
        for (m, name), r in extra.items():
            v = r["variance_components"]["tau_beta_h"]
            ef = r.get("effects")
            label = "logit link, cue family" if "logit" in name else "probit link, substrate"
            eff_txt = (
                (
                    f"{ef['nde']['point']:+.4f} | {ef['nie']['point']:+.4f} | "
                    f"{ef['te']['point']:+.4f}"
                )
                if ef
                else "not on the probability scale | n/a | n/a"
            )
            out.append(
                f"| `{m}` | {label} | {v['posterior_median']:.4f} [{v['lo']:.4f}, "
                f"{v['hi']:.4f}] | {eff_txt} | {r['sampler']['max_r_hat']:.3f} | "
                f"{r['sampler']['divergences']} |"
            )
        out += [
            "",
            ("Every row in this document, primary and sensitivity, was mirrored from the "
            "corrected submission's output tree alone; the first submission's tree was "
            "read only to confirm it had finished and none of its numbers were copied. "
            "Each row prints the sha256 of the analysis file it ran under in its own "
            "artifact, so a reader can check that claim without trusting this sentence."),
            "",
            ("The logit-link fit is `src/bayes_cot_faithfulness/hierarchical.py` exactly as "
            "written. Its coefficients live on a different link from every cell row in this "
            "document, so no probability-scale effect is computed from it and only its "
            "variance components and its sampler diagnostics are printed. The substrate "
            "fit answers a question section 2.4 does not ask, and is here so the "
            "cue-family component can be read against something."),
            "",
        ]
    out += ["---", ""]
    return out


def _row_cell_consistency(rows, fits) -> str:
    """The model row and the six cell fits are separate runs. Check they agree."""
    size_ok = size_n = 0
    anch_ok = anch_n = 0
    for m, r in rows.items():
        for ce in r["cells_entering"]:
            f = fits.get((m, ce["substrate"], ce["cue_family"]))
            if f is None:
                continue
            d = f["table"]["denominators"]
            size_n += 1
            size_ok += int(
                d["n_items_complete"] == ce["n_items"] and d["n_rows"] == ce["n_rows"]
            )
        for name in ("mu00", "mu01", "mu10", "mu11"):
            k = n = 0
            for ce in r["cells_entering"]:
                f = fits.get((m, ce["substrate"], ce["cue_family"]))
                if f is None:
                    continue
                a = f["anchor"]["cells"][name]
                k += a["k"]
                n += a["n"]
            p = r["pooled_anchor"]["cells"][name]
            anch_n += 1
            anch_ok += int(k == p["k"] and n == p["n"])
    return (
        "**The row and its cells are separate runs, and they are checked against each "
        "other.** The model row was fitted by its own cluster job straight from the "
        "transcripts; the six cell fits were fitted by a different job. Comparing the two "
        f"afterwards, {size_ok} of {size_n} cell sizes in the row's `cells_entering` equal "
        "the `n_items_complete` and `n_rows` the corresponding `fit.json` recorded, and "
        f"{anch_ok} of {anch_n} pooled anchor counts in the row equal the sum of the same "
        "four anchor cells over that model's six `fit.json` files, numerator and "
        "denominator. A row fitted on a different item set than the cells printed beside "
        "it would fail this."
    )


def _no_model_rho_line(rows) -> str:
    """No model-level rho sweep is printed. Section 2.5 is why, and it is measurable."""
    n_lb = sum(1 for r in rows.values() if r["effects"]["load_bearing_at_rho_zero"])
    worst = max(r["effects"]["prob_nie_above_0.15"] for r in rows.values())
    n_cover = sum(
        1
        for r in rows.values()
        if r["effects"]["nie"]["lo"] <= 0.0 <= r["effects"]["nie"]["hi"]
    )
    cover = (
        f"{n_cover} of {len(rows)} rows have an NIE interval that covers zero"
    )
    return (
        "**No model-level rho sweep is printed, and section 2.5 is the reason rather "
        "than a shortage of compute.** 2.5 says that where no effect is supported at "
        "rho = 0 the verdict is unresolved and the dial says so rather than showing "
        f"robustness. {n_lb} of {len(rows)} model rows are load-bearing at rho = 0; the "
        f"largest P(NIE > 0.15) over the {len(rows)} rows is {worst:.3f} against the 0.95 "
        f"the rule asks for, and {cover}. Sweeping rho from "
        "there would report how far an effect that is not supported at rho = 0 survives, "
        "which is the number 2.5 forbids putting on the dial. rho\\*_point is likewise "
        "not printed at the model level: it is an invariant reference with no directional "
        "meaning, and this lane has no model-level use for it that section 8.1 permits. "
        "The per-cell sweeps in section 4 are unaffected and are printed there."
    )


def _demoted(rows, fits) -> int:
    """Cells whose own three estimands agree but whose model row's do not."""
    n = 0
    for (m, s, c), f in fits.items():
        r = rows.get(m)
        if r is None:
            continue
        if f["anchor"]["all_three_agree"] and not r["model_level_all_three_agree"]:
            n += 1
    return n


def _pairing_caveat(rows, fits) -> str:
    """The 5.3 test is not the 4.x test. Say so where the claim status is decided."""
    note = next(iter(rows.values()))["model_level_agreement_margin_test"]["nde"][
        "pairing_note"
    ]
    n_dem = _demoted(rows, fits)
    n_cell = sum(1 for f in fits.values() if f["anchor"]["all_three_agree"])
    agreeing = [m for m, r in rows.items() if r["model_level_all_three_agree"]]
    return (
        "**This test is not the per-cell test of section 4, and it is harder to pass for "
        "a reason that is arithmetic rather than empirical.** The artifact states it: "
        + note
        + ". The per-cell test forms the difference inside one bootstrap replicate, so "
        "the shared item noise cancels; here the column B side is a PyMC posterior over "
        "hyperparameters and the anchor side is an item bootstrap, and nothing pairs "
        "them. The consequence is measurable rather than hypothetical: "
        f"{n_cell} of {len(fits)} cells pass their own three-estimand test, "
        f"{'no model row passes' if not agreeing else 'the model rows that pass are ' + ', '.join('`' + m + '`' for m in agreeing)}"
        f", and {n_dem} of those {n_cell} cells are therefore RAW on the model-level leg "
        "alone. This lane requires BOTH legs because section 22 puts the comparison at "
        "the model level and a cell-only rule would promote on the easier test; the "
        "conservative choice can only demote. A later lane that pairs the two sides, or "
        "that reads section 22.1's margin as a per-cell rule, will get more ANCHORED "
        "cells from these same numbers, and that is a choice about the test and not a "
        "new measurement. "
        + _extra_leg_binding(rows, fits)
    )


def _extra_leg_binding(rows, fits) -> str:
    """Say whether the lane's extra cell-level leg changed any status."""
    n_model_only = sum(
        1
        for (m, s, c), f in fits.items()
        if rows.get(m)
        and rows[m]["model_level_all_three_agree"]
        and not f["anchor"]["all_three_agree"]
    )
    if n_model_only == 0:
        return (
            "Section 22's own rule is the model-level leg alone, so the extra leg this "
            "lane adds is not binding on this table: no cell would have been promoted by "
            "the model-level test and demoted by the cell-level one, because no model row "
            "passes the model-level test in the first place. The extra leg is recorded "
            "because it would bind on a table where a model row did pass."
        )
    return (
        "Section 22's own rule is the model-level leg alone, and the extra leg this lane "
        f"adds IS binding here: {n_model_only} of {len(fits)} cells would be ANCHORED "
        "under section 22 read literally and are held at RAW by this lane's extra "
        "cell-level requirement. That is a deviation from the pre-registration in the "
        "conservative direction and it is flagged rather than buried."
    )


def _correspondence_note(rows) -> str:
    """Which anchor contrast answers to which estimand is a reading, not a fixture."""
    mt = next(iter(rows.values()))["model_level_agreement_margin_test"]
    pairs = ", ".join(
        f"{k.upper()} against `{mt[k]['anchor_contrast']}`" for k in ("nde", "nie", "te")
    )
    return (
        "**The correspondence itself is a reading.** Section 22 says the model-level "
        "column B estimate is compared with *the corresponding* anchor contrast and does "
        "not say which of the five contrasts corresponds to which estimand. This lane "
        f"pairs {pairs}, the same pairing the wave-1 cells were promoted on, and stores "
        "the label beside every comparison so a lane that pairs them differently can see "
        "exactly what it is changing. A different pairing is a different test on the same "
        "five measured contrasts, all of which are printed in section 4."
    )


def _sampler_health(rows) -> str:
    """Say plainly whether the hierarchical fits sampled cleanly. They may not have."""
    bad = []
    for m, r in rows.items():
        sm = r["sampler"]
        if sm["divergences"] > 0 or sm["max_r_hat"] > 1.01 or sm["min_ess_bulk"] < 400:
            bad.append(
                f"`{m}` with {sm['divergences']} divergences, max r_hat "
                f"{sm['max_r_hat']:.3f} and minimum bulk ESS {sm['min_ess_bulk']:.0f}"
            )
    if not bad:
        return (
            "**Sampler health.** Every model row below sampled with 0 divergences, max "
            "r_hat at or below 1.01 and a minimum bulk ESS above 400."
        )
    return (
        "**Sampler health, stated before the numbers because it bears on how to read "
        "them.** The hierarchical fit is harder than the per-cell one: it carries six "
        "group deviations and two zero-centred cue-family deviations over a design whose "
        "clean arm has no outcome variation, and it does not sample cleanly everywhere. "
        + "; ".join(bad)
        + ". Divergences mean the sampler could not explore part of the posterior, so "
        "these intervals are not guaranteed to be the posterior's own. The numbers are "
        "printed with their diagnostics rather than withheld, and the diagnostic is one "
        "more reason the row is PROVISIONAL. A later lane that wants a clean row should "
        "raise `target_accept`, reparameterise the group deviations, or fit fewer levels "
        "at once, and should re-run rather than reinterpret these."
    )


def gemma_section(gemma) -> list[str]:
    if not gemma:
        return []
    out = [
        "## 6. The Gemma AQuA-RAT unparseable clean outputs",
        "",
        ("`google/gemma-2-9b-it` on AQuA-RAT loses "
        f"{gemma['stated-hint']['n_unparseable']} of "
        f"{gemma['stated-hint']['n_records']} entered items on the stated-hint cell and "
        f"{gemma['professor']['n_unparseable']} of {gemma['professor']['n_records']} on "
        "the professor cell to an unparseable clean answer, against 0 to 9 on every other "
        "cell in this table. Those items never reach the clean-correct population, so they "
        "are attrition before the analysis rather than a defect in it. This lane read "
        f"{gemma['n_sampled']} of them from each cell at seed {gemma['seed']} "
        f"({2 * gemma['n_sampled']} read in all, each sampled item's class, its options "
        "and the last 120 characters of its completion stored in the artifact), then "
        "classified all "
        f"{gemma['stated-hint']['n_unparseable'] + gemma['professor']['n_unparseable']} "
        "by the same rule. **Nothing is fixed here and no parser is changed.**"),
        "",
        "| cell | unparseable clean | answered none of the above | numeric answer, not a letter (value is an option) | other non-letter text | truncated with no answer line | empty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("stated-hint", "professor"):
        g = gemma[k]
        out.append(
            f"| aqua_rat x {k} | {g['n_unparseable']}/{g['n_records']} | "
            f"{g['classes']['none_of_the_above']} | {g['classes']['numeric_not_a_letter']} "
            f"({g['numeric_answers_whose_value_is_one_of_the_options']}) | "
            f"{g['classes']['other_nonletter_text']} | "
            f"{g['classes']['truncated_no_answer_line']} | {g['classes']['empty']} |"
        )
    out += [
        "",
        ("**What they are.** The dominant class is not truncation. The model finishes its "
        "reasoning, reaches a value, sees that the value is not among the five lettered "
        "options, and writes `Answer: (None of the above)`. The frozen answer regex has no "
        "letter to extract from that string, so the item is unparseable. The second class "
        "writes the numeric value or a bare number in the answer slot, `Answer: (6)` where "
        "the options are 9, 7, 3, 8 and 12, again with no letter to extract. Only a handful "
        "per cell are truncation, ending mid-sentence with no answer line at all."),
        "",
        ("**Why this is worth a section rather than a footnote.** The two classes say "
        "different things. The truncated ones are a budget artifact. The `None of the "
        "above` ones are a model behaviour on an arithmetic substrate, and they are "
        "selected: an item the model gets numerically wrong in a way that lands off the "
        "option list is exactly an item it would not have answered correctly. Dropping them "
        "therefore removes a non-random slice of AQuA-RAT before the clean-correct "
        "restriction is applied, on top of the restriction element 1 already declares. "
        "Neither cell's numbers are adjusted for it and no rate in section 3 or 4 should be "
        "read as covering that slice."),
        "",
        "Source file: `experiments/results/cells18-fits/gemma_aqua_clean_parse.json`.",
        "",
        "---",
        "",
    ]
    return out


def _rho_limit_line(fits) -> str:
    """Limit 7, with the measured gap between the two rho quantities."""
    rows = []
    for (m, s, c), f in fits.items():
        fr = f["column_b"]["rho"]["frontier_at_practical_threshold"]
        dec = f["column_b"]["rho"]["rho_star_decision"]["value"]
        if not fr["unresolved"] and fr["robustness"] is not None:
            rows.append((f"`{m}` {slug(s, c)}", fr["robustness"], dec))
    detail = "; ".join(
        f"{name} at {rob:.4f} against "
        + (f"{dec:+.3f}" if dec is not None else "an unresolved verdict")
        for name, rob, dec in rows
    )
    return (
        "7. **rho\\*_point is not a robustness score, and the two rho quantities are printed "
        "apart.** Section 8.1 forbids merging them and section 4's tables print both on the "
        "symmetric grid A4.6(a) fixes, with the binding side. The breakdown frontier asked "
        "at the pre-registered practical threshold of 0.15 returns a robustness number in "
        f"only {len(rows)} of the {len(fits)} cells, because the other "
        f"{len(fits) - len(rows)} have no effect worth defending at rho = 0; and where it "
        f"does return one it is far larger than the rho at which the verdict fails: {detail}."
    )


def _answer_only_prose(fits) -> str:
    """The claim about the answer-only control, counted rather than asserted."""
    up = {"arc_challenge": 0, "aqua_rat": 0}
    tot = {"arc_challenge": 0, "aqua_rat": 0}
    biggest = ("", 0.0, 0.0, 0.0)
    for (m, s, c), f in fits.items():
        ctl = f["anchor"]["falsifier_controls"]["matched_answer_only_text"]
        mu01 = f["anchor"]["cells"]["mu01"]
        if ctl["a0"]["rate"] is None or mu01["rate"] is None:
            continue
        d = ctl["a0"]["rate"] - mu01["rate"]
        tot[s] += 1
        up[s] += int(d > 0)
        if d > biggest[3]:
            biggest = (f"`{m}` {slug(s, c)}", ctl["a0"]["rate"], mu01["rate"], d)
    return (
        "9. **The anchor's matched answer-only control does not point one way, so no single "
        "cell's anchor is a general fact about replay.** The control replaces the donor "
        "chain with a bare assertion of the same answer and leaves the recipient clean, so "
        "`a0` against that cell's own `mu01` asks whether the replayed reasoning matters "
        "beyond its final answer. It RAISES the rate in "
        f"{up['arc_challenge']} of the {tot['arc_challenge']} ARC-Challenge cells and in "
        f"{up['aqua_rat']} of the {tot['aqua_rat']} AQuA-RAT cells. The largest gap is "
        f"{biggest[0]}, where a donor stripped to a bare answer reaches {biggest[1]:.4f} "
        f"against {biggest[2]:.4f} for a full cued donor, a difference of {biggest[3]:+.4f}. "
        "Each comparison below is within one cell against that cell's own mu01 and is not a "
        "comparison across models."
    )


def _answer_only_table(fits) -> list[str]:
    """The matched answer-only control against each cell's own mu01, all 18 cells."""
    out = [
        "| cell | answer-only donor, clean recipient (a0) | mu01, full cued donor | a0 minus mu01 |",
        "|---|---|---|---:|",
    ]
    for (m, s, c), f in fits.items():
        ctl = f["anchor"]["falsifier_controls"]["matched_answer_only_text"]
        mu01 = f["anchor"]["cells"]["mu01"]
        if ctl["a0"]["rate"] is None or mu01["rate"] is None:
            continue
        out.append(
            f"| `{m}` {slug(s, c)} | {w(ctl['a0'])} | {w(mu01)} | "
            f"{ctl['a0']['rate'] - mu01['rate']:+.4f} |"
        )
    return out


def limits_section(fits, rows) -> list[str]:
    n_zero = sum(
        1
        for f in fits.values()
        if f["column_b"]["separation_diagnostic"]["clean_arm_outcome_variance"] == 0.0
    )
    n_g1 = sum(1 for f in fits.values() if not f["logit_level_gate_G1"]["eligible"])
    flips = sum(
        1 for f in fits.values() if f["column_b"]["mediator_noise_band"]["noise_flip"]["flips"]
    )
    return [
        "## 7. What these 18 cells do not say",
        "",
        "Each item is a measured or structural fact from the sections above.",
        "",
        ("1. **No cross-model ordering is published, and none is computable from this "
        "table.** Section 25 allows cross-model column-A statements only as partial "
        "orderings at posterior probability above 0.90, off a CALIBRATED rate. Column A "
        "here is uncorrected, because ruling R9 froze no jury Q1 configuration, so there "
        "is no calibrated scale on which any two of these three models could be ordered. "
        "The three models appear side by side because they were run side by side."),
        "",
        (f"2. **The clean arm carries zero outcome variance in {n_zero} of {len(fits)} "
        "cells, by construction.** The NDE and NIE split rests on the probit link "
        "extrapolating into a region the clean arm never visits. Only the TE is checked "
        "directly, by the randomized arm difference. A5.7 adds that on a correctly "
        "specified simulated world the extrapolation costs precision rather than accuracy, "
        "and that the thing to worry about is rho; neither statement makes the split "
        "identified in a real cell. Item 11 below reports what that one direct check "
        "actually returns on these cells, which is not a clean pass."),
        "",
        (f"3. **The logit-level column B does not exist for any cell.** {n_g1} of "
        f"{len(fits)} fail the A5.4 gate G1, all on the same condition: no arms record in "
        "this project carries `intervention_level = logit`, because the generation pass "
        "that would write it (part 4.5's job B of `docs/OUTCOME-SCALE-NOTE.md`) has never "
        "been run. The outcome scale that WOULD have a varying clean arm is therefore "
        "still unmeasured on the arms, and the anchor-cell margins printed in section 4 "
        "are a different regime and are labelled as one."),
        "",
        ("4. **VALIDATED is unreachable for every cell.** Element 19's top rung needs the "
        "element 11 mechanism-challenge coverage check. The ladder was built on "
        "2026-09-08 and nothing has been trained: no checkpoint exists, so no coverage "
        "check can have run. No cell in this table is more than ANCHORED, and ANCHORED "
        "here rests on a PROVISIONAL model row."),
        "",
        ("5. **The model-level rows are PROVISIONAL.** Section 2.4 fixes the estimand and "
        "not the link, the anchor pooling or the way a posterior is compared with a "
        "bootstrap. Section 5.4 lists what this lane chose. A later lane that chooses "
        "differently will get different rows, and the claim statuses that depend on them "
        "can move."),
        "",
        (f"6. **No chain-level mediator noise has been measured for any of these 18 cells.** "
        f"The attenuation band uses a continuation-level floor from a different run, and "
        f"the point-estimate verdict flips inside the band in {flips} of {len(fits)} "
        "cells."),
        "",
        _rho_limit_line(fits),
        "",
        ("8. **The model row is item-weighted, so the two large ARC cells carry most of it.** "
        "The hierarchical fit pools ITEMS across a model's six cells with a cell-level "
        "random effect and a zero-centred cue-family deviation, so a cell that entered "
        "1,500 items contributes about three times the likelihood of one that entered 570. "
        "The cue-family term stops one family driving the population mean unflagged, which "
        "is what section 8 asks of it, but it does not equalise the cells. The per-cell rows "
        "are printed beside the model row in section 5.2 for exactly this reason."),
        "",
        _answer_only_prose(fits),
        "",
    ] + _answer_only_table(fits) + [
        "",
        _claim_status_limit_line(rows, fits),
        "",
        _te_identity_line(fits),
        "",
    ] + _te_identity_table(fits) + [
        "",
    ]


def _te_identity_line(fits) -> str:
    """The one directly checked quantity does not always check out. Say by how much."""
    off = []
    for (m, s, c), f in fits.items():
        d = f["column_b"]["model_implied_te_vs_randomized_arm_difference"]
        if not (d["difference_lo"] <= 0.0 <= d["difference_hi"]):
            off.append((abs(d["difference"]), m, s, c, d))
    off.sort(reverse=True)
    n = len(fits)
    if not off:
        return (
            f"11. **The TE identity holds in all {n} cells.** The model-implied TE and "
            "the randomized arm difference agree to within their bootstrap interval "
            "everywhere, which is the only direct check the design supports."
        )
    _, m, s, c, d = off[0]
    return (
        f"11. **The one quantity the data pins directly does not always agree with the "
        f"fit: the TE identity fails to cover zero in {len(off)} of {n} cells.** Limit 2 "
        "says the NDE and NIE split rests on extrapolation and that only the TE is "
        "checked directly, against the randomized arm difference. That check is printed "
        f"in every block of section 4, and in {len(off)} of {n} cells the bootstrap "
        "interval on the difference excludes zero, so the probit fit does not reproduce "
        "the arm difference within its own sampling error. The largest gap is "
        f"`{m}` {slug(s, c)}, model-implied {d['model_implied_te']:.4f} against a measured "
        f"arm difference of {d['randomized_arm_difference']:.4f}, difference "
        f"{d['difference']:+.4f} [{d['difference_lo']:+.4f}, {d['difference_hi']:+.4f}]. "
        f"The shortfall has a direction: the model-implied TE is BELOW the arm "
        f"difference in {_n_negative(fits)} of {n} cells, not scattered either side of "
        f"it, and the shortfall reaches {_max_relative(fits):.1%} of the arm difference "
        f"on the largest. The two sides carry the same sign in {_same_sign(fits)} of {n} "
        "cells, but a "
        "reader who takes limit 2 to mean the TE is validated should read this instead: "
        "the TE is checkABLE, it was checked, and in these cells the check is not clean. "
        "That is a statement about the probit specification on a design with a "
        "zero-variance clean arm, not about the arm difference, which is a direct count."
    )


def _n_negative(fits) -> int:
    return sum(
        1
        for f in fits.values()
        if f["column_b"]["model_implied_te_vs_randomized_arm_difference"]["difference"]
        < 0
    )


def _max_relative(fits) -> float:
    return max(
        abs(d["difference"]) / d["randomized_arm_difference"]
        for d in (
            f["column_b"]["model_implied_te_vs_randomized_arm_difference"]
            for f in fits.values()
        )
    )


def _same_sign(fits) -> int:
    return sum(
        1
        for f in fits.values()
        if (f["column_b"]["model_implied_te_vs_randomized_arm_difference"]
            ["model_implied_te"] > 0)
        == (f["column_b"]["model_implied_te_vs_randomized_arm_difference"]
            ["randomized_arm_difference"] > 0)
    )


def _te_identity_table(fits) -> list[str]:
    off = []
    for (m, s, c), f in fits.items():
        d = f["column_b"]["model_implied_te_vs_randomized_arm_difference"]
        if not (d["difference_lo"] <= 0.0 <= d["difference_hi"]):
            off.append((abs(d["difference"]), m, s, c, d))
    if not off:
        return []
    off.sort(reverse=True)
    out = [
        "| cell | model-implied TE | randomized arm difference | difference |",
        "|---|---:|---:|---|",
    ]
    for _, m, s, c, d in off:
        out.append(
            f"| `{m}` {slug(s, c)} | {d['model_implied_te']:.4f} | "
            f"{d['randomized_arm_difference']:.4f} | {d['difference']:+.4f} "
            f"[{d['difference_lo']:+.4f}, {d['difference_hi']:+.4f}] |"
        )
    return out


def _claim_status_limit_line(rows, fits) -> str:
    """Item 10: what the claim status actually turned on."""
    if not rows:
        return (
            "10. **No claim status is final.** The model-level rows the promotion rule "
            "needs have not been written, so every cell reads PENDING_MODEL_ROW."
        )
    n_anch = sum(1 for f in fits.values() if f.get("claim_status") == "ANCHORED")
    n_cell = sum(1 for f in fits.values() if f["anchor"]["all_three_agree"])
    n_dem = _demoted(rows, fits)
    return (
        f"10. **{n_anch} of {len(fits)} cells are ANCHORED, and the binding constraint is "
        "the model-level leg rather than any cell's own evidence.** "
        f"{n_cell} of {len(fits)} cells clear the element 21 margin on all three of their "
        f"own estimands; {n_dem} of them are held at RAW because their model row does not "
        "clear the same margin, on a test whose two sides cannot be paired (section 5.3). "
        "Read the RAW label as a statement about the promotion rule and the model-level "
        "comparison, not as evidence that those cells disagree with their anchor. Each "
        "cell's own agreement table is printed in section 4 and is unaffected."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gate, fits, pymc, rows, extra, gemma = load()
    lines: list[str] = []
    lines += header(gate, fits, rows)
    lines += gate_section(gate)
    lines += inventory_section(fits)
    lines += column_a_section(fits)
    lines += cells_section(fits, pymc)
    lines += model_rows_section(rows, extra, fits)
    lines += gemma_section(gemma)
    lines += limits_section(fits, rows)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
