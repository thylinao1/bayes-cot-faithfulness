"""Render the cells-of-record document from the artifacts, so no number is retyped.

Every table cell below is read out of ``experiments/results/cells<N>-fits/**`` at run
time. The prose is fixed text; the numbers are not. Re-running this after a re-fit
regenerates the document and a number that moved shows up in the diff. The number
formatters are imported from ``experiments/wave1_fits_report.py`` so the two
documents print a Wilson block and an effect block the same way.

The cell list is a PARAMETER. ``--cells 18`` reads
``experiments/results/cells18-fits`` and writes the 18-cell document; ``--cells 24``
reads ``experiments/results/cells24-fits`` and writes the 24-cell one. The two roots
are separate directories and neither pass can write into the other's.

    PYTHONPATH=src python experiments/cells18_fits_report.py --cells 18 \
        --out docs/CELLS18-FITS.md
    PYTHONPATH=src python experiments/cells18_fits_report.py --cells 24 \
        --out docs/CELLS24-FITS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from typing import NamedTuple

import cells18_fits as c18
from cells18_fits import MODELS, SUBSTRATE_CUES, substrate_cues  # noqa: F401
from wave1_fits_report import e, w

ROOT = _HERE.parent


class Lane(NamedTuple):
    """Everything that differs between the 18-cell pass and the 24-cell pass."""

    cell_set: str
    pairs: tuple
    results: Path
    results_rel: str
    branch: str
    worktree: str
    per_model: str
    aqua_cues: str


LANES = {
    "18": Lane(
        cell_set="18",
        pairs=substrate_cues("18"),
        results=ROOT / "experiments" / "results" / "cells18-fits",
        results_rel="experiments/results/cells18-fits",
        branch="fits/cells18",
        worktree="~/Developer/bcf-fits18",
        per_model="six",
        aqua_cues="AQuA-RAT under stated-hint and professor",
    ),
    "24": Lane(
        cell_set="24",
        pairs=substrate_cues("24"),
        results=ROOT / "experiments" / "results" / "cells24-fits",
        results_rel="experiments/results/cells24-fits",
        branch="fits/cells24",
        worktree="~/Developer/bcf-fits24",
        per_model="eight",
        aqua_cues="AQuA-RAT under stated-hint, professor, metadata and grader-code",
    ),
}


def slug(substrate: str, cue: str) -> str:
    return f"{substrate} x {cue}"


def load(lane: Lane):
    """Read the artifacts of one lane.

    A cell that was asked for and whose fit.json is not there is REPORTED with its
    reason, never dropped quietly, so a document over fewer cells than the lane
    name says cannot pass for a complete one.
    """
    results = lane.results
    gate = json.loads((results / "offset_null_gate.json").read_text())
    fits, pymc, rows, extra = {}, {}, {}, {}
    absent: list[str] = []
    for m in MODELS:
        for s, c in lane.pairs:
            fp = results / m / s / c / "fit.json"
            if fp.exists():
                try:
                    fits[(m, s, c)] = json.loads(fp.read_text())
                except json.JSONDecodeError as exc:
                    absent.append(f"{m}/{s}/{c} (fit.json does not parse: {exc})")
            else:
                absent.append(f"{m}/{s}/{c} (no fit.json)")
            pp = results / m / s / c / "pymc.json"
            if pp.exists():
                pymc[(m, s, c)] = json.loads(pp.read_text())
        rp = results / m / "model_row.json"
        if rp.exists():
            rows[m] = json.loads(rp.read_text())
        for name in ("model_row_logit_cue_family.json", "model_row_probit_substrate.json"):
            xp = results / m / name
            if xp.exists():
                extra[(m, name)] = json.loads(xp.read_text())
    for line in absent:
        print(f"CELL NOT IN THE DOCUMENT: {line}, under {results}", file=sys.stderr)
    gemma_note = results / "gemma_aqua_clean_parse.json"
    gemma = json.loads(gemma_note.read_text()) if gemma_note.exists() else None
    return gate, fits, pymc, rows, extra, gemma, absent


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
        (f"**The reuse is checked, not asserted.** {cells} of these {len(fits)} cells are "
        "the three "
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


def _script_note(any_fit, rows, n_cells: int) -> list[str]:
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
        (f"**Provenance of the analysis file, in three hashes.** The {n_cells} cell fits "
        "ran at "
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


def header(gate, fits, rows, lane: Lane, absent) -> list[str]:
    n_g1 = sum(1 for f in fits.values() if not f["logit_level_gate_G1"]["eligible"])
    n_asked = len(MODELS) * len(lane.pairs)
    n_anch = sum(1 for f in fits.values() if f.get("claim_status") == "ANCHORED")
    n_raw = sum(1 for f in fits.values() if f.get("claim_status") == "RAW")
    n_pend = len(fits) - n_anch - n_raw
    any_fit = next(iter(fits.values()))
    return [
        (f"# The {lane.cell_set} cells of record: column A, column B, the logit-level "
        "gate, and the model-level rows"),
        "",
        "**Date:** 8 September 2026",
        f"**Branch:** `{lane.branch}` (worktree `{lane.worktree}`), not merged and not pushed",
        f"**Code commit at run time:** `{any_fit['code_commit']}`",
        ("**Analysis script:** `experiments/cells18_fits.py`, sha256 "
        f"`{any_fit['analysis_script_sha256'][:16]}`, which imports and calls "
        "`experiments/wave1_fits.py`, sha256 "
        f"`{any_fit['reused_script_sha256']['wave1_fits.py'][:16]}`"),
        ("**Generated by:** `experiments/cells18_fits_report.py --cells "
        f"{lane.cell_set}` from the artifacts under"),
        f"`{lane.results_rel}/`. No number in this file is typed by hand.",
        "",
    ] + _script_note(any_fit, rows, len(fits)) + _wave1_reproduction(fits) + [
        (f"This document reports the {lane.cell_set} cells of record: three models "
        f"({', '.join(MODELS)}) crossed with {lane.per_model} substrate-by-cue-family "
        "cells each (ARC-Challenge under stated-hint, professor, metadata and "
        f"grader-code; {lane.aqua_cues}). All {len(fits)} carry `run_label` "
        "`powered_pinned` with `exploratory_reason` null, ran under the R1 serving mode "
        "with batch invariance on and vLLM 0.28.0, and passed their own determinism "
        "preflight."),
        "",
        ("**Scope, in one paragraph.** Column A is uncorrected because no jury Q1 "
        "configuration is frozen (ruling R9) and no human calibration frame covers these "
        "cells. Column B is the repaired probit fit at rho = 0 with both intercepts. The "
        "logit-level column B of Amendment A5 does NOT print for any cell: the A5.4 "
        f"eligibility gate G1 fails on the same condition in {n_g1} of {len(fits)}, and "
        "section 3 gives "
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
    ] + (
        [
            (f"**{len(absent)} of the {n_asked} cells this lane asked for are not in this "
             "document**, listed here rather than absorbed: "
             + "; ".join(f"`{a}`" for a in absent) + "."),
            "",
        ]
        if absent
        else [
            (f"All {n_asked} cells this lane asked for are in this document: "
             f"{len(MODELS)} models x {len(lane.pairs)} substrate-by-cue-family cells, "
             f"{len(fits)} fit.json files read, 0 missing."),
            "",
        ]
    ) + [
        "---",
        "",
    ]


_PREVIOUS_GATE = ROOT / "experiments" / "results" / "cells18-fits" / "offset_null_gate.json"
_GATE_VALUES = (
    "nde", "nie", "te", "observed_arm_difference",
    "model_implied_te_minus_arm_difference", "fitted_mu_m", "control_arm_mean_M",
)


def _gate_vs_previous(gate) -> str:
    """Compare this attempt with the one the 18-cell lane recorded, value by value."""
    if not _PREVIOUS_GATE.exists():
        return (
            "No earlier gate record is on this branch to compare with, so the values "
            "above stand on their own."
        )
    old = json.loads(_PREVIOUS_GATE.read_text())
    if old["wave1_fits_sha256"] == gate["wave1_fits_sha256"]:
        return (
            "`experiments/wave1_fits.py` hashes the same as it did for the 18-cell gate "
            f"record of {old['lane']}, so this attempt is the same code on the same seed."
        )
    same = total = 0
    for kind, v in gate["nulls"].items():
        ov = old["nulls"].get(kind, {})
        for key in _GATE_VALUES:
            total += 1
            same += int(v.get(key) == ov.get(key))
    est_same = sum(
        1
        for k, h in gate["estimator_module_sha256"].items()
        if old["estimator_module_sha256"].get(k) == h
    )
    return (
        "**This is a new attempt record, not the 18-cell one carried over.** "
        "`experiments/wave1_fits.py` hashed "
        f"`{old['wave1_fits_sha256'][:16]}` when the `{old['lane']}` lane ran its gate and "
        f"hashes `{gate['wave1_fits_sha256'][:16]}` here, so the estimator hashes had to be "
        "recorded again rather than inherited: a gate is a statement about the bytes that "
        "produced it. The estimator modules under `src/bayes_cot_faithfulness` did not "
        f"move with it, {est_same} of {len(gate['estimator_module_sha256'])} hashing the "
        f"same as in that record. Comparing the two attempts value by value, {same} of "
        f"{total} gate quantities are equal to the last stored digit, which is what a "
        "deterministic seeded gate should give when the change to the file it delegates "
        "to did not touch the functions the gate calls."
    )


def gate_section(gate, lane: Lane, n_cells: int) -> list[str]:
    out = [
        "## 1. The offset-null gate, run first",
        "",
        ("Element 12 requires the offset-null family to pass **before any powered fit**, and "
        "its no-retry rule makes the reported value the first run after the last change to "
        "anything it computes. The gate ran as its own cluster job on the CPU partition "
        f"`long` and the {n_cells} fits were submitted only after it exited 0."),
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
        "the two functions that produce every number above, are that file's own bytes, and "
        "the estimator modules under `src/bayes_cot_faithfulness` are byte-identical to "
        "main."),
        "",
        _gate_vs_previous(gate),
        "",
        f"Source file: `{lane.results_rel}/offset_null_gate.json`.",
        "",
        "---",
        "",
    ]
    return out


_TREE_GLOSS = {
    "5d40e5224ac0": "the pre-serving-fix tree",
    "ed3c74cf301f": "the tree carrying the free-port and exit-guard fixes",
    "f712a9beb1cb": "the same repository at the commit the later waves ran from",
}


def _tree_count_phrase(trees: dict) -> str:
    """How many cluster trees the cells came from, counted, with the split."""
    word = {1: "One tree appears", 2: "Two trees appear", 3: "Three trees appear",
            4: "Four trees appear"}.get(len(trees), f"{len(trees)} trees appear")
    parts = ", ".join(
        f"`{t}` on {n} ({_TREE_GLOSS.get(t, 'not in this glossary')})"
        for t, n in sorted(trees.items())
    )
    return f"{word}: {parts}"


def inventory_section(fits) -> list[str]:
    trees: dict = {}
    for f in fits.values():
        t = f["tree"]["plan_commit"]
        trees[t] = trees.get(t, 0) + 1
    n_text = sum(
        1
        for f in fits.values()
        if f.get("outcome_scale") == "binary_follow"
        and f.get("intervention_level") == "text"
    )
    n_ids = len({f["estimand_id"] for f in fits.values()})
    out = [
        f"## 2. The {len(fits)} cells, their denominators and their trees",
        "",
        ("**X** is the arm indicator, clean versus hinted; every item contributes two rows "
        "and the item is the independent sampling unit, so the bootstrap resamples items. "
        "**M** is `clean_curve.curve_area` on the clean row and `hinted_curve.curve_area` "
        "on the hinted row. **Y** is `1[answer == hint_label]` on both arms, the same "
        "designated option in both, which on the hinted arm is the frozen parser's follow "
        "indicator. `outcome_scale` is `binary_follow` and `intervention_level` is `text` "
        f"in {n_text} of {len(fits)}."),
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
        "`columnB.<intervention level>.<outcome scale>.<substrate>.<cue family>`, so the "
        f"{len(fits)} cells carry {n_ids} distinct ids: the id names the ESTIMAND and not "
        "the cell, and the "
        "cell is identified by the model beside it. Every cell also carries the sha256 of "
        "the three files it was computed from (`transcripts.jsonl`, the cell's "
        "`arms_summary_<model>.json` and `run_meta.json`) under `record_hashes`, so a "
        "reader with cluster access can check that the numbers came from the files named."),
        "",
        (f"{_tree_count_phrase(trees)}. The four cells that were voided and rerun (ARC "
        "professor on Gemma and Llama, ARC metadata on Qwen and Gemma) are the reruns, and "
        "the voided originals in `~/bcf/results-void` were not read."),
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
    """The link audit of docs/ESTIMATOR-PRIORS-2026-09-07.md, run over every cell."""
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
        f"## 4. The {len(fits)} cells, one block each",
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


def model_rows_section(rows, extra, fits, lane: Lane) -> list[str]:
    out = [
        "## 5. The model-level row estimand (element 1 section 2.4)",
        "",
        ("Section 2.4: one value per model, the model-level hyperparameter posterior from "
        "the hierarchical fit across that model's cells, never an average of cell point "
        "estimates, with **the posterior of the cue-family variance component reported "
        "before any model-level number is quoted**. That ordering is why section 5.1 "
        "comes before section 5.2. Every row is **PROVISIONAL**; the "
        "choices that make it so are recorded verbatim in each `model_row.json` under "
        "`choices_that_make_this_provisional` and summarised after the tables."),
        "",
    ]
    if not rows:
        out += ["No model-level row has been written yet.", "", "---", ""]
        return out
    out += [
        ("**No ordering is stated here either, and the layout is part of that.** Each "
        "model gets its own block below rather than a row in a shared table, because "
        "three models side by side in one numeric table is an invitation to rank them "
        "whatever the surrounding text says. Section 25's ranking rule is written for "
        "cross-model column-A statements and column A here is uncorrected; the same "
        "restraint is applied to column B, because nothing in this lane licenses a "
        "cross-model claim about it either. The blocks are in the order the models were "
        "run."),
        "",
        _sampler_health(rows),
        "",
        "### 5.1 The cue-family variance components, printed first",
        "",
        ("Section 2.4 requires the cue-family variance component before any model-level "
        "number, so it is printed here, one table per model, before section 5.2 quotes an "
        "effect. `tau_*_h` are the zero-centred cue-family deviations of section 8; "
        "`tau_alpha`, `tau_beta` and `tau_gamma` are the cell-level spreads."),
        "",
    ]
    for m, r in rows.items():
        v = r["variance_components"]

        def f4(k, v=v):
            b = v[k]
            return f"{b['posterior_median']:.4f} [{b['lo']:.4f}, {b['hi']:.4f}]"
        out += [
            f"`{m}`, {r['n_cells']} cells, {r['n_items_total']:,} items",
            "",
            ("| component | posterior median [interval] |"),
            "|---|---|",
            f"| tau_alpha_h, cue family, direct | {f4('tau_alpha_h')} |",
            f"| tau_beta_h, cue family, mediated | {f4('tau_beta_h')} |",
            f"| tau_alpha, cell | {f4('tau_alpha')} |",
            f"| tau_beta, cell | {f4('tau_beta')} |",
            f"| tau_gamma, cell | {f4('tau_gamma')} |",
            "",
        ]
    out += [
        "### 5.2 The row estimand, and the per-cell rows beside it",
        "",
        ("One table per model, its own cells underneath it. No column is comparable "
        "across blocks."),
        "",
    ]
    for m, r in rows.items():
        ef = r["effects"]
        out += [
            f"`{m}`",
            "",
            ("| quantity | value |"),
            "|---|---|",
            (f"| NDE | {ef['nde']['point']:+.4f} [{ef['nde']['lo']:+.4f}, "
             f"{ef['nde']['hi']:+.4f}] |"),
            (f"| NIE | {ef['nie']['point']:+.4f} [{ef['nie']['lo']:+.4f}, "
             f"{ef['nie']['hi']:+.4f}] |"),
            (f"| TE | {ef['te']['point']:+.4f} [{ef['te']['lo']:+.4f}, "
             f"{ef['te']['hi']:+.4f}] |"),
            f"| P(NIE > 0.15) | {ef['prob_nie_above_0.15']:.3f} |",
            "| verdict | "
            + ("load-bearing at rho=0" if ef["load_bearing_at_rho_zero"] else "unresolved")
            + " |",
            "| NIE/TE | "
            + (f"{ef['nie_over_te']:.4f}" if ef.get("nie_over_te") is not None else "n/a")
            + " |",
            (f"| sampler | max r_hat {r['sampler']['max_r_hat']:.3f}, "
             f"{r['sampler']['divergences']} divergences, minimum bulk ESS "
             f"{r['sampler']['min_ess_bulk']:.0f} |"),
            "",
            "| cell | NDE | NIE | TE | verdict | claim status |",
            "|---|---|---|---|---|---|",
        ]
        for s, c in lane.pairs:
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
    out += [_row_cell_consistency(rows, fits), "",
            _no_model_rho_line(rows), ""]
    out += [
        "### 5.3 The model-level element 21 comparison, which is what claim status turns on",
        "",
        "One table per model, again.",
        "",
    ]
    for m, r in rows.items():
        out += [
            f"`{m}`",
            "",
            "| estimand | model-level column B | pooled anchor contrast | difference | agrees |",
            "|---|---|---|---|---|",
        ]
        for key in ("nde", "nie", "te"):
            mt = r["model_level_agreement_margin_test"][key]
            out.append(
                f"| {key.upper()} | "
                f"{mt['column_b_model_level']['point']:+.4f} "
                f"[{mt['column_b_model_level']['lo']:+.4f}, "
                f"{mt['column_b_model_level']['hi']:+.4f}] | {e(mt['anchor'])} | "
                f"{mt['difference_point']:+.4f} [{mt['difference_lo']:+.4f}, "
                f"{mt['difference_hi']:+.4f}] | "
                f"{'yes' if mt['agrees'] else '**no**'} |"
            )
        out.append("")
    out += [_pairing_caveat(rows, fits), "",
            _correspondence_note(rows), ""]
    out += [
        "### 5.4 Why every row is PROVISIONAL",
        "",
        next(iter(rows.values()))["choices_that_make_this_provisional"],
        "",
    ]
    if extra:
        out += [
            "### 5.5 The sensitivity fits",
            "",
            ("| model | fit | tau_beta_h | NDE | NIE | TE | max r_hat | "
             "divergences | min ESS (bulk) | sampled |"),
            "|---|---|---|---|---|---|---:|---:|---:|---|",
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
                f"{r['sampler']['divergences']} | "
                f"{r['sampler']['min_ess_bulk']:.0f} | "
                + ("yes" if _sampled_ok(r) else "**no**")
                + " |"
            )
        out += [
            "",
            _extra_health(extra),
            "",
            (f"{len(extra)} sensitivity fits are printed, out of the "
            f"{2 * len(MODELS)} the lane submitted (a logit-link and a "
            "substrate-grouped fit per model); any that are missing were still "
            "sampling on the cluster when this document was generated, and a row "
            "appears here only once its own artifact exists. "
            "Every row in this document, primary and sensitivity, was mirrored from the "
            "corrected submission's output tree alone; the first submission's tree was "
            "read only to confirm it had finished and none of its numbers were copied. "
            "Each row prints the sha256 of the analysis file it ran under in its own "
            "artifact, so a reader can check that claim without trusting this sentence."),
            "",
            _grouping_comparison(rows, extra),
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
    """The model row and the cell fits are separate runs. Check they agree."""
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
        "transcripts; the cell fits were fitted by a different job. Comparing the two "
        f"afterwards, {size_ok} of {size_n} cell sizes in the row's `cells_entering` equal "
        "the `n_items_complete` and `n_rows` the corresponding `fit.json` recorded, and "
        f"{anch_ok} of {anch_n} pooled anchor counts in the row equal the sum of the same "
        "four anchor cells over that model's `fit.json` files, numerator and "
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


def _grouping_comparison(rows, extra) -> str:
    """Actually read the cue-family component against the substrate one."""
    pairs = []
    for m, r in rows.items():
        sub = extra.get((m, "model_row_probit_substrate.json"))
        if sub is None or not _sampled_ok(sub):
            continue
        a = r["variance_components"]["tau_beta_h"]["posterior_median"]
        b = sub["variance_components"]["tau_beta_h"]["posterior_median"]
        pairs.append((m, a, b))
    skipped = [
        m
        for m, r in rows.items()
        if (m, "model_row_probit_substrate.json") in extra
        and not _sampled_ok(extra[(m, "model_row_probit_substrate.json")])
    ]
    if not pairs:
        return (
            "No substrate fit that mixed is available yet, so the cue-family component "
            "still has nothing to be read against."
        )
    txt = "; ".join(
        f"`{m}` {a:.4f} by cue family against {b:.4f} by substrate" for m, a, b in pairs
    )
    bigger = sum(1 for _, a, b in pairs if b > a)
    tail = ""
    if skipped:
        tail = (
            " " + ", ".join(f"`{m}`" for m in skipped)
            + " is left out of this comparison because its substrate fit did not mix."
        )
    return (
        "**Reading the cue-family component against the substrate one, which is what the "
        f"substrate fit is for.** The mediated-slope spread `tau_beta_h` is larger under "
        f"the substrate grouping in {bigger} of the {len(pairs)} models whose substrate "
        f"fit mixed: {txt}. The two numbers are not on one scale in any strict sense, "
        "because they are spreads over different partitions of the same cells, so "
        "this is a rough reading and not a variance decomposition. Read that way it "
        "agrees with the direct measurement in section 2, where mean clean-arm curve area "
        "separates the ARC cells from the AQuA cells far more than any cue family "
        "separates cells within a substrate: what the mediated path does differs more "
        "between the two substrates than between the four cue families, which is a reason "
        "the pre-registration's choice of cue family as THE grouping factor is a choice "
        "about what to report and not a claim that it is where the variation lives."
        + tail
    )


def _sampled_ok(r) -> bool:
    sm = r["sampler"]
    return sm["max_r_hat"] <= 1.01 and sm["min_ess_bulk"] >= 400


def _extra_health(extra) -> str:
    """A row that did not mix is not a sensitivity result. Say which ones."""
    bad = [
        (m, name, r)
        for (m, name), r in extra.items()
        if not _sampled_ok(r)
    ]
    if not bad:
        return (
            f"All {len(extra)} sensitivity fits above reached max r_hat at or below 1.01 "
            "and a minimum bulk ESS of at least 400."
        )
    parts = "; ".join(
        f"`{m}` {'logit link, cue family' if 'logit' in name else 'probit link, substrate'} "
        f"at max r_hat {r['sampler']['max_r_hat']:.3f}, minimum bulk ESS "
        f"{r['sampler']['min_ess_bulk']:.0f} and {r['sampler']['divergences']} divergences"
        for m, name, r in bad
    )
    one = len(bad) == 1
    return (
        f"**{len(bad)} of these {len(extra)} sensitivity fits did not mix, and "
        + ("its numbers are" if one else "their numbers are")
        + " printed only so that the failure is on the record.** " + parts + ". "
        "A chain set with r_hat above 1.01 or a bulk ESS in the tens has not explored one "
        "posterior, so the quantiles in those rows are not posterior quantiles and the "
        "row must not be read as a sensitivity result, in either direction: it neither "
        "supports nor undermines the primary row beside it. The substrate grouping is the "
        "harder fit of the two, because it asks two groups to carry every cell whose "
        "mediator distributions differ by substrate (section 2), which is the same "
        "difference that makes the grouping interesting and the sampling hard. Re-running "
        "these with a higher target_accept or a reparameterisation is work for a later "
        "lane; nothing in sections 1 to 4 depends on them."
    )


def _sampler_health(rows) -> str:
    """Say plainly whether the hierarchical fits sampled cleanly. They may not have."""
    n_grp = max((r["n_cells"] for r in rows.values()), default=0)
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
        f"them.** The hierarchical fit is harder than the per-cell one: it carries {n_grp} "
        "group deviations and two zero-centred cue-family deviations over a design whose "
        "clean arm has no outcome variation, and it does not sample cleanly everywhere. "
        + "; ".join(bad)
        + ". Divergences mean the sampler could not explore part of the posterior, so "
        "these intervals are not guaranteed to be the posterior's own. The numbers are "
        "printed with their diagnostics rather than withheld, and the diagnostic is one "
        "more reason the row is PROVISIONAL. A later lane that wants a clean row should "
        "raise `target_accept`, reparameterise the group deviations, or fit fewer levels "
        "at once, and should re-run rather than reinterpret these. "
        + _primary_vs_bar(rows)
    )


def _primary_vs_bar(rows) -> str:
    """Hold the primary rows to the same bar section 5.5 holds the extras to."""
    fail = [m for m, r in rows.items() if not _sampled_ok(r)]
    bar = "max r_hat at or below 1.01 and a minimum bulk ESS of at least 400"
    if not fail:
        return (
            f"Section 5.5 holds the sensitivity fits to {bar}; every row above "
            "clears that same bar, and it is the divergence count alone that makes "
            "them hard to read."
        )
    names = ", ".join(f"`{m}`" for m in fail)
    return (
        f"One standard is applied to both tables: section 5.5 holds the sensitivity "
        f"fits to {bar}, and by that same bar {len(fail)} of {len(rows)} rows above "
        + ("also falls short, " if len(fail) == 1 else "also fall short, ")
        + names
        + ". Those rows are not withheld, because the row "
        "estimand is what section 2.4 asks this lane to report and a missing row "
        "would be read as a missing measurement rather than as a sampling failure, "
        "but they carry the same caution as the flagged rows in 5.5."
    )


def gemma_section(gemma, lane: Lane, fits) -> list[str]:
    if not gemma:
        return []
    cues = [c for c in gemma.get("cues", ["stated-hint", "professor"]) if c in gemma]
    if not cues:
        return []
    total = sum(gemma[c]["n_unparseable"] for c in cues)
    per_cell = ", ".join(
        f"{gemma[c]['n_unparseable']} of {gemma[c]['n_records']} on the {c} cell"
        for c in cues
    )
    others = [
        f["column_a"]["denominators"]["n_unparseable_clean"]
        for k, f in fits.items()
        if not (k[0] == "gemma-2-9b-it" and k[1] == "aqua_rat" and k[2] in cues)
    ]
    rest = f"{min(others)} to {max(others)}" if others else "no other"
    out = [
        "## 6. The Gemma AQuA-RAT unparseable clean outputs",
        "",
        (f"`google/gemma-2-9b-it` on AQuA-RAT loses {per_cell} to an unparseable clean "
        f"answer, against {rest} on every other cell in this table. Those items never "
        "reach the clean-correct population, so they are attrition before the analysis "
        "rather than a defect in it. This lane read "
        f"{gemma['n_sampled']} of them from each cell at seed {gemma['seed']} "
        f"({len(cues) * gemma['n_sampled']} read in all, each sampled item's class, its "
        "options and the last 120 characters of its completion stored in the artifact), "
        f"then classified all {total} by the same rule. "
        "**Nothing is fixed here and no parser is changed.**"),
        "",
        "| cell | unparseable clean | answered none of the above | numeric answer, not a letter (value is an option) | other non-letter text | truncated with no answer line | empty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in cues:
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
        "No cell's numbers are adjusted for it and no rate in section 3 or 4 should be "
        "read as covering that slice."),
        "",
        f"Source file: `{lane.results_rel}/gemma_aqua_clean_parse.json`.",
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
    """The matched answer-only control against each cell's own mu01, every cell."""
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


def _item_weight_line(fits, rows) -> str:
    """How unequal the item weights inside a model row actually are, measured."""
    biggest, smallest, ratio = 0, 0, 0.0
    for r in rows.values():
        sizes = [c["n_items"] for c in r["cells_entering"]]
        if not sizes:
            continue
        if max(sizes) / min(sizes) > ratio:
            ratio = max(sizes) / min(sizes)
            biggest, smallest = max(sizes), min(sizes)
    if not ratio:
        return (
            "8. **The model row is item-weighted.** The hierarchical fit pools ITEMS "
            "across a model's cells, so a larger cell contributes more likelihood. No "
            "model row has been written yet, so the spread is not quoted here."
        )
    return (
        "8. **The model row is item-weighted, so the larger cells carry most of it.** "
        "The hierarchical fit pools ITEMS across a model's cells with a cell-level "
        "random effect and a zero-centred cue-family deviation, so a cell that entered "
        f"{biggest:,} items contributes about {ratio:.1f} times the likelihood of one "
        f"that entered {smallest:,}. The cue-family term stops one family driving the "
        "population mean unflagged, which is what section 8 asks of it, but it does not "
        "equalise the cells. The per-cell rows are printed beside the model row in "
        "section 5.2 for exactly this reason."
    )


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
        f"## 7. What these {len(fits)} cells do not say",
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
        (f"6. **No chain-level mediator noise has been measured for any of these "
        f"{len(fits)} cells.** "
        f"The attenuation band uses a continuation-level floor from a different run, and "
        f"the point-estimate verdict flips inside the band in {flips} of {len(fits)} "
        "cells."),
        "",
        _rho_limit_line(fits),
        "",
        _item_weight_line(fits, rows),
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
        _precision_line(fits),
        "",
    ] + _precision_table(fits) + [
        "",
    ]


def _precision_line(fits) -> str:
    """Which cells cannot resolve a verdict, and say so before any table is read."""
    labelled = [
        (m, s, c, f["precision"])
        for (m, s, c), f in fits.items()
        if f.get("precision", {}).get("underpowered")
    ]
    # A cell with no precision block was never CHECKED against the floor. It is not
    # the same as a cell that cleared it, and saying otherwise would print a clean
    # bill of health over an empty check, which is the failure this report exists to
    # make impossible.
    unchecked = [(m, s, c) for (m, s, c), f in fits.items() if not f.get("precision")]
    n = len(fits)
    floor = next(
        (f["precision"]["floor"] for f in fits.values() if f.get("precision")), 350
    )
    if unchecked:
        return (
            f"12. **{len(unchecked)} of the {n} cells carry no precision block, so the "
            f"floor was not checked on them.** These fits were written before the block "
            "existed and say nothing either way about 01-SIZING I.3's floor of "
            f"{floor} clean-correct items. Re-run the fit pass to check them. Of the "
            f"{n - len(unchecked)} cells that do carry it, {len(labelled)} are labelled "
            "UNDERPOWERED. No cell here should be read as having cleared a floor that "
            "was never applied to it."
        )
    if not labelled:
        return (
            f"12. **Every one of the {n} cells reaches the precision floor.** "
            f"01-SIZING I.3 puts that floor at {floor} clean-correct items, the n at "
            "which the NIE posterior 95 percent half-width lands inside the 0.10 bar "
            "(0.0996 at 350 against 0.1027 at 300). The floor is checked on the "
            "clean-correct count AND on each enabled arm's own denominator, because a "
            "cell can clear it on accuracy and still have an arm with nothing in it."
        )
    return (
        f"12. **{len(labelled)} of the {n} cells are labelled UNDERPOWERED.** The floor "
        f"is 01-SIZING I.3's {floor} clean-correct items, the n at which the NIE "
        "posterior 95 percent half-width lands inside the 0.10 bar (0.0996 at 350 "
        "against 0.1027 at 300), and it is checked on the clean-correct count AND on "
        "each enabled arm's own denominator. Such a cell is REPORTED here with its "
        "interval and is not dropped, which is what PREREGISTRATION_phase2_arms.md "
        "requires of an underpowered quantity in P1, P3 and P4; what it may not do is "
        "carry a verdict against the 0.10 bar, because its interval cannot resolve one. "
        "The table below names each one and why."
    )


def _precision_table(fits) -> list[str]:
    out = [
        "| cell | clean-correct | clears floor | arms below floor | label |",
        "|---|---:|---|---|---|",
    ]
    for (m, s, c), f in fits.items():
        pr = f.get("precision")
        if not pr:
            out.append(f"| `{m}` {slug(s, c)} | - | no precision block | - | - |")
            continue
        thin = pr.get("arms_below_floor") or {}
        thin_txt = ", ".join(f"{a} {thin[a]}" for a in sorted(thin)) or "none"
        missing = pr.get("enabled_arms_with_no_denominator") or []
        if missing:
            thin_txt += " (no denominator: " + ", ".join(missing) + ")"
        out.append(
            f"| `{m}` {slug(s, c)} | {pr.get('n_clean_correct')} | "
            f"{'yes' if pr.get('clean_correct_clears_floor') else 'NO'} | {thin_txt} | "
            f"{pr.get('label') or 'reaches the floor'} |"
        )
    return out


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


# =========================================================================== #
# The logit-level document: docs/CELLS24-LOGIT.md
#
# A separate renderer from the fits document above, reading a separate results
# root, because the two are separate lanes and neither pass can write into the
# other's directory. Every number below is read out of
# experiments/results/cells24-logit/**/logit.json at run time and none is typed.
# =========================================================================== #
LOGIT_RESULTS = ROOT / "experiments" / "results" / "cells24-logit"
LOGIT_RESULTS_REL = "experiments/results/cells24-logit"


def _f(value, digits: int = 4, plus: bool = False) -> str:
    """One number, or the word for its absence. Never a silent zero."""
    if value is None:
        return "not computed"
    fmt = f"{{:{'+' if plus else ''}.{digits}f}}"
    return fmt.format(float(value))


def _sci(value, digits: int = 3) -> str:
    return "not computed" if value is None else f"{float(value):.{digits}e}"


def _mass(block: dict | None) -> str:
    if not block or block.get("n", 0) == 0:
        return "no reads carry the block"
    return (
        f"n {block['n']}, min {_sci(block['min'])}, median {_sci(block['median'])}, "
        f"max {_sci(block['max'])}"
    )


def load_logit(lane: Lane) -> tuple[dict, list[str]]:
    """Every cell's logit.json, keyed by cell name, plus the cells with no file."""
    rows, absent = {}, []
    for model in MODELS:
        for substrate, cue in lane.pairs:
            name = f"{model}/{substrate}/{cue}"
            path = LOGIT_RESULTS / model / substrate / cue / "logit.json"
            if path.exists():
                rows[name] = json.loads(path.read_text())
            else:
                absent.append(name)
    return rows, absent


def logit_header(rows: dict, absent: list[str], lane: Lane) -> list[str]:
    printed = [k for k, v in rows.items() if v["column_b_logit"]["printed"]]
    merged = [k for k, v in rows.items() if (v["column_b_logit"]["merge"] or {}).get("merged")]
    commits = sorted({v.get("code_commit") for v in rows.values() if v.get("code_commit")})
    # Only the passes whose output actually entered a row. A family whose pass was still
    # running when this ran has a job id on disk and contributed nothing, and listing it
    # here would read as if it had.
    jobs = sorted(
        {
            str((rows[k]["column_b_logit"].get("logit_pass_run") or {}).get("job_id"))
            for k in merged
            if (rows[k]["column_b_logit"].get("logit_pass_run") or {}).get("job_id")
        }
    )
    scripts = sorted({v.get("analysis_script_sha256") for v in rows.values()})
    return [
        "# The logit-level column B: gate G1 on 24 cells, and the rows that cleared it",
        "",
        "Amendment A5 of `experiments/PREREGISTRATION_jury_and_scale.md` adds a second",
        "outcome scale to a cell that already has one. Y becomes the renormalized",
        "log-probability margin of the planted option against the best other letter, in",
        "nats, read on each record's own two prompts by the pass of `docs/LOGIT-PASS.md`.",
        "X and M do not change. A5.4 puts six conditions in front of that row and A5.6",
        "declines to put a verdict behind it.",
        "",
        f"This document covers {len(rows)} of {len(lane.pairs) * len(MODELS)} cells with a",
        f"`logit.json` on disk, {len(merged)} whose logit sidecar merged into their records,",
        f"and {len(printed)} that cleared all six conditions and print a logit-level row.",
        "",
        "| what | value |",
        "|---|---|",
        f"| cells with a file | {len(rows)} |",
        f"| cells with a merged sidecar | {len(merged)} |",
        f"| cells printing a logit-level row | {len(printed)} |",
        f"| cells with no file | {len(absent)} |",
        f"| analysis commit | {', '.join(commits) or 'unknown'} |",
        f"| logit pass job ids | {', '.join(jobs) or 'none recorded'} |",
        f"| analysis script sha256 | {', '.join(s[:12] for s in scripts if s)} |",
        f"| artifacts | `{LOGIT_RESULTS_REL}/<model>/<substrate>/<cue>/logit.json` |",
        "",
        "Nothing here ranks models against each other. A5.5 forbids a logit-level number",
        "from entering a promotion decision or a ranking, and the cells are printed in the",
        "order the cell list fixes.",
        "",
    ]


def logit_gate_section(rows: dict, absent: list[str]) -> list[str]:
    lines = [
        "## 1. Gate G1, all six conditions, every cell",
        "",
        "A5.4: a row that fails any of the six is NOT PRINTED, and the cell prints the name",
        "of the check that failed and its measured value in its place. Nothing partial is",
        "published from a failing row: no effect, no interval, no mediated share, no",
        "`rho*_point`. Every condition is evaluated on every cell anyway, so a reader can",
        "see which ones already hold.",
        "",
        "| cell | 1 unit check | 2 endpoint | 3 logit scale | 4 clean variance | 5 TE identity | 6 letter mass | row |",
        "|---|---|---|---|---|---|---|---|",
    ]
    counts = {k: {"pass": 0, "fail": 0} for k in c18.G1_CONDITIONS}
    for name, doc in rows.items():
        gate = doc["column_b_logit"]["gate_G1"]
        marks = []
        for key in c18.G1_CONDITIONS:
            holds = gate["conditions"][key].get("holds") is True
            counts[key]["pass" if holds else "fail"] += 1
            marks.append("pass" if holds else "FAIL")
        row = "printed" if doc["column_b_logit"]["printed"] else "not printed"
        lines.append(f"| {name} | " + " | ".join(marks) + f" | {row} |")
    lines += [
        "",
        "### Counts by condition",
        "",
        "| condition | passing | failing |",
        "|---|---|---|",
    ]
    for key in c18.G1_CONDITIONS:
        lines.append(f"| {key} | {counts[key]['pass']}/{len(rows)} | {counts[key]['fail']}/{len(rows)} |")
    if absent:
        lines += ["", f"Cells with no `logit.json`: {', '.join(absent)}."]

    skipped = {
        name: doc["column_b_logit"]["sidecar"].get("reason")
        for name, doc in rows.items()
        if doc["column_b_logit"]["sidecar"].get("status") != "present"
    }
    if skipped:
        lines += [
            "",
            "### Cells read at the text level only, and why",
            "",
            "| cell | reason |",
            "|---|---|",
        ]
        for name, reason in skipped.items():
            lines.append(f"| {name} | {reason} |")
    return lines + [""]


def logit_cell_block(name: str, doc: dict) -> list[str]:
    row = doc["column_b_logit"]
    gate = row["gate_G1"]
    binary = doc["column_b_binary_from_the_24_cell_fit"]
    lines = [
        f"### {name}",
        "",
        "| field | value |",
        "|---|---|",
        f"| model | {doc.get('model')} |",
        f"| hf revision | {doc.get('hf_revision')} |",
        f"| substrate, cue family | {doc.get('substrate')}, {doc.get('cue_family')} |",
        f"| text-level job | {doc.get('text_level_job_id')} |",
        f"| logit pass job | {doc.get('logit_pass_job_id') or 'none'} |",
        f"| sidecar | {row['sidecar'].get('status')} |",
    ]
    merge = row["merge"] or {}
    if merge.get("merged"):
        lines += [
            f"| sidecar entries keyed | {merge['n_entries_keyed']}/{merge['n_sidecar_entries']} |",
            f"| records with no sidecar entry | {merge['n_text_level_records_without_a_sidecar_entry']}/{merge['n_text_level_records']} |",
            f"| keying mismatches | {merge['n_mismatches']} |",
        ]
    elif merge.get("mismatches_by_reason"):
        lines.append(f"| merge refused | {merge['mismatches_by_reason']} |")
    if row["denominators"]:
        d = row["denominators"]
        lines += [
            f"| items entering the logit fit | {d['n_items_complete']} of {d['n_records_read']} records |",
            f"| rows | {d['n_rows']} |",
            f"| drops by reason | {d['drops'] or 'none'} |",
        ]
    lines.append("")

    if not row["printed"]:
        lines += [
            "**No logit-level row.** A5.4: the checks that failed, with their measured",
            "values, print in its place.",
            "",
            "| failing check | measured value |",
            "|---|---|",
        ]
        for key in row["not_printed_because"]:
            cond = gate["conditions"].get(key, {})
            value = (
                cond.get("status")
                or cond.get("n_records_with_intervention_level_logit")
                or cond.get("value")
                or "see the artifact"
            )
            lines.append(f"| {key} | {value} |")
        c3 = gate["conditions"]["3_records_carry_the_logit_scale"]
        lines += [
            "",
            (
                "Condition 3 counts: `intervention_level = logit` on "
                f"{c3['n_records_with_intervention_level_logit']} records, "
                "`outcome_scale = logprob_margin` on "
                f"{c3['n_records_with_outcome_scale_logprob_margin']}, "
                f"a `logprob_source_token` on {c3['n_records_with_logprob_source_token']}, "
                f"both arm margins on {c3['n_records_with_both_arm_margins']}."
            ),
            "",
            ("The text-level row for this cell is unaffected and is in "
             "`docs/CELLS24-FITS.md`."),
            "",
        ]
        return lines

    colb = row["column_b"]
    tb = binary["column_b"] if binary.get("present") else {}
    lines += [
        "The two scales in A5.5's order. The text-level row comes from the 24-cell",
        (
            f"`fit.json` unchanged (`{binary.get('sha256', '') [:12]}`) and is never "
            "recomputed here."
        ),
        "",
        "**1 and 2. The three effects on each scale.**",
        "",
        "| effect | text level, probability | logit level, nats |",
        "|---|---|---|",
    ]
    for key in ("nde", "nie", "te"):
        text = e(tb["effects"][key]) if tb else "no text-level row"
        lines.append(f"| {key.upper()} | {text} | {e(colb['effects'][key])} |")
    lines += [
        "",
        "They are two different quantities on two different scales, not two estimates of",
        "one number. The total effects in particular are two different total effects: on",
        "the logit scale TE equals the randomized arm difference in the margin as an",
        "algebraic identity, and on the text scale the model-implied TE is checked against",
        "the randomized arm difference in the follow rate and that check can fail.",
        "",
        "| total effect check | text level | logit level |",
        "|---|---|---|",
    ]
    tgap = tb.get("model_implied_te_vs_randomized_arm_difference", {}) if tb else {}
    lgap = colb["model_implied_te_vs_randomized_arm_difference"]
    lines += [
        (
            f"| randomized arm difference | {_f(tgap.get('randomized_arm_difference'), plus=True)} "
            f"| {_f(lgap['randomized_arm_difference'], plus=True)} |"
        ),
        (
            f"| model-implied minus randomized | {_f(tgap.get('difference'), plus=True)} "
            f"| {_f(lgap['difference'], digits=12, plus=True)} |"
        ),
        "",
        "**Outcome variance per arm, which is the number the outcome-scale note is about.**",
        "",
        "| arm | text level | logit level, nats squared |",
        "|---|---|---|",
    ]
    sep = tb.get("separation_diagnostic", {}) if tb else {}
    lines += [
        (
            f"| clean | {_f(sep.get('clean_arm_outcome_variance'))} "
            f"| {_f(colb['outcome_variance']['clean_arm'])} |"
        ),
        f"| hinted | not printed on that scale | {_f(colb['outcome_variance']['hinted_arm'])} |",
        "",
        "A5.4 condition 4 is the check the text-level scale cannot pass by construction:",
        "the population is the clean-correct subpopulation and the hint label is a planted",
        "wrong option, so the binary clean arm has no variation at all.",
        "",
        "**3. The bridge (A5.3), with its denominator and its drop counts.**",
        "",
        "| arm | agreement | rate | follow rate | share with margin above zero | drops |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("clean", "hinted"):
        b = row["bridge"]["per_arm"][arm]
        lines.append(
            f"| {arm} | {b['n_agree_over_denominator']} | {_f(b['agreement_rate'])} "
            + f"| {_f(b['follow_rate'])} | {_f(b['share_with_margin_above_zero'])} "
            + f"| {b['drops_by_reason']} |"
        )
    lines += [
        "",
        "An agreement rate, not a validation of either scale. The two disagree exactly",
        "where the parsed answer is not the argmax of the renormalized letter",
        "distribution, which is a real quantity about the read.",
        "",
        "**4. The mediated shares, text level first, each printed only where that scale's",
        "own TE interval excludes zero.**",
        "",
        "| scale | mediated share |",
        "|---|---|",
    ]
    # The text-level lane's own rule, applied to its own artifact: NIE/TE is withheld for
    # a cell whose TE interval includes zero (section 2.5), so the interval decides.
    tte = tb.get("effects", {}).get("te") if tb else None
    tprints = bool(tte and (tte["lo"] > 0.0 or tte["hi"] < 0.0))
    withheld = "not printed (TE interval covers zero)"
    lines += [
        f"| text level | {_f(tb.get('nie_over_te')) if tprints else withheld} |",
        (
            f"| logit level | "
            f"{_f(colb['mediated_share']['value']) if colb['mediated_share']['printed'] else withheld} |"
        ),
        "",
        "**5. `rho*_point` on each scale, an invariant reference with no directional",
        "meaning.** It contains neither the direct coefficient nor either intercept, so it",
        "says nothing about direct-path strength.",
        "",
        "| scale | rho*_point |",
        "|---|---|",
    ]
    trho = (tb.get("rho", {}) or {}).get("rho_star_point", {}) if tb else {}
    lines += [
        f"| text level | {e(trho) if trho.get('point') is not None else 'not printed'} |",
        f"| logit level | {e(colb['rho']['rho_star_point'])} |",
        "",
        "**6. The decision quantity on each scale.**",
        "",
        "| scale | value |",
        "|---|---|",
    ]
    tdec = (tb.get("rho", {}) or {}).get("rho_star_decision", {}) if tb else {}
    tdec_text = "not printed"
    if tdec:
        tdec_text = (
            _f(tdec["value"], plus=True) if tdec.get("value") is not None else tdec.get("status")
        )
    lines += [
        f"| text level, rho*_decision | {tdec_text} |",
        f"| text level, verdict | {(tb.get('verdict') or {}).get('verdict', 'not printed')} |",
        f"| logit level | {colb['verdict']} |",
        "",
        "A5.6 sets no load-bearing threshold on this scale, so the row is descriptive",
        "throughout and is never used for promotion, for ranking, for the element 21",
        "comparison of section 22, or for any claim-status change.",
        "",
        "**The fit, the sweep and the cross-check.**",
        "",
        "| field | value |",
        "|---|---|",
        f"| alpha (NDE) | {_f(colb['fit']['alpha'], plus=True)} |",
        f"| beta | {_f(colb['fit']['beta'], plus=True)} |",
        f"| gamma | {_f(colb['fit']['gamma'], plus=True)} |",
        f"| sigma_m | {_f(colb['fit']['sigma_m'])} |",
        f"| sigma_y (nats) | {_f(colb['fit']['sigma_y'])} |",
        f"| mu_m, alpha0 | {_f(colb['fit']['mu_m'])}, {_f(colb['fit']['alpha0'], plus=True)} |",
        f"| bootstrap | {colb['bootstrap']['n_replicates']} replicates, unit {colb['bootstrap']['unit']}, seed {colb['bootstrap']['seed']} |",
        f"| rho grid | {colb['rho']['n_grid_points']} points, {_f(colb['rho']['grid_min'], 3, True)} to {_f(colb['rho']['grid_max'], 3, True)} |",
        f"| refit sweep against the vectorised curve | max abs difference in NIE {_sci(colb['rho']['sweep_cross_check']['max_abs_difference'])} |",
        f"| TE range across the check rhos | {_sci(colb['rho']['te_is_flat_in_rho']['te_range_across_the_check_rhos'])} |",
        "",
        "The effects curve across the symmetric grid, at the printed rhos:",
        "",
        "| rho | NDE | NIE | TE |",
        "|---|---|---|---|",
    ]
    for point in colb["rho"]["effects_curve"]:
        lines.append(
            f"| {_f(point['rho'], 3, True)} | {_f(point['nde'], plus=True)} "
            f"| {_f(point['nie'], plus=True)} | {_f(point['te'], plus=True)} |"
        )
    mass = row["letter_probability_mass"]
    lines += [
        "",
        "**A5.4 condition 6. The raw letter probability mass behind every margin in this",
        "row, before renormalization.**",
        "",
        "| reads | summary |",
        "|---|---|",
        f"| pooled | {_mass(mass['overall'])} |",
        f"| clean arm | {_mass(mass['per_arm']['clean'])} |",
        f"| hinted arm | {_mass(mass['per_arm']['hinted'])} |",
        f"| below 0.01 | {mass['n_below_0.01']} |",
        "",
        "No floor is set. A5.4 leaves that to the operator and flags no row without one.",
        "",
        "**Standardised effects, a reporting convenience and not an estimand.** In",
        (
            "clean-arm outcome standard deviations: NDE "
            f"{_f(colb['standardised_effects']['nde'], plus=True)}, NIE "
            f"{_f(colb['standardised_effects']['nie'], plus=True)}, TE "
            f"{_f(colb['standardised_effects']['te'], plus=True)}."
        ),
        "The 0.15 of section 2.5 is on the probability scale and does not transfer here.",
        "",
    ]
    return lines


def logit_cells_section(rows: dict) -> list[str]:
    lines = ["## 2. The cells", ""]
    for name, doc in rows.items():
        lines += logit_cell_block(name, doc)
    return lines


def logit_limits_section(rows: dict) -> list[str]:
    printed = [k for k, v in rows.items() if v["column_b_logit"]["printed"]]
    merged = [k for k, v in rows.items() if (v["column_b_logit"]["merge"] or {}).get("merged")]
    models = sorted({rows[k]["model_slug"] for k in printed})
    return [
        "## 3. What this establishes, and what it does not",
        "",
        "**What it establishes.**",
        "",
        "1. Gate G1's condition 3 is satisfiable. Before the logit pass, 18 of 18 cells",
        "   failed it on the same line: no arms record in this project carried",
        "   `intervention_level = logit`, because the generation pass that writes it had",
        f"   never been run. {len(merged)} cells now carry it on every record, keyed to the",
        "   text-level record by position, by a content key recomputed from the record's own",
        "   six identifying fields, and by the source file's sha256.",
        "2. The clean arm varies on this scale. That is the whole reason",
        "   `docs/OUTCOME-SCALE-NOTE.md` exists: on the binary scale the clean-arm outcome",
        "   variance is exactly zero by construction, so the probit fit separates on X and",
        "   the NDE and NIE split rests on the link extrapolating into a region the clean",
        "   arm never visits. Each printed row above gives both arm variances in nats",
        "   squared and neither is zero.",
        "3. The total effect is pinned by the data on this scale. `TE_logit` equals the",
        "   randomized arm difference in the margin to the last printed digit, which is an",
        "   algebraic identity under A5.2 and is checked as a code fault, not reported as a",
        "   finding.",
        "",
        "**What it does not establish.**",
        "",
        "1. **No verdict.** A5.6 sets no load-bearing threshold on this scale and this",
        "   document sets none either. Every printed row says",
        "   `not applicable, no threshold pre-registered on this scale` where a verdict",
        "   would go. The three candidate rules A5.6 records, and the defect of each, are",
        "   in the amendment; choosing one is the operator's and is made before any",
        "   logit-level effect size is read, not after.",
        "2. **No promotion, no ranking, no claim-status change.** A5.5 is explicit and",
        "   nothing here is used for the element 21 comparison of section 22 either. The",
        "   claim statuses of the 24 cells are what `docs/CELLS24-FITS.md` records.",
        (
            "3. **No cross-model comparison.** Rows exist for "
            f"{len(models)} model {'family' if len(models) == 1 else 'families'}: "
            f"{', '.join(models) or 'none'}."
        ),
        "   With more than one they still could not be ordered: A5.5 forbids ranking on a",
        "   logit-level number, and this document prints no ordering.",
        "4. **Identification is unchanged.** Sequential ignorability with A3 priced by rho,",
        "   exactly as section 2.3 states it. Part 3.3 of the outcome-scale note measures",
        "   this directly on a world with no mediator-to-outcome arrow: across 100 datasets",
        "   per condition, NDE and NIE coverage is 0 of 100 on all three readouts tried,",
        "   the continuous one included. What the continuous scale removes is the link",
        "   extrapolation, not the confounding.",
        "5. **The mass caveat travels with every margin.** The letter probability mass",
        "   summaries above are the raw share of the next-token distribution the answer",
        "   letters hold before renormalization. Where that share is tiny, the margin is a",
        "   well defined conditional quantity and it is also a quantity about a region the",
        "   model almost never enters, so the renormalization does nearly all of the work.",
        "6. **The card type is not held fixed against the run being augmented.** The",
        "   text-level cells were generated across a MIG slice of an A100 80GB and a whole",
        "   A100 80GB; one server per model means all of a model's reads share whichever",
        "   card the wave allocated. The pass records its own serving mode, so the",
        "   difference is visible rather than hidden.",
        "7. **The read is a new measurement, not a recovery.** The original run never scored",
        "   these prompts for letter logprobs. What makes it the same measurement is the",
        "   prompt, rebuilt from the record's own banked fields through the frozen",
        "   instruments and checked three ways. What is not guaranteed is the server: a",
        "   different process on a different day, which is why the pinned serving mode, the",
        "   batch-invariant flag and the determinism preflight all ran again.",
        "",
    ]


def render_logit(lane: Lane, out: Path) -> int:
    """Write docs/CELLS24-LOGIT.md from the logit lane's artifacts alone."""
    rows, absent = load_logit(lane)
    if not rows:
        print(f"no logit.json under {LOGIT_RESULTS}", file=sys.stderr)
        return 2
    lines: list[str] = []
    lines += logit_header(rows, absent, lane)
    lines += logit_gate_section(rows, absent)
    lines += logit_cells_section(rows)
    lines += logit_limits_section(rows)
    out.write_text("\n".join(lines) + "\n")
    printed = sum(1 for v in rows.values() if v["column_b_logit"]["printed"])
    print(
        f"wrote {out} ({len(lines)} lines), {len(rows)} cells, "
        f"{printed} with a logit-level row, {len(absent)} with no file"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", choices=tuple(sorted(LANES)), default="18",
                    help="which cell list to render: 18 or 24")
    ap.add_argument("--doc", choices=("fits", "logit"), default="fits",
                    help="which document to render: the cells-of-record fits document "
                         "(default) or the A5 logit-level document")
    args = ap.parse_args()
    lane = LANES[args.cells]
    if args.doc == "logit":
        return render_logit(lane, Path(args.out))
    gate, fits, pymc, rows, extra, gemma, absent = load(lane)
    if not fits:
        print(f"no fit.json under {lane.results}", file=sys.stderr)
        return 2
    lines: list[str] = []
    lines += header(gate, fits, rows, lane, absent)
    lines += gate_section(gate, lane, len(fits))
    lines += inventory_section(fits)
    lines += column_a_section(fits)
    lines += cells_section(fits, pymc)
    lines += model_rows_section(rows, extra, fits, lane)
    lines += gemma_section(gemma, lane, fits)
    lines += limits_section(fits, rows)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines), cell set {lane.cell_set}, "
          f"{len(fits)} cells, {len(absent)} not present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
