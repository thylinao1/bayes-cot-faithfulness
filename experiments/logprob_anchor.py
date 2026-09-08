"""Job A of ``docs/OUTCOME-SCALE-NOTE.md`` part 4.5: the element 21 anchor contrast read
on the LOGPROB scale, from records that already exist. No new generation.

Two modes, in this order.

``extract``
    One pass per cell of record. Reads the cell's arms records with
    ``wave1_fits.load_records`` (the same ``source_file`` filter the binary table uses),
    reads the four anchor cells' stored letter-logprob blocks, applies the six required
    field checks of part 4.5 step 2, and writes the four cell means, the five section 22
    contrasts and the letter-probability-mass summary to
    ``experiments/results/logprob-anchor/<model>/<substrate>/<cue>/anchor_logprob.json``.
    Aggregates only: no per-item value is written out.

``report``
    Renders ``docs/LOGPROB-ANCHOR.md`` from those artifacts and from the binary contrasts
    already in ``experiments/results/cells18-fits/**/fit.json``. Every number in the
    document is read at run time, so none of them is typed by hand.

What this file deliberately does NOT do.

* It never recomputes Y. ``Y_c`` is the stored ``logprob_margin`` of the anchor cell,
  which ``bayes_cot_faithfulness.outcome_scale.letter_logprob_fields`` defines as the
  renormalized log odds of the target letter against the BEST OTHER letter. Part 4.5 step
  3 says the stored value is the one that travels, and the definition is restated in
  every artifact so a reader does not have to go and find it.
* It never compares a number on this scale against a Column B estimate. Part 4.5 step 6
  forbids it until a margin exists on this scale, and Amendment A5.6 declines to set one:
  a logit-level row is descriptive, prints no verdict, and is "never used for promotion,
  for ranking, for the element 21 comparison of section 22, or for any claim-status
  change".
* It computes no cross-model ordering. Cells are listed in the declaration order of
  ``cells18_fits.CELLS`` and that order is not a result.

Run:

    PYTHONPATH=src python experiments/logprob_anchor.py --mode extract \\
        --fits-root experiments/results/cells18-fits \\
        --results-root ~/bcf/results --out-root experiments/results/logprob-anchor

    PYTHONPATH=src python experiments/logprob_anchor.py --mode report \\
        --out-root experiments/results/logprob-anchor \\
        --fits-root experiments/results/cells18-fits --doc docs/LOGPROB-ANCHOR.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cells18_fits import CELLS  # the 18 cells of record, in their declaration order
from openai_client import _strip_token  # the same decoder the writer matched letters with
from wave1_fits import (
    ANCHOR_CELLS,
    N_BOOTSTRAP,
    load_records,
    quantile_interval,
    sha256_of,
)

from bayes_cot_faithfulness.interventions import CHOICE_LABELS

# The five contrasts of section 22, with the weights ``wave1_fits.anchor_block`` gives
# them. That function holds them in a local dict, so this is the one thing here that
# mirrors rather than imports; ``tests/test_logprob_anchor.py`` pins the mirror by
# recomputing the binary contrast points of a committed fit.json from its own cell rates
# with these weights and requiring the recorded points back.
ANCHOR_CONTRAST_DEF = {
    "text_source_given_cued_recipient": (("mu11", 1.0), ("mu10", -1.0)),
    "text_source_given_clean_recipient": (("mu01", 1.0), ("mu00", -1.0)),
    "cue_effect_given_clean_donor": (("mu10", 1.0), ("mu00", -1.0)),
    "interaction": (("mu11", 1.0), ("mu10", -1.0), ("mu01", -1.0), ("mu00", 1.0)),
    "joint_replay_regime": (("mu11", 1.0), ("mu00", -1.0)),
}

Y_DEFINITION = (
    "Y_c is the STORED anchor-cell logprob_margin, never recomputed here: the "
    "renormalized log odds of the target letter against the BEST OTHER letter, as "
    "src/bayes_cot_faithfulness/outcome_scale.py::letter_logprob_fields writes it. "
    "Element 0 says 'the log-probability margin of the planted option' without "
    "disambiguating best-other from against-the-rest; the records carry best-other, so "
    "best-other is what this reports. Units are nats."
)

NO_PROMOTION = (
    "No number on this scale is compared against any Column B estimate. Part 4.5 step 6 "
    "holds the element 21 promotion rule until a margin exists on this scale, and "
    "Amendment A5.6 sets none: section 22.1's 0.10 is two thirds of the "
    "probability-scale 0.15 of section 2.5, and there is no honest way to carry a "
    "probability into nats. A logit-level row is descriptive and prints no verdict."
)

MASS_CAVEAT = (
    "Section 4.4: the raw letter mass on the anchor prompts is small, so the "
    "renormalized distribution these margins come from is a ratio of small numbers. The "
    "mass summary prints beside every margin table and is never omitted."
)


# --------------------------------------------------------------------------- #
# Step 2. The required field checks. Any failure drops the ITEM, counted by reason.
# --------------------------------------------------------------------------- #
def labels_for(rec: dict) -> tuple[str, ...]:
    """The item's allowed answer labels, as ``QAItem.labels`` builds them from choices."""
    choices = rec.get("choices")
    if not isinstance(choices, list) or not choices:
        return ()
    return tuple(CHOICE_LABELS[: len(choices)])


def cell_failures(block: object, hint_label: object, labels: tuple[str, ...]) -> list[str]:
    """The part 4.5 step 2 checks on one anchor cell's stored logprob block."""
    if not isinstance(block, dict) or not block:
        return ["missing_logprob_block"]
    out: list[str] = []
    if block.get("intervention_level") != "logit":
        out.append("intervention_level_not_logit")
    if block.get("outcome_scale") != "logprob_margin":
        out.append("outcome_scale_not_logprob_margin")
    if block.get("unavailable_reason") is not None:
        out.append("unavailable_reason_present")
    if block.get("target_letter") != hint_label:
        out.append("target_letter_mismatch")
    logps = block.get("answer_logprobs")
    if not isinstance(logps, dict) or set(logps) != set(labels):
        out.append("answer_logprobs_label_set_mismatch")
    tokens = block.get("logprob_source_token")
    if not isinstance(tokens, dict) or any(
        _strip_token(str(tokens.get(letter, ""))) != letter for letter in labels
    ):
        out.append("source_token_mismatch")
    if block.get("logprob_margin") is None:
        out.append("logprob_margin_null")
    return out


def item_failures(rec: dict) -> tuple[list[str], dict, dict]:
    """Every reason this record fails, plus its four margins and four masses when it does not."""
    hint = rec.get("hint_label")
    labels = labels_for(rec)
    reasons: list[str] = []
    if hint is None:
        reasons.append("no_hint_label")
    if not labels:
        reasons.append("no_choices")
    anchor = rec.get("anchor")
    if not isinstance(anchor, dict) or not isinstance(anchor.get("cells"), dict):
        reasons.append("no_anchor_block")
        return reasons, {}, {}
    if reasons:
        return reasons, {}, {}
    margins: dict[str, float] = {}
    masses: dict[str, float] = {}
    for cell in ANCHOR_CELLS:
        block = (anchor["cells"].get(cell) or {}).get("logprob")
        failed = cell_failures(block, hint, labels)
        for reason in failed:
            if reason not in reasons:
                reasons.append(reason)
        if failed:
            continue
        margins[cell] = float(block["logprob_margin"])
        mass = block.get("letter_probability_mass")
        masses[cell] = None if mass is None else float(mass)
    return reasons, margins, masses


def collect_items(records: list[dict]) -> tuple[list[dict], Counter]:
    """Keep the items that pass every check in all four anchor cells; count the rest."""
    items, drops = [], Counter()
    for idx, rec in enumerate(records):
        reasons, margins, masses = item_failures(rec)
        if reasons:
            for reason in reasons:
                drops[reason] += 1
            drops["items_dropped"] += 1
            continue
        items.append({"index": idx, "y": margins, "mass": masses})
    return items, drops


# --------------------------------------------------------------------------- #
# Steps 4 and 5. The mass summary, the cell means, the item bootstrap, the contrasts.
# --------------------------------------------------------------------------- #
def mass_summary(items: list[dict]) -> dict:
    """min, median and max of the raw letter mass per anchor cell, with denominators."""
    out = {}
    for cell in ANCHOR_CELLS:
        vals = [it["mass"][cell] for it in items if it["mass"].get(cell) is not None]
        out[cell] = {
            "n": len(vals),
            "n_items": len(items),
            "min": min(vals) if vals else None,
            "median": float(statistics.median(vals)) if vals else None,
            "max": max(vals) if vals else None,
            "n_below_0.01": int(sum(1 for v in vals if v < 0.01)),
        }
    return out


def cell_means(items: list[dict]) -> dict:
    """mu_ab as the item mean of the stored margin, with its spread and its denominator."""
    out = {}
    for cell in ANCHOR_CELLS:
        vals = np.array([it["y"][cell] for it in items], dtype=float)
        out[cell] = {
            "n": int(vals.size),
            "mean": float(vals.mean()) if vals.size else None,
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
        }
    return out


def bootstrap_cells(items: list[dict], seed: int, n_replicates: int) -> dict:
    """A paired ITEM bootstrap: one resample index feeds all four cells, so contrasts pair."""
    n = len(items)
    y = {c: np.array([it["y"][c] for it in items], dtype=float) for c in ANCHOR_CELLS}
    boot = {c: np.empty(n_replicates) for c in ANCHOR_CELLS}
    rng = np.random.default_rng(seed)
    for b in range(n_replicates):
        idx = rng.integers(0, n, n)
        for cell in ANCHOR_CELLS:
            boot[cell][b] = float(y[cell][idx].mean())
    return boot


def contrasts_from(means: dict, boot: dict) -> dict:
    """The five section 22 contrasts, point from the means and interval from the bootstrap."""
    out = {}
    for name, terms in ANCHOR_CONTRAST_DEF.items():
        point = sum(sign * means[c]["mean"] for c, sign in terms)
        series = sum(sign * boot[c] for c, sign in terms)
        lo, hi = quantile_interval(series)
        out[name] = {"point": float(point), "lo": lo, "hi": hi}
    return out


def with_intervals(means: dict, boot: dict) -> dict:
    """Attach the bootstrap interval of each cell mean to its summary."""
    for cell in ANCHOR_CELLS:
        lo, hi = quantile_interval(boot[cell])
        means[cell]["boot_lo"], means[cell]["boot_hi"] = lo, hi
    return means


# --------------------------------------------------------------------------- #
# One cell, end to end
# --------------------------------------------------------------------------- #
def binary_side(fit: dict) -> dict:
    """The binary anchor of record, copied through from fit.json and never recomputed."""
    anchor = fit.get("anchor") or {}
    return {
        "source": "experiments/results/cells18-fits/<cell>/fit.json, key anchor",
        "outcome_scale": fit.get("outcome_scale"),
        "intervention_level": fit.get("intervention_level"),
        "cells": anchor.get("cells"),
        "contrasts": anchor.get("contrasts"),
    }


def extract_cell(fit_path: Path, records_path: Path) -> dict:
    """Everything part 4.5 Job A asks for, for one cell, as a small aggregate record."""
    fit = json.loads(fit_path.read_text())
    seed = fit["column_b"]["bootstrap"]["seed"]
    records, file_counts = load_records(records_path)
    items, drops = collect_items(records)
    started = time.time()
    means = cell_means(items)
    boot = bootstrap_cells(items, seed, N_BOOTSTRAP) if items else {}
    if items:
        means = with_intervals(means, boot)
        contrasts = contrasts_from(means, boot)
    else:
        contrasts = {}
    recorded = (fit.get("record_hashes") or {}).get("transcripts.jsonl")
    digest = sha256_of(records_path)
    return {
        "spec": "docs/OUTCOME-SCALE-NOTE.md part 4.5, Job A",
        "cell": fit["cell"],
        "model_slug": fit["model_slug"],
        "model": fit["model"],
        "substrate": fit["substrate"],
        "cue_family": fit["cue_family"],
        "generation_job_id": fit.get("job_id"),
        "outcome_scale": "logprob_margin",
        "intervention_level": "logit",
        "y_definition": Y_DEFINITION,
        "records_path": str(records_path),
        "records_sha256": digest,
        "records_sha256_in_fit_json": recorded,
        "records_match_fit_json": bool(recorded == digest),
        "analysis_script_sha256": sha256_of(Path(__file__).resolve()),
        "file_counts": file_counts,
        "denominators": {
            "n_records_read": len(records),
            "n_items_kept": len(items),
            "n_items_dropped": int(drops.get("items_dropped", 0)),
            "drops": {k: int(v) for k, v in sorted(drops.items())},
        },
        "letter_probability_mass": mass_summary(items),
        "mass_caveat": MASS_CAVEAT,
        "cells": means,
        "contrasts": contrasts,
        "bootstrap": {
            "n_replicates": N_BOOTSTRAP,
            "unit": "item (all four anchor cells of an item resampled together)",
            "seed": seed,
            "seed_source": "fit.json column_b.bootstrap.seed",
        },
        "binary_from_fit_json": binary_side(fit),
        "promotion": NO_PROMOTION,
        "elapsed_s": round(time.time() - started, 3),
    }


def run_extract(args) -> int:
    fits_root = Path(args.fits_root).expanduser()
    results_root = Path(args.results_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    done, missing = 0, []
    for model, substrate, cue in CELLS:
        fit_path = fits_root / model / substrate / cue / "fit.json"
        records_path = results_root / model / substrate / cue / "transcripts.jsonl"
        if not fit_path.exists() or not records_path.exists():
            missing.append(f"{model}/{substrate}/{cue}")
            continue
        block = extract_cell(fit_path, records_path)
        out_dir = out_root / model / substrate / cue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "anchor_logprob.json").write_text(json.dumps(block, indent=2) + "\n")
        d = block["denominators"]
        print(
            f"[ok] {block['cell']}: read {d['n_records_read']}, kept {d['n_items_kept']}, "
            f"dropped {d['n_items_dropped']}, records_match_fit_json "
            f"{block['records_match_fit_json']}",
            flush=True,
        )
        done += 1
    print(f"[done] {done} of {len(CELLS)} cells; missing inputs: {missing}", flush=True)
    return 0 if done == len(CELLS) else 1


# --------------------------------------------------------------------------- #
# The document. Numbers are read from the artifacts, prose is fixed text.
# --------------------------------------------------------------------------- #
def num(value: int) -> str:
    """Counts print with thousands separators so a denominator stays readable."""
    return f"{value:,}"


def n4(value) -> str:
    return "not defined" if value is None else f"{value:+.4f}"


def f4(value) -> str:
    return "not defined" if value is None else f"{value:.4f}"


def sci(value) -> str:
    return "not defined" if value is None else f"{value:.3e}"


def interval(block: dict) -> str:
    return f"{n4(block['point'])} [{n4(block['lo'])}, {n4(block['hi'])}]"


def binary_interval(block: dict | None) -> str:
    if not block:
        return "not in fit.json"
    return f"{block['point']:+.4f} [{block['lo']:+.4f}, {block['hi']:+.4f}]"


def drops_table(drops: dict, denominator: int) -> list[str]:
    reasons = [(k, v) for k, v in sorted(drops.items()) if k != "items_dropped"]
    if not reasons:
        return [f"No item was dropped, out of {num(denominator)} records read.", ""]
    out = ["| drop reason | items | of records read |", "|---|---:|---:|"]
    for name, count in reasons:
        out.append(f"| {name} | {num(count)} | {num(denominator)} |")
    out += [
        (f"| **items dropped (any reason)** | **{num(drops.get('items_dropped', 0))}** "
         f"| **{num(denominator)}** |"),
        "",
    ]
    return out


def cell_section(block: dict) -> list[str]:
    d, mass = block["denominators"], block["letter_probability_mass"]
    out = [
        f"### {block['cell']}",
        "",
        (f"Records read {num(d['n_records_read'])}, items kept {num(d['n_items_kept'])}, "
         f"items dropped {num(d['n_items_dropped'])}. Generation job "
         f"{block['generation_job_id']}. "
         f"The records file hashes to the value the cell's fit.json recorded: "
         f"{str(block['records_match_fit_json']).lower()}."),
        "",
    ]
    out += drops_table(d["drops"], d["n_records_read"])
    out += ["Raw letter probability mass on the four anchor prompts (section 4.4).", ""]
    out += ["| anchor cell | n | min | median | max | below 0.01 |", "|---|---:|---:|---:|---:|---:|"]
    for cell in ANCHOR_CELLS:
        m = mass[cell]
        out.append(
            f"| {cell} | {num(m['n'])} | {sci(m['min'])} | {sci(m['median'])} | "
            f"{sci(m['max'])} | {num(m['n_below_0.01'])} |"
        )
    out += ["", "The four cell means on the logprob scale, in nats.", ""]
    out += [
        "| anchor cell | n | mean | bootstrap interval | sd | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in ANCHOR_CELLS:
        c = block["cells"][cell]
        out.append(
            f"| {cell} | {num(c['n'])} | {n4(c['mean'])} | "
            f"[{n4(c.get('boot_lo'))}, {n4(c.get('boot_hi'))}] | {f4(c['sd'])} "
            f"| {n4(c['min'])} | {n4(c['max'])} |"
        )
    out += ["", "The five contrasts of section 22, both scales side by side.", ""]
    out += [
        "| contrast | logprob scale, nats | binary follow scale, probability |",
        "|---|---:|---:|",
    ]
    binary = (block["binary_from_fit_json"].get("contrasts") or {})
    for name in ANCHOR_CONTRAST_DEF:
        out.append(
            f"| {name} | {interval(block['contrasts'][name])} | "
            f"{binary_interval(binary.get(name))} |"
        )
    out += [""]
    return out


def sign_agreement(blocks: list[dict]) -> list[str]:
    """How often the two printed columns of one contrast point the same way.

    This is a description of the two columns above and nothing more: no threshold, no
    test, and no comparison against a Column B estimate. It exists because the sign is
    the one thing a reader can compare across scales without a bridge, and a table that
    prints both columns and says nothing about them invites the reader to invent one.
    """
    out = [
        "| contrast | cells where both scales share the sign | of cells |",
        "|---|---:|---:|",
    ]
    for name in ANCHOR_CONTRAST_DEF:
        same, total = 0, 0
        for block in blocks:
            logprob = block["contrasts"].get(name)
            binary = (block["binary_from_fit_json"].get("contrasts") or {}).get(name)
            if not logprob or not binary:
                continue
            total += 1
            same += int((logprob["point"] >= 0.0) == (binary["point"] >= 0.0))
        out.append(f"| {name} | {same} | {total} |")
    out.append("")
    return out


def totals_section(blocks: list[dict]) -> list[str]:
    total_read = sum(b["denominators"]["n_records_read"] for b in blocks)
    total_kept = sum(b["denominators"]["n_items_kept"] for b in blocks)
    total_drop = sum(b["denominators"]["n_items_dropped"] for b in blocks)
    summed = Counter()
    for b in blocks:
        for reason, count in b["denominators"]["drops"].items():
            if reason != "items_dropped":
                summed[reason] += count
    out = [
        (f"Across the {len(blocks)} cells: {num(total_read)} records read, "
         f"{num(total_kept)} items kept, {num(total_drop)} items dropped."),
        "",
    ]
    if not summed:
        out += [
            ("No item was dropped in any cell, so every required check of part 4.5 step 2 "
             "held on every record in every one of the four anchor cells."),
            "",
        ]
        return out
    out += ["| drop reason | items, summed over cells | of records read |", "|---|---:|---:|"]
    for reason, count in sorted(summed.items()):
        out.append(f"| {reason} | {num(count)} | {num(total_read)} |")
    out += [
        (f"| **items dropped (any reason)** | **{num(total_drop)}** "
         f"| **{num(total_read)}** |"),
        "",
    ]
    return out


def one_or_list(values: set) -> str:
    """One value printed bare, several printed as a list, so the sentence stays honest."""
    ordered = sorted(values)
    return str(ordered[0]) if len(ordered) == 1 else ", ".join(str(v) for v in ordered)


def provenance(blocks: list[dict], meta: dict | None) -> list[str]:
    """Where these numbers came from, read out of the artifacts rather than retyped."""
    matched = sum(1 for b in blocks if b["records_match_fit_json"])
    scripts = one_or_list({b["analysis_script_sha256"] for b in blocks})
    out = [
        (f"The records file of {matched} of the {len(blocks)} cells hashes to the value "
         "that cell's own `fit.json` recorded, so these margins and the binary numbers "
         "beside them were read off the same bytes."),
        "",
        f"Extraction script sha256 {scripts}.",
        "",
    ]
    if meta:
        out += [
            (f"Cluster job {meta.get('job_id')} on {meta.get('node')}, "
             f"{meta.get('cpus_per_task')} CPUs and {meta.get('mem_per_node_mb')} MB, "
             f"partition requested {meta.get('partition_requested')} and allocated "
             f"{meta.get('partition_allocated')}, started {meta.get('started')} and "
             f"finished {meta.get('finished')} with exit code {meta.get('exit_code')}. "
             f"Python {meta.get('python')}, numpy {meta.get('numpy')}. "
             f"GPU used: {meta.get('gpu')}."),
            "",
        ]
    return out


def intro_and_method(seeds: str, reps: str) -> list[str]:
    """Sections 1 and 2: what this is, what Y is, and what a record had to carry."""
    return [
        "# The element 21 anchor contrast on the logprob scale",
        "",
        ("Generated by `experiments/logprob_anchor.py --mode report` from the artifacts "
         "under `experiments/results/logprob-anchor/`. No number below is typed by hand."),
        "",
        "## 1. What this is, and what it is not",
        "",
        ("This is Job A of `docs/OUTCOME-SCALE-NOTE.md` part 4.5. It reads the four anchor "
         "cells of each arms record, takes the letter-logprob block those records already "
         "carry, and reports the four cell means and the five section 22 contrasts on the "
         "logprob scale, printed beside the binary ones from `fit.json` and never instead "
         "of them. No generation pass ran and no record was rewritten."),
        "",
        NO_PROMOTION,
        "",
        ("The cells are listed in the declaration order of `cells18_fits.CELLS`, which is "
         "the order that file writes them in and not an ordering by any measured value. "
         "Section 25 forbids a cross-model ordering off these numbers and none is made "
         "here."),
        "",
        "## 2. What Y is, and what a record had to carry to be counted",
        "",
        Y_DEFINITION,
        "",
        ("An item enters only when all four of its anchor cells pass every check of part "
         "4.5 step 2: `intervention_level` is `logit`, `outcome_scale` is "
         "`logprob_margin`, no `unavailable_reason` is present, `target_letter` equals the "
         "record's own `hint_label`, the scored letter set equals the item's allowed "
         "labels, and every logprob was read off a token that decodes to its own letter "
         "(the same `_strip_token` comparison the client made when it wrote the value). "
         "Two structural checks come first, a missing `hint_label` and a missing choice "
         "list, and a null `logprob_margin` is a failure too because there would be no Y. "
         "Any failure drops the ITEM, not the cell, and is counted by reason the way "
         "`wave1_fits.build_table` counts drops today."),
        "",
        (f"mu_ab is the item mean of Y_c. Intervals come from a {reps} replicate ITEM "
         f"bootstrap at seed {seeds}, which is the seed each cell's own `fit.json` records "
         "under `column_b.bootstrap.seed`. One resample index feeds all four cells, so the "
         "contrasts are paired the way the binary ones are."),
        "",
        MASS_CAVEAT,
        "",
        "## 3. Provenance",
        "",
    ]


def closing_section() -> list[str]:
    """Section 7: what the tables above establish, and what they do not."""
    return [
        "## 7. What this establishes, and what it does not",
        "",
        ("**What it establishes.** The four anchor cells of every counted item carry a "
         "letter-logprob margin toward the same designated option, measured on the same "
         "four prompts the binary anchor was read on, so the four means and the five "
         "contrasts above are the same design read on a second outcome scale. Where the "
         "binary outcome in the mu00 cell is 0 by construction, this scale still varies, "
         "and the sd column of each cell table says by how much. That is the A4.6(b) "
         "degeneracy and its absence, measured on the same items."),
        "",
        ("**What it does not establish.** No promotion, and no margin. Part 4.5 step 6 "
         "holds the element 21 comparison until a margin exists on this scale, and A5.6 "
         "declines to set one, so nothing above is compared with a Column B estimate and "
         "no verdict is printed. Section 22.1's 0.10 belongs to the probability scale it "
         "was derived on."),
        "",
        ("**The caveat of section 4.4, which a reader must not skip past.** The mass "
         "tables above are part of the result, not decoration. A margin computed where "
         "the answer letters hold a tiny share of the next-token mass is a well defined "
         "conditional quantity, and it is also a quantity about a region the model almost "
         "never enters: the renormalization does the work. Whether that makes it a good "
         "outcome is an empirical question this document does not settle. Whether it is "
         "reportable without the mass beside it is settled, and the answer is no."),
        "",
        ("**What this is not evidence about.** These are anchor cells, which are replay "
         "constructions, so `joint_replay_regime` is explicitly not the native cue total "
         "effect absent a generation-to-replay bridge (element 21). A native logprob-scale "
         "Column B needs the generation pass of part 4.5 Job B, which has not run."),
        "",
    ]


def render(blocks: list[dict], meta: dict | None = None) -> str:
    seeds = one_or_list({b["bootstrap"]["seed"] for b in blocks})
    reps = one_or_list({b["bootstrap"]["n_replicates"] for b in blocks})
    lines = intro_and_method(seeds, reps)
    lines += provenance(blocks, meta)
    lines += ["## 4. Drop counts with their denominators", ""]
    lines += totals_section(blocks)
    lines += ["## 5. Per cell", ""]
    for block in blocks:
        lines += cell_section(block)
    lines += [
        "## 6. How often the two columns point the same way",
        "",
        ("The sign is the one thing that can be read across the two scales without a "
         "bridge, so it is counted here and nothing else is. This is a description of the "
         "columns printed in section 5, not a test, and it carries no threshold."),
        "",
    ]
    lines += sign_agreement(blocks)
    lines += closing_section()
    return "\n".join(lines) + "\n"


def run_report(args) -> int:
    out_root = Path(args.out_root).expanduser()
    blocks = []
    for model, substrate, cue in CELLS:
        path = out_root / model / substrate / cue / "anchor_logprob.json"
        if path.exists():
            blocks.append(json.loads(path.read_text()))
    if not blocks:
        print(f"[fail] no anchor_logprob.json under {out_root}", flush=True)
        return 1
    meta_path = out_root / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    doc = Path(args.doc).expanduser()
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render(blocks, meta))
    print(f"[ok] wrote {doc} from {len(blocks)} of {len(CELLS)} cells", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("extract", "report"), required=True)
    ap.add_argument("--fits-root", default="experiments/results/cells18-fits")
    ap.add_argument("--results-root", default="~/bcf/results")
    ap.add_argument("--out-root", default="experiments/results/logprob-anchor")
    ap.add_argument("--doc", default="docs/LOGPROB-ANCHOR.md")
    args = ap.parse_args()
    return run_extract(args) if args.mode == "extract" else run_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
