#!/usr/bin/env python
"""Ruling R3(ii): turn one enrichment pass into the item list its cells will read.

An enrichment pass is a normal cell run with ONE arm enabled, `sampling`, over the whole
1,500-item pool of a substrate (`bcf/waves/enrich-*.tsv`). It leaves the per-item element
9.2 blocks in the run directory. This script reads them back and writes three files:

  enrichment_items.jsonl   one line per scored item: pool index, question hash,
                           normalized entropy, modal answer, whether the mode is correct,
                           the stratum, and the right-but-uncertain flag at 0.30
  uncertain_items.json     the LIST FILE: exactly the right-but-uncertain items, in pool
                           order, in the schema `bayes_cot_faithfulness.item_list` reads
                           (`08_additive_arms.py --item-list`)
  enrichment_report.json   the counts and the denominators behind them

It computes nothing about entropy itself. Every per-item quantity is the one the sampling
arm already wrote through `summarize_item_samples`, so this script cannot disagree with
the arm; what it adds is the POOL INDEX, which the arm has no reason to know, by matching
each record's (question, choices) key back to the substrate file the pass ran on.

An item the pass never scored (a clean-incorrect item, or one the substrate pass could
not generate) is reported in the counts and is NOT in the list. Right-but-uncertain is a
conjunction that includes a correct mode, so a clean-incorrect item can only ever be
excluded; saying so in the counts is what keeps the exclusion visible.

  python bcf/enrich_report.py --out-dir <cell dir> --data experiments/data/arc_challenge.json
  echo $?   # 0 wrote the files, 3 nothing scorable was found, 2 bad usage/inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from bayes_cot_faithfulness.item_list import (  # noqa: E402
    SCHEMA,
    SELECTOR_RIGHT_BUT_UNCERTAIN,
    question_sha16,
)
from bayes_cot_faithfulness.sampling_arm import (  # noqa: E402
    SAMPLING_K,
    SAMPLING_TEMPERATURE,
    UNCERTAIN_ENTROPY_THRESHOLD,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOTHING_SCORED = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_record_file(out_dir: Path) -> Path | None:
    """The run artifact holding the per-item sampling blocks, transcripts first.

    The published transcripts are preferred because they are what the analysis reads;
    the checkpoint is the fallback because it exists even when a pass stopped before the
    cue pass, and it carries the same `sampling` block under the same key.
    """
    for pattern in ("arms_transcripts_*.json", "arms_checkpoint_*.json"):
        hits = sorted(out_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):  # a checkpoint
        rows = payload.get("records", [])
    else:  # a transcripts array
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} holds no record array")
    return rows


def index_pool(pool: list[dict]) -> dict[tuple[str, tuple[str, ...]], int]:
    """Pool position by (question, choices), the same key the resume merge uses."""
    out: dict[tuple[str, tuple[str, ...]], int] = {}
    for i, row in enumerate(pool):
        key = (row["question"], tuple(row["choices"]))
        out.setdefault(key, i)  # a duplicate key keeps its FIRST position
    return out


def item_rows(records: list[dict], by_key: dict) -> tuple[list[dict], dict]:
    """One row per record that carries a sampling block, plus the accounting."""
    rows = []
    n_no_block = 0
    n_unmatched = 0
    for r in records:
        block = r.get("sampling")
        if not block:
            n_no_block += 1
            continue
        key = (r["question"], tuple(r["choices"]))
        idx = by_key.get(key)
        if idx is None:
            n_unmatched += 1
            continue
        rows.append({
            "index": idx,
            "question_sha16": question_sha16(r["question"]),
            "answer_label": r.get("answer_label"),
            "clean_correct": r.get("clean_correct"),
            "k": block.get("k"),
            "n_scorable": block.get("n_scorable"),
            "n_unscorable": block.get("n_unscorable"),
            "n_out_of_set": block.get("n_out_of_set"),
            "n_options": block.get("n_options"),
            "normalized_entropy": block.get("normalized_entropy"),
            "modal_answer": block.get("modal_answer"),
            "modal_tie": block.get("modal_tie"),
            "modal_correct": block.get("modal_correct"),
            "entropy_threshold": block.get("entropy_threshold"),
            "right_but_uncertain": bool(block.get("right_but_uncertain")),
            "stratum": block.get("stratum"),
        })
    rows.sort(key=lambda r: r["index"])
    return rows, {"n_records": len(records), "n_without_sampling_block": n_no_block,
                  "n_not_matched_to_the_pool": n_unmatched}


def build_list_file(rows: list[dict], *, pool_path: Path, pool: list[dict],
                    substrate: str, model: str | None, run_meta: dict) -> dict:
    uncertain = [r for r in rows if r["right_but_uncertain"]]
    thresholds = {r["entropy_threshold"] for r in rows if r["entropy_threshold"] is not None}
    return {
        "schema": SCHEMA,
        "selector": SELECTOR_RIGHT_BUT_UNCERTAIN,
        "substrate": substrate,
        "pool_path": str(pool_path),
        "pool_sha256": _sha256(pool_path),
        "pool_size": len(pool),
        "model": model,
        "sampling_k": run_meta.get("sampling_k", SAMPLING_K),
        "sampling_temperature": run_meta.get("sampling_temperature", SAMPLING_TEMPERATURE),
        "prompt": "clean",
        # The threshold that the ARM used, read back off the records rather than restated
        # here, so a list built by an arm running a different threshold cannot be labelled
        # with the frozen one. The frozen value is carried beside it for comparison.
        "entropy_threshold_used": (sorted(thresholds)[0] if len(thresholds) == 1
                                   else sorted(thresholds)),
        "entropy_threshold_frozen": UNCERTAIN_ENTROPY_THRESHOLD,
        "source_job_id": run_meta.get("job_id"),
        "source_plan_commit": run_meta.get("plan_commit"),
        "source_run_label": run_meta.get("run_label"),
        "n_items": len(uncertain),
        "items": [{"index": r["index"], "question_sha16": r["question_sha16"]}
                  for r in uncertain],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="the enrichment pass's run directory (holds the arm artifacts)")
    ap.add_argument("--data", type=Path, required=True,
                    help="the substrate pool the pass ran on; the index in the list file "
                         "is a position in THIS file")
    ap.add_argument("--substrate", default=None,
                    help="substrate name for the list file (default: the --data stem)")
    ap.add_argument("--records", type=Path, default=None,
                    help="read the records from this file instead of discovering them")
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    if not out_dir.is_dir():
        print(f"[enrich] REFUSING: no such run directory: {out_dir}")
        return EXIT_USAGE
    if not args.data.is_file():
        print(f"[enrich] REFUSING: no such pool file: {args.data}")
        return EXIT_USAGE
    record_file = args.records or find_record_file(out_dir)
    if record_file is None or not record_file.is_file():
        print(f"[enrich] REFUSING: no arms transcripts or checkpoint in {out_dir}; the "
              "pass wrote nothing this script can read.")
        return EXIT_USAGE

    pool = json.loads(args.data.read_text())
    records = load_records(record_file)
    rows, accounting = item_rows(records, index_pool(pool))
    if not rows:
        print(f"[enrich] {record_file.name} holds {accounting['n_records']} record(s) and "
              "NONE carries a sampling block. Nothing was written; this pass has not run "
              "its arm yet.")
        return EXIT_NOTHING_SCORED

    run_meta = {}
    meta_path = out_dir / "run_meta.json"
    if meta_path.is_file():
        run_meta = json.loads(meta_path.read_text())
    substrate = args.substrate or args.data.stem
    list_file = build_list_file(rows, pool_path=args.data, pool=pool, substrate=substrate,
                               model=run_meta.get("model"), run_meta=run_meta)

    items_path = out_dir / "enrichment_items.jsonl"
    items_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    list_path = out_dir / "uncertain_items.json"
    list_path.write_text(json.dumps(list_file, indent=2))

    n_uncertain = list_file["n_items"]
    n_modal_correct = sum(1 for r in rows if r["modal_correct"])
    report = {
        "run_dir": str(out_dir),
        "record_file": str(record_file),
        "pool": str(args.data),
        "pool_sha256": list_file["pool_sha256"],
        "pool_size": len(pool),
        "substrate": substrate,
        "model": run_meta.get("model"),
        "job_id": run_meta.get("job_id"),
        "n_items_scored": len(rows),
        "n_modal_correct": n_modal_correct,
        "n_right_but_uncertain": n_uncertain,
        "right_but_uncertain_fraction": (n_uncertain / len(rows)) if rows else None,
        "coverage_of_the_pool": len(rows) / len(pool) if pool else None,
        "entropy_threshold_used": list_file["entropy_threshold_used"],
        "accounting": accounting,
        "list_file": str(list_path),
        "items_file": str(items_path),
    }
    (out_dir / "enrichment_report.json").write_text(json.dumps(report, indent=2))

    print(f"[enrich] {len(rows)}/{len(pool)} pool item(s) scored by the sampling arm; "
          f"{n_modal_correct} with a correct mode; {n_uncertain} right-but-uncertain at "
          f"{list_file['entropy_threshold_used']}")
    print(f"[enrich] wrote {items_path.name}, {list_path.name} and enrichment_report.json")
    if n_uncertain == 0:
        print("[enrich] NOTE: the list file selects nothing, so there is no enrichment "
              "cell to run for this model and substrate. That is a finding, not a "
              "failure: this model's uncertain stratum is empty on this pool.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
