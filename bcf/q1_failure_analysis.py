"""Per-item failure analysis for the two Q1 gate classes that fail on every judge.

Reads the run-0, unswapped Q1 votes out of every banked gate run and, for the two classes
that carry the failing metrics, lists which corpus item each judge x prompt-file pair got
wrong. Nothing is relabelled here: the corpus truth is read off `meta.truth.q1` and the
scoring direction is the one `gate.py` uses (recall counts YES on a yes-truth class,
specificity counts NO on a no-truth class).

    python bcf/q1_failure_analysis.py --out experiments/results/jury-prescreen/failure_analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

VOTES_ROOT = Path("/Users/maksimsilchenko/Developer/bcf-jury/experiments/results/jury-gate")
ITEMS = Path("experiments/results/jury-gate/gate_items.jsonl")

# The banked runs of record, by (judge_key, q1 prompt file letter) -> votes.jsonl.
RUNS: dict[tuple[str, str], str] = {
    ("llama-3.3-70b-fp8", "a"): "llama-3.3-70b-fp8",
    ("llama-3.3-70b-fp8", "b"): "llama-3.3-70b-fp8-q1b",
    ("llama-3.3-70b-fp8", "c"): "llama-3.3-70b-fp8-q1c",
    ("gemma-3-27b-it", "a"): "gemma-3-27b-it-h200-q1a",
    ("gemma-3-27b-it", "b"): "gemma-3-27b-it-h200-q1b",
    ("gemma-3-27b-it", "c"): "gemma-3-27b-it-h200-q1c",
    ("gpt-oss-20b", "a"): "gpt-oss-20b-h100-47-q1a",
    ("gpt-oss-20b", "b"): "gpt-oss-20b-h100-47-q1b",
    ("gpt-oss-20b", "c"): "gpt-oss-20b-h100-47-q1c",
    ("qwen3-32b", "a"): "qwen3-32b-h200-np1024-q1a",
    ("qwen3-32b", "b"): "qwen3-32b-h200-np1024-q1b",
    ("qwen3-32b", "c"): "qwen3-32b-h200-np1024-q1c",
}

FOCUS = ("paraphrased_disclosure", "restated_cue_only")
UNAVAILABLE = {"malformed", "abstain", "", None}


def load_items() -> dict[str, dict]:
    return {
        r["item_id"]: r
        for r in (json.loads(x) for x in ITEMS.read_text(encoding="utf-8").splitlines() if x.strip())
    }


def load_q1_run0(slug: str) -> dict[str, dict]:
    """item_id -> the run-0, unswapped Q1 vote row for one banked run."""
    path = VOTES_ROOT / slug / "arc_challenge" / "stated-hint" / "votes.jsonl"
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("question") != "Q1" or r.get("run_idx") != 0 or r.get("position_swap"):
            continue
        out[r["item_id"]] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    items = load_items()
    per_class_items = {c: sorted(i for i, v in items.items() if v["meta"]["gate_class"] == c)
                       for c in FOCUS}

    report: dict = {
        "votes_root": str(VOTES_ROOT),
        "items_file": str(ITEMS),
        "runs": {},
        "class_sizes": {c: len(v) for c, v in per_class_items.items()},
    }
    # failure counts per item, across the 12 judge x file cells
    fail_count: dict[str, Counter] = {c: Counter() for c in FOCUS}
    fail_by_cell: dict[str, dict[str, list[str]]] = {c: {} for c in FOCUS}
    votes_by_cell: dict[str, dict[str, dict[str, str]]] = {c: {} for c in FOCUS}

    for (judge, letter), slug in sorted(RUNS.items()):
        cell = f"{judge}|q1{letter}"
        run = load_q1_run0(slug)
        report["runs"][cell] = {"slug": slug, "q1_rows_run0": len(run)}
        for cls in FOCUS:
            truth_yes = bool(items[per_class_items[cls][0]]["meta"]["truth"]["q1"])
            want = "yes" if truth_yes else "no"
            failing, avail, unavail = [], 0, 0
            cellvotes = {}
            for iid in per_class_items[cls]:
                row = run.get(iid)
                vote = (row or {}).get("vote")
                cellvotes[iid] = vote if vote is not None else "MISSING"
                if vote in UNAVAILABLE or row is None:
                    unavail += 1
                    continue
                avail += 1
                if vote != want:
                    failing.append(iid)
            fail_by_cell[cls][cell] = failing
            votes_by_cell[cls][cell] = cellvotes
            for iid in failing:
                fail_count[cls][iid] += 1
            report["runs"][cell][cls] = {
                "truth": want,
                "scorable": avail,
                "unavailable": unavail,
                "correct": avail - len(failing),
                "failing": len(failing),
                "denominator_class": len(per_class_items[cls]),
            }

    report["fail_by_cell"] = fail_by_cell
    report["votes_by_cell"] = votes_by_cell
    report["fail_count_per_item"] = {c: dict(fail_count[c].most_common()) for c in FOCUS}
    report["always_failing"] = {}
    report["never_failing"] = {}
    for cls in FOCUS:
        cells_scoring = {cell: fail_by_cell[cls][cell] for cell in fail_by_cell[cls]}
        # how many cells actually had this item scorable
        scorable_cells: dict[str, int] = defaultdict(int)
        for cell, cv in votes_by_cell[cls].items():
            for iid, v in cv.items():
                if v not in UNAVAILABLE and v != "MISSING":
                    scorable_cells[iid] += 1
        report["always_failing"][cls] = sorted(
            i for i in per_class_items[cls]
            if scorable_cells[i] > 0 and fail_count[cls][i] == scorable_cells[i]
        )
        report["never_failing"][cls] = sorted(
            i for i in per_class_items[cls]
            if scorable_cells[i] > 0 and fail_count[cls][i] == 0
        )
        report.setdefault("scorable_cells_per_item", {})[cls] = dict(scorable_cells)
        _ = cells_scoring
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for cls in FOCUS:
        print(f"== {cls}: {len(per_class_items[cls])} items")
        for cell in sorted(report["runs"]):
            s = report["runs"][cell][cls]
            print(f"   {cell:28s} correct {s['correct']:>3d}/{s['scorable']:<3d} "
                  f"failing {s['failing']:>3d}  unavailable {s['unavailable']:>3d}")
        print(f"   always failing where scorable: {len(report['always_failing'][cls])}")
        print(f"   never failing anywhere:        {len(report['never_failing'][cls])}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
