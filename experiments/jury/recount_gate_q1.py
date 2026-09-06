"""Recount the Q1 gate rows straight from a votes.jsonl, so a check does not have to
trust the report it is checking.

Why this exists. The gate metrics are not all counted in the same direction. For a class
whose Q1 truth is TRUE the metric is `recall_<class>` and its numerator is the YES votes.
For a class whose Q1 truth is FALSE the metric is `specificity_<class>` and its numerator
is the NO votes. `specificity_restated_cue_only` 26/69 and "43 of 69 items got a yes" are
therefore the SAME measurement stated from the two ends, not a disagreement, and reading
one against the other looks like a large error when nothing is wrong.

The second trap is the run. Every gate metric is computed on run 0 unswapped, which is the
vote a panel label would actually use. The corpus is also scored on three seeded
temperature-0 runs for test-retest, so a count of "items with at least one yes anywhere"
takes the union over three runs and can exceed the run-0 count by the number of rows that
did not repeat. Both numbers are printed below, side by side, with their denominators.

    python -m experiments.jury.recount_gate_q1 \
        experiments/results/jury-gate/llama-3.3-70b-fp8-q1b/arc_challenge/stated-hint/votes.jsonl

Add --json for machine-readable output. The gate items file supplies each item's class and
defaults to experiments/results/jury-gate/gate_items.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .synthetic_gate import TRUTH

DEFAULT_ITEMS = Path("experiments/results/jury-gate/gate_items.jsonl")


def load_classes(items_path: Path) -> dict[str, str]:
    """item_id to gate_class, read from the built corpus."""
    out: dict[str, str] = {}
    with items_path.open() as fh:
        for line in fh:
            item = json.loads(line)
            out[item["item_id"]] = item["meta"]["gate_class"]
    return out


def recount(votes_path: Path, classes: dict[str, str], judge_key: str | None = None) -> dict:
    """Run 0 unswapped yes/no per class, plus the union-over-runs yes count."""
    run0: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0, "other": 0})
    union_yes: dict[str, set[str]] = defaultdict(set)
    seen_items: dict[str, set[str]] = defaultdict(set)
    total_votes = 0

    with votes_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            total_votes += 1
            if judge_key and row.get("judge_key") != judge_key:
                continue
            if row.get("question") != "Q1":
                continue
            cls = classes.get(row["item_id"])
            if cls is None:
                continue
            seen_items[cls].add(row["item_id"])
            vote = row.get("vote")
            if vote == "yes":
                union_yes[cls].add(row["item_id"])
            if row.get("run_idx") == 0 and not row.get("position_swap"):
                key = vote if vote in ("yes", "no") else "other"
                run0[cls][key] += 1

    per_class = {}
    for cls in sorted(seen_items):
        counts = run0[cls]
        available = counts["yes"] + counts["no"]
        truth = TRUTH.get(cls, {}).get("q1")
        if truth is True:
            metric, numerator = f"recall_{cls}", counts["yes"]
        elif truth is False:
            metric, numerator = f"specificity_{cls}", counts["no"]
        else:
            metric, numerator = None, None
        per_class[cls] = {
            "q1_truth": truth,
            "items": len(seen_items[cls]),
            "run0_yes": counts["yes"],
            "run0_no": counts["no"],
            "run0_unavailable": counts["other"],
            "run0_available": available,
            "gate_metric": metric,
            "gate_numerator": numerator,
            "gate_denominator": available,
            "union_any_run_yes": len(union_yes[cls]),
        }
    return {"votes_file": str(votes_path), "total_vote_rows": total_votes, "per_class": per_class}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("votes", type=Path, help="path to a votes.jsonl")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS, help="gate items jsonl")
    ap.add_argument("--judge", default=None, help="restrict to one judge_key")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    result = recount(args.votes, load_classes(args.items), args.judge)
    if args.as_json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"{result['votes_file']}  ({result['total_vote_rows']} vote rows)")
    header = f"{'class':24s} {'truth':6s} {'run0 yes':>9s} {'run0 no':>8s} {'gate metric':>34s} {'union yes':>10s}"
    print(header)
    print("-" * len(header))
    for cls, c in result["per_class"].items():
        metric = f"{c['gate_metric']} {c['gate_numerator']}/{c['gate_denominator']}" if c["gate_metric"] else "-"
        truth = {True: "yes", False: "no", None: "-"}[c["q1_truth"]]
        print(f"{cls:24s} {truth:6s} {c['run0_yes']:9d} {c['run0_no']:8d} {metric:>34s} {c['union_any_run_yes']:10d}")
    print()
    print("run0 yes plus run0 no is the metric denominator. For a truth-false class the gate")
    print("numerator is run0 no, NOT run0 yes. union yes takes all three seeded runs.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
