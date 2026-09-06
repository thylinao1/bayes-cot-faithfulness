"""Run the synthetic judge gate for one or more judges and write the report.

The report carries a numerator and a denominator for every number, the measured votes per
second per server (which is what the judging budget table in `budget.md` is recomputed
from), the prompt SHA-256s, the pinned judge revision, and the SHA-256 of the thresholds
file so that a later edit to the thresholds is visible in the diff of any report.

A judge that misses a threshold is reported FAIL with its numbers. Nothing is retuned.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import records as rec
from .aggregate import test_retest
from .family_map import JUDGE_BY_KEY
from .gate_thresholds import HIGHER_IS_BETTER, THRESHOLDS, thresholds_sha256
from .prompt_files import load_default_prompts
from .runner import JuryItem, JuryRunner, load_items
from .synthetic_gate import TRUTH


def _first_run_votes(rows: list[dict], question: str) -> list[dict]:
    """Run 0, unswapped: the votes the panel label would use."""
    return [r for r in rows if r.get("question") == question
            and r.get("run_idx") == 0 and not r.get("position_swap")]


def per_class_counts(rows: list[dict], items_by_id: dict[str, dict]) -> dict:
    """Q1 and gate outcomes per gate class, with every denominator kept."""
    out: dict[str, dict] = {}
    for row in rows:
        item = items_by_id.get(row["item_id"])
        if item is None:
            continue
        cls = item["meta"]["gate_class"]
        bucket = out.setdefault(cls, {
            "q1": {"yes": 0, "no": 0, "unavailable": 0, "total": 0},
            "gate": {"coherent": 0, "silent_override": 0, "unavailable": 0, "total": 0},
            "q2": {"supported": 0, "unsupported": 0, "uncertain": 0, "unavailable": 0, "total": 0},
        })
        q = row["question"]
        vote = row["vote"]
        key = {"Q1": "q1", "gate": "gate", "Q2": "q2"}.get(q)
        if key is None:
            continue
        bucket[key]["total"] += 1
        if vote in rec.UNAVAILABLE_VOTES:
            bucket[key]["unavailable"] += 1
        elif vote in bucket[key]:
            bucket[key][vote] += 1
    return out


def score_judge(rows: list[dict], items_by_id: dict[str, dict], judge_key: str) -> dict:
    """Every gate metric for one judge, each as numerator, denominator and rate."""
    mine = [r for r in rows if r.get("judge_key") == judge_key]
    first = [r for r in mine if r.get("run_idx") == 0 and not r.get("position_swap")]
    counts = per_class_counts(first, items_by_id)
    metrics: dict[str, dict] = {}

    def add(name: str, num: int, den: int) -> None:
        metrics[name] = {
            "numerator": num, "denominator": den,
            "rate": (num / den) if den else None,
        }

    for cls, truth in TRUTH.items():
        c = counts.get(cls)
        if not c:
            continue
        avail_q1 = c["q1"]["yes"] + c["q1"]["no"]
        if truth.get("q1") is True:
            add(f"recall_{cls}", c["q1"]["yes"], avail_q1)
        elif truth.get("q1") is False:
            add(f"specificity_{cls}", c["q1"]["no"], avail_q1)
        gate_truth = truth.get("gate")
        if gate_truth:
            avail_gate = c["gate"]["coherent"] + c["gate"]["silent_override"]
            add(f"gate_accuracy_{cls}", c["gate"][gate_truth], avail_gate)

    malformed = sum(1 for r in mine if r.get("vote") == rec.MALFORMED)
    abstain = sum(1 for r in mine if r.get("vote") == rec.ABSTAIN)
    add("malformed_rate_max", malformed, len(mine))
    add("abstain_rate", abstain, len(mine))
    agree, multi = test_retest(mine, question="Q1")
    add("test_retest_q1_min", agree, multi)

    swap_rows = [r for r in mine if r.get("position_swap")]
    plain_by_key = {
        (r["item_id"], r["question"], r["run_idx"]): r["vote"]
        for r in mine if not r.get("position_swap")
    }
    swap_agree = sum(
        1 for r in swap_rows
        if plain_by_key.get((r["item_id"], r["question"], r["run_idx"])) == r["vote"]
    )
    add("position_swap_agreement", swap_agree, len(swap_rows))

    verdicts = {}
    for name, bar in THRESHOLDS.items():
        m = metrics.get(name)
        if m is None or m["rate"] is None:
            verdicts[name] = {"verdict": "NO DATA", "threshold": bar, **(m or {})}
            continue
        higher = name in HIGHER_IS_BETTER
        passed = m["rate"] >= bar if higher else m["rate"] <= bar
        verdicts[name] = {
            "verdict": "PASS" if passed else "FAIL",
            "threshold": bar, "direction": "at least" if higher else "at most", **m,
        }
    failures = [k for k, v in verdicts.items() if v["verdict"] == "FAIL"]
    no_data = [k for k, v in verdicts.items() if v["verdict"] == "NO DATA"]
    return {
        "judge_key": judge_key,
        "judge_model": JUDGE_BY_KEY[judge_key].hf_id,
        "judge_revision": JUDGE_BY_KEY[judge_key].revision,
        "votes": len(mine),
        "per_class_counts": counts,
        "metrics": metrics,
        "gate_verdicts": verdicts,
        "failed_metrics": failures,
        "no_data_metrics": no_data,
        "verdict": "FAIL" if failures else ("INCOMPLETE" if no_data else "PASS"),
    }


def build_report(out_dir: Path, items: list[dict], run_summary: dict, judge_keys: list[str]) -> dict:
    rows = rec.read_votes(out_dir / "votes.jsonl")
    items_by_id = {i["item_id"]: i for i in items}
    per_judge = [score_judge(rows, items_by_id, k) for k in judge_keys]
    class_sizes: dict[str, int] = {}
    for item in items:
        cls = item["meta"]["gate_class"]
        class_sizes[cls] = class_sizes.get(cls, 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "thresholds": THRESHOLDS,
        "thresholds_sha256": thresholds_sha256(),
        "prompt_sha256": {q: p.sha256 for q, p in load_default_prompts().items()},
        "corpus": {"items": len(items), "per_class": class_sizes},
        "run_summary": run_summary,
        "votes_per_second_per_server": run_summary.get("votes_per_second"),
        "per_judge": per_judge,
        "verdict": "FAIL" if any(j["verdict"] == "FAIL" for j in per_judge) else (
            "INCOMPLETE" if any(j["verdict"] == "INCOMPLETE" for j in per_judge) else "PASS"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    from .backends import vllm_endpoint

    ap = argparse.ArgumentParser(description="Run the synthetic judge gate.")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True, help="must contain /<substrate>/<cue_family>")
    ap.add_argument("--judge", action="append", required=True, help="judge_key=base_url")
    ap.add_argument("--substrate", default="arc_challenge")
    ap.add_argument("--cue-family", default="stated-hint")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    raw = [json.loads(x) for x in Path(args.items).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        raw = raw[: args.limit]
    items = [JuryItem(**dict(r)) for r in raw]
    endpoints = {}
    judge_keys = []
    for spec in args.judge:
        key, _, url = spec.partition("=")
        if key not in JUDGE_BY_KEY:
            raise SystemExit(f"unknown judge key {key!r}")
        judge_keys.append(key)
        if not args.report_only:
            endpoints[key] = vllm_endpoint(key, url, seed=args.seed)
    out_dir = Path(args.out)
    runner = JuryRunner(
        endpoints=endpoints, prompts=load_default_prompts(), out_dir=out_dir,
        substrate=args.substrate, cue_family=args.cue_family, seed=args.seed,
        mode="three-seeded", position_swap="first-run", num_predict=args.num_predict,
        resume=args.resume or args.report_only,
        # The gate serves one judge at a time under the card budget, so it scores exactly
        # the judges it was given and marks the labels partial.
        judge_filter=tuple(judge_keys) if not args.report_only else None,
    )
    if args.report_only:
        summary = json.loads((runner.out_dir / "run_summary.json").read_text()) \
            if (runner.out_dir / "run_summary.json").exists() else {}
    else:
        summary = runner.run(items, concurrency=args.concurrency)
        (runner.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    report = build_report(runner.out_dir, raw, summary, judge_keys)
    (runner.out_dir / "gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"],
        "per_judge": {j["judge_key"]: {"verdict": j["verdict"], "failed": j["failed_metrics"]}
                      for j in report["per_judge"]},
        "votes_per_second": report["votes_per_second_per_server"],
        "corpus": report["corpus"],
    }, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
