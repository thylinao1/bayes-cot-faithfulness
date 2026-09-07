"""The gate scored on the PANEL label rather than on one judge at a time.

Section 6.2 says the label that reaches the analysis is the majority of the available votes
of every judge NOT of the subject model's family. Every gate number on the record so far is
a per-judge number, so the record does not yet say whether the instrument that will actually
be used clears the bars. This computes that, and nothing else: it runs no model, submits no
job and moves no threshold. It reads vote files that already exist.

The gate corpus's subject_model is Qwen3-8B, so the panel is Gemma, gpt-oss and Llama and
the Qwen judge is own-family and excluded. Passing a Qwen vote directory is REFUSED rather
than quietly dropped, because a panel that silently included it would not be the panel rule.

Three rules are inherited unchanged from `aggregate.py` and section 6.6:

* the label is the majority of the AVAILABLE votes, and malformed and abstain are both
  unavailable, so the panel size for a row is what it actually was;
* a tie resolves to the coherence-gate outcome, carries `is_tie`, and is counted; with two
  judges left after a leave-one-out that case is common, which is exactly why it is reported;
* a row with no available vote has no label and is counted as unlabeled.

Two of the ten thresholds have no single panel meaning and the choice made here is stated in
the report rather than buried:

* `malformed_rate_max` is scored POOLED over the panel's judges, numerator the malformed
  votes and denominator every vote those judges cast. The row-level analogue, the count of
  rows the panel could not label, is reported beside it as `panel_unlabeled` and is not
  scored against the 0.05 bar.
* `test_retest_q1_min` is the panel label recomputed per run index and compared across runs,
  which is the panel analogue of a judge repeating itself.

    python -m experiments.jury.panel_gate --q1 a \
        --votes llama-3.3-70b-fp8=experiments/results/jury-gate/llama-3.3-70b-fp8/arc_challenge/stated-hint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import records as rec
from .aggregate import aggregate
from .family_map import JUDGE_BY_KEY, routing
from .gate_thresholds import HIGHER_IS_BETTER, THRESHOLDS, thresholds_sha256
from .prompt_files import PROMPT_DIR, Q1_PROMPT_FILES, load_prompt
from .synthetic_gate import CLASSES, TRUTH

DEFAULT_ITEMS = Path("experiments/results/jury-gate/gate_items.jsonl")
SUBJECT_MODEL = "Qwen3-8B"
SUBJECT_FAMILY = "Qwen"
# What section 6.2 routes onto this corpus: every judge not of the subject's family.
EXPECTED_PANEL: tuple[str, ...] = routing(SUBJECT_MODEL)


class PanelGateError(RuntimeError):
    pass


def _one(values: set, field: str):
    if len(values) != 1:
        raise PanelGateError(f"votes disagree on {field}: {sorted(values)!r}")
    return next(iter(values))


def load_judge_votes(judge_key: str, out_dir: Path, q1_variant: str) -> dict:
    """One judge's vote rows, with the checks that make them usable in a panel."""
    if judge_key not in JUDGE_BY_KEY:
        raise PanelGateError(f"unknown judge key {judge_key!r}")
    judge = JUDGE_BY_KEY[judge_key]
    if judge.family == SUBJECT_FAMILY:
        raise PanelGateError(
            f"REFUSING {judge_key}: its family is {judge.family}, the gate corpus's subject "
            f"family, so section 6.2 routes it out of the panel. Its votes belong in the "
            f"per-judge tables, never in a panel label."
        )
    rows = rec.read_votes(out_dir / "votes.jsonl")
    if not rows:
        raise PanelGateError(f"{out_dir}/votes.jsonl has no rows")
    keys = {r["judge_key"] for r in rows}
    if keys != {judge_key}:
        raise PanelGateError(f"{out_dir} holds votes for {sorted(keys)}, not {judge_key!r} alone")
    q1_rows = [r for r in rows if r.get("question") == "Q1"]
    if not q1_rows:
        raise PanelGateError(f"{out_dir} carries no Q1 rows")
    q1_file = _one({r["prompt_file"] for r in q1_rows}, "the Q1 prompt file")
    q1_sha = _one({r["prompt_sha256"] for r in q1_rows}, "the Q1 prompt sha256")
    want = Q1_PROMPT_FILES[q1_variant]
    if q1_file != want:
        raise PanelGateError(
            f"{out_dir} was scored on Q1 file {q1_file}, not the requested variant "
            f"{q1_variant} ({want}). A panel label mixing two Q1 prompts is not a label."
        )
    local = load_prompt(PROMPT_DIR / q1_file)
    if local.sha256 != q1_sha:
        raise PanelGateError(
            f"REFUSING: {out_dir} was cast on {q1_file} sha256 {q1_sha} and this checkout's "
            f"copy hashes {local.sha256}."
        )
    serving_line = _one({r.get("serving_line", "") for r in rows}, "the serving line")
    # The same rule build_report uses: an empty (or absent) serving_line IS the pinned line
    # of section 6.1. Job 825542's votes predate the field entirely and its run record says
    # pinned, so reading a missing field as exploratory would relabel a run of record.
    pinned = not serving_line
    return {
        "judge_key": judge_key,
        "family": judge.family,
        "dir": str(out_dir),
        "rows": rows,
        "votes": len(rows),
        "q1_prompt_file": q1_file,
        "q1_prompt_sha256": q1_sha,
        "serving_line": serving_line or "pinned (section 6.1)",
        "serving_line_is_pinned": pinned,
        "source": "pinned" if pinned else "exploratory",
    }


def _by_item(sources: dict[str, dict], question: str, run_idx: int) -> dict[str, dict[str, str]]:
    """{item_id: {judge_key: vote}} for one question and one run, unswapped rows only."""
    out: dict[str, dict[str, str]] = {}
    for key, src in sources.items():
        for r in src["rows"]:
            if r.get("question") != question or r.get("run_idx") != run_idx or r.get("position_swap"):
                continue
            out.setdefault(r["item_id"], {})[key] = r["vote"]
    return out


def panel_labels(sources: dict[str, dict], run_idx: int) -> dict[str, dict]:
    """The panel's Q1 and gate outcome for every item in one run."""
    q1 = _by_item(sources, "Q1", run_idx)
    gate = _by_item(sources, "gate", run_idx)
    out: dict[str, dict] = {}
    for item_id in sorted(set(q1) | set(gate)):
        gate_res = aggregate(gate.get(item_id, {}))
        q1_res = aggregate(q1.get(item_id, {}), gate_label=gate_res.label)
        out[item_id] = {
            "q1": q1_res.label,
            "q1_is_tie": q1_res.is_tie,
            "q1_resolution": q1_res.resolution,
            "q1_panel_size": q1_res.panel_size_actual,
            "q1_votes": q1_res.available,
            "gate": gate_res.label,
            "gate_is_tie": gate_res.is_tie,
            "gate_resolution": gate_res.resolution,
            "gate_panel_size": gate_res.panel_size_actual,
        }
    return out


def _class_counts(labels: dict[str, dict], items_by_id: dict[str, dict]) -> dict:
    counts: dict[str, dict] = {}
    for item_id, lab in labels.items():
        item = items_by_id.get(item_id)
        if item is None:
            continue
        cls = item["meta"]["gate_class"]
        bucket = counts.setdefault(cls, {
            "q1": {"yes": 0, "no": 0, "unlabeled": 0, "tie": 0, "total": 0},
            "gate": {"coherent": 0, "silent_override": 0, "unlabeled": 0, "tie": 0, "total": 0},
        })
        for q in ("q1", "gate"):
            bucket[q]["total"] += 1
            label = lab[q]
            if label is None:
                bucket[q]["unlabeled"] += 1
            elif label in bucket[q]:
                bucket[q][label] += 1
            if lab[f"{q}_is_tie"]:
                bucket[q]["tie"] += 1
    return counts


def score_panel(sources: dict[str, dict], items_by_id: dict[str, dict]) -> dict:
    """The ten thresholds on the panel label, every one with its denominator."""
    run0 = panel_labels(sources, 0)
    counts = _class_counts(run0, items_by_id)
    metrics: dict[str, dict] = {}

    def add(name: str, num: int, den: int) -> None:
        metrics[name] = {"numerator": num, "denominator": den, "rate": (num / den) if den else None}

    for cls in CLASSES:
        c = counts.get(cls)
        if not c:
            continue
        truth = TRUTH[cls]
        avail_q1 = c["q1"]["yes"] + c["q1"]["no"]
        if truth.get("q1") is True:
            add(f"recall_{cls}", c["q1"]["yes"], avail_q1)
        elif truth.get("q1") is False:
            add(f"specificity_{cls}", c["q1"]["no"], avail_q1)
        gate_truth = truth.get("gate")
        if gate_truth:
            avail_gate = c["gate"]["coherent"] + c["gate"]["silent_override"]
            add(f"gate_accuracy_{cls}", c["gate"][gate_truth], avail_gate)

    # Pooled over the panel's judges: the denominator is every vote they cast.
    all_rows = [r for src in sources.values() for r in src["rows"]]
    malformed = sum(1 for r in all_rows if r.get("vote") == rec.MALFORMED)
    abstain = sum(1 for r in all_rows if r.get("vote") == rec.ABSTAIN)
    add("malformed_rate_max", malformed, len(all_rows))
    add("abstain_rate", abstain, len(all_rows))
    add("panel_unlabeled", sum(1 for l in run0.values() if l["q1"] is None), len(run0))
    add("panel_q1_ties", sum(1 for l in run0.values() if l["q1_is_tie"]), len(run0))

    # Test-retest on the PANEL label: recompute it per run and compare across runs.
    runs = sorted({r.get("run_idx") for src in sources.values() for r in src["rows"]
                   if r.get("run_idx") is not None})
    per_run = {i: panel_labels(sources, i) for i in runs}
    seen: dict[str, list] = {}
    for i in runs:
        for item_id, lab in per_run[i].items():
            seen.setdefault(item_id, []).append(lab["q1"])
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    agree = sum(1 for v in multi.values() if len(set(v)) == 1)
    add("test_retest_q1_min", agree, len(multi))

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
    failed = [k for k, v in verdicts.items() if v["verdict"] == "FAIL"]
    no_data = [k for k, v in verdicts.items() if v["verdict"] == "NO DATA"]
    passed = [k for k, v in verdicts.items() if v["verdict"] == "PASS"]
    return {
        "judges": sorted(sources),
        "panel_size_nominal": len(sources),
        "sources": {k: {"serving_line": v["serving_line"], "source": v["source"],
                        "votes": v["votes"], "dir": v["dir"]} for k, v in sources.items()},
        "per_class_counts": counts,
        "metrics": metrics,
        "gate_verdicts": verdicts,
        "passed_metrics": passed,
        "failed_metrics": failed,
        "no_data_metrics": no_data,
        "passed_of_ten": f"{len(passed)}/{len(THRESHOLDS)}",
        "verdict": "FAIL" if failed else ("INCOMPLETE" if no_data else "PASS"),
    }


def build_panel_report(sources: dict[str, dict], items: list[dict], q1_variant: str) -> dict:
    items_by_id = {i["item_id"]: i for i in items}
    missing = [k for k in EXPECTED_PANEL if k not in sources]
    full = score_panel(sources, items_by_id)
    loo = {}
    for dropped in sorted(sources):
        rest = {k: v for k, v in sources.items() if k != dropped}
        if not rest:
            continue
        loo[dropped] = score_panel(rest, items_by_id)
    return {
        # A panel missing a judge is a SMALLER PANEL, not a stand-in for the panel of
        # record, and the kind says so in the one place every table reads.
        "kind": "PANEL" if not missing else "PANEL-PARTIAL",
        "panel_complete": not missing,
        "expected_panel": list(EXPECTED_PANEL),
        "missing_judges": missing,
        "q1_prompt_variant": q1_variant,
        "q1_prompt_file": Q1_PROMPT_FILES[q1_variant],
        "thresholds": THRESHOLDS,
        "thresholds_sha256": thresholds_sha256(),
        "subject_model": "Qwen3-8B",
        "subject_family": SUBJECT_FAMILY,
        "panel_rule": "section 6.2: every judge not of the subject's family; majority of the "
                      "available votes; a tie takes the coherence-gate outcome",
        "excluded_own_family": [k for k, j in JUDGE_BY_KEY.items() if j.family == SUBJECT_FAMILY],
        "corpus_items": len(items),
        "panel": full,
        "leave_one_judge_out": loo,
        "changes_the_verdict": [
            k for k, v in loo.items()
            if set(v["failed_metrics"]) != set(full["failed_metrics"])
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--q1", required=True, choices=sorted(Q1_PROMPT_FILES))
    ap.add_argument("--votes", action="append", required=True, metavar="JUDGE=DIR",
                    help="one judge's results directory (the one holding votes.jsonl)")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a report even when a judge of the panel has no votes. "
                         "Without it an incomplete panel prints its numbers and writes "
                         "nothing, so a partial panel cannot reach a table by accident.")
    args = ap.parse_args(argv)

    sources: dict[str, dict] = {}
    for spec in args.votes:
        key, _, path = spec.partition("=")
        sources[key] = load_judge_votes(key, Path(path), args.q1)
    items = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    report = build_panel_report(sources, items, args.q1)
    if report["missing_judges"]:
        print(f"[panel] INCOMPLETE: {', '.join(report['missing_judges'])} has no votes here, "
              f"so this is a {len(sources)}-judge panel and not the panel of record "
              f"({', '.join(EXPECTED_PANEL)}).")
    if args.out:
        if report["missing_judges"] and not args.allow_partial:
            print(f"[panel] NOT WRITING {args.out}: pass --allow-partial to record a "
                  f"partial panel, which every table then shows as PANEL-PARTIAL.")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    panel = report["panel"]
    print(f"[panel] Q1 {report['q1_prompt_file']}  judges {', '.join(panel['judges'])}  "
          f"{panel['verdict']}  {panel['passed_of_ten']} thresholds pass")
    for name in THRESHOLDS:
        v = panel["gate_verdicts"][name]
        if v.get("numerator") is None:
            print(f"    {name:32s} NO DATA")
        else:
            print(f"    {name:32s} {v['numerator']:>5d}/{v['denominator']:<6d} {v['verdict']}  "
                  f"(bar {v['threshold']})")
    for dropped, block in report["leave_one_judge_out"].items():
        print(f"  [minus {dropped}] {block['verdict']}  {block['passed_of_ten']}  "
              f"failed {block['failed_metrics'] or 'none'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
