"""Rebuild a gate report from a fetched vote file, taking every parameter from the votes.

A report that was written on the cluster is a claim; this recomputes the same numbers from
the rows themselves so the claim can be checked on the Mac. Nothing is guessed from a
directory name: the judge keys, the Q1 prompt file and its SHA-256, the serving line and
any input transform all come off the vote records, and the run REFUSES if the prompt file
named in the votes is not byte-identical to the copy in this checkout, which is the check
that makes a recomputation worth doing at all.

    python -m experiments.jury.recompute_report \
        experiments/results/jury-gate/gemma-3-27b-it-h200-q1a/arc_challenge/stated-hint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import records as rec
from .gate import build_report
from .prompt_files import PROMPT_DIR, Q1_PROMPT_FILES, load_prompt, load_prompts

DEFAULT_ITEMS = Path("experiments/results/jury-gate/gate_items.jsonl")


class RecomputeError(RuntimeError):
    pass


def _one(values: set, field: str):
    if len(values) != 1:
        raise RecomputeError(f"votes disagree on {field}: {sorted(values)!r}")
    return next(iter(values))


def recompute(out_dir: Path, items_path: Path = DEFAULT_ITEMS) -> dict:
    rows = rec.read_votes(out_dir / "votes.jsonl")
    if not rows:
        raise RecomputeError(f"{out_dir}/votes.jsonl has no rows; there is nothing to recompute")
    q1_rows = [r for r in rows if r.get("question") == "Q1"]
    if not q1_rows:
        raise RecomputeError(f"{out_dir}/votes.jsonl carries no Q1 rows")

    q1_file = _one({r["prompt_file"] for r in q1_rows}, "the Q1 prompt file")
    q1_sha = _one({r["prompt_sha256"] for r in q1_rows}, "the Q1 prompt sha256")
    variant = next((k for k, v in Q1_PROMPT_FILES.items() if v == q1_file), None)
    if variant is None:
        raise RecomputeError(f"the votes name Q1 prompt {q1_file!r}, which is not in Q1_PROMPT_FILES")
    local = load_prompt(PROMPT_DIR / q1_file)
    if local.sha256 != q1_sha:
        raise RecomputeError(
            f"REFUSING: the votes were cast on {q1_file} sha256 {q1_sha}, and this checkout's "
            f"copy hashes {local.sha256}. The recomputation would be against different bytes."
        )
    serving_line = _one({r.get("serving_line", "") for r in rows}, "the serving line")
    serving_note = _one({r.get("serving_line_note", "") for r in rows}, "the serving-line note")
    pinned = _one({bool(r.get("serving_line_is_pinned")) for r in rows}, "serving_line_is_pinned")
    transform = _one({json.dumps(r.get("input_transform") or {}, sort_keys=True)
                      for r in rows}, "the input transform")
    all_rows = _one({bool(r.get("all_judge_row")) for r in rows}, "all_judge_row")
    judge_keys = sorted({r["judge_key"] for r in rows})

    summary_path = out_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    raw = [json.loads(x) for x in items_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    prompts = load_prompts(q1=variant)
    report = build_report(
        out_dir, raw, summary, judge_keys, prompts=prompts,
        # A pinned run recorded an empty serving_line, and build_report reads empty as
        # pinned, so the two have to be passed back the way they were recorded.
        serving_line="" if pinned else serving_line,
        serving_line_note=serving_note,
        all_judge_rows=all_rows,
        input_transform=json.loads(transform),
    )
    report["recomputed_from"] = str(out_dir / "votes.jsonl")
    report["recomputed_vote_rows"] = len(rows)
    report["q1_prompt_variant"] = variant
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--copy-to", type=Path, default=Path("experiments/jury"),
                    help="also write gate_report_<slug>.json here")
    args = ap.parse_args(argv)

    failures = 0
    for out_dir in args.dirs:
        slug = out_dir.parts[-3] if len(out_dir.parts) >= 3 else out_dir.name
        try:
            report = recompute(out_dir, args.items)
        except (RecomputeError, FileNotFoundError) as exc:
            print(f"[{slug}] NOT RECOMPUTED: {exc}")
            failures += 1
            continue
        (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.copy_to:
            args.copy_to.mkdir(parents=True, exist_ok=True)
            (args.copy_to / f"gate_report_{slug}.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
        judge = report["per_judge"][0]
        print(f"[{slug}] {report['verdict']}  Q1 {report['prompt_files']['Q1']} "
              f"({report['q1_prompt_variant']})  serving_line {report['serving_line']}  "
              f"votes {judge['votes']}  failed {judge['failed_metrics'] or 'none'}")
        for name, v in judge["gate_verdicts"].items():
            num = v.get("numerator")
            den = v.get("denominator")
            if num is None:
                print(f"    {name:32s} NO DATA")
            else:
                print(f"    {name:32s} {num:>5d}/{den:<6d} {v['verdict']}  (bar {v['threshold']})")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
