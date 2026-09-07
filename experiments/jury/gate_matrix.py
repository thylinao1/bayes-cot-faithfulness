"""One table for every gate run there is: judge x Q1 file x serving line, ten thresholds.

`gate_compare.py` puts a handful of configurations side by side as columns, which stops
working once there are four judges, three Q1 files, two serving lines and a projection.
This writes the same numbers long-form, one ROW per configuration, so the table grows
downwards and a reader can sort it by judge or by prompt.

Three kinds of row, and the kind is a column so it can never be lost:

  MEASURED   a gate report recomputed from a vote file
  PROJECTED  a projection under the option (d) stipulation, from project_echo_strip.py
  PARTIAL    a run that did not finish; its numbers are shown with the votes it wrote
  PANEL      the section 6.2 panel label scored on the same bars, from panel_gate.py, with
             PANEL-LOO for a leave-one-judge-out and PANEL-PARTIAL when a judge of the panel
             has no votes, which makes the row a smaller panel and not the panel of record

Nothing is averaged across rows and nothing is ranked. Every cell is numerator/denominator
with the verdict against the bar that was written before the first run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gate_thresholds import THRESHOLDS
from .synthetic_gate import CLASSES, TRUTH

SHORT = {
    "recall_planted_mention": "planted",
    "recall_paraphrased_disclosure": "paraphrase",
    "recall_quoted_denied": "quoted-denied",
    "specificity_clean": "clean",
    "specificity_deleted_step": "deleted-step",
    "specificity_restated_cue_only": "restated",
    "gate_accuracy_gate_positive": "gate-override",
    "gate_accuracy_clean": "gate-coherent",
    "malformed_rate_max": "malformed",
    "test_retest_q1_min": "test-retest",
}


def _cell(verdicts: dict, name: str) -> str:
    v = verdicts.get(name)
    if not v or v.get("numerator") is None:
        return "no data"
    return f"{v['numerator']}/{v['denominator']} {v['verdict']}"


def _panel_rows(data: dict) -> list[dict]:
    """A panel report becomes one PANEL row plus one PANEL-LOO row per dropped judge.

    The panel is not a judge and its row says so: the Judge column names the composition,
    and the Serving line column names where each judge's votes came from, because a panel
    label built from one pinned and two exploratory judges is not a pinned-line measurement
    and must never be read as one.
    """
    out: list[dict] = []

    def line_of(block: dict) -> str:
        parts = []
        for key in block["judges"]:
            src = block["sources"][key]
            # The line comes off the votes, never off a guess: the string was hardcoded
            # "exploratory-h200" back when the h200 was the only exploratory pool, and it
            # printed h200 for the gpt-oss run that was actually on exploratory-h100-47.
            line = "pinned" if src["source"] == "pinned" else (src.get("serving_line") or "exploratory")
            parts.append(f"{key}:{line}")
        return " + ".join(parts)

    def row(block: dict, kind: str, label: str) -> dict:
        return {
            "kind": kind,
            "slug": f"panel-q1{data['q1_prompt_variant']}",
            "job": "-",
            "judge": label,
            "q1_file": data["q1_prompt_file"],
            "serving_line": line_of(block),
            "votes": sum(s["votes"] for s in block["sources"].values()),
            "verdict": block["verdict"],
            "failed": block["failed_metrics"],
            "verdicts": block["gate_verdicts"],
            "per_class": block["per_class_counts"],
            "note": "panel label, section 6.2 majority of available votes",
        }

    kind = "PANEL" if data.get("panel_complete", True) else "PANEL-PARTIAL"
    out.append(row(data["panel"], kind, "PANEL " + "+".join(data["panel"]["judges"])))
    for dropped, block in data.get("leave_one_judge_out", {}).items():
        out.append(row(block, f"{kind}-LOO", f"PANEL minus {dropped}"))
    return out


def rows_from_report(path: Path, jobs: dict) -> list[dict]:
    """A gate report (per_judge), a projection, or a panel report becomes 1..n rows."""
    data = json.loads(path.read_text())
    slug = path.name[len("gate_report_"):-len(".json")] if path.name.startswith("gate_report_") else path.stem
    meta = jobs.get(slug, {})
    out: list[dict] = []
    if str(data.get("kind", "")).startswith("PANEL"):
        return _panel_rows(data)
    if "per_judge" in data:
        transform = data.get("input_transform") or {}
        line = data.get("serving_line", "")
        if transform.get("echo_strip"):
            line = f"{line} + echo strip {transform['echo_strip_min_chars']}"
        for judge in data["per_judge"]:
            out.append({
                "kind": meta.get("kind", "MEASURED"),
                "slug": slug,
                "job": meta.get("job", "-"),
                "judge": judge["judge_key"],
                "q1_file": data["prompt_files"]["Q1"],
                "serving_line": line,
                "votes": judge["votes"],
                "verdict": judge["verdict"],
                "failed": judge["failed_metrics"],
                "verdicts": judge["gate_verdicts"],
                "per_class": judge["per_class_counts"],
                "note": meta.get("note", ""),
            })
        return out
    if "observed" in data and "projected" in data:
        base = slug.removeprefix("projected_")
        meta = jobs.get(base, {})
        for judge_key, block in data["projected"].items():
            out.append({
                "kind": "PROJECTED",
                "slug": slug,
                "job": meta.get("job", "-"),
                "judge": judge_key,
                "q1_file": data.get("q1_prompt_file", "?"),
                "serving_line": f"{data.get('serving_line', '?')} + option (d) stipulated",
                "votes": block["votes"],
                "verdict": block["verdict"],
                "failed": block["failed_metrics"],
                "verdicts": block["gate_verdicts"],
                "per_class": block["per_class_counts"],
                "note": "restated scores as its clean twin; every other class keeps its votes",
            })
        return out
    raise ValueError(f"{path} is neither a gate report nor a projection")


def threshold_matrix(rows: list[dict]) -> list[str]:
    head = ["Kind", "Judge", "Q1 file", "Serving line", "Job", "Votes", "Verdict"] + \
        [f"{SHORT[n]} ({'<=' if n == 'malformed_rate_max' else '>='}{THRESHOLDS[n]})" for n in THRESHOLDS]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for row in rows:
        cells = [row["kind"], row["judge"], row["q1_file"], row["serving_line"],
                 str(row["job"]), str(row["votes"]), row["verdict"]]
        cells += [_cell(row["verdicts"], name) for name in THRESHOLDS]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def class_matrix(rows: list[dict]) -> list[str]:
    head = ["Kind", "Judge", "Q1 file", "Serving line"] + [f"{c} ({'yes' if TRUTH[c].get('q1') else 'no'})" for c in CLASSES]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for row in rows:
        cells = [row["kind"], row["judge"], row["q1_file"], row["serving_line"]]
        for cls in CLASSES:
            counts = (row["per_class"].get(cls) or {}).get("q1")
            if not counts:
                cells.append("no data")
                continue
            cells.append(f"{counts['yes']} yes / {counts['no']} no")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="append", required=True, type=Path)
    ap.add_argument("--jobs", type=Path, default=Path("experiments/jury/gate_runs.json"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    jobs = json.loads(args.jobs.read_text()) if args.jobs and args.jobs.exists() else {}
    rows: list[dict] = []
    for path in args.report:
        if not path.exists():
            print(f"[skip] {path}")
            continue
        rows.extend(rows_from_report(path, jobs))
    rows.sort(key=lambda r: (r["judge"], r["q1_file"], r["kind"] != "MEASURED", r["serving_line"]))

    body = ["## Every gate run, one row each: judge x Q1 file x serving line", ""]
    body += threshold_matrix(rows)
    body += ["", "## Q1 yes and no per gate class, run 0 unswapped, same rows", ""]
    body += class_matrix(rows)
    text = "\n".join(body) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
