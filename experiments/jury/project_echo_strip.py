"""The OFFLINE projection of option (d) onto the gate runs that already exist.

THE STIPULATION, and it is a stipulation and not a measurement. Assume the echo strip
works as option (d) intends: the reproduced item text is gone, so a `restated_cue_only`
response is indistinguishable from its matched `clean` counterpart and the judge votes on
it exactly as it voted on that clean item. Every other class keeps the votes it cast.

Under that assumption this recomputes all ten thresholds, per judge and per Q1 file, from
the vote files already on disk. Every row of the output is labelled PROJECTED. No judge is
called, no threshold is touched and nothing here is a run.

THE STIPULATION DOES NOT HOLD ON THIS CORPUS AT THE SPECIFIED 200 CHARACTERS. The strip
removes nothing from any of the 483 items, because the restated class plants an 86 to 99
character quote and not a copied block (`experiments/jury/echo_strip.py`, and
`tests/test_echo_strip.py` pins the numbers). The projection is therefore an upper bound
on what option (d) could buy if the corpus carried a bulk echo, which is exactly what makes
it worth computing before a card is spent on it.

    python -m experiments.jury.project_echo_strip \
        experiments/results/jury-gate/llama-3.3-70b-fp8-q1b/arc_challenge/stated-hint/votes.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gate import score_judge
from .gate_thresholds import THRESHOLDS

DEFAULT_ITEMS = Path("experiments/results/jury-gate/gate_items.jsonl")


def load_items(items_path: Path) -> list[dict]:
    return [json.loads(x) for x in items_path.read_text(encoding="utf-8").splitlines() if x.strip()]


def clean_twin_map(items: list[dict]) -> dict[str, str]:
    """restated_cue_only item_id to the item_id of the clean item built from the same
    banked row. The two differ by exactly the restated header, which is what the strip is
    supposed to remove."""
    clean_by_base = {
        (i["meta"]["base_source"], i["meta"]["base_index"]): i["item_id"]
        for i in items
        if i["meta"]["gate_class"] == "clean"
    }
    out = {}
    for item in items:
        if item["meta"]["gate_class"] != "restated_cue_only":
            continue
        twin = clean_by_base.get((item["meta"]["base_source"], item["meta"]["base_index"]))
        if twin:
            out[item["item_id"]] = twin
    return out


def project_rows(rows: list[dict], items: list[dict]) -> tuple[list[dict], dict]:
    """Rewrite the Q1 votes on restated items to the vote cast on the clean twin."""
    twins = clean_twin_map(items)
    by_key = {
        (r.get("judge_key"), r.get("item_id"), r.get("question"), r.get("run_idx"),
         bool(r.get("position_swap"))): r
        for r in rows
    }
    projected: list[dict] = []
    substituted = missing = 0
    for row in rows:
        if row.get("question") != "Q1" or row.get("item_id") not in twins:
            projected.append(row)
            continue
        key = (row.get("judge_key"), twins[row["item_id"]], "Q1", row.get("run_idx"),
               bool(row.get("position_swap")))
        twin_row = by_key.get(key)
        if twin_row is None:
            missing += 1
            projected.append(row)
            continue
        new = dict(row)
        new["vote"] = twin_row["vote"]
        new["projected_from_item_id"] = twin_row["item_id"]
        new["projection"] = "echo_strip_stipulation"
        projected.append(new)
        substituted += 1
    return projected, {"substituted_q1_votes": substituted, "twin_not_found": missing,
                       "restated_items": len(twins)}


def project(votes_path: Path, items: list[dict], judge_keys: list[str] | None = None) -> dict:
    rows = [json.loads(x) for x in votes_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    items_by_id = {i["item_id"]: i for i in items}
    keys = judge_keys or sorted({r["judge_key"] for r in rows if r.get("judge_key")})
    projected_rows, stats = project_rows(rows, items)
    q1_rows = [r for r in rows if r.get("question") == "Q1"]
    q1_files = sorted({r["prompt_file"] for r in q1_rows})
    # Job 825542's votes predate the serving_line fields, so absence means the judge ran on
    # its pinned line. That is what the run record says it did, and it is stated rather than
    # left to be inferred from an empty cell.
    lines = sorted({(r.get("serving_line") or "pinned (section 6.1)")
                    if not r.get("serving_line_is_pinned", True) else "pinned (section 6.1)"
                    for r in rows})
    return {
        "votes_file": str(votes_path),
        "label": "PROJECTED",
        "q1_prompt_file": q1_files[0] if len(q1_files) == 1 else q1_files,
        "q1_prompt_sha256": sorted({r["prompt_sha256"] for r in q1_rows})[0] if q1_rows else "",
        "serving_line": lines[0] if len(lines) == 1 else lines,
        "input_transform": rows[0].get("input_transform") or {},
        "stipulation": "restated_cue_only scores exactly as its matched clean item; every "
                       "other class keeps its votes",
        "projection_stats": stats,
        "thresholds": THRESHOLDS,
        "observed": {k: score_judge(rows, items_by_id, k) for k in keys},
        "projected": {k: score_judge(projected_rows, items_by_id, k) for k in keys},
    }


def _line(name: str, block: dict) -> str:
    verdicts = block["gate_verdicts"]
    v = verdicts.get(name, {})
    if not v or v.get("numerator") is None:
        return f"{'NO DATA':>18s}"
    return f"{v['numerator']:>5d}/{v['denominator']:<5d} {v['verdict']:<5s}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("votes", nargs="+", type=Path)
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    items = load_items(args.items)
    results = []
    for path in args.votes:
        if not path.exists():
            print(f"[skip] {path} does not exist")
            continue
        result = project(path, items)
        results.append(result)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            slug = path.parts[-4] if len(path.parts) >= 4 else path.stem
            (args.out_dir / f"gate_report_projected_{slug}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")

    if args.as_json:
        print(json.dumps(results, indent=2))
        return 0

    for result in results:
        print(f"\n{result['votes_file']}")
        print(f"  PROJECTED under: {result['stipulation']}")
        print(f"  substituted Q1 votes: {result['projection_stats']['substituted_q1_votes']} "
              f"over {result['projection_stats']['restated_items']} restated items "
              f"(twins not found: {result['projection_stats']['twin_not_found']})")
        for key in result["observed"]:
            obs, proj = result["observed"][key], result["projected"][key]
            print(f"  judge {key}: observed {obs['verdict']} -> PROJECTED {proj['verdict']}")
            header = f"    {'threshold':32s} {'bar':>6s} {'observed':>18s} {'PROJECTED':>18s}"
            print(header)
            print("    " + "-" * (len(header) - 4))
            for name, bar in THRESHOLDS.items():
                print(f"    {name:32s} {bar:>6.2f} {_line(name, obs)} {_line(name, proj)}")
            print(f"    observed failed : {', '.join(obs['failed_metrics']) or 'none'}")
            print(f"    PROJECTED failed: {', '.join(proj['failed_metrics']) or 'none'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
