"""Does the panel code score the same thing the per-judge reports score?

Run experiments/jury/panel_gate.py with EXACTLY ONE judge. The panel label is then that
judge's own vote by definition, so every one of the ten scored metrics must come back
identical to that run's committed gate_report.json, numerator and denominator, verdict
included. Anything else means a panel number and a per-judge number are not commensurable
and no panel row could stand in the same table as a per-judge row.

tests/test_panel_gate.py pins this for Q1 file a. This runs a, b and c and prints a
machine comparison rather than two lists a reader has to align by eye. It reads vote
files that already exist, runs no model, submits nothing and writes no report.

    python bcf/panel_one_judge_check.py     # exit 0 when all three reproduce, 1 otherwise
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from experiments.jury.gate_thresholds import THRESHOLDS  # noqa: E402
from experiments.jury.panel_gate import (  # noqa: E402
    DEFAULT_ITEMS,
    build_panel_report,
    load_judge_votes,
)

RUNS = {
    "a": ("llama-3.3-70b-fp8", 825542),
    "b": ("llama-3.3-70b-fp8-q1b", 826010),
    "c": ("llama-3.3-70b-fp8-q1c", 826017),
}
JUDGE = "llama-3.3-70b-fp8"
ROOT = pathlib.Path("experiments/results/jury-gate")


def main() -> int:
    items = [json.loads(x) for x in DEFAULT_ITEMS.read_text(encoding="utf-8").splitlines() if x.strip()]
    differences = 0
    for variant, (slug, job) in RUNS.items():
        d = ROOT / slug / "arc_challenge" / "stated-hint"
        committed = json.loads((d / "gate_report.json").read_text())
        cm = committed["per_judge"][0]["metrics"]
        report = build_panel_report({JUDGE: load_judge_votes(JUDGE, d, variant)}, items, variant)
        panel = report["panel"]
        pv = panel["gate_verdicts"]
        print(f"=== Q1 {variant}, judge {JUDGE}, job {job}, kind {report['kind']} ===")
        vsame = committed["verdict"] == panel["verdict"]
        differences += 0 if vsame else 1
        print(f"    verdict   committed {committed['verdict']:<10s} one-judge panel "
              f"{panel['verdict']:<10s} {'SAME' if vsame else 'DIFFERS'}")
        for metric in THRESHOLDS:
            c = cm.get(metric) or {}
            p = pv.get(metric) or {}
            same = (c.get("numerator"), c.get("denominator")) == (p.get("numerator"), p.get("denominator"))
            differences += 0 if same else 1
            print(f"    {metric:33s} committed {c.get('numerator')}/{c.get('denominator')}"
                  f"   panel {p.get('numerator')}/{p.get('denominator')}"
                  f"   {'SAME' if same else 'DIFFERS'}")
        print()
    print(f"metric-or-verdict differences across the three files: {differences}")
    return 0 if differences == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
