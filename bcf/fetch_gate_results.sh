#!/bin/bash
# Bring one judge's gate results back from the cluster and recompute the judging budget
# from the rate that run measured. Read-only on the cluster side.
#
#   bcf/fetch_gate_results.sh llama-3.3-70b-fp8
set -euo pipefail
JUDGE="${1:?usage: fetch_gate_results.sh <judge_key>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="bcf/results/jury-gate/${JUDGE}/arc_challenge/stated-hint"
LOCAL="${REPO}/experiments/results/jury-gate/${JUDGE}/arc_challenge/stated-hint"
mkdir -p "$LOCAL"
rsync -a \
  --include='votes.jsonl' --include='panel_labels.jsonl' --include='gate_report.json' \
  --include='run_summary.json' --include='checkpoint.json' --include='run.log' \
  --include='exit_code.txt' --include='rerun_queue.jsonl' --exclude='*' \
  "soc:${REMOTE}/" "${LOCAL}/"
echo "[fetch] into ${LOCAL}"
ls -la "$LOCAL"
if [ -f "${LOCAL}/gate_report.json" ]; then
  python -m experiments.jury.budget --gate-report "${LOCAL}/gate_report.json" \
    --out "${REPO}/experiments/jury/budget.md"
elif [ -f "${LOCAL}/checkpoint.json" ]; then
  echo "[fetch] no gate_report yet; building the budget from the checkpoint's partial rate"
  python - "$LOCAL" <<'PY'
import json, sys, pathlib
from experiments.jury import budget as b
cp = json.loads((pathlib.Path(sys.argv[1]) / "checkpoint.json").read_text())
measured = {
    "votes_per_second": cp["votes_per_second"], "source": str(pathlib.Path(sys.argv[1]) / "checkpoint.json"),
    "judge": ", ".join(cp.get("judges", {})), "concurrency": None,
    "votes": cp.get("votes_written"), "seconds": cp.get("elapsed_s"),
    "items": cp.get("items_total"), "gpu": None,
    "note": "PARTIAL: read from a running job's checkpoint, not from a finished gate report.",
}
pathlib.Path("experiments/jury/budget.md").write_text(b.build_markdown(measured))
print(json.dumps({"partial_rate": measured["votes_per_second"], "votes": measured["votes"]}, indent=2))
PY
else
  echo "[fetch] nothing measured yet; budget.md not written"
fi
