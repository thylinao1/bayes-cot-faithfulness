#!/bin/bash
# Prove the R1 determinism preflight can REFUSE, at the command line, with exit codes.
#
# The unit tests assert the gate's arithmetic. This script asserts the thing a cluster
# job actually depends on: that `bcf/determinism_preflight.py` returns 10 and writes a
# REFUSE verdict when one of the thirty completions differs, and 0 when none does. The
# two probe files are built by the SAME comparison code the cluster runs
# (concurrency_probe.compare), so the injection is into the comparison, not into a
# hand-typed summary string.
#
#   bash bcf/preflight_falsification.sh [output_dir]
#
# Writes <output_dir>/EXIT_CODES.txt and the two verdicts. Exits non-zero if the gate
# did NOT behave as claimed, so a broken gate fails this script instead of being
# recorded as proof.

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO}/docs/a3f-proofs}"
WORK="$(mktemp -d)"
mkdir -p "$OUT_DIR"

# The payload builder lives in python; it receives the paths through the environment so
# no path is interpolated into the heredoc.
BCF_WORK="$WORK" BCF_REPO_DIR="$REPO" python - <<'PY'
import importlib.util, json, os, sys
from pathlib import Path

repo = Path(os.environ["BCF_REPO_DIR"])
work = Path(os.environ["BCF_WORK"])
spec = importlib.util.spec_from_file_location("cp", repo / "bcf" / "concurrency_probe.py")
cp = importlib.util.module_from_spec(spec)
sys.modules["cp"] = cp
spec.loader.exec_module(cp)

LETTERS = ("A", "B", "C", "D")
N = 30
comps = [f"chain for item {i}\nANSWER: {LETTERS[i % 4]}" for i in range(N)]
lps = [{L: -0.5 - i * 0.01 - j for j, L in enumerate(LETTERS)} for i in range(N)]


def payload(c32):
    base = {"_completions": comps, "_logprobs": lps}
    rows = []
    for level, c in ((1, comps), (32, c32)):
        rows.append({
            "concurrency": level,
            "generations_per_second": 0.1433 if level == 1 else 2.9196,
            "vs_concurrency_1": cp.compare(base, {"_completions": c, "_logprobs": lps}),
        })
    return {"model": "Qwen/Qwen3-8B", "n_items": N, "label": "determinism_preflight",
            "batch_invariant_env": "1", "rows": rows}


(work / "probe_clean.json").write_text(json.dumps(payload(list(comps)), indent=2))
injected = list(comps)
injected[17] = injected[17] + " (one token different)"
(work / "probe_injected.json").write_text(json.dumps(payload(injected), indent=2))
print(f"[falsify] built two probe files in {work}; the injected one alters item 17 only")
PY

CODES="${OUT_DIR}/EXIT_CODES.txt"
{
  echo "# bcf/determinism_preflight.py, proven able to refuse"
  echo "# generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by bcf/preflight_falsification.sh"
  echo "#"
  echo "# Both probe files are built by concurrency_probe.compare on real completion and"
  echo "# logprob lists. They differ in ONE of the thirty completions at 32 in flight."
} > "$CODES"

python "${REPO}/bcf/determinism_preflight.py" \
  --probe-json "${WORK}/probe_clean.json" \
  --out "${OUT_DIR}/verdict_clean.json" > "${OUT_DIR}/preflight_clean.log" 2>&1
CLEAN=$?
echo "clean probe (30/30 identical, max logprob diff 0.0): exit ${CLEAN}  (expected 0)" >> "$CODES"

python "${REPO}/bcf/determinism_preflight.py" \
  --probe-json "${WORK}/probe_injected.json" \
  --out "${OUT_DIR}/verdict_injected.json" > "${OUT_DIR}/preflight_injected.log" 2>&1
INJECTED=$?
echo "one altered completion at 32 in flight:            exit ${INJECTED}  (expected 10)" >> "$CODES"

{
  echo ""
  echo "verdict_clean.json    : $(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["verdict"])' "${OUT_DIR}/verdict_clean.json")"
  echo "verdict_injected.json : $(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["verdict"])' "${OUT_DIR}/verdict_injected.json")"
  echo "injected identical fraction: $(python -c 'import json,sys;d=json.load(open(sys.argv[1]));print([r["identical_completions"] for r in d["levels_checked"] if r["concurrency"]==32][0])' "${OUT_DIR}/verdict_injected.json")"
} >> "$CODES"

rm -rf "$WORK"
cat "$CODES"

if [ "$CLEAN" -ne 0 ] || [ "$INJECTED" -ne 10 ]; then
  echo "[falsify] FAILED: the gate did not behave as claimed (clean ${CLEAN}, injected ${INJECTED})" >&2
  exit 1
fi
echo "[falsify] the gate passes a clean probe and REFUSES an injected one."
