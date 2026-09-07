#!/bin/bash
# Regenerate the machine-written half of experiments/jury/GATE-Q1-COMPARISON.md.
#
# Every gate_report_*.json in experiments/jury becomes one row of the matrix, projections
# included and marked PROJECTED. The prose around the markers is hand-written and is left
# alone; the tables between them are replaced, so a table in that document can never drift
# from the reports it claims to summarise.
#
#   bcf/rebuild_comparison.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${BCF_VENV_PY:-/Users/maksimsilchenko/Developer/bayes-cot-faithfulness/.venv/bin/python}"
DOC="${REPO}/experiments/jury/GATE-Q1-COMPARISON.md"
GEN="${REPO}/experiments/jury/gate_matrix_2026-09-07.md"

args=()
for f in "${REPO}"/experiments/jury/gate_report_*.json; do
  [ -f "$f" ] || continue
  args+=(--report "$f")
done
[ ${#args[@]} -gt 0 ] || { echo "no gate_report_*.json to summarise" >&2; exit 2; }

( cd "$REPO" && PYTHONPATH="${REPO}/src" "$VENV_PY" -m experiments.jury.gate_matrix "${args[@]}" --out "$GEN" >/dev/null ) || exit 1

python3 - "$DOC" "$GEN" <<'PY'
import sys, pathlib
doc, gen = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = doc.read_text()
begin, end = "<!-- BEGIN GENERATED MATRIX -->", "<!-- END GENERATED MATRIX -->"
i, j = text.find(begin), text.find(end)
if i < 0 or j < 0:
    raise SystemExit(f"{doc} has no generated-matrix markers; refusing to guess where to splice")
doc.write_text(text[: i + len(begin)] + "\n\n" + gen.read_text().rstrip() + "\n\n" + text[j:])
print(f"spliced {gen} into {doc}")
PY
