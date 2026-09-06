#!/bin/bash
# The four cluster verbs this lane needs, in one place, so a step is never done two ways.
# Runs on the MAC. Every one of them needs `ssh soc` to answer; if it times out during
# banner exchange the NUS VPN is down and reconnecting it is the operator's.
#
#   bcf/w2c.sh sync                     rsync this worktree to ~/bcf/repo-jury
#   bcf/w2c.sh status                   the queue, and the votes and exit code of every
#                                       jury-gate results directory on the cluster
#   bcf/w2c.sh submit <judges.tsv>      one wave through bcf/jury_wave.sh, which refuses on
#                                       the live cap counting every campaign's jobs
#   bcf/w2c.sh fetch <slug> [<slug>...] rsync the results back and recompute each report
#                                       from the vote file
#
# It never cancels anything and never uses a wildcard scancel.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${BCF_VENV_PY:-/Users/maksimsilchenko/Developer/bayes-cot-faithfulness/.venv/bin/python}"
CMD="${1:?usage: w2c.sh sync|status|submit <tsv>|fetch <slug>...}"
shift || true

case "$CMD" in
  sync)
    rsync -a --delete \
      --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
      --exclude 'experiments/results' --exclude '*.pyc' \
      "${REPO}/" soc:bcf/repo-jury/
    ssh soc 'ls -la ~/bcf/repo-jury/bcf/exit_guard.sh ~/bcf/repo-jury/experiments/jury/echo_strip.py && sha256sum ~/bcf/repo-jury/experiments/jury/gate_thresholds.py ~/bcf/repo-jury/experiments/jury/prompts/*.md'
    ;;
  status)
    ssh soc 'echo "=== queue, every job on this account ==="; squeue --me -o "%.10i %.30j %.9P %.9T %.11M %.11l %.20b %R"; echo; echo "=== a100-80 and h200-141 cards in use, RUNNING and PENDING, all campaigns ==="; squeue --me -h -t RUNNING,PENDING -O "tres-alloc:200" | tr "," "\n" | grep -oE "gres/gpu:[a-z0-9-]+=[0-9]+" | awk -F= "{s[\$1]+=\$2} END {for (k in s) print k\"=\"s[k]}"; echo; echo "=== jury-gate results on the cluster ==="; for d in ~/bcf/results/jury-gate/*/arc_challenge/stated-hint; do [ -d "$d" ] || continue; printf "%-48s votes=%-6s exit=%s\n" "$(basename "$(dirname "$(dirname "$d")")")" "$(wc -l < "$d/votes.jsonl" 2>/dev/null || echo 0)" "$(cat "$d/exit_code.txt" 2>/dev/null || echo -)"; done'
    ;;
  submit)
    TSV="${1:?usage: w2c.sh submit <judges.tsv, path inside the repo>}"
    ssh soc "cd ~/bcf/repo-jury && bash bcf/jury_wave.sh '${TSV}'"
    ;;
  fetch)
    [ $# -gt 0 ] || { echo "usage: w2c.sh fetch <slug> [<slug>...]" >&2; exit 2; }
    dirs=()
    for slug in "$@"; do
      local_dir="${REPO}/experiments/results/jury-gate/${slug}/arc_challenge/stated-hint"
      mkdir -p "$local_dir"
      rsync -a \
        --include='votes.jsonl' --include='panel_labels.jsonl' --include='gate_report.json' \
        --include='run_summary.json' --include='checkpoint.json' --include='run.log' \
        --include='exit_code.txt' --include='server-*.log' --include='rerun_queue.jsonl' \
        --exclude='*' \
        "soc:bcf/results/jury-gate/${slug}/arc_challenge/stated-hint/" "$local_dir/" || true
      printf '%-48s votes=%-6s exit=%s\n' "$slug" \
        "$(wc -l < "$local_dir/votes.jsonl" 2>/dev/null | tr -d ' ' || echo 0)" \
        "$(cat "$local_dir/exit_code.txt" 2>/dev/null | tr -d '[:space:]' || echo -)"
      dirs+=("experiments/results/jury-gate/${slug}/arc_challenge/stated-hint")
    done
    echo
    echo "=== recomputed from the vote files, not read off the cluster ==="
    ( cd "$REPO" && PYTHONPATH="${REPO}/src" "$VENV_PY" -m experiments.jury.recompute_report "${dirs[@]}" )
    ;;
  *)
    echo "unknown command $CMD" >&2; exit 2 ;;
esac
