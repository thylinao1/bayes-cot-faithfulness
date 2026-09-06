#!/bin/bash
# Finish the exploratory h200 gate lane after the VPN comes back.
#
# W2b left this lane part-run: the FP8 judge is complete on all three Q1 prompts, the
# Gemma-3-27B-it job was mid-way through its third, and the other two judges have rows
# ready that were never submitted because the one h200-141 card was held. Nothing here
# touches the a100-80 runs of record, and every job it submits labels itself exploratory.
#
#   ssh soc          # must work first; if it times out the NUS VPN is down
#   bcf/finish_exploratory_h200.sh status
#   bcf/finish_exploratory_h200.sh fetch
#   bcf/finish_exploratory_h200.sh submit-gptoss
#   bcf/finish_exploratory_h200.sh submit-qwen
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CMD="${1:?usage: finish_exploratory_h200.sh status|fetch|submit-gptoss|submit-qwen}"

SLUGS=(
  gemma-3-27b-it-h200 gemma-3-27b-it-h200-q1a gemma-3-27b-it-h200-q1b gemma-3-27b-it-h200-q1c
  gpt-oss-20b-h200 gpt-oss-20b-h200-q1a gpt-oss-20b-h200-q1b gpt-oss-20b-h200-q1c
  qwen3-32b-h200-np1024 qwen3-32b-h200-np1024-q1a qwen3-32b-h200-np1024-q1b qwen3-32b-h200-np1024-q1c
)

case "$CMD" in
  status)
    ssh soc 'squeue --me -o "%.10i %.28j %.9P %.8T %.10M %R"; echo; for d in ~/bcf/results/jury-gate/*/arc_challenge/stated-hint; do printf "%-46s votes=%-6s exit=%s\n" "$(basename "$(dirname "$(dirname "$d")")")" "$(wc -l < "$d/votes.jsonl" 2>/dev/null || echo 0)" "$(cat "$d/exit_code.txt" 2>/dev/null || echo -)"; done'
    ;;
  fetch)
    for slug in "${SLUGS[@]}"; do
      local_dir="${REPO}/experiments/results/jury-gate/${slug}/arc_challenge/stated-hint"
      mkdir -p "$local_dir"
      rsync -a --include='votes.jsonl' --include='panel_labels.jsonl' --include='gate_report.json' \
        --include='run_summary.json' --include='checkpoint.json' --include='run.log' \
        --include='exit_code.txt' --include='server-*.log' --exclude='*' \
        "soc:bcf/results/jury-gate/${slug}/arc_challenge/stated-hint/" "$local_dir/" 2>/dev/null || true
      [ -f "${local_dir}/gate_report.json" ] && cp "${local_dir}/gate_report.json" \
        "${REPO}/experiments/jury/gate_report_${slug}.json"
    done
    echo "[fetch] done; budget.md untouched"
    ;;
  submit-gptoss)
    ssh soc 'cd ~/bcf/repo-jury && bash bcf/jury_wave.sh bcf/judges_gate_h200_explore_gptoss.tsv'
    ;;
  submit-qwen)
    ssh soc 'cd ~/bcf/repo-jury && bash bcf/jury_wave.sh bcf/judges_gate_h200_explore_qwen_np1024.tsv'
    ;;
  *)
    echo "unknown command $CMD" >&2; exit 2 ;;
esac
