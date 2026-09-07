#!/bin/bash
# W6 external validity: submit whatever the cap now admits, then report.
#
# Run it on the cluster. It never cancels anything, it never bypasses the wave guard, and
# it is safe to run repeatedly: bcf/w6_wave.sh refuses when a cap would break, and
# score_faithcot resumes from votes already on disk rather than regenerating them.
#
#   ssh soc 'cd ~/bcf/repo-w6 && bash bcf/w6_resume.sh'
#
# The h200-141 cap is ONE card for the whole account. The wave counts pending as well as
# running, so an h200 job cannot be queued behind another one: it goes in when the card is
# actually free.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
ITEMS="$HOME/bcf/repo-w6/experiments/data/external/w6_items_combined.jsonl"
[ -f "$ITEMS" ] || { echo "no items file at $ITEMS" >&2; exit 2; }

for table in bcf/w6_judges_a100_gemma.tsv bcf/w6_judges_h200_llama.tsv; do
  key="$(awk 'NR==1{print $1}' "$table")"
  if squeue --me -h -n "bcf-w6-${key}" -t RUNNING,PENDING | grep -q .; then
    echo "[resume] bcf-w6-${key} already in the queue, skipping"
    continue
  fi
  if [ -s "$HOME/bcf/results/w6-external/${key}/faithcot/no-cue/score_summary.json" ]; then
    echo "[resume] ${key} already finished (score_summary.json exists), skipping"
    continue
  fi
  echo "[resume] trying ${table}"
  bash bcf/w6_wave.sh --items "$ITEMS" "$table"
done

echo
echo "[resume] state:"
squeue --me -o '%.10i %.26j %.9P %.8T %.10M %R' | grep -E 'JOBID|bcf-w6-' || echo "  no bcf-w6-* jobs in the queue"
for d in "$HOME"/bcf/results/w6-external/*/faithcot/no-cue; do
  [ -d "$d" ] || continue
  n=$(wc -l < "$d/votes.jsonl" 2>/dev/null || echo 0)
  echo "  $(basename "$(dirname "$(dirname "$d")")"): ${n} votes, exit_code=$(cat "$d/exit_code.txt" 2>/dev/null || echo none)"
done
