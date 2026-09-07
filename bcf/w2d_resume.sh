#!/bin/bash
# The W2d cluster sequence, in order, one verb at a time, for the moment `ssh soc` answers.
#
# Every step is idempotent and every step REFUSES rather than guessing when the cluster is
# unreachable, so a half-connected VPN cannot leave a job half-submitted. It never cancels
# anything, never uses a wildcard scancel, and never submits two a100-80 tables at once.
#
#   bcf/w2d_resume.sh reachable        is ssh soc answering right now (exit 0 yes, 99 no)
#   bcf/w2d_resume.sh sync             rsync this worktree to ~/bcf/repo-jury
#   bcf/w2d_resume.sh gemma            fetch the three Gemma exploratory runs and recompute
#   bcf/w2d_resume.sh why <slug>       print the run-log lines that explain that run's exit
#   bcf/w2d_resume.sh explore-qwen     submit the h200 Qwen at num_predict 1024
#   bcf/w2d_resume.sh explore-gptoss   submit the h200 gpt-oss
#   bcf/w2d_resume.sh a100-qwen        submit the pinned a100-80 Qwen  (ONE table at a time)
#   bcf/w2d_resume.sh a100-gemma       submit the pinned a100-80 Gemma (ONE table at a time)
#   bcf/w2d_resume.sh a100-gptoss      submit the pinned a100-80 gpt-oss (ONE at a time)
#   bcf/w2d_resume.sh watch            the queue and every jury-gate results directory
#   bcf/w2d_resume.sh panel            the panel gate for Q1 a, b and c from whatever exists
#   bcf/w2d_resume.sh tables           rebuild the tables in GATE-Q1-COMPARISON.md
#
# The order that matters: sync BEFORE any submit (the cluster copy lacks the exit guard
# until then), and each a100-80 table only after the previous one has left the queue, which
# `watch` shows and which bcf/jury_wave.sh enforces on the live per-user cap counting every
# campaign's jobs.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${BCF_VENV_PY:-/Users/maksimsilchenko/Developer/bayes-cot-faithfulness/.venv/bin/python}"
RETRY="${REPO}/bcf/ssh_retry.sh"
GEMMA_SLUGS=(gemma-3-27b-it-h200-q1a gemma-3-27b-it-h200-q1b gemma-3-27b-it-h200-q1c)

need_cluster() {
  bash "$RETRY" --wall 45 --tries 2 --gap 10 -- 'true' >/dev/null 2>&1 && return 0
  echo "[w2d] REFUSING: ssh soc does not answer. Nothing was submitted and nothing was fetched." >&2
  echo "[w2d] Check the VPN: a connected Cisco client shows a utun with a 10.195.x inet address" >&2
  echo "[w2d]   ifconfig | grep -B4 '10\\.195\\.'   and   route -n get 192.168.51.148" >&2
  exit 99
}

CMD="${1:-}"; shift || true
case "$CMD" in
  reachable)
    bash "$RETRY" --wall 45 --tries 1 -- 'hostname; date' && exit 0 || exit 99 ;;
  sync)     need_cluster; bash "${REPO}/bcf/w2c.sh" sync ;;
  watch)    need_cluster; bash "${REPO}/bcf/w2c.sh" status ;;
  gemma)    need_cluster; bash "${REPO}/bcf/w2c.sh" fetch "${GEMMA_SLUGS[@]}" ;;
  why)
    need_cluster
    SLUG="${1:?usage: w2d_resume.sh why <results slug>}"
    # gate.py returns 1 for a FAIL VERDICT and something else for a failure, so the exit
    # code alone cannot tell the two apart. These are the lines that can.
    bash "$RETRY" --wall 60 --tries 3 -- "
      d=\$HOME/bcf/results/jury-gate/${SLUG}/arc_challenge/stated-hint;
      echo '--- exit code ---'; cat \$d/exit_code.txt 2>/dev/null;
      echo '--- the variant-finished and verdict lines ---';
      grep -hE 'finished at|worst status|\"verdict\"|Traceback|Error|error\":' \$d/../../*.log \$d/run.log 2>/dev/null | tail -40;
      echo '--- last 25 lines of the run log ---'; tail -25 \$d/run.log 2>/dev/null" ;;
  explore-qwen)   need_cluster; bash "${REPO}/bcf/w2c.sh" submit bcf/judges_gate_h200_explore_qwen_np1024.tsv ;;
  explore-gptoss) need_cluster; bash "${REPO}/bcf/w2c.sh" submit bcf/judges_gate_h200_explore_gptoss.tsv ;;
  a100-qwen)      need_cluster; bash "${REPO}/bcf/w2c.sh" submit bcf/judges_gate_a100_qwen_2026-09-07.tsv ;;
  a100-gemma)     need_cluster; bash "${REPO}/bcf/w2c.sh" submit bcf/judges_gate_a100_gemma_2026-09-07.tsv ;;
  a100-gptoss)    need_cluster; bash "${REPO}/bcf/w2c.sh" submit bcf/judges_gate_a100_gptoss_2026-09-07.tsv ;;
  panel)
    # Local only. It uses whichever of the three judges have votes on this Mac and says so
    # in the report; a panel missing a judge is a smaller panel, not a substitute for one.
    cd "$REPO" || exit 2
    for v in a b c; do
      args=()
      for pair in "gemma-3-27b-it:gemma-3-27b-it-a100-q1${v} gemma-3-27b-it-h200-q1${v}" \
                  "gpt-oss-20b:gpt-oss-20b-a100-q1${v} gpt-oss-20b-h200-q1${v}" \
                  "llama-3.3-70b-fp8:llama-3.3-70b-fp8-q1${v} llama-3.3-70b-fp8"; do
        key="${pair%%:*}"
        for slug in ${pair#*:}; do
          d="experiments/results/jury-gate/${slug}/arc_challenge/stated-hint"
          # The PINNED directory is listed first in every pair, so a pinned run wins over
          # an exploratory one for the same judge and the fallback is visible in the report.
          if [ -s "$d/votes.jsonl" ]; then args+=(--votes "${key}=${d}"); break; fi
        done
      done
      if [ ${#args[@]} -eq 0 ]; then echo "[panel] Q1 ${v}: no vote files at all, skipped"; continue; fi
      PYTHONPATH="${REPO}/src" "$VENV_PY" -m experiments.jury.panel_gate --q1 "$v" \
        "${args[@]}" --out "experiments/jury/panel_report_q1${v}.json"
    done ;;
  tables)   bash "${REPO}/bcf/rebuild_comparison.sh" ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
