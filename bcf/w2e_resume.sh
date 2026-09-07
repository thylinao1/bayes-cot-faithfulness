#!/bin/bash
# The W2e cluster sequence, in the order that survives a link that keeps dropping.
#
# The policy this file encodes: SUBMIT EVERYTHING THAT CAN BE SUBMITTED FIRST, fetch
# afterwards. A submitted job survives a VPN drop; an unsubmitted one is lost time. So
# the smoke run goes out before any gate, and the one h200 card and the a100-80 pool are
# each fed one job at a time because that is what their caps allow.
#
#   bcf/w2e_resume.sh reachable      is ssh soc answering right now (exit 0 yes, 99 no)
#   bcf/w2e_resume.sh sync           rsync this worktree to ~/bcf/repo-jury  (DO FIRST)
#   bcf/w2e_resume.sh smoke          submit the a3f skeleton smoke run (bcf-a3f-skel)
#   bcf/w2e_resume.sh smoke-status   its queue row, its exit code and its preflight json
#   bcf/w2e_resume.sh smoke-fetch    mirror it into experiments/results/a3f-skeleton/
#   bcf/w2e_resume.sh explore-qwen   submit the h200 Qwen3-32B at num_predict 1024
#   bcf/w2e_resume.sh explore-gptoss submit the h200 gpt-oss-20b (AFTER the Qwen ends)
#   bcf/w2e_resume.sh a100-qwen      submit the pinned a100-80 Qwen  (ONE at a time)
#   bcf/w2e_resume.sh a100-gemma     submit the pinned a100-80 Gemma (ONE at a time)
#   bcf/w2e_resume.sh a100-gptoss    submit the pinned a100-80 gpt-oss (ONE at a time)
#   bcf/w2e_resume.sh gemma          fetch the three Gemma exploratory runs, recompute
#   bcf/w2e_resume.sh why <slug>     the run-log lines that explain that run's exit code
#   bcf/w2e_resume.sh fetch <slug>.. fetch any jury-gate slug and recompute its report
#   bcf/w2e_resume.sh watch          the queue and every jury-gate results directory
#   bcf/w2e_resume.sh panel          the panel gate for Q1 a, b and c from what exists
#   bcf/w2e_resume.sh tables         rebuild the tables in GATE-Q1-COMPARISON.md
#
# THE ORDERING CONSTRAINTS, none of them optional:
#   * sync before any submit. The cluster copy has neither this session's smoke script
#     nor the chain-repeats arm until then.
#   * the h200-141 per-user cap is ONE card and xgpk0 is the only h200 node, and it sits
#     in the 3-hour `gpu` partition. So explore-qwen and explore-gptoss are serial and
#     each is a 2:50 job that resumes across submissions.
#   * the a100-80 cap is FOUR cards counting every campaign on this account, and the
#     alta campaign holds some of them. jury_wave.sh counts live and refuses with the
#     denominators printed; a refusal is the correct outcome, not an error to work
#     around, and nothing of alta's is ever cancelled to make room.
#   * the a100-40 pool the smoke run uses is a different type with its own cap of 8, so
#     the smoke run does not compete with any of the gates.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${BCF_VENV_PY:-/Users/maksimsilchenko/Developer/bayes-cot-faithfulness/.venv/bin/python}"
RETRY="${REPO}/bcf/ssh_retry.sh"
W2D="${REPO}/bcf/w2d_resume.sh"
SMOKE_REMOTE='$HOME/bcf/results/a3f-skeleton/qwen3-8b/arc_challenge/stated-hint'
SMOKE_LOCAL="${REPO}/experiments/results/a3f-skeleton/qwen3-8b/arc_challenge/stated-hint"

need_cluster() {
  bash "$RETRY" --wall 45 --tries 2 --gap 10 -- 'true' >/dev/null 2>&1 && return 0
  echo "[w2e] REFUSING: ssh soc does not answer. Nothing was submitted and nothing was fetched." >&2
  echo "[w2e] The VPN: a connected Cisco client shows a utun with a 10.195.x inet address." >&2
  echo "[w2e]   ifconfig | grep -B4 '10\\.195\\.'   and   route -n get 192.168.51.148" >&2
  exit 99
}

CMD="${1:-}"; shift || true
case "$CMD" in
  smoke)        need_cluster; bash "${REPO}/bcf/a3f_smoke.sh" ;;
  smoke-status) bash "${REPO}/bcf/a3f_smoke.sh" status ;;
  smoke-fetch)
    need_cluster
    mkdir -p "$SMOKE_LOCAL"
    # Named includes, not a bare rsync of the directory: a smoke cell also writes the
    # per-arm transcripts, which are large and are not what this lane reads.
    rsync -a \
      --include='run_meta.json' --include='determinism_preflight.json' \
      --include='determinism_preflight_probe.json' --include='logprob_check.json' \
      --include='arms_summary.json' --include='run.log' --include='exit_code.txt' \
      --include='throughput.json' --include='server.log' \
      --include='chain_repeats*.json' --include='*_summary.json' \
      --exclude='*' \
      "soc:bcf/results/a3f-skeleton/qwen3-8b/arc_challenge/stated-hint/" "$SMOKE_LOCAL/" || true
    echo "=== mirrored into ${SMOKE_LOCAL}"
    ls -la "$SMOKE_LOCAL"
    ;;
  fetch)
    [ $# -gt 0 ] || { echo "usage: w2e_resume.sh fetch <slug> [<slug>...]" >&2; exit 2; }
    need_cluster; bash "${REPO}/bcf/w2c.sh" fetch "$@" ;;
  reachable|sync|gemma|why|explore-qwen|explore-gptoss|a100-qwen|a100-gemma|a100-gptoss|watch|panel|tables)
    bash "$W2D" "$CMD" "$@" ;;
  *) sed -n '2,38p' "$0"; exit 2 ;;
esac
