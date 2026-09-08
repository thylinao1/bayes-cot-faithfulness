#!/bin/bash
# A stand-in for bcf/env.sh, used ONLY by tests/harness/render_judge_command.sh.
#
# bcf/judge_serve.sbatch sources "${BCF_ENV_SH:-$HOME/bcf/repo-jury/bcf/env.sh}", which is
# the documented extension point, so pointing BCF_ENV_SH here needs no change to the
# script under test. Every function it calls before it runs the judging is stubbed to a
# fixed answer, so what comes out depends on the SCRIPT and on the exported BCF_*
# variables and on nothing else: no cluster, no Hub call, no free-port draw, no weights.
#
# The exit guard is NOT stubbed. bcf/exit_guard.sh is sourced from the repository under
# test, so the exit codes this harness renders are the ones the real guard writes.

set -uo pipefail

export BCF_ROOT="${BCF_RENDER_ROOT:?BCF_RENDER_ROOT must be set}"
export SCRATCH="${BCF_ROOT}/scratch"
export HF_HOME="${SCRATCH}/hf"
mkdir -p "$SCRATCH" "$HF_HOME"
export PYTHONUNBUFFERED=1

bcf_pick_port() { echo "${BCF_RENDER_PORT:-18000}"; }

bcf_revision() { echo "${BCF_RENDER_HUB_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}"; }

bcf_assert_devices() { echo "[devices] render stub: pretending ${1} device(s)"; return 0; }

bcf_wait_server_ready() { echo "[serve] render stub: server ready"; return 0; }

bcf_prewarm_harmony() { echo "[harmony] render stub: vocab cached"; return 0; }

# The real one, copied in behaviour and kept here so the render does not depend on which
# env.sh a machine has: 0 and 1 with no completion marker become 12, anything above 1
# passes through.
bcf_gate_stage_status() {
  local status="${1:?}" report="${2:?}"
  if [ "$status" -gt 1 ]; then printf '%s\n' "$status"
  elif [ -f "$report" ]; then printf '%s\n' "$status"
  else printf '%s\n' 12
  fi
  return 0
}
