#!/bin/bash
# A stand-in for bcf/env.sh, used ONLY by tests/harness/render_serve_command.sh.
#
# bcf/serve_and_run.sbatch sources "${BCF_ENV_SH:-$HOME/bcf/repo/bcf/env.sh}", which is
# the documented extension point, so pointing BCF_ENV_SH here needs no change to the
# script under test. Every function the script calls before it launches the server is
# stubbed to a fixed answer, so the render depends on the SCRIPT and on the exported
# BCF_* variables and on nothing else: no cluster, no Hub call, no free-port draw.
#
# bcf_guard_register_child is the stop point. The script calls it on the line after it
# backgrounds `vllm serve`, so waiting for that child and then exiting 99 captures the
# rendered command with nothing downstream of the serve line having run.

set -uo pipefail

export BCF_ROOT="${BCF_RENDER_ROOT:?BCF_RENDER_ROOT must be set}"
export BCF_REPO="${BCF_ROOT}/repo"
export BCF_RESULTS="${BCF_ROOT}/results"
export SCRATCH="${BCF_ROOT}/scratch"
export HF_HOME="${SCRATCH}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export OUTLINES_CACHE_DIR="${SCRATCH}/outlines"
export VLLM_CACHE_ROOT="${SCRATCH}/vllm"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$OUTLINES_CACHE_DIR" "$VLLM_CACHE_ROOT"
export VLLM_USE_FLASHINFER_SAMPLER="${BCF_KEEP_FLASHINFER_SAMPLER:-0}"
export PYTHONUNBUFFERED=1

bcf_pick_port() { echo "${BCF_RENDER_PORT:-18000}"; }

bcf_revision() {
  if [ "${BCF_RENDER_REVISION_FAILS:-0}" = "1" ]; then return 1; fi
  echo "${BCF_RENDER_HUB_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
}

bcf_assert_devices() { echo "[devices] render stub: pretending ${1} device(s)"; return 0; }
bcf_install_exit_guard() { :; }
bcf_wait_server_ready() { return 0; }
bcf_guard_register_child() { wait "${1:-}" 2>/dev/null; exit 99; }
