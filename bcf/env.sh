#!/bin/bash
# Sourced by every bcf sbatch script. Sets up the conda env, node-local scratch, and
# the paths the runners expect. Never run standalone; it assumes $SLURM_JOB_ID exists.
#
# Built by ~/bcf/env-setup.sbatch (job 825198, EXIT_CODE=0, 2026-09-06):
#   vllm 0.28.0, nnsight 0.7.0, numpyro 0.21.0, torch 2.13.0+cu130.

# NOT `set -e`: this file is sourced, so -e would leak into the calling sbatch script,
# where a plain `[ -n "$OPTIONAL" ] && args+=(...)` that legitimately returns 1 kills the
# job. Each caller decides its own error mode.
set -uo pipefail

source ~/miniconda3/bin/activate
conda activate bcf

# Home is persistent but has a shared budget; model weights live on node-local scratch
# and die with the job (NUS-COMPUTE.md 1.5, and the "home quota" risk row in the plan).
export SCRATCH="/tmp/${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
export HF_HOME="${SCRATCH}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export OUTLINES_CACHE_DIR="${SCRATCH}/outlines"
export VLLM_CACHE_ROOT="${SCRATCH}/vllm"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$OUTLINES_CACHE_DIR" "$VLLM_CACHE_ROOT"

export BCF_ROOT="${BCF_ROOT:-$HOME/bcf}"
export BCF_REPO="${BCF_REPO:-$BCF_ROOT/repo}"
export BCF_RESULTS="${BCF_RESULTS:-$BCF_ROOT/results}"
export PYTHONPATH="${BCF_REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- helpers -------------------------------------------------------------------

# Number of devices vLLM will actually see. Asked of torch, not of the SBATCH request
# and not of a grep over `nvidia-smi -L`, because on this cluster all three disagree:
# Slurm's a100-40 and a100-80 GRES hand out MIG slices, so `nvidia-smi -L` prints ONE
# "GPU 0:" line with two indented "MIG ... Device N:" lines under it, while CUDA sees
# two devices. Job 825248 refused on exactly that mismatch. torch.cuda.device_count()
# is what the serving process will see, so it is the number to assert on.
bcf_visible_devices() {
  python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0
}

# True when the allocation is MIG slices rather than whole cards. This matters more
# than the count: MIG instances have no NVLink or peer-to-peer path between them, so
# NCCL cannot build a tensor-parallel group across them however many are visible.
bcf_is_mig() {
  nvidia-smi -L 2>/dev/null | grep -q '^ *MIG '
}

# Refuse BEFORE any weight download if the card count is not what the model needs.
# A 70B started on one card downloads 140 GB to scratch and then OOMs; this turns
# that into a five-second exit.
bcf_assert_devices() {
  local want="$1" got
  got="$(bcf_visible_devices)"
  echo "[devices] torch.cuda.device_count()=${got}; requested tensor-parallel size ${want}"
  nvidia-smi -L || true
  if [ "$got" -ne "$want" ]; then
    echo "[devices] REFUSING: visible device count ${got} != requested tensor-parallel size ${want}" >&2
    return 1
  fi
  if [ "$want" -gt 1 ] && bcf_is_mig; then
    echo "[devices] REFUSING: this allocation is MIG slices (nvidia-smi -L shows MIG devices)." >&2
    echo "[devices]   MIG instances have no peer-to-peer path, so NCCL cannot form a" >&2
    echo "[devices]   tensor-parallel group across them. Use a GPU type served as whole" >&2
    echo "[devices]   cards (h100-96) for any tensor-parallel size above 1." >&2
    return 1
  fi
  return 0
}

# Resolve and print the HF revision sha for the CONTRACT hf_revision field, and pin
# the served weights to it. A floating 'main' makes a run unreproducible.
bcf_revision() {
  python - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
print(HfApi().model_info(sys.argv[1]).sha)
PY
}
