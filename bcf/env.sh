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

# No CUDA JIT between the allocation and the run. vLLM 0.28.0 picks FlashInfer for
# top-p / top-k sampling and JIT-compiles flashinfer's sampling.cu on first use, AFTER the
# weights are loaded. That compile is node dependent: job 825246 ran it fine on xgph12,
# job 825480 died on xgph10 with
#   flashinfer/sampling.cuh(623): error: class "cub::_V_300302_SM_800::BlockAdjacentDifference<...>"
#   ninja: build stopped: subcommand failed
# after 5 minutes of allocation, model download and load. Every generation in this project
# runs at temperature 0 (frozen decoding constant) except the pre-registered k=32 T=0.7
# arm, so the sampler kernel is not on the measurement path, and trading it for the
# PyTorch sampler removes a node-dependent build from between a scarce card and a run.
# Unset BCF_KEEP_FLASHINFER_SAMPLER to opt back in on a node where it works.
export VLLM_USE_FLASHINFER_SAMPLER="${BCF_KEEP_FLASHINFER_SAMPLER:-0}"

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
  if [ "$want" -gt 1 ] && bcf_is_mig; then
    echo "[devices] REFUSING: this allocation is MIG slices, not whole cards." >&2
    nvidia-smi -L | sed 's/^/[devices]   /' >&2
    echo "[devices]   CUDA exposes at most ONE MIG instance to a process, so" >&2
    echo "[devices]   torch.cuda.device_count() reports ${got} even though Slurm granted" >&2
    echo "[devices]   ${SLURM_GPUS_PER_NODE:-?} and CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}." >&2
    echo "[devices]   Tensor parallelism across MIG slices is impossible, not merely slow." >&2
    echo "[devices]   Verified 2026-09-06: a100-40 is MIG 3g.40gb of an A100 80GB (job" >&2
    echo "[devices]   825248) and h100-47 is MIG 3g.47gb of an H100 NVL (job 825283)." >&2
    echo "[devices]   h100-96 is the only whole-card multi-GPU type within this account's" >&2
    echo "[devices]   caps (2 per node, per-user cap 2)." >&2
    return 1
  fi
  if [ "$got" -ne "$want" ]; then
    echo "[devices] REFUSING: visible device count ${got} != requested tensor-parallel size ${want}" >&2
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
