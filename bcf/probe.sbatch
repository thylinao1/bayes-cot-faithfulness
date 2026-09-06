#!/bin/bash
#SBATCH --job-name=bcf-trackg-probe
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Track G determinism and capacity probe. Serves Qwen3-8B on ONE a100-40 MIG slice with
# the same vLLM flags as the Phase-1 skeleton, then runs a fixed set of 30 generations
# and 30 forced-logprob items at each concurrency level.
#
# This is a measurement, not a powered run: 30 items, no arms, no cell, nothing written
# under a substrate/cue path. It exists because every rate measured so far was taken
# through a one-request-at-a-time client.
#
# --exclude=xgpj0: that node reports CUDA device count 0 while nvidia-smi lists the card
# (NUS-COMPUTE.md 1.8), and job 825479 died on it in one second.
# --mem=64G: the partition default is 3G of host RAM and a vLLM engine core dies under it
# with "Engine core initialization failed" and no root cause in the server log (job
# 825536, sacct State=OUT_OF_MEMORY, ReqMem=3G, MaxRSS=6.3G).

set -uo pipefail

export BCF_REPO="${BCF_REPO:-$HOME/bcf/repo-trackg}"
source "${BCF_ENV_SH:-$BCF_REPO/bcf/env.sh}"
set +e

MODEL="${BCF_MODEL:-Qwen/Qwen3-8B}"
PORT="${BCF_PORT:-8000}"
SEED="${BCF_SEED:-7}"
MAX_LEN="${BCF_MAX_LEN:-8192}"
GPU_UTIL="${BCF_GPU_UTIL:-0.90}"
NUM_PREDICT="${BCF_NUM_PREDICT:-320}"
N_ITEMS="${BCF_N_ITEMS:-30}"
LEVELS="${BCF_LEVELS:-1,8,32,64}"
# BCF_BATCH_INVARIANT=1 runs the probe TWICE in this one job, on one card:
#   phase A: the server WITHOUT the flag, levels "1" only, writing its concurrency-1
#            completions and letter logprobs to baseline_off_raw.json;
#   phase B: the server WITH VLLM_BATCH_INVARIANT=1, every level, comparing each level
#            against its own flag-on concurrency-1 row AND against phase A's file.
# Phase A exists because job 825548 wrote no raw completions (probe_results.json keeps
# only the summary), so the flag-off baseline cannot be quoted across jobs. Re-measuring
# it on the same node in the same job is also the stronger comparison.
# Unset (the default) leaves this script on exactly the path job 825548 took.
BATCH_INVARIANT="${BCF_BATCH_INVARIANT:-}"
SERVER_WAIT="${BCF_SERVER_WAIT:-2400}"
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}"
# The pools are gitignored (licence), so the substrate file is read from the repo the
# earlier waves already populated, never copied into this lane's tree.
DATA="${BCF_DATA:-$HOME/bcf/repo/experiments/data/arc_challenge.json}"
OUT_DIR="${BCF_OUT_DIR:-$BCF_RESULTS/trackg-probe}"

mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
EXIT_FILE="${OUT_DIR}/exit_code.txt"
BASE_URL="http://127.0.0.1:${PORT}/v1"
export BCF_REQUEST_LOG="${OUT_DIR}/requests.jsonl"

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

finish() {
  local code=$?
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  echo "$code" > "$EXIT_FILE"
  # ReqMem and MaxRSS into the run's own meta, per the 2026-09-07 memory ruling.
  sacct -j "${SLURM_JOB_ID}" --format=JobID,State,ExitCode,ReqMem,MaxRSS,Elapsed,NodeList \
    > "${OUT_DIR}/sacct.txt" 2>/dev/null || true
  echo "[done] exit_code=${code} -> ${EXIT_FILE}" | ts | tee -a "$LOG"
  exit "$code"
}
trap finish EXIT

{
  echo "=== bcf trackg probe: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "model=${MODEL} n_items=${N_ITEMS} levels=${LEVELS} data=${DATA}"
  echo "repo=${BCF_REPO} out_dir=${OUT_DIR}"
  echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
  echo "BCF_BATCH_INVARIANT=${BATCH_INVARIANT:-<unset>}"
} | ts | tee -a "$LOG"

if [ ! -f "$DATA" ]; then
  echo "[data] REFUSING: ${DATA} does not exist" | ts | tee -a "$LOG" >&2
  exit 2
fi

if ! bcf_assert_devices 1 2>&1 | ts | tee -a "$LOG"; then
  exit 3
fi

REVISION="$(bcf_revision "$MODEL" 2>>"$LOG")"
if [ -z "$REVISION" ]; then
  echo "[revision] REFUSING: no HF revision sha for ${MODEL}" | ts | tee -a "$LOG" >&2
  exit 4
fi
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
echo "[revision] ${MODEL} @ ${REVISION} (vllm ${VLLM_VERSION})" | ts | tee -a "$LOG"

BCF_META_TEMPLATE_KWARGS="$TEMPLATE_KWARGS" python - <<PY >> "$LOG" 2>&1
import json, os, pathlib
pathlib.Path("${OUT_DIR}/run_meta.json").write_text(json.dumps({
    "job_id": "${SLURM_JOB_ID}", "model": "${MODEL}", "hf_revision": "${REVISION}",
    "backend": "openai", "backend_version": "vllm ${VLLM_VERSION}",
    "purpose": "track G concurrency probe; NOT a cell, NOT a powered run",
    "tensor_parallel_size": 1, "seed": ${SEED},
    "temperature": 0.0, "num_predict": ${NUM_PREDICT}, "n_items": ${N_ITEMS},
    "levels": "${LEVELS}", "data": "${DATA}",
    "chat_template_kwargs": json.loads(os.environ["BCF_META_TEMPLATE_KWARGS"]),
    "gpu": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip(),
    "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    "batch_invariant_switch": "${BATCH_INVARIANT}",
    "two_phase": "${BATCH_INVARIANT}" == "1",
}, indent=2))
PY

VLLM_ARGS=(
  --revision "$REVISION" --served-model-name "$MODEL"
  --port "$PORT" --host 127.0.0.1
  --tensor-parallel-size 1
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --seed "$SEED"
  --download-dir "${HF_HOME}/download"
)

# start_server <phase-tag> <0|1 batch invariant>. Writes server-<tag>.log so a refusal
# or an attention-backend fallback is readable per phase rather than interleaved.
start_server() {
  local tag="$1" bi="$2"
  local slog="${OUT_DIR}/server-${tag}.log"
  if [ "$bi" = "1" ]; then
    export VLLM_BATCH_INVARIANT=1
  else
    unset VLLM_BATCH_INVARIANT
  fi
  echo "[serve:${tag}] VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-<unset>}" | ts | tee -a "$LOG"
  echo "[serve:${tag}] vllm serve ${MODEL} ${VLLM_ARGS[*]}" | ts | tee -a "$LOG"
  vllm serve "$MODEL" "${VLLM_ARGS[@]}" >> "$slog" 2>&1 &
  SERVER_PID=$!
  local deadline=$(( $(date +%s) + SERVER_WAIT ))
  until curl -sf "${BASE_URL}/models" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[serve:${tag}] server died during startup; last 60 lines:" | ts | tee -a "$LOG"
      tail -60 "$slog" | ts | tee -a "$LOG"
      # A server that REFUSES the flag is the result, not a fault: the lines saying so
      # are already in $slog and the exit code names which phase died.
      return 5
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "[serve:${tag}] no answer within ${SERVER_WAIT}s" | ts | tee -a "$LOG"
      tail -60 "$slog" | ts | tee -a "$LOG"
      return 6
    fi
    sleep 10
  done
  echo "[serve:${tag}] up at ${BASE_URL}" | ts | tee -a "$LOG"
  # Which attention backend the server actually selected. The batch-invariant path is
  # honored by FLASH_ATTN (num_splits forced to 1); a fallback to another backend would
  # make the flag a no-op, so the line is grepped out rather than assumed.
  grep -iE "attention backend|Using .* backend|batch.?invariant" "$slog" \
    | tail -20 | sed "s/^/[serve:${tag}] /" | ts | tee -a "$LOG" || true
  return 0
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 10
    kill -9 "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  # The port must be free before the next server binds it.
  sleep 10
}

RAW_OFF="${OUT_DIR}/baseline_off_raw.json"

if [ "$BATCH_INVARIANT" = "1" ]; then
  echo "[phase A] flag OFF, concurrency 1 only, writing the raw baseline" | ts | tee -a "$LOG"
  start_server "off" 0 || exit $?
  python "${BCF_REPO}/bcf/concurrency_probe.py" \
    --base-url "$BASE_URL" --model "$MODEL" --data "$DATA" \
    --n-items "$N_ITEMS" --levels 1 --num-predict "$NUM_PREDICT" \
    --seed "$SEED" --chat-template-kwargs "$TEMPLATE_KWARGS" \
    --label "batch_invariant_off" \
    --raw-out "$RAW_OFF" \
    --out "${OUT_DIR}/probe_results_off.json" 2>&1 | ts | tee -a "$LOG"
  PHASE_A=${PIPESTATUS[0]}
  echo "[phase A] finished with status ${PHASE_A}" | ts | tee -a "$LOG"
  stop_server
  if [ "$PHASE_A" -ne 0 ]; then exit "$PHASE_A"; fi

  echo "[phase B] flag ON, levels ${LEVELS}" | ts | tee -a "$LOG"
  start_server "bi" 1 || exit $?
  python "${BCF_REPO}/bcf/concurrency_probe.py" \
    --base-url "$BASE_URL" --model "$MODEL" --data "$DATA" \
    --n-items "$N_ITEMS" --levels "$LEVELS" --num-predict "$NUM_PREDICT" \
    --seed "$SEED" --chat-template-kwargs "$TEMPLATE_KWARGS" \
    --label "batch_invariant_on" \
    --baseline-raw "$RAW_OFF" \
    --raw-out "${OUT_DIR}/baseline_on_raw.json" \
    --out "${OUT_DIR}/probe_results.json" 2>&1 | ts | tee -a "$LOG"
  PROBE_STATUS=${PIPESTATUS[0]}
else
  start_server "single" 0 || exit $?
  python "${BCF_REPO}/bcf/concurrency_probe.py" \
    --base-url "$BASE_URL" --model "$MODEL" --data "$DATA" \
    --n-items "$N_ITEMS" --levels "$LEVELS" --num-predict "$NUM_PREDICT" \
    --seed "$SEED" --chat-template-kwargs "$TEMPLATE_KWARGS" \
    --out "${OUT_DIR}/probe_results.json" 2>&1 | ts | tee -a "$LOG"
  PROBE_STATUS=${PIPESTATUS[0]}
fi
echo "[probe] finished with status ${PROBE_STATUS}" | ts | tee -a "$LOG"
exit "$PROBE_STATUS"
