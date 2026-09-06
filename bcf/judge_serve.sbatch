#!/bin/bash
#SBATCH --job-name=bcf-jury-serve
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --exclude=xgpj0
#
# Serve one or two jury judges on the allocated card, run the synthetic judge gate against
# them, and EXIT. No idle server: the gate is the job, and when it finishes the card goes
# back to the pool.
#
# xgpj0 is excluded in the header, not left to the shell: that node's conda entrypoints are
# broken (NUS-COMPUTE 1.8), CUDA reports no device on it, and job 825479 spent an a100-80
# slot discovering that at 02:10 on 2026-09-07.
#
# The four judges and their pinned revisions live in experiments/jury/family_map.py and are
# read from there by bcf/judge_spec.py. This script never writes an HF id or a revision.
#
#   sbatch --job-name=bcf-jury-llama-fp8 --gpus=h200-141 \
#          --export=ALL,BCF_JUDGES=llama-3.3-70b-fp8,BCF_GATE_ITEMS=$HOME/bcf/gate_items.jsonl \
#          ~/bcf/repo-jury/bcf/judge_serve.sbatch
#
#   sbatch --job-name=bcf-jury-a100-pair --gpus=a100-80 \
#          --export=ALL,BCF_JUDGES=gemma-3-27b-it,gpt-oss-20b,... (see BCF_JUDGES note)
#
# Required : BCF_JUDGES (comma separated judge keys), BCF_GATE_ITEMS
# Optional : BCF_OUT_ROOT BCF_PORT_BASE BCF_CONCURRENCY BCF_GATE_LIMIT BCF_SEED
#            BCF_NUM_PREDICT BCF_SERVER_WAIT BCF_TP BCF_JUDGE_CACHE BCF_CACHE_BUDGET_GB
#            BCF_RESUME

set -uo pipefail

# env.sh defaults BCF_REPO to $BCF_ROOT/repo, which is the tree the OTHER lane's live jobs
# (825510, 825511) read at runtime. This lane must never run out of it, and must never
# write into it. Pin repo-jury BEFORE sourcing, and refuse if anything points back.
export BCF_REPO="${BCF_REPO:-$HOME/bcf/repo-jury}"
source "${BCF_ENV_SH:-$HOME/bcf/repo-jury/bcf/env.sh}"
set +e
case "$BCF_REPO" in
  */repo|*/repo/) echo "[repo] REFUSING: BCF_REPO=${BCF_REPO} is the other lane's tree" >&2; exit 10 ;;
esac
if [ ! -d "${BCF_REPO}/experiments/jury" ]; then
  echo "[repo] REFUSING: ${BCF_REPO} carries no experiments/jury; wrong tree" >&2
  exit 10
fi

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

JUDGES="${BCF_JUDGES:?BCF_JUDGES is required, e.g. qwen3-32b or gemma-3-27b-it,gpt-oss-20b}"
GATE_ITEMS="${BCF_GATE_ITEMS:?BCF_GATE_ITEMS is required}"
OUT_ROOT="${BCF_OUT_ROOT:-$HOME/bcf/results/jury-gate}"
PORT_BASE="${BCF_PORT_BASE:-8100}"
CONCURRENCY="${BCF_CONCURRENCY:-8}"
GATE_LIMIT="${BCF_GATE_LIMIT:-0}"
SEED="${BCF_SEED:-7}"
NUM_PREDICT="${BCF_NUM_PREDICT:-256}"
SERVER_WAIT="${BCF_SERVER_WAIT:-3600}"
TP="${BCF_TP:-1}"
SUBSTRATE="arc_challenge"
CUE="stated-hint"
RESUME_FLAG=""
[ -n "${BCF_RESUME:-}" ] && RESUME_FLAG="--resume"

# vLLM 0.28 JIT-compiles the FlashInfer top-k/top-p sampler on first use and that build is
# node dependent (job 825480 died on xgph10 after loading the weights). Every judge call is
# temperature 0, so the sampler kernel is off the measurement path. env.sh sets this too;
# it is repeated here so a judge job started with a different env file still gets it.
export VLLM_USE_FLASHINFER_SAMPLER=0

SLUG="$(echo "$JUDGES" | tr ',' '-')"
OUT_DIR="${OUT_ROOT}/${SLUG}/${SUBSTRATE}/${CUE}"
case "$OUT_DIR" in
  *"/${SUBSTRATE}/${CUE}") : ;;
  *) echo "[path] REFUSING: ${OUT_DIR} does not end in /${SUBSTRATE}/${CUE}" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
EXIT_FILE="${OUT_DIR}/exit_code.txt"

SERVER_PIDS=()
finish() {
  local code=$?
  for pid in "${SERVER_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null || true
  done
  sleep 5
  for pid in "${SERVER_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -9 "$pid" 2>/dev/null || true
  done
  echo "$code" > "$EXIT_FILE"
  echo "[done] exit_code=${code} -> ${EXIT_FILE}" | ts | tee -a "$LOG"
  exit "$code"
}
trap finish EXIT

{
  echo "=== bcf judge_serve: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "judges=${JUDGES} items=${GATE_ITEMS} out=${OUT_DIR} concurrency=${CONCURRENCY}"
} | ts | tee -a "$LOG"

# --- 1. device assertion, BEFORE any download ----------------------------------
if ! bcf_assert_devices "$TP" 2>&1 | ts | tee -a "$LOG"; then
  exit 3
fi

# --- 2. the judge weight cache -------------------------------------------------
# Judge weights are large and the judges are restarted often, so a per-job /tmp cache pays
# the download every single run. Home is persistent, but the W2 brief's "no home quota" is
# wrong: NUS-COMPUTE 1.2 records a hidden ~500 GB per-user quota, falsified live on
# 2026-07-29 with "Disk quota exceeded (122)" at about 501 GB, and home held 361 GB on
# 2026-09-07. So the cache is persistent WITH A PREFLIGHT: measure the used space, keep a
# working ceiling, and fall back to node-local scratch rather than filling the quota and
# killing every other campaign's writes.
JUDGE_CACHE="${BCF_JUDGE_CACHE:-$HOME/bcf/hf-judges}"
CACHE_BUDGET_GB="${BCF_CACHE_BUDGET_GB:-450}"
mkdir -p "$JUDGE_CACHE"
HOME_USED_GB="$(du -sBG "$HOME" 2>/dev/null | awk '{gsub("G","",$1); print $1+0}')"
HEADROOM_GB=$(( CACHE_BUDGET_GB - HOME_USED_GB ))
# A rough footprint per judge, only used to decide persistent vs scratch.
NEED_GB=0
for key in $(echo "$JUDGES" | tr ',' ' '); do
  case "$key" in
    qwen3-32b) NEED_GB=$(( NEED_GB + 66 )) ;;
    gemma-3-27b-it) NEED_GB=$(( NEED_GB + 56 )) ;;
    gpt-oss-20b) NEED_GB=$(( NEED_GB + 15 )) ;;
    llama-3.3-70b-fp8) NEED_GB=$(( NEED_GB + 72 )) ;;
    *) NEED_GB=$(( NEED_GB + 80 )) ;;
  esac
done
echo "[cache] home used ${HOME_USED_GB} GB of a ${CACHE_BUDGET_GB} GB working ceiling; headroom ${HEADROOM_GB} GB; this job needs about ${NEED_GB} GB" | ts | tee -a "$LOG"
if [ "$NEED_GB" -le "$HEADROOM_GB" ]; then
  export HF_HOME="$JUDGE_CACHE"
  mkdir -p "$HF_HOME"
  echo "[cache] PERSISTENT: HF_HOME=${HF_HOME} (weights survive the job, restarts are free)" | ts | tee -a "$LOG"
else
  echo "[cache] SCRATCH: not enough home headroom, falling back to ${SCRATCH}/hf. The download repeats next run; the quota does not blow up." | ts | tee -a "$LOG"
fi

# --- 3. serve every judge in this job ------------------------------------------
i=0
PORTS=()
KEYS=()
for key in $(echo "$JUDGES" | tr ',' ' '); do
  if ! eval "$(python "${BCF_REPO}/bcf/judge_spec.py" "$key")"; then
    echo "[judge] REFUSING: unknown judge key ${key}" | ts | tee -a "$LOG" >&2
    exit 8
  fi
  PORT=$(( PORT_BASE + i ))
  # The pinned revision is the point of the pin: check it still resolves and that the live
  # sha for the repo has not moved under the pin. A mismatch is reported, not silently
  # served, and the PINNED sha is what vllm is given either way.
  LIVE_SHA="$(bcf_revision "$JUDGE_HF_ID" 2>>"$LOG")"
  if [ -z "$LIVE_SHA" ]; then
    echo "[judge] REFUSING: could not reach the HF API for ${JUDGE_HF_ID}" | ts | tee -a "$LOG" >&2
    exit 4
  fi
  if [ "$LIVE_SHA" != "$JUDGE_REVISION" ]; then
    echo "[judge] NOTE: ${JUDGE_HF_ID} main is now ${LIVE_SHA}; serving the PINNED ${JUDGE_REVISION}" | ts | tee -a "$LOG"
  fi
  VLLM_ARGS=(
    --revision "$JUDGE_REVISION" --served-model-name "$JUDGE_HF_ID"
    --port "$PORT" --host 127.0.0.1
    --tensor-parallel-size "$TP"
    --max-model-len "$JUDGE_MAX_LEN"
    --gpu-memory-utilization "$JUDGE_GPU_UTIL"
    --seed "$SEED"
    --download-dir "${HF_HOME}/download"
  )
  [ -n "$JUDGE_QUANT" ] && VLLM_ARGS+=(--quantization "$JUDGE_QUANT")
  echo "[serve] ${key}: ${JUDGE_HF_ID} @ ${JUDGE_REVISION} on port ${PORT} (${JUDGE_SERVING_LINE}, util ${JUDGE_GPU_UTIL})" | ts | tee -a "$LOG"
  vllm serve "$JUDGE_HF_ID" "${VLLM_ARGS[@]}" >> "${OUT_DIR}/server-${key}.log" 2>&1 &
  SERVER_PIDS+=($!)
  PORTS+=("$PORT")
  KEYS+=("$key")
  i=$(( i + 1 ))
done

# --- 4. wait for every server --------------------------------------------------
DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
for idx in "${!KEYS[@]}"; do
  key="${KEYS[$idx]}"; PORT="${PORTS[$idx]}"; pid="${SERVER_PIDS[$idx]}"
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve] ${key} died during startup; last 40 lines:" | ts | tee -a "$LOG"
      tail -40 "${OUT_DIR}/server-${key}.log" | ts | tee -a "$LOG"
      exit 5
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
      echo "[serve] ${key} did not answer within ${SERVER_WAIT}s" | ts | tee -a "$LOG"
      tail -40 "${OUT_DIR}/server-${key}.log" | ts | tee -a "$LOG"
      exit 6
    fi
    sleep 10
  done
  echo "[serve] ${key} up on port ${PORT}" | ts | tee -a "$LOG"
done

# --- 5. the gate ---------------------------------------------------------------
JUDGE_FLAGS=()
for idx in "${!KEYS[@]}"; do
  JUDGE_FLAGS+=(--judge "${KEYS[$idx]}=http://127.0.0.1:${PORTS[$idx]}/v1")
done
LIMIT_FLAG=()
[ "$GATE_LIMIT" -gt 0 ] && LIMIT_FLAG=(--limit "$GATE_LIMIT")

echo "[gate] starting at $(date -Is)" | ts | tee -a "$LOG"
cd "$BCF_REPO" || exit 9
python -m experiments.jury.gate \
  --items "$GATE_ITEMS" --out "$OUT_DIR" \
  --substrate "$SUBSTRATE" --cue-family "$CUE" \
  --seed "$SEED" --concurrency "$CONCURRENCY" --num-predict "$NUM_PREDICT" \
  $RESUME_FLAG "${LIMIT_FLAG[@]}" "${JUDGE_FLAGS[@]}" 2>&1 | ts | tee -a "$LOG"
GATE_STATUS=${PIPESTATUS[0]}
echo "[gate] finished at $(date -Is) with status ${GATE_STATUS}" | ts | tee -a "$LOG"

# Exit 0 when the gate RAN, whatever its verdict: a judge failing a threshold is a result,
# not a job failure, and mailing FAIL on it would train the operator to ignore the mail.
# The verdict lives in gate_report.json.
if [ "$GATE_STATUS" -gt 1 ]; then
  exit "$GATE_STATUS"
fi
exit 0
