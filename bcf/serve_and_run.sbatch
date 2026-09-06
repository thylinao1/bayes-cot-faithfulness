#!/bin/bash
#SBATCH --job-name=bcf-run
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80
#SBATCH --exclude=xgpj0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#
# One model x substrate x cue family: start a vLLM server on the allocated card(s),
# then run the arms against it through the OpenAI-compatible backend.
#
# Every knob arrives through --export, so this file is submitted unedited for every
# cell. wave.sh is the only thing that should call sbatch on it.
#
#   sbatch --job-name=bcf-qwen3-8b-arc \
#          --export=ALL,BCF_MODEL=Qwen/Qwen3-8B,BCF_SUBSTRATE=arc_challenge,\
# BCF_CUE=stated-hint,BCF_N_ITEMS=500,BCF_TP=1 \
#          ~/bcf/repo/bcf/serve_and_run.sbatch
#
# Required : BCF_MODEL BCF_SUBSTRATE BCF_CUE
# Optional : BCF_TP(1) BCF_N_ITEMS(500) BCF_NUM_PREDICT(320) BCF_CURVE_CAP(500)
#            BCF_ARMS(all) BCF_SEED(7) BCF_OUT_ROOT($BCF_RESULTS) BCF_PORT(8000)
#            BCF_MAX_LEN(8192) BCF_GPU_UTIL(0.90) BCF_QUANT() BCF_EXTRA_VLLM()
#            BCF_TEMPLATE_KWARGS('{"enable_thinking": false}') BCF_SERVER_WAIT(2400)

set -uo pipefail
source "${BCF_ENV_SH:-$HOME/bcf/repo/bcf/env.sh}"
set +e   # explicit, regardless of what env.sh leaves behind: this script checks every
         # status itself and must not die on an optional flag test returning 1

# Prefix every line this script prints with an ISO-8601 timestamp. bcf/throughput.py
# reads those timestamps back to find the arm boundaries, so the per-arm rate is
# measured from the run instead of estimated.
ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

MODEL="${BCF_MODEL:?BCF_MODEL is required}"
SUBSTRATE="${BCF_SUBSTRATE:?BCF_SUBSTRATE is required}"
CUE="${BCF_CUE:?BCF_CUE is required}"
TP="${BCF_TP:-1}"
N_ITEMS="${BCF_N_ITEMS:-500}"
NUM_PREDICT="${BCF_NUM_PREDICT:-320}"
CURVE_CAP="${BCF_CURVE_CAP:-500}"
SEED="${BCF_SEED:-7}"
PORT="${BCF_PORT:-8000}"
MAX_LEN="${BCF_MAX_LEN:-8192}"
GPU_UTIL="${BCF_GPU_UTIL:-0.90}"
SERVER_WAIT="${BCF_SERVER_WAIT:-2400}"
CONCURRENCY="${BCF_CONCURRENCY:-1}"
# BCF_BATCH_INVARIANT=1 sets VLLM_BATCH_INVARIANT on the SERVER. vLLM 0.28.0 honors it
# in the FlashAttention backend (num_splits forced to 1) with Triton batch-invariant
# matmul and softmax kernels. It is a run parameter, not a default: whether it is worth
# its throughput cost is the operator's ruling, and this records which way a run went.
BATCH_INVARIANT="${BCF_BATCH_INVARIANT:-}"
if [ "$BATCH_INVARIANT" = "1" ]; then
  export VLLM_BATCH_INVARIANT=1
else
  unset VLLM_BATCH_INVARIANT
fi
# Element 9.2 and element 8.3 draw parameters. The defaults ARE the frozen values; they
# are passed explicitly so a run's own command line carries them.
SAMPLING_K="${BCF_SAMPLING_K:-32}"
SAMPLING_TEMPERATURE="${BCF_SAMPLING_TEMPERATURE:-0.7}"
CURVE_REPEATS="${BCF_CURVE_REPEATS:-3}"
REPEAT_TEMPERATURES="${BCF_REPEAT_TEMPERATURES:-0.0,0.7}"
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}"
OUT_ROOT="${BCF_OUT_ROOT:-$BCF_RESULTS}"

MODEL_SLUG="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="${OUT_ROOT}/${MODEL_SLUG}/${SUBSTRATE}/${CUE}"

# CONTRACT: the results path must carry substrate and cue family. Assert it here,
# before a single generation, rather than discovering a mislabelled cell at analysis.
case "$OUT_DIR" in
  *"/${SUBSTRATE}/${CUE}") : ;;
  *) echo "[path] REFUSING: ${OUT_DIR} does not end in /${SUBSTRATE}/${CUE}" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
EXIT_FILE="${OUT_DIR}/exit_code.txt"
BASE_URL="http://127.0.0.1:${PORT}/v1"
export BCF_REQUEST_LOG="${OUT_DIR}/requests.jsonl"

# exit_code.txt is written on EVERY exit path, including a kill: a cell with no exit
# file is an unfinished cell, and that has to be readable without the Slurm log.
finish() {
  local code=$?
  # Enumerate and kill only THIS job's server. Never a wildcard kill: other campaigns
  # share this account (NUS-COMPUTE.md 1.9 etiquette).
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  echo "$code" > "$EXIT_FILE"
  echo "[done] exit_code=${code} -> ${EXIT_FILE}" | ts | tee -a "$LOG"
  exit "$code"
}
trap finish EXIT

{
  echo "=== bcf serve_and_run: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "model=${MODEL} substrate=${SUBSTRATE} cue=${CUE} tp=${TP} n_items=${N_ITEMS}"
  echo "out_dir=${OUT_DIR}"
} | ts | tee -a "$LOG"

# --- 1. device assertion, BEFORE the download ----------------------------------
if ! bcf_assert_devices "$TP" 2>&1 | ts | tee -a "$LOG"; then
  exit 3
fi

# --- 2. pin the revision -------------------------------------------------------
REVISION="$(bcf_revision "$MODEL" 2>>"$LOG")"
if [ -z "$REVISION" ]; then
  echo "[revision] REFUSING: could not resolve an HF revision sha for ${MODEL}" | ts | tee -a "$LOG" >&2
  exit 4
fi
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
echo "[revision] ${MODEL} @ ${REVISION}  (vllm ${VLLM_VERSION})" | ts | tee -a "$LOG"
# Element 10 pins a revision per roster row and says the runner compares against it.
# When the wave manifest carries BCF_REVISION, a drift between the pinned sha and what
# the Hub resolves today stops the cell here rather than producing a cell whose
# hf_revision field quietly disagrees with the pre-registration.
if [ -n "${BCF_REVISION:-}" ] && [ "${BCF_REVISION}" != "${REVISION}" ]; then
  echo "[revision] REFUSING: the wave pins ${BCF_REVISION} but the Hub resolves ${REVISION}" \
    | ts | tee -a "$LOG" >&2
  exit 8
fi
BCF_META_TEMPLATE_KWARGS="$TEMPLATE_KWARGS" python - <<PY >> "$LOG" 2>&1
import json, os, pathlib
pathlib.Path("${OUT_DIR}/run_meta.json").write_text(json.dumps({
    "job_id": "${SLURM_JOB_ID}", "model": "${MODEL}", "hf_revision": "${REVISION}",
    "backend": "openai", "backend_version": "vllm ${VLLM_VERSION}",
    "substrate": "${SUBSTRATE}", "cue_family": "${CUE}",
    "tensor_parallel_size": ${TP}, "seed": ${SEED}, "concurrency": ${CONCURRENCY},
    # Which way the batch-determinism switch went, on the record. Job 825548 measured
    # 11/30 identical completions at 8 in flight with the flag off, so a run's
    # concurrency is meaningless without this beside it.
    "batch_invariant": "${BATCH_INVARIANT}" == "1",
    "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
    "sampling_k": ${SAMPLING_K}, "sampling_temperature": ${SAMPLING_TEMPERATURE},
    "curve_repeats": ${CURVE_REPEATS},
    "repeat_temperatures": [float(x) for x in "${REPEAT_TEMPERATURES}".split(",")],
    "plan_commit": "${BCF_PLAN_COMMIT:-}", "bcf_repo": "${BCF_REPO}",
    "temperature": 0.0, "num_predict": ${NUM_PREDICT}, "curve_cap": ${CURVE_CAP},
    # CONTRACT record schema. Round-1 verification check 4 failed because
    # intervention_level was absent from every summary; it is written here AND by the
    # runner into arms_summary, from the same constant, so the two cannot disagree.
    "intervention_level": "text", "outcome_scale": "binary_follow",
    # JSON, parsed as JSON: interpolating it into a Python literal makes `false` a
    # NameError, which is exactly how job 825241 died.
    "chat_template_kwargs": json.loads(os.environ["BCF_META_TEMPLATE_KWARGS"]),
    "gpu": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip(),
}, indent=2))
PY

# --- 3. serve ------------------------------------------------------------------
# Only long-stable flags here. vLLM renames its logging flags between releases, and a
# rejected flag kills the server AFTER the weights are downloaded; anything version
# specific belongs in BCF_EXTRA_VLLM where it is a run parameter, not a silent default.
VLLM_ARGS=(
  --revision "$REVISION" --served-model-name "$MODEL"
  --port "$PORT" --host 127.0.0.1
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --seed "$SEED"
  --download-dir "${HF_HOME}/download"
)
if [ -n "${BCF_QUANT:-}" ]; then VLLM_ARGS+=(--quantization "$BCF_QUANT"); fi
# shellcheck disable=SC2206
if [ -n "${BCF_EXTRA_VLLM:-}" ]; then VLLM_ARGS+=(${BCF_EXTRA_VLLM}); fi

echo "[serve] VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-<unset>}" | ts | tee -a "$LOG"
echo "[serve] vllm serve ${MODEL} ${VLLM_ARGS[*]}" | ts | tee -a "$LOG"
vllm serve "$MODEL" "${VLLM_ARGS[@]}" >> "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "[serve] waiting up to ${SERVER_WAIT}s for the server (pid ${SERVER_PID})" | ts | tee -a "$LOG"
DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
until curl -sf "${BASE_URL}/models" > /dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] server died during startup; last 40 lines:" | ts | tee -a "$LOG"
    tail -40 "${OUT_DIR}/server.log" | ts | tee -a "$LOG"
    exit 5
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[serve] server did not answer within ${SERVER_WAIT}s" | ts | tee -a "$LOG"
    tail -40 "${OUT_DIR}/server.log" | ts | tee -a "$LOG"
    exit 6
  fi
  sleep 10
done
grep -iE "attention backend|Using .* backend|batch.?invariant" "${OUT_DIR}/server.log" \
  | tail -20 | sed 's/^/[serve] /' | ts | tee -a "$LOG" || true
echo "[serve] up at ${BASE_URL} after $(( SERVER_WAIT - (DEADLINE - $(date +%s)) ))s" | ts | tee -a "$LOG"

# --- 4. forced-logprob unit check on the live server ---------------------------
# The SAME --chat-template-kwargs the arms use. Under a different template the forced
# letter sits after different preceding tokens (for Qwen3 with thinking on, after an
# opening <think>), so the logprob read is of a different position and certifies
# nothing about the run. Job 825246 measured 0.00034 total letter mass this way.
python "${BCF_REPO}/bcf/logprob_check.py" \
  --base-url "$BASE_URL" --model "$MODEL" --seed "$SEED" \
  --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --out "${OUT_DIR}/logprob_check.json" 2>&1 | ts | tee -a "$LOG"
LP_STATUS=${PIPESTATUS[0]}
if [ "$LP_STATUS" -ne 0 ]; then
  echo "[logprob] REFUSING: the forced-answer-logprob check failed (${LP_STATUS})" | ts | tee -a "$LOG" >&2
  exit 7
fi

# --- 5. the arms ---------------------------------------------------------------
ARM_FLAGS=()
# PLUS signs, spaces, or commas. sbatch --export is itself a COMMA-separated list, so a
# comma inside a value silently truncates it: BCF_ARMS=replay,placebo,direct arrives as
# BCF_ARMS=replay and the rest are read as names of variables to inherit, which do not
# exist and are dropped without an error. Job 826024 was submitted that way and would
# have run the replay arm alone; it was cancelled by id before it generated anything.
# A value carrying a SPACE breaks the export string a different way, so '+' is the
# separator a multi-arm list has to travel under. All three are accepted here.
for arm in $(echo "${BCF_ARMS:-replay placebo direct twostep filler curves transplant anchor specificity}" | tr '+,' '  '); do
  ARM_FLAGS+=(--arm "$arm")
done
echo "[arms] enabled: ${ARM_FLAGS[*]}" | ts | tee -a "$LOG"
# A truncated arm list is indistinguishable from a deliberate short one, so a caller
# that means N arms says so and this refuses on a mismatch. That is exactly the failure
# job 826024 had: eleven arms asked for, one arrived.
N_ARMS_SEEN=$(( ${#ARM_FLAGS[@]} / 2 ))
if [ -n "${BCF_N_ARMS:-}" ] && [ "$N_ARMS_SEEN" -ne "${BCF_N_ARMS}" ]; then
  echo "[arms] REFUSING: BCF_N_ARMS=${BCF_N_ARMS} but ${N_ARMS_SEEN} arm(s) arrived: ${ARM_FLAGS[*]}" \
    | ts | tee -a "$LOG" >&2
  echo "[arms]   sbatch --export is comma separated; use '+' between arm names." | ts | tee -a "$LOG" >&2
  exit 9
fi
TAXONOMY_FLAGS=()
if [ "$CUE" != "stated-hint" ]; then TAXONOMY_FLAGS=(--taxonomy "$CUE"); fi

echo "[arms] starting at $(date -Is)" | ts | tee -a "$LOG"
python "${BCF_REPO}/experiments/08_additive_arms.py" \
  --backend openai --base-url "$BASE_URL" \
  --model "$MODEL" --seed "$SEED" \
  --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --data "${BCF_REPO}/experiments/data/${SUBSTRATE}.json" \
  --specificity-holdout "${BCF_REPO}/experiments/data/specificity_holdout.json" \
  --n-items "$N_ITEMS" --num-predict "$NUM_PREDICT" --curve-cap "$CURVE_CAP" \
  --concurrency "$CONCURRENCY" \
  --sampling-k "$SAMPLING_K" --sampling-temperature "$SAMPLING_TEMPERATURE" \
  --curve-repeats "$CURVE_REPEATS" --repeat-temperatures "$REPEAT_TEMPERATURES" \
  --out "$OUT_DIR" --timeout 600 --resume \
  "${ARM_FLAGS[@]}" "${TAXONOMY_FLAGS[@]}" 2>&1 | ts | tee -a "$LOG"
ARMS_STATUS=${PIPESTATUS[0]}
echo "[arms] finished at $(date -Is) with status ${ARMS_STATUS}" | ts | tee -a "$LOG"

# --- 6. CONTRACT filenames -----------------------------------------------------
python "${BCF_REPO}/bcf/contract_layout.py" --out-dir "$OUT_DIR" 2>&1 | ts | tee -a "$LOG"

# --- 7. throughput, measured from this run's own log ---------------------------
python "${BCF_REPO}/bcf/throughput.py" --run-log "$LOG" --out-dir "$OUT_DIR" \
  2>&1 | ts | tee -a "$LOG" || true

exit "$ARMS_STATUS"
