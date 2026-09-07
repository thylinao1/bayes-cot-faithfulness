#!/bin/bash
#SBATCH --job-name=bcf-audit-serving
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Element 9.4 serving test (Track V audit lane). One model per job: start the pinned
# vLLM server on the allocated slice, then run experiments/audit/serving_test.py twice,
# configuration A then configuration B. DATA ONLY; nothing here writes into the sweep's
# result tree and nothing here is a cell of record.
#
#   sbatch --job-name=bcf-audit-st-<slug> \
#     --export=ALL,BCF_MODEL=...,BCF_REV=...,BCF_SLUG=... \
#     ~/bcf/repo-audit1/bcf/audit1_serving_test.sbatch
#
# Exit codes: 2 output path, 5 server died, 6 server timeout, 1 a configuration failed.

set -uo pipefail
source "${BCF_ENV_SH:-$HOME/bcf/repo-audit1/bcf/env.sh}"
set +e

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

MODEL="${BCF_MODEL:?BCF_MODEL is required}"
REV="${BCF_REV:?BCF_REV is required}"
SLUG="${BCF_SLUG:?BCF_SLUG is required}"
PREFILL="${BCF_PREFILL:-$'<think>\n\n</think>\n\n'}"
SUBSTRATE="${BCF_SUBSTRATE:-arc_challenge}"
N_ITEMS="${BCF_N_ITEMS:-30}"
NP_A="${BCF_NP_A:-320}"
NP_B="${BCF_NP_B:-4096}"
CONC="${BCF_CONCURRENCY:-32}"
SEED="${BCF_SEED:-7}"
PORT="${BCF_PORT:-8000}"
MAX_LEN="${BCF_MAX_LEN:-8192}"
GPU_UTIL="${BCF_GPU_UTIL:-0.90}"
SERVER_WAIT="${BCF_SERVER_WAIT:-2400}"
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}"
# env.sh already exports BCF_REPO (pointing at the sweep tree), so this lane uses its own
# variable name; a ${BCF_REPO:-...} default would never fire and the job would run the
# wrong tree, which is exactly how job 826904 failed.
REPO="${BCF_AUDIT_REPO:-$HOME/bcf/repo-audit1}"
OUT_DIR="${BCF_OUT_DIR:-$HOME/bcf/results/audit1-${SLUG}}"

mkdir -p "$OUT_DIR" || exit 2
LOG="${OUT_DIR}/run.log"
: > "$LOG"

echo "=== bcf audit1 serving test: job ${SLURM_JOB_ID:-none} on $(hostname) ===" | ts | tee -a "$LOG"
echo "model=${MODEL} rev=${REV} slug=${SLUG} n=${N_ITEMS} npA=${NP_A} npB=${NP_B} conc=${CONC}" | ts | tee -a "$LOG"
nvidia-smi -L 2>&1 | ts | tee -a "$LOG"

export VLLM_BATCH_INVARIANT=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME="${HF_HOME_OVERRIDE:-/tmp/${SLURM_JOB_ID:-audit1}/hf}"
mkdir -p "$HF_HOME"
echo "[serve] VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=${HF_HOME}" | ts | tee -a "$LOG"

vllm serve "$MODEL" --revision "$REV" --served-model-name "$MODEL" \
  --port "$PORT" --host 127.0.0.1 --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" --seed "$SEED" \
  --download-dir "${HF_HOME}/download" > "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "[serve] waiting up to ${SERVER_WAIT}s (pid ${SERVER_PID})" | ts | tee -a "$LOG"

BASE_URL="http://127.0.0.1:${PORT}/v1"
DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
while true; do
  curl -sf "${BASE_URL}/models" > /dev/null 2>&1 && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] server died during startup; last 30 lines:" | ts | tee -a "$LOG" >&2
    tail -30 "${OUT_DIR}/server.log" | ts | tee -a "$LOG" >&2
    echo 5 > "${OUT_DIR}/exit_code.txt"
    exit 5
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[serve] TIMEOUT after ${SERVER_WAIT}s" | ts | tee -a "$LOG" >&2
    kill "$SERVER_PID" 2>/dev/null
    echo 6 > "${OUT_DIR}/exit_code.txt"
    exit 6
  fi
  sleep 5
done
echo "[serve] up" | ts | tee -a "$LOG"

RC=0
echo "[config A] thinking suppressed by prefill, num_predict ${NP_A}" | ts | tee -a "$LOG"
python "${REPO}/experiments/audit/serving_test.py" \
  --base-url "$BASE_URL" --model "$MODEL" --config A \
  --data "${REPO}/experiments/data/${SUBSTRATE}.json" \
  --n-items "$N_ITEMS" --num-predict "$NP_A" --concurrency "$CONC" --seed "$SEED" \
  --prefill "$PREFILL" --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --out "${OUT_DIR}/config_A.json" 2>&1 | ts | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then RC=1; fi

echo "[config B] thinking allowed, num_predict ${NP_B}, answer after the think block" | ts | tee -a "$LOG"
python "${REPO}/experiments/audit/serving_test.py" \
  --base-url "$BASE_URL" --model "$MODEL" --config B \
  --data "${REPO}/experiments/data/${SUBSTRATE}.json" \
  --n-items "$N_ITEMS" --num-predict "$NP_B" --concurrency "$CONC" --seed "$SEED" \
  --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --out "${OUT_DIR}/config_B.json" 2>&1 | ts | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then RC=1; fi

BCF_META_MODEL="$MODEL" BCF_META_REV="$REV" BCF_META_SLUG="$SLUG" \
BCF_META_SUB="$SUBSTRATE" BCF_META_N="$N_ITEMS" BCF_META_NPA="$NP_A" \
BCF_META_NPB="$NP_B" BCF_META_CONC="$CONC" BCF_META_SEED="$SEED" \
python - > "${OUT_DIR}/run_meta.json" <<'PYEOF'
import json, os
print(json.dumps({
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "model": os.environ["BCF_META_MODEL"],
    "hf_revision": os.environ["BCF_META_REV"],
    "slug": os.environ["BCF_META_SLUG"],
    "substrate": os.environ["BCF_META_SUB"],
    "n_items": int(os.environ["BCF_META_N"]),
    "num_predict_A": int(os.environ["BCF_META_NPA"]),
    "num_predict_B": int(os.environ["BCF_META_NPB"]),
    "concurrency": int(os.environ["BCF_META_CONC"]),
    "seed": int(os.environ["BCF_META_SEED"]),
    "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
    "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    "hostname": os.uname().nodename,
    "run_label": "audit_serving_test_exploratory",
    "note": "element 9.4 serving test; data only, no ruling, not a cell of record",
}, indent=1))
PYEOF

kill "$SERVER_PID" 2>/dev/null
echo "$RC" > "${OUT_DIR}/exit_code.txt"
echo "[done] exit_code=${RC}" | ts | tee -a "$LOG"
exit "$RC"
