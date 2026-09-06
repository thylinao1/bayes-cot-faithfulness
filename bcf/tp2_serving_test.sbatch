#!/bin/bash
#SBATCH --job-name=bcf-skel-tp2
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100-96:2
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#
# Tensor-parallel-2 serving test on two h100-96. The 70B dense subjects
# (Llama-3.3-70B, R1-Distill-Llama-70B) need ~140 GB and will not fit one 80 GB card,
# so this proves the two-card serving line BEFORE a 24-hour sweep job depends on it.
#
# NOT a100-80, though CONTRACT.md line 18 says so: every a100-80 node on this cluster
# carries exactly ONE card (sinfo -o "%n %G", 2026-09-06; 11 nodes, gpu:a100-80:1), so
# `--nodes=1 --gpus-per-node=a100-80:2` is rejected at submit time with "Requested node
# configuration is not available". h100-96 (11 nodes x 2 = 192 GB, per-user cap 8) is
# the replacement serving line for the 70B dense subjects.
#
# It answers exactly three questions and then stops:
#   1. does `--nodes=1 --gpus-per-node=h100-96:2` actually deliver two visible devices;
#   2. does vLLM start with --tensor-parallel-size 2 on them;
#   3. does the server answer one real chat request and one forced-logprob request.
#
# It serves the SKELETON model, not a 70B: the question here is the plumbing, and
# pulling 140 GB to prove a device count is a waste of the download budget. A separate
# 70B fit test runs when the Llama licence lands.
#
# Budget: 20 minutes of serving; --time is 40 to leave room for the download.

set -uo pipefail
source "${BCF_ENV_SH:-$HOME/bcf/repo/bcf/env.sh}"
set +e   # see serve_and_run.sbatch: this script checks every status itself

MODEL="${BCF_MODEL:-Qwen/Qwen3-8B}"
TP="${BCF_TP:-2}"
PORT="${BCF_PORT:-8100}"
SEED="${BCF_SEED:-7}"
OUT_DIR="${BCF_OUT_DIR:-$BCF_RESULTS/phase1-skeleton/tp2-serving-test}"
SERVE_MINUTES="${BCF_SERVE_MINUTES:-20}"
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}"

mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
EXIT_FILE="${OUT_DIR}/exit_code.txt"
BASE_URL="http://127.0.0.1:${PORT}/v1"

finish() {
  local code=$?
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  echo "$code" > "$EXIT_FILE"
  echo "[done] exit_code=${code} -> ${EXIT_FILE}" | tee -a "$LOG"
  exit "$code"
}
trap finish EXIT

{
  echo "=== bcf tp2 serving test: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "model=${MODEL} tensor_parallel=${TP} serve_minutes=${SERVE_MINUTES}"
  echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
} | tee -a "$LOG"

N_VISIBLE="$(bcf_visible_devices)"
echo "[devices] visible device count: ${N_VISIBLE} (requested tensor-parallel size ${TP})" | tee -a "$LOG"
nvidia-smi -L 2>&1 | tee -a "$LOG"
if ! bcf_assert_devices "$TP" 2>&1 | tee -a "$LOG"; then
  exit 3
fi

REVISION="$(bcf_revision "$MODEL" 2>>"$LOG")"
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
echo "[revision] ${MODEL} @ ${REVISION} (vllm ${VLLM_VERSION})" | tee -a "$LOG"

vllm serve "$MODEL" --revision "$REVISION" --served-model-name "$MODEL" \
  --port "$PORT" --host 127.0.0.1 --tensor-parallel-size "$TP" \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --seed "$SEED" \
  --download-dir "${HF_HOME}/download" >> "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

DEADLINE=$(( $(date +%s) + 1800 ))
until curl -sf "${BASE_URL}/models" > /dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] server died during startup:" | tee -a "$LOG"
    tail -40 "${OUT_DIR}/server.log" | tee -a "$LOG"
    exit 5
  fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "[serve] timed out" | tee -a "$LOG"; exit 6; }
  sleep 10
done
echo "[serve] up at ${BASE_URL} with tensor-parallel-size ${TP}" | tee -a "$LOG"

# One real request, then one forced-logprob request, both through the same client the
# sweep uses. Serving for the full window afterwards is what makes this a SERVING test
# and not just a startup test: a TP group that dies under a few minutes of idle plus
# traffic would pass a one-shot check.
python - <<PY 2>&1 | tee -a "$LOG"
import json, sys, time
sys.path.insert(0, "${BCF_REPO}/experiments")
from openai_client import OpenAIClient

client = OpenAIClient(base_url="${BASE_URL}", model="${MODEL}", seed=${SEED},
                      temperature=0.0, chat_template_kwargs=${TEMPLATE_KWARGS})
started = time.time()
text = client.generate(
    "Which is prime?\\n(A) 21\\n(B) 22\\n(C) 23\\n(D) 24\\nAnswer with the letter only.",
    num_predict=64,
)
elapsed = time.time() - started
forced = client.forced_answer_logprobs(
    "Which is prime?\\n(A) 21\\n(B) 22\\n(C) 23\\n(D) 24\\nAnswer with the letter only.",
    ["A", "B", "C", "D"],
)
report = {
    "visible_devices": ${N_VISIBLE},
    "tensor_parallel_size": ${TP},
    "model": "${MODEL}",
    "hf_revision": "${REVISION}",
    "vllm_version": "${VLLM_VERSION}",
    "server_version": client.server_version(),
    "one_request_seconds": elapsed,
    "one_request_text": text,
    "forced_logprob_method": forced.method,
    "forced_logprobs": forced.logprobs,
    "forced_tokens": forced.tokens,
    "n_letters_scored": len(forced.logprobs),
    "n_letters_requested": 4,
}
print(json.dumps(report, indent=2))
open("${OUT_DIR}/tp2_report.json", "w").write(json.dumps(report, indent=2))
PY
CHECK_STATUS=${PIPESTATUS[0]}

echo "[serve] holding the server for ${SERVE_MINUTES} minutes" | tee -a "$LOG"
HOLD_END=$(( $(date +%s) + SERVE_MINUTES * 60 ))
while [ "$(date +%s)" -lt "$HOLD_END" ]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] server DIED during the hold window" | tee -a "$LOG"
    exit 8
  fi
  curl -sf "${BASE_URL}/models" > /dev/null || { echo "[serve] /models stopped answering" | tee -a "$LOG"; exit 9; }
  sleep 60
done
echo "[serve] survived the ${SERVE_MINUTES}-minute hold with ${TP} cards" | tee -a "$LOG"

exit "$CHECK_STATUS"
