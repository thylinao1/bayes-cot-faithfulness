#!/bin/bash
#SBATCH --job-name=bcf-skel-lpcheck
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#
# Serve one model and run ONLY the forced-answer-logprob unit check against it.
#
# Why this exists as its own job: the check must run under the SAME chat template as
# the run it certifies. Job 825246 ran it without `enable_thinking: false`, so for
# Qwen3 the forced letter was scored after an opening <think> token and the four
# letters together held 0.00034 of the probability mass. The alignment assertions
# still passed (the logprob really was the letter's), but the position was not the
# one the sweep generates from, so the check certified the wrong thing.
#
# Cheap: server start plus eight requests. Do not fold this into serve_and_run's
# --resume, which would overwrite that cell's measured throughput with a second leg.
#
# Exit codes: 3 device assertion, 5 server died, 6 server timeout, 12 FINISHED WITHOUT
# A COMPLETION MARKER (logprob_check_thinking_off.json never written despite an exit 0;
# bcf_install_exit_guard, bcf/env.sh), 13 SERVE PORT NOT OURS. 128+signal on a Slurm
# cancel or interrupt (143 SIGTERM, 130 SIGINT).

set -uo pipefail
source "${BCF_ENV_SH:-$HOME/bcf/repo/bcf/env.sh}"
set +e

MODEL="${BCF_MODEL:-Qwen/Qwen3-8B}"
# Unset BCF_PORT: pick a free loopback port for this job. A fixed port is what let
# two of our jobs share one server on xgph12 on 2026-09-07.
PORT="$(bcf_pick_port "${BCF_PORT:-}")"
SEED="${BCF_SEED:-7}"
OUT_DIR="${BCF_OUT_DIR:-$BCF_RESULTS/phase1-skeleton/logprob-check}"
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}"

mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# exit_code.txt is written on EVERY exit path, including a kill, and a 0 is written
# ONLY once logprob_check_thinking_off.json actually exists. See
# bcf/env.sh:bcf_install_exit_guard for the docstring (the 827052/827096 2026-09-07
# postmortem this closes) and tests/test_exit_guard.py for the revert proof.
bcf_install_exit_guard "$OUT_DIR" "logprob_check_thinking_off.json"

echo "=== bcf logprob recheck: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ===" | tee -a "$LOG"
if ! bcf_assert_devices 1 2>&1 | tee -a "$LOG"; then exit 3; fi

REVISION="$(bcf_revision "$MODEL" 2>>"$LOG")"
echo "[revision] ${MODEL} @ ${REVISION}" | tee -a "$LOG"

vllm serve "$MODEL" --revision "$REVISION" --served-model-name "$MODEL" \
  --port "$PORT" --host 127.0.0.1 --tensor-parallel-size 1 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --seed "$SEED" \
  --download-dir "${HF_HOME}/download" >> "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
bcf_guard_register_child "$SERVER_PID"

DEADLINE=$(( $(date +%s) + 2400 ))
# Ready means OUR server answers, not just that something does: the model list names
# ${MODEL}, our pid is alive, and the listener on the port is that pid or a descendant.
# Exit 5 server died, 6 timeout, 13 the port belongs to someone else.
bcf_wait_server_ready "$BASE_URL" "$MODEL" "$SERVER_PID" "$DEADLINE" \
  "${OUT_DIR}/server.log" "${OUT_DIR}/serve_ready.json" | tee -a "$LOG"
READY_RC=${PIPESTATUS[0]}
[ "$READY_RC" -ne 0 ] && exit "$READY_RC"
echo "[serve] up at ${BASE_URL} on port ${PORT} (pid ${SERVER_PID}), ownership checked" | tee -a "$LOG"

# Both templates, so the difference is on the record rather than asserted from memory.
python "${BCF_REPO}/bcf/logprob_check.py" --base-url "$BASE_URL" --model "$MODEL" \
  --seed "$SEED" --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --out "${OUT_DIR}/logprob_check_thinking_off.json" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}

echo "--- same check WITHOUT the template kwargs, for comparison ---" | tee -a "$LOG"
python "${BCF_REPO}/bcf/logprob_check.py" --base-url "$BASE_URL" --model "$MODEL" \
  --seed "$SEED" --out "${OUT_DIR}/logprob_check_default_template.json" 2>&1 | tee -a "$LOG"

exit "$STATUS"
