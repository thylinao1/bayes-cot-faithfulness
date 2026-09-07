#!/bin/bash
#SBATCH --job-name=bcf-gate-twopath
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#
# RULING R12(2): the two-path equivalence gate. Runs ONCE, on ONE server, before any
# reasoning_mode "off" cell is a cell of record.
#
# 30 ARC items on Qwen/Qwen3-8B at the pinned revision, under the R1 serving mode
# (VLLM_BATCH_INVARIANT=1, VLLM_USE_FLASHINFER_SAMPLER=0), each sent through
# /chat/completions and through the rendered /completions path with reasoning_mode
# DEFAULT semantics: nothing is closed and nothing is appended, so the two paths should
# render the same prompt and return the same bytes. PASS is 30/30 byte-identical
# completions, a max absolute letter-logprob difference of exactly 0.0, and a clean
# rendered-prompt round trip. Anything else and the request path change is a
# serving-mode change that needs its own preflight line.
#
#   sbatch --export=ALL,BCF_REPO=$HOME/bcf/repo-rmode ~/bcf/repo-rmode/bcf/gate_twopath.sbatch
#
# Optional: BCF_MODEL(Qwen/Qwen3-8B) BCF_REVISION(the pinned Qwen3-8B sha)
#           BCF_SUBSTRATE(arc_challenge) BCF_N_ITEMS(30) BCF_NUM_PREDICT(320)
#           BCF_CONCURRENCY(1) BCF_SEED(7) BCF_PORT(a free port) BCF_MAX_LEN(8192)
#           BCF_GPU_UTIL(0.90) BCF_SERVER_WAIT(2400) BCF_OUT_DIR
#           BCF_TEMPLATE_KWARGS('{"enable_thinking": false}')
#
# Concurrency is 1 by DEFAULT and that is deliberate: this gate asks whether the two
# request paths agree, and running it under load would fold the batching question into
# the answer. Ruling R1's own preflight is what measures batching, on its own job.
#
# Exit codes: 2 output path, 3 device assertion, 4 revision unresolvable, 5 server died,
# 6 server timeout, 8 revision drift, 10 GATE REFUSED, 12 finished without writing
# gate_twopath.json, 13 the port answers but not from this job's server.

set -uo pipefail
source "${BCF_ENV_SH:-$HOME/bcf/repo/bcf/env.sh}"
set +e

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

MODEL="${BCF_MODEL:-Qwen/Qwen3-8B}"
# The roster's pinned revision for this row (bcf/waves/a100-40-01.tsv). Pinned here as a
# default rather than left to the Hub, because a gate measured on a different weight
# revision certifies nothing about the cells that cite it.
PINNED_REV="${BCF_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
SUBSTRATE="${BCF_SUBSTRATE:-arc_challenge}"
N_ITEMS="${BCF_N_ITEMS:-30}"
NUM_PREDICT="${BCF_NUM_PREDICT:-320}"
CONCURRENCY="${BCF_CONCURRENCY:-1}"
SEED="${BCF_SEED:-7}"
PORT="$(bcf_pick_port "${BCF_PORT:-}")"
MAX_LEN="${BCF_MAX_LEN:-8192}"
GPU_UTIL="${BCF_GPU_UTIL:-0.90}"
SERVER_WAIT="${BCF_SERVER_WAIT:-2400}"
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}"
OUT_DIR="${BCF_OUT_DIR:-${BCF_RESULTS}/gate-twopath}"

mkdir -p "$OUT_DIR" || exit 2
LOG="${OUT_DIR}/run.log"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# Same guard as every cell job: exit_code.txt on EVERY exit path, and a 0 only once the
# marker this job exists to produce is actually on disk. A gate that was cancelled and
# read as PASS would be worse than no gate at all.
bcf_install_exit_guard "$OUT_DIR" "gate_twopath.json"

# RULING R1's serving mode. The gate has to run under the mode the cells run under, or
# it certifies the path change on a kernel path no cell uses.
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

{
  echo "=== bcf gate_twopath (ruling R12(2)): job ${SLURM_JOB_ID:-none} on $(hostname) ==="
  echo "model=${MODEL} rev=${PINNED_REV} substrate=${SUBSTRATE} n=${N_ITEMS}"
  echo "num_predict=${NUM_PREDICT} concurrency=${CONCURRENCY} seed=${SEED} port=${PORT}"
  echo "out_dir=${OUT_DIR} repo=${BCF_REPO}"
} | ts | tee -a "$LOG"

if ! bcf_assert_devices 1 2>&1 | ts | tee -a "$LOG"; then
  exit 3
fi

REVISION="$(bcf_revision "$MODEL" 2>>"$LOG")"
if [ -z "$REVISION" ]; then
  echo "[revision] REFUSING: could not resolve an HF revision sha for ${MODEL}" | ts | tee -a "$LOG" >&2
  exit 4
fi
if [ "$REVISION" != "$PINNED_REV" ]; then
  echo "[revision] REFUSING: the gate pins ${PINNED_REV} but the Hub resolves ${REVISION}" \
    | ts | tee -a "$LOG" >&2
  exit 8
fi
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
echo "[revision] ${MODEL} @ ${REVISION} (vllm ${VLLM_VERSION})" | ts | tee -a "$LOG"

vllm serve "$MODEL" --revision "$REVISION" --served-model-name "$MODEL" \
  --port "$PORT" --host 127.0.0.1 --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" --seed "$SEED" \
  --download-dir "${HF_HOME}/download" >> "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
bcf_guard_register_child "$SERVER_PID"
echo "[serve] waiting up to ${SERVER_WAIT}s for pid ${SERVER_PID} on port ${PORT}" | ts | tee -a "$LOG"

DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
bcf_wait_server_ready "$BASE_URL" "$MODEL" "$SERVER_PID" "$DEADLINE" \
  "${OUT_DIR}/server.log" "${OUT_DIR}/serve_ready.json" | ts | tee -a "$LOG"
READY_RC=${PIPESTATUS[0]}
if [ "$READY_RC" -ne 0 ]; then
  exit "$READY_RC"
fi
echo "[serve] up at ${BASE_URL}" | ts | tee -a "$LOG"

python "${BCF_REPO}/experiments/gate_twopath.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --data "${BCF_REPO}/experiments/data/${SUBSTRATE}.json" \
  --n-items "$N_ITEMS" --num-predict "$NUM_PREDICT" \
  --concurrency "$CONCURRENCY" --seed "$SEED" \
  --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --out "${OUT_DIR}/gate_twopath.json" 2>&1 | ts | tee -a "$LOG"
GATE_STATUS=${PIPESTATUS[0]}

BCF_META_MODEL="$MODEL" BCF_META_REV="$REVISION" BCF_META_OUT="$OUT_DIR" \
BCF_META_VLLM="$VLLM_VERSION" BCF_META_STATUS="$GATE_STATUS" \
python - >> "$LOG" 2>&1 <<'PYEOF'
import json, os, pathlib
out = pathlib.Path(os.environ["BCF_META_OUT"])
(out / "run_meta.json").write_text(json.dumps({
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "gate": "R12(2) two-path equivalence",
    "model": os.environ["BCF_META_MODEL"],
    "hf_revision": os.environ["BCF_META_REV"],
    "backend_version": "vllm " + os.environ["BCF_META_VLLM"],
    "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
    "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    "hostname": os.uname().nodename,
    "gate_exit_status": int(os.environ["BCF_META_STATUS"]),
    "note": "reasoning_mode default semantics on both paths; nothing closed, nothing "
            "appended. A PASS is what lets a reasoning_mode 'off' cell be a cell of "
            "record (ruling R12(2)).",
}, indent=2))
PYEOF

echo "[gate] exit ${GATE_STATUS}" | ts | tee -a "$LOG"
exit "$GATE_STATUS"
