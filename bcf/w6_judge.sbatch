#!/bin/bash
#SBATCH --job-name=bcf-w6-serve
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --exclude=xgpj0
#
# W6 external validity. Serve ONE judge and score the FaithCoT-Bench (FINE-CoT) corpus on
# Q1 files a, b and c and on Q2, then exit. Adapted from bcf/judge_serve.sbatch; the
# differences are that it runs experiments.external.score_faithcot rather than the gate,
# and that it reads a third-party item file rather than the synthetic gate corpus.
#
# EXPLORATORY. No Q1 configuration has passed the frozen gate, so every number this
# produces is labelled EXPLORATORY and no Q1 file is ever selected on these results.
#
# The FaithCoT data is used under the written evaluation-only permission recorded in
# PERMISSIONS.md. It is never committed and never redistributed. Cite arXiv:2510.04040.
#
#   Required : BCF_JUDGES (ONE judge key), BCF_GATE_ITEMS (the items jsonl)
#   Optional : BCF_OUT_ROOT BCF_PORT_BASE BCF_CONCURRENCY BCF_SEED BCF_NUM_PREDICT
#
# Exit codes: 2 items file, 3 device assertion, 4 HF API, 5 server died, 6 server
# timeout, 8 judge key, 9 repo cd, 10 wrong repo tree, 13 SERVE PORT NOT OURS (the
# port answers with a foreign model list or from a process this job did not start).
# BCF_PORT_BASE unset means a free loopback port is picked for this job.
#              BCF_SERVER_WAIT BCF_TP BCF_JUDGE_CACHE BCF_CACHE_BUDGET_GB BCF_RUNS
#              BCF_LIMIT BCF_SERVING_LINE BCF_OUT_SLUG

set -uo pipefail

export BCF_REPO="${BCF_REPO:-$HOME/bcf/repo-w6}"
source "${BCF_ENV_SH:-$HOME/bcf/repo-w6/bcf/env.sh}"
set +e
case "$BCF_REPO" in
  */repo|*/repo/) echo "[repo] REFUSING: BCF_REPO=${BCF_REPO} is another lane's tree" >&2; exit 10 ;;
  */repo-jury|*/repo-jury/) echo "[repo] REFUSING: this lane must never run out of repo-jury" >&2; exit 10 ;;
esac
if [ ! -f "${BCF_REPO}/experiments/external/score_faithcot.py" ]; then
  echo "[repo] REFUSING: ${BCF_REPO} carries no experiments/external/score_faithcot.py" >&2
  exit 10
fi

source "${BCF_EXIT_GUARD_SH:-${BCF_REPO}/bcf/exit_guard.sh}"
ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

JUDGES="${BCF_JUDGES:?BCF_JUDGES is required, one judge key}"
JUDGES="$(printf '%s' "$JUDGES" | tr '+' ',')"
case "$JUDGES" in
  *,*) echo "[judge] REFUSING: this lane serves ONE judge per job, got ${JUDGES}" >&2; exit 8 ;;
esac
ITEMS="${BCF_GATE_ITEMS:?BCF_GATE_ITEMS is required}"
[ -f "$ITEMS" ] || { echo "[items] REFUSING: no such items file ${ITEMS}" >&2; exit 2; }
OUT_ROOT="${BCF_OUT_ROOT:-$HOME/bcf/results/w6-external}"
PORT_BASE="${BCF_PORT_BASE:-}"   # unset: pick a free port (see bcf_pick_port)
CONCURRENCY="${BCF_CONCURRENCY:-8}"
SEED="${BCF_SEED:-7}"
NUM_PREDICT="${BCF_NUM_PREDICT:-320}"
SERVER_WAIT="${BCF_SERVER_WAIT:-3600}"
TP="${BCF_TP:-1}"
RUNS="${BCF_RUNS:-3}"
LIMIT="${BCF_LIMIT:-0}"
SERVING_LINE="${BCF_SERVING_LINE:-}"
SUBSTRATE="faithcot"
CUE="no-cue"

# vLLM 0.28 JIT-compiles the FlashInfer sampler on first use and that build is node
# dependent. Every judge call is temperature 0, so the sampler is off the measurement path.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Batch invariance: the same transcript must get the same vote whatever else is in the
# batch, or a concurrency change would move the numbers this lane reports.
export VLLM_BATCH_INVARIANT=1

SLUG="${BCF_OUT_SLUG:-$JUDGES}"
OUT_DIR="${OUT_ROOT}/${SLUG}/${SUBSTRATE}/${CUE}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
EXIT_FILE="${OUT_DIR}/exit_code.txt"
bcf_exit_guard_init "$EXIT_FILE"

SERVER_PIDS=()
finish() {
  local raw=$?
  local code
  for pid in "${SERVER_PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
  sleep 5
  for pid in "${SERVER_PIDS[@]:-}"; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
  code="$(bcf_exit_guard_code "$raw")"
  bcf_exit_guard_write "$code"
  echo "[done] exit_code=${code} -> ${EXIT_FILE}" | ts | tee -a "$LOG"
  exit "$code"
}
trap finish EXIT

{
  echo "=== bcf w6 external: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "judge=${JUDGES} items=${ITEMS} out=${OUT_DIR} concurrency=${CONCURRENCY} runs=${RUNS} limit=${LIMIT}"
  echo "VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT} VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
} | ts | tee -a "$LOG"

if ! bcf_assert_devices "$TP" 2>&1 | ts | tee -a "$LOG"; then exit 3; fi

# --- judge weight cache, same preflight as the jury lane -----------------------
JUDGE_CACHE="${BCF_JUDGE_CACHE:-$HOME/bcf/hf-judges}"
CACHE_BUDGET_GB="${BCF_CACHE_BUDGET_GB:-200}"
mkdir -p "$JUDGE_CACHE"
CACHE_USED_GB="$(du -sBG "$JUDGE_CACHE" 2>/dev/null | awk '{gsub("G","",$1); print $1+0}')"
HOME_USAGE_CACHE="${JUDGE_CACHE%/}/.home-usage-rest-gb"
REST_GB=""
if [ -f "$HOME_USAGE_CACHE" ]; then
  age_s=$(( $(date +%s) - $(stat -c %Y "$HOME_USAGE_CACHE" 2>/dev/null || echo 0) ))
  [ "$age_s" -lt 3600 ] && REST_GB="$(awk '{print $1+0}' "$HOME_USAGE_CACHE" 2>/dev/null)"
fi
if [ -z "$REST_GB" ]; then
  TOTAL_GB="$(du -sBG "$HOME" 2>/dev/null | awk '{gsub("G","",$1); print $1+0}')"
  REST_GB=$(( TOTAL_GB - CACHE_USED_GB )); [ "$REST_GB" -lt 0 ] && REST_GB=0
  echo "$REST_GB" > "$HOME_USAGE_CACHE"
fi
HOME_USED_GB=$(( REST_GB + CACHE_USED_GB ))
HEADROOM_GB=$(( CACHE_BUDGET_GB - HOME_USED_GB ))
NEED_GB=0
case "$JUDGES" in
  qwen3-32b) NEED_GB=66 ;; gemma-3-27b-it) NEED_GB=56 ;;
  gpt-oss-20b) NEED_GB=15 ;; llama-3.3-70b-fp8) NEED_GB=72 ;; *) NEED_GB=80 ;;
esac
HAVE_GB=0
if eval "$(python "${BCF_REPO}/bcf/judge_spec.py" "$JUDGES" 2>/dev/null)"; then
  pattern="models--$(echo "$JUDGE_HF_ID" | sed 's|/|--|')"
  HAVE_GB=$(du -sBG "$JUDGE_CACHE"/*/"${pattern}"* "$JUDGE_CACHE/${pattern}"* 2>/dev/null \
    | awk '{gsub("G","",$1); s+=$1+0} END {print s+0}')
fi
NET_GB=$(( NEED_GB - HAVE_GB )); [ "$NET_GB" -lt 0 ] && NET_GB=0
echo "[cache] home ${HOME_USED_GB} GB of ${CACHE_BUDGET_GB} GB ceiling; headroom ${HEADROOM_GB} GB; net need ${NET_GB} GB" | ts | tee -a "$LOG"
if [ "$NET_GB" -le "$HEADROOM_GB" ]; then
  export HF_HOME="$JUDGE_CACHE"; mkdir -p "$HF_HOME"
  echo "[cache] PERSISTENT HF_HOME=${HF_HOME}" | ts | tee -a "$LOG"
else
  echo "[cache] SCRATCH ${SCRATCH}/hf" | ts | tee -a "$LOG"
fi

# --- serve ---------------------------------------------------------------------
if ! eval "$(python "${BCF_REPO}/bcf/judge_spec.py" "$JUDGES")"; then
  echo "[judge] REFUSING: unknown judge key ${JUDGES}" | ts | tee -a "$LOG" >&2; exit 8
fi
PORT="$(bcf_pick_port "$PORT_BASE")"
LIVE_SHA="$(bcf_revision "$JUDGE_HF_ID" 2>>"$LOG")"
[ -z "$LIVE_SHA" ] && { echo "[judge] REFUSING: HF API unreachable for ${JUDGE_HF_ID}" | ts | tee -a "$LOG" >&2; exit 4; }
[ "$LIVE_SHA" != "$JUDGE_REVISION" ] && echo "[judge] NOTE: main is ${LIVE_SHA}; serving PINNED ${JUDGE_REVISION}" | ts | tee -a "$LOG"

VLLM_ARGS=(
  --revision "$JUDGE_REVISION" --served-model-name "$JUDGE_HF_ID"
  --port "$PORT" --host 127.0.0.1
  --tensor-parallel-size "$TP"
  --max-model-len "$JUDGE_MAX_LEN"
  --gpu-memory-utilization "$JUDGE_GPU_UTIL"
  --seed "$SEED"
  --download-dir "${HF_HOME:-$SCRATCH/hf}/download"
)
[ -n "$JUDGE_QUANT" ] && VLLM_ARGS+=(--quantization "$JUDGE_QUANT")
echo "[serve] ${JUDGES}: ${JUDGE_HF_ID} @ ${JUDGE_REVISION} on ${PORT} (${JUDGE_SERVING_LINE})" | ts | tee -a "$LOG"
vllm serve "$JUDGE_HF_ID" "${VLLM_ARGS[@]}" >> "${OUT_DIR}/server-${JUDGES}.log" 2>&1 &
SERVER_PIDS+=($!)

DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
# Ready means OUR server answers: the model list names this judge, the pid we started is
# alive, and the process holding the port is that pid or a descendant of it. A port that
# answers for anyone else is refused (13) rather than scored against.
bcf_wait_server_ready "http://127.0.0.1:${PORT}/v1" "$JUDGE_HF_ID" "${SERVER_PIDS[0]}" \
  "$DEADLINE" "${OUT_DIR}/server-${JUDGES}.log" "${OUT_DIR}/serve_ready.json" \
  | ts | tee -a "$LOG"
READY_RC=${PIPESTATUS[0]}
[ "$READY_RC" -ne 0 ] && exit "$READY_RC"
echo "[serve] up on port ${PORT} (pid ${SERVER_PIDS[0]}), ownership checked" | ts | tee -a "$LOG"

# The serving facts of this job, beside the scores: which port, which pid, and whether
# the listener was proved ours or the check was skipped and why.
BCF_SERVE_META_DIR="$OUT_DIR" python - <<'SERVEMETA' 2>&1 | ts | tee -a "$LOG"
import json, os, pathlib
out_dir = pathlib.Path(os.environ["BCF_SERVE_META_DIR"])
ready_path = out_dir / "serve_ready.json"
ready = json.loads(ready_path.read_text()) if ready_path.exists() else {}
meta = {
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "port": ready.get("port"),
    "server_pid": ready.get("server_pid"),
    "port_owner_check": ready.get("owner_check", "missing: no serve_ready file"),
    "port_owner_method": ready.get("owner_method"),
    "port_listener_pids": ready.get("listener_pids"),
    "served_model_list": ready.get("models"),
}
(out_dir / "serve_meta.json").write_text(json.dumps(meta, indent=2))
print(f"[serve] port={meta['port']} server_pid={meta['server_pid']} "
      f"port_owner_check={meta['port_owner_check']}")
SERVEMETA

# --- score ---------------------------------------------------------------------
LINE_FLAG=()
[ -n "$SERVING_LINE" ] && LINE_FLAG=(--serving-line "$SERVING_LINE")
cd "$BCF_REPO" || exit 9
python -m experiments.external.score_faithcot \
  --items "$ITEMS" --out "$OUT_DIR" \
  --judge "${JUDGES}=http://127.0.0.1:${PORT}/v1" \
  --seed "$SEED" --runs "$RUNS" --limit "$LIMIT" \
  --concurrency "$CONCURRENCY" --num-predict "$NUM_PREDICT" \
  "${LINE_FLAG[@]}" 2>&1 | ts | tee -a "$LOG"
ST=${PIPESTATUS[0]}
echo "[score] finished at $(date -Is) with status ${ST}" | ts | tee -a "$LOG"
bcf_exit_guard_complete
exit "$ST"
