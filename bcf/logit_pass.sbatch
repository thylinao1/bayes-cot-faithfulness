#!/bin/bash
#SBATCH --job-name=bcf-logit
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#
# Job B of docs/OUTCOME-SCALE-NOTE.md part 4.5: ONE model, ALL of its cells, ONE server.
#
# It starts a vLLM server on the allocated card exactly the way bcf/serve_and_run.sbatch
# does (same pinned constants, same revision pin, same batch-invariant flag, same free
# port and ownership readiness, same exit guard), then runs experiments/logit_pass.py over
# every cell directory named in BCF_CELLS. One server for all of a model's cells, because
# the expensive part is loading the weights and the cheap part is the reads.
#
# Every knob arrives through --export, so this file is submitted unedited for every model.
# bcf/logit_wave.sh is the only thing that should call sbatch on it.
#
#   sbatch --job-name=bcf-logit-qwen3-8b \
#          --export=ALL,BCF_MODEL=Qwen/Qwen3-8B,BCF_CELLS=<dir>+<dir>,BCF_TP=1 \
#          ~/bcf/repo-<sha>/bcf/logit_pass.sbatch
#
# Required : BCF_MODEL BCF_CELLS
# Optional : BCF_TP(1) BCF_SEED(7) BCF_PORT(a free port) BCF_MAX_LEN(8192)
#            BCF_GPU_UTIL(0.90) BCF_QUANT() BCF_EXTRA_VLLM() BCF_REVISION()
#            BCF_TEMPLATE_KWARGS('{"enable_thinking": false}') BCF_SERVER_WAIT(2400)
#            BCF_CONCURRENCY(32) BCF_BATCH_INVARIANT(1) BCF_PREFLIGHT(1)
#            BCF_PREFLIGHT_ITEMS(30) BCF_PREFLIGHT_LEVELS(1,$BCF_CONCURRENCY)
#            BCF_LOGIT_ROOT($BCF_RESULTS/logit-pass) BCF_LOGIT_EXTRA()
#
# BCF_CELLS is a list of CELL DIRECTORIES, separated by '+' (spaces and commas are also
# accepted and folded). It must be '+' in an --export string: sbatch --export is itself a
# comma-separated list, so a comma inside a value silently truncates it and the rest of the
# list is read as names of variables to inherit, which is how job 826024 lost ten arms.
#
# WHERE THINGS ARE WRITTEN, and why the two are separate.
#   the CELL directories get the pass's products: arms_transcripts_<model>.logit.json,
#     logit_check.json, logit_pass_meta.json, logit_pass_checkpoint.json. They go there
#     because that is where the fits read a cell from.
#   $BCF_LOGIT_ROOT/<model slug> gets THIS JOB's own files: run.log, server.log,
#     run_meta.json, logprob_check.json, determinism_preflight.json, exit_code.txt and the
#     completion marker logit_pass_done.json. They are NOT written into a cell directory,
#     because a cell already has a run.log and an exit_code.txt from the run that generated
#     it and overwriting either would destroy the record of that run.
#
# THE FAMILY GATE. A5.4 condition 1 and section 9.1: the per-family forced-logprob unit
# check must pass before any of that family's logit-level cells are reported, and a family
# that fails is reported at the TEXT level only. Two checks stand here, in this order:
# bcf/logprob_check.py on the live server before any cell runs (exit 7, the same script and
# the same exit code serve_and_run.sbatch uses), and then logit_pass.py's own check over the
# cell's real reads, which also exits 7. The first cell that fails it stops the model: the
# remaining cells are not run, because their rows could not be printed either.
#
# Exit codes: 2 a named cell directory is not usable, 3 device assertion, 4 revision
# unresolvable, 5 server died, 6 server timeout, 7 THE FAMILY UNIT CHECK FAILED (this
# family is text level only), 8 revision drift against BCF_REVISION, 10 determinism
# preflight refused (ruling R1), 12 FINISHED WITHOUT A COMPLETION MARKER (the guard in
# bcf/env.sh; a job Slurm cancelled used to read 0 here), 13 SERVE PORT NOT OURS,
# 15 a cell's logit pass failed for a reason of its own (its code is in the run log and in
# that cell's logit_pass_meta.json). 128+signal on a cancel (143 SIGTERM, 130 SIGINT).

set -uo pipefail
source "${BCF_ENV_SH:-$HOME/bcf/repo/bcf/env.sh}"
set +e   # this script checks every status itself and must not die on an optional test

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

MODEL="${BCF_MODEL:?BCF_MODEL is required}"
CELLS_RAW="${BCF_CELLS:?BCF_CELLS is required (cell directories separated by '+')}"
TP="${BCF_TP:-1}"
SEED="${BCF_SEED:-7}"
PORT="$(bcf_pick_port "${BCF_PORT:-}")"
MAX_LEN="${BCF_MAX_LEN:-8192}"
GPU_UTIL="${BCF_GPU_UTIL:-0.90}"
SERVER_WAIT="${BCF_SERVER_WAIT:-2400}"
CONCURRENCY="${BCF_CONCURRENCY:-32}"
# RULING R1 (A3.6). The pinned serving mode is VLLM_BATCH_INVARIANT=1 at up to 32 requests
# in flight, and the two defaults move together for the reason serve_and_run.sbatch gives:
# job 826020 measured 30/30 identical completions and a max letter-logprob difference of
# exactly 0.0 at 1, 8, 32 and 64 in flight with the flag on, against 13/30 and 0.875 nats
# at 32 with it off. The letter logprobs THIS job reads are the quantity that measurement
# was taken on, so the flag is not optional decoration here.
BATCH_INVARIANT="${BCF_BATCH_INVARIANT:-1}"
RUN_LABEL="powered_pinned"
EXPLORATORY_REASON=""
if [ "$BATCH_INVARIANT" = "1" ]; then
  export VLLM_BATCH_INVARIANT=1
else
  unset VLLM_BATCH_INVARIANT
  RUN_LABEL="exploratory"
  EXPLORATORY_REASON="BCF_BATCH_INVARIANT=${BATCH_INVARIANT} (explicit opt-out of the A3.6 pinned serving mode)"
fi
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
PREFLIGHT="${BCF_PREFLIGHT:-1}"
PREFLIGHT_ITEMS="${BCF_PREFLIGHT_ITEMS:-30}"
PREFLIGHT_LEVELS="${BCF_PREFLIGHT_LEVELS:-1,${CONCURRENCY}}"
if [ "$PREFLIGHT" != "1" ]; then
  RUN_LABEL="exploratory"
  EXPLORATORY_REASON="${EXPLORATORY_REASON}${EXPLORATORY_REASON:+; }BCF_PREFLIGHT=${PREFLIGHT} (no determinism preflight on this server)"
fi
# A literal JSON default outside ${VAR:-word}: bash 3.2's word scanner ends the expansion
# at the first bare } inside word, which corrupted this exact string until it was moved
# out. See the same note in bcf/serve_and_run.sbatch.
DEFAULT_TEMPLATE_KWARGS='{"enable_thinking": false}'
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-$DEFAULT_TEMPLATE_KWARGS}"

MODEL_SLUG="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]')"
LOGIT_ROOT="${BCF_LOGIT_ROOT:-${BCF_RESULTS}/logit-pass}"
OUT_DIR="${LOGIT_ROOT}/${MODEL_SLUG}"
case "$OUT_DIR" in
  *"/${MODEL_SLUG}") : ;;
  *) echo "[path] REFUSING: ${OUT_DIR} does not end in /${MODEL_SLUG}" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
BASE_URL="http://127.0.0.1:${PORT}/v1"
export BCF_REQUEST_LOG="${OUT_DIR}/requests.jsonl"

# exit_code.txt on EVERY exit path, and a 0 only once logit_pass_done.json exists. The
# marker is written by the last step below, after every cell has been through the pass.
bcf_install_exit_guard "$OUT_DIR" "logit_pass_done.json"

# --- 0. the cells, resolved and checked BEFORE the weight download ---------------
# A named cell that is not there, or that holds no arms transcripts, is a wave mistake and
# costs nothing to catch here; catching it after the download costs the queue wait and the
# weights, and not catching it at all would produce a model whose "done" marker covers
# fewer cells than the wave asked for.
CELL_DIRS=()
for cell in $(printf '%s' "$CELLS_RAW" | tr '+,' '  '); do
  CELL_DIRS+=("$cell")
done
N_CELLS=${#CELL_DIRS[@]}
if [ "$N_CELLS" -eq 0 ]; then
  echo "[cells] REFUSING: BCF_CELLS resolved to no directories" >&2
  exit 2
fi
{
  echo "=== bcf logit_pass: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "model=${MODEL} tp=${TP} cells=${N_CELLS} concurrency=${CONCURRENCY}"
  echo "out_dir=${OUT_DIR} (this job's own files; the pass writes into the cells)"
  echo "serving port=${PORT} (BCF_PORT=${BCF_PORT:-<unset>, picked free for this job})"
} | ts | tee -a "$LOG"

BAD_CELLS=0
for cell in "${CELL_DIRS[@]}"; do
  if [ ! -d "$cell" ]; then
    echo "[cells] MISSING: ${cell}" | ts | tee -a "$LOG" >&2
    BAD_CELLS=$(( BAD_CELLS + 1 ))
    continue
  fi
  # The count deliberately excludes the sidecar this job writes, so a rerun of an already
  # augmented cell still sees exactly one input file.
  n_tx=$(ls "${cell}"/arms_transcripts_*.json 2>/dev/null | grep -vc '\.logit\.json$')
  if [ "${n_tx:-0}" -lt 1 ]; then
    echo "[cells] NO ARMS TRANSCRIPTS: ${cell}" | ts | tee -a "$LOG" >&2
    BAD_CELLS=$(( BAD_CELLS + 1 ))
    continue
  fi
  echo "[cells] ok: ${cell} (${n_tx} arms transcripts file(s))" | ts | tee -a "$LOG"
done
if [ "$BAD_CELLS" -gt 0 ]; then
  echo "[cells] REFUSING: ${BAD_CELLS} of ${N_CELLS} named cell(s) are not usable. Nothing" \
    | ts | tee -a "$LOG" >&2
  echo "[cells]   was served and nothing was written." | ts | tee -a "$LOG" >&2
  exit 2
fi
# The substrate of the FIRST cell, for the determinism preflight's item pool. It is read
# from that cell's own run_meta.json rather than parsed out of the path.
SUBSTRATE="$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("substrate",""))' \
  "${CELL_DIRS[0]}/run_meta.json" 2>/dev/null)"

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
# The cells being augmented were generated at a pinned revision. Reading their logprobs off
# DIFFERENT weights would produce a logit-level row for a model the text-level row is not
# about, which is the one drift that cannot be seen in the output. The wave passes the
# roster's revision in BCF_REVISION and this stops the job on a mismatch.
if [ -n "${BCF_REVISION:-}" ] && [ "${BCF_REVISION}" != "${REVISION}" ]; then
  echo "[revision] REFUSING: the wave pins ${BCF_REVISION} but the Hub resolves ${REVISION}" \
    | ts | tee -a "$LOG" >&2
  exit 8
fi
# And the same check against what each CELL recorded, which is the number that matters:
# the cell's own run_meta.json says which weights produced the text-level records.
for cell in "${CELL_DIRS[@]}"; do
  cell_rev="$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("hf_revision",""))' \
    "${cell}/run_meta.json" 2>/dev/null)"
  if [ -n "$cell_rev" ] && [ "$cell_rev" != "$REVISION" ]; then
    echo "[revision] REFUSING: ${cell} was generated at ${cell_rev} and this job serves" \
      | ts | tee -a "$LOG" >&2
    echo "[revision]   ${REVISION}. A logit-level row read off other weights is not about" \
      | ts | tee -a "$LOG" >&2
    echo "[revision]   the model the text-level row is about." | ts | tee -a "$LOG" >&2
    exit 8
  fi
done

BCF_META_TEMPLATE_KWARGS="$TEMPLATE_KWARGS" BCF_META_CELLS="$(printf '%s\n' "${CELL_DIRS[@]}")" \
python - <<PY >> "$LOG" 2>&1
import json, os, pathlib
pathlib.Path("${OUT_DIR}/run_meta.json").write_text(json.dumps({
    "job": "logit_pass", "job_id": "${SLURM_JOB_ID}",
    "spec": "docs/OUTCOME-SCALE-NOTE.md part 4.5 job B; prereg Amendment A5",
    "intervention_level": "logit", "outcome_scale": "logprob_margin",
    "model": "${MODEL}", "hf_revision": "${REVISION}",
    "backend": "openai", "backend_version": "vllm ${VLLM_VERSION}",
    "cells": os.environ["BCF_META_CELLS"].split(),
    "tensor_parallel_size": ${TP}, "seed": ${SEED}, "concurrency": ${CONCURRENCY},
    "batch_invariant": "${BATCH_INVARIANT}" == "1",
    "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
    "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    "vllm_version": "${VLLM_VERSION}",
    "run_label": "${RUN_LABEL}",
    "exploratory_reason": "${EXPLORATORY_REASON}" or None,
    "determinism_preflight": "${PREFLIGHT}" == "1",
    "determinism_preflight_levels": "${PREFLIGHT_LEVELS}",
    "determinism_preflight_n_items": ${PREFLIGHT_ITEMS},
    "determinism_preflight_substrate": "${SUBSTRATE}",
    "attention_backend": None,
    "attention_backend_lines": [],
    "port": ${PORT}, "server_pid": None, "port_owner_check": None,
    "temperature": 0.0, "logprob_mode": "prompt_logprobs",
    "chat_template_kwargs": json.loads(os.environ["BCF_META_TEMPLATE_KWARGS"]),
    "plan_commit": "${BCF_PLAN_COMMIT:-}", "bcf_repo": "${BCF_REPO}",
    "gpu": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip(),
}, indent=2))
PY

# --- 3. serve ------------------------------------------------------------------
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
bcf_guard_register_child "$SERVER_PID"

echo "[serve] waiting up to ${SERVER_WAIT}s for the server (pid ${SERVER_PID}) on port ${PORT}" \
  | ts | tee -a "$LOG"
DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
# "Something answers on the port" is NOT readiness: MIG slices of one node share the host
# loopback, and on xgph12 on 2026-09-07 one job's probe was answered by another job's
# server. This requires the /v1/models list to name OUR model, our pid to be alive, and the
# listener to be that pid or a descendant of it.
READY_REPORT="${OUT_DIR}/serve_ready.json"
bcf_wait_server_ready "$BASE_URL" "$MODEL" "$SERVER_PID" "$DEADLINE" \
  "${OUT_DIR}/server.log" "$READY_REPORT" | ts | tee -a "$LOG"
READY_RC=${PIPESTATUS[0]}
if [ "$READY_RC" -ne 0 ]; then
  exit "$READY_RC"
fi
echo "[serve] up at ${BASE_URL}" | ts | tee -a "$LOG"

# --- 3b. the attention backend the server ACTUALLY selected, into run_meta.json -----
BCF_META_OUT_DIR="$OUT_DIR" python - <<'PY' >> "$LOG" 2>&1
import json, os, pathlib, re

out_dir = pathlib.Path(os.environ["BCF_META_OUT_DIR"])
meta_path = out_dir / "run_meta.json"
log_path = out_dir / "server.log"
log = log_path.read_text(errors="replace") if log_path.exists() else ""
lines = [ln.strip() for ln in log.splitlines()
         if re.search(r"attention backend|Using .* backend|batch.?invariant", ln, re.I)]
backend = None
for ln in lines:
    m = re.search(r"([A-Z][A-Z0-9_]{3,})\s*backend", ln)
    if m:
        backend = m.group(1)
        break
meta = json.loads(meta_path.read_text())
meta["attention_backend"] = backend
meta["attention_backend_lines"] = lines[-20:]
ready_path = out_dir / "serve_ready.json"
if ready_path.exists():
    ready = json.loads(ready_path.read_text())
    meta["port"] = ready.get("port", meta.get("port"))
    meta["server_pid"] = ready.get("server_pid")
    meta["port_owner_check"] = ready.get("owner_check")
    meta["port_owner_method"] = ready.get("owner_method")
    meta["port_listener_pids"] = ready.get("listener_pids")
    meta["served_model_list"] = ready.get("models")
else:
    meta["port_owner_check"] = "missing: serve_ready.json was not written"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"[serve] attention_backend={backend} port={meta.get('port')} "
      f"server_pid={meta.get('server_pid')} port_owner_check={meta.get('port_owner_check')}")
PY

# --- 4. the per-family forced-logprob unit check, on the live server -------------
# A5.4 condition 1. The SAME script and the same --chat-template-kwargs the cells were
# generated with: under a different template the forced letter sits after different
# preceding tokens, so the check would certify a different read.
python "${BCF_REPO}/bcf/logprob_check.py" \
  --base-url "$BASE_URL" --model "$MODEL" --seed "$SEED" \
  --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --out "${OUT_DIR}/logprob_check.json" 2>&1 | ts | tee -a "$LOG"
LP_STATUS=${PIPESTATUS[0]}
if [ "$LP_STATUS" -ne 0 ]; then
  echo "[logprob] REFUSING (${LP_STATUS}): the per-family forced-answer-logprob check did" \
    | ts | tee -a "$LOG" >&2
  echo "[logprob]   not pass, so this family is reported at the TEXT level only (A5.4" \
    | ts | tee -a "$LOG" >&2
  echo "[logprob]   condition 1). No cell was touched." | ts | tee -a "$LOG" >&2
  exit 7
fi

# --- 4b. RULING R1: the determinism preflight, on THIS server, before any cell ----
if [ "$PREFLIGHT" = "1" ] && [ -n "$SUBSTRATE" ]; then
  echo "[preflight] ${PREFLIGHT_ITEMS} items at ${PREFLIGHT_LEVELS} in flight, flag ${VLLM_BATCH_INVARIANT:-<unset>}" \
    | ts | tee -a "$LOG"
  python "${BCF_REPO}/bcf/determinism_preflight.py" \
    --base-url "$BASE_URL" --model "$MODEL" --seed "$SEED" \
    --data "${BCF_REPO}/experiments/data/${SUBSTRATE}.json" \
    --n-items "$PREFLIGHT_ITEMS" --levels "$PREFLIGHT_LEVELS" \
    --chat-template-kwargs "$TEMPLATE_KWARGS" \
    --out "${OUT_DIR}/determinism_preflight.json" \
    --probe-out "${OUT_DIR}/determinism_preflight_probe.json" \
    2>&1 | ts | tee -a "$LOG"
  PF_STATUS=${PIPESTATUS[0]}
  if [ "$PF_STATUS" -ne 0 ]; then
    echo "[preflight] REFUSING (${PF_STATUS}): this server did not reproduce its own" \
      | ts | tee -a "$LOG" >&2
    echo "[preflight]   concurrency-1 letter logprobs under load. The letter logprobs are" \
      | ts | tee -a "$LOG" >&2
    echo "[preflight]   exactly what this job reads, so there is nothing to salvage here." \
      | ts | tee -a "$LOG" >&2
    exit 10
  fi
  echo "[preflight] PASS" | ts | tee -a "$LOG"
else
  echo "[preflight] SKIPPED (BCF_PREFLIGHT=${PREFLIGHT}, substrate '${SUBSTRATE}'); this run is EXPLORATORY" \
    | ts | tee -a "$LOG"
fi

# --- 5. the pass, one cell at a time against the one server ----------------------
DONE=0
FAILED_CELL=""
FAILED_STATUS=0
STATUS_LINES=()
for cell in "${CELL_DIRS[@]}"; do
  echo "[pass] ${cell}" | ts | tee -a "$LOG"
  # shellcheck disable=SC2206
  EXTRA=(${BCF_LOGIT_EXTRA:-})
  python "${BCF_REPO}/experiments/logit_pass.py" \
    --cell-dir "$cell" --model "$MODEL" --endpoint "$BASE_URL" \
    --out "$cell" --seed "$SEED" \
    --chat-template-kwargs "$TEMPLATE_KWARGS" \
    --logprob-mode prompt_logprobs \
    --concurrency "$CONCURRENCY" \
    ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | ts | tee -a "$LOG"
  CELL_STATUS=${PIPESTATUS[0]}
  STATUS_LINES+=("${CELL_STATUS} ${cell}")
  if [ "$CELL_STATUS" -eq 0 ]; then
    DONE=$(( DONE + 1 ))
    continue
  fi
  FAILED_CELL="$cell"
  FAILED_STATUS="$CELL_STATUS"
  if [ "$CELL_STATUS" -eq 7 ]; then
    echo "[pass] THE FAMILY UNIT CHECK FAILED on ${cell}. Section 9.1 and A5.4 condition 1:" \
      | ts | tee -a "$LOG" >&2
    echo "[pass]   this model family is reported at the TEXT level only, so the remaining" \
      | ts | tee -a "$LOG" >&2
    echo "[pass]   $(( N_CELLS - DONE - 1 )) cell(s) are not run: their rows could not print either." \
      | ts | tee -a "$LOG" >&2
  else
    echo "[pass] cell failed with ${CELL_STATUS}; stopping rather than running the rest" \
      | ts | tee -a "$LOG" >&2
  fi
  break
done

printf '%s\n' "${STATUS_LINES[@]}" > "${OUT_DIR}/cell_status.txt"
echo "[pass] ${DONE}/${N_CELLS} cell(s) augmented" | ts | tee -a "$LOG"

if [ -n "$FAILED_CELL" ]; then
  # No completion marker: the guard writes the code below, and a reader of exit_code.txt
  # alone learns that this model has no complete logit-level pass.
  if [ "$FAILED_STATUS" -eq 7 ]; then exit 7; fi
  exit 15
fi

# --- 6. the completion marker --------------------------------------------------
BCF_META_OUT_DIR="$OUT_DIR" BCF_META_CELLS="$(printf '%s\n' "${CELL_DIRS[@]}")" \
python - <<'PY' | ts | tee -a "$LOG"
import json, os, pathlib
out_dir = pathlib.Path(os.environ["BCF_META_OUT_DIR"])
cells = os.environ["BCF_META_CELLS"].split()
rows = []
for cell in cells:
    meta_path = pathlib.Path(cell) / "logit_pass_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows.append({
        "cell": cell,
        "sidecar": (meta.get("sidecar") or {}).get("file"),
        "n_items_scored": meta.get("n_items_scored"),
        "n_items_dropped": meta.get("n_items_dropped"),
        "drops_by_reason": meta.get("drops_by_reason"),
        "family_unit_check_passed": (meta.get("family_unit_check") or {}).get("passed"),
        "clean_arm_margin": meta.get("clean_arm_margin"),
        "hinted_arm_margin": meta.get("hinted_arm_margin"),
    })
(out_dir / "logit_pass_done.json").write_text(json.dumps({
    "n_cells": len(cells), "cells": rows,
    "note": ("every cell named in BCF_CELLS was augmented and its family unit check "
             "passed; this file is the completion marker the exit guard requires before "
             "it writes a 0"),
}, indent=2))
print(f"[done] logit_pass_done.json covers {len(cells)} cell(s)")
PY

exit 0
