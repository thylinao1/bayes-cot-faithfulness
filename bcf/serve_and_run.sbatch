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
#            BCF_ARMS(all) BCF_SEED(7) BCF_OUT_ROOT($BCF_RESULTS) BCF_PORT(a free port)
#            BCF_MAX_LEN(8192) BCF_GPU_UTIL(0.90) BCF_QUANT() BCF_EXTRA_VLLM()
#            BCF_TEMPLATE_KWARGS('{"enable_thinking": false}') BCF_SERVER_WAIT(2400)
#            BCF_CONCURRENCY(32) BCF_BATCH_INVARIANT(1) BCF_PREFLIGHT(1)
#            BCF_PREFLIGHT_ITEMS(30) BCF_PREFLIGHT_LEVELS(1,$BCF_CONCURRENCY)
#            BCF_CHAIN_REPEATS(3) BCF_CHAIN_REPEAT_TEMPERATURE(0.7)
#            BCF_CHAIN_REPEAT_SEED(20260907)
#            BCF_ITEM_LIST() BCF_ENRICH_REPORT(0) BCF_OUT_SUBROOT()
#            BCF_REASONING_MODE(default) BCF_LOCAL_CHECKPOINT()
#
# BCF_LOCAL_CHECKPOINT=<dir>, element 11's ladder. A merged LoRA checkpoint has no Hub
# id and no Hub revision, so with this set the script serves that directory, takes
# REVISION from the checkpoint's own manifest hash (the `ladder-<sha16>` string
# serve_manifest.checkpoint_revision builds, which is what the ladder wave manifest
# pins), drops --revision from the serve line, and records local_checkpoint=true in
# run_meta.json. UNSET, which is every one of the 216 sweep cells, nothing below
# changes: tests/test_serve_local_checkpoint.py renders the default serve argv and the
# default run_meta.json and diffs them against fixtures taken from this file BEFORE the
# branch existed. --served-model-name stays $BCF_MODEL either way, because the runner
# and bcf_wait_server_ready both address the server by that name.
#
# RULING R12 (2026-09-07), BCF_REASONING_MODE, for a roster row whose chat template opens
# a reasoning block and documents no switch (OLMo-3-7B-Think, R1-Distill-Llama-8B,
# Phi-4-reasoning, R1-0528-Qwen3-8B):
#   default  today's behaviour, byte for byte. Every other row uses this and nothing
#            about those cells changes.
#   off      the CELL OF RECORD. The prompt is rendered through the model's own template
#            (/tokenize plus /detokenize) with the reasoning block CLOSED, generation
#            moves to /completions, and every element 15 constant stays where it is
#            (num_predict 320 full, 24 forced). R12(2): the two-path equivalence gate
#            (bcf/gate_twopath.sbatch) must have PASSED before an off cell is a cell of
#            record, because the request path is part of the serving mode.
#   on       the EXPLORATORY additive arm. Ordinary chat path, num_predict 4096 for FULL
#            generations on this row only (the forced budget stays 24), the answer read
#            after the closing think tag. Reported beside the off cell per model and
#            NEVER pooled with it. Raise BCF_MAX_LEN (16384) with it.
#
# RULING R3(ii), the two enrichment shapes this one script serves:
#   the enrichment PASS   BCF_ARMS=sampling, BCF_N_ITEMS=<the whole pool>,
#                         BCF_ENRICH_REPORT=1. One k = 32 draw per item on the clean
#                         prompt; the report step writes uncertain_items.json.
#   the enrichment CELL   BCF_ITEM_LIST=<that uncertain_items.json>, the hinted arms.
#                         The runner runs on exactly those items, every record carries
#                         enrichment=true, and the summary counts the regular and the
#                         enrichment populations apart.
#
# Exit codes: 2 results path, 3 device assertion, 4 revision unresolvable (a Hub sha
#    that will not resolve, or a BCF_LOCAL_CHECKPOINT directory with no checkpoint
#    manifest to hash), 5 server died,
# 6 server timeout, 7 forced-logprob unit check, 8 revision drift, 9 arm-count mismatch,
# 10 DETERMINISM PREFLIGHT REFUSED (ruling R1), 11 item list named but not readable,
# 12 FINISHED WITHOUT A COMPLETION MARKER: the shell reached its own exit with status 0
#    but arms_summary.json was never written (bcf_install_exit_guard, bcf/env.sh; the
#    827052/827096 postmortem, 2026-09-07 -- a job Slurm CANCELLED with SIGTERM used to
#    read exit_code.txt=0 here), 13 SERVE PORT NOT OURS: the port answers, but with a
#    foreign model list or from a process this job did not start (bcf/serve_ready.py,
#    which returns 11 for that case; the job maps it to 13 because 11 already means the
#    item list here), 14 BCF_REASONING_MODE is not one of default/off/on (ruling R12;
#    refused here rather than in the runner so it costs no queue wait). 128+signal on a
#    Slurm cancel or interrupt (143 SIGTERM, 130 SIGINT), from the same guard.

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
# A FIXED default port put two of our jobs on one server (xgph12, 2026-09-07: the
# second job's probe was answered by the first job's server). Unset BCF_PORT now
# means "pick a free loopback port for this job"; a set BCF_PORT is still honored,
# because the operator pinning a port is a deliberate act.
PORT="$(bcf_pick_port "${BCF_PORT:-}")"
MAX_LEN="${BCF_MAX_LEN:-8192}"
GPU_UTIL="${BCF_GPU_UTIL:-0.90}"
SERVER_WAIT="${BCF_SERVER_WAIT:-2400}"
# RULING R1 (2026-09-07, Amendment A3.9). The pinned serving mode of a powered cell is
# VLLM_BATCH_INVARIANT=1 at up to 32 requests in flight. 32 is the default here BECAUSE
# the flag is: job 826020 measured 30/30 identical completions and a max letter-logprob
# difference of exactly 0.0 at 1, 8, 32 and 64 in flight with the flag on, against 13/30
# and 0.875 nats at 32 with it off (job 825548). The flag is what makes concurrency safe,
# so the two defaults move together and neither is set without the other on the record.
CONCURRENCY="${BCF_CONCURRENCY:-32}"
# BCF_BATCH_INVARIANT=1 sets VLLM_BATCH_INVARIANT on the SERVER. vLLM 0.28.0 honors it
# in the FlashAttention backend (num_splits forced to 1) with Triton batch-invariant
# matmul and softmax kernels. Default ON. The ONLY way past it is an explicit
# BCF_BATCH_INVARIANT=0, which stamps the whole run EXPLORATORY in run_meta.json: a
# flag-off run is a different kernel path, matches only 10 of 30 flag-on completions on
# the probe set, and is never compared at the token level with a powered cell.
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
# VLLM_USE_FLASHINFER_SAMPLER=0 is part of the same pin. It is exported here rather than
# assumed from the submitting shell, because a cell whose sampler differs is a different
# serving mode under a name that says it is not.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# The preflight of ruling R1: 30 items at 1 and at 32 in flight on THIS server before any
# arm runs. BCF_PREFLIGHT=0 skips it, which also stamps the run exploratory, because a
# cell with no determinism measurement of its own is not a cell of record.
PREFLIGHT="${BCF_PREFLIGHT:-1}"
PREFLIGHT_ITEMS="${BCF_PREFLIGHT_ITEMS:-30}"
PREFLIGHT_LEVELS="${BCF_PREFLIGHT_LEVELS:-1,${CONCURRENCY}}"
if [ "$PREFLIGHT" != "1" ]; then
  RUN_LABEL="exploratory"
  EXPLORATORY_REASON="${EXPLORATORY_REASON}${EXPLORATORY_REASON:+; }BCF_PREFLIGHT=${PREFLIGHT} (no determinism preflight on this server)"
fi
# Element 9.2 and element 8.3 draw parameters. The defaults ARE the frozen values; they
# are passed explicitly so a run's own command line carries them.
SAMPLING_K="${BCF_SAMPLING_K:-32}"
SAMPLING_TEMPERATURE="${BCF_SAMPLING_TEMPERATURE:-0.7}"
CURVE_REPEATS="${BCF_CURVE_REPEATS:-3}"
REPEAT_TEMPERATURES="${BCF_REPEAT_TEMPERATURES:-0.0,0.7}"
# Ruling R4: the CHAIN-level resamples. r = 3 at temperature 0.7 are the ruling's own
# values, passed explicitly so a run's command line carries them.
CHAIN_REPEATS="${BCF_CHAIN_REPEATS:-3}"
CHAIN_REPEAT_TEMPERATURE="${BCF_CHAIN_REPEAT_TEMPERATURE:-0.7}"
CHAIN_REPEAT_SEED="${BCF_CHAIN_REPEAT_SEED:-20260907}"
# A literal JSON default has to live OUTSIDE ${VAR:-word}: bash's word scanner for
# ${VAR:-word} treats the first bare } inside word as the expansion's own terminator,
# so `${BCF_TEMPLATE_KWARGS:-{\"enable_thinking\": false\}}` (the previous line here)
# rendered, with BCF_TEMPLATE_KWARGS unset, as the 26-byte string
# `{"enable_thinking": false\}` -- a stray backslash, no closing brace, invalid JSON --
# under bash 3.2 (macOS's frozen system /bin/bash). Verified NOT to happen under bash
# 5 (the cluster's interpreter, and CI's): the old line renders the correct
# {"enable_thinking": false} there. So this never actually corrupted a run_meta.json on
# the cluster; it only fired inside tests/harness/render_serve_command.sh when that
# harness was run on a Mac, which is what produced
# tests/fixtures/serve_render_default_bare_before_fix.txt and made that fixture disagree
# with the Linux CI runner's own correct render. Fixed anyway, so the default no longer
# depends on which bash build happens to be running it: a plain variable reference has
# no braces of its own for any version's word-scanner to mis-parse.
DEFAULT_TEMPLATE_KWARGS='{"enable_thinking": false}'
TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-$DEFAULT_TEMPLATE_KWARGS}"
OUT_ROOT="${BCF_OUT_ROOT:-$BCF_RESULTS}"
# A relative subdirectory under the results root, so a wave file can separate a line's
# output without hard-coding a home directory into a manifest that travels between
# machines. The enrichment pass uses it (BCF_OUT_SUBROOT=enrichment): its model,
# substrate and cue are the SAME three values as the regular stated-hint cell's, so
# without this the pass and the cell would share one directory, one checkpoint and one
# summary file.
if [ -n "${BCF_OUT_SUBROOT:-}" ]; then OUT_ROOT="${OUT_ROOT}/${BCF_OUT_SUBROOT}"; fi
# RULING R3(ii). An enrichment CELL names its items in a list file written by the
# enrichment pass; an enrichment PASS asks for the report step that writes one. Both are
# empty on a regular cell, which is why a regular cell's command line is unchanged.
ITEM_LIST="${BCF_ITEM_LIST:-}"
ENRICH_REPORT="${BCF_ENRICH_REPORT:-0}"
# Element 11's ladder. Empty for every sweep cell, and the ONLY thing that selects any
# of the three local-checkpoint branches below. LOCAL_CHECKPOINT_META is the fragment
# spliced into the run_meta.json literal: empty by default, so the default case's
# run_meta.json is byte-identical to what this script wrote before the branch existed.
LOCAL_CHECKPOINT="${BCF_LOCAL_CHECKPOINT:-}"
LOCAL_CHECKPOINT_META=""
# RULING R12. Refused HERE, before the path assertion and long before the weight
# download, because a typo'd mode that fell through to a default would produce a cell
# whose run_meta names a configuration it did not run.
REASONING_MODE="${BCF_REASONING_MODE:-default}"
case "$REASONING_MODE" in
  default|off|on) : ;;
  *) echo "[reasoning] REFUSING: BCF_REASONING_MODE=${REASONING_MODE} is not default, off or on" >&2
     exit 14 ;;
esac
# R12(1): 4,096 for FULL generations on an "on" row only. The forced-continuation budget
# is not touched anywhere; it is the element 15 constant 24 inside the runner. The
# PREFLIGHT keeps NUM_PREDICT (see step 4b): it is a determinism measurement of the
# server, and running it at 4,096 is what made the audit's concurrency-1 leg prohibitive.
NUM_PREDICT_FULL="$NUM_PREDICT"
if [ "$REASONING_MODE" = "on" ]; then NUM_PREDICT_FULL=4096; fi

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
BASE_URL="http://127.0.0.1:${PORT}/v1"
export BCF_REQUEST_LOG="${OUT_DIR}/requests.jsonl"

# exit_code.txt is written on EVERY exit path, including a kill, and a 0 is written
# ONLY once arms_summary.json (bcf/contract_layout.py's step 6, below) actually exists:
# jobs 827052 and 827096 were CANCELLED by Slurm with SIGTERM on 2026-09-07 and still
# wrote exit_code.txt=0 with no arms_summary.json, because the old finish() here had no
# TERM trap and just read $? whenever the EXIT trap happened to fire. See
# bcf/env.sh:bcf_install_exit_guard for the full docstring and tests/test_exit_guard.py
# for the revert proof (the old shape really does write 0 on this exact scenario).
bcf_install_exit_guard "$OUT_DIR" "arms_summary.json"

# A named list file that is not there means the enrichment pass this cell depends on has
# not landed. Refusing HERE costs nothing; refusing after the server is up costs the
# weight download and the queue wait, and refusing NOT AT ALL would silently run the
# cell on the first BCF_N_ITEMS of the pool under a name that says otherwise.
if [ -n "$ITEM_LIST" ] && [ ! -f "$ITEM_LIST" ]; then
  echo "[item-list] REFUSING: BCF_ITEM_LIST=${ITEM_LIST} does not exist. This cell would" >&2
  echo "[item-list]   otherwise run on the regular n under an enrichment name." >&2
  exit 11
fi

{
  echo "=== bcf serve_and_run: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "model=${MODEL} substrate=${SUBSTRATE} cue=${CUE} tp=${TP} n_items=${N_ITEMS}"
  echo "out_dir=${OUT_DIR}"
  echo "serving port=${PORT} (BCF_PORT=${BCF_PORT:-<unset>, picked free for this job})"
} | ts | tee -a "$LOG"

# --- 1. device assertion, BEFORE the download ----------------------------------
if ! bcf_assert_devices "$TP" 2>&1 | ts | tee -a "$LOG"; then
  exit 3
fi

# --- 2. pin the revision -------------------------------------------------------
if [ -n "$LOCAL_CHECKPOINT" ]; then
# LADDER. The revision of a local checkpoint IS the sha256 of its own manifest, computed
# by the SAME function that wrote the wave manifest's pinned string
# (bayes_cot_faithfulness.ladder.serve_manifest.checkpoint_revision), so the drift check
# below compares like with like instead of being skipped. BCF_MODEL_PATH points at the
# merged directory and the manifest sits beside it in the checkpoint directory, so both
# are looked in, in that order, and neither is guessed at.
if [ ! -d "$LOCAL_CHECKPOINT" ]; then
  echo "[revision] REFUSING: BCF_LOCAL_CHECKPOINT=${LOCAL_CHECKPOINT} is not a directory" | ts | tee -a "$LOG" >&2
  exit 4
fi
REVISION="$(BCF_LOCAL_CKPT_DIR="$LOCAL_CHECKPOINT" python - 2>>"$LOG" <<'PY'
import os, pathlib, sys
from bayes_cot_faithfulness.ladder.serve_manifest import checkpoint_revision
d = pathlib.Path(os.environ["BCF_LOCAL_CKPT_DIR"]).expanduser()
for cand in (d / "manifest.json", d.parent / "manifest.json"):
    if cand.exists():
        print(checkpoint_revision(cand))
        sys.exit(0)
sys.exit(1)
PY
)"
if [ -z "$REVISION" ]; then
  echo "[revision] REFUSING: no checkpoint manifest.json in ${LOCAL_CHECKPOINT} or its" | ts | tee -a "$LOG" >&2
  echo "[revision]   parent, so this checkpoint has no revision string and a transcript" | ts | tee -a "$LOG" >&2
  echo "[revision]   from it could not name the weights that produced it." | ts | tee -a "$LOG" >&2
  exit 4
fi
LOCAL_CHECKPOINT_META="\"local_checkpoint\": True, \"local_checkpoint_path\": \"${LOCAL_CHECKPOINT}\","
else
REVISION="$(bcf_revision "$MODEL" 2>>"$LOG")"
if [ -z "$REVISION" ]; then
  echo "[revision] REFUSING: could not resolve an HF revision sha for ${MODEL}" | ts | tee -a "$LOG" >&2
  exit 4
fi
fi
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
echo "[revision] ${MODEL} @ ${REVISION}  (vllm ${VLLM_VERSION})" | ts | tee -a "$LOG"
# Element 10 pins a revision per roster row and says the runner compares against it.
# When the wave manifest carries BCF_REVISION, a drift between the pinned sha and what
# the Hub resolves today stops the cell here rather than producing a cell whose
# hf_revision field quietly disagrees with the pre-registration.
# For a local checkpoint the same check is an equality check between the revision the
# manifest pinned and the checkpoint's own manifest hash, which is the failure worth
# catching there: a wave row pointing at a checkpoint other than the one it names.
RESOLVED_BY="the Hub"
if [ -n "$LOCAL_CHECKPOINT" ]; then RESOLVED_BY="the checkpoint manifest"; fi
if [ -n "${BCF_REVISION:-}" ] && [ "${BCF_REVISION}" != "${REVISION}" ]; then
  echo "[revision] REFUSING: the wave pins ${BCF_REVISION} but ${RESOLVED_BY} resolves ${REVISION}" \
    | ts | tee -a "$LOG" >&2
  exit 8
fi
BCF_META_TEMPLATE_KWARGS="$TEMPLATE_KWARGS" python - <<PY >> "$LOG" 2>&1
import json, os, pathlib
pathlib.Path("${OUT_DIR}/run_meta.json").write_text(json.dumps({
    "job_id": "${SLURM_JOB_ID}", "model": "${MODEL}", "hf_revision": "${REVISION}",
    ${LOCAL_CHECKPOINT_META}
    "backend": "openai", "backend_version": "vllm ${VLLM_VERSION}",
    "substrate": "${SUBSTRATE}", "cue_family": "${CUE}",
    "tensor_parallel_size": ${TP}, "seed": ${SEED}, "concurrency": ${CONCURRENCY},
    # Which way the batch-determinism switch went, on the record. Job 825548 measured
    # 11/30 identical completions at 8 in flight with the flag off, so a run's
    # concurrency is meaningless without this beside it.
    "batch_invariant": "${BATCH_INVARIANT}" == "1",
    "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
    "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    "vllm_version": "${VLLM_VERSION}",
    # RULING R1. A cell of record is flag ON with its own determinism preflight passed.
    # Anything else is EXPLORATORY and says why, in the file the analysis reads.
    "run_label": "${RUN_LABEL}",
    "exploratory_reason": "${EXPLORATORY_REASON}" or None,
    "determinism_preflight": "${PREFLIGHT}" == "1",
    "determinism_preflight_levels": "${PREFLIGHT_LEVELS}",
    "determinism_preflight_n_items": ${PREFLIGHT_ITEMS},
    # Filled in after the server answers, from server.log: which attention backend vLLM
    # actually selected. The flag is honored by FLASH_ATTN; a build that fell back to
    # another backend would lose the property with no other visible symptom.
    "attention_backend": None,
    "attention_backend_lines": [],
    "sampling_k": ${SAMPLING_K}, "sampling_temperature": ${SAMPLING_TEMPERATURE},
    "curve_repeats": ${CURVE_REPEATS},
    "repeat_temperatures": [float(x) for x in "${REPEAT_TEMPERATURES}".split(",")],
    "chain_repeats": ${CHAIN_REPEATS},
    "chain_repeat_temperature": ${CHAIN_REPEAT_TEMPERATURE},
    "chain_repeat_seed": ${CHAIN_REPEAT_SEED},
    "plan_commit": "${BCF_PLAN_COMMIT:-}", "bcf_repo": "${BCF_REPO}",
    # Which loopback port this job served on, and (filled in once the server answers)
    # the pid we launched plus the verdict of the port-ownership check. Two jobs on one
    # node used to be indistinguishable in the record; these three fields are what makes
    # "this cell had its own server" checkable after the fact.
    "port": ${PORT},
    "server_pid": None,
    "port_owner_check": None,
    # RULING R3(ii). Which of the two enrichment shapes this job is, on the record that
    # the analysis reads, so an enrichment cell can never be pooled with a regular one by
    # accident. Both are absent/false on a regular cell.
    "item_list": "${ITEM_LIST}" or None,
    "enrichment_cell": bool("${ITEM_LIST}"),
    "enrichment_pass": "${ENRICH_REPORT}" == "1",
    "temperature": 0.0, "num_predict": ${NUM_PREDICT}, "curve_cap": ${CURVE_CAP},
    # RULING R12. Written HERE from the submitted value, so a job that dies before the
    # arms finish still records which configuration it was asked for; the runner merges
    # the measured half (the close branch, the rendered prompt tail, the continuation
    # reopen counts) into the same file when it finalizes.
    "reasoning_mode": "${REASONING_MODE}",
    "reasoning_path": ("completions" if "${REASONING_MODE}" == "off" else "chat"),
    "reasoning_block_closed": (True if "${REASONING_MODE}" == "off" else None),
    "num_predict_full": ${NUM_PREDICT_FULL},
    # The determinism preflight and the forced-logprob check both run on the CHAT path,
    # whatever the mode: they measure the server, and R12(2) is what certifies that the
    # rendered /completions path renders the same prompt. Recorded rather than implied.
    "reasoning_preflight_path": "chat",
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
# The two things, and the only two things, a local checkpoint changes about the serve
# line: what vLLM loads, and whether --revision is passed at all (a Hub revision flag on
# a local directory is not a no-op, it is an error). Both hold the Hub-path values when
# BCF_LOCAL_CHECKPOINT is unset, so the array renders exactly as it did.
SERVE_TARGET="$MODEL"
REVISION_ARGS=(--revision "$REVISION")
if [ -n "$LOCAL_CHECKPOINT" ]; then
  SERVE_TARGET="$LOCAL_CHECKPOINT"
  REVISION_ARGS=()
fi
VLLM_ARGS=(
  ${REVISION_ARGS[@]+"${REVISION_ARGS[@]}"} --served-model-name "$MODEL"
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
echo "[serve] vllm serve ${SERVE_TARGET} ${VLLM_ARGS[*]}" | ts | tee -a "$LOG"
vllm serve "$SERVE_TARGET" "${VLLM_ARGS[@]}" >> "${OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
bcf_guard_register_child "$SERVER_PID"

echo "[serve] waiting up to ${SERVER_WAIT}s for the server (pid ${SERVER_PID}) on port ${PORT}" | ts | tee -a "$LOG"
DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
# "Something answers on the port" is NOT readiness. bcf_wait_server_ready requires the
# /v1/models list to name OUR served model, our server pid to be alive, and the process
# listening on the port to be that pid or a descendant of it. Anything else is refused
# on the first poll that answers, with exit 13, rather than run against.
READY_REPORT="${OUT_DIR}/serve_ready.json"
bcf_wait_server_ready "$BASE_URL" "$MODEL" "$SERVER_PID" "$DEADLINE" \
  "${OUT_DIR}/server.log" "$READY_REPORT" | ts | tee -a "$LOG"
READY_RC=${PIPESTATUS[0]}
if [ "$READY_RC" -ne 0 ]; then
  exit "$READY_RC"
fi
grep -iE "attention backend|Using .* backend|batch.?invariant" "${OUT_DIR}/server.log" \
  | tail -20 | sed 's/^/[serve] /' | ts | tee -a "$LOG" || true
echo "[serve] up at ${BASE_URL} after $(( SERVER_WAIT - (DEADLINE - $(date +%s)) ))s" | ts | tee -a "$LOG"

# --- 3b. the attention backend the server ACTUALLY selected, into run_meta.json -----
# RULING R1 pins the backend as part of the serving mode, and vLLM chooses it at startup
# from the build, the head size and the dtype. The flag's batch-invariant kernels are
# honored by FLASH_ATTN; a fallback to another backend would keep the environment
# variable set and lose the property, so the selected backend is read back out of the
# server's own log rather than assumed from the variable.
BCF_META_OUT_DIR="$OUT_DIR" python - <<'PY' >> "$LOG" 2>&1
import json, os, pathlib, re

out_dir = pathlib.Path(os.environ["BCF_META_OUT_DIR"])
meta_path = out_dir / "run_meta.json"
log = (out_dir / "server.log").read_text(errors="replace") if (out_dir / "server.log").exists() else ""
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

# The port-ownership verdict, from the file bcf/serve_ready.py just wrote. The record
# has to say WHICH port this cell served on, WHICH pid served it, and whether the
# listener was proved to be that pid ("passed") or could not be named on this node
# ("skipped: ..."). A cell whose port_owner_check is skipped is not wrong, but it is
# not proof either, and the analysis can now tell the two apart.
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
print(f"[serve] attention_backend={backend} from {len(lines)} matching server.log line(s)")
print(f"[serve] port={meta.get('port')} server_pid={meta.get('server_pid')} "
      f"port_owner_check={meta.get('port_owner_check')}")
PY


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

# --- 4b. RULING R1: the determinism preflight, on THIS server, before any arm ------
# The first 30 items of this cell's own pool, generated and read as forced letter
# logprobs at 1 and at ${CONCURRENCY} in flight. The cell runs only on 30/30 identical
# completions and a max absolute letter-logprob difference of exactly 0.0. Exit 10 is
# reserved for this refusal so it is readable from exit_code.txt alone.
#
# The cost is 60 generations and 120 forced logprob reads, about half a minute on the
# measured 8B line, against a cell that runs for hours. What it buys is that the pinned
# serving mode is a property of THIS server rather than of one 2026-09-07 probe on one
# model: a build that fell back to another attention backend keeps the environment
# variable set and silently loses the property.
if [ "$PREFLIGHT" = "1" ]; then
  echo "[preflight] 30 items at ${PREFLIGHT_LEVELS} in flight, flag ${VLLM_BATCH_INVARIANT:-<unset>}" \
    | ts | tee -a "$LOG"
  python "${BCF_REPO}/bcf/determinism_preflight.py" \
    --base-url "$BASE_URL" --model "$MODEL" --seed "$SEED" \
    --data "${BCF_REPO}/experiments/data/${SUBSTRATE}.json" \
    --n-items "$PREFLIGHT_ITEMS" --levels "$PREFLIGHT_LEVELS" \
    --num-predict "$NUM_PREDICT" \
    --chat-template-kwargs "$TEMPLATE_KWARGS" \
    --out "${OUT_DIR}/determinism_preflight.json" \
    --probe-out "${OUT_DIR}/determinism_preflight_probe.json" \
    2>&1 | ts | tee -a "$LOG"
  PF_STATUS=${PIPESTATUS[0]}
  if [ "$PF_STATUS" -ne 0 ]; then
    echo "[preflight] REFUSING THE CELL (${PF_STATUS}): this server did not reproduce its" \
      | ts | tee -a "$LOG" >&2
    echo "[preflight]   own concurrency-1 completions and letter logprobs under load, so" \
      | ts | tee -a "$LOG" >&2
    echo "[preflight]   every outcome it produced would carry batch-induced noise that no" \
      | ts | tee -a "$LOG" >&2
    echo "[preflight]   seed controls. Reasons are in ${OUT_DIR}/determinism_preflight.json" \
      | ts | tee -a "$LOG" >&2
    exit 10
  fi
  echo "[preflight] PASS" | ts | tee -a "$LOG"
else
  echo "[preflight] SKIPPED by BCF_PREFLIGHT=${PREFLIGHT}; this run is EXPLORATORY" \
    | ts | tee -a "$LOG"
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
ITEM_LIST_FLAGS=()
if [ -n "$ITEM_LIST" ]; then
  ITEM_LIST_FLAGS=(--item-list "$ITEM_LIST")
  echo "[item-list] this cell runs on the items named in ${ITEM_LIST}" | ts | tee -a "$LOG"
fi

echo "[reasoning] mode=${REASONING_MODE} num_predict_full=${NUM_PREDICT_FULL} (forced stays 24)" \
  | ts | tee -a "$LOG"
if [ "$REASONING_MODE" = "off" ]; then
  echo "[reasoning] the arms generate through /completions on a rendered, block-closed" \
    | ts | tee -a "$LOG"
  echo "[reasoning]   prompt; the preflight above measured the CHAT path. R12(2): the" \
    | ts | tee -a "$LOG"
  echo "[reasoning]   two-path equivalence gate certifies the two render the same prompt." \
    | ts | tee -a "$LOG"
fi
if [ "$REASONING_MODE" = "on" ] && [ "$MAX_LEN" -lt 16384 ]; then
  echo "[reasoning] WARNING: BCF_MAX_LEN=${MAX_LEN} with a 4096-token full generation;" \
    | ts | tee -a "$LOG"
  echo "[reasoning]   16384 is what the element 9.4 serving test used." | ts | tee -a "$LOG"
fi
echo "[arms] starting at $(date -Is)" | ts | tee -a "$LOG"
python "${BCF_REPO}/experiments/08_additive_arms.py" \
  --backend openai --base-url "$BASE_URL" \
  --model "$MODEL" --seed "$SEED" \
  --chat-template-kwargs "$TEMPLATE_KWARGS" \
  --data "${BCF_REPO}/experiments/data/${SUBSTRATE}.json" \
  --specificity-holdout "${BCF_REPO}/experiments/data/specificity_holdout.json" \
  --n-items "$N_ITEMS" --num-predict "$NUM_PREDICT_FULL" --curve-cap "$CURVE_CAP" \
  --reasoning-mode "$REASONING_MODE" \
  --concurrency "$CONCURRENCY" \
  --sampling-k "$SAMPLING_K" --sampling-temperature "$SAMPLING_TEMPERATURE" \
  --curve-repeats "$CURVE_REPEATS" --repeat-temperatures "$REPEAT_TEMPERATURES" \
  --chain-repeats "$CHAIN_REPEATS" \
  --chain-repeat-temperature "$CHAIN_REPEAT_TEMPERATURE" \
  --chain-repeat-seed "$CHAIN_REPEAT_SEED" \
  --out "$OUT_DIR" --timeout 600 --resume \
  "${ARM_FLAGS[@]}" "${TAXONOMY_FLAGS[@]}" "${ITEM_LIST_FLAGS[@]}" 2>&1 | ts | tee -a "$LOG"
ARMS_STATUS=${PIPESTATUS[0]}
echo "[arms] finished at $(date -Is) with status ${ARMS_STATUS}" | ts | tee -a "$LOG"

# --- 6. CONTRACT filenames -----------------------------------------------------
python "${BCF_REPO}/bcf/contract_layout.py" --out-dir "$OUT_DIR" 2>&1 | ts | tee -a "$LOG"

# --- 7. throughput, measured from this run's own log ---------------------------
python "${BCF_REPO}/bcf/throughput.py" --run-log "$LOG" --out-dir "$OUT_DIR" \
  2>&1 | ts | tee -a "$LOG" || true

# --- 8. RULING R3(ii): the enrichment pass's own product ------------------------
# Only for a pass (BCF_ENRICH_REPORT=1). It reads the per-item sampling blocks this run
# just wrote and turns them into uncertain_items.json, the list file an enrichment cell
# is submitted with. It never changes the arms' own artifacts, and its status is reported
# but does NOT replace the arms status: a pass whose arm finished and whose report failed
# is a pass whose data is intact.
if [ "$ENRICH_REPORT" = "1" ]; then
  python "${BCF_REPO}/bcf/enrich_report.py" \
    --out-dir "$OUT_DIR" --data "${BCF_REPO}/experiments/data/${SUBSTRATE}.json" \
    --substrate "$SUBSTRATE" 2>&1 | ts | tee -a "$LOG"
  ENRICH_STATUS=${PIPESTATUS[0]}
  echo "$ENRICH_STATUS" > "${OUT_DIR}/enrich_report_exit.txt"
  echo "[enrich] report exit ${ENRICH_STATUS} -> ${OUT_DIR}/enrich_report_exit.txt" \
    | ts | tee -a "$LOG"
fi

exit "$ARMS_STATUS"
