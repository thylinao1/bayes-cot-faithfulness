#!/bin/bash
#SBATCH --job-name=bcf-jury-serve
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
# Serve one or two jury judges on the allocated card, run the synthetic judge gate against
# them, and EXIT. No idle server: the gate is the job, and when it finishes the card goes
# back to the pool.
#
# --mem is in the header because leaving it out is fatal here and looks like something
# else. Slurm's default on this cluster is 3G of host RAM and 1 CPU. Job 825536 asked for
# an h200-141, got 3G, and its vLLM EngineCore was OOM-killed at 02:42 with no traceback
# in the server log, only "Engine core initialization failed"; sacct is where the truth was
# (State=OUT_OF_MEMORY, ExitCode 0:125, ReqMem=3G, MaxRSS=6.3G). A 70B needs tens of GB of
# host RAM to stream its shards to the card, so the judge jobs ask for 96G and the 70B row
# asks for more through jury_wave.sh's MEM= field.
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
#            BCF_RESUME BCF_Q1_PROMPT BCF_OUT_SLUG BCF_SERVING_LINE BCF_SERVING_NOTE
#            BCF_HOME_USAGE_TTL_H BCF_ALL_JUDGE_ROWS BCF_ECHO_STRIP
#
# BCF_Q1_PROMPT names which Q1 prompt file the gate loads ("a" is the file of record,
# "b" and "c" the revisions). It also takes a list, separated by "+" in a job table and by
# "," or "+" on a hand-run sbatch ("a+c"), in which case the gate
# runs once per variant into ${SLUG}-q1<variant> and the served judge is paid for only
# once. BCF_OUT_SLUG overrides the results directory name so two Q1 prompts
# on the same judge do not write into one vote file. BCF_SERVING_LINE labels a run served
# on a card other than the one section 6.1 pins for that judge; it lands on every vote
# record and in the report, so an exploratory run can never be read as the run of record.

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

# The exit-code guard. Slurm runs this file from its own spool directory, so $0 is not in
# the repo and the guard is sourced by absolute path like env.sh is.
source "${BCF_EXIT_GUARD_SH:-${BCF_REPO}/bcf/exit_guard.sh}"

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

JUDGES="${BCF_JUDGES:?BCF_JUDGES is required, e.g. qwen3-32b or gemma-3-27b-it+gpt-oss-20b}"
# A co-hosted pair arrives with '+' between the keys, because sbatch --export is itself
# comma separated: "BCF_JUDGES=gemma-3-27b-it,gpt-oss-20b" reaches the job as
# BCF_JUDGES=gemma-3-27b-it plus a stray variable name, and the job would then serve ONE
# judge while its results directory and its report claimed two. Both separators are
# accepted here so a hand-run sbatch that quotes the value properly still works.
JUDGES="$(printf '%s' "$JUDGES" | tr '+' ',')"
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
Q1_PROMPT="${BCF_Q1_PROMPT:-a}"
SERVING_LINE="${BCF_SERVING_LINE:-}"
# The note carries spaces and commas, and jury_wave.sh word-splits the tsv's extra fields
# and joins them with commas into --export. So the note is NOT passed through the job
# table: it is defaulted here from the serving line, and BCF_SERVING_NOTE only overrides it
# for a hand-run sbatch where the quoting is under the caller's control.
SERVING_NOTE="${BCF_SERVING_NOTE:-}"
if [ -n "$SERVING_LINE" ] && [ -z "$SERVING_NOTE" ]; then
  SERVING_NOTE="exploratory: served on ${SERVING_LINE}, NOT the line section 6.1 pins for this judge; the run of record is the pinned a100-80 line and it runs unchanged from bcf/judges_gate_a100*.tsv"
fi
RESUME_FLAG=""
[ -n "${BCF_RESUME:-}" ] && RESUME_FLAG="--resume"
# The panel rule routes a judge away from its own family, so serving ONLY the Qwen judge
# against a corpus whose subject_model is Qwen3-8B plans zero votes and writes an empty
# report. Set this to score every item with the served judge and record the own-family vote
# as one.
ALL_ROWS_FLAG=""
[ -n "${BCF_ALL_JUDGE_ROWS:-}" ] && ALL_ROWS_FLAG="--all-judge-rows"
# EXPLORATORY option (d): strip verbatim reproductions of the prompt out of the response
# before the judge sees it. Unset or 0 is off, which is what every run of record uses. The
# strip's parameters and the SHA-256 of experiments/jury/echo_strip.py land on every vote
# and in the report, so a stripped run cannot be read as an unstripped one.
ECHO_STRIP_FLAG=""
[ "${BCF_ECHO_STRIP:-0}" -gt 0 ] 2>/dev/null && ECHO_STRIP_FLAG="--echo-strip ${BCF_ECHO_STRIP}"

# vLLM 0.28 JIT-compiles the FlashInfer top-k/top-p sampler on first use and that build is
# node dependent (job 825480 died on xgph10 after loading the weights). Every judge call is
# temperature 0, so the sampler kernel is off the measurement path. env.sh sets this too;
# it is repeated here so a judge job started with a different env file still gets it.
export VLLM_USE_FLASHINFER_SAMPLER=0

SLUG="${BCF_OUT_SLUG:-$(echo "$JUDGES" | tr ',' '-')}"
OUT_DIR="${OUT_ROOT}/${SLUG}/${SUBSTRATE}/${CUE}"
case "$OUT_DIR" in
  *"/${SUBSTRATE}/${CUE}") : ;;
  *) echo "[path] REFUSING: ${OUT_DIR} does not end in /${SUBSTRATE}/${CUE}" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"
EXIT_FILE="${OUT_DIR}/exit_code.txt"
# Written NON-ZERO here, before anything can go wrong, and overwritten by finish() only
# with a code that the guard has checked. Job 826023 was cancelled mid-variant and its
# EXIT trap wrote 0; nothing downstream could tell 2 percent of a run from all of it.
bcf_exit_guard_init "$EXIT_FILE"

SERVER_PIDS=()
finish() {
  local raw=$?
  local code
  for pid in "${SERVER_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null || true
  done
  sleep 5
  for pid in "${SERVER_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -9 "$pid" 2>/dev/null || true
  done
  # A cancel, a kill or an unwind that never reached the completion marker cannot record a
  # zero: 128+signal for a trapped signal, 250 for a zero that skipped the marker, and the
  # 255 already on disk for anything that never got here at all.
  code="$(bcf_exit_guard_code "$raw")"
  bcf_exit_guard_write "$code"
  if [ "$code" != "$raw" ]; then
    echo "[done] raw status ${raw} recorded as ${code} (signal=${BCF_EXIT_SIGNAL}, completed=${BCF_EXIT_COMPLETE})" | ts | tee -a "$LOG"
  fi
  echo "[done] exit_code=${code} -> ${EXIT_FILE}" | ts | tee -a "$LOG"
  exit "$code"
}
trap finish EXIT

{
  echo "=== bcf judge_serve: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "judges=${JUDGES} items=${GATE_ITEMS} out=${OUT_DIR} concurrency=${CONCURRENCY}"
  echo "q1_prompt=${Q1_PROMPT} serving_line=${SERVING_LINE:-pinned}"
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
# A full `du -sBG $HOME` over this NFS home costs 90 to 120 s and every judge job paid it.
# Split the number instead: the REST of home changes only when another campaign writes,
# so it is measured once and memoised with a TTL, while the judge cache is the only part
# this lane grows and is measured fresh every run in about a second. HOME_USED_GB is still
# the same quantity, and the ceiling it is compared against does not move.
HOME_USAGE_CACHE="${JUDGE_CACHE%/}/.home-usage-rest-gb"
HOME_USAGE_TTL_H="${BCF_HOME_USAGE_TTL_H:-24}"
CACHE_USED_GB="$(du -sBG "$JUDGE_CACHE" 2>/dev/null | awk '{gsub("G","",$1); print $1+0}')"
REST_GB=""
if [ -f "$HOME_USAGE_CACHE" ]; then
  age_s=$(( $(date +%s) - $(stat -c %Y "$HOME_USAGE_CACHE" 2>/dev/null || echo 0) ))
  if [ "$age_s" -lt $(( HOME_USAGE_TTL_H * 3600 )) ]; then
    REST_GB="$(awk '{print $1+0}' "$HOME_USAGE_CACHE" 2>/dev/null)"
    echo "[cache] home-minus-cache ${REST_GB} GB from ${HOME_USAGE_CACHE} (${age_s}s old, ttl ${HOME_USAGE_TTL_H}h)" | ts | tee -a "$LOG"
  fi
fi
if [ -z "$REST_GB" ]; then
  scan_start=$(date +%s)
  TOTAL_GB="$(du -sBG "$HOME" 2>/dev/null | awk '{gsub("G","",$1); print $1+0}')"
  REST_GB=$(( TOTAL_GB - CACHE_USED_GB ))
  [ "$REST_GB" -lt 0 ] && REST_GB=0
  echo "$REST_GB" > "$HOME_USAGE_CACHE"
  echo "[cache] full home scan took $(( $(date +%s) - scan_start ))s: total ${TOTAL_GB} GB, cache ${CACHE_USED_GB} GB, rest ${REST_GB} GB memoised for ${HOME_USAGE_TTL_H}h" | ts | tee -a "$LOG"
fi
HOME_USED_GB=$(( REST_GB + CACHE_USED_GB ))
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
# What is ALREADY in the persistent cache for these judges does not need downloading
# again, and home usage already counts it, so the decision has to be made on the NET need.
# Without this the preflight re-downloads a model it is already holding: job 825542 found
# home at 439 GB (365 plus the 74 GB it had just cached), computed 11 GB of headroom
# against a 72 GB gross need, and fell back to scratch to fetch the same 74 GB again.
HAVE_GB=0
for key in $(echo "$JUDGES" | tr ',' ' '); do
  if eval "$(python "${BCF_REPO}/bcf/judge_spec.py" "$key" 2>/dev/null)"; then
    # Replace the SLASH with "--", not the first hyphen: an org name may itself contain a
    # hyphen (meta-llama), and "tr / - | sed s/-/--/" would then double the wrong one.
    pattern="models--$(echo "$JUDGE_HF_ID" | sed 's|/|--|')"
    have=$(du -sBG "$JUDGE_CACHE"/*/"${pattern}"* "$JUDGE_CACHE/${pattern}"* 2>/dev/null \
      | awk '{gsub("G","",$1); s+=$1+0} END {print s+0}')
    HAVE_GB=$(( HAVE_GB + have ))
  fi
done
NET_GB=$(( NEED_GB - HAVE_GB ))
[ "$NET_GB" -lt 0 ] && NET_GB=0
echo "[cache] home used ${HOME_USED_GB} GB (rest ${REST_GB} + cache ${CACHE_USED_GB}) of a ${CACHE_BUDGET_GB} GB working ceiling; headroom ${HEADROOM_GB} GB; gross need ${NEED_GB} GB, already cached ${HAVE_GB} GB, net need ${NET_GB} GB" | ts | tee -a "$LOG"
if [ "$NET_GB" -le "$HEADROOM_GB" ]; then
  export HF_HOME="$JUDGE_CACHE"
  mkdir -p "$HF_HOME"
  echo "[cache] PERSISTENT: HF_HOME=${HF_HOME} (weights survive the job, restarts are free)" | ts | tee -a "$LOG"
else
  echo "[cache] SCRATCH: net need ${NET_GB} GB exceeds ${HEADROOM_GB} GB of headroom, falling back to ${SCRATCH}/hf. The download repeats next run; the quota does not blow up." | ts | tee -a "$LOG"
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

  # EXPLORATORY LINES ONLY: the memory fraction in the section 6.1 table describes the card
  # that table names. gpt-oss-20b carries 0.25 because the table co-hosted it with Gemma on
  # one a100-80; R7 dissolved that pair, and 0.25 of a 47 GB MIG slice is 11.7 GB, under the
  # 13 GB the mxfp4 weights need, so vLLM loads the model and then has nothing left for the
  # KV cache (job 826859, "No available memory for the cache blocks", exit 5). A row may
  # override the fraction, and only a row whose serving line says exploratory: a run of
  # record serves at the table's number or it does not serve.
  # The row's own BCF_SERVING_LINE is what decides, not JUDGE_SERVING_LINE, which is the
  # table's prose and says "a100-80" on every gpt-oss row whatever card the job holds.
  if [ -n "${BCF_JUDGE_GPU_UTIL:-}" ]; then
    case "${BCF_SERVING_LINE:-}" in
      exploratory*)
        echo "[serve] ${key}: gpu-memory-utilization OVERRIDDEN ${JUDGE_GPU_UTIL} -> ${BCF_JUDGE_GPU_UTIL} (row serving line '${BCF_SERVING_LINE}'; the table fraction belongs to '${JUDGE_SERVING_LINE}', a card this line is not on)" | ts | tee -a "$LOG"
        JUDGE_GPU_UTIL="${BCF_JUDGE_GPU_UTIL}"
        ;;
      *)
        echo "[judge] REFUSING: BCF_JUDGE_GPU_UTIL=${BCF_JUDGE_GPU_UTIL} with BCF_SERVING_LINE='${BCF_SERVING_LINE:-<unset>}', which is not exploratory. A run of record serves at the section 6.1 fraction ${JUDGE_GPU_UTIL}." | ts | tee -a "$LOG" >&2
        exit 12
        ;;
    esac
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
LINE_FLAGS=()
[ -n "$SERVING_LINE" ] && LINE_FLAGS+=(--serving-line "$SERVING_LINE")
[ -n "$SERVING_NOTE" ] && LINE_FLAGS+=(--serving-line-note "$SERVING_NOTE")

cd "$BCF_REPO" || exit 9
# '+' as well as ',' separates the variants. A comma cannot travel through a job table:
# jury_wave.sh joins the tsv's extra fields into sbatch --export, which is itself comma
# separated, so BCF_Q1_PROMPT=a,c would arrive as BCF_Q1_PROMPT=a plus a stray name "c".
IFS=',' read -r -a Q1_VARIANTS <<< "$(printf '%s' "$Q1_PROMPT" | tr '+' ',')"
GATE_STATUS=0
for variant in "${Q1_VARIANTS[@]}"; do
  # One served judge, one gate run per Q1 prompt. With a single variant the results path
  # is unchanged, so a resumed run finds its own vote file where it left it.
  if [ "${#Q1_VARIANTS[@]}" -gt 1 ]; then
    RUN_OUT="${OUT_ROOT}/${SLUG}-q1${variant}/${SUBSTRATE}/${CUE}"
    mkdir -p "$RUN_OUT"
  else
    RUN_OUT="$OUT_DIR"
  fi
  echo "[gate] Q1 variant ${variant} starting at $(date -Is) -> ${RUN_OUT}" | ts | tee -a "$LOG"
  # Same rule one level down: the variant's own exit file starts at 255, so a run
  # cancelled inside variant c leaves c non-zero and does not disturb a and b.
  bcf_exit_guard_open_stage "${RUN_OUT}/exit_code.txt"
  python -m experiments.jury.gate \
    --items "$GATE_ITEMS" --out "$RUN_OUT" \
    --substrate "$SUBSTRATE" --cue-family "$CUE" \
    --seed "$SEED" --concurrency "$CONCURRENCY" --num-predict "$NUM_PREDICT" \
    --q1-prompt "$variant" \
    $RESUME_FLAG $ALL_ROWS_FLAG $ECHO_STRIP_FLAG "${LIMIT_FLAG[@]}" "${LINE_FLAGS[@]}" "${JUDGE_FLAGS[@]}" 2>&1 | ts | tee -a "$LOG"
  st=${PIPESTATUS[0]}
  echo "[gate] Q1 variant ${variant} finished at $(date -Is) with status ${st}" | ts | tee -a "$LOG"
  # 1 is a FAILED THRESHOLD, which is a result. Anything above 1 is the job breaking.
  [ "$st" -gt "$GATE_STATUS" ] && GATE_STATUS=$st
  echo "$st" > "${RUN_OUT}/exit_code.txt"
done
echo "[gate] all variants finished at $(date -Is), worst status ${GATE_STATUS}" | ts | tee -a "$LOG"

# Exit 0 when the gate RAN, whatever its verdict: a judge failing a threshold is a result,
# not a job failure, and mailing FAIL on it would train the operator to ignore the mail.
# The verdict lives in gate_report.json.
bcf_exit_guard_complete
if [ "$GATE_STATUS" -gt 1 ]; then
  exit "$GATE_STATUS"
fi
exit 0
