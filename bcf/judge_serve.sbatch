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
# Exit codes of this script: 2 results path, 3 device assertion, 4 HF API, 5 server died,
# 6 server timeout, 8 unknown judge key, 9 repo cd, 10 wrong repo tree, 12 gpu-util
# override on a non-exploratory line, 13 SERVE PORT NOT OURS (the port answers with a
# foreign model list, or from a process this job did not start; bcf/serve_ready.py),
# 14 the gpt-oss harmony vocab did not prewarm, 15 the SECONDARY sweep mode was asked for
# and its items manifest does not authorise it. A gate variant that crashed without
# writing gate_report.json is recorded as 12 and raises the job's own status to 12.
#
# TWO MODES, and the default is the gate. BCF_JURY_MODE=three-seeded (the default, and
# what every existing job table asks for by saying nothing) serves the judges and runs the
# synthetic gate, unchanged. BCF_JURY_MODE=audit is the SECONDARY sweep lane of RULING R15
# part 2 (2026-09-08): the same serving code, the same port-ownership probe and the same
# exit guard, but it scores ONE CELL of banked sweep records with
# experiments.jury.runner --mode audit (one run plus the seeded 10 percent three-run
# audit of section 6.4) and stamps every vote with the configuration name. It reads two
# more variables and refuses without them:
#
#   BCF_SWEEP_ITEMS      the cell's items JSONL, built by experiments/jury/sweep_items.py
#   BCF_ITEMS_MANIFEST   that file's manifest, default manifest.json beside it. It has to
#                        say "secondary": true, name a configuration, and carry the
#                        sha256 of the items file the job was handed, or the job stops at
#                        15 before a single vote is cast. The substrate and the cue family
#                        of the results path are READ FROM IT, so a cell's votes cannot be
#                        written under another cell's name.
#
# Required : BCF_JUDGES (comma separated judge keys), BCF_GATE_ITEMS (gate mode only)
# Optional : BCF_OUT_ROOT BCF_PORT_BASE BCF_CONCURRENCY BCF_GATE_LIMIT BCF_SEED
#            BCF_NUM_PREDICT BCF_SERVER_WAIT BCF_TP BCF_JUDGE_CACHE BCF_CACHE_BUDGET_GB
#            BCF_RESUME BCF_Q1_PROMPT BCF_OUT_SLUG BCF_SERVING_LINE BCF_SERVING_NOTE
#            BCF_HOME_USAGE_TTL_H BCF_ALL_JUDGE_ROWS BCF_ECHO_STRIP
#            BCF_JURY_MODE BCF_SWEEP_ITEMS BCF_ITEMS_MANIFEST
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
# The mode, decided before anything else reads a variable, because it decides WHICH items
# variable is required. Unset is the gate, which is what every job table written before
# 2026-09-08 asks for by saying nothing.
JURY_MODE="${BCF_JURY_MODE:-three-seeded}"
case "$JURY_MODE" in
  three-seeded|audit) : ;;
  *) echo "[mode] REFUSING: BCF_JURY_MODE=${JURY_MODE} is neither three-seeded nor audit" >&2; exit 15 ;;
esac
if [ "$JURY_MODE" = "three-seeded" ]; then
  GATE_ITEMS="${BCF_GATE_ITEMS:?BCF_GATE_ITEMS is required}"
  OUT_ROOT="${BCF_OUT_ROOT:-$HOME/bcf/results/jury-gate}"
else
  # jury_wave.sh puts BCF_GATE_ITEMS on every --export line it builds, so it is present
  # here and is deliberately NOT read: the audit lane's items come from BCF_SWEEP_ITEMS.
  GATE_ITEMS=""
  OUT_ROOT="${BCF_OUT_ROOT:-$HOME/bcf/results/jury-secondary}"
fi
# Unset means "a free loopback port per judge". A fixed base put two jobs of ours on
# one server on xgph12 on 2026-09-07 (see DECISION-LOG 17:41); a pinned BCF_PORT_BASE
# is still honored and still lays the judges out at BASE, BASE+1, ...
PORT_BASE="${BCF_PORT_BASE:-}"
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

# --- the SECONDARY sweep lane's items, and the manifest that authorises them ----
# Read BEFORE the results path exists, because the results path is read out of it: the
# cell's substrate and cue family come from the manifest, not from a variable a job table
# could get wrong, so a cell's votes cannot land under another cell's name. The checks
# that decide whether this job may run AT ALL are made further down, once the exit guard
# is armed, so a refusal is recorded in exit_code.txt like any other failure.
SWEEP_ITEMS=""
ITEMS_MANIFEST=""
ITEMS_CONFIGURATION=""
ITEMS_SECONDARY=""
ITEMS_SHA256=""
ITEMS_SHA256_ACTUAL=""
ITEMS_COUNT=""
ITEMS_SUBJECT=""
if [ "$JURY_MODE" = "audit" ]; then
  SWEEP_ITEMS="${BCF_SWEEP_ITEMS:-}"
  if [ ! -f "$SWEEP_ITEMS" ]; then
    echo "[items] REFUSING: BCF_JURY_MODE=audit needs BCF_SWEEP_ITEMS pointing at a cell's items JSONL (got '${SWEEP_ITEMS}')" >&2
    exit 15
  fi
  ITEMS_MANIFEST="${BCF_ITEMS_MANIFEST:-$(dirname "$SWEEP_ITEMS")/manifest.json}"
  if [ ! -f "$ITEMS_MANIFEST" ]; then
    echo "[items] REFUSING: no items manifest at ${ITEMS_MANIFEST}; experiments/jury/sweep_items.py writes one beside every items file" >&2
    exit 15
  fi
  # The echo strip is a gate-only option: experiments.jury.runner has no such flag, and
  # accepting the variable while dropping the flag would run an unstripped sweep under a
  # row that says otherwise.
  if [ -n "$ECHO_STRIP_FLAG" ]; then
    echo "[items] REFUSING: BCF_ECHO_STRIP=${BCF_ECHO_STRIP} is a gate-only option and this is BCF_JURY_MODE=audit" >&2
    exit 15
  fi
  case "$Q1_PROMPT" in
    *+*|*,*)
      echo "[items] REFUSING: the secondary configuration names ONE Q1 file; got BCF_Q1_PROMPT=${Q1_PROMPT}" >&2
      exit 15 ;;
  esac
  MANIFEST_ENV="$(python - "$ITEMS_MANIFEST" "$SWEEP_ITEMS" <<'MANIFEST'
import hashlib
import json
import shlex
import sys

manifest_path, items_path = sys.argv[1], sys.argv[2]
try:
    with open(manifest_path, encoding="utf-8") as fh:
        m = json.load(fh)
except (OSError, ValueError) as exc:
    print(f"[items] REFUSING: {manifest_path} is not readable JSON: {exc}", file=sys.stderr)
    raise SystemExit(1) from None
try:
    fields = {
        "ITEMS_SUBSTRATE": m["cell"]["substrate"],
        "ITEMS_CUE": m["cell"]["cue_family"],
        "ITEMS_SUBJECT": m["cell"]["subject_model"],
        "ITEMS_CONFIGURATION": m["configuration"],
        "ITEMS_SECONDARY": "true" if m.get("secondary") is True else "false",
        "ITEMS_SHA256": m["items"]["sha256"],
        "ITEMS_COUNT": m["items"]["count"],
    }
except KeyError as exc:
    print(f"[items] REFUSING: {manifest_path} has no {exc}; it is not a "
          f"sweep_items.py manifest", file=sys.stderr)
    raise SystemExit(1) from None
digest = hashlib.sha256()
with open(items_path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        digest.update(chunk)
fields["ITEMS_SHA256_ACTUAL"] = digest.hexdigest()
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
MANIFEST
)" || exit 15
  eval "$MANIFEST_ENV"
  SUBSTRATE="$ITEMS_SUBSTRATE"
  CUE="$ITEMS_CUE"
fi

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
  [ "$JURY_MODE" = "audit" ] && echo "mode=audit sweep_items=${SWEEP_ITEMS} manifest=${ITEMS_MANIFEST}"
} | ts | tee -a "$LOG"

# The check that decides whether the SECONDARY lane may run at all. It sits after the EXIT
# trap on purpose, so a refusal is recorded as 15 in exit_code.txt rather than leaving the
# guard's "started, never finished" 255 behind. What it is for: these votes are a reading
# reported BESIDE column A of record and never selected on (RULING R15 part 2 (g)), and
# the stamp the manifest authorises, which the runner then writes onto every row, is the
# only thing that keeps them that way.
if [ "$JURY_MODE" = "audit" ]; then
  if [ "$ITEMS_SECONDARY" != "true" ]; then
    echo "[items] REFUSING: ${ITEMS_MANIFEST} does not say \"secondary\": true. This lane writes secondary votes only." | ts | tee -a "$LOG" >&2
    exit 15
  fi
  case "$ITEMS_CONFIGURATION" in
    secondary-*) : ;;
    *)
      echo "[items] REFUSING: configuration '${ITEMS_CONFIGURATION}' in ${ITEMS_MANIFEST} does not name a secondary configuration" | ts | tee -a "$LOG" >&2
      exit 15 ;;
  esac
  if [ "$ITEMS_SHA256" != "$ITEMS_SHA256_ACTUAL" ]; then
    echo "[items] REFUSING: ${SWEEP_ITEMS} hashes to ${ITEMS_SHA256_ACTUAL} and its manifest describes ${ITEMS_SHA256}; the manifest does not belong to this items file" | ts | tee -a "$LOG" >&2
    exit 15
  fi
  echo "[items] SECONDARY ${ITEMS_CONFIGURATION}: ${ITEMS_COUNT} items, subject ${ITEMS_SUBJECT}, cell ${SUBSTRATE}/${CUE}, sha256 ${ITEMS_SHA256}" | ts | tee -a "$LOG"
fi

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
CACHE_BUDGET_GB="${BCF_CACHE_BUDGET_GB:-200}"  # DECISION-LOG 2026-09-07 03:00 and 14:5x: home is shared with another campaign against a ~500 GB quota; 200 forces node scratch until the operator frees space
mkdir -p "$JUDGE_CACHE"
# A full `du -sBG $HOME` over this NFS home costs 90 to 120 s and every judge job paid it.
# Split the number instead: the REST of home changes only when another campaign writes,
# so it is measured once and memoised with a TTL, while the judge cache is the only part
# this lane grows and is measured fresh every run in about a second. HOME_USED_GB is still
# the same quantity, and the ceiling it is compared against does not move.
HOME_USAGE_CACHE="${JUDGE_CACHE%/}/.home-usage-rest-gb"
HOME_USAGE_TTL_H="${BCF_HOME_USAGE_TTL_H:-1}"  # another campaign writes tens of GB per hour; a day-old estimate chose a 66 GB home download on 2026-09-07
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
SERVED_NAMES=()
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
  if [ -n "$PORT_BASE" ]; then
    PORT=$(( PORT_BASE + i ))
  else
    # Two picks in a row can in principle hand back the same free port, and the second
    # server would then fail to bind after its weights had downloaded. One cheap retry
    # loop is worth more than that discovery.
    PORT="$(bcf_pick_port)"
    while [[ " ${PORTS[*]:-} " == *" ${PORT} "* ]]; do PORT="$(bcf_pick_port)"; done
  fi
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
  # gpt-oss only: vLLM 0.28 renders its chat requests through openai_harmony, which
  # downloads its tiktoken vocab on first use INSIDE the request handler. On xgpk0 that
  # made the first chat completion of job 828627 an HTTP 500. Pull the download here,
  # where a failure is loud and costs no votes. NOT piped into ts: a pipe would run the
  # function in a subshell and its export of TIKTOKEN_RS_CACHE_DIR would never reach
  # this shell, so the server below would start without the cache that was just filled.
  case "$key" in
    gpt-oss-20b)
      bcf_prewarm_harmony > "${OUT_DIR}/harmony-prewarm-${key}.log" 2>&1
      prewarm_rc=$?
      ts < "${OUT_DIR}/harmony-prewarm-${key}.log" | tee -a "$LOG"
      if [ "$prewarm_rc" -ne 0 ]; then
        echo "[serve] REFUSING: ${key} harmony vocab prewarm failed (${prewarm_rc}); not starting a server whose first request would download it" | ts | tee -a "$LOG" >&2
        exit 14
      fi
      echo "[serve] ${key}: harmony vocab cached in ${TIKTOKEN_RS_CACHE_DIR}" | ts | tee -a "$LOG"
      ;;
  esac
  echo "[serve] ${key}: ${JUDGE_HF_ID} @ ${JUDGE_REVISION} on port ${PORT} (${JUDGE_SERVING_LINE}, util ${JUDGE_GPU_UTIL})" | ts | tee -a "$LOG"
  vllm serve "$JUDGE_HF_ID" "${VLLM_ARGS[@]}" >> "${OUT_DIR}/server-${key}.log" 2>&1 &
  SERVER_PIDS+=($!)
  PORTS+=("$PORT")
  KEYS+=("$key")
  # The name vLLM is serving under, kept per index: the readiness check below asks the
  # port for its model list and requires THIS string back.
  SERVED_NAMES+=("$JUDGE_HF_ID")
  i=$(( i + 1 ))
done

# --- 4. wait for every server --------------------------------------------------
DEADLINE=$(( $(date +%s) + SERVER_WAIT ))
for idx in "${!KEYS[@]}"; do
  key="${KEYS[$idx]}"; PORT="${PORTS[$idx]}"; pid="${SERVER_PIDS[$idx]}"
  # Ready means OUR server answers: the model list names this judge, the pid we started
  # is alive, and the process holding the port is that pid or a descendant. A port that
  # answers for anyone else is refused (13) instead of judged against.
  bcf_wait_server_ready "http://127.0.0.1:${PORT}/v1" "${SERVED_NAMES[$idx]}" "$pid" \
    "$DEADLINE" "${OUT_DIR}/server-${key}.log" "${OUT_DIR}/serve_ready-${key}.json" \
    | ts | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "[serve] ${key}: readiness failed with ${rc}" | ts | tee -a "$LOG"
    exit "$rc"
  fi
  echo "[serve] ${key} up on port ${PORT} (pid ${pid}), ownership checked" | ts | tee -a "$LOG"
done

# --- 4b. what this job served, on the record -----------------------------------
# judge_serve writes no run_meta.json of its own (the gate writes the vote records), so
# the serving facts that used to be unrecoverable after the fact, which port, which pid,
# and whether the listener was proved ours, land in their own file next to the votes.
BCF_SERVE_META_DIR="$OUT_DIR" BCF_SERVE_META_KEYS="${KEYS[*]}" python - <<'SERVEMETA' 2>&1 | ts | tee -a "$LOG"
import json, os, pathlib
out_dir = pathlib.Path(os.environ["BCF_SERVE_META_DIR"])
rows = []
for key in os.environ["BCF_SERVE_META_KEYS"].split():
    path = out_dir / f"serve_ready-{key}.json"
    if not path.exists():
        rows.append({"judge": key, "port_owner_check": "missing: no serve_ready file"})
        continue
    ready = json.loads(path.read_text())
    rows.append({
        "judge": key,
        "port": ready.get("port"),
        "server_pid": ready.get("server_pid"),
        "port_owner_check": ready.get("owner_check"),
        "port_owner_method": ready.get("owner_method"),
        "port_listener_pids": ready.get("listener_pids"),
        "served_model_list": ready.get("models"),
    })
meta = {"job_id": os.environ.get("SLURM_JOB_ID"), "servers": rows}
(out_dir / "serve_meta.json").write_text(json.dumps(meta, indent=2))
for row in rows:
    print(f"[serve] {row['judge']}: port={row.get('port')} pid={row.get('server_pid')} "
          f"owner_check={row.get('port_owner_check')}")
SERVEMETA

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
  if [ "$JURY_MODE" = "audit" ]; then
    # The SECONDARY sweep lane. Same served judge, same ports, same guard; the runner
    # rather than the gate, one cell of banked records rather than the synthetic corpus,
    # and --mode audit, which is section 6.4's sweep protocol: one run per item plus the
    # seeded 10 percent that gets three. --only-served-judges is what makes a
    # single-judge configuration possible at all: the panel rule routes three judges onto
    # a Gemma or Llama subject and only one of them is served here, so without it the
    # runner refuses on the first unreachable judge. Every row still records the full
    # routed panel and the labels are marked partial.
    # The completion marker is run_summary.json, which the runner writes on the one path
    # that means the scoring finished.
    STAGE_MARKER="${RUN_OUT}/run_summary.json"
    python -m experiments.jury.runner \
      --items "$SWEEP_ITEMS" --out "$RUN_OUT" \
      --substrate "$SUBSTRATE" --cue-family "$CUE" \
      --seed "$SEED" --concurrency "$CONCURRENCY" --num-predict "$NUM_PREDICT" \
      --q1-prompt "$variant" --mode audit --only-served-judges \
      --secondary --configuration "$ITEMS_CONFIGURATION" \
      $RESUME_FLAG $ALL_ROWS_FLAG "${LIMIT_FLAG[@]}" "${LINE_FLAGS[@]}" "${JUDGE_FLAGS[@]}" 2>&1 | ts | tee -a "$LOG"
  else
    STAGE_MARKER="${RUN_OUT}/gate_report.json"
    python -m experiments.jury.gate \
      --items "$GATE_ITEMS" --out "$RUN_OUT" \
      --substrate "$SUBSTRATE" --cue-family "$CUE" \
      --seed "$SEED" --concurrency "$CONCURRENCY" --num-predict "$NUM_PREDICT" \
      --q1-prompt "$variant" \
      $RESUME_FLAG $ALL_ROWS_FLAG $ECHO_STRIP_FLAG "${LIMIT_FLAG[@]}" "${LINE_FLAGS[@]}" "${JUDGE_FLAGS[@]}" 2>&1 | ts | tee -a "$LOG"
  fi
  st=${PIPESTATUS[0]}
  echo "[gate] Q1 variant ${variant} finished at $(date -Is) with status ${st}" | ts | tee -a "$LOG"
  # 1 is a FAILED THRESHOLD, which is a result. Anything above 1 is the job breaking.
  # A 0 or a 1 with no gate_report.json is neither: a Python traceback also exits 1, and
  # a gate that ran always wrote a report. bcf_gate_stage_status turns that case into 12.
  rec_st="$(bcf_gate_stage_status "$st" "$STAGE_MARKER")"
  if [ "$rec_st" != "$st" ]; then
    echo "[gate] variant ${variant} wrote no $(basename "$STAGE_MARKER"): crash, not a verdict (status ${st})" | ts | tee -a "$LOG"
  fi
  [ "$rec_st" -gt "$GATE_STATUS" ] && GATE_STATUS=$rec_st
  echo "$rec_st" > "${RUN_OUT}/exit_code.txt"
done
echo "[gate] all variants finished at $(date -Is), worst status ${GATE_STATUS}" | ts | tee -a "$LOG"

# Exit 0 when the gate RAN, whatever its verdict: a judge failing a threshold is a result,
# not a job failure, and mailing FAIL on it would train the operator to ignore the mail.
# The verdict lives in gate_report.json.
# 12 is a variant that crashed without writing a report, and it is never a zero here.
bcf_exit_guard_complete
if [ "$GATE_STATUS" -gt 1 ]; then
  exit "$GATE_STATUS"
fi
exit 0
