#!/bin/bash
# The ONLY thing that should call sbatch on bcf/logit_pass.sbatch.
#
#   bcf/logit_wave.sh --check-only bcf/waves/logit/logit-a100-40-01.tsv
#   bcf/logit_wave.sh              bcf/waves/logit/logit-a100-40-01.tsv
#
# Manifests live in bcf/waves/logit/, one directory DOWN from the sweep manifests, because
# bcf/check_wave_manifests.py and tests/test_wave_plan.py glob bcf/waves/*.tsv without
# recursing and count everything they find that is not enrich-, ladder- or -resub- as sweep
# cell rows of the 216-cell grid. A model row beside them would be counted as a cell.
#
# MANIFEST FORMAT, one MODEL per row (not one cell per row: one server does all of a
# model's cells), tab separated, '#' comments allowed:
#
#   MODEL <tab> KEY=VALUE <tab> KEY=VALUE ...
#   Qwen/Qwen3-8B	GPU=a100-40	BCF_CELLS=<dir>+<dir>	BCF_TP=1	MEM=64G	CPUS=8
#
# Every KEY=VALUE is exported to the job EXCEPT the five this script consumes itself:
#   GPU=    the card type for the row (all rows in one manifest must agree; the caps below
#           are per pool and a mixed wave would need two sets of counts to be honest)
#   MEM=    becomes sbatch --mem. Exporting it would not reserve a byte: Slurm's default on
#           this account is 3G of host RAM, and a vLLM engine core dies under it with only
#           "Engine core initialization failed" in the server log (job 825536,
#           State=OUT_OF_MEMORY, ReqMem=3G, MaxRSS=6.3G)
#   CPUS=   becomes sbatch --cpus-per-task, same reasoning
#   TIME=   becomes sbatch --time for that row
#   BCF_EXPECTED_HOURS=  read for the wall-clock note, and also exported
#
# FIELD ORDER DOES NOT MATTER, and the last field is not special. Rows are split by
# `row_fields` in bcf/wave_lib.sh, which is the fixed version: the shape it replaced wrote
# no trailing newline, so a `while read` loop dropped every row's LAST field. That was live
# on 2026-09-08 (job 828564 lost its own BCF_OUT_SUBROOT and wrote into a cell of record
# before being caught). Put BCF_CELLS wherever it reads best.
#
# THE FOUR CAPS, which are wave.sh's and are checked the same way here. They are copied
# rather than called because bcf/wave.sh only knows the job types CONTRACT.md gives a split
# to, and `logit` is not one of them; a wave.sh invocation would be refused at its
# TYPE_BUDGET lookup. If a constant below ever disagrees with bcf/wave.sh, bcf/wave.sh is
# the one that is right.
#   1. MaxSubmitJobs=32 jobs in system, account-wide. The 33rd sbatch is REJECTED at submit
#      time, not queued.
#   2. The per-user MaxTRESPU cap for the GPU TYPE, counting EVERY campaign on the account.
#      Going over does not reject, it makes these jobs queue behind another campaign's,
#      which is worse because it is invisible.
#   3. The per-user TOTAL gpu cap of 12 across all types.
#   4. This job type's own budget. CONTRACT.md gives the logit pass NO split of its own, so
#      the budget here is 1 card, which is the ladder's and the probe's posture ("probe or
#      ladder 1"). One model at a time. Widening it is a CONTRACT.md decision and not this
#      script's, and the POOL GUARD is on by default for the same reason it is on for the
#      enrich type: a job type with no history cannot break one.
#
# It never cancels anything, and an injected counter can never cause a submission.

set -uo pipefail

MAX_SUBMIT_JOBS=32
GPU_TOTAL_CAP=12
JOB_TYPE="logit"
SHA_LEN=12

# QOS MaxTRESPU per GPU type, re-verified live 2026-09-07 with
#   sacctmgr show qos normal format=Name,MaxTRESPU
# Written as a case rather than a `declare -A` so this script also runs under the Mac's
# system bash 3.2, which has neither associative arrays nor mapfile.
user_cap() {
  case "$1" in
    a100-40) echo 8 ;; a100-80) echo 4 ;; h100-47) echo 4 ;;
    h100-96) echo 2 ;; h200-141) echo 1 ;; *) echo "" ;;
  esac
}
# Cards PER NODE, from `sinfo -o "%n %G"` on 2026-09-06. This decides whether a
# tensor-parallel request is submittable at all.
cards_per_node() {
  case "$1" in
    a100-40) echo 2 ;; a100-80) echo 1 ;; h100-47) echo 4 ;;
    h100-96) echo 2 ;; h200-141) echo 4 ;; *) echo 1 ;;
  esac
}
partition_for() {
  case "$1" in h200-141) echo gpu ;; *) echo gpu-long ;; esac
}
partition_wall_hours() {
  case "$1" in gpu) echo 3 ;; *) echo 72 ;; esac
}
# The --time asked for when a row gives none. It must sit UNDER the partition ceiling: the
# single h200-141 node is in partition `gpu`, whose wall is 3 hours, and a longer request
# is rejected with "Requested node configuration is not available", which reads like a busy
# cluster and is not.
partition_default_wall() {
  case "$1" in gpu) echo 02:50:00 ;; *) echo 08:00:00 ;; esac
}
# CONTRACT.md gives this job type no split. 1 is the deliberate conservative reading.
type_budget() {
  case "$1" in
    a100-40|a100-80|h100-47|h100-96|h200-141) echo 1 ;; *) echo "" ;;
  esac
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bcf/wave_lib.sh
. "${HERE}/wave_lib.sh"

GPU_TYPE=""
WALL_TIME=""
DRY_RUN=0
REPO_TREE=""
SBATCH_SCRIPT=""
MANIFEST=""
MAX_ROWS=0

while [ $# -gt 0 ]; do
  arg="$1"; val=""
  case "$arg" in
    --*=*) val="${arg#*=}"; arg="${arg%%=*}"; shift ;;
    --gpu-type|--time|--repo-tree|--sbatch|--manifest|--max) val="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
  case "$arg" in
    --gpu-type) GPU_TYPE="$val" ;;
    --time) WALL_TIME="$val" ;;
    --repo-tree) REPO_TREE="$val" ;;
    --sbatch) SBATCH_SCRIPT="$val" ;;
    --manifest) MANIFEST="$val" ;;
    --max) MAX_ROWS="$val" ;;
    --check-only|--dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    -*) echo "unknown option '${arg}'" >&2; exit 2 ;;
    *) MANIFEST="$arg" ;;
  esac
done

[ -n "$MANIFEST" ] || { echo "usage: logit_wave.sh [--check-only] models.tsv" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no such manifest: $MANIFEST" >&2; exit 2; }
case "$(basename "$MANIFEST")" in
  logit-*) : ;;
  *) echo "[logit-wave] REFUSING: ${MANIFEST} is not named logit-*.tsv. The prefix and the" >&2
     echo "[logit-wave]   bcf/waves/logit/ directory are both what keep a model row out of" >&2
     echo "[logit-wave]   the 216-cell grid count in bcf/check_wave_manifests.py." >&2
     exit 2 ;;
esac

# --- read the rows (no mapfile: bash 3.2) ---------------------------------------
ROWS=()
while IFS= read -r line; do ROWS+=("$line"); done \
  < <(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST")
N_ROWS=${#ROWS[@]}
[ "$N_ROWS" -gt 0 ] || { echo "manifest has no rows" >&2; exit 2; }

# --- parse them, and refuse what no amount of waiting fixes ----------------------
CARDS_WANTED=0
MAX_TP=1
MAX_HOURS=0
ROW_GPU=""
SEEN_MODELS=""
for row in "${ROWS[@]}"; do
  model="$(printf '%s' "$row" | cut -f1)"
  tp=1
  hours=0
  cells=""
  gpu=""
  while IFS= read -r field; do
    case "$field" in
      GPU=*) gpu="${field#GPU=}" ;;
      BCF_TP=*) tp="${field#BCF_TP=}" ;;
      BCF_CELLS=*) cells="${field#BCF_CELLS=}" ;;
      BCF_EXPECTED_HOURS=*) hours="${field#BCF_EXPECTED_HOURS=}" ;;
    esac
  done < <(row_fields "$row")
  if [ -z "$model" ] || [ -z "$cells" ]; then
    echo "[logit-wave] REFUSING: a row carries no MODEL or no BCF_CELLS, so its job would" >&2
    echo "[logit-wave]   serve a model with nothing to read: ${row}" >&2
    exit 1
  fi
  case "$cells" in
    *,*|*" "*)
      echo "[logit-wave] REFUSING: BCF_CELLS for ${model} contains a comma or a space." >&2
      echo "[logit-wave]   sbatch --export is a COMMA-separated list, so the value would be" >&2
      echo "[logit-wave]   truncated at the first comma and the rest read as variable names" >&2
      echo "[logit-wave]   to inherit, silently. Separate cell directories with '+'." >&2
      exit 1 ;;
  esac
  case " ${SEEN_MODELS} " in
    *" ${model} "*)
      echo "[logit-wave] REFUSING: ${model} appears twice. Two jobs augmenting one model's" >&2
      echo "[logit-wave]   cells would race on the same sidecar files." >&2
      exit 1 ;;
  esac
  SEEN_MODELS="${SEEN_MODELS} ${model}"
  [ -z "$gpu" ] && gpu="$GPU_TYPE"
  if [ -z "$gpu" ]; then
    echo "[logit-wave] REFUSING: ${model} names no GPU= and no --gpu-type was given." >&2
    exit 1
  fi
  if [ -n "$ROW_GPU" ] && [ "$gpu" != "$ROW_GPU" ]; then
    echo "[logit-wave] REFUSING: this manifest mixes ${ROW_GPU} and ${gpu}. Every cap below" >&2
    echo "[logit-wave]   is per pool, so a mixed wave would print one set of counts against" >&2
    echo "[logit-wave]   two pools. Split it into one manifest per card type." >&2
    exit 1
  fi
  ROW_GPU="$gpu"
  CARDS_WANTED=$(( CARDS_WANTED + tp ))
  [ "$tp" -gt "$MAX_TP" ] && MAX_TP=$tp
  h_int="${hours%%.*}"
  [ -z "$h_int" ] && h_int=0
  [ "$h_int" -gt "$MAX_HOURS" ] && MAX_HOURS=$h_int
done
GPU_TYPE="$ROW_GPU"

GPU_USER_CAP="$(user_cap "$GPU_TYPE")"
[ -n "$GPU_USER_CAP" ] || {
  echo "unknown GPU type '${GPU_TYPE}'; known: a100-40 a100-80 h100-47 h100-96 h200-141" >&2
  exit 2; }
BUDGET="$(type_budget "$GPU_TYPE")"
PART="$(partition_for "$GPU_TYPE")"
PART_WALL="$(partition_wall_hours "$PART")"
PER_NODE="$(cards_per_node "$GPU_TYPE")"

echo "[logit-wave] file ${MANIFEST}: ${N_ROWS} model row(s), ${CARDS_WANTED} card(s), max tp ${MAX_TP}"
echo "[logit-wave] type ${JOB_TYPE} on ${GPU_TYPE}, partition ${PART} (wall ceiling ${PART_WALL} h)"

if [ "$MAX_TP" -gt "$PER_NODE" ]; then
  echo "[logit-wave] REFUSING: a row asks for tensor-parallel ${MAX_TP} on ${GPU_TYPE}, but"
  echo "[logit-wave]   every ${GPU_TYPE} node carries only ${PER_NODE} card(s). No amount of"
  echo "[logit-wave]   waiting fixes this."
  exit 1
fi
if [ -n "$WALL_TIME" ]; then
  req_h="${WALL_TIME%%:*}"
  case "$req_h" in *-*) req_h=$(( ${req_h%%-*} * 24 + ${req_h#*-} )) ;; esac
  if [ "$req_h" -gt "$PART_WALL" ]; then
    echo "[logit-wave] REFUSING: --time ${WALL_TIME} is above partition ${PART}'s ceiling of ${PART_WALL} h."
    echo "[logit-wave]   sbatch answers that with 'Requested node configuration is not"
    echo "[logit-wave]   available', which looks like a busy cluster and is not."
    exit 1
  fi
else
  WALL_TIME="$(partition_default_wall "$PART")"
fi
if [ "$MAX_HOURS" -ge "$PART_WALL" ]; then
  echo "[logit-wave] NOTE: the longest row here is projected at ${MAX_HOURS} h against a"
  echo "[logit-wave]   ${PART_WALL} h ceiling. experiments/logit_pass.py resumes from its own"
  echo "[logit-wave]   checkpoint, so that is a resubmit count and not a blocker."
fi

# --- the immutable tree a powered job reads --------------------------------------
# DECISION-LOG 2026-09-07 03:58 ruling (b): a powered job NEVER reads ~/bcf/repo. Live jobs
# read that path, and rsyncing a lane's worktree over it while a job is mid-read swaps the
# code under a running measurement. One tree per planning commit instead.
SYNC_ROOT="${BCF_SYNC_ROOT:-$HOME/bcf}"
FORBIDDEN_TREE="${SYNC_ROOT}/repo"
SRC_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)"
PLAN_SHA=""
DIRTY=""
if [ -n "$SRC_ROOT" ]; then
  PLAN_SHA="$(git -C "$SRC_ROOT" rev-parse --short=${SHA_LEN} HEAD 2>/dev/null || true)"
  DIRTY="$(git -C "$SRC_ROOT" status --porcelain 2>/dev/null | grep -v '^??' || true)"
fi
if [ -z "$REPO_TREE" ]; then
  if [ -z "$PLAN_SHA" ]; then
    echo "[logit-wave] REFUSING: ${HERE} is not inside a git checkout, so there is no"
    echo "[logit-wave]   planning commit to name the tree after. Pass --repo-tree."
    exit 1
  fi
  REPO_TREE="${SYNC_ROOT}/repo-${PLAN_SHA}"
fi
REPO_TREE_ABS="${REPO_TREE%/}"
if [ -z "$PLAN_SHA" ] && [ -f "${REPO_TREE_ABS}/.bcf_sync.json" ]; then
  PLAN_SHA="$(sed -n 's/.*"commit": "\([0-9a-f]*\)".*/\1/p' "${REPO_TREE_ABS}/.bcf_sync.json" | head -1)"
  [ -n "$PLAN_SHA" ] && echo "[logit-wave] planning commit ${PLAN_SHA} taken from the synced tree's marker"
fi
if [ "$REPO_TREE_ABS" = "${FORBIDDEN_TREE%/}" ]; then
  echo "[logit-wave] REFUSING: the target tree is ${REPO_TREE_ABS}, the shared ~/bcf/repo"
  echo "[logit-wave]   that LIVE JOBS READ."
  exit 1
fi
if [ "${BCF_REPO:-}" = "${FORBIDDEN_TREE%/}" ]; then
  echo "[logit-wave] REFUSING: BCF_REPO is set to ${BCF_REPO}, the shared tree live jobs read."
  exit 1
fi
if [ -n "$DIRTY" ]; then
  echo "[logit-wave] REFUSING: the checkout at ${SRC_ROOT} has uncommitted tracked changes,"
  echo "[logit-wave]   so ${PLAN_SHA} does not describe the code a job would run. Commit first."
  echo "$DIRTY" | sed 's/^/[logit-wave]     /'
  exit 1
fi
SYNC_MARKER="${REPO_TREE_ABS}/.bcf_sync.json"
echo "[logit-wave] planning commit ${PLAN_SHA:-<none>} at ${SRC_ROOT:-<no checkout>}"
echo "[logit-wave] jobs will read BCF_REPO=${REPO_TREE_ABS}"

sync_tree() {
  # The CODE comes from the COMMIT via git archive, so a file edited after the commit
  # cannot reach a powered job. The substrate pools are gitignored for licence reasons, so
  # they are copied from the working tree and verified against the committed manifest.
  if [ -e "$SYNC_MARKER" ]; then
    if grep -q "\"commit\": \"${PLAN_SHA}\"" "$SYNC_MARKER" 2>/dev/null; then
      echo "[logit-wave] ${REPO_TREE_ABS} already holds ${PLAN_SHA}; leaving it untouched"
      return 0
    fi
    echo "[logit-wave] REFUSING: ${REPO_TREE_ABS} was synced from a DIFFERENT commit (see"
    echo "[logit-wave]   ${SYNC_MARKER}). A synced tree is never rewritten: a job may be"
    echo "[logit-wave]   reading it."
    return 1
  fi
  mkdir -p "$REPO_TREE_ABS" || return 1
  git -C "$SRC_ROOT" archive --format=tar "$PLAN_SHA" | tar -x -C "$REPO_TREE_ABS" || return 1
  copied=0
  for pool in arc_challenge aqua_rat logiqa2 specificity_holdout toy_mcq; do
    if [ -f "${SRC_ROOT}/experiments/data/${pool}.json" ]; then
      cp "${SRC_ROOT}/experiments/data/${pool}.json" \
         "${REPO_TREE_ABS}/experiments/data/${pool}.json" || return 1
      copied=$(( copied + 1 ))
    fi
  done
  echo "[logit-wave] copied ${copied} gitignored data file(s) into the tree"
  ( cd "$REPO_TREE_ABS" && python scripts/write_pool_manifest.py --check ) || {
    echo "[logit-wave] REFUSING: the pools copied into ${REPO_TREE_ABS} do not match the"
    echo "[logit-wave]   manifest committed at ${PLAN_SHA}."
    return 1
  }
  printf '{\n  "commit": "%s",\n  "source": "%s",\n  "synced_at": "%s",\n  "synced_by": "bcf/logit_wave.sh"\n}\n' \
    "$PLAN_SHA" "$SRC_ROOT" "$(date -Is)" > "$SYNC_MARKER"
  echo "[logit-wave] synced ${PLAN_SHA} -> ${REPO_TREE_ABS} (marker ${SYNC_MARKER})"
}

# --- the injected counters, for proving the refusals -----------------------------
# Same shape and the same rule as bcf/wave.sh: a refusal that has never been seen refuse is
# a claim, not a check. Any injected counter FORCES --check-only.
FAKE_ANY=0
[ -n "${BCF_WAVE_FAKE_INSYSTEM:-}" ] && FAKE_ANY=1
[ -n "${BCF_WAVE_FAKE_CARDS:-}" ] && FAKE_ANY=1
if [ "$FAKE_ANY" -eq 1 ] && [ "$DRY_RUN" -ne 1 ]; then
  echo "[logit-wave] REFUSING: BCF_WAVE_FAKE_INSYSTEM / BCF_WAVE_FAKE_CARDS are set, which"
  echo "[logit-wave]   makes every count below a fiction. They are for --check-only proofs."
  exit 2
fi
fake_count() {
  printf '%s\n' "${BCF_WAVE_FAKE_CARDS:-}" | tr ',' '\n' \
    | grep -E "^${1}=" | cut -d= -f2 | head -1
}

count_cards() {  # $1 = squeue state list, $2 = gres pattern
  { squeue --me -h -t "$1" -O "tres-alloc:200" \
    | tr ',' '\n' | grep -o "$2=[0-9]*" || true; } \
    | awk -F= '{s+=$2} END {print s+0}'
}
count_named_cards() {  # $1 = job-name prefix regex
  { squeue --me -h -t RUNNING,PENDING -O "Name:80,tres-alloc:200" \
    | grep -E "$1" \
    | tr ',' '\n' | grep -o "gres/gpu:${GPU_TYPE}=[0-9]*" || true; } \
    | awk -F= '{s+=$2} END {print s+0}'
}

IN_SYSTEM=$(squeue --me -h -t RUNNING,PENDING | wc -l | tr -d ' ')
[ -n "${BCF_WAVE_FAKE_INSYSTEM:-}" ] && IN_SYSTEM="$BCF_WAVE_FAKE_INSYSTEM"
AFTER=$(( IN_SYSTEM + N_ROWS ))
CARDS_ALL=$(count_cards RUNNING "gres/gpu:${GPU_TYPE}")
CARDS_PENDING=$(count_cards PENDING "gres/gpu:${GPU_TYPE}")
GPU_ALL=$(count_cards RUNNING "gres/gpu")
CARDS_THIS_TYPE=$(count_named_cards "^bcf-${JOB_TYPE}-")
CARDS_POOL=$(count_named_cards "^bcf-")
inj=$(fake_count "$GPU_TYPE"); [ -n "$inj" ] && CARDS_ALL="$inj"
inj=$(fake_count gpu);         [ -n "$inj" ] && GPU_ALL="$inj"
inj=$(fake_count own);         [ -n "$inj" ] && CARDS_THIS_TYPE="$inj"
inj=$(fake_count pool);        [ -n "$inj" ] && CARDS_POOL="$inj"

echo "[logit-wave] jobs in system: ${IN_SYSTEM}/${MAX_SUBMIT_JOBS}; this wave adds ${N_ROWS} -> ${AFTER}/${MAX_SUBMIT_JOBS}"
echo "[logit-wave] ${GPU_TYPE} cards RUNNING, ALL campaigns: ${CARDS_ALL}/${GPU_USER_CAP} (${CARDS_PENDING} more pending, not counted)"
echo "[logit-wave] gpu cards RUNNING, ALL types, ALL campaigns: ${GPU_ALL}/${GPU_TOTAL_CAP}"
echo "[logit-wave] ${GPU_TYPE} cards in use, bcf-${JOB_TYPE}: ${CARDS_THIS_TYPE}/${BUDGET} (no CONTRACT split for this type; 1 is the probe/ladder posture)"
echo "[logit-wave] POOL GUARD on: every bcf-* card on ${GPU_TYPE}, running AND pending, all job types: ${CARDS_POOL}/${GPU_USER_CAP}"

REFUSE=""
[ "$AFTER" -gt "$MAX_SUBMIT_JOBS" ] && REFUSE="${REFUSE}
  jobs in system would reach ${AFTER} > ${MAX_SUBMIT_JOBS}; the excess sbatch calls are REJECTED, not queued"
[ $(( CARDS_ALL + CARDS_WANTED )) -gt "$GPU_USER_CAP" ] && REFUSE="${REFUSE}
  ${GPU_TYPE} cards would reach $(( CARDS_ALL + CARDS_WANTED )) > ${GPU_USER_CAP} (per-user cap, all campaigns)"
[ $(( GPU_ALL + CARDS_WANTED )) -gt "$GPU_TOTAL_CAP" ] && REFUSE="${REFUSE}
  total gpu cards would reach $(( GPU_ALL + CARDS_WANTED )) > ${GPU_TOTAL_CAP} (per-user cap across ALL types)"
[ $(( CARDS_THIS_TYPE + CARDS_WANTED )) -gt "$BUDGET" ] && REFUSE="${REFUSE}
  bcf-${JOB_TYPE} cards on ${GPU_TYPE} would reach $(( CARDS_THIS_TYPE + CARDS_WANTED )) > ${BUDGET} (this job type's budget)"
[ $(( CARDS_POOL + CARDS_WANTED )) -gt "$GPU_USER_CAP" ] && REFUSE="${REFUSE}
  this campaign's own ${GPU_TYPE} cards would reach $(( CARDS_POOL + CARDS_WANTED )) > ${GPU_USER_CAP} counting its PENDING jobs of every type (pool guard)"

if [ -n "$REFUSE" ]; then
  echo "[logit-wave] REFUSING to submit:${REFUSE}"
  echo "[logit-wave] nothing was submitted and nothing was cancelled. Wait for jobs to drain,"
  echo "[logit-wave]   or submit one model row at a time."
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  if [ -e "$SYNC_MARKER" ]; then
    echo "[logit-wave] CHECK-ONLY would reuse the tree at ${REPO_TREE_ABS} (marker present)"
  else
    echo "[logit-wave] CHECK-ONLY would sync ${SRC_ROOT} @ ${PLAN_SHA} -> ${REPO_TREE_ABS}; nothing was written"
  fi
else
  sync_tree || exit 1
fi
[ -n "$SBATCH_SCRIPT" ] || SBATCH_SCRIPT="${REPO_TREE_ABS}/bcf/logit_pass.sbatch"

# --- submit ---------------------------------------------------------------------
echo "[logit-wave] all checks pass; $([ "$DRY_RUN" -eq 1 ] && echo 'would submit' || echo 'submitting') ${N_ROWS} job(s)"
REJECTED=0
NOT_PLACEABLE=0
SUBMITTED=0
LIMIT="$N_ROWS"
if [ "$MAX_ROWS" -gt 0 ] && [ "$MAX_ROWS" -lt "$LIMIT" ]; then LIMIT="$MAX_ROWS"; fi
i=0
for row in "${ROWS[@]}"; do
  i=$(( i + 1 ))
  [ "$i" -gt "$LIMIT" ] && break
  MODEL="$(printf '%s' "$row" | cut -f1)"
  # The FULL basename lowercased, non-alphanumerics folded to '-'. A slug that kept only
  # the text after the last hyphen made Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B and
  # DeepSeek-R1-Distill-Llama-8B all "8b", and job names are how ownership is read out of
  # squeue by the count above.
  slug="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' \
          | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
  name="bcf-${JOB_TYPE}-${slug}"
  tp=1
  mem=""
  cpus=""
  row_time=""
  exports="ALL,BCF_REPO=${REPO_TREE_ABS},BCF_ENV_SH=${REPO_TREE_ABS}/bcf/env.sh"
  exports="${exports},BCF_PLAN_COMMIT=${PLAN_SHA},BCF_MODEL=${MODEL}"
  first=1
  while IFS= read -r field; do
    [ -n "$field" ] || continue
    if [ "$first" -eq 1 ]; then first=0; continue; fi   # field 1 is MODEL
    case "$field" in
      GPU=*)  continue ;;
      MEM=*)  mem="${field#MEM=}";  continue ;;
      CPUS=*) cpus="${field#CPUS=}"; continue ;;
      TIME=*) row_time="${field#TIME=}"; continue ;;
      BCF_TP=*) tp="${field#BCF_TP=}" ;;
    esac
    exports="${exports},${field}"
  done < <(row_fields "$row")
  gpus_flag=(--gpus="${GPU_TYPE}")
  [ "$tp" -gt 1 ] && gpus_flag=(--nodes=1 --gpus-per-node="${GPU_TYPE}:${tp}")
  extra=(--time="${row_time:-$WALL_TIME}")
  [ -n "$mem" ] && extra+=(--mem="$mem")
  [ -n "$cpus" ] && extra+=(--cpus-per-task="$cpus")
  cmd=(sbatch --job-name="$name" --partition="$PART" --exclude=xgpj0 \
       "${gpus_flag[@]}" "${extra[@]}" --export="$exports" "$SBATCH_SCRIPT")
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[logit-wave] CHECK-ONLY would run:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    # Slurm's own validator: it parses every flag, resolves the partition, the GRES and the
    # excluded node, and estimates a start time without allocating anything. ADVISORY, and
    # deliberately not fatal: Slurm answers "this line is invalid" and "no node is free
    # right now" with the same string.
    if [ "${BCF_SKIP_TEST_ONLY:-0}" != "1" ] && command -v sbatch >/dev/null 2>&1; then
      if out=$(sbatch --test-only "${cmd[@]:1}" 2>&1); then
        echo "[logit-wave]   sbatch --test-only ACCEPTS it: ${out}"
      else
        echo "[logit-wave]   sbatch --test-only CANNOT PLACE IT NOW: ${out}"
        NOT_PLACEABLE=$(( NOT_PLACEABLE + 1 ))
      fi
    fi
  else
    if "${cmd[@]}"; then
      SUBMITTED=$(( SUBMITTED + 1 ))
    else
      echo "[logit-wave] sbatch REJECTED the row for ${MODEL}" >&2
      REJECTED=$(( REJECTED + 1 ))
    fi
  fi
done

if [ "$REJECTED" -gt 0 ]; then
  echo "[logit-wave] ${REJECTED} of ${N_ROWS} row(s) were REJECTED by sbatch" >&2
  exit 1
fi
if [ "$NOT_PLACEABLE" -gt 0 ]; then
  echo "[logit-wave] ${NOT_PLACEABLE} of ${N_ROWS} row(s) could not be placed right now (advisory)"
fi
echo "[logit-wave] done ($([ "$DRY_RUN" -eq 1 ] && echo 'check-only, nothing submitted' || echo "${SUBMITTED} submitted"))"
exit 0
