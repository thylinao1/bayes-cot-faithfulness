#!/bin/bash
# The ONLY thing that should call sbatch on serve_and_run.sbatch.
#
# Four independent caps have to hold before a wave goes out, and they fail differently:
#
#   1. MaxSubmitJobs=32 jobs in system, account-wide (NUS-COMPUTE.md 1.4). The 33rd
#      sbatch is REJECTED at submit time, not queued, and a fire-and-forget loop drops
#      the rest silently. Array elements count individually.
#   2. The per-user MaxTRESPU cap for the GPU TYPE, which counts EVERY campaign on the
#      account (a100-40 8, a100-80 4, h100-96 2, h100-47 4, h200-141 1). Going over does
#      not reject; it makes this campaign's jobs queue behind another campaign's, which
#      is worse because it is invisible. Re-verified live 2026-09-07.
#   3. The per-user TOTAL gpu cap of 12 across all types. A wave can fit its own pool and
#      still be held by cards this account is spending somewhere else.
#   4. The CONTRACT.md pool split by job type, so the sweep cannot eat the judges' cards.
#
# Two things it refuses BEFORE any of that, because no amount of waiting fixes them:
# a tensor-parallel size larger than the cards on one node of that type, and a wall
# clock longer than the partition's own ceiling.
#
# This script refuses on any of them, prints every denominator, and never cancels
# anything.
#
#   bcf/wave.sh --type sweep --gpu-type a100-40 --check-only  bcf/waves/a100-40-01.tsv
#   bcf/wave.sh --type sweep --gpu-type a100-40               bcf/waves/a100-40-01.tsv
#
# cells.tsv: tab-separated, one cell per line, '#' comments allowed
#   MODEL <tab> SUBSTRATE <tab> CUE [<tab> KEY=VALUE ...]
#   Qwen/Qwen3-8B	arc_challenge	stated-hint	BCF_TP=1	BCF_N_ITEMS=1500
#
# KEY=VALUE fields are exported to the job, EXCEPT MEM= and CPUS=, which become sbatch
# --mem and --cpus-per-task flags. That distinction is load bearing: Slurm's default on
# this account is 3G of host RAM and 1 CPU, a vLLM engine core dies under it with only
# "Engine core initialization failed" and no root cause in the server log, and the
# evidence is only legible in sacct (job 825536, State=OUT_OF_MEMORY, ReqMem=3G,
# MaxRSS=6.3G). Exporting a MEM variable would not have reserved a byte.

set -uo pipefail

MAX_SUBMIT_JOBS=32          # Slurm association limit, verified live 2026-07-31

# QOS MaxTRESPU per GPU type. Re-verified live 2026-09-06 and again 2026-09-07 with
#   sacctmgr show qos normal format=Name,MaxTRESPU
# and it is TIGHTER than NUS-COMPUTE.md 1.4 and CONTRACT.md's original record:
#   cpu=1024, a100-40=8, a100-80=4, h100-47=4, h100-96=2, h200-141=1, h200-71=1,
#   nv=8, gpu=12 (total across all types)
declare -A USER_CAP=( [a100-40]=8 [a100-80]=4 [h100-47]=4 [h100-96]=2 [h200-141]=1 [nv]=8 )
GPU_TOTAL_CAP=12

# Cards PER NODE, from `sinfo -o "%n %G"` on 2026-09-06. This is the constraint that
# decides whether a tensor-parallel job is submittable at all: EVERY a100-80 node
# carries exactly ONE card, so `--gpus-per-node=a100-80:2` is rejected with
# "Requested node configuration is not available" no matter how free the cluster is.
#   a100-80  11 nodes x 1     a100-40  8 x 1 and 6 x 2
#   h100-96  11 nodes x 2     h100-47  10 x 4      h200-141  1 node x 4
declare -A CARDS_PER_NODE=( [a100-40]=2 [a100-80]=1 [h100-47]=4 [h100-96]=2 [h200-141]=4 )

# Partition per GPU type and that partition's wall ceiling, from `sinfo -o "%P|%n|%G"`
# and `sinfo -o "%P %l"` on 2026-09-07. xgpk0 is the ONLY h200-141 node and it sits in
# partition `gpu` alone, whose wall is 3 hours; an h200 job submitted to gpu-long is
# rejected with "Requested node configuration is not available", which reads like a busy
# cluster and is not. A cell longer than the ceiling has to run with --resume across
# several jobs, and this script says so instead of letting sbatch refuse it.
declare -A PARTITION=( [a100-40]=gpu-long [a100-80]=gpu-long [h100-47]=gpu-long \
                       [h100-96]=gpu-long [h200-141]=gpu )
declare -A PARTITION_WALL_HOURS=( [gpu-long]=72 [gpu]=3 [test]=72 )
# The --time this script asks for when the caller gives none. It must sit UNDER the
# partition ceiling: serve_and_run.sbatch's own header says 24:00:00, and submitting that
# to partition `gpu` (the only partition the single h200-141 node is in) is rejected with
# "Requested node configuration is not available", which reads like a busy cluster and is
# not. Verified with sbatch --test-only on 2026-09-07: the identical h200 request is
# ACCEPTED at --time=02:50:00 and REJECTED at the header's 24 h.
declare -A PARTITION_DEFAULT_WALL=( [gpu-long]=48:00:00 [gpu]=02:50:00 [test]=48:00:00 )

GPU_TYPE="a100-80"

# CONTRACT.md card budget by job type, CORRECTED 2026-09-07 for the live caps. Cards,
# not jobs: a TP=2 job takes two. The a100-80 sweep figure is 2, not 4, because the two
# a100-80 judge servers hold the other 2 of that pool's 4 whenever they are up.
declare -A TYPE_BUDGET=( [sweep_a100-40]=8 [sweep_a100-80]=2 [sweep_h100-96]=2 \
                         [sweep_h200-141]=1 \
                         [judge_a100-80]=2 [judge_h200-141]=1 \
                         [probe_a100-40]=1 [probe_a100-80]=1 \
                         [ladder_a100-40]=1 [ladder_a100-80]=1 \
                         [skel_a100-40]=3 [skel_a100-80]=3 [skel_h100-96]=2 \
                         [enrich_a100-40]=8 )
KNOWN_TYPES="sweep judge probe ladder skel enrich"

# The POOL GUARD (added 2026-09-07 with the enrich type). The three checks above count,
# in order, every campaign's RUNNING cards of this type, every campaign's RUNNING cards
# of every type, and THIS job type's own cards counting pending. None of them counts this
# campaign's OTHER job types' PENDING cards, and that is the hole the enrichment line
# opens: a100-40 carries the sweep at 8 of 8 and the enrichment pass wants the same pool,
# so an enrich wave submitted while 8 sweep cells sit PENDING would put 16 jobs against a
# cap of 8. They would not be rejected; they would queue behind this campaign's own work,
# which is the invisible failure this script exists to prevent.
# It is ON for the enrich type, which has no history to change, and available to every
# other type through BCF_WAVE_POOL_GUARD=1. It is deliberately not on by default for the
# existing types: it would refuse waves those types submit today (an a100-80 sweep wave
# beside three pending bcf-jury cards, for one), and turning it on for them is a
# CONTRACT.md decision rather than this script's.
POOL_GUARD_TYPES="enrich"

JOB_TYPE="sweep"
JOB_NAME=""
WALL_TIME=""      # overrides the sbatch header's --time when set
DRY_RUN=0
SBATCH_SCRIPT=""  # resolved from the synced tree below unless --sbatch overrides it
CELLS=""

# --- the tree a powered job reads ----------------------------------------------
# DECISION-LOG 2026-09-07 03:58 ruling (b): a powered job NEVER reads ~/bcf/repo. Live
# jobs read that path, and rsyncing a lane's worktree over it while a job is mid-arm
# swaps the code under a running measurement. One immutable tree per planning commit
# instead: ~/bcf/repo-<short sha>, synced from THE COMMIT (not the working tree) before
# submission and not touched afterwards, with BCF_REPO pointing at it.
SYNC_ROOT="${BCF_SYNC_ROOT:-$HOME/bcf}"
FORBIDDEN_TREE="${SYNC_ROOT}/repo"
REPO_TREE=""       # --repo-tree, for a tree already synced by an earlier wave
SHA_LEN=12

# Accept both `--opt value` and `--opt=value`, and REFUSE an unrecognised flag instead
# of letting it fall through to the positional. The old catch-all swallowed
# `--time=24:00:00` as the cells file and then overwrote it with the real one, so an
# explicit wall clock was silently dropped and the header's 24 h was used.
while [ $# -gt 0 ]; do
  arg="$1"
  val=""
  case "$arg" in
    --*=*) val="${arg#*=}"; arg="${arg%%=*}"; shift ;;
    --type|--job-name|--time|--gpu-type|--sbatch|--repo-tree) val="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
  case "$arg" in
    --type) JOB_TYPE="$val" ;;
    --job-name) JOB_NAME="$val" ;;
    --time) WALL_TIME="$val" ;;
    --gpu-type) GPU_TYPE="$val" ;;
    --sbatch) SBATCH_SCRIPT="$val" ;;
    --repo-tree) REPO_TREE="$val" ;;
    --dry-run|--check-only) DRY_RUN=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    -*) echo "unknown option '${arg}'" >&2; exit 2 ;;
    *) CELLS="$arg" ;;
  esac
done

[ -n "$CELLS" ] || { echo "usage: wave.sh [--type sweep] [--gpu-type a100-40] [--check-only] cells.tsv" >&2; exit 2; }
[ -f "$CELLS" ] || { echo "no such cells file: $CELLS" >&2; exit 2; }
case " $KNOWN_TYPES " in
  *" $JOB_TYPE "*) : ;;
  *) echo "unknown --type '$JOB_TYPE'; known: $KNOWN_TYPES" >&2; exit 2 ;;
esac
[ -n "${USER_CAP[$GPU_TYPE]:-}" ] || {
  echo "unknown --gpu-type '$GPU_TYPE'; known: ${!USER_CAP[*]}" >&2; exit 2; }
BUDGET_KEY="${JOB_TYPE}_${GPU_TYPE}"
[ -n "${TYPE_BUDGET[$BUDGET_KEY]:-}" ] || {
  echo "REFUSING: no CONTRACT budget for --type ${JOB_TYPE} on ${GPU_TYPE}." >&2
  echo "  Known pairs: ${!TYPE_BUDGET[*]}" >&2
  echo "  A pool with no stated split for this job type is a planning gap, not a default." >&2
  exit 2; }
GPU_USER_CAP=${USER_CAP[$GPU_TYPE]}
BUDGET=${TYPE_BUDGET[$BUDGET_KEY]}
PART=${PARTITION[$GPU_TYPE]}
PART_WALL=${PARTITION_WALL_HOURS[$PART]}

# The per-type card count below is a grep over job names, so an override that does not
# carry the type prefix would make its own cards invisible to the next wave's check.
if [ -n "$JOB_NAME" ] && [ "${JOB_NAME#bcf-${JOB_TYPE}-}" = "$JOB_NAME" ]; then
  echo "--job-name '$JOB_NAME' must start with 'bcf-${JOB_TYPE}-' or its cards go uncounted" >&2
  exit 2
fi

# --- the immutable tree for this planning commit --------------------------------
# Resolved BEFORE any cap check, because a wave that cannot name the tree its jobs will
# read has nothing to submit, and finding that out after the caps pass wastes the check.
WAVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(git -C "$WAVE_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
PLAN_SHA=""
DIRTY=""
if [ -n "$SRC_ROOT" ]; then
  PLAN_SHA="$(git -C "$SRC_ROOT" rev-parse --short=${SHA_LEN} HEAD 2>/dev/null || true)"
  DIRTY="$(git -C "$SRC_ROOT" status --porcelain 2>/dev/null | grep -v '^??' || true)"
fi

if [ -z "$REPO_TREE" ]; then
  if [ -z "$PLAN_SHA" ]; then
    echo "[wave] REFUSING: ${WAVE_DIR} is not inside a git checkout, so there is no"
    echo "[wave]   planning commit to name the tree after. A powered job has to read a"
    echo "[wave]   tree whose contents are pinned by a sha; pass --repo-tree explicitly"
    echo "[wave]   if you have already synced one."
    exit 1
  fi
  REPO_TREE="${SYNC_ROOT}/repo-${PLAN_SHA}"
fi

# The one path that is never the target, however it was arrived at.
REPO_TREE_ABS="${REPO_TREE%/}"
# A tooling copy outside any git checkout (the feeder's deployed copy) has no HEAD to
# name a planning commit. When it is handed an already synced tree, the tree's marker IS
# the pin: the tree is never rewritten, so its recorded commit is the code a job reads.
if [ -z "$PLAN_SHA" ] && [ -n "$REPO_TREE" ] && [ -f "${REPO_TREE_ABS}/.bcf_sync.json" ]; then
  PLAN_SHA="$(sed -n 's/.*"commit": "\([0-9a-f]*\)".*/\1/p' "${REPO_TREE_ABS}/.bcf_sync.json" | head -1)"
  [ -n "$PLAN_SHA" ] && echo "[wave] no checkout here; planning commit ${PLAN_SHA} taken from the synced tree's marker"
fi
if [ "$REPO_TREE_ABS" = "${FORBIDDEN_TREE%/}" ]; then
  echo "[wave] REFUSING: the target tree is ${REPO_TREE_ABS}, which is the shared"
  echo "[wave]   ~/bcf/repo that LIVE JOBS READ. Syncing over it swaps the code under a"
  echo "[wave]   running measurement (DECISION-LOG 2026-09-07 03:58 ruling (b))."
  echo "[wave]   The target must be ${SYNC_ROOT}/repo-<short sha of the planning commit>."
  exit 1
fi
if [ "${BCF_REPO:-}" = "${FORBIDDEN_TREE%/}" ]; then
  echo "[wave] REFUSING: BCF_REPO is set to ${BCF_REPO}, the shared tree live jobs read."
  echo "[wave]   Unset it; this script sets BCF_REPO to the tree it syncs."
  exit 1
fi
if [ -n "$DIRTY" ]; then
  echo "[wave] REFUSING: the checkout at ${SRC_ROOT} has uncommitted tracked changes, so"
  echo "[wave]   ${PLAN_SHA} does not describe the code a job would run. Commit first."
  echo "$DIRTY" | sed 's/^/[wave]     /'
  exit 1
fi

SYNC_MARKER="${REPO_TREE_ABS}/.bcf_sync.json"
echo "[wave] planning commit ${PLAN_SHA:-<none>} at ${SRC_ROOT:-<no checkout>}"
echo "[wave] jobs will read BCF_REPO=${REPO_TREE_ABS}"

sync_tree() {
  # The CODE comes from the commit, via git archive, so a file edited after the commit
  # cannot reach a powered job. The three substrate pools do NOT: they are gitignored
  # for licence reasons, so they are copied from the working tree and then verified
  # against the manifest that IS in the commit. A pool that does not match its committed
  # hashes stops the wave here.
  if [ -e "$SYNC_MARKER" ]; then
    if grep -q "\"commit\": \"${PLAN_SHA}\"" "$SYNC_MARKER" 2>/dev/null; then
      echo "[wave] ${REPO_TREE_ABS} already holds ${PLAN_SHA}; leaving it untouched"
      return 0
    fi
    echo "[wave] REFUSING: ${REPO_TREE_ABS} exists and was synced from a DIFFERENT commit"
    echo "[wave]   (see ${SYNC_MARKER}). A synced tree is never rewritten: a job may be"
    echo "[wave]   reading it. Sync the new commit to its own ${SYNC_ROOT}/repo-<sha>."
    return 1
  fi
  mkdir -p "$REPO_TREE_ABS" || return 1
  git -C "$SRC_ROOT" archive --format=tar "$PLAN_SHA" | tar -x -C "$REPO_TREE_ABS" || return 1
  local copied=0
  for pool in arc_challenge aqua_rat logiqa2 specificity_holdout toy_mcq; do
    if [ -f "${SRC_ROOT}/experiments/data/${pool}.json" ]; then
      cp "${SRC_ROOT}/experiments/data/${pool}.json" \
         "${REPO_TREE_ABS}/experiments/data/${pool}.json" || return 1
      copied=$(( copied + 1 ))
    fi
  done
  echo "[wave] copied ${copied} gitignored data file(s) into the tree"
  ( cd "$REPO_TREE_ABS" && python scripts/write_pool_manifest.py --check ) || {
    echo "[wave] REFUSING: the pools copied into ${REPO_TREE_ABS} do not match the"
    echo "[wave]   manifest committed at ${PLAN_SHA}."
    return 1
  }
  printf '{\n  "commit": "%s",\n  "source": "%s",\n  "synced_at": "%s",\n  "synced_by": "bcf/wave.sh"\n}\n' \
    "$PLAN_SHA" "$SRC_ROOT" "$(date -Is)" > "$SYNC_MARKER"
  echo "[wave] synced ${PLAN_SHA} -> ${REPO_TREE_ABS} (marker ${SYNC_MARKER})"
}

# --- read the cells ------------------------------------------------------------
mapfile -t ROWS < <(grep -vE '^[[:space:]]*(#|$)' "$CELLS")
N_CELLS=${#ROWS[@]}
[ "$N_CELLS" -gt 0 ] || { echo "cells file has no rows" >&2; exit 2; }

# Split a row on TABS only. `for f in $(... | tr '\t' '\n')` also splits on spaces, which
# silently shatters a value like `BCF_EXTRA_VLLM=--max-num-seqs 64` into two fields and
# exports a variable named `64`.
row_fields() { printf '%s' "$1" | tr '\t' '\n'; }

CARDS_WANTED=0
MAX_TP=1
MAX_HOURS=0
for row in "${ROWS[@]}"; do
  tp=1
  hours=0
  while IFS= read -r field; do
    case "$field" in
      BCF_TP=*) tp="${field#BCF_TP=}" ;;
      BCF_EXPECTED_HOURS=*) hours="${field#BCF_EXPECTED_HOURS=}" ;;
    esac
  done < <(row_fields "$row")
  CARDS_WANTED=$(( CARDS_WANTED + tp ))
  [ "$tp" -gt "$MAX_TP" ] && MAX_TP=$tp
  # Integer compare on the whole-hours part; this is a ceiling check, not accounting.
  h_int="${hours%%.*}"
  [ -z "$h_int" ] && h_int=0
  [ "$h_int" -gt "$MAX_HOURS" ] && MAX_HOURS=$h_int
done

echo "[wave] file ${CELLS}: ${N_CELLS} cell(s), ${CARDS_WANTED} card(s), max tp ${MAX_TP}"
echo "[wave] type ${JOB_TYPE} on ${GPU_TYPE}, partition ${PART} (wall ceiling ${PART_WALL} h)"

# --- refusal 1: physically impossible tensor-parallel request -------------------
PER_NODE=${CARDS_PER_NODE[$GPU_TYPE]:-1}
if [ "$MAX_TP" -gt "$PER_NODE" ]; then
  echo "[wave] REFUSING: a cell asks for tensor-parallel ${MAX_TP} on ${GPU_TYPE}, but"
  echo "[wave]   every ${GPU_TYPE} node carries only ${PER_NODE} card(s) (sinfo -o '%n %G')."
  echo "[wave]   No amount of waiting fixes this. Use a GPU type with >= ${MAX_TP} cards per node:"
  for g in "${!CARDS_PER_NODE[@]}"; do
    [ "${CARDS_PER_NODE[$g]}" -ge "$MAX_TP" ] && echo "[wave]     --gpu-type ${g} (${CARDS_PER_NODE[$g]} per node, cap ${USER_CAP[$g]})"
  done
  exit 1
fi

# --- wall clock: set it, cap it at the partition ceiling, and say what that costs
# A cell longer than the partition ceiling is NOT refused: serve_and_run.sbatch always
# passes --resume, so such a cell runs across several jobs and picks up where it stopped.
# What is refused is an explicit --time above the ceiling, because that is silently
# rejected by sbatch with a message about node configuration.
if [ -n "$WALL_TIME" ]; then
  req_h="${WALL_TIME%%:*}"
  case "$req_h" in *-*) req_h=$(( ${req_h%%-*} * 24 + ${req_h#*-} )) ;; esac
  if [ "$req_h" -gt "$PART_WALL" ]; then
    echo "[wave] REFUSING: --time ${WALL_TIME} is above partition ${PART}'s ceiling of ${PART_WALL} h."
    echo "[wave]   sbatch answers that with 'Requested node configuration is not available',"
    echo "[wave]   which looks like a busy cluster and is not."
    exit 1
  fi
else
  WALL_TIME="${PARTITION_DEFAULT_WALL[$PART]}"
fi
if [ "$MAX_HOURS" -ge "$PART_WALL" ]; then
  legs=$(( (MAX_HOURS + PART_WALL - 1) / PART_WALL ))
  echo "[wave] NOTE: the longest cell here is projected at ${MAX_HOURS} h and ${GPU_TYPE} lives"
  echo "[wave]   in partition ${PART}, ceiling ${PART_WALL} h, so it needs about ${legs} --resume"
  echo "[wave]   legs. serve_and_run.sbatch always passes --resume, so this is a resubmit"
  echo "[wave]   count, not a blocker. Wall requested: ${WALL_TIME}."
fi

# --- the injected counters, for proving the refusals ---------------------------
# Same shape as bcf/jury_wave.sh's, and it exists for the same reason: a refusal that has
# never been seen refuse is a claim, not a check. BCF_WAVE_FAKE_CARDS is a comma list of
# key=value over the four counters below, e.g. "a100-40=0,gpu=0,own=0,pool=0".
#
# It can NEVER cause a submission: any injected counter forces --check-only, refusing
# with exit 2 if the caller asked for a real wave. An injected count that made a real
# sbatch happen would be the worst possible bug in this file.
FAKE_ANY=0
[ -n "${BCF_WAVE_FAKE_INSYSTEM:-}" ] && FAKE_ANY=1
[ -n "${BCF_WAVE_FAKE_CARDS:-}" ] && FAKE_ANY=1
if [ "$FAKE_ANY" -eq 1 ] && [ "$DRY_RUN" -ne 1 ]; then
  echo "[wave] REFUSING: BCF_WAVE_FAKE_INSYSTEM / BCF_WAVE_FAKE_CARDS are set, which"
  echo "[wave]   makes every count below a fiction. They are for --check-only proofs."
  echo "[wave]   Nothing was submitted."
  exit 2
fi
fake_count() {  # $1 = key; echoes the injected value or nothing
  printf '%s\n' "${BCF_WAVE_FAKE_CARDS:-}" | tr ',' '\n' \
    | grep -E "^${1}=" | cut -d= -f2 | head -1
}

# --- check 3: jobs in system ---------------------------------------------------
IN_SYSTEM=$(squeue --me -h -t RUNNING,PENDING | wc -l | tr -d ' ')
[ -n "${BCF_WAVE_FAKE_INSYSTEM:-}" ] && IN_SYSTEM="$BCF_WAVE_FAKE_INSYSTEM"
AFTER=$(( IN_SYSTEM + N_CELLS ))
echo "[wave] jobs in system: ${IN_SYSTEM}/${MAX_SUBMIT_JOBS}; this wave adds ${N_CELLS} -> ${AFTER}/${MAX_SUBMIT_JOBS}"

# --- check 4: cards, account-wide, by total, and by type -----------------------
# The per-user cap is a CONCURRENCY limit, so it is measured on RUNNING allocations.
# Exceeding it does not reject anything: the extra jobs sit PENDING and Slurm starts
# them as cards free (NUS-COMPUTE.md 1.4). Pending array elements are reported too, but
# not counted: a `--array=0-9%2` shows its remaining elements as one PENDING row that
# holds no card and cannot start until a running sibling exits, so counting it as a full
# card refuses waves that would have fitted.
count_cards() {  # $1 = squeue state list, $2 = gres pattern
  { squeue --me -h -t "$1" -O "tres-alloc:200" \
    | tr ',' '\n' | grep -o "$2=[0-9]*" || true; } \
    | awk -F= '{s+=$2} END {print s+0}'
}
CARDS_ALL=$(count_cards RUNNING "gres/gpu:${GPU_TYPE}")
CARDS_PENDING=$(count_cards PENDING "gres/gpu:${GPU_TYPE}")
GPU_ALL=$(count_cards RUNNING "gres/gpu")
inj=$(fake_count "$GPU_TYPE");   [ -n "$inj" ] && CARDS_ALL="$inj"
inj=$(fake_count gpu);           [ -n "$inj" ] && GPU_ALL="$inj"
# This campaign's own cards of this type, identified by job name (bcf-<type>-...).
# Job names carry the prefix precisely so ownership is readable from squeue. This count
# DOES include PENDING: the CONTRACT split exists to stop this campaign over-committing
# itself, and a queued bcf job has already spent its share of the split.
CARDS_THIS_TYPE=$({ squeue --me -h -t RUNNING,PENDING -O "Name:80,tres-alloc:200" \
  | grep -E "^bcf-${JOB_TYPE}-" \
  | tr ',' '\n' | grep -o "gres/gpu:${GPU_TYPE}=[0-9]*" || true; } \
  | awk -F= '{s+=$2} END {print s+0}')
inj=$(fake_count own); [ -n "$inj" ] && CARDS_THIS_TYPE="$inj"
# The pool guard's own count: EVERY bcf-* job of this card type, running and pending,
# whatever its job type. This is the campaign's real footprint on the pool.
CARDS_POOL=$({ squeue --me -h -t RUNNING,PENDING -O "Name:80,tres-alloc:200" \
  | grep -E "^bcf-" \
  | tr ',' '\n' | grep -o "gres/gpu:${GPU_TYPE}=[0-9]*" || true; } \
  | awk -F= '{s+=$2} END {print s+0}')
inj=$(fake_count pool); [ -n "$inj" ] && CARDS_POOL="$inj"

echo "[wave] ${GPU_TYPE} cards RUNNING, ALL campaigns: ${CARDS_ALL}/${GPU_USER_CAP} (${CARDS_PENDING} more pending, not counted)"
echo "[wave] gpu cards RUNNING, ALL types, ALL campaigns: ${GPU_ALL}/${GPU_TOTAL_CAP}"
echo "[wave] ${GPU_TYPE} cards in use, bcf-${JOB_TYPE}: ${CARDS_THIS_TYPE}/${BUDGET} (CONTRACT.md split)"
echo "[wave] this wave wants ${CARDS_WANTED} card(s) across ${N_CELLS} job(s)"

REFUSE=""
[ "$AFTER" -gt "$MAX_SUBMIT_JOBS" ] && REFUSE="${REFUSE}
  jobs in system would reach ${AFTER} > ${MAX_SUBMIT_JOBS}; the excess sbatch calls are REJECTED, not queued"
[ $(( CARDS_ALL + CARDS_WANTED )) -gt "$GPU_USER_CAP" ] && REFUSE="${REFUSE}
  ${GPU_TYPE} cards would reach $(( CARDS_ALL + CARDS_WANTED )) > ${GPU_USER_CAP} (per-user cap, all campaigns)"
[ $(( GPU_ALL + CARDS_WANTED )) -gt "$GPU_TOTAL_CAP" ] && REFUSE="${REFUSE}
  total gpu cards would reach $(( GPU_ALL + CARDS_WANTED )) > ${GPU_TOTAL_CAP} (per-user cap across ALL types, all campaigns)"
[ $(( CARDS_THIS_TYPE + CARDS_WANTED )) -gt "$BUDGET" ] && REFUSE="${REFUSE}
  bcf-${JOB_TYPE} cards on ${GPU_TYPE} would reach $(( CARDS_THIS_TYPE + CARDS_WANTED )) > ${BUDGET} (CONTRACT.md ${JOB_TYPE} budget)"

POOL_GUARD=0
case " ${POOL_GUARD_TYPES} " in *" ${JOB_TYPE} "*) POOL_GUARD=1 ;; esac
[ "${BCF_WAVE_POOL_GUARD:-0}" = "1" ] && POOL_GUARD=1
if [ "$POOL_GUARD" -eq 1 ]; then
  echo "[wave] POOL GUARD on: every bcf-* card on ${GPU_TYPE}, running AND pending, all job types: ${CARDS_POOL}/${GPU_USER_CAP}"
  [ $(( CARDS_POOL + CARDS_WANTED )) -gt "$GPU_USER_CAP" ] && REFUSE="${REFUSE}
  this campaign's own ${GPU_TYPE} cards would reach $(( CARDS_POOL + CARDS_WANTED )) > ${GPU_USER_CAP} counting its PENDING jobs of every type (pool guard)"
fi

if [ -n "$REFUSE" ]; then
  echo "[wave] REFUSING to submit:${REFUSE}"
  echo "[wave] nothing was submitted and nothing was cancelled. Wait for jobs to drain, or split the wave."
  exit 1
fi

# --- the tree ------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  if [ -e "$SYNC_MARKER" ]; then
    echo "[wave] CHECK-ONLY would reuse the tree at ${REPO_TREE_ABS} (marker present)"
  else
    echo "[wave] CHECK-ONLY would sync ${SRC_ROOT} @ ${PLAN_SHA} -> ${REPO_TREE_ABS}"
    echo "[wave]   (git archive of the commit, plus the gitignored pools verified"
    echo "[wave]    against the manifest committed at ${PLAN_SHA}); nothing was written"
  fi
else
  sync_tree || exit 1
fi
[ -n "$SBATCH_SCRIPT" ] || SBATCH_SCRIPT="${REPO_TREE_ABS}/bcf/serve_and_run.sbatch"

# --- submit --------------------------------------------------------------------
echo "[wave] all checks pass; $([ "$DRY_RUN" -eq 1 ] && echo 'would submit' || echo 'submitting') ${N_CELLS} job(s)"
REJECTED=0
NOT_PLACEABLE=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r MODEL SUBSTRATE CUE REST <<< "$row"
  # The FULL basename, lowercased, non-alphanumerics folded to '-'. The old slug kept
  # only the text after the last hyphen, so Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B and
  # DeepSeek-R1-Distill-Llama-8B all became "8b" and three jobs in one wave carried the
  # same name. Job names are how ownership is read out of squeue, so a collision makes
  # the next wave's per-type card count unreadable.
  slug="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' \
          | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
  name="${JOB_NAME:-bcf-${JOB_TYPE}-${slug}-${SUBSTRATE}-${CUE}}"
  tp=1
  mem=""
  cpus=""
  exports="ALL,BCF_REPO=${REPO_TREE_ABS},BCF_ENV_SH=${REPO_TREE_ABS}/bcf/env.sh"
  exports="${exports},BCF_PLAN_COMMIT=${PLAN_SHA}"
  exports="${exports},BCF_MODEL=${MODEL},BCF_SUBSTRATE=${SUBSTRATE},BCF_CUE=${CUE}"
  while IFS= read -r field; do
    [ -n "$field" ] || continue
    case "$field" in
      MEM=*)  mem="${field#MEM=}";  continue ;;
      CPUS=*) cpus="${field#CPUS=}"; continue ;;
      BCF_TP=*) tp="${field#BCF_TP=}" ;;
    esac
    exports="${exports},${field}"
  done < <(row_fields "$REST")
  gpus_flag=(--gpus="${GPU_TYPE}")
  [ "$tp" -gt 1 ] && gpus_flag=(--nodes=1 --gpus-per-node="${GPU_TYPE}:${tp}")
  extra=(--time="$WALL_TIME")
  [ -n "$mem" ] && extra+=(--mem="$mem")
  [ -n "$cpus" ] && extra+=(--cpus-per-task="$cpus")
  cmd=(sbatch --job-name="$name" --partition="$PART" --exclude=xgpj0 \
       "${gpus_flag[@]}" "${extra[@]}" --export="$exports" "$SBATCH_SCRIPT")
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[wave] CHECK-ONLY would run:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    # Hand the exact command line to Slurm's own validator. --test-only parses every
    # flag, resolves the partition, the GRES and the excluded node, and estimates a start
    # time WITHOUT allocating anything. It is the difference between "this looks right"
    # and "sbatch accepts this": a --export value carrying a space, an --exclude naming a
    # node outside the partition, or a GRES the partition cannot satisfy all fail here
    # rather than at 3 a.m. on the first real wave.
    if [ "${BCF_SKIP_TEST_ONLY:-0}" != "1" ] && command -v sbatch >/dev/null 2>&1; then
      # ADVISORY, deliberately not fatal. Slurm answers both "this command line is
      # invalid" and "no node can hold this right now" with the same string,
      # "Requested node configuration is not available". Proven on 2026-09-07: the
      # tensor-parallel-2 h100-96 line failed that way while every h100-96 node had both
      # cards allocated, and the same line is what job 825253 is sitting PENDING on. So
      # a failure here is reported and counted, and the wave's exit code stays with the
      # caps and the physical refusals, which are unambiguous.
      if out=$(sbatch --test-only "${cmd[@]:1}" 2>&1); then
        echo "[wave]   sbatch --test-only ACCEPTS it: ${out}"
      else
        echo "[wave]   sbatch --test-only CANNOT PLACE IT NOW: ${out}"
        echo "[wave]   (that message covers both an invalid line and a full pool; check"
        echo "[wave]    the pool with sinfo before reading it as a configuration error)"
        NOT_PLACEABLE=$(( NOT_PLACEABLE + 1 ))
      fi
    fi
  else
    # Check sbatch's own status per row: printing "N submitted" when sbatch rejected a
    # job is how a wave silently loses cells.
    if ! "${cmd[@]}"; then
      echo "[wave] sbatch REJECTED the row for ${MODEL} ${SUBSTRATE} ${CUE}" >&2
      REJECTED=$(( REJECTED + 1 ))
    fi
  fi
done
if [ "$REJECTED" -gt 0 ]; then
  echo "[wave] ${REJECTED} of ${N_CELLS} row(s) were REJECTED by sbatch" >&2
  exit 1
fi
if [ "$NOT_PLACEABLE" -gt 0 ]; then
  echo "[wave] ${NOT_PLACEABLE} of ${N_CELLS} row(s) could not be placed right now (advisory)"
fi
echo "[wave] done ($([ "$DRY_RUN" -eq 1 ] && echo 'check-only, nothing submitted' || echo "${N_CELLS} submitted"))"
