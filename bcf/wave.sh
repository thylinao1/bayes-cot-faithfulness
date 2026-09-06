#!/bin/bash
# The ONLY thing that should call sbatch on serve_and_run.sbatch.
#
# Two independent caps have to hold before a wave goes out, and they fail differently:
#
#   1. MaxSubmitJobs=32 jobs in system, account-wide (NUS-COMPUTE.md 1.4). The 33rd
#      sbatch is REJECTED at submit time, not queued, and a fire-and-forget loop drops
#      the rest silently. Array elements count individually.
#   2. The CONTRACT.md a100-80 split (sweep 4, judges 2, probe/ladder 1, reserve 1)
#      inside a per-user concurrency cap of 8 that counts EVERY campaign on the
#      account. Going over does not reject; it makes this campaign's jobs queue behind
#      another campaign's, which is worse because it is invisible.
#
# This script refuses on either, prints both denominators, and never cancels anything.
#
#   bcf/wave.sh --type sweep --dry-run  cells.tsv
#   bcf/wave.sh --type sweep            cells.tsv
#
# cells.tsv: tab-separated, one cell per line, '#' comments allowed
#   MODEL <tab> SUBSTRATE <tab> CUE [<tab> KEY=VALUE ...]
#   Qwen/Qwen3-8B	arc_challenge	stated-hint	BCF_TP=1	BCF_N_ITEMS=500

set -euo pipefail

MAX_SUBMIT_JOBS=32          # Slurm association limit, verified live 2026-07-31

# QOS MaxTRESPU per GPU type. Re-verified live 2026-09-06 with
#   sacctmgr show qos normal format=Name,MaxTRESPU
# and it is TIGHTER than NUS-COMPUTE.md 1.4 and CONTRACT.md record. The live line is:
#   cpu=1024, a100-40=8, a100-80=4, h100-47=4, h100-96=2, h200-141=1, h200-71=1,
#   nv=8, gpu=12 (total across all types)
# CONTRACT.md's a100-80 split (sweep 4 + judges 2 + probe 1 + reserve 1 = 8) and its
# "h200-141 (cap 2)" line both assume the old, looser numbers and do not fit.
declare -A USER_CAP=( [a100-40]=8 [a100-80]=4 [h100-47]=4 [h100-96]=2 [h200-141]=1 [nv]=8 )
GPU_TOTAL_CAP=12

# Cards PER NODE, from `sinfo -o "%n %G"` on 2026-09-06. This is the constraint that
# decides whether a tensor-parallel job is submittable at all: EVERY a100-80 node
# carries exactly ONE card, so `--gpus-per-node=a100-80:2` is rejected with
# "Requested node configuration is not available" no matter how free the cluster is.
#   a100-80  11 nodes x 1     a100-40  8 x 1 and 6 x 2
#   h100-96  11 nodes x 2     h100-47  10 x 4      h200-141  1 node x 4
declare -A CARDS_PER_NODE=( [a100-40]=2 [a100-80]=1 [h100-47]=4 [h100-96]=2 [h200-141]=4 )

GPU_TYPE="a100-80"

# CONTRACT.md a100-80 card budget by job type. Cards, not jobs: a TP=2 job takes two.
declare -A TYPE_BUDGET=( [sweep]=4 [judge]=2 [probe]=1 [ladder]=1 [reserve]=1 )
# Phase 1 only. 'skel' is CARVED OUT of the sweep's 4 cards (1 for the skeleton cell,
# 2 for the tensor-parallel-2 serving test), never additional to them: the skeleton and
# the sweep do not run at the same time. The account-wide check below is what actually
# stops the two budgets from being spent together.
TYPE_BUDGET[skel]=3

JOB_TYPE="sweep"
JOB_NAME=""
WALL_TIME=""      # overrides the sbatch header's --time when set
DRY_RUN=0
SBATCH_SCRIPT="${BCF_SBATCH:-$HOME/bcf/repo/bcf/serve_and_run.sbatch}"
CELLS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --type) JOB_TYPE="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --time) WALL_TIME="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --sbatch) SBATCH_SCRIPT="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) CELLS="$1"; shift ;;
  esac
done

[ -n "$CELLS" ] || { echo "usage: wave.sh [--type sweep] [--dry-run] cells.tsv" >&2; exit 2; }
[ -f "$CELLS" ] || { echo "no such cells file: $CELLS" >&2; exit 2; }
[ -n "${TYPE_BUDGET[$JOB_TYPE]:-}" ] || {
  echo "unknown --type '$JOB_TYPE'; known: ${!TYPE_BUDGET[*]}" >&2; exit 2; }
[ -n "${USER_CAP[$GPU_TYPE]:-}" ] || {
  echo "unknown --gpu-type '$GPU_TYPE'; known: ${!USER_CAP[*]}" >&2; exit 2; }
GPU_USER_CAP=${USER_CAP[$GPU_TYPE]}
# The per-type card count below is a grep over job names, so an override that does not
# carry the type prefix would make its own cards invisible to the next wave's check.
if [ -n "$JOB_NAME" ] && [ "${JOB_NAME#bcf-${JOB_TYPE}-}" = "$JOB_NAME" ]; then
  echo "--job-name '$JOB_NAME' must start with 'bcf-${JOB_TYPE}-' or its cards go uncounted" >&2
  exit 2
fi

# --- read the cells ------------------------------------------------------------
mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "$CELLS")
N_CELLS=${#ROWS[@]}
[ "$N_CELLS" -gt 0 ] || { echo "cells file has no rows" >&2; exit 2; }

CARDS_WANTED=0
MAX_TP=1
for row in "${ROWS[@]}"; do
  tp=1
  for field in $(printf '%s\n' "$row" | tr '\t' '\n'); do
    case "$field" in BCF_TP=*) tp="${field#BCF_TP=}" ;; esac
  done
  CARDS_WANTED=$(( CARDS_WANTED + tp ))
  [ "$tp" -gt "$MAX_TP" ] && MAX_TP=$tp
done

# Refuse a physically impossible request HERE, with the reason, rather than letting
# sbatch answer "Requested node configuration is not available" and leave you guessing
# whether the cluster is merely busy.
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

# --- check 1: jobs in system ---------------------------------------------------
IN_SYSTEM=$(squeue --me -h -t RUNNING,PENDING | wc -l | tr -d ' ')
AFTER=$(( IN_SYSTEM + N_CELLS ))
echo "[wave] jobs in system: ${IN_SYSTEM}/${MAX_SUBMIT_JOBS}; this wave adds ${N_CELLS} -> ${AFTER}/${MAX_SUBMIT_JOBS}"

# --- check 2: a100-80 cards, account-wide and by type --------------------------
# The per-user cap is a CONCURRENCY limit, so it is measured on RUNNING allocations.
# Exceeding it does not reject anything: the extra jobs sit PENDING and Slurm starts
# them as cards free (NUS-COMPUTE.md 1.4). Pending array elements are reported too, but
# not counted: a `--array=0-9%2` shows its remaining elements as one PENDING row that
# holds no card and cannot start until a running sibling exits, so counting it as a full
# card refuses waves that would have fitted.
CARDS_ALL=$({ squeue --me -h -t RUNNING -O "tres-alloc:200" \
  | tr ',' '\n' | grep -o "gres/gpu:${GPU_TYPE}=[0-9]*" || true; } \
  | awk -F= '{s+=$2} END {print s+0}')
CARDS_PENDING=$({ squeue --me -h -t PENDING -O "tres-alloc:200" \
  | tr ',' '\n' | grep -o "gres/gpu:${GPU_TYPE}=[0-9]*" || true; } \
  | awk -F= '{s+=$2} END {print s+0}')
# This campaign's own cards of this type, identified by job name (bcf-<type>-...).
# Job names carry the prefix precisely so ownership is readable from squeue. This count
# DOES include PENDING: the CONTRACT split exists to stop this campaign over-committing
# itself, and a queued bcf job has already spent its share of the split.
CARDS_THIS_TYPE=$({ squeue --me -h -t RUNNING,PENDING -O "Name:80,tres-alloc:200" \
  | grep -E "^bcf-${JOB_TYPE}-" \
  | tr ',' '\n' | grep -o "gres/gpu:${GPU_TYPE}=[0-9]*" || true; } \
  | awk -F= '{s+=$2} END {print s+0}')
BUDGET=${TYPE_BUDGET[$JOB_TYPE]}

echo "[wave] ${GPU_TYPE} cards RUNNING, ALL campaigns: ${CARDS_ALL}/${GPU_USER_CAP} (${CARDS_PENDING} more pending, not counted)"
echo "[wave] ${GPU_TYPE} cards in use, bcf-${JOB_TYPE}: ${CARDS_THIS_TYPE}/${BUDGET} (CONTRACT.md split)"
echo "[wave] this wave wants ${CARDS_WANTED} card(s) across ${N_CELLS} job(s)"

REFUSE=""
[ "$AFTER" -gt "$MAX_SUBMIT_JOBS" ] && REFUSE="${REFUSE}
  jobs in system would reach ${AFTER} > ${MAX_SUBMIT_JOBS}; the excess sbatch calls are REJECTED, not queued"
[ $(( CARDS_ALL + CARDS_WANTED )) -gt "$GPU_USER_CAP" ] && REFUSE="${REFUSE}
  ${GPU_TYPE} cards would reach $(( CARDS_ALL + CARDS_WANTED )) > ${GPU_USER_CAP} (per-user cap, all campaigns)"
[ $(( CARDS_THIS_TYPE + CARDS_WANTED )) -gt "$BUDGET" ] && REFUSE="${REFUSE}
  bcf-${JOB_TYPE} cards would reach $(( CARDS_THIS_TYPE + CARDS_WANTED )) > ${BUDGET} (CONTRACT.md ${JOB_TYPE} budget)"

if [ -n "$REFUSE" ]; then
  echo "[wave] REFUSING to submit:${REFUSE}"
  echo "[wave] nothing was submitted and nothing was cancelled. Wait for jobs to drain, or split the wave."
  exit 1
fi

# --- submit --------------------------------------------------------------------
echo "[wave] both checks pass; submitting ${N_CELLS} job(s)"
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r MODEL SUBSTRATE CUE REST <<< "$row"
  slug="$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | sed 's/^.*-//')"
  name="${JOB_NAME:-bcf-${JOB_TYPE}-${slug}-${SUBSTRATE}}"
  tp=1
  exports="ALL,BCF_MODEL=${MODEL},BCF_SUBSTRATE=${SUBSTRATE},BCF_CUE=${CUE}"
  for field in $(printf '%s\n' "$REST" | tr '\t' '\n'); do
    [ -n "$field" ] || continue
    exports="${exports},${field}"
    case "$field" in BCF_TP=*) tp="${field#BCF_TP=}" ;; esac
  done
  gpus_flag="--gpus=${GPU_TYPE}"
  [ "$tp" -gt 1 ] && gpus_flag="--nodes=1 --gpus-per-node=${GPU_TYPE}:${tp}"
  time_flag=()
  [ -n "$WALL_TIME" ] && time_flag=(--time="$WALL_TIME")
  cmd=(sbatch --job-name="$name" $gpus_flag "${time_flag[@]}" --export="$exports" "$SBATCH_SCRIPT")
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[wave] DRY-RUN ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
done
echo "[wave] done ($([ "$DRY_RUN" -eq 1 ] && echo 'dry run, nothing submitted' || echo "${N_CELLS} submitted"))"
