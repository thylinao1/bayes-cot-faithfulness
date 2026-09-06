#!/bin/bash
# The only thing that should call sbatch on bcf/judge_serve.sbatch.
#
# Counts, before EVERY submission, across EVERY job on this Unix account (alta, bcf, any
# other campaign; the caps are per account, not per campaign, NUS-COMPUTE 1.9):
#
#   1. jobs in system        MaxSubmitJobs = 32. The 33rd sbatch is REJECTED, not queued.
#   2. jobs running          MaxJobs = 16.
#   3. cards per GPU type    QOS MaxTRESPU, re-verified live 2026-09-07:
#                            a100-40=8  a100-80=4  h100-47=4  h100-96=2  h200-141=1
#   4. cards in total        gpu = 12 across all types.
#
# Cards are counted over RUNNING **and** PENDING, not RUNNING alone. A pending job has
# already claimed its share: counting only running jobs lets a wave go out that then sits
# behind the account's own queue, which is the invisible failure mode.
#
# It never cancels anything and it never uses a wildcard scancel.
#
#   bcf/jury_wave.sh --dry-run judges.tsv
#   bcf/jury_wave.sh judges.tsv
#
# judges.tsv: tab separated, '#' comments allowed
#   JUDGE_KEYS <tab> GPU_TYPE <tab> WALLTIME [<tab> KEY=VALUE ...]
#   llama-3.3-70b-fp8	h200-141	05:00:00	BCF_CONCURRENCY=8
#   gemma-3-27b-it,gpt-oss-20b	a100-80	05:00:00
#
# Fail proof (this is run and recorded, not assumed):
#   BCF_WAVE_FAKE_CARDS="h200-141=1" bcf/jury_wave.sh --dry-run judges.tsv   -> exit 1

set -uo pipefail

MAX_SUBMIT_JOBS=32
MAX_RUNNING_JOBS=16
GPU_TOTAL_CAP=12
declare -A USER_CAP=( [a100-40]=8 [a100-80]=4 [h100-47]=4 [h100-96]=2 [h200-141]=1 [h200-71]=1 [nv]=8 )
declare -A CARDS_PER_NODE=( [a100-40]=2 [a100-80]=1 [h100-47]=4 [h100-96]=2 [h200-141]=4 )

# Which partition actually carries each card type, and that partition's wall ceiling.
# Found the hard way on 2026-09-07: xgpk0 is the ONLY h200-141 node on the cluster and it
# sits in `gpu` alone, not in `gpu-long`. An h200 job submitted to gpu-long is rejected
# outright with "Requested node configuration is not available", which reads like a busy
# cluster and is not. `gpu` caps the wall at 3 hours, so an h200 judge checkpoints and
# resumes across jobs instead of running one long one.
declare -A PARTITION_FOR=( [a100-40]=gpu-long [a100-80]=gpu-long [h100-47]=gpu-long [h100-96]=gpu-long [h200-141]=gpu [h200-71]=gpu [nv]=gpu-long )
declare -A PARTITION_MAX_H=( [gpu]=3 [gpu-long]=72 )

SBATCH_SCRIPT="${BCF_JURY_SBATCH:-$HOME/bcf/repo-jury/bcf/judge_serve.sbatch}"
GATE_ITEMS="${BCF_GATE_ITEMS:-$HOME/bcf/gate_items.jsonl}"
DRY_RUN=0
JOBS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --sbatch) SBATCH_SCRIPT="$2"; shift 2 ;;
    --items) GATE_ITEMS="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) JOBS="$1"; shift ;;
  esac
done

[ -n "$JOBS" ] || { echo "usage: jury_wave.sh [--dry-run] judges.tsv" >&2; exit 2; }
[ -f "$JOBS" ] || { echo "no such judges file: $JOBS" >&2; exit 2; }

mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "$JOBS")
N_JOBS=${#ROWS[@]}
[ "$N_JOBS" -gt 0 ] || { echo "judges file has no rows" >&2; exit 2; }

# --- live counts, every job on the account -------------------------------------
IN_SYSTEM=$(squeue --me -h -t RUNNING,PENDING | wc -l | tr -d ' ')
RUNNING=$(squeue --me -h -t RUNNING | wc -l | tr -d ' ')
[ -n "${BCF_WAVE_FAKE_INSYSTEM:-}" ] && IN_SYSTEM="$BCF_WAVE_FAKE_INSYSTEM"

count_cards() {  # $1 = gpu type
  local t="$1" injected
  injected=$(printf '%s\n' "${BCF_WAVE_FAKE_CARDS:-}" | tr ',' '\n' | grep -E "^${t}=" | cut -d= -f2 | head -1)
  if [ -n "$injected" ]; then echo "$injected"; return; fi
  { squeue --me -h -t RUNNING,PENDING -O "tres-alloc:200" \
      | tr ',' '\n' | grep -o "gres/gpu:${t}=[0-9]*" || true; } \
    | awk -F= '{s+=$2} END {print s+0}'
}

TOTAL_GPU=$({ squeue --me -h -t RUNNING,PENDING -O "tres-alloc:200" \
  | tr ',' '\n' | grep -oE "gres/gpu:[a-z0-9-]+=[0-9]+" || true; } \
  | awk -F= '{s+=$2} END {print s+0}')

# --- what this wave wants ------------------------------------------------------
declare -A WANT=()
CARDS_WANTED=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r KEYS GPU_TYPE WALL REST <<< "$row"
  [ -n "${USER_CAP[$GPU_TYPE]:-}" ] || {
    echo "[wave] REFUSING: unknown gpu type '${GPU_TYPE}' (known: ${!USER_CAP[*]})" >&2; exit 2; }
  tp=1
  for field in $(printf '%s\n' "${REST:-}" | tr '\t' '\n'); do
    case "$field" in BCF_TP=*) tp="${field#BCF_TP=}" ;; esac
  done
  per_node=${CARDS_PER_NODE[$GPU_TYPE]:-1}
  if [ "$tp" -gt "$per_node" ]; then
    echo "[wave] REFUSING: ${KEYS} asks tensor-parallel ${tp} on ${GPU_TYPE}, which has ${per_node} card(s) per node."
    echo "[wave]   No amount of waiting fixes that (sinfo -o '%n %G')."
    exit 1
  fi
  part="${PARTITION_FOR[$GPU_TYPE]:-gpu-long}"
  max_h="${PARTITION_MAX_H[$part]:-3}"
  wall_h="${WALL%%:*}"
  if [ "$((10#$wall_h))" -ge "$max_h" ]; then
    echo "[wave] REFUSING: ${KEYS} asks ${WALL} on ${GPU_TYPE}, which lives in partition"
    echo "[wave]   '${part}' with a ${max_h} hour ceiling. Slurm rejects the job at submit"
    echo "[wave]   time; it does not queue. Shorten the wall and resume across jobs."
    exit 1
  fi
  WANT[$GPU_TYPE]=$(( ${WANT[$GPU_TYPE]:-0} + tp ))
  CARDS_WANTED=$(( CARDS_WANTED + tp ))
done

echo "[wave] denominators, live, ALL campaigns on this account ($(date -Is)):"
echo "[wave]   jobs in system : ${IN_SYSTEM}/${MAX_SUBMIT_JOBS}   (this wave adds ${N_JOBS} -> $(( IN_SYSTEM + N_JOBS ))/${MAX_SUBMIT_JOBS})"
echo "[wave]   jobs running   : ${RUNNING}/${MAX_RUNNING_JOBS}"
echo "[wave]   gpu cards total: ${TOTAL_GPU}/${GPU_TOTAL_CAP}   (this wave adds ${CARDS_WANTED} -> $(( TOTAL_GPU + CARDS_WANTED ))/${GPU_TOTAL_CAP})"

REFUSE=""
[ $(( IN_SYSTEM + N_JOBS )) -gt "$MAX_SUBMIT_JOBS" ] && REFUSE="${REFUSE}
  jobs in system would reach $(( IN_SYSTEM + N_JOBS )) > ${MAX_SUBMIT_JOBS}; the excess sbatch calls are REJECTED, not queued"
[ $(( TOTAL_GPU + CARDS_WANTED )) -gt "$GPU_TOTAL_CAP" ] && REFUSE="${REFUSE}
  total gpu cards would reach $(( TOTAL_GPU + CARDS_WANTED )) > ${GPU_TOTAL_CAP} (QOS gpu cap, all types, all campaigns)"

for t in "${!WANT[@]}"; do
  used=$(count_cards "$t")
  cap=${USER_CAP[$t]}
  want=${WANT[$t]}
  echo "[wave]   ${t} cards    : ${used}/${cap}   (this wave adds ${want} -> $(( used + want ))/${cap})"
  if [ $(( used + want )) -gt "$cap" ]; then
    REFUSE="${REFUSE}
  ${t} cards would reach $(( used + want )) > ${cap} (per-user QOS cap, counting every campaign)"
  fi
done

if [ -n "$REFUSE" ]; then
  echo "[wave] REFUSING to submit:${REFUSE}"
  echo "[wave] nothing was submitted and nothing was cancelled. Wait for cards to free, or split the wave."
  exit 1
fi

echo "[wave] every cap holds; submitting ${N_JOBS} job(s)"
N_SUBMITTED=0
N_REJECTED=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r KEYS GPU_TYPE WALL REST <<< "$row"
  slug="$(echo "$KEYS" | tr ',' '-')"
  name="bcf-jury-${slug}"
  exports="ALL,BCF_JUDGES=${KEYS},BCF_GATE_ITEMS=${GATE_ITEMS}"
  tp=1
  # MEM= and CPUS= are sbatch flags, not exports. Slurm's default here is 3G of host RAM
  # and 1 CPU, which OOM-kills a vLLM engine core with no message in the server log
  # (job 825536; sacct said OUT_OF_MEMORY while the log said "initialization failed").
  mem_flag=()
  cpus_flag=()
  for field in $(printf '%s\n' "${REST:-}" | tr '\t' '\n'); do
    [ -n "$field" ] || continue
    case "$field" in
      MEM=*)  mem_flag=(--mem="${field#MEM=}"); continue ;;
      CPUS=*) cpus_flag=(--cpus-per-task="${field#CPUS=}"); continue ;;
    esac
    exports="${exports},${field}"
    case "$field" in BCF_TP=*) tp="${field#BCF_TP=}" ;; esac
  done
  gpus_flag="--gpus=${GPU_TYPE}"
  [ "$tp" -gt 1 ] && gpus_flag="--nodes=1 --gpus-per-node=${GPU_TYPE}:${tp}"
  part="${PARTITION_FOR[$GPU_TYPE]:-gpu-long}"
  cmd=(sbatch --job-name="$name" --partition="$part" $gpus_flag "${mem_flag[@]}" "${cpus_flag[@]}" --time="$WALL" --export="$exports" "$SBATCH_SCRIPT")
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[wave] DRY-RUN ${cmd[*]}"
    N_SUBMITTED=$(( N_SUBMITTED + 1 ))
  else
    # sbatch REJECTS rather than queues on a bad node configuration or the submit cap, and
    # a loop that ignores its status reports a submission that never happened.
    if "${cmd[@]}"; then
      N_SUBMITTED=$(( N_SUBMITTED + 1 ))
    else
      N_REJECTED=$(( N_REJECTED + 1 ))
      echo "[wave] SUBMIT REJECTED for ${name} (sbatch exited non-zero). Nothing queued for it."
    fi
  fi
done
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[wave] done (dry run, ${N_SUBMITTED} would be submitted, nothing was)"
  exit 0
fi
echo "[wave] done (${N_SUBMITTED} submitted, ${N_REJECTED} rejected of ${N_JOBS})"
[ "$N_REJECTED" -eq 0 ] || exit 1
