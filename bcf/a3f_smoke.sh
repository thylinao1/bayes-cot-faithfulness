#!/bin/bash
# The A3-final skeleton SMOKE RUN, submitted from the Mac, one job, one MIG slice.
#
#   bcf/a3f_smoke.sh --dry-run     print the exact sbatch line, submit nothing
#   bcf/a3f_smoke.sh               submit it
#   bcf/a3f_smoke.sh status        the job's queue row and its results directory
#
# WHAT IT IS. Thirty ARC items, stated-hint, Qwen3-8B at the element-10 pinned revision,
# on ONE a100-40 GRES, which on this cluster is a MIG 3g.40gb slice of an A100 80GB
# (env.sh, job 825248). It is NOT a powered cell and it is NOT at any pre-registered n:
# n is 30, the smallest number that exercises every arm, and its purpose is to run
# ruling R1's determinism preflight against a real server for the first time and to
# produce the first measured chain-level lambda for A3.5.
#
# TWELVE ARMS, not nine: the nine A2 arms of the wave manifests
# (replay placebo direct twostep filler curves transplant anchor specificity) plus the
# three the wave manifests do not carry, `sampling` (element 9.2), `repeat-curves`
# (element 8.3) and `chain-repeats` (ruling R4). BCF_N_ARMS=12 makes a truncated
# --export list exit 9 instead of running a short arm list that looks deliberate.
#
# Arm names travel under '+', never a comma: sbatch --export is itself comma separated
# and a comma silently truncates the value (job 826024 was submitted that way).
#
# The a100-40 per-user cap is 8 cards and the total gpu cap is 12, both counting every
# campaign on the account. This asks for one card of one type the alta campaign is not
# using; the denominators are printed live before the submit anyway.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RETRY="${REPO}/bcf/ssh_retry.sh"

MODEL="Qwen/Qwen3-8B"
# Element 10 roster revision, the same sha every a100-40 wave manifest pins for this row.
# serve_and_run.sbatch exits 8 rather than running if the Hub resolves anything else.
REVISION="b968826d9c46dd6066d109eabc6255188de91218"
SUBSTRATE="arc_challenge"
CUE="stated-hint"
N_ITEMS=30
ARMS="replay+placebo+direct+twostep+filler+curves+sampling+repeat-curves+chain-repeats+transplant+anchor+specificity"
N_ARMS=12
JOB_NAME="bcf-a3f-skel"
OUT_ROOT='$HOME/bcf/results/a3f-skeleton'
REPO_TREE='$HOME/bcf/repo-jury'

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  status)
    bash "$RETRY" --wall 60 --tries 3 -- "
      squeue --me -o '%.10i %.20j %.9P %.9T %.11M %.11l %.20b %R' | grep -E 'JOBID|a3f' || echo '(no a3f job in the queue)';
      d=\$HOME/bcf/results/a3f-skeleton/qwen3-8b/arc_challenge/stated-hint;
      echo '--- results dir ---'; ls -la \$d 2>/dev/null || echo '(not created yet)';
      echo '--- exit code ---'; cat \$d/exit_code.txt 2>/dev/null || echo '(none)';
      echo '--- preflight ---'; cat \$d/determinism_preflight.json 2>/dev/null || echo '(none)'"
    exit $? ;;
  "") : ;;
  *) echo "usage: a3f_smoke.sh [--dry-run|status]" >&2; exit 2 ;;
esac

EXPORTS="ALL"
EXPORTS="${EXPORTS},BCF_REPO=${REPO_TREE}"
EXPORTS="${EXPORTS},BCF_ENV_SH=${REPO_TREE}/bcf/env.sh"
EXPORTS="${EXPORTS},BCF_MODEL=${MODEL}"
EXPORTS="${EXPORTS},BCF_REVISION=${REVISION}"
EXPORTS="${EXPORTS},BCF_SUBSTRATE=${SUBSTRATE}"
EXPORTS="${EXPORTS},BCF_CUE=${CUE}"
EXPORTS="${EXPORTS},BCF_N_ITEMS=${N_ITEMS}"
EXPORTS="${EXPORTS},BCF_CURVE_CAP=${N_ITEMS}"
EXPORTS="${EXPORTS},BCF_ARMS=${ARMS}"
EXPORTS="${EXPORTS},BCF_N_ARMS=${N_ARMS}"
EXPORTS="${EXPORTS},BCF_TP=1"
EXPORTS="${EXPORTS},BCF_CONCURRENCY=32"
EXPORTS="${EXPORTS},BCF_BATCH_INVARIANT=1"
EXPORTS="${EXPORTS},BCF_PREFLIGHT=1"
EXPORTS="${EXPORTS},BCF_OUT_ROOT=${OUT_ROOT}"

SBATCH_CMD="sbatch --job-name=${JOB_NAME} --partition=gpu-long --gpus=a100-40 --exclude=xgpj0"
SBATCH_CMD="${SBATCH_CMD} --mem=64G --cpus-per-task=8 --time=04:00:00"
# The --export value is left UNQUOTED on the remote command line on purpose: it carries
# no spaces, and $HOME inside it has to be expanded by the CLUSTER's shell. Inside
# single quotes sbatch would receive the four characters $HOME and BCF_REPO would name
# a directory that does not exist, which serve_and_run.sbatch cannot detect because it
# sources that path before it checks anything.
SBATCH_CMD="${SBATCH_CMD} --export=${EXPORTS} ${REPO_TREE}/bcf/serve_and_run.sbatch"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[a3f] DRY-RUN, nothing submitted:"
  echo "$SBATCH_CMD"
  exit 0
fi

echo "[a3f] live denominators, ALL campaigns on this account, before the submit:"
bash "$RETRY" --wall 60 --tries 3 -- '
  echo "  jobs in system : $(squeue --me -h -t RUNNING,PENDING | wc -l | tr -d " ")/32"
  echo "  jobs running   : $(squeue --me -h -t RUNNING | wc -l | tr -d " ")/16"
  squeue --me -h -t RUNNING,PENDING -O "tres-alloc:200" | tr "," "\n" \
    | grep -oE "gres/gpu:[a-z0-9-]+=[0-9]+" | awk -F= "{s[\$1]+=\$2} END {for (k in s) print \"  \"k\"=\"s[k]}"' \
  || { echo "[a3f] REFUSING: ssh soc does not answer. Nothing was submitted." >&2; exit 99; }

echo "[a3f] submitting:"
echo "$SBATCH_CMD"
bash "$RETRY" --wall 90 --tries 3 -- "cd ${REPO_TREE} && ${SBATCH_CMD}" \
  || { echo "[a3f] REFUSING: the sbatch did not go through. Nothing is queued for it." >&2; exit 99; }
