#!/bin/bash
# The gpt-oss-20b serving line: one configuration per invocation, one job per
# configuration, submitted through the unmodified bcf/serve_and_run.sbatch.
#
# Wave-1 cell 826740 (a100-40 MIG 3g.40gb, VLLM_BATCH_INVARIANT=1) died at engine init
# with NotImplementedError from the vLLM mxfp4 MoE oracle. This script runs the three
# configurations that separate the two candidate causes: the compute capability of the
# card, and the batch-invariant flag.
#
#   bcf/gptoss_line.sh a100-40-flagoff     (a) diagnostic, exploratory by construction
#   bcf/gptoss_line.sh h100-47-flagon      (b) sm90 with the R1 pinned serving mode
#   bcf/gptoss_line.sh a100-80-flagon      (c) whole sm80 card, only if the cap admits it
#   bcf/gptoss_line.sh a100-40-flagoff-np4096   fourth row when 320 does not parse
#
# Every job carries --exclude=xgpj0, an explicit --mem=64G and --cpus-per-task=8, and
# VLLM_USE_FLASHINFER_SAMPLER=0. Results land under ~/bcf/results/gptoss-line-*, never in
# another lane's cell directory.
set -uo pipefail

CFG="${1:?usage: gptoss_line.sh <a100-40-flagoff|h100-47-flagon|a100-80-flagon|a100-40-flagoff-np4096>}"

REPO="$HOME/bcf/repo-gptoss"
MODEL="openai/gpt-oss-20b"
# Element 10 roster row 7 pins this revision; serve_and_run refuses on drift (exit 8).
REV="6cee5e81ee83917806bbde320786a8fb61efebee"
# gpt-oss is a reasoning model served through the harmony template with the
# openai_gptoss reasoning parser, so the analysis channel lands in reasoning_content and
# experiments/openai_client.py reads message.content only. The effort setting is the
# knob that decides whether anything reaches content within num_predict, so it is set
# explicitly and recorded rather than left at the template default.
TPL='{"reasoning_effort":"low"}'
NP=320

case "$CFG" in
  a100-40-flagoff)        GPUTYPE=a100-40 ; BI=0 ; PF=1 ; PORT=8010 ;;
  a100-40-flagoff-np4096) GPUTYPE=a100-40 ; BI=0 ; PF=0 ; PORT=8013 ; NP=4096 ;;
  a100-40-flagoff-acc)    GPUTYPE=a100-40 ; BI=0 ; PF=0 ; PORT=8014 ;;
  h100-47-flagon)         GPUTYPE=h100-47 ; BI=1 ; PF=1 ; PORT=8011 ;;
  a100-80-flagon)         GPUTYPE=a100-80 ; BI=1 ; PF=1 ; PORT=8012 ;;
  *) echo "unknown configuration '$CFG'" >&2; exit 2 ;;
esac

JOB="bcf-gptoss-${CFG}"
OUT_ROOT="$HOME/bcf/results/gptoss-line-${CFG}"
mkdir -p "$OUT_ROOT" "$HOME/bcf/logs"

# The per-user MaxTRESPU cap counts EVERY campaign on this account, so it is read from
# live RUNNING allocations rather than from this campaign's own bookkeeping.
declare -A USER_CAP=( [a100-40]=8 [a100-80]=4 [h100-47]=4 [h100-96]=2 [h200-141]=1 )
running() { squeue --me -h -t RUNNING -O "tres-alloc:200" | grep -o "gres/gpu:${1}=[0-9]*" \
            | cut -d= -f2 | paste -sd+ - | sed 's/^$/0/' | bc; }
HELD=$(running "$GPUTYPE"); CAP=${USER_CAP[$GPUTYPE]}
GPU_ALL=$(squeue --me -h -t RUNNING -O "tres-alloc:200" | grep -o 'gres/gpu=[0-9]*' \
          | cut -d= -f2 | paste -sd+ - | sed 's/^$/0/' | bc)
MINE=$(squeue --me -h -t RUNNING,PENDING -o "%j" | grep -c '^bcf-gptoss-' || true)
echo "[cap] ${GPUTYPE}: ${HELD} running of ${CAP}; gpu total ${GPU_ALL} of 12; this lane has ${MINE} job(s) in system"
if [ "$HELD" -ge "$CAP" ]; then
  echo "[cap] REFUSING: ${GPUTYPE} is at its per-user cap; a cap refusal is the correct outcome, never a scancel" >&2
  exit 3
fi
if [ "$GPU_ALL" -ge 12 ]; then
  echo "[cap] REFUSING: the account holds ${GPU_ALL} of 12 gpus" >&2; exit 3
fi
if [ "$MINE" -ge 2 ]; then
  echo "[cap] REFUSING: this lane already has ${MINE} jobs in system; at most two run at once" >&2; exit 3
fi

EXPORT="ALL"
EXPORT="${EXPORT},BCF_ENV_SH=${REPO}/bcf/env.sh,BCF_REPO=${REPO}"
EXPORT="${EXPORT},BCF_MODEL=${MODEL},BCF_REVISION=${REV}"
EXPORT="${EXPORT},BCF_SUBSTRATE=arc_challenge,BCF_CUE=stated-hint"
EXPORT="${EXPORT},BCF_N_ITEMS=30,BCF_TP=1,BCF_CONCURRENCY=32"
EXPORT="${EXPORT},BCF_NUM_PREDICT=${NP},BCF_CURVE_CAP=30"
EXPORT="${EXPORT},BCF_BATCH_INVARIANT=${BI},BCF_PREFLIGHT=${PF},BCF_PREFLIGHT_ITEMS=30"
# BCF_PREFLIGHT_LEVELS is deliberately NOT exported. determinism_preflight.py splits
# --levels on commas and sbatch --export is itself comma separated, so "1,32" sent
# that way would truncate to "1". The sbatch builds 1,${BCF_CONCURRENCY} inside the
# job from BCF_CONCURRENCY=32, which is the same value and cannot be truncated.
# One arm. The clean pass at [1/3] runs whatever the arm list says, and `direct` is the
# cheapest arm, so this is the clean pass plus the R1 preflight and nothing else.
EXPORT="${EXPORT},BCF_ARMS=direct,BCF_N_ARMS=1"
EXPORT="${EXPORT},BCF_PORT=${PORT},BCF_OUT_ROOT=${OUT_ROOT}"
EXPORT="${EXPORT},VLLM_USE_FLASHINFER_SAMPLER=0"
EXPORT="${EXPORT},BCF_TEMPLATE_KWARGS=${TPL}"

set -x
sbatch --job-name="$JOB" \
       --partition=gpu-long \
       --gpus="${GPUTYPE}" \
       --exclude=xgpj0 \
       --mem=64G \
       --cpus-per-task=8 \
       --time=02:00:00 \
       --output="$HOME/bcf/logs/%x-%j.out" \
       --export="$EXPORT" \
       "${REPO}/bcf/serve_and_run.sbatch"
