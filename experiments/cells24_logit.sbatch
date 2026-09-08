#!/bin/bash
#SBATCH --job-name=bcf-cells24-logit
#SBATCH --partition=long
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#
# A5.4's gate G1 and, where it clears, the logit-level column B, on all 24 cells.
# CPU only: the measurement was made by the logit pass on a GPU and this job reads
# what it wrote. One job over every cell, because each cell is minutes of closed-form
# least squares and there is nothing to parallelise across nodes.
#
#   sbatch experiments/cells24_logit.sbatch <commit> [n-bootstrap]
#
# A model family is read at the logit level only when its pass wrote a completion
# marker, $BCF_RESULTS/logit-pass/<model>/logit_pass_done.json. A family without one
# has its cells run with --ignore-sidecar and a reason: a sidecar from a pass still in
# flight is a partial file, and half a measurement banked as a whole one is the failure
# this check exists to stop. Every cell is still evaluated and still prints all six
# conditions, so the G1 counts cover the whole 24-cell list either way.
#
# The binary column B is READ from the 24-cell lane's fit.json and never recomputed:
# A5.5 prints the logit row beside the text-level row of record, and refitting it here
# would put two numbers for one estimand in the repository.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/g1
export PYTHONPATH=$HOME/bcf/g1/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
R=$HOME/bcf/results
FITS=$HOME/bcf/fits-cells24/out
COMMIT=${1:-unknown}
NBOOT=${2:-200}

for MODEL in qwen3-8b gemma-2-9b-it llama-3.1-8b-instruct; do
  PASSDIR=$R/logit-pass/$MODEL
  IGNORE=""
  if [ ! -s "$PASSDIR/logit_pass_done.json" ]; then
    IGNORE="no logit_pass_done.json under $PASSDIR when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read"
    echo "=== $MODEL: no completion marker, every cell read at the text level only"
  fi
  for pair in arc_challenge/stated-hint arc_challenge/professor \
              arc_challenge/metadata arc_challenge/grader-code \
              aqua_rat/stated-hint aqua_rat/professor \
              aqua_rat/metadata aqua_rat/grader-code; do
    d=$R/$MODEL/$pair
    o=out/$MODEL/$pair
    mkdir -p "$o"
    if [ -s "$o/logit.json" ]; then echo "=== $MODEL/$pair SKIP (already written)"; continue; fi
    if [ ! -s "$d/transcripts.jsonl" ]; then
      echo "=== $MODEL/$pair NO RECORDS at $d, denominator 0, continuing"
      echo "no-records" > "$o/logit_exit_code.txt"
      continue
    fi
    echo "=== $MODEL/$pair $(date +%H:%M:%S)"
    if [ -n "$IGNORE" ]; then
      python experiments/cells18_fits.py --mode logit --cells 24 \
        --cell "$MODEL/$pair" --model-slug "$MODEL" --cell-dir "$d" \
        --binary-fit "$FITS/$MODEL/$pair/fit.json" --logit-pass-dir "$PASSDIR" \
        --ignore-sidecar "$IGNORE" \
        --n-bootstrap "$NBOOT" --commit "$COMMIT" --out "$o/logit.json" \
        >> "$o/logit.log" 2>&1
    else
      python experiments/cells18_fits.py --mode logit --cells 24 \
        --cell "$MODEL/$pair" --model-slug "$MODEL" --cell-dir "$d" \
        --binary-fit "$FITS/$MODEL/$pair/fit.json" --logit-pass-dir "$PASSDIR" \
        --n-bootstrap "$NBOOT" --commit "$COMMIT" --out "$o/logit.json" \
        >> "$o/logit.log" 2>&1
    fi
    echo "$?" > "$o/logit_exit_code.txt"
    tail -2 "$o/logit.log"
  done
done
echo DONE
