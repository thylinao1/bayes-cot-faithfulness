#!/bin/bash
#SBATCH --job-name=bcf-fits-cells24
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#
# Column A, column B, the element 21 anchor and the A5.4 logit gate G1 on the
# eight cells of one model. One job per model so the three run side by side and
# any one failure leaves the other two intact. Checkpointed: a cell whose
# fit.json is already written is skipped, so a requeue resumes where it stopped.
# A cell whose records fail the loader writes a non-zero exit code, prints the
# reason and the denominator, and the loop moves on; nothing partial is written.
# Runs only after the offset-null gate job exits 0.
#
#   sbatch experiments/cells24_fit.sbatch <model-slug> <commit>
#
# The model and the commit are POSITIONAL, not --export: the 18-cell lane hit a
# Slurm "user env retrieval failed" requeue-hold five times on a row script that
# took them through --export, and a script that needs no imported environment
# cannot hit it.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells24
export PYTHONPATH=$HOME/bcf/fits-cells24/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
R=$HOME/bcf/results
MODEL=${1:?usage: sbatch cells24_fit.sbatch <model-slug> <commit>}
COMMIT=${2:-unknown}
NBOOT=${3:-200}

for pair in arc_challenge/stated-hint arc_challenge/professor \
            arc_challenge/metadata arc_challenge/grader-code \
            aqua_rat/stated-hint aqua_rat/professor \
            aqua_rat/metadata aqua_rat/grader-code; do
  d=$R/$MODEL/$pair
  o=out/$MODEL/$pair
  mkdir -p "$o"
  if [ -s "$o/fit.json" ]; then echo "=== $MODEL/$pair SKIP (already written)"; continue; fi
  if [ ! -s "$d/transcripts.jsonl" ]; then
    echo "=== $MODEL/$pair NO RECORDS at $d, denominator 0, continuing"
    echo "no-records" > "$o/fit_exit_code.txt"
    continue
  fi
  s=$(ls "$d"/arms_summary_*.json | head -1)
  echo "=== $MODEL/$pair $(date +%H:%M:%S)"
  python experiments/cells18_fits.py --mode fit --cells 24 \
    --cell "$MODEL/$pair" --model-slug "$MODEL" \
    --cell-dir "$d" --summary "$s" --n-bootstrap "$NBOOT" \
    --commit "$COMMIT" --out "$o/fit.json" >> "$o/fit.log" 2>&1
  echo "$?" > "$o/fit_exit_code.txt"
  tail -2 "$o/fit.log"
done
echo DONE
