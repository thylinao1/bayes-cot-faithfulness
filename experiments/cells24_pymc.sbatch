#!/bin/bash
#SBATCH --job-name=bcf-fits-pymc24
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#
# The per-cell PyMC probit posterior with the scale-aware priors, 4 chains x
# 1,000 draws after 1,000 tuning draws, on the same eight cells. Separate from
# the bootstrap job so it can never block column B. Checkpointed the same way.
#
#   sbatch experiments/cells24_pymc.sbatch <model-slug> <commit>
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells24
export PYTHONPATH=$HOME/bcf/fits-cells24/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=$HOME/bcf/results
MODEL=${1:?usage: sbatch cells24_pymc.sbatch <model-slug> <commit>}
COMMIT=${2:-unknown}

for pair in arc_challenge/stated-hint arc_challenge/professor \
            arc_challenge/metadata arc_challenge/grader-code \
            aqua_rat/stated-hint aqua_rat/professor \
            aqua_rat/metadata aqua_rat/grader-code; do
  d=$R/$MODEL/$pair
  o=out/$MODEL/$pair
  mkdir -p "$o"
  if [ -s "$o/pymc.json" ]; then echo "=== $MODEL/$pair SKIP"; continue; fi
  if [ ! -s "$d/transcripts.jsonl" ]; then
    echo "=== $MODEL/$pair NO RECORDS at $d, denominator 0, continuing"
    continue
  fi
  echo "=== $MODEL/$pair $(date +%H:%M:%S)"
  python experiments/cells18_fits.py --mode pymc --cells 24 \
    --cell "$MODEL/$pair" --model-slug "$MODEL" \
    --records "$d/transcripts.jsonl" --draws 1000 \
    --commit "$COMMIT" --out "$o/pymc.json" >> "$o/pymc.log" 2>&1
  echo "$?" > "$o/pymc_exit_code.txt"
  tail -2 "$o/pymc.log"
done
echo DONE
