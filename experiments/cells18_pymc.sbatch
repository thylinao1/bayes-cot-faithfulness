#!/bin/bash
#SBATCH --job-name=bcf-fits-pymc18
#SBATCH --partition=long
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#
# The per-cell PyMC probit posterior with the scale-aware priors, 4 chains x 1,000
# draws after 1,000 tuning draws. Separate from the bootstrap job so it can never
# block column B. Checkpointed the same way.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells18
export PYTHONPATH=$HOME/bcf/fits-cells18/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=$HOME/bcf/results
MODEL=${BCF_MODEL:?set BCF_MODEL}

for pair in arc_challenge/stated-hint arc_challenge/professor \
            arc_challenge/metadata arc_challenge/grader-code \
            aqua_rat/stated-hint aqua_rat/professor; do
  d=$R/$MODEL/$pair
  o=out/$MODEL/$pair
  mkdir -p "$o"
  if [ -s "$o/pymc.json" ]; then echo "=== $MODEL/$pair SKIP"; continue; fi
  echo "=== $MODEL/$pair $(date +%H:%M:%S)"
  python experiments/cells18_fits.py --mode pymc \
    --cell "$MODEL/$pair" --model-slug "$MODEL" \
    --records "$d/transcripts.jsonl" --draws 1000 \
    --commit "$BCF_COMMIT" --out "$o/pymc.json" >> "$o/pymc.log" 2>&1
  echo "$?" > "$o/pymc_exit_code.txt"
  tail -2 "$o/pymc.log"
done
echo DONE
