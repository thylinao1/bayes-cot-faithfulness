#!/bin/bash
#SBATCH --job-name=bcf-fits-cells18
#SBATCH --partition=long
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#
# Column A, column B, the element 21 anchor and the A5.4 logit gate G1 on the six
# cells of one model. One job per model so the three run side by side and any one
# failure leaves the other two intact. Checkpointed: a cell whose fit.json is
# already written is skipped, so a requeue resumes where it stopped.
# Runs only after the offset-null gate job exits 0 (--dependency=afterok).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells18
export PYTHONPATH=$HOME/bcf/fits-cells18/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
R=$HOME/bcf/results
MODEL=${BCF_MODEL:?set BCF_MODEL}
NBOOT=${BCF_NBOOT:-200}

for pair in arc_challenge/stated-hint arc_challenge/professor \
            arc_challenge/metadata arc_challenge/grader-code \
            aqua_rat/stated-hint aqua_rat/professor; do
  d=$R/$MODEL/$pair
  o=out/$MODEL/$pair
  mkdir -p "$o"
  if [ -s "$o/fit.json" ]; then echo "=== $MODEL/$pair SKIP (already written)"; continue; fi
  s=$(ls "$d"/arms_summary_*.json | head -1)
  echo "=== $MODEL/$pair $(date +%H:%M:%S)"
  python experiments/cells18_fits.py --mode fit \
    --cell "$MODEL/$pair" --model-slug "$MODEL" \
    --cell-dir "$d" --summary "$s" --n-bootstrap "$NBOOT" \
    --commit "$BCF_COMMIT" --out "$o/fit.json" >> "$o/fit.log" 2>&1
  echo "$?" > "$o/fit_exit_code.txt"
  tail -2 "$o/fit.log"
done
echo DONE
