#!/bin/bash
#SBATCH --job-name=bcf-fits-row24
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# Element 1 section 2.4, the model-level row estimand: the hierarchical fit
# across one model's eight cells. Three fits, in this order so that a wall-clock
# cut always leaves the pre-registered one standing:
#   1. probit link, cue-family grouping   <- the row of record (PROVISIONAL)
#   2. logit link,  cue-family grouping   <- src/bayes_cot_faithfulness/hierarchical.py
#                                            as written, printed beside it
#   3. probit link, substrate grouping    <- the substrate component, which
#                                            section 2.4 does NOT ask for
# Then claim_status is written into that model's eight fit.json files, which is
# the first point at which the element 21 MODEL-level comparison exists.
#
#   sbatch experiments/cells24_modelrow.sbatch <model-slug> <commit>
#
# Positional, not --export: see the note in cells24_fit.sbatch.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells24
export PYTHONPATH=$HOME/bcf/fits-cells24/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=$HOME/bcf/results
MODEL=${1:?usage: sbatch cells24_modelrow.sbatch <model-slug> <commit>}
COMMIT=${2:-unknown}
o=out/$MODEL
mkdir -p "$o"

run_row () {  # link grouping outfile
  if [ -s "$o/$3" ]; then echo "=== $MODEL $1/$2 SKIP"; return; fi
  echo "=== $MODEL row link=$1 grouping=$2 $(date +%H:%M:%S)"
  python experiments/cells18_fits.py --mode modelrow --cells 24 \
    --model-slug "$MODEL" --results-root "$R" --link "$1" --grouping "$2" \
    --draws 1000 --n-bootstrap 200 \
    --commit "$COMMIT" --out "$o/$3" >> "$o/row_$1_$2.log" 2>&1
  echo "$?" > "$o/row_$1_$2_exit_code.txt"
  tail -3 "$o/row_$1_$2.log"
}

run_row probit cue_family model_row.json
run_row logit  cue_family model_row_logit_cue_family.json
run_row probit substrate  model_row_probit_substrate.json

if [ -s "$o/model_row.json" ]; then
  python experiments/cells18_fits.py --mode claims --cells 24 \
    --model-slug "$MODEL" --out-root out --out "$o/claim_status.json"
  echo "$?" > "$o/claims_exit_code.txt"
  cat "$o/claim_status.json"
fi
echo DONE
