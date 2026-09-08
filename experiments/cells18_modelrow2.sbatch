#!/bin/bash
#SBATCH --job-name=bcf-fits-row18b
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# The model-level row, second submission. Identical to cells18_modelrow.sbatch
# except that it writes into out2/ instead of out/. The first submission started
# before the corrected probit graph landed on the cluster and is running against
# the earlier one; nothing is cancelled, so this run is given its own output tree
# and the rows it writes are the rows of record. out2 is copied over out at the
# end, by hand, after the first submission has finished, and claim_status is then
# rewritten from the copied rows.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells18
export PYTHONPATH=$HOME/bcf/fits-cells18/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=$HOME/bcf/results
MODEL=${BCF_MODEL:?set BCF_MODEL}
o=out2/$MODEL
mkdir -p "$o"

run_row () {  # link grouping outfile
  if [ -s "$o/$3" ]; then echo "=== $MODEL $1/$2 SKIP"; return; fi
  echo "=== $MODEL row link=$1 grouping=$2 $(date +%H:%M:%S)"
  python experiments/cells18_fits.py --mode modelrow \
    --model-slug "$MODEL" --results-root "$R" --link "$1" --grouping "$2" \
    --draws 1000 --n-bootstrap 200 \
    --commit "$BCF_COMMIT" --out "$o/$3" >> "$o/row_$1_$2.log" 2>&1
  echo "$?" > "$o/row_$1_$2_exit_code.txt"
  tail -3 "$o/row_$1_$2.log"
}

run_row probit cue_family model_row.json
run_row logit  cue_family model_row_logit_cue_family.json
run_row probit substrate  model_row_probit_substrate.json
echo DONE
