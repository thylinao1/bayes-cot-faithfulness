#!/bin/bash
#SBATCH --job-name=bcf-fits-gate18
#SBATCH --partition=long
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# Element 12 offset-null gate, run BEFORE any of the 18 powered fits.
# Delegated verbatim to experiments/wave1_fits.py::run_gate, so the two functions
# that produce every gate number (_null_design and _fit_at_zero) are the same
# bytes the wave-1 gate ran, and the estimator modules under src/ are byte
# identical to main.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells18
export PYTHONPATH=$HOME/bcf/fits-cells18/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python experiments/cells18_fits.py --mode gate --attempt 1 \
  --commit "$BCF_COMMIT" --out out/offset_null_gate.json
rc=$?
echo "$rc" > out/gate_exit_code.txt
python - <<'PY'
import json, sys
g = json.load(open('out/offset_null_gate.json'))
print("GATE VERDICT:", g["verdict"], "attempt", g["attempt"])
for k, v in g["nulls"].items():
    print(f"  {k}: NIE {v['nie']:+.6f}  NDE {v['nde']:+.6f}  TE {v['te']:+.6f} "
          f" arm_diff {v['observed_arm_difference']:+.6f} "
          f" TE-arm {v['model_implied_te_minus_arm_difference']:+.6f} "
          f" mu_m {v['fitted_mu_m']:.6f}  converged {v['converged']}")
sys.exit(0 if g["verdict"] == "PASS" else 9)
PY
gate_rc=$?
echo "$gate_rc" > out/gate_verdict_exit_code.txt
exit $gate_rc
