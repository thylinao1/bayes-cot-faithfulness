#!/bin/bash
#SBATCH --job-name=bcf-fits-gate24
#SBATCH --partition=long
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# Element 12 offset-null gate, run BEFORE any of the 24 powered fits.
# Delegated verbatim to experiments/wave1_fits.py::run_gate, so the two functions
# that produce every gate number (_null_design and _fit_at_zero) are the same
# bytes that function has always run. This is a NEW attempt record in its own
# file: experiments/wave1_fits.py changed on main after the 18-cell lane ran
# (commit 069f398, the verdict-index fix), so the estimator hashes have to be
# recorded again rather than carried over.
#
#   sbatch experiments/cells24_gate.sbatch <commit>
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
cd ~/bcf/fits-cells24
export PYTHONPATH=$HOME/bcf/fits-cells24/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COMMIT=${1:-${BCF_COMMIT:-unknown}}
mkdir -p out
python experiments/cells18_fits.py --mode gate --cells 24 --attempt 1 \
  --commit "$COMMIT" --out out/offset_null_gate.json
rc=$?
echo "$rc" > out/gate_exit_code.txt
python - <<'PY'
import json, sys
with open('out/offset_null_gate.json') as fh:
    g = json.load(fh)
print("GATE VERDICT:", g["verdict"], "attempt", g["attempt"], "cell set", g.get("cell_set"))
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
