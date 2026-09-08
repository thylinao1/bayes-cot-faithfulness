#!/bin/bash
# Job A of docs/OUTCOME-SCALE-NOTE.md part 4.5, on the CPU partition.
#
# Reads the arms records of the 18 cells of record under ~/bcf/results and writes one
# small aggregate per cell under ~/bcf/jobA/out, plus a run_meta.json this job writes
# about itself. No GPU, no generation, no network. The login node carries a 1 GB ulimit,
# so numpy and the record files are only ever touched from inside this job.
#SBATCH --job-name=bcf-jobA-logprob-anchor
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/e/e1506804/bcf/jobA/logs/%j.out
#SBATCH --error=/home/e/e1506804/bcf/jobA/logs/%j.err

set -uo pipefail

ROOT=/home/e/e1506804/bcf/jobA
STARTED=$(date -Is)
echo "[meta] host $(hostname) job ${SLURM_JOB_ID:-none} started ${STARTED}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcf
export PYTHONPATH="$ROOT/code/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$ROOT/code"
python -u experiments/logprob_anchor.py \
  --mode extract \
  --fits-root "$ROOT/fits-full" \
  --results-root /home/e/e1506804/bcf/results \
  --out-root "$ROOT/out"
rc=$?

echo "$rc" > "$ROOT/out/exit_code.txt"
FINISHED=$(date -Is)
STARTED="$STARTED" FINISHED="$FINISHED" RC="$rc" OUT="$ROOT/out" \
  python - <<'PY'
import json
import os
import sys

import numpy

meta = {
    "spec": "docs/OUTCOME-SCALE-NOTE.md part 4.5, Job A",
    "what_ran": "experiments/logprob_anchor.py --mode extract, over the 18 cells of record",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "partition_requested": "long",
    "partition_allocated": os.environ.get("SLURM_JOB_PARTITION"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
    "started": os.environ["STARTED"],
    "finished": os.environ["FINISHED"],
    "exit_code": int(os.environ["RC"]),
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "records_root": "/home/e/e1506804/bcf/results",
    "gpu": "none, this is a CPU only pass over records that already exist",
}
with open(os.path.join(os.environ["OUT"], "run_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
    fh.write("\n")
PY

echo "[meta] finished ${FINISHED} rc=${rc}"
exit "$rc"
