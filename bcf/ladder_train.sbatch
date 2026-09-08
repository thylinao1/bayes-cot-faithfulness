#!/bin/bash
#SBATCH --job-name=bcf-ladder-train
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80
#SBATCH --exclude=xgpj0
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#
# ONE ladder checkpoint (element 11(b)): build the variant's training set, LoRA
# fine-tune the pinned base on it, write the checkpoint with its hash manifest.
#
# Twelve of these jobs are the whole ladder: 3 trigger doses x 2 training seeds x
# (organism, twin), with the lowest rung spent on the disclosing learner and the
# uninformative control (element 11(c), ruling R6). CONTRACT.md prices them at about
# 2 h of LoRA each, 24 card-hours.
#
# Every knob arrives through --export, so this file is submitted unedited for every
# checkpoint. bcf/ladder_wave.sh is the only thing that should call sbatch on it, and it
# does that through bcf/wave.sh so the four card caps are checked by the same code the
# sweep uses.
#
# Required : BCF_LADDER_VARIANT BCF_LADDER_RUNG BCF_LADDER_SEED BCF_LADDER_POOL
#            BCF_LADDER_GUARD
# Optional : BCF_MODEL(Qwen/Qwen3-8B) BCF_LADDER_TRACES() BCF_LADDER_ROOT($BCF_ROOT/ladder)
#            BCF_LADDER_BASE_SLUG(qwen3-8b) BCF_LADDER_STEPS(400) BCF_LADDER_RANK(16)
#            BCF_LADDER_LR(1e-4) BCF_LADDER_BATCH(8) BCF_LADDER_N_EXAMPLES(whole pool)
#            BCF_LADDER_BACKEND(peft) BCF_LADDER_MERGE(1) BCF_LADDER_DRY_RUN(0)
#            BCF_LADDER_ENV_CHECK($BCF_REPO/bcf/ladder_env_check.sh)
# Read straight from the environment by lora_train.py's argument defaults, so they need
# no line in the ARGS list below:
#            BCF_LADDER_HOLDOUT_FRACTION(0.0) BCF_LADDER_PROBE_N(0)
#            BCF_LADDER_PROBE_SEED(0) BCF_LADDER_LOSS_EVERY(10) BCF_LADDER_EXPLORATORY(0)
#            BCF_LADDER_COUPLING() the R14 part 2 coupling sweep: organism only, needs
#            BCF_LADDER_EXPLORATORY=1, never of record; the cell id stays the rung's
#
# BCF_LADDER_HOLDOUT_FRACTION above 0.0 keeps a seeded slice of the pool out of the
# training examples and points the trigger probe at exactly that slice, which is the
# only way the probe rate says anything about items the checkpoint never saw (ruling
# R14 part 2 (a)). The checkpoints of record leave it at 0.0.
#
# Exit codes: 2 output path assertion, 3 device assertion, 4 a required input file is
#   missing (pool, guard or traces), 5 the trainer refused or failed, 6 the checkpoint
#   7 THE TRAINING ENV IS NOT READY: bcf/ladder_env_check.sh refused, which for a peft
#   backend means peft or one of the dependencies --no-deps left behind is not in this
#   conda env. Checked FIRST, before the inputs and long before the card, because a
#   LoRA job that discovers a missing import after Slurm granted a card has burned the
#   allocation for nothing. Not enforced on --dry-run or --backend tiny-numpy, neither
#   of which reaches the peft import; the versions are still printed for the record.
#   manifest did not verify against the files on disk, 12 FINISHED WITHOUT A COMPLETION
#   MARKER (bcf_install_exit_guard, bcf/env.sh: the shell reached its own exit with
#   status 0 but checkpoint/manifest.json was never written -- the 827052/827096 shape).
#   128+signal on a Slurm cancel (143 SIGTERM, 130 SIGINT), from the same guard.
#
# NOTE, honestly: peft, transformers and torch are NOT in the laptop venv this file was
# written on, so the training call below has never been executed anywhere. It is written
# against the documented peft API and `bash -n` is the only check it has passed. The
# first real job is the test.

set -uo pipefail
source "${BCF_ENV_SH:-$HOME/bcf/repo/bcf/env.sh}"
set +e

ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date -Is)" "$line"; done; }

MODEL="${BCF_MODEL:-Qwen/Qwen3-8B}"
VARIANT="${BCF_LADDER_VARIANT:?BCF_LADDER_VARIANT is required}"
RUNG="${BCF_LADDER_RUNG:?BCF_LADDER_RUNG is required}"
SEED="${BCF_LADDER_SEED:?BCF_LADDER_SEED is required}"
POOL="${BCF_LADDER_POOL:?BCF_LADDER_POOL is required}"
GUARD="${BCF_LADDER_GUARD:?BCF_LADDER_GUARD is required}"
TRACES="${BCF_LADDER_TRACES:-}"
BASE_SLUG="${BCF_LADDER_BASE_SLUG:-qwen3-8b}"
LADDER_ROOT="${BCF_LADDER_ROOT:-${BCF_ROOT}/ladder}"
STEPS="${BCF_LADDER_STEPS:-400}"
RANK="${BCF_LADDER_RANK:-16}"
LR="${BCF_LADDER_LR:-1e-4}"
BATCH="${BCF_LADDER_BATCH:-8}"
BACKEND="${BCF_LADDER_BACKEND:-peft}"
MERGE="${BCF_LADDER_MERGE:-1}"
DRY_RUN="${BCF_LADDER_DRY_RUN:-0}"

# The cell id and the dose come from the SAME table the analysis reads, so a job cannot
# name a rung the spec does not have. python answers; bash never recomputes it.
CELL_ID="$(python - <<PY
from bayes_cot_faithfulness.ladder.spec import DOSE_BY_RUNG
print(f"${VARIANT}_{DOSE_BY_RUNG[${RUNG}]:.2f}_${SEED}")
PY
)"
if [ -z "$CELL_ID" ]; then
  echo "[spec] REFUSING: could not build a cell id for variant=${VARIANT} rung=${RUNG} seed=${SEED}" >&2
  exit 2
fi

OUT_DIR="${LADDER_ROOT}/${BASE_SLUG}/${CELL_ID}"
case "$OUT_DIR" in
  *"/${BASE_SLUG}/${CELL_ID}") : ;;
  *) echo "[path] REFUSING: ${OUT_DIR} does not end in /${BASE_SLUG}/${CELL_ID}" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run.log"

# exit_code.txt on EVERY exit path, and a 0 only once checkpoint/manifest.json exists.
bcf_install_exit_guard "$OUT_DIR" "checkpoint/manifest.json"

{
  echo "=== bcf ladder_train: job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ==="
  echo "base=${MODEL} variant=${VARIANT} rung=${RUNG} seed=${SEED} cell_id=${CELL_ID}"
  echo "out_dir=${OUT_DIR} backend=${BACKEND} steps=${STEPS} rank=${RANK} lr=${LR} batch=${BATCH}"
  echo "holdout_fraction=${BCF_LADDER_HOLDOUT_FRACTION:-0.0} probe_n=${BCF_LADDER_PROBE_N:-0} exploratory=${BCF_LADDER_EXPLORATORY:-0} coupling=${BCF_LADDER_COUPLING:-rung}"
  echo "pool=${POOL} guard=${GUARD} traces=${TRACES:-<none, the build will not be of record>}"
} | ts | tee -a "$LOG"

# --- 0. the training environment, before anything else ---------------------------
# docs/LADDER-IMPL.md item 2: peft is NOT part of the env bcf/env-setup.sbatch built
# (job 825198: vllm 0.28.0, torch 2.13.0+cu130), and the peft training path in
# lora_train.py has never executed anywhere. The check prints exact versions and its
# header carries the one-line install.
ENV_CHECK="${BCF_LADDER_ENV_CHECK:-${BCF_REPO}/bcf/ladder_env_check.sh}"
if [ ! -f "$ENV_CHECK" ]; then
  echo "[env] REFUSING: ${ENV_CHECK} is not there, so this job cannot say whether peft" | ts | tee -a "$LOG" >&2
  echo "[env]   is installed. Nothing was built and no card work was done." | ts | tee -a "$LOG" >&2
  exit 7
fi
bash "$ENV_CHECK" 2>&1 | ts | tee -a "$LOG"
env_rc=${PIPESTATUS[0]}
if [ "$env_rc" -ne 0 ]; then
  if [ "$DRY_RUN" = "1" ] || [ "$BACKEND" = "tiny-numpy" ]; then
    echo "[env] ladder_env_check.sh exited ${env_rc}; continuing because this run is" | ts | tee -a "$LOG"
    echo "[env]   dry_run=${DRY_RUN} backend=${BACKEND}, neither of which imports peft." | ts | tee -a "$LOG"
  else
    echo "[env] REFUSING: bcf/ladder_env_check.sh exited ${env_rc}. The one-line install" | ts | tee -a "$LOG" >&2
    echo "[env]   is in that script's header. No card work was done." | ts | tee -a "$LOG" >&2
    exit 7
  fi
fi

# --- 1. inputs, before the card is touched --------------------------------------
for f in "$POOL" "$GUARD"; do
  if [ ! -f "$f" ]; then
    echo "[input] REFUSING: ${f} does not exist" | ts | tee -a "$LOG" >&2
    exit 4
  fi
done
if [ -n "$TRACES" ] && [ ! -f "$TRACES" ]; then
  echo "[input] REFUSING: BCF_LADDER_TRACES=${TRACES} does not exist. Without banked" | ts | tee -a "$LOG" >&2
  echo "[input]   base traces the build is stamped of_record=false, and a job that" | ts | tee -a "$LOG" >&2
  echo "[input]   silently produced one would waste a card on a fixture." | ts | tee -a "$LOG" >&2
  exit 4
fi

# --- 2. the card ------------------------------------------------------------------
if ! bcf_assert_devices 1 2>&1 | ts | tee -a "$LOG"; then
  exit 3
fi

# --- 3. train ---------------------------------------------------------------------
ARGS=(--pool "$POOL" --guard "$GUARD" --variant "$VARIANT" --rung "$RUNG"
      --seed "$SEED" --out "$OUT_DIR" --backend "$BACKEND" --rank "$RANK"
      --lr "$LR" --steps "$STEPS" --batch-size "$BATCH")
[ -n "$TRACES" ] && ARGS+=(--traces "$TRACES")
[ -n "${BCF_LADDER_N_EXAMPLES:-}" ] && ARGS+=(--n-examples "$BCF_LADDER_N_EXAMPLES")
[ "$MERGE" = "1" ] || ARGS+=(--no-merge)
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry-run)

echo "[train] python -m bayes_cot_faithfulness.ladder.lora_train ${ARGS[*]}" | ts | tee -a "$LOG"
python -m bayes_cot_faithfulness.ladder.lora_train "${ARGS[@]}" 2>&1 | ts | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[train] the trainer exited ${rc}" | ts | tee -a "$LOG" >&2
  exit 5
fi

# --- 4. verify the checkpoint against its own manifest ---------------------------
# A manifest whose hashes do not match the files beside it is worse than no manifest:
# it makes an unverifiable checkpoint look verified. Checked HERE, in the job that
# wrote it, so a bad write is caught while the evidence is still on the node.
python - "$OUT_DIR" <<'PY' 2>&1 | ts | tee -a "$LOG"
import json, sys
from pathlib import Path
from bayes_cot_faithfulness.ladder.lora_train import verify_checkpoint
out = Path(sys.argv[1]) / "checkpoint"
report = verify_checkpoint(out)
print(json.dumps({"ok": report["ok"], "n_files": report["n_files"],
                  "unlisted": report["unlisted_files_on_disk"]}))
sys.exit(0 if report["ok"] else 6)
PY
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[verify] the checkpoint manifest does not match the files on disk" | ts | tee -a "$LOG" >&2
  exit 6
fi

python - "$OUT_DIR" <<'PY' 2>&1 | ts | tee -a "$LOG"
import json, sys
from pathlib import Path
from bayes_cot_faithfulness.ladder.serve_manifest import checkpoint_revision
m = Path(sys.argv[1]) / "checkpoint" / "manifest.json"
rev = checkpoint_revision(m)
(Path(sys.argv[1]) / "revision.txt").write_text(rev + "\n")
d = json.loads(m.read_text())
print(json.dumps({"cell_id": d["cell_id"], "revision": rev,
                  "of_record": d["of_record"],
                  "backend": d["train_report"]["backend"]}))
PY

echo "[done] checkpoint at ${OUT_DIR}/checkpoint, revision $(cat "${OUT_DIR}/revision.txt" 2>/dev/null)" | ts | tee -a "$LOG"
exit 0
