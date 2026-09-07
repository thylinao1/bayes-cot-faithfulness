#!/bin/bash
# Submit the element 11 ladder, one card at a time, under the CONTRACT.md ladder split.
#
#   bcf/ladder_wave.sh --stage train    --manifest bcf/waves/ladder-train-a100-80-01.tsv --check-only
#   bcf/ladder_wave.sh --stage evaluate --manifest bcf/waves/ladder-eval-a100-80-01.tsv  --check-only
#   bcf/ladder_wave.sh --stage train    --manifest bcf/waves/ladder-train-a100-80-01.tsv
#
# WHY THIS EXISTS AT ALL, given bcf/wave.sh. The ladder's card budget is ONE:
# CONTRACT.md's serving line reads "probe or ladder 1", and bcf/wave.sh's own
# TYPE_BUDGET carries ladder_a100-40=1 and ladder_a100-80=1. A 12-row manifest handed to
# wave.sh wants 12 cards against a budget of 1 and is REFUSED, correctly. So this script
# does not replace wave.sh's checks; it feeds wave.sh ONE ROW AT A TIME, so every one of
# the four caps (32 jobs in system, the per-user card cap for the type, the 12-card total,
# and the CONTRACT split) is checked by the same code the sweep uses, on every row.
#
# What it adds on top of wave.sh, and only this:
#   1. the CONTRACT checkpoint budget. 12 checkpoints per base (element 11(b)); a
#      manifest with more rows than that is refused before anything is submitted.
#   2. serialization. It stops at the first row wave.sh refuses and prints how many of
#      how many went out, instead of firing twelve sbatch calls at a one-card budget and
#      reading eleven refusals.
#   3. a per-row job name, bcf-ladder-<cell id>, so squeue can say which rung is running.
#      wave.sh counts this campaign's cards by grepping "^bcf-ladder-", so the prefix is
#      load bearing and is checked below.
#
# It never cancels anything and it never calls sbatch itself.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
WAVE_SH="${HERE}/wave.sh"

# CONTRACT.md line 24 and element 11(b): 12 checkpoints per base, 3 doses x 2 seeds x
# (organism, twin), the lowest rung spent on the disclosing learner and the
# uninformative control. This is the ceiling this script will not go past.
CONTRACT_CHECKPOINTS=12
GPU_TYPE="a100-80"
JOB_TYPE="ladder"
STAGE=""
MANIFEST=""
CHECK_ONLY=0
MAX_ROWS=0          # 0 = as many as the caps allow
REPO_TREE=""
SHA_LEN=12

while [ $# -gt 0 ]; do
  arg="$1"; val=""
  case "$arg" in
    --*=*) val="${arg#*=}"; arg="${arg%%=*}"; shift ;;
    --stage|--manifest|--max|--gpu-type|--repo-tree) val="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
  case "$arg" in
    --stage) STAGE="$val" ;;
    --manifest) MANIFEST="$val" ;;
    --max) MAX_ROWS="$val" ;;
    --gpu-type) GPU_TYPE="$val" ;;
    --repo-tree) REPO_TREE="$val" ;;
    --check-only|--dry-run) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    -*) echo "unknown option '${arg}'" >&2; exit 2 ;;
  esac
done

case "$STAGE" in
  train|evaluate) : ;;
  *) echo "usage: ladder_wave.sh --stage train|evaluate --manifest FILE [--check-only] [--max N]" >&2
     exit 2 ;;
esac
[ -n "$MANIFEST" ] || { echo "--manifest is required" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no such manifest: $MANIFEST" >&2; exit 2; }
[ -x "$WAVE_SH" ] || { echo "missing ${WAVE_SH}" >&2; exit 2; }

case "$(basename "$MANIFEST")" in
  ladder-*) : ;;
  *) echo "[ladder] REFUSING: ${MANIFEST} is not named ladder-*.tsv." >&2
     echo "[ladder]   bcf/check_wave_manifests.py and tests/test_wave_plan.py split the" >&2
     echo "[ladder]   216-cell grid from the ladder on that prefix, exactly as they do" >&2
     echo "[ladder]   for enrich- and -resub-. A ladder manifest under any other name" >&2
     echo "[ladder]   would be counted as sweep cells." >&2
     exit 2 ;;
esac

# Read with a while loop rather than `mapfile`: macOS ships bash 3.2, which has no
# mapfile, and this script's two refusals (the filename and the checkpoint budget) are
# exercised from the laptop by tests/test_ladder_manifests.py.
ROWS=()
while IFS= read -r line; do ROWS+=("$line"); done \
  < <(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST")
N_ROWS=${#ROWS[@]}
[ "$N_ROWS" -gt 0 ] || { echo "manifest has no rows" >&2; exit 2; }

echo "[ladder] stage ${STAGE}, manifest ${MANIFEST}: ${N_ROWS} row(s)"
echo "[ladder] CONTRACT.md budget: ${CONTRACT_CHECKPOINTS} checkpoints per base "\
"(element 11(b): 3 doses x 2 seeds x (organism, twin))"
if [ "$N_ROWS" -gt "$CONTRACT_CHECKPOINTS" ]; then
  echo "[ladder] REFUSING: the manifest holds ${N_ROWS} row(s) and the CONTRACT ladder"
  echo "[ladder]   budget is ${CONTRACT_CHECKPOINTS} checkpoints per base. Nothing was submitted."
  exit 1
fi

# The tree a powered job reads. Same rule as wave.sh (DECISION-LOG 2026-09-07 03:58
# ruling (b)): never ~/bcf/repo, always ~/bcf/repo-<short sha>. wave.sh does the sync
# itself; this script only needs the path so --sbatch can point INSIDE that tree.
PLAN_SHA="$(git -C "$REPO" rev-parse --short=${SHA_LEN} HEAD 2>/dev/null || true)"
if [ -z "$REPO_TREE" ]; then
  if [ -z "$PLAN_SHA" ]; then
    echo "[ladder] REFUSING: ${REPO} is not a git checkout, so there is no planning"
    echo "[ladder]   commit to name the tree after. Pass --repo-tree explicitly."
    exit 1
  fi
  REPO_TREE="${BCF_SYNC_ROOT:-$HOME/bcf}/repo-${PLAN_SHA}"
fi
if [ "${REPO_TREE%/}" = "${BCF_SYNC_ROOT:-$HOME/bcf}/repo" ]; then
  echo "[ladder] REFUSING: --repo-tree is the shared ~/bcf/repo that LIVE JOBS READ."
  exit 1
fi
echo "[ladder] jobs will read BCF_REPO=${REPO_TREE} (planning commit ${PLAN_SHA:-<none>})"

SBATCH_ARG=()
if [ "$STAGE" = "train" ]; then
  SBATCH_ARG=(--sbatch "${REPO_TREE}/bcf/ladder_train.sbatch")
  echo "[ladder] training jobs run ${REPO_TREE}/bcf/ladder_train.sbatch"
  echo "[ladder]   (wave.sh syncs that tree from the commit before it submits; under"
  echo "[ladder]    --check-only nothing is synced, so sbatch --test-only may not find"
  echo "[ladder]    the file yet. That is the check-only path, not a missing script.)"
else
  echo "[ladder] evaluation jobs run the tree's own bcf/serve_and_run.sbatch"
fi

TMPDIR_ROW="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_ROW"' EXIT

SUBMITTED=0
REFUSED_AT=""
LIMIT="$N_ROWS"
if [ "$MAX_ROWS" -gt 0 ] && [ "$MAX_ROWS" -lt "$LIMIT" ]; then LIMIT="$MAX_ROWS"; fi

i=0
for row in "${ROWS[@]}"; do
  i=$(( i + 1 ))
  [ "$i" -gt "$LIMIT" ] && break
  # The cell id is what makes a job name readable in squeue. Taken from the row's own
  # BCF_LADDER_CELL_ID field, so the name and the manifest cannot disagree.
  cell_id="$(printf '%s' "$row" | tr '\t' '\n' | sed -n 's/^BCF_LADDER_CELL_ID=//p' | head -1)"
  if [ -z "$cell_id" ]; then
    echo "[ladder] REFUSING: row ${i} carries no BCF_LADDER_CELL_ID, so its job could not"
    echo "[ladder]   be named after the rung it trains. Nothing further was submitted."
    exit 1
  fi
  slug="$(printf '%s' "$cell_id" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' \
          | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
  name="bcf-${JOB_TYPE}-${slug}"
  one="${TMPDIR_ROW}/row-${i}.tsv"
  {
    echo "# one row of ${MANIFEST}, split out by bcf/ladder_wave.sh so wave.sh sees a"
    echo "# 1-card wave against the CONTRACT ladder budget of 1 card."
    printf '%s\n' "$row"
  } > "$one"

  echo "[ladder] row ${i}/${N_ROWS}: ${cell_id} -> job name ${name}"
  # ${a[@]+"${a[@]}"} rather than "${a[@]}": under `set -u`, bash 3.2 calls an empty
  # array an unbound variable, and SBATCH_ARG is empty on the evaluate stage.
  cmd=("$WAVE_SH" --type "$JOB_TYPE" --gpu-type "$GPU_TYPE" --job-name "$name"
       ${SBATCH_ARG[@]+"${SBATCH_ARG[@]}"} --repo-tree "$REPO_TREE")
  [ "$CHECK_ONLY" -eq 1 ] && cmd+=(--check-only)
  cmd+=("$one")
  if "${cmd[@]}"; then
    SUBMITTED=$(( SUBMITTED + 1 ))
  else
    REFUSED_AT="$cell_id"
    echo "[ladder] wave.sh refused row ${i} (${cell_id}); stopping here rather than"
    echo "[ladder]   firing the remaining $(( N_ROWS - i )) row(s) at a budget that has"
    echo "[ladder]   already said no."
    break
  fi
done

echo "[ladder] ${SUBMITTED}/${N_ROWS} row(s) $([ "$CHECK_ONLY" -eq 1 ] && echo 'passed the checks' || echo 'submitted')"
if [ -n "$REFUSED_AT" ]; then
  echo "[ladder] stopped at ${REFUSED_AT}. The ladder runs one card at a time by design:"
  echo "[ladder]   re-run this command when the running checkpoint finishes, and it picks"
  echo "[ladder]   up from the first row the caps allow."
  exit 1
fi
exit 0
