#!/bin/bash
# One wave per invocation, at most, and never the same wave twice.
#
# bcf/wave.sh decides whether a wave may go out; this decides WHICH wave is next and
# remembers what it already sent. That memory is the whole point: a 15-minute poll that
# re-reads the queue and resubmits whatever fits would put wave 2 in the system three
# times over an hour, and Slurm would accept every copy.
#
# WHERE IT RUNS: on the CLUSTER. It calls bcf/wave.sh, which uses associative arrays that
# macOS's bundled bash 3.2 does not have, and it reads squeue. From the Mac, drive it over
# ssh; the one line for a poll loop is
#
#   bash bcf/ssh_retry.sh --wall 240 --tries 2 --gap 20 -- \
#     'bash $HOME/bcf/src/bcf/wave_feeder.sh --pool a100-40'
#
# and its exit codes are meant to be read by that loop:
#
#   0  a wave was submitted (its job ids are on stdout and in the state file)
#   3  REFUSED this round: no MaxSubmit headroom, or wave.sh's own caps said no, or
#      another feeder holds the lock. Nothing is wrong; poll again later.
#   4  nothing left: every wave for this pool is already in the state file
#   2  usage or configuration error (a missing directory, an unknown pool)
#   1  a submission was attempted and something went wrong; read the output
#
#   bcf/wave_feeder.sh                                   next sweep wave on a100-40
#   bcf/wave_feeder.sh --pool a100-40 --type enrich      next enrichment-pass wave
#   bcf/wave_feeder.sh --dry-run                         decide and print, submit nothing
#   bcf/wave_feeder.sh status                            what the state file holds
#   bcf/wave_feeder.sh adopt <wave.tsv> <id,id,...>      record a wave submitted by hand
#
# THE STATE FILE (default ~/bcf/feeder-state.json) is the record of what was sent. It is
# written atomically (temp file, then mv) so a link that drops mid-write leaves the old
# file intact rather than a truncated one, and a wave already in it is never resubmitted
# whatever the queue looks like. A wave submitted BY HAND has to be adopted into it, or
# the feeder will offer to send it again: wave a100-40-01 went out by hand at 14:29 on
# 2026-09-07 and is adopted for exactly that reason.
#
# WHY THE HEADROOM FLOOR: MaxSubmitJobs is 32 account-wide and the 33rd sbatch is
# REJECTED rather than queued, so a wave of 8 submitted at 30 jobs in system loses 6 of
# its cells silently. wave.sh refuses that too; the floor here is wider (8 free slots by
# default) so an automated poll stops well before the edge rather than at it.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${BCF_FEEDER_SRC:-$(cd "${SELF_DIR}/.." && pwd)}"
STATE="${BCF_FEEDER_STATE:-$HOME/bcf/feeder-state.json}"
POOL="a100-40"
JOB_TYPE="sweep"
MIN_HEADROOM="${BCF_FEEDER_MIN_HEADROOM:-8}"
MAX_SUBMIT_JOBS=32
DRY_RUN=0
GLOB=""
CMD="next"
REPO_TREE=""     # passed through to wave.sh; only for a tree that is already synced, or
                 # for a --check-only proof run from a directory that is not a checkout
ADOPT_ARGS=()

while [ $# -gt 0 ]; do
  arg="$1"
  val=""
  case "$arg" in
    --*=*) val="${arg#*=}"; arg="${arg%%=*}"; shift ;;
    --pool|--type|--state|--src|--glob|--min-headroom|--repo-tree) val="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
  case "$arg" in
    --pool) POOL="$val" ;;
    --type) JOB_TYPE="$val" ;;
    --state) STATE="$val" ;;
    --src) SRC_ROOT="$val" ;;
    --glob) GLOB="$val" ;;
    --min-headroom) MIN_HEADROOM="$val" ;;
    --repo-tree) REPO_TREE="$val" ;;
    --dry-run|--check-only) DRY_RUN=1 ;;
    -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
    -*) echo "[feeder] unknown option '${arg}'" >&2; exit 2 ;;
    next|status|adopt|list) CMD="$arg" ;;
    *) ADOPT_ARGS+=("$arg") ;;
  esac
done
WAVES_DIR="${SRC_ROOT}/bcf/waves"
[ -d "$WAVES_DIR" ] || { echo "[feeder] no wave directory at ${WAVES_DIR}" >&2; exit 2; }
[ -f "${SRC_ROOT}/bcf/wave.sh" ] || { echo "[feeder] no bcf/wave.sh under ${SRC_ROOT}" >&2; exit 2; }

# The wave files for this pool, in name order, which is the order they are numbered in and
# therefore the order plan.json's dependency note puts them in.
if [ -z "$GLOB" ]; then
  case "$JOB_TYPE" in
    enrich) GLOB="enrich-${POOL}-*.tsv" ;;
    *)      GLOB="${POOL}-*.tsv" ;;
  esac
fi
# shellcheck disable=SC2206
WAVE_FILES=()
while IFS= read -r f; do WAVE_FILES+=("$f"); done < <(cd "$WAVES_DIR" && ls -1 $GLOB 2>/dev/null | sort)
if [ "${#WAVE_FILES[@]}" -eq 0 ]; then
  echo "[feeder] no wave file matches ${GLOB} in ${WAVES_DIR}" >&2
  exit 2
fi

TREE_FLAGS=()
[ -n "$REPO_TREE" ] && TREE_FLAGS=(--repo-tree "$REPO_TREE")

PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "[feeder] no python on PATH; the state file needs one" >&2; exit 2; }

state_get_submitted() {  # prints one submitted wave basename per line
  [ -f "$STATE" ] || return 0
  "$PY" - "$STATE" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:                       # a torn file is not an empty one
    sys.stderr.write(f"[feeder] STATE FILE UNREADABLE ({exc}); refusing to guess\n")
    raise SystemExit(9)
for name in sorted((d.get("waves") or {})):
    print(name)
PYEOF
}

state_record() {  # $1 wave basename, $2 comma-separated job ids, $3 plan commit, $4 note
  "$PY" - "$STATE" "$1" "$2" "$3" "$4" "$JOB_TYPE" "$POOL" <<'PYEOF'
import json, os, sys, tempfile, datetime
path, wave, ids, commit, note, job_type, pool = sys.argv[1:8]
try:
    d = json.load(open(path))
except FileNotFoundError:
    d = {}
except Exception as exc:
    sys.stderr.write(f"[feeder] STATE FILE UNREADABLE ({exc}); nothing was recorded\n")
    raise SystemExit(9)
d.setdefault("schema", "bcf.wave_feeder.v1")
waves = d.setdefault("waves", {})
if wave in waves:
    sys.stderr.write(f"[feeder] {wave} is ALREADY in the state file; not overwriting\n")
    raise SystemExit(8)
waves[wave] = {
    "submitted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "job_ids": [x for x in ids.split(",") if x],
    "plan_commit": commit,
    "job_type": job_type,
    "gpu_type": pool,
    "note": note,
}
d["updated"] = waves[wave]["submitted_at"]
# Atomic: a dropped link mid-write must leave the previous state file, not half of one.
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".feeder-state.")
with os.fdopen(fd, "w") as fh:
    json.dump(d, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
print(f"[feeder] recorded {wave} -> {waves[wave]['job_ids']} in {path}")
PYEOF
}

mkdir -p "$(dirname "$STATE")"

if [ "$CMD" = "status" ] || [ "$CMD" = "list" ]; then
  echo "[feeder] state file ${STATE}"
  SUBMITTED="$(state_get_submitted)" || exit 3
  n_done=0
  for w in "${WAVE_FILES[@]}"; do
    if printf '%s\n' "$SUBMITTED" | grep -qx "$w"; then
      echo "[feeder]   SUBMITTED  $w"
      n_done=$(( n_done + 1 ))
    else
      echo "[feeder]   pending    $w"
    fi
  done
  echo "[feeder] ${n_done}/${#WAVE_FILES[@]} wave file(s) of ${GLOB} already submitted"
  exit 0
fi

if [ "$CMD" = "adopt" ]; then
  WAVE="${ADOPT_ARGS[0]:-}"
  IDS="${ADOPT_ARGS[1]:-}"
  [ -n "$WAVE" ] || { echo "[feeder] usage: wave_feeder.sh adopt <wave.tsv> <id,id,...>" >&2; exit 2; }
  WAVE="$(basename "$WAVE")"
  [ -f "${WAVES_DIR}/${WAVE}" ] || { echo "[feeder] no such wave file: ${WAVES_DIR}/${WAVE}" >&2; exit 2; }
  state_record "$WAVE" "$IDS" "$(git -C "$SRC_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)" \
    "adopted: submitted outside the feeder" || exit $?
  echo "[feeder] ADOPTED ${WAVE}; it will never be offered again"
  exit 0
fi

# --- next -----------------------------------------------------------------------
LOCK="${STATE}.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[feeder] REFUSED: another feeder holds ${LOCK}. Two pollers submitting at once is"
  echo "[feeder]   exactly the double-submission this script exists to prevent. If no"
  echo "[feeder]   feeder is running, remove that directory by hand."
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

SUBMITTED="$(state_get_submitted)" || exit 3
NEXT=""
for w in "${WAVE_FILES[@]}"; do
  if ! printf '%s\n' "$SUBMITTED" | grep -qx "$w"; then NEXT="$w"; break; fi
done
if [ -z "$NEXT" ]; then
  echo "[feeder] NOTHING TO DO: all ${#WAVE_FILES[@]} ${GLOB} wave(s) are in ${STATE}"
  exit 4
fi
echo "[feeder] next unsubmitted wave for ${JOB_TYPE}/${POOL}: ${NEXT}"

IN_SYSTEM=$(squeue --me -h -t RUNNING,PENDING 2>/dev/null | wc -l | tr -d ' ')
[ -n "${BCF_WAVE_FAKE_INSYSTEM:-}" ] && IN_SYSTEM="$BCF_WAVE_FAKE_INSYSTEM"
HEADROOM=$(( MAX_SUBMIT_JOBS - IN_SYSTEM ))
echo "[feeder] jobs in system ${IN_SYSTEM}/${MAX_SUBMIT_JOBS}; headroom ${HEADROOM}, floor ${MIN_HEADROOM}"
if [ "$HEADROOM" -lt "$MIN_HEADROOM" ]; then
  echo "[feeder] REFUSED: headroom ${HEADROOM} is under the floor of ${MIN_HEADROOM}."
  echo "[feeder]   Nothing was submitted; ${NEXT} stays next."
  exit 3
fi

# wave.sh is the authority on the caps and the CONTRACT split. Asking it first, with
# --check-only, means the feeder never has its own opinion about whether a wave may go.
echo "[feeder] asking wave.sh --check-only about ${NEXT}"
CHECK_OUT="$( cd "$SRC_ROOT" && bash bcf/wave.sh --type "$JOB_TYPE" --gpu-type "$POOL" \
              "${TREE_FLAGS[@]}" --check-only "bcf/waves/${NEXT}" 2>&1 )"
CHECK_RC=$?
printf '%s\n' "$CHECK_OUT" | sed 's/^/[feeder]   | /'
if [ "$CHECK_RC" -ne 0 ]; then
  echo "[feeder] REFUSED by wave.sh (exit ${CHECK_RC}): the split or a cap has no room."
  echo "[feeder]   Nothing was submitted; ${NEXT} stays next."
  exit 3
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[feeder] DRY RUN: wave.sh accepts ${NEXT} and this is where it would be submitted."
  echo "[feeder]   No sbatch was called and the state file was not written."
  exit 0
fi

echo "[feeder] SUBMITTING ${NEXT}"
SUBMIT_OUT="$( cd "$SRC_ROOT" && bash bcf/wave.sh --type "$JOB_TYPE" --gpu-type "$POOL" \
               "${TREE_FLAGS[@]}" "bcf/waves/${NEXT}" 2>&1 )"
SUBMIT_RC=$?
printf '%s\n' "$SUBMIT_OUT" | sed 's/^/[feeder]   | /'
IDS="$(printf '%s\n' "$SUBMIT_OUT" | grep -oE 'Submitted batch job [0-9]+' \
       | awk '{print $4}' | paste -sd, -)"
PLAN_COMMIT="$(printf '%s\n' "$SUBMIT_OUT" | grep -oE 'planning commit [0-9a-f]+' \
               | awk '{print $3}' | head -1)"
[ -n "$PLAN_COMMIT" ] || PLAN_COMMIT="unknown"

if [ -z "$IDS" ]; then
  echo "[feeder] NOTHING WAS SUBMITTED: wave.sh exited ${SUBMIT_RC} and printed no job id."
  echo "[feeder]   ${NEXT} is NOT recorded, so the next poll will offer it again."
  exit 1
fi
# Recorded even when wave.sh exited non-zero with SOME ids: those jobs exist, and a wave
# whose ids are not written down is a wave that gets submitted twice.
state_record "$NEXT" "$IDS" "$PLAN_COMMIT" "wave.sh exit ${SUBMIT_RC}" || exit 1
echo "[feeder] SUBMITTED ${NEXT} as ${IDS} (wave.sh exit ${SUBMIT_RC})"
[ "$SUBMIT_RC" -eq 0 ] || {
  echo "[feeder] WARNING: wave.sh exited ${SUBMIT_RC}; some rows may have been rejected."
  exit 1
}
exit 0
