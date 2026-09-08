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
#   4  nothing left: every wave for this pool is already in the state file, or (--held-only)
#      no recorded wave for this pool/type is currently holding any row
#   2  usage or configuration error (a missing directory, an unknown pool, a malformed
#      line in the hold file)
#   1  a submission was attempted and something went wrong; read the output
#   5  the next wave was recorded as SKIPPED: every one of its rows is on the hold list,
#      so nothing was submitted but the wave is not offered again either
#
#   bcf/wave_feeder.sh                                   next sweep wave on a100-40
#   bcf/wave_feeder.sh --pool a100-40 --type enrich      next enrichment-pass wave
#   bcf/wave_feeder.sh --dry-run                         decide and print, submit nothing
#   bcf/wave_feeder.sh status                            what the state file holds
#   bcf/wave_feeder.sh adopt <wave.tsv> <id,id,...>      record a wave submitted by hand
#   bcf/wave_feeder.sh held-only                         build+submit exactly the rows
#                                                         every recorded wave held
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
#
# THE HOLD LIST (bcf/waves/HOLD_MODELS.txt, read by bcf/hold_filter.py) names HF model
# ids whose rows must not go out yet, each because of a specific ruling still open. Before
# asking wave.sh about a wave, this script filters that wave's rows against the hold list:
#   * nothing held             -> the wave is submitted exactly as before this existed.
#   * some rows held           -> a filtered copy, header and comments kept, is written to
#     bcf/waves/partial/<wave>-<n>rows.tsv, and THAT file is what gets checked and
#     submitted; the state entry records which models went and which were held.
#   * every row held           -> nothing is submitted; the wave is recorded SKIPPED
#     (exit 5) so the next poll moves on to the wave after it rather than offering the
#     same fully-held wave forever.
# `held-only` reads back every recorded wave's held rows for this pool and job type,
# recombines them, and checks them AGAIN against the current hold list (a hold that has
# only partly lifted still holds what it names), so lifting one row's hold and running
# `held-only` submits exactly the rows that are now clear, from every wave already past.
# Three things it does NOT do, each of them a defect fixed on 2026-09-08:
#   * it does not reissue the row text the state file froze when the row was first held.
#     Every class-2 row gained BCF_REASONING_MODE=off after those rows were recorded, and
#     a wave rebuilt from the frozen text would run in the reasoning mode R12(1) rejected
#     and record it truthfully, so nothing downstream would catch it. The row's identity
#     is its (model, substrate, cue) triple; its content comes from the manifest today.
#   * it does not build one wave out of every clear row. At most BCF_FEEDER_MAX_HELD_ROWS
#     (default 8, the a100-40 slice count) go in a wave; the rest are DEFERRED, the run
#     still exits 0, and running held-only again takes the next batch.
#   * it does not clear a source wave's whole held_rows list. Only the rows this wave
#     actually submitted are cleared, so deferred and still-held rows survive for the
#     next run instead of being dropped with no record that they existed.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${BCF_FEEDER_SRC:-$(cd "${SELF_DIR}/.." && pwd)}"
STATE="${BCF_FEEDER_STATE:-$HOME/bcf/feeder-state.json}"
POOL="a100-40"
JOB_TYPE="sweep"
MIN_HEADROOM="${BCF_FEEDER_MIN_HEADROOM:-8}"
MAX_SUBMIT_JOBS=32
# The most rows one held-only wave may carry. The a100-40 pool has 8 MIG slices and
# bcf/wave.sh counts its cap as running cards plus the wave's own cards, so a held-only
# rebuild of every clear row (30 were recorded on 2026-09-08) is a wave wave.sh will never
# accept. Rows past this cap are deferred, stay recorded against their source waves, and
# are taken by the next held-only run. 0 or less means no cap.
MAX_HELD_ROWS="${BCF_FEEDER_MAX_HELD_ROWS:-8}"
DRY_RUN=0
GLOB=""
CMD="next"
REPO_TREE=""     # passed through to wave.sh; only for a tree that is already synced, or
                 # for a --check-only proof run from a directory that is not a checkout
HOLD_FILE="${BCF_FEEDER_HOLD_FILE:-}"   # resolved against WAVES_DIR below once known
ADOPT_ARGS=()

while [ $# -gt 0 ]; do
  arg="$1"
  val=""
  case "$arg" in
    --*=*) val="${arg#*=}"; arg="${arg%%=*}"; shift ;;
    --pool|--type|--state|--src|--glob|--min-headroom|--repo-tree|--hold-file) val="${2:-}"; shift 2 ;;
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
    --hold-file) HOLD_FILE="$val" ;;
    --dry-run|--check-only) DRY_RUN=1 ;;
    -h|--help) sed -n '2,74p' "$0"; exit 0 ;;
    -*) echo "[feeder] unknown option '${arg}'" >&2; exit 2 ;;
    next|status|adopt|list|held-only) CMD="$arg" ;;
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

[ -n "$HOLD_FILE" ] || HOLD_FILE="${WAVES_DIR}/HOLD_MODELS.txt"
HOLD_FILTER="${SRC_ROOT}/bcf/hold_filter.py"
[ -f "$HOLD_FILTER" ] || { echo "[feeder] no bcf/hold_filter.py under ${SRC_ROOT}" >&2; exit 2; }

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

state_record() {  # $1 wave basename  $2 comma ids  $3 plan commit  $4 note
                   # $5 (optional) path to an extra-fields JSON file written by
                   #    hold_filter.py: status, submitted_models, held_models,
                   #    held_rows, partial_file. Empty/missing defaults to a wave
                   #    with nothing held, exactly the pre-hold-list shape.
  "$PY" - "$STATE" "$1" "$2" "$3" "$4" "$JOB_TYPE" "$POOL" "${5:-}" <<'PYEOF'
import json, os, sys, tempfile, datetime
path, wave, ids, commit, note, job_type, pool, extra_path = sys.argv[1:9]
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
extra = {
    "status": "submitted",
    "submitted_models": [],
    "held_models": [],
    "held_rows": [],
    "partial_file": None,
}
if extra_path:
    extra.update(json.load(open(extra_path)))
waves[wave] = {
    "submitted_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "job_ids": [x for x in ids.split(",") if x],
    "plan_commit": commit,
    "job_type": job_type,
    "gpu_type": pool,
    "note": note,
    **extra,
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

state_lift_held() {  # $1 the held-only wave's name  $2 path to its sources JSON
                      # ({"source_waves": [...]} from hold_filter.py collect)
                      # $3 path to the compose extra JSON, whose submitted_rows say which
                      #    rows actually went out this time
                      #
                      # Only the rows that WERE SUBMITTED are cleared. A held-only wave is
                      # capped at BCF_FEEDER_MAX_HELD_ROWS rows, so a run can leave rows
                      # deferred, and a row still on the hold list was never going out at
                      # all; clearing every held_row of every source wave would drop both
                      # kinds on the floor with no record anywhere that they existed.
  "$PY" - "$STATE" "$1" "$2" "$3" <<'PYEOF'
import json, os, sys, tempfile
path, held_only_wave, sources_path, extra_path = sys.argv[1:5]
try:
    d = json.load(open(path))
except Exception as exc:
    sys.stderr.write(f"[feeder] STATE FILE UNREADABLE ({exc}); nothing lifted\n")
    raise SystemExit(9)
sources = json.load(open(sources_path)).get("source_waves", [])
extra = json.load(open(extra_path))
submitted = extra.get("submitted_rows")
if submitted is None:
    sys.stderr.write("[feeder] the compose output names no submitted_rows; nothing lifted, "
                     "every source wave keeps its held rows\n")
    raise SystemExit(10)
# A row's identity is its first three columns (model, substrate, cue). The KEY=VALUE tail
# of what was submitted is the CURRENT manifest text, which is the whole point of the
# rebuild, so it will not match the recorded text and must not be compared.
sent = {tuple(r.split("\t")[:3]) for r in submitted}
waves = d.get("waves", {})
n_rows_lifted = 0
n_waves = 0
n_still_recorded = 0
for w in sources:
    entry = waves.get(w)
    if not entry or not entry.get("held_rows"):
        continue
    before = list(entry["held_rows"])
    remaining = [r for r in before if tuple(r.split("\t")[:3]) not in sent]
    n_still_recorded += len(remaining)
    if len(remaining) == len(before):
        continue
    # held_models stays as the historical record of what WAS held, even for a wave whose
    # held_rows is now empty; held_lifted_by names the wave that carried these rows out.
    entry.setdefault("held_lifted_by", [])
    if held_only_wave not in entry["held_lifted_by"]:
        entry["held_lifted_by"].append(held_only_wave)
    entry["held_rows"] = remaining
    n_rows_lifted += len(before) - len(remaining)
    n_waves += 1
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".feeder-state.")
with os.fdopen(fd, "w") as fh:
    json.dump(d, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
print(f"[feeder] lifted {n_rows_lifted} held row(s) from {n_waves} source wave(s), "
      f"consumed by {held_only_wave}; {n_still_recorded} row(s) stay recorded as held")
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

if [ "$CMD" = "held-only" ]; then
  LOCK="${STATE}.lock"
  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[feeder] REFUSED: another feeder holds ${LOCK}. Two pollers submitting at once is"
    echo "[feeder]   exactly the double-submission this script exists to prevent. If no"
    echo "[feeder]   feeder is running, remove that directory by hand."
    exit 3
  fi
  ROWS_FILE="$(mktemp "${STATE}.held-rows.XXXXXX" 2>/dev/null || mktemp)"
  SOURCES_JSON="$(mktemp "${STATE}.held-sources.XXXXXX" 2>/dev/null || mktemp)"
  EXTRA_JSON="$(mktemp "${STATE}.extra.XXXXXX" 2>/dev/null || mktemp)"
  trap 'rm -f "$ROWS_FILE" "$SOURCES_JSON" "$EXTRA_JSON"; rmdir "$LOCK" 2>/dev/null' EXIT

  # Step 1: gather every currently-held row recorded for this pool and job type, from
  # every kind of recorded wave (an original sweep/enrich wave, or an earlier held-only
  # wave whose own rebuild still left something held).
  # --waves-dir is what makes this read the CURRENT manifest row for each recorded triple
  # instead of the text frozen when the row was first held.
  COLLECT_OUT="$("$PY" "$HOLD_FILTER" collect "$STATE" "$POOL" "$JOB_TYPE" \
                 --waves-dir "$WAVES_DIR" \
                 --rows-out "$ROWS_FILE" --sources-out "$SOURCES_JSON" 2>&1)"
  COLLECT_RC=$?
  if [ "$COLLECT_RC" -ne 0 ]; then
    echo "[feeder] REFUSED: could not read ${STATE} for held rows." >&2
    printf '%s\n' "$COLLECT_OUT" | sed 's/^/[feeder]   | /' >&2
    exit 2
  fi
  N_SOURCE_WAVES="" N_HELD_ROWS_COLLECTED="" N_STALE=""
  while IFS='=' read -r k v; do
    case "$k" in
      N_SOURCE_WAVES) N_SOURCE_WAVES="$v" ;;
      N_HELD_ROWS) N_HELD_ROWS_COLLECTED="$v" ;;
      N_STALE) N_STALE="$v" ;;
    esac
  done <<< "$COLLECT_OUT"
  [ -n "$N_STALE" ] || N_STALE=0
  echo "[feeder] collected ${N_HELD_ROWS_COLLECTED} held row(s) from ${N_SOURCE_WAVES} recorded ${JOB_TYPE}/${POOL} wave(s)"
  if [ "$N_STALE" -gt 0 ]; then
    echo "[feeder]   ${N_STALE} row(s) are NOT in their manifest any more and kept their recorded text:"
    printf '%s\n' "$COLLECT_OUT" | grep -F 'WARNING:' | sed 's/^/[feeder]   | /'
  fi
  if [ "$N_HELD_ROWS_COLLECTED" -eq 0 ]; then
    echo "[feeder] NOTHING TO DO: no recorded ${JOB_TYPE}/${POOL} wave is currently holding any row."
    exit 4
  fi

  # Step 2: name this held-only wave the next unused held-<pool>-<type>-NN, so it is a
  # normal state-file entry (never resubmitted once recorded) and, if it still leaves
  # something held, a FUTURE held-only run collects from it in turn.
  N_PRIOR=$(state_get_submitted | grep -c "^held-${POOL}-${JOB_TYPE}-" 2>/dev/null || true)
  [ -n "$N_PRIOR" ] || N_PRIOR=0
  HELD_NAME="held-${POOL}-${JOB_TYPE}-$(printf '%02d' $((N_PRIOR + 1)))"

  # Step 3: check the collected rows against the CURRENT hold list. A hold that lifted
  # for four models and not the fifth still holds the fifth here.
  COMPOSE_OUT="$("$PY" "$HOLD_FILTER" compose "$ROWS_FILE" "$HOLD_FILE" \
                 --waves-dir "$WAVES_DIR" --name "$HELD_NAME" --max-rows "$MAX_HELD_ROWS" \
                 --extra-out "$EXTRA_JSON" 2>&1)"
  COMPOSE_RC=$?
  if [ "$COMPOSE_RC" -ne 0 ]; then
    echo "[feeder] REFUSED: the hold list ${HOLD_FILE} could not be read." >&2
    printf '%s\n' "$COMPOSE_OUT" | sed 's/^/[feeder]   | /' >&2
    exit 2
  fi
  N_TOTAL="" N_KEPT="" N_CLEAR="" N_DEFERRED="" N_HELD=""
  FSTATUS="" PARTIAL_FILE="" HELD_MODELS="" SUBMITTED_MODELS=""
  while IFS='=' read -r k v; do
    case "$k" in
      N_TOTAL) N_TOTAL="$v" ;;
      N_KEPT) N_KEPT="$v" ;;
      N_CLEAR) N_CLEAR="$v" ;;
      N_DEFERRED) N_DEFERRED="$v" ;;
      N_HELD) N_HELD="$v" ;;
      STATUS) FSTATUS="$v" ;;
      PARTIAL_FILE) PARTIAL_FILE="$v" ;;
      HELD_MODELS) HELD_MODELS="$v" ;;
      SUBMITTED_MODELS) SUBMITTED_MODELS="$v" ;;
    esac
  done <<< "$COMPOSE_OUT"
  [ -n "$N_CLEAR" ] || N_CLEAR="$N_KEPT"
  [ -n "$N_DEFERRED" ] || N_DEFERRED=0
  echo "[feeder] ${HELD_NAME}: ${N_TOTAL} row(s) collected, ${N_CLEAR} now clear, ${N_KEPT} in this wave (cap ${MAX_HELD_ROWS}), ${N_DEFERRED} deferred, ${N_HELD} still held"
  [ "$N_HELD" -gt 0 ] && echo "[feeder]   still held: ${HELD_MODELS}"

  if [ "$N_KEPT" -eq 0 ]; then
    echo "[feeder] EVERY collected row is still on the hold list; nothing to submit."
    exit 4
  fi

  SUBMIT_REL="bcf/waves/partial/${PARTIAL_FILE}"
  echo "[feeder] asking wave.sh --check-only about ${SUBMIT_REL}"
  CHECK_OUT="$( cd "$SRC_ROOT" && bash bcf/wave.sh --type "$JOB_TYPE" --gpu-type "$POOL" \
                "${TREE_FLAGS[@]}" --check-only "$SUBMIT_REL" 2>&1 )"
  CHECK_RC=$?
  printf '%s\n' "$CHECK_OUT" | sed 's/^/[feeder]   | /'
  if [ "$CHECK_RC" -ne 0 ]; then
    echo "[feeder] REFUSED by wave.sh (exit ${CHECK_RC}): the split or a cap has no room."
    echo "[feeder]   Nothing was submitted; the source waves' held rows are unchanged."
    exit 3
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[feeder] DRY RUN: wave.sh accepts ${SUBMIT_REL} (${HELD_NAME}) and this is where it would be submitted."
    echo "[feeder]   No sbatch was called and the state file was not written."
    [ "$N_DEFERRED" -gt 0 ] && \
      echo "[feeder] held-only: ${N_DEFERRED} rows deferred, run held-only again when slices free up"
    exit 0
  fi

  echo "[feeder] SUBMITTING ${SUBMIT_REL} as ${HELD_NAME}"
  SUBMIT_OUT="$( cd "$SRC_ROOT" && bash bcf/wave.sh --type "$JOB_TYPE" --gpu-type "$POOL" \
                 "${TREE_FLAGS[@]}" "$SUBMIT_REL" 2>&1 )"
  SUBMIT_RC=$?
  printf '%s\n' "$SUBMIT_OUT" | sed 's/^/[feeder]   | /'
  IDS="$(printf '%s\n' "$SUBMIT_OUT" | grep -oE 'Submitted batch job [0-9]+' \
         | awk '{print $4}' | paste -sd, -)"
  PLAN_COMMIT="$(printf '%s\n' "$SUBMIT_OUT" | grep -oE 'planning commit [0-9a-f]+' \
                 | awk '{print $3}' | head -1)"
  [ -n "$PLAN_COMMIT" ] || PLAN_COMMIT="unknown"

  if [ -z "$IDS" ]; then
    echo "[feeder] NOTHING WAS SUBMITTED: wave.sh exited ${SUBMIT_RC} and printed no job id."
    echo "[feeder]   Nothing was recorded; the source waves' held rows are unchanged."
    exit 1
  fi
  state_record "$HELD_NAME" "$IDS" "$PLAN_COMMIT" \
    "held-only, sources: $(cat "$SOURCES_JSON")" "$EXTRA_JSON" || exit 1
  echo "[feeder] SUBMITTED ${HELD_NAME} as ${IDS} (wave.sh exit ${SUBMIT_RC})"
  state_lift_held "$HELD_NAME" "$SOURCES_JSON" "$EXTRA_JSON" \
    || echo "[feeder] WARNING: lift bookkeeping failed; a source wave's held_rows may still show these rows" >&2
  # A capped wave is a normal, successful outcome, not a failure: the deferred rows are
  # still recorded against their source waves and the next held-only run takes them, so
  # the exit code stays 0 and the caller's poll loop keeps going.
  if [ "$N_DEFERRED" -gt 0 ]; then
    echo "[feeder] held-only: ${N_DEFERRED} rows deferred, run held-only again when slices free up"
  fi
  [ "$SUBMIT_RC" -eq 0 ] || {
    echo "[feeder] WARNING: wave.sh exited ${SUBMIT_RC}; some rows may have been rejected."
    exit 1
  }
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

# The hold list, before wave.sh ever sees this wave. hold_filter.py reads column 1 of
# every data row, keeps every '#' and blank line verbatim, and writes a filtered copy
# under bcf/waves/partial/ when only SOME rows are held; nothing is written when nothing
# is held, so the pre-hold-list wave file is what gets checked and submitted unchanged.
EXTRA_JSON="$(mktemp "${STATE}.extra.XXXXXX" 2>/dev/null || mktemp)"
trap 'rm -f "$EXTRA_JSON"; rmdir "$LOCK" 2>/dev/null' EXIT
FILTER_OUT="$("$PY" "$HOLD_FILTER" filter "${WAVES_DIR}/${NEXT}" "$HOLD_FILE" \
              --waves-dir "$WAVES_DIR" --extra-out "$EXTRA_JSON" 2>&1)"
FILTER_RC=$?
if [ "$FILTER_RC" -ne 0 ]; then
  echo "[feeder] REFUSED: the hold list ${HOLD_FILE} could not be read." >&2
  printf '%s\n' "$FILTER_OUT" | sed 's/^/[feeder]   | /' >&2
  exit 2
fi
N_TOTAL="" N_KEPT="" N_HELD="" FSTATUS="" PARTIAL_FILE="" HELD_MODELS="" SUBMITTED_MODELS=""
while IFS='=' read -r k v; do
  case "$k" in
    N_TOTAL) N_TOTAL="$v" ;;
    N_KEPT) N_KEPT="$v" ;;
    N_HELD) N_HELD="$v" ;;
    STATUS) FSTATUS="$v" ;;
    PARTIAL_FILE) PARTIAL_FILE="$v" ;;
    HELD_MODELS) HELD_MODELS="$v" ;;
    SUBMITTED_MODELS) SUBMITTED_MODELS="$v" ;;
  esac
done <<< "$FILTER_OUT"
echo "[feeder] hold list ${HOLD_FILE}: ${NEXT} has ${N_TOTAL} row(s), ${N_KEPT} submittable, ${N_HELD} held"
[ "$N_HELD" -gt 0 ] && echo "[feeder]   held: ${HELD_MODELS}"

if [ "$N_KEPT" -eq 0 ]; then
  echo "[feeder] WAVE ${NEXT} IS FULLY HELD: every row is on the hold list; nothing to submit."
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[feeder] DRY RUN: would record ${NEXT} as SKIPPED (all rows held); nothing written."
    exit 0
  fi
  SKIP_COMMIT="$(git -C "$SRC_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  state_record "$NEXT" "" "$SKIP_COMMIT" "all ${N_TOTAL} row(s) held by ${HOLD_FILE}" "$EXTRA_JSON" || exit $?
  echo "[feeder] RECORDED ${NEXT} as SKIPPED (all rows held)"
  exit 5
fi

SUBMIT_REL="bcf/waves/${NEXT}"
if [ "$N_HELD" -gt 0 ]; then
  SUBMIT_REL="bcf/waves/partial/${PARTIAL_FILE}"
  echo "[feeder] partial wave written: ${SUBMIT_REL} (${N_KEPT} of ${N_TOTAL} rows)"
fi

# wave.sh is the authority on the caps and the CONTRACT split. Asking it first, with
# --check-only, means the feeder never has its own opinion about whether a wave may go.
echo "[feeder] asking wave.sh --check-only about ${SUBMIT_REL}"
CHECK_OUT="$( cd "$SRC_ROOT" && bash bcf/wave.sh --type "$JOB_TYPE" --gpu-type "$POOL" \
              "${TREE_FLAGS[@]}" --check-only "$SUBMIT_REL" 2>&1 )"
CHECK_RC=$?
printf '%s\n' "$CHECK_OUT" | sed 's/^/[feeder]   | /'
if [ "$CHECK_RC" -ne 0 ]; then
  echo "[feeder] REFUSED by wave.sh (exit ${CHECK_RC}): the split or a cap has no room."
  echo "[feeder]   Nothing was submitted; ${NEXT} stays next."
  exit 3
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[feeder] DRY RUN: wave.sh accepts ${SUBMIT_REL} and this is where it would be submitted."
  echo "[feeder]   No sbatch was called and the state file was not written."
  exit 0
fi

echo "[feeder] SUBMITTING ${SUBMIT_REL}"
SUBMIT_OUT="$( cd "$SRC_ROOT" && bash bcf/wave.sh --type "$JOB_TYPE" --gpu-type "$POOL" \
               "${TREE_FLAGS[@]}" "$SUBMIT_REL" 2>&1 )"
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
# whose ids are not written down is a wave that gets submitted twice. Recorded under NEXT
# (the original wave name), not the partial filename, so state lookups and --held-only's
# own collection stay keyed by the one name the wave is known by everywhere else.
state_record "$NEXT" "$IDS" "$PLAN_COMMIT" "wave.sh exit ${SUBMIT_RC}" "$EXTRA_JSON" || exit 1
echo "[feeder] SUBMITTED ${NEXT} as ${IDS} (wave.sh exit ${SUBMIT_RC})"
[ "$SUBMIT_RC" -eq 0 ] || {
  echo "[feeder] WARNING: wave.sh exited ${SUBMIT_RC}; some rows may have been rejected."
  exit 1
}
exit 0
