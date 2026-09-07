#!/bin/bash
# Proof that a jury judge job cancelled or killed mid-variant records a NON-ZERO exit code.
#
# Runs on the Mac (bash 3.2) and on the cluster (bash 5). No GPU, no network, no Slurm: the
# real bcf/judge_serve.sbatch is invoked with a stub env.sh and stub vllm, curl, du and
# python on PATH, so the file under test is the file that runs on the cluster.
#
# Six unit scenarios on bcf/exit_guard.sh, three integration scenarios on the real sbatch
# script, and a LEGACY scenario that reruns the cancel against the pre-fix logic and asserts
# it writes 0. That last one is what makes this test able to fail: without it, a test that
# only ever sees the fixed script cannot tell a working guard from a lucky one.
#
#   bcf/test_exit_guard.sh          # prints one line per scenario, exit 0 if all pass
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GUARD="${REPO}/bcf/exit_guard.sh"
SBATCH="${REPO}/bcf/judge_serve.sbatch"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/bcf-exit-guard-XXXXXX")"
trap '[ -n "${BCF_TEST_KEEP:-}" ] || rm -rf "$TMP"' EXIT
[ -n "${BCF_TEST_KEEP:-}" ] && echo "[keep] scratch at $TMP"

N_PASS=0
N_FAIL=0
check() {  # name  want  got
  if [ "$3" = "$2" ]; then
    printf 'PASS  %-46s exit_code=%s\n' "$1" "$3"
    N_PASS=$(( N_PASS + 1 ))
  else
    printf 'FAIL  %-46s want=%s got=%s\n' "$1" "$2" "$3"
    N_FAIL=$(( N_FAIL + 1 ))
  fi
}
read_code() { cat "$1" 2>/dev/null | tr -d '[:space:]'; }

# ---------------------------------------------------------------- unit scenarios
cat > "$TMP/stubjob.sh" <<'EOS'
#!/bin/bash
set -uo pipefail
source "$BCF_GUARD"
EXIT_FILE="$1"
BEHAVIOUR="$2"
STAGE_FILE="$(dirname "$EXIT_FILE")/stage_exit_code.txt"
finish() {
  raw=$?
  code="$(bcf_exit_guard_code "$raw")"
  bcf_exit_guard_write "$code"
  exit "$code"
}
bcf_exit_guard_init "$EXIT_FILE"
trap finish EXIT
bcf_exit_guard_open_stage "$STAGE_FILE"
case "$BEHAVIOUR" in
  complete)       bcf_exit_guard_complete; exit 0 ;;
  threshold_fail) bcf_exit_guard_complete; exit 1 ;;
  crash7)         exit 7 ;;
  zero_no_marker) exit 0 ;;
  hang)           echo $$ > "$(dirname "$EXIT_FILE")/ready"; while true; do sleep 0.2; done ;;
esac
EOS
chmod +x "$TMP/stubjob.sh"

cat > "$TMP/legacyjob.sh" <<'EOS'
#!/bin/bash
# The pre-fix shape: one EXIT trap that records whatever $? happened to be.
set -uo pipefail
EXIT_FILE="$1"
BEHAVIOUR="$2"
finish() { code=$?; echo "$code" > "$EXIT_FILE"; exit "$code"; }
trap finish EXIT
case "$BEHAVIOUR" in
  complete)       exit 0 ;;
  zero_no_marker) exit 0 ;;
  hang)           echo $$ > "$(dirname "$EXIT_FILE")/ready"; while true; do sleep 0.2; done ;;
esac
EOS
chmod +x "$TMP/legacyjob.sh"

run_unit() {  # name behaviour want [--kill SIG]
  name="$1"; behaviour="$2"; want="$3"; sig="${4:-}"
  d="$TMP/unit-$name"; mkdir -p "$d"
  if [ -n "$sig" ]; then
    BCF_GUARD="$GUARD" "$TMP/stubjob.sh" "$d/exit_code.txt" "$behaviour" >/dev/null 2>&1 &
    jobpid=$!
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      [ -s "$d/ready" ] && break
      sleep 0.2
    done
    kill -"$sig" "$jobpid" 2>/dev/null
    kill -"$sig" "$(cat "$d/ready" 2>/dev/null)" 2>/dev/null
    wait "$jobpid" 2>/dev/null
  else
    BCF_GUARD="$GUARD" "$TMP/stubjob.sh" "$d/exit_code.txt" "$behaviour" >/dev/null 2>&1
  fi
  check "unit $name" "$want" "$(read_code "$d/exit_code.txt")"
}

echo "--- unit scenarios on bcf/exit_guard.sh ---"
run_unit complete        complete        0
run_unit threshold_fail  threshold_fail  1
run_unit crash7          crash7          7
run_unit zero_no_marker  zero_no_marker  250
run_unit cancelled_term  hang            143 TERM
run_unit hard_kill       hang            255 KILL

# the stage file of a cancelled run must be non-zero as well
d="$TMP/unit-cancelled_term"
check "unit cancelled_term stage file" 143 "$(read_code "$d/stage_exit_code.txt")"

echo "--- legacy discriminator (this is what makes the test able to fail) ---"
for behaviour in zero_no_marker; do
  d="$TMP/legacy-$behaviour"; mkdir -p "$d"
  "$TMP/legacyjob.sh" "$d/exit_code.txt" "$behaviour" >/dev/null 2>&1
  check "legacy $behaviour records a success" 0 "$(read_code "$d/exit_code.txt")"
done
d="$TMP/legacy-cancelled"; mkdir -p "$d"
"$TMP/legacyjob.sh" "$d/exit_code.txt" hang >/dev/null 2>&1 &
jobpid=$!
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  [ -s "$d/ready" ] && break
  sleep 0.2
done
kill -TERM "$jobpid" 2>/dev/null
kill -TERM "$(cat "$d/ready" 2>/dev/null)" 2>/dev/null
wait "$jobpid" 2>/dev/null
check "legacy cancelled records a success" 0 "$(read_code "$d/exit_code.txt")"

# ------------------------------------------------ integration on the real sbatch
STUBS="$TMP/stubs"; mkdir -p "$STUBS"
cat > "$STUBS/du" <<'EOS'
#!/bin/bash
echo "0G ${!#}"
EOS
cat > "$STUBS/curl" <<'EOS'
#!/bin/bash
exit 0
EOS
cat > "$STUBS/vllm" <<'EOS'
#!/bin/bash
exec sleep 300
EOS
cat > "$STUBS/python" <<'EOS'
#!/bin/bash
for arg in "$@"; do
  case "$arg" in
    experiments.jury.gate)
      case "${BCF_TEST_GATE:-ok}" in
        crash7) exit 7 ;;
        hang)   echo $$ > "${BCF_TEST_READY}"; while true; do sleep 0.2; done ;;
        *)      exit 0 ;;
      esac
      ;;
  esac
done
exec python3 "$@"
EOS
chmod +x "$STUBS"/*

cat > "$TMP/env_stub.sh" <<'EOS'
#!/bin/bash
set -uo pipefail
export SCRATCH="${BCF_TEST_SCRATCH:?}"
mkdir -p "$SCRATCH/hf"
export HF_HOME="$SCRATCH/hf"
bcf_assert_devices() { echo "[stub] devices ok for tp=$1"; return 0; }
bcf_revision() { echo "${BCF_TEST_LIVE_SHA:-f50dbad2c84590ca17dc51e207c34321b65ff14b}"; }
EOS

# The legacy copy of the real sbatch: the same file with every guard call removed and the
# pre-fix finish() restored. Built from the file under test so it cannot drift from it.
python3 - "$SBATCH" "$TMP/judge_serve_legacy.sbatch" <<'PY'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
s = s.replace('bcf_exit_guard_init "$EXIT_FILE"', ': # legacy: no guard')
s = s.replace('  bcf_exit_guard_open_stage "${RUN_OUT}/exit_code.txt"', '  : # legacy')
s = s.replace('\nbcf_exit_guard_complete\n', '\n')
s = s.replace('  code="$(bcf_exit_guard_code "$raw")"\n  bcf_exit_guard_write "$code"\n',
              '  code="$raw"\n  echo "$code" > "$EXIT_FILE"\n')
assert 'bcf_exit_guard_code' not in s and 'bcf_exit_guard_init' not in s, "legacy build failed"
open(dst, "w").write(s)
PY
chmod +x "$TMP/judge_serve_legacy.sbatch"

run_sbatch() {  # name script gate_behaviour want [kill]
  name="$1"; script="$2"; gate="$3"; want="$4"; do_kill="${5:-}"
  root="$TMP/int-$name"; mkdir -p "$root"
  out="$root/results"
  ready="$root/gate.pid"
  env_common=(
    "BCF_REPO=$REPO"
    "BCF_ENV_SH=$TMP/env_stub.sh"
    "BCF_JUDGES=llama-3.3-70b-fp8"
    "BCF_GATE_ITEMS=$REPO/experiments/results/jury-gate/gate_items.jsonl"
    "BCF_OUT_ROOT=$out"
    "BCF_OUT_SLUG=testjudge"
    "BCF_Q1_PROMPT=a"
    "BCF_JUDGE_CACHE=$root/cache"
    "BCF_TEST_SCRATCH=$root/scratch"
    "BCF_TEST_GATE=$gate"
    # Both of these keep an optional bash array non-empty. macOS ships bash 3.2, where
    # "${EMPTY_ARRAY[@]}" is an unbound-variable error under set -u; the cluster runs
    # bash 5.2 where it expands to nothing. Setting them means this test exercises the
    # gate loop on either shell rather than dying in the expansion before it gets there.
    "BCF_GATE_LIMIT=1"
    "BCF_SERVING_LINE=test-harness"
    "BCF_TEST_READY=$ready"
    "SLURM_JOB_ID=testjob"
    "PATH=$STUBS:$PATH"
  )
  exit_file="$out/testjudge/arc_challenge/stated-hint/exit_code.txt"
  if [ -n "$do_kill" ]; then
    env "${env_common[@]}" bash "$script" >"$root/stdout.txt" 2>&1 &
    jobpid=$!
    for _ in $(seq 1 60); do
      [ -s "$ready" ] && break
      sleep 0.5
    done
    kill -TERM "$jobpid" 2>/dev/null
    kill -TERM "$(cat "$ready" 2>/dev/null)" 2>/dev/null
    wait "$jobpid" 2>/dev/null
  else
    env "${env_common[@]}" bash "$script" >"$root/stdout.txt" 2>&1
  fi
  check "sbatch $name" "$want" "$(read_code "$exit_file")"
}

echo "--- integration scenarios on the real bcf/judge_serve.sbatch ---"
run_sbatch gate_ok            "$SBATCH" ok     0
run_sbatch gate_crash_midrun  "$SBATCH" crash7 7
run_sbatch cancelled_midrun   "$SBATCH" hang   143 kill
echo "--- the same cancel against the pre-fix script ---"
run_sbatch legacy_cancelled   "$TMP/judge_serve_legacy.sbatch" hang 0 kill

echo
echo "passed ${N_PASS}, failed ${N_FAIL}"
[ "$N_FAIL" -eq 0 ]
