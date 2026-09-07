#!/bin/bash
# Proof for the exploratory gpu-memory-utilization override in bcf/judge_serve.sbatch.
#
# The block under test is EXTRACTED from the sbatch file rather than retyped, so what runs
# here is the text that will run on the cluster. Three cases: the exploratory row gets the
# override, a run of record is refused with exit 12, and an unset override changes nothing.
set -uo pipefail
SBATCH="${1:?usage: util_guard_proof.sh <path to judge_serve.sbatch>}"
BLOCK="$(awk '/^  if \[ -n "\$\{BCF_JUDGE_GPU_UTIL:-\}" \]; then$/,/^  fi$/' "$SBATCH")"
if [ -z "$BLOCK" ]; then echo "EXTRACTION FAILED: the block is not in $SBATCH"; exit 99; fi
echo "--- the block under test, as extracted from $SBATCH"
echo "$BLOCK"

run_case() {
  local name="$1" serving="$2" override="$3"
  local script; script="$(mktemp)"
  {
    echo 'ts() { cat; }'
    echo 'LOG=/dev/null'
    echo 'key=gpt-oss-20b'
    echo 'JUDGE_GPU_UTIL=0.25'
    echo 'JUDGE_SERVING_LINE="mxfp4, shares the same a100-80 at gpu-memory-utilization 0.25"'
    [ -n "$serving" ] && echo "BCF_SERVING_LINE='${serving}'"
    [ -n "$override" ] && echo "BCF_JUDGE_GPU_UTIL='${override}'"
    echo "$BLOCK"
    echo 'echo "RESULT util=${JUDGE_GPU_UTIL}"'
  } > "$script"
  echo
  echo "--- ${name}"
  bash "$script"
  echo "exit=$?"
  rm -f "$script"
}

run_case "A. exploratory row, override 0.90: taken" "exploratory-h100-47" "0.90"
run_case "B. the pinned a100-80 run of record, same override: refused" "pinned-a100-80" "0.90"
run_case "C. no BCF_SERVING_LINE at all, override set: refused" "" "0.90"
run_case "D. exploratory row, no override: the table fraction stands" "exploratory-h100-47" ""
