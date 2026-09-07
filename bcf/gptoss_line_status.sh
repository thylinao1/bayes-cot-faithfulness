#!/bin/bash
# One line per gpt-oss serving-line configuration: queue state, exit code, the MoE
# backend the server selected, the batch-invariance rejection if it fired, the model
# load size and the last preflight line. Read-only; it submits nothing and cancels
# nothing. Configurations are named on the command line, defaulting to the three tests.
set -uo pipefail
CFGS=("$@")
[ ${#CFGS[@]} -gt 0 ] || CFGS=(a100-40-flagoff h100-47-flagon a100-80-flagon a100-40-flagoff-acc a100-40-flagoff-np4096)
for c in "${CFGS[@]}"; do
  D="$HOME/bcf/results/gptoss-line-${c}/gpt-oss-20b/arc_challenge/stated-hint"
  st=$(squeue --me -h -n "bcf-gptoss-${c}" -o "%T" 2>/dev/null | tr '\n' ' ')
  [ -n "$st" ] || st="GONE"
  ec="-"; [ -f "$D/exit_code.txt" ] && ec=$(tr -d '\n' < "$D/exit_code.txt")
  bk=$(grep -o "Using '[A-Z_]*' Mxfp4 MoE backend" "$D/server.log" 2>/dev/null | tail -1)
  bi=$(grep -o "backend: [A-Z_]*, reason: kernel does not support batch invariance" "$D/server.log" 2>/dev/null | head -2 | tr '\n' ';')
  ld=$(grep -o "Model loading took [0-9.]* GiB" "$D/server.log" 2>/dev/null | tail -1)
  pf=$(grep -o "\[preflight\].*" "$D/run.log" 2>/dev/null | tail -1 | cut -c1-160)
  ar=$(grep -o "clean accuracy: .*" "$D/run.log" 2>/dev/null | tail -1 | cut -c1-80)
  echo "${c} state=${st% } exit=${ec} | ${bk:-no-backend-line} | ${bi:-no-bi-reject} | ${ld:-not-loaded} | ${pf:-no-preflight-line} | ${ar:-no-accuracy-line}"
done
