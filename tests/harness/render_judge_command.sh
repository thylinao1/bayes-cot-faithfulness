#!/bin/bash
# Render what bcf/judge_serve.sbatch would ask the jury to do, without a cluster.
#
# WHY this exists. The BCF_JURY_MODE=audit branch has to be ADDITIVE: with the variable
# unset, the gate lane must run exactly the command it ran before the branch existed. The
# script has no dry-run mode, so this harness supplies one from the outside: stub
# executables on PATH for `vllm`, `du`, `stat` and for the two `python -m
# experiments.jury.*` calls, a stub bcf/env.sh through the script's own BCF_ENV_SH hook,
# and the real bcf/exit_guard.sh. What comes out is the argv the script would have handed
# the judging module, and the exit code its guard recorded, with the sandbox root
# normalised to <ROOT> so the fixture is machine independent.
#
# Usage: render_judge_command.sh <output-file> [VAR=value ...]
#   Any VAR=value pairs are exported into the run, which is how a caller asks for the
#   default gate case (none) or the audit case (BCF_JURY_MODE=audit BCF_SWEEP_ITEMS=...).
#   BCF_RENDER_SCRIPT names the script to render, which is how the same harness renders
#   this file's own pre-change version out of git.

set -uo pipefail

OUT="${1:?usage: render_judge_command.sh <output-file> [VAR=value ...]}"
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
SCRIPT="${BCF_RENDER_SCRIPT:-${REPO}/bcf/judge_serve.sbatch}"

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/bcf-judge-render.XXXXXX")"
if [ "${BCF_RENDER_KEEP:-0}" != "1" ]; then trap 'rm -rf "$ROOT"' EXIT; else echo "[render] root ${ROOT}" >&2; fi
BIN="${ROOT}/bin"
mkdir -p "$BIN"

# The judging call is the command under test. Everything else python does here (the judge
# spec, the serve_meta writer) runs on the real interpreter, so the metadata in the render
# is the metadata the script produces.
cat > "${BIN}/python" <<'PYSTUB'
#!/bin/bash
if [ "${1:-}" = "-m" ]; then
  case "${2:-}" in
    experiments.jury.gate|experiments.jury.runner)
      : > "${BCF_RENDER_ARGV:?}"
      out=""
      prev=""
      for a in "$@"; do
        printf '%s\n' "$a" >> "${BCF_RENDER_ARGV}"
        [ "$prev" = "--out" ] && out="$a"
        prev="$a"
      done
      # The completion marker each mode's own module writes. The script asks
      # bcf_gate_stage_status for a DIFFERENT file per mode, and a stub that wrote
      # neither would make both modes look like a crash.
      if [ -n "$out" ] && [ "${BCF_RENDER_WRITE_MARKER:-1}" = "1" ]; then
        case "${2}" in
          experiments.jury.gate) printf '{}\n' > "${out}/gate_report.json" ;;
          experiments.jury.runner) printf '{}\n' > "${out}/run_summary.json" ;;
        esac
      fi
      exit "${BCF_RENDER_JUDGE_STATUS:-0}"
      ;;
  esac
fi
exec "${BCF_RENDER_PYTHON:-python3}" "$@"
PYSTUB

# The server itself is backgrounded by the script and killed by its own EXIT trap, so
# whether this stub finishes before the kill is a race and its argv is NOT rendered. What
# vLLM is asked to serve is pinned by the section 6.1 judge table and bcf/judge_spec.py,
# neither of which this branch touches.
cat > "${BIN}/vllm" <<'VSTUB'
#!/bin/bash
sleep 30
VSTUB

# `date -Is` is GNU. On a BSD date every log line would carry an error instead of a time,
# which makes the rendered log unreadable for no gain.
cat > "${BIN}/date" <<'DSTUB'
#!/bin/bash
if [ "${1:-}" = "-Is" ]; then echo "2026-09-08T00:00:00+00:00"; exit 0; fi
exec /bin/date "$@"
DSTUB

# The cache preflight walks $HOME with du and reads a mtime with stat. Both are stubbed:
# a real du -sBG over a home directory is minutes of I/O and would make the render depend
# on the machine it ran on.
cat > "${BIN}/du" <<'DUSTUB'
#!/bin/bash
echo "0G ${!#}"
DUSTUB
cat > "${BIN}/stat" <<'STATSTUB'
#!/bin/bash
echo 0
STATSTUB
chmod +x "${BIN}/python" "${BIN}/vllm" "${BIN}/du" "${BIN}/stat" "${BIN}/date"

export PATH="${BIN}:${PATH}"
export BCF_RENDER_ROOT="$ROOT"
export BCF_RENDER_ARGV="${ROOT}/judge-argv.txt"
export BCF_ENV_SH="${HERE}/judge_render/env_stub.sh"
export BCF_EXIT_GUARD_SH="${REPO}/bcf/exit_guard.sh"
export BCF_REPO="$REPO"
export SLURM_JOB_ID=999999
export TZ=UTC

# The defaults this harness renders. One judge on its pinned line, the Q1 file of record,
# a gate corpus that exists; every other knob is left at the script's own default so the
# fixture records the DEFAULTS and a change to one of them shows up here.
export BCF_JUDGES="${BCF_JUDGES:-qwen3-32b}"
export BCF_OUT_ROOT="${BCF_OUT_ROOT:-${ROOT}/results}"
export BCF_JUDGE_CACHE="${BCF_JUDGE_CACHE:-${ROOT}/hf-judges}"
if [ -z "${BCF_GATE_ITEMS:-}" ]; then
  export BCF_GATE_ITEMS="${ROOT}/gate_items.jsonl"
  : > "$BCF_GATE_ITEMS"
fi

for kv in "$@"; do export "${kv?}"; done

bash "$SCRIPT" > "${ROOT}/stdout.txt" 2>&1
RC=$?

{
  echo "# exit_status ${RC}"
  echo "# judge-argv"
  if [ -f "$BCF_RENDER_ARGV" ]; then cat "$BCF_RENDER_ARGV"; else echo "<no judging invocation>"; fi
  echo "# exit_code.txt"
  found=0
  while IFS= read -r f; do
    printf '%s %s\n' "${f}" "$(cat "$f")"
    found=1
  done < <(find "${BCF_OUT_ROOT}" -name exit_code.txt 2>/dev/null | sort)
  [ "$found" = "1" ] || echo "<no exit_code.txt>"
} | sed "s#${ROOT}#<ROOT>#g" > "$OUT"

# The log goes to its own file, not into the rendered fixture: it carries a hostname, a
# job id and a timestamp per line, and a fixture that moves every second pins nothing.
sed "s#${ROOT}#<ROOT>#g" "${ROOT}/stdout.txt" > "${OUT}.stdout"

exit 0
