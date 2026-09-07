#!/bin/bash
# Render what bcf/serve_and_run.sbatch would ask vLLM to do, without a cluster.
#
# WHY this exists. The local-checkpoint branch of that script has to be ADDITIVE: with
# BCF_LOCAL_CHECKPOINT unset the sweep's live runner must behave exactly as it did. The
# script has no dry-run mode, so this harness supplies one from the outside: stub
# executables on PATH for `vllm`, `python -c 'import vllm'` and `nvidia-smi`, a stub
# bcf/env.sh through the script's own BCF_ENV_SH hook, and a stop at
# bcf_guard_register_child, which the script calls on the line after it backgrounds the
# server. What comes out is the serve argv and the run_meta.json the script wrote, with
# the sandbox root normalised to <ROOT> so the fixture is machine independent.
#
# Usage: render_serve_command.sh <output-file> [VAR=value ...]
#   Any VAR=value pairs are exported into the run, which is how a caller asks for the
#   default case (none) or the local-checkpoint case (BCF_LOCAL_CHECKPOINT=...).

set -uo pipefail

OUT="${1:?usage: render_serve_command.sh <output-file> [VAR=value ...]}"
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
SCRIPT="${BCF_RENDER_SCRIPT:-${REPO}/bcf/serve_and_run.sbatch}"

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/bcf-render.XXXXXX")"
if [ "${BCF_RENDER_KEEP:-0}" != "1" ]; then trap 'rm -rf "$ROOT"' EXIT; else echo "[render] root ${ROOT}" >&2; fi
BIN="${ROOT}/bin"
mkdir -p "$BIN"

# `python -c 'import vllm; print(vllm.__version__)'` is the only python call whose
# answer this harness invents. Everything else (the run_meta.json writer) runs on the
# real interpreter, so the metadata in the fixture is the metadata the script produces.
cat > "${BIN}/python" <<'PYSTUB'
#!/bin/bash
if [ "${1:-}" = "-c" ]; then
  case "$2" in
    *vllm*) echo "0.28.0" ; exit 0 ;;
  esac
fi
exec "${BCF_RENDER_PYTHON:-python3}" "$@"
PYSTUB

cat > "${BIN}/vllm" <<'VSTUB'
#!/bin/bash
# The command under test. One argument per line, so a fixture diff points at the
# argument that moved rather than at a re-wrapped line.
: > "${BCF_RENDER_ARGV:?}"
for a in "$@"; do printf '%s\n' "$a" >> "${BCF_RENDER_ARGV}"; done
VSTUB

cat > "${BIN}/nvidia-smi" <<'NSTUB'
#!/bin/bash
exit 1
NSTUB
chmod +x "${BIN}/python" "${BIN}/vllm" "${BIN}/nvidia-smi"

export PATH="${BIN}:${PATH}"
export BCF_RENDER_ROOT="$ROOT"
export BCF_RENDER_ARGV="${ROOT}/argv.txt"
export BCF_ENV_SH="${HERE}/serve_render/env_stub.sh"
export SLURM_JOB_ID=999999
export BCF_PLAN_COMMIT=renderfixture
export TZ=UTC

# The cell this harness renders. Roster row 1 on the substrate and cue family the ladder
# is measured with; every other knob is left at the script's own default so the fixture
# records the DEFAULTS and a change to one of them shows up here.
export BCF_MODEL="${BCF_MODEL:-Qwen/Qwen3-8B}"
export BCF_SUBSTRATE="${BCF_SUBSTRATE:-arc_challenge}"
export BCF_CUE="${BCF_CUE:-stated-hint}"

for kv in "$@"; do export "${kv?}"; done

bash "$SCRIPT" > "${ROOT}/stdout.txt" 2>&1
RC=$?

MODEL_SLUG="$(echo "${BCF_MODEL##*/}" | tr '[:upper:]' '[:lower:]')"
META="${BCF_OUT_ROOT:-${ROOT}/results}/${MODEL_SLUG}/${BCF_SUBSTRATE}/${BCF_CUE}/run_meta.json"

{
  echo "# exit_status ${RC}"
  echo "# vllm-argv"
  if [ -f "$BCF_RENDER_ARGV" ]; then cat "$BCF_RENDER_ARGV"; else echo "<no vllm invocation>"; fi
  echo "# run_meta.json"
  if [ -f "$META" ]; then
    "${BCF_RENDER_PYTHON:-python3}" -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1])),indent=2,sort_keys=True))' "$META"
  else
    echo "<no run_meta.json>"
  fi
} | sed "s#${ROOT}#<ROOT>#g" > "$OUT"

exit 0
