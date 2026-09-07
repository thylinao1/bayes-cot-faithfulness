#!/bin/bash
# Sourced by every bcf sbatch script. Sets up the conda env, node-local scratch, and
# the paths the runners expect. Never run standalone; it assumes $SLURM_JOB_ID exists.
#
# Built by ~/bcf/env-setup.sbatch (job 825198, EXIT_CODE=0, 2026-09-06):
#   vllm 0.28.0, nnsight 0.7.0, numpyro 0.21.0, torch 2.13.0+cu130.

# NOT `set -e`: this file is sourced, so -e would leak into the calling sbatch script,
# where a plain `[ -n "$OPTIONAL" ] && args+=(...)` that legitimately returns 1 kills the
# job. Each caller decides its own error mode.
set -uo pipefail

# A shell cancelled BEFORE the caller reaches bcf_install_exit_guard must still exit
# truthfully. Everything between this line and that call is slow on a real node (the
# conda activation just below measured 4.1 s on the login node, 2026-09-07), and a
# SIGTERM landing inside that window used to kill bash with the signal's DEFAULT
# disposition: wait status -15, no trap, no exit_code.txt. These two bootstrap traps
# hold the window until the caller installs the real guard, which replaces them. They
# deliberately inline the exit rather than call a function, because the functions
# below do not exist yet while this file is still being sourced.
trap 'BCF_GUARD_SIGNAL=15; exit 143' TERM
trap 'BCF_GUARD_SIGNAL=2; exit 130' INT

# Activate the campaign env, but only on a machine that actually has it. The
# unguarded pair this replaces still spent seconds inside a `conda activate` that
# could never succeed when $HOME has no miniconda (measured 2026-09-07 on the login
# node with a faked $HOME: 1.1 s for the doomed activate alone, longer on a cold CI
# runner), and every one of those seconds is time the calling script runs with no
# exit guard installed.
BCF_CONDA_PREFIX="${BCF_CONDA_PREFIX:-$HOME/miniconda3}"
if [ -f "${BCF_CONDA_PREFIX}/bin/activate" ]; then
  source "${BCF_CONDA_PREFIX}/bin/activate"
  conda activate "${BCF_CONDA_ENV:-bcf}"
else
  echo "[env] no conda at ${BCF_CONDA_PREFIX}; continuing with the ambient python" >&2
fi

# Home is persistent but has a shared budget; model weights live on node-local scratch
# and die with the job (NUS-COMPUTE.md 1.5, and the "home quota" risk row in the plan).
export SCRATCH="/tmp/${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
export HF_HOME="${SCRATCH}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export OUTLINES_CACHE_DIR="${SCRATCH}/outlines"
export VLLM_CACHE_ROOT="${SCRATCH}/vllm"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$OUTLINES_CACHE_DIR" "$VLLM_CACHE_ROOT"

# No CUDA JIT between the allocation and the run. vLLM 0.28.0 picks FlashInfer for
# top-p / top-k sampling and JIT-compiles flashinfer's sampling.cu on first use, AFTER the
# weights are loaded. That compile is node dependent: job 825246 ran it fine on xgph12,
# job 825480 died on xgph10 with
#   flashinfer/sampling.cuh(623): error: class "cub::_V_300302_SM_800::BlockAdjacentDifference<...>"
#   ninja: build stopped: subcommand failed
# after 5 minutes of allocation, model download and load. Every generation in this project
# runs at temperature 0 (frozen decoding constant) except the pre-registered k=32 T=0.7
# arm, so the sampler kernel is not on the measurement path, and trading it for the
# PyTorch sampler removes a node-dependent build from between a scarce card and a run.
# Unset BCF_KEEP_FLASHINFER_SAMPLER to opt back in on a node where it works.
export VLLM_USE_FLASHINFER_SAMPLER="${BCF_KEEP_FLASHINFER_SAMPLER:-0}"

export BCF_ROOT="${BCF_ROOT:-$HOME/bcf}"
export BCF_REPO="${BCF_REPO:-$BCF_ROOT/repo}"
export BCF_RESULTS="${BCF_RESULTS:-$BCF_ROOT/results}"
export PYTHONPATH="${BCF_REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- helpers -------------------------------------------------------------------

# Number of devices vLLM will actually see. Asked of torch, not of the SBATCH request
# and not of a grep over `nvidia-smi -L`, because on this cluster all three disagree:
# Slurm's a100-40 and a100-80 GRES hand out MIG slices, so `nvidia-smi -L` prints ONE
# "GPU 0:" line with two indented "MIG ... Device N:" lines under it, while CUDA sees
# two devices. Job 825248 refused on exactly that mismatch. torch.cuda.device_count()
# is what the serving process will see, so it is the number to assert on.
bcf_visible_devices() {
  python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0
}

# True when the allocation is MIG slices rather than whole cards. This matters more
# than the count: MIG instances have no NVLink or peer-to-peer path between them, so
# NCCL cannot build a tensor-parallel group across them however many are visible.
bcf_is_mig() {
  nvidia-smi -L 2>/dev/null | grep -q '^ *MIG '
}

# Refuse BEFORE any weight download if the card count is not what the model needs.
# A 70B started on one card downloads 140 GB to scratch and then OOMs; this turns
# that into a five-second exit.
bcf_assert_devices() {
  local want="$1" got
  got="$(bcf_visible_devices)"
  echo "[devices] torch.cuda.device_count()=${got}; requested tensor-parallel size ${want}"
  nvidia-smi -L || true
  if [ "$want" -gt 1 ] && bcf_is_mig; then
    echo "[devices] REFUSING: this allocation is MIG slices, not whole cards." >&2
    nvidia-smi -L | sed 's/^/[devices]   /' >&2
    echo "[devices]   CUDA exposes at most ONE MIG instance to a process, so" >&2
    echo "[devices]   torch.cuda.device_count() reports ${got} even though Slurm granted" >&2
    echo "[devices]   ${SLURM_GPUS_PER_NODE:-?} and CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}." >&2
    echo "[devices]   Tensor parallelism across MIG slices is impossible, not merely slow." >&2
    echo "[devices]   Verified 2026-09-06: a100-40 is MIG 3g.40gb of an A100 80GB (job" >&2
    echo "[devices]   825248) and h100-47 is MIG 3g.47gb of an H100 NVL (job 825283)." >&2
    echo "[devices]   h100-96 is the only whole-card multi-GPU type within this account's" >&2
    echo "[devices]   caps (2 per node, per-user cap 2)." >&2
    return 1
  fi
  if [ "$got" -ne "$want" ]; then
    echo "[devices] REFUSING: visible device count ${got} != requested tensor-parallel size ${want}" >&2
    return 1
  fi
  return 0
}

# Resolve and print the HF revision sha for the CONTRACT hf_revision field, and pin
# the served weights to it. A floating 'main' makes a run unreproducible.
bcf_revision() {
  python - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
print(HfApi().model_info(sys.argv[1]).sha)
PY
}

# --- serving ports and readiness ------------------------------------------------

# A FREE loopback port for this job, unless the operator pins one in BCF_PORT.
#
# Every serving sbatch used to default to a FIXED port (8000, 8100, 8200, 8300). MIG
# slices of one A100 node share the host loopback, so two of our jobs on one node meant
# two servers fighting for one port: on xgph12 on 2026-09-07 the second job's readiness
# probe was answered by the FIRST job's server and the run went on against it. A free
# port is picked here, and the readiness check below proves the answer comes from OUR
# server, because the pick itself races (another process can take the port between the
# probe socket closing and vllm binding). Belt and braces, on purpose.
bcf_pick_port() {
  if [ -n "${1:-}" ]; then echo "$1"; return 0; fi
  local py
  py="$(command -v python3 || command -v python)"
  "$py" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

# Wait for a server this job started, and prove it is ours before returning ready.
#
#   bcf_wait_server_ready <base-url> <served-name> <server-pid> <deadline-epoch> \
#                         <server-log> <report-json> | ts | tee -a "$LOG"
#   rc=${PIPESTATUS[0]}
#
# Prints to stdout only (the caller owns the timestamping and the log file) and
# returns the caller's own exit code:
#   0   ready, and the responder is ours
#   5   our server died during startup
#   6   the server did not become ready before the deadline
#   13  REFUSED: something else answers on this port (see bcf/serve_ready.py)
# The per-poll ownership verdict is written to <report-json> for run_meta.json.
bcf_wait_server_ready() {
  local base_url="$1" served="$2" pid="$3" deadline="$4" slog="$5" report="$6"
  local py rc
  py="$(command -v python3 || command -v python)"
  while true; do
    "$py" "${BCF_REPO}/bcf/serve_ready.py" --base-url "$base_url" \
      --served-name "$served" --server-pid "$pid" --report "$report" 2>&1
    rc=$?
    case "$rc" in
      0) return 0 ;;
      3) : ;;  # nothing listening yet, keep polling
      11)
        echo "[serve] REFUSING: ${base_url} answers, but not from this job's server (see the line above and ${report}). Not running against a server this job does not own."
        return 13
        ;;
      *)
        # The checker itself broke (a traceback, a bad argument, no python). That is not
        # a verdict, and "we could not check" must never read as "it is ours".
        echo "[serve] REFUSING: bcf/serve_ready.py exited ${rc}, which is not a verdict. Refusing rather than assuming the server on ${base_url} is ours."
        return 13
        ;;
    esac
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve] server died during startup; last 40 lines of ${slog}:"
      tail -40 "$slog" 2>/dev/null
      return 5
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "[serve] server did not become ready before the deadline; last 40 lines of ${slog}:"
      tail -40 "$slog" 2>/dev/null
      return 6
    fi
    sleep 10
  done
}

# --- exit-code guard for serving sbatch scripts ----------------------------------
#
# THE DEFECT THIS CLOSES. Jobs 827052 and 827096 were CANCELLED by Slurm with SIGTERM
# on 2026-09-07 (Slurm printed "CANCELLED ... DUE TO SIGNAL Terminated"), and five
# seconds later each job's run.log printed "[done] exit_code=0" with no
# arms_summary.json ever written. Their `trap finish EXIT` read $? at trap time and
# trusted it verbatim; on the cancel path $? read 0 by the time the trap ran, so a
# killed cell read exactly like a clean one. bcf/judge_serve.sbatch already trapped
# TERM/INT/HUP/USR1 explicitly (bcf/exit_guard.sh) and wrote 143 on its own first live
# cancel; this helper brings serve_and_run.sbatch, probe.sbatch, logprob_recheck.sbatch
# and tp2_serving_test.sbatch to that same bar, from one place instead of four copies.
#
#   bcf_install_exit_guard OUT_DIR MARKER
#     OUT_DIR  the run's output directory. exit_code.txt is written there.
#     MARKER   filename of the run's own completion marker, resolved under OUT_DIR
#               unless it is already an absolute path (arms_summary.json for
#               serve_and_run.sbatch, probe_results.json for probe.sbatch,
#               logprob_check_thinking_off.json for logprob_recheck.sbatch,
#               tp2_report.json for tp2_serving_test.sbatch -- read each script's own
#               success path rather than assuming; contract_layout.py is what actually
#               produces the canonical arms_summary.json, and it runs after the arms
#               command, so the marker is only ever present once that step has run).
#
# Installs three traps in the CALLING shell (not a subshell, so `exit` from inside a
# handler ends the caller's own process):
#   TERM -> writes 128+15=143 and forwards SIGTERM to every pid registered with
#           bcf_guard_register_child, so the server dies with the job instead of being
#           orphaned. Never a wildcard kill (NUS-COMPUTE.md 1.9 etiquette: other
#           campaigns share this account).
#   INT  -> writes 128+2=130, same forwarding.
#   EXIT -> the normal path, and the one that closes the 827052/827096 defect: a 0 is
#           written ONLY when the command that set $? returned 0 AND the marker file
#           exists. A 0 with no marker becomes 12, "finished without a completion
#           marker" -- exactly the 827052/827096 shape, a shell that reached its own
#           exit with nothing to show for it. Any other status passes through
#           unchanged: 1 still means a failed threshold, a result rather than a job
#           failure, and 2-11/13 keep meaning whatever the calling script's own header
#           table says they mean.
#
#   bcf_guard_register_child PID   forward TERM/INT/a cleanup kill to this pid. Call it
#                                   right after `cmd & PID=$!` for every server (or
#                                   client) process the script itself started and owns.
#
# Call bcf_install_exit_guard ONCE, as early as OUT_DIR is known, and do not also
# `trap ... EXIT` afterwards in the caller: the two traps would race and only the last
# one installed survives, silently dropping this guard. Sourcing this file already
# installed bootstrap TERM/INT traps (top of the file) so the window between the
# source and this call is not a raw-signal death; installing the real guard replaces
# them, and OUT_DIR is what the bootstrap pair could not know.
#
# Exercised without Slurm or a GPU by tests/test_exit_guard.py (a tiny bash script
# sources this file, installs the guard, and is sent real signals from Python).
BCF_GUARD_OUT_DIR=""
BCF_GUARD_MARKER=""
BCF_GUARD_SIGNAL=0
BCF_GUARD_CHILDREN=""

bcf_guard_register_child() {
  BCF_GUARD_CHILDREN="${BCF_GUARD_CHILDREN} $1"
}

# True when at least one child was registered, so the exit path can tell "shut the
# server down gracefully" from "there is nothing to be graceful to".
_bcf_guard_has_children() {
  local pid
  for pid in $BCF_GUARD_CHILDREN; do
    [ -n "$pid" ] && return 0
  done
  return 1
}

# Best-effort signal to every registered child. Never fails the guard: a child that
# already exited, or a pid that was never really ours, just gets ESRCH and is ignored.
_bcf_guard_kill_children() {
  local sig="${1:-TERM}" pid
  for pid in $BCF_GUARD_CHILDREN; do
    [ -n "$pid" ] && kill -"$sig" "$pid" 2>/dev/null
  done
  return 0
}

_bcf_guard_on_signal() {
  local sig="$1"
  BCF_GUARD_SIGNAL="$sig"
  _bcf_guard_kill_children "$sig"
  exit $(( 128 + sig ))
}

_bcf_guard_finish() {
  local code=$? marker
  if [ "${BCF_GUARD_SIGNAL:-0}" -ne 0 ]; then
    # The signal handler already forwarded the real signal above; this is a fast
    # best-effort sweep, not a second grace period, because Slurm is already tearing
    # the job step down on its own clock.
    code=$(( 128 + BCF_GUARD_SIGNAL ))
    _bcf_guard_kill_children KILL
  else
    if [ "$code" -eq 0 ]; then
      marker="$BCF_GUARD_MARKER"
      case "$marker" in
        /*) : ;;
        *) marker="${BCF_GUARD_OUT_DIR}/${marker}" ;;
      esac
      if [ ! -e "$marker" ]; then
        code=12
      fi
    fi
    # A normal (non-signal) exit: give a server we own a real chance to shut down
    # before the hard kill, same as the finish() this replaces. Only when there IS a
    # registered child: the unconditional 5 s this replaces sat on the exit path of
    # every run that never registered one, which on a CI runner was most of the
    # distance to a test timeout for no shutdown that anything was waiting on.
    if _bcf_guard_has_children; then
      _bcf_guard_kill_children TERM
      sleep 5
      _bcf_guard_kill_children KILL
    fi
  fi
  mkdir -p "$BCF_GUARD_OUT_DIR" 2>/dev/null
  printf '%s\n' "$code" > "${BCF_GUARD_OUT_DIR}/exit_code.txt"
  if [ -f "${BCF_GUARD_OUT_DIR}/run.log" ]; then
    printf '[%s] [done] exit_code=%s -> %s/exit_code.txt\n' \
      "$(date -Is 2>/dev/null || echo unknown)" "$code" "$BCF_GUARD_OUT_DIR" \
      >> "${BCF_GUARD_OUT_DIR}/run.log" 2>/dev/null
  fi
  # Optional extension point: a caller that needs one more thing done at exit (e.g.
  # bcf/probe.sbatch's sacct capture) defines a function named bcf_guard_after_finish
  # BEFORE this trap fires and it runs here, AFTER exit_code.txt is written. This
  # exists so a script never needs its own `trap ... EXIT`, which would silently
  # replace this one (only the last EXIT trap installed survives) and drop the guard.
  if declare -f bcf_guard_after_finish >/dev/null 2>&1; then
    bcf_guard_after_finish
  fi
  return 0
}

bcf_install_exit_guard() {
  BCF_GUARD_OUT_DIR="${1:?bcf_install_exit_guard needs OUT_DIR}"
  BCF_GUARD_MARKER="${2:?bcf_install_exit_guard needs MARKER}"
  BCF_GUARD_SIGNAL=0
  BCF_GUARD_CHILDREN=""
  mkdir -p "$BCF_GUARD_OUT_DIR"
  trap '_bcf_guard_on_signal 15' TERM
  trap '_bcf_guard_on_signal 2' INT
  trap _bcf_guard_finish EXIT
}
