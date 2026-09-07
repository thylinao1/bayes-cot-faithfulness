"""Proof that ``bcf_install_exit_guard`` (bcf/env.sh) writes a truthful exit_code.txt.

THE DEFECT. Jobs 827052 and 827096 were CANCELLED by Slurm with SIGTERM on
2026-09-07 (Slurm printed "CANCELLED ... DUE TO SIGNAL Terminated"), and five seconds
later each job's run.log printed "[done] exit_code=0", with no arms_summary.json ever
written. The finish()/``trap finish EXIT`` pattern that was in serve_and_run.sbatch
(and, before this fix, in probe.sbatch, logprob_recheck.sbatch and
tp2_serving_test.sbatch too) had no explicit TERM trap: it read ``$?`` whenever the
EXIT trap happened to fire, and on the cancel path that read 0. A results directory
holding a fraction of a run therefore carried a success code, indistinguishable
downstream from a complete one.

REVERT PROOF. ``test_old_guard_pattern_writes_zero_on_cancel`` below runs the EXACT
pre-fix shape -- ``trap finish EXIT`` with ``finish(){ code=$?; echo "$code" > ...;
exit "$code"; }`` and no TERM trap -- against the same sleep-30-then-SIGTERM scenario
the new guard is tested against below, and asserts it writes 0. That is what makes the
other tests in this file able to fail: without this case, a test that only ever
exercises the fixed helper cannot tell a working guard from a lucky one.

The remaining cases source the real bcf/env.sh (no Slurm, no GPU, no network --
SLURM_JOB_ID is set to a throwaway id so env.sh's own
``${SLURM_JOB_ID:?SLURM_JOB_ID must be set}`` guard does not abort the source; the
miniconda activate line env.sh runs is allowed to fail quietly on a machine with no
miniconda3, since it is not on ``set -e``) and exercise ``bcf_install_exit_guard``
itself through a tiny bash harness script:

  test_cancelled_writes_143            sleep 30 in the foreground, SIGTERM from
                                        Python mid-sleep -> 143 (128 + SIGTERM)
  test_zero_without_marker_writes_12   the script exits 0 but never creates the
                                        marker file -> 12, "finished without a
                                        completion marker"
  test_zero_with_marker_writes_0       the script creates the marker file, then
                                        exits 0 -> 0
  test_signal_forwarded_to_registered_child   a registered "server" child is still
                                        alive when SIGTERM reaches the parent, and is
                                        gone shortly after -- proof of the "never a
                                        wildcard kill, but never an orphan either"
                                        requirement.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENV_SH = REPO / "bcf" / "env.sh"

SIGTERM_CODE = 128 + 15  # 143
SIGINT_CODE = 128 + 2  # 130
ZERO_WITHOUT_MARKER_CODE = 12

# How long the harness's foreground "work" runs for. Long enough that the SIGTERM
# sent a few hundred ms after start always lands mid-run rather than after a
# fast-finishing test host somehow already reached exit; if the trap did not fire the
# test times out instead of silently passing.
FOREGROUND_SLEEP_SECONDS = 30
SEND_SIGNAL_AFTER = 0.4
WAIT_TIMEOUT = 10


def _job_id() -> str:
    return f"testguard-{uuid.uuid4().hex[:12]}"


def _read_exit_code(out_dir: Path) -> str:
    return (out_dir / "exit_code.txt").read_text().strip()


def _run_and_signal(
    script: Path, args: list[str], out_dir: Path, *, sig: int | None, job_id: str
) -> subprocess.CompletedProcess:
    """Launch ``bash script args...``, optionally signal it mid-run, and wait.

    The harness is started in its OWN process group (``start_new_session=True``) and,
    when ``sig`` is given, the signal is sent to that whole group with ``os.killpg``
    rather than to the bash pid alone. This mirrors how Slurm actually cancels a job
    (SIGTERM reaches the whole job step's process tree, which is why the real
    827052/827096 incident showed the run finishing within ~5 seconds of the cancel,
    not lingering for its full remaining runtime) and it is load-bearing for the test:
    bash defers running a trap until the CURRENT foreground command completes on its
    own (documented behaviour, confirmed against this Mac's bash 3.2.57 -- a bare
    ``sleep 30`` signalled ONLY at the bash pid does not trigger its TERM trap until
    all 30 seconds elapse). Signalling the group means ``sleep`` -- the harness's own
    foreground command -- receives SIGTERM directly and dies immediately, which is
    what lets bash regain control and run the trap right away, exactly as it does in
    production when Slurm signals the job's server and client processes at the same
    moment as the batch script itself.
    """
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = job_id
    proc = subprocess.Popen(
        ["bash", str(script), *args],
        cwd=out_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if sig is not None:
        time.sleep(SEND_SIGNAL_AFTER)
        assert proc.poll() is None, "harness exited before the signal was sent"
        os.killpg(os.getpgid(proc.pid), sig)
    out, _ = proc.communicate(timeout=WAIT_TIMEOUT)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, None)


# --------------------------------------------------------------------- revert proof
LEGACY_HARNESS = """\
#!/bin/bash
# The pre-fix shape: one EXIT trap, no TERM trap, reads $? verbatim.
set -uo pipefail
OUT_DIR="$1"
finish() { code=$?; echo "$code" > "$OUT_DIR/exit_code.txt"; exit "$code"; }
trap finish EXIT
sleep __SLEEP_SECONDS__
""".replace("__SLEEP_SECONDS__", str(FOREGROUND_SLEEP_SECONDS))


def test_old_guard_pattern_writes_zero_on_cancel(tmp_path: Path):
    out_dir = tmp_path / "legacy"
    out_dir.mkdir()
    script = tmp_path / "legacy.sh"
    script.write_text(LEGACY_HARNESS)
    result = _run_and_signal(
        script, [str(out_dir)], out_dir, sig=signal.SIGTERM, job_id=_job_id()
    )
    assert _read_exit_code(out_dir) == "0", (
        "the pre-fix finish()/trap-EXIT pattern was expected to write a false 0 on "
        f"cancel (this is the 827052/827096 shape); got stdout:\n{result.stdout}"
    )


# ------------------------------------------------------------- bcf_install_exit_guard
GUARD_HARNESS = """\
#!/bin/bash
set -uo pipefail
source "__ENV_SH__"
OUT_DIR="$1"
BEHAVIOUR="$2"
MARKER="${3:-arms_summary.json}"
mkdir -p "$OUT_DIR"
bcf_install_exit_guard "$OUT_DIR" "$MARKER"
case "$BEHAVIOUR" in
  hang)
    sleep __SLEEP_SECONDS__
    ;;
  zero_no_marker)
    exit 0
    ;;
  zero_with_marker)
    : > "$OUT_DIR/$MARKER"
    exit 0
    ;;
  *)
    echo "unknown behaviour: $BEHAVIOUR" >&2
    exit 99
    ;;
esac
""".replace("__ENV_SH__", str(ENV_SH)).replace(
    "__SLEEP_SECONDS__", str(FOREGROUND_SLEEP_SECONDS)
)


@pytest.fixture()
def guard_script(tmp_path: Path) -> Path:
    script = tmp_path / "guard_harness.sh"
    script.write_text(GUARD_HARNESS)
    return script


def test_cancelled_writes_143(tmp_path: Path, guard_script: Path):
    out_dir = tmp_path / "cancelled"
    out_dir.mkdir()
    result = _run_and_signal(
        guard_script,
        [str(out_dir), "hang"],
        out_dir,
        sig=signal.SIGTERM,
        job_id=_job_id(),
    )
    assert result.returncode == SIGTERM_CODE, (
        f"harness process exit status: want {SIGTERM_CODE}, got {result.returncode}; "
        f"stdout:\n{result.stdout}"
    )
    assert _read_exit_code(out_dir) == str(SIGTERM_CODE), (
        f"exit_code.txt: want {SIGTERM_CODE} (128+SIGTERM), got "
        f"{_read_exit_code(out_dir)!r}; a cancelled run must never read as a clean 0"
    )


def test_cancelled_with_sigint_writes_130(tmp_path: Path, guard_script: Path):
    out_dir = tmp_path / "interrupted"
    out_dir.mkdir()
    _run_and_signal(
        guard_script,
        [str(out_dir), "hang"],
        out_dir,
        sig=signal.SIGINT,
        job_id=_job_id(),
    )
    assert _read_exit_code(out_dir) == str(SIGINT_CODE)


def test_zero_without_marker_writes_12(tmp_path: Path, guard_script: Path):
    out_dir = tmp_path / "no-marker"
    out_dir.mkdir()
    result = _run_and_signal(
        guard_script,
        [str(out_dir), "zero_no_marker"],
        out_dir,
        sig=None,
        job_id=_job_id(),
    )
    assert not (out_dir / "arms_summary.json").exists()
    assert _read_exit_code(out_dir) == str(ZERO_WITHOUT_MARKER_CODE), (
        "a script that exits 0 without producing its own completion marker must not "
        f"read as a clean run; stdout:\n{result.stdout}"
    )


def test_zero_with_marker_writes_0(tmp_path: Path, guard_script: Path):
    out_dir = tmp_path / "with-marker"
    out_dir.mkdir()
    result = _run_and_signal(
        guard_script,
        [str(out_dir), "zero_with_marker"],
        out_dir,
        sig=None,
        job_id=_job_id(),
    )
    assert (out_dir / "arms_summary.json").exists()
    assert _read_exit_code(out_dir) == "0", (
        f"a run that produced its own completion marker and exited 0 must read 0; "
        f"stdout:\n{result.stdout}"
    )


# ------------------------------------------------------- signal forwarding to a child
CHILD_FORWARD_HARNESS = """\
#!/bin/bash
set -uo pipefail
source "__ENV_SH__"
OUT_DIR="$1"
mkdir -p "$OUT_DIR"
bcf_install_exit_guard "$OUT_DIR" "arms_summary.json"
sleep __SLEEP_SECONDS__ &
CHILD_PID=$!
bcf_guard_register_child "$CHILD_PID"
echo "$CHILD_PID" > "$OUT_DIR/child.pid"
wait "$CHILD_PID"
""".replace("__ENV_SH__", str(ENV_SH)).replace(
    "__SLEEP_SECONDS__", str(FOREGROUND_SLEEP_SECONDS)
)


def test_signal_forwarded_to_registered_child(tmp_path: Path):
    # Never a wildcard kill (NUS-COMPUTE.md 1.9): the guard must reach ONLY a child
    # registered with bcf_guard_register_child, and it must actually reach it -- a
    # server left running past the job that started it is an orphan on a shared node.
    out_dir = tmp_path / "forwarding"
    out_dir.mkdir()
    script = tmp_path / "child_forward.sh"
    script.write_text(CHILD_FORWARD_HARNESS)
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = _job_id()
    proc = subprocess.Popen(
        ["bash", str(script), str(out_dir)],
        cwd=out_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    child_pid_file = out_dir / "child.pid"
    deadline = time.time() + WAIT_TIMEOUT
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert child_pid_file.exists(), "harness never registered its child"
    child_pid = int(child_pid_file.read_text().strip())

    def child_alive() -> bool:
        try:
            os.kill(child_pid, 0)
            return True
        except OSError:
            return False

    assert child_alive(), "the registered child should still be running before SIGTERM"
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=WAIT_TIMEOUT)

    assert _read_exit_code(out_dir) == str(SIGTERM_CODE), out
    deadline = time.time() + 3
    while child_alive() and time.time() < deadline:
        time.sleep(0.05)
    assert not child_alive(), (
        "the registered child (the 'server') outlived the parent's SIGTERM handler; "
        "it would be orphaned on a shared node"
    )
