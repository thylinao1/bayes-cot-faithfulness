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
``${SLURM_JOB_ID:?SLURM_JOB_ID must be set}`` guard does not abort the source; env.sh
skips its conda activation entirely on a machine with no miniconda under ``$HOME``)
and exercise ``bcf_install_exit_guard`` itself through a tiny bash harness script:

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
  test_sourcing_env_alone_survives_sigterm_with_143   a cancel that arrives after
                                        env.sh is sourced but BEFORE the caller
                                        installs the guard exits 143 rather than
                                        dying of the raw signal.
  test_the_installed_guard_owns_the_signal_traps   the armed TERM/INT handlers are
                                        the guard's own, not env.sh's bootstrap pair,
                                        which is dumber and would otherwise hide the
                                        deletion of the real ones.

The last two groups exercise the other two env.sh helpers that decide whether a run
gets to call itself finished, in the same no-Slurm no-GPU no-network way:

  bcf_gate_stage_status   a gate variant that exits 0 or 1 without writing
                          gate_report.json is recorded as 12, not as a verdict.
  bcf_prewarm_harmony     the gpt-oss vocab download, pulled forward out of the first
                          chat request, with a fake python on PATH standing in for it.
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
SEND_SIGNAL_AFTER = 0.2
WAIT_TIMEOUT = 30

# The harness touches this file the instant the guard is installed, and the signal is
# sent only once it appears. A fixed delay instead of this handshake is a race against
# however long the harness's own setup takes, and that setup is NOT fast everywhere:
# bcf/env.sh activates a conda env, which costs 4.1 s on the cluster login node and
# seconds on a CI runner, against 2 ms on a Mac with no miniconda at all. That is the
# whole reason these tests passed on macOS and failed on Linux -- the Linux signal
# landed while bash was still inside the activation, before any trap existed, so bash
# died of the raw SIGTERM (wait status -15) and no exit_code.txt was ever written.
GUARD_READY_FILE = "guard_ready"
READY_TIMEOUT = 60


def _job_id() -> str:
    return f"testguard-{uuid.uuid4().hex[:12]}"


def _read_exit_code(out_dir: Path) -> str:
    return (out_dir / "exit_code.txt").read_text().strip()


def _wait_for_ready(out_dir: Path, proc: subprocess.Popen) -> None:
    """Block until the harness says its guard is armed."""
    ready = out_dir / GUARD_READY_FILE
    deadline = time.time() + READY_TIMEOUT
    while not ready.exists() and time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"harness exited (status {proc.returncode}) before arming its guard"
            )
        time.sleep(0.02)
    assert ready.exists(), f"harness never armed its guard within {READY_TIMEOUT}s"


def _run_and_signal(
    script: Path, args: list[str], out_dir: Path, *, sig: int | None, job_id: str
) -> subprocess.CompletedProcess:
    """Launch ``bash script args...``, optionally signal it mid-run, and wait.

    The signal is sent only after the harness reports, by touching
    ``GUARD_READY_FILE``, that its guard is installed. What is under test is the
    guard, not a race between a fixed delay and however long sourcing bcf/env.sh
    takes on this machine.

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
        _wait_for_ready(out_dir, proc)
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
: > "$OUT_DIR/__READY__"
sleep __SLEEP_SECONDS__
""".replace("__SLEEP_SECONDS__", str(FOREGROUND_SLEEP_SECONDS)).replace(
    "__READY__", GUARD_READY_FILE
)


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
: > "$OUT_DIR/__READY__"
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
).replace("__READY__", GUARD_READY_FILE)


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


# ------------------------------- the window BEFORE the caller installs the real guard
PRE_INSTALL_HARNESS = """\
#!/bin/bash
# Sources env.sh and never installs the guard: the shape of a job cancelled during
# env.sh's own conda activation, before the sbatch script reaches its guard call.
set -uo pipefail
OUT_DIR="$1"
mkdir -p "$OUT_DIR"
source "__ENV_SH__"
: > "$OUT_DIR/__READY__"
sleep __SLEEP_SECONDS__
""".replace("__ENV_SH__", str(ENV_SH)).replace(
    "__SLEEP_SECONDS__", str(FOREGROUND_SLEEP_SECONDS)
).replace("__READY__", GUARD_READY_FILE)


def test_sourcing_env_alone_survives_sigterm_with_143(tmp_path: Path):
    """A cancel before bcf_install_exit_guard must not kill bash with the raw signal.

    Every serving sbatch spends real time between ``source bcf/env.sh`` and its
    ``bcf_install_exit_guard`` call: the conda activation inside env.sh measured 4.1 s
    on the cluster login node on 2026-09-07. A SIGTERM landing in that window used to
    kill bash under the signal's default disposition, so the job's wait status was
    "killed by 15" and no handler of ours ever ran. env.sh now arms a bootstrap pair
    at the top of the file for exactly this window.

    exit_code.txt is deliberately NOT asserted here and is NOT written: the bootstrap
    pair runs before any caller has said where OUT_DIR is. What it buys is a truthful
    exit status, not a file.
    """
    out_dir = tmp_path / "pre-install"
    out_dir.mkdir()
    script = tmp_path / "pre_install.sh"
    script.write_text(PRE_INSTALL_HARNESS)
    result = _run_and_signal(
        script, [str(out_dir)], out_dir, sig=signal.SIGTERM, job_id=_job_id()
    )
    assert result.returncode == SIGTERM_CODE, (
        "a shell that has sourced env.sh but not yet installed the guard must exit "
        f"{SIGTERM_CODE}, not die of the raw signal; got {result.returncode}, "
        f"stdout:\n{result.stdout}"
    )
    assert not (out_dir / "exit_code.txt").exists()


# --------------------------------------------- the guard owns the traps, not the stub
TRAP_INSPECT_HARNESS = """\
#!/bin/bash
set -uo pipefail
source "__ENV_SH__"
OUT_DIR="$1"
mkdir -p "$OUT_DIR"
bcf_install_exit_guard "$OUT_DIR" "arms_summary.json"
{ trap -p TERM; trap -p INT; trap -p EXIT; } > "$OUT_DIR/traps.txt"
: > "$OUT_DIR/arms_summary.json"
exit 0
""".replace("__ENV_SH__", str(ENV_SH))


def test_the_installed_guard_owns_the_signal_traps(tmp_path: Path):
    """The guard's OWN handlers must be the armed ones, not env.sh's bootstrap pair.

    Sourcing env.sh now arms a bootstrap ``trap 'BCF_GUARD_SIGNAL=15; exit 143' TERM``
    so that the seconds between the source and bcf_install_exit_guard (4.1 s of conda
    activation on the cluster login node, measured 2026-09-07) are not a raw-signal
    death. That bootstrap is deliberately dumber than the real handler: it cannot
    forward the signal to a child registered with bcf_guard_register_child, and it
    knows no OUT_DIR.

    This case exists because the bootstrap otherwise HIDES a regression. Deleting
    ``trap '_bcf_guard_on_signal 15' TERM`` from bcf_install_exit_guard leaves every
    behavioural test above still green (checked, 2026-09-07): the bootstrap catches
    the signal, the EXIT trap still writes 143, and _bcf_guard_finish's KILL sweep
    still reaches the child. Only reading the armed traps back tells the two apart.
    """
    out_dir = tmp_path / "traps"
    out_dir.mkdir()
    script = tmp_path / "trap_inspect.sh"
    script.write_text(TRAP_INSPECT_HARNESS)
    result = _run_and_signal(
        script, [str(out_dir)], out_dir, sig=None, job_id=_job_id()
    )
    assert _read_exit_code(out_dir) == "0", result.stdout
    traps = (out_dir / "traps.txt").read_text()
    assert "_bcf_guard_on_signal 15" in traps, (
        "TERM is not handled by the guard's own handler; env.sh's bootstrap trap or "
        f"something else is armed instead:\n{traps}"
    )
    assert "_bcf_guard_on_signal 2" in traps, (
        f"INT is not handled by the guard's own handler:\n{traps}"
    )
    assert "_bcf_guard_finish" in traps, f"EXIT trap is not the guard's:\n{traps}"


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


# ------------------------------------------- a gate variant status that is recorded


def _source_env(snippet: str, *, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run one bash snippet with bcf/env.sh sourced, the way an sbatch script does."""
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = _job_id()
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", "-c", f'source "{ENV_SH}"\n{snippet}'],
        env=env,
        capture_output=True,
        text=True,
        timeout=WAIT_TIMEOUT,
        check=False,
    )


CRASH_WITHOUT_REPORT_CODE = 12


def test_gate_stage_status_keeps_a_verdict_that_wrote_its_report(tmp_path: Path):
    report = tmp_path / "gate_report.json"
    report.write_text("{}")
    for status in ("0", "1"):
        res = _source_env(f'bcf_gate_stage_status {status} "{report}"')
        assert res.stdout.strip() == status, res.stderr


def test_gate_stage_status_turns_a_missing_report_into_12(tmp_path: Path):
    """The 828627 shape: the gate crashed, wrote no report, and exited a verdict code."""
    missing = tmp_path / "gate_report.json"
    for status in ("0", "1"):
        res = _source_env(f'bcf_gate_stage_status {status} "{missing}"')
        assert res.stdout.strip() == str(CRASH_WITHOUT_REPORT_CODE), (
            f"status {status} with no report at {missing} must not stay a verdict; "
            f"stdout={res.stdout!r} stderr={res.stderr!r}"
        )


def test_gate_stage_status_passes_a_job_failure_through(tmp_path: Path):
    """Above 1 the job already broke, and that code carries its own meaning."""
    report = tmp_path / "gate_report.json"
    res = _source_env(f'bcf_gate_stage_status 3 "{report}"')
    assert res.stdout.strip() == "3", res.stderr
    report.write_text("{}")
    res = _source_env(f'bcf_gate_stage_status 5 "{report}"')
    assert res.stdout.strip() == "5", res.stderr


# ------------------------------------------------------ the gpt-oss harmony prewarm
#
# A fake `python3` on PATH stands in for the openai_harmony import: it counts its own
# invocations and fails the first SHIM_FAIL_TIMES of them. That is the whole shape of
# the real failure (a vocab download that fails and then works), with no network.
PYTHON_SHIM = """\
#!/bin/bash
n=$(cat "$SHIM_COUNT_FILE" 2>/dev/null || echo 0)
n=$(( n + 1 ))
echo "$n" > "$SHIM_COUNT_FILE"
if [ "$n" -le "${SHIM_FAIL_TIMES:-0}" ]; then
  echo "openai_harmony.HarmonyError: error downloading or loading vocab file" >&2
  exit 1
fi
exit 0
"""


def _prewarm(tmp_path: Path, *, fail_times: int) -> tuple[subprocess.CompletedProcess, Path]:
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "python3"
    shim.write_text(PYTHON_SHIM)
    shim.chmod(0o755)
    counts = tmp_path / "attempts.txt"
    cache = tmp_path / "tiktoken-rs-cache"
    # PATH is set AFTER the source so nothing env.sh does can shadow the shim.
    snippet = (
        f'export PATH="{shim_dir}:$PATH"\n'
        "bcf_prewarm_harmony\n"
        "rc=$?\n"
        'echo "cache=${TIKTOKEN_RS_CACHE_DIR}"\n'
        'exit "$rc"\n'
    )
    res = _source_env(snippet, env_extra={
        "SHIM_COUNT_FILE": str(counts),
        "SHIM_FAIL_TIMES": str(fail_times),
        "BCF_TIKTOKEN_CACHE": str(cache),
        "BCF_PREWARM_SLEEP": "0",
    })
    return res, counts


def test_prewarm_harmony_retries_and_succeeds(tmp_path: Path):
    """Two failed vocab loads then a good one is a success, not a dead job."""
    res, counts = _prewarm(tmp_path, fail_times=2)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert counts.read_text().strip() == "3", res.stdout
    for attempt in (1, 2, 3):
        assert f"prewarm attempt {attempt} of 3" in res.stdout, res.stdout
    assert f"cache={tmp_path / 'tiktoken-rs-cache'}" in res.stdout
    assert (tmp_path / "tiktoken-rs-cache").is_dir()


def test_prewarm_harmony_that_never_loads_returns_non_zero(tmp_path: Path):
    """The caller must be able to exit BEFORE the server starts, loudly."""
    res, counts = _prewarm(tmp_path, fail_times=99)
    assert res.returncode != 0, res.stdout
    assert counts.read_text().strip() == "3", res.stdout
    assert "REFUSING" in res.stderr, res.stderr
