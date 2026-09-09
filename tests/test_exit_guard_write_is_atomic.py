"""Finalising an exit code must never leave a run recorded two different ways.

bcf_exit_guard_write records one final code into the run's exit_code.txt and into every
stage file still marked started. The run file is the authoritative record, so the
invariant is:

    if the run file carries a final code, no stage file is still marked started

Until 2026-09-09 the writer broke that under a second signal. It wrote the run file FIRST
and the stage files after, and the loop forked a `cat` per stage file, so the window was
wide. bcf/test_exit_guard.sh sends TERM twice on purpose, mirroring how Slurm signals a
whole job step, and a signal landing in that window left exit_code.txt=143 next to
stage_exit_code.txt=255: the same cancelled run reading as "cancelled" in one file and
"started, never finished" in the other. It reproduced in about one run in five, which is
why it survived: the suite passed most of the time.

This test forces the race instead of waiting for it, by signalling while the writer is
mid-loop.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "bcf" / "exit_guard.sh"

# enough stage files that the loop is still running when the signal arrives
N_STAGES = 4000
DELAYS = (0.01, 0.03, 0.05, 0.08, 0.12)


def _started_code() -> str:
    m = re.search(r'^BCF_EXIT_STARTED=["\']?(\d+)', GUARD.read_text(), re.MULTILINE)
    assert m, "could not read BCF_EXIT_STARTED from bcf/exit_guard.sh"
    return m.group(1)


def _run_and_signal(tmp: Path, delay: float) -> tuple[str, list[str]]:
    started = _started_code()
    run_file = tmp / "exit_code.txt"
    run_file.write_text(started + "\n")
    stages = []
    for i in range(N_STAGES):
        f = tmp / f"stage_{i}.txt"
        f.write_text(started + "\n")
        stages.append(f)

    listfile = tmp / "stages.txt"
    listfile.write_text(" ".join(str(s) for s in stages))
    script = tmp / "drive.sh"
    script.write_text(
        "set -u\n"
        f'. "{GUARD}"\n'
        f'BCF_EXIT_FILE="{run_file}"\n'
        f'BCF_EXIT_STAGE_FILES="$(cat "{listfile}")"\n'
        "bcf_exit_guard_write 143\n"
    )
    proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
    time.sleep(delay)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=60)
    return run_file.read_text().strip(), [s.read_text().strip() for s in stages]


def test_a_signal_mid_finalisation_never_splits_the_record(tmp_path):
    started = _started_code()
    violations = []
    for i, delay in enumerate(DELAYS):
        d = tmp_path / f"attempt_{i}"
        d.mkdir()
        run_code, stage_codes = _run_and_signal(d, delay)
        stale = sum(1 for c in stage_codes if c == started)
        if run_code != started and stale:
            violations.append(
                f"delay={delay}s: run file says {run_code} (final) while {stale} of "
                f"{len(stage_codes)} stage files still say {started} (started)"
            )
    assert not violations, (
        "the run was recorded two different ways at once:\n  " + "\n  ".join(violations)
    )


def test_finalisation_completes_despite_the_signal(tmp_path):
    """With the trap in place the writer finishes, so everything reads 143."""
    d = tmp_path / "complete"
    d.mkdir()
    run_code, stage_codes = _run_and_signal(d, 0.05)
    assert run_code == "143", f"run file did not finalise: {run_code}"
    assert set(stage_codes) == {"143"}, (
        f"stage files did not all finalise: {sorted(set(stage_codes))}"
    )


def test_the_writer_returns_zero_when_no_run_file_is_set(tmp_path):
    """It is called from an EXIT trap; a non-zero return would change what is recorded."""
    script = tmp_path / "d.sh"
    script.write_text(
        "set -u\n"
        f'. "{GUARD}"\n'
        'BCF_EXIT_FILE=""\n'
        'BCF_EXIT_STAGE_FILES=""\n'
        "bcf_exit_guard_write 7\n"
        'echo "rc=$?"\n'
    )
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                       timeout=30, check=False)
    assert "rc=0" in r.stdout, f"writer returned non-zero: {r.stdout} {r.stderr}"


# ------------------------------------------------------- the second-signal race itself

DOUBLE_SIGNAL_ATTEMPTS = 5

# The EXIT trap deliberately pauses before it records. That is not the bug being simulated,
# it only WIDENS a window that really exists: bcf_exit_guard_signal sets the signal number
# and calls exit, and the EXIT trap is what actually writes the code. Everything the shell
# does between those two points ran with the TERM trap still installed. Sending the second
# signal into that pause is the same event as a Slurm cancel reaching a second process in
# the job step a moment after the first; without the pause the two signals coalesce, since
# standard signals are not queued, and the test silently stops testing anything.
JOB = """\
set -u
. "{guard}"
bcf_exit_guard_init "{exit_file}"
finish() {{
  raw=$?
  sleep 0.6
  code="$(bcf_exit_guard_code "$raw")"
  bcf_exit_guard_write "$code"
}}
trap finish EXIT
echo $$ > "{ready}"
sleep 30
"""


def test_a_second_signal_cannot_stop_a_cancel_being_recorded(tmp_path):
    """Slurm signals a whole job step, so a second signal is normal, not exotic.

    Until 2026-09-09 the signal traps stayed installed while the exit path ran, so a second
    TERM re-entered the handler and killed the shell before the code was written. The
    cancelled run then kept its 255 "started, never finished" marker instead of 143, which
    is exactly the confusion this guard exists to remove. In bcf/test_exit_guard.sh it
    showed up about one run in five.
    """
    started = _started_code()
    wrong = []
    for attempt in range(DOUBLE_SIGNAL_ATTEMPTS):
        d = tmp_path / f"dbl_{attempt}"
        d.mkdir()
        exit_file = d / "exit_code.txt"
        ready = d / "ready"
        script = d / "job.sh"
        script.write_text(JOB.format(guard=GUARD, exit_file=exit_file, ready=ready))

        proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
        deadline = time.time() + 20
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "the job never armed its guard"

        pgid = os.getpgid(proc.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(0.25)          # let the handler start and reach the pause
            os.killpg(pgid, signal.SIGTERM)   # the second signal, into the real window
        except ProcessLookupError:
            pass
        proc.wait(timeout=60)

        got = exit_file.read_text().strip()
        if got != "143":
            wrong.append(f"attempt {attempt}: recorded {got}")

    assert not wrong, (
        f"a cancelled run was not recorded as 143 in {len(wrong)} of "
        f"{DOUBLE_SIGNAL_ATTEMPTS} attempts (a bare {started} means the second signal "
        "killed the shell before the EXIT trap could record anything):\n  "
        + "\n  ".join(wrong)
    )
