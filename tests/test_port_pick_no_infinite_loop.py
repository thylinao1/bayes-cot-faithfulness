"""The port picker must never loop forever, and an undefined picker must be fatal.

This pins a defect that cost real disk. bcf/judge_serve.sbatch picked a port with

    PORT="$(bcf_pick_port)"
    while [[ " ${PORTS[*]:-} " == *" ${PORT} "* ]]; do PORT="$(bcf_pick_port)"; done

If bcf_pick_port is not defined, the substitution yields "" and the pattern becomes
*" "* , which matches any string containing a space. " ${PORTS[*]:-} " always contains
one, so the loop condition was always true. On 2026-09-07 two orphaned runs of the exit
guard's integration harness sat in that loop for two days and wrote 35 GB of
"bcf_pick_port: command not found" into a single temp file, filling the machine.

The test drives the real block out of the real script with a stub environment, so it
fails if the guard is removed rather than only if someone edits a comment.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "bcf" / "judge_serve.sbatch"

# The block under test, lifted verbatim from the script so the test exercises the shipped
# text. If the shape of the block changes, the extraction fails loudly and this test is
# updated deliberately rather than silently passing against something else.
START = 'if ! command -v bcf_pick_port >/dev/null 2>&1; then'
END = '    done'


def _port_block() -> str:
    text = SBATCH.read_text()
    assert START in text, "the port-pick guard is gone from bcf/judge_serve.sbatch"
    body = text[text.index(START):]
    body = body[: body.index(END) + len(END)]
    # the script logs through `ts` and $LOG; neither exists in this harness
    return body.replace('| ts | tee -a "$LOG" >&2', ">&2")


def _run(defined: bool, picks: str = "", timeout: int = 10) -> subprocess.CompletedProcess:
    body = picks or "echo 5000"
    stub = f"bcf_pick_port() {{ {body}; }}\n" if defined else ""
    script = (
        "set -u\n"
        + stub
        + "PORTS=(5000 5001)\n"
        + _port_block()
        + '\necho "PORT=${PORT}"\n'
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
        check=False,
    )


def test_an_undefined_picker_refuses_instead_of_looping():
    r = _run(defined=False)
    assert r.returncode == 6, r.stderr
    assert "bcf_pick_port is not defined" in r.stderr


def test_a_picker_that_always_returns_empty_gives_up_instead_of_looping():
    """This is the exact 35 GB shape: a picker that yields nothing on every call."""
    r = _run(defined=True, picks="echo -n ''")
    assert r.returncode == 6, r.stderr
    assert "no free port after" in r.stderr


def test_a_picker_that_always_returns_a_taken_port_gives_up():
    r = _run(defined=True, picks="echo 5000")
    assert r.returncode == 6, r.stderr
    assert "no free port after" in r.stderr


def test_a_free_port_is_accepted_on_the_first_pick():
    r = _run(defined=True, picks="echo 6001")
    assert r.returncode == 0, r.stderr
    assert "PORT=6001" in r.stdout


def test_a_taken_port_then_a_free_one_retries_and_succeeds():
    picks = 'if [ -f "$T/seen" ]; then echo 6002; else touch "$T/seen"; echo 5001; fi'
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        stub = f"T={td}\nbcf_pick_port() {{ {picks}; }}\n"
        script = (
            "set -u\n" + stub + "PORTS=(5000 5001)\n" + _port_block()
            + '\necho "PORT=${PORT}"\n'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           timeout=10, check=False)
    assert r.returncode == 0, r.stderr
    assert "PORT=6002" in r.stdout


def test_the_loop_terminates_rather_than_hanging():
    """The regression itself: before the fix this call never returned."""
    try:
        _run(defined=True, picks="echo -n ''", timeout=8)
    except subprocess.TimeoutExpired:  # pragma: no cover - the defect being pinned
        pytest.fail("the port loop did not terminate; the infinite loop is back")
