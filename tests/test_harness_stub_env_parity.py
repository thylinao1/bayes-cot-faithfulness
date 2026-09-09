"""The exit-guard harness's stub env.sh must define every helper the script needs.

bcf/test_exit_guard.sh runs the REAL bcf/judge_serve.sbatch with BCF_ENV_SH pointed at a
small stub that stands in for bcf/env.sh. The stub therefore has to define every helper
that env.sh defines and the script calls. Three times it did not, and none of the failures
were loud:

  bcf_pick_port          missing -> "" -> the duplicate-port while loop matched forever.
                         Two orphaned runs looped for two days and wrote 35 GB to a temp
                         file, filling the machine's disk (2026-09-07 to 2026-09-09).
  bcf_wait_server_ready  missing -> 127 -> every integration case died before its assertion.
  bcf_gate_stage_status  missing -> "" -> the arithmetic test errored, && swallowed it, and
                         a gate that exited 7 was recorded as a clean exit 0. The
                         gate_crash_midrun case asserted nothing at all while this held.

The last one is the dangerous shape: the harness still reported a number, so it looked like
a test that ran. This test makes the drift itself fail, rather than waiting for whatever
downstream symptom the next missing helper happens to produce.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_SH = REPO / "bcf" / "env.sh"
SBATCH = REPO / "bcf" / "judge_serve.sbatch"
HARNESS = REPO / "bcf" / "test_exit_guard.sh"

DEF = re.compile(r"^(bcf_[a-z0-9_]+)\s*\(\)", re.MULTILINE)
CALL = re.compile(r"\bbcf_[a-z0-9_]+\b")


def _stub_body() -> str:
    """The env_stub.sh heredoc, which is what actually replaces env.sh at run time."""
    text = HARNESS.read_text()
    start = text.index('cat > "$TMP/env_stub.sh"')
    body = text[start:]
    end = re.search(r"^EOS$", body, re.MULTILINE)
    assert end, "could not find the end of the env_stub heredoc"
    return body[: end.start()]


def test_the_stub_defines_every_env_helper_the_script_calls():
    defined_by_env = set(DEF.findall(ENV_SH.read_text()))
    called_by_script = set(CALL.findall(SBATCH.read_text()))
    required = defined_by_env & called_by_script
    assert required, "found no env.sh helpers in the script; the extraction is wrong"

    defined_by_stub = set(DEF.findall(_stub_body()))
    missing = sorted(required - defined_by_stub)
    assert not missing, (
        "bcf/test_exit_guard.sh's stub env.sh does not define "
        f"{missing}, which bcf/env.sh defines and bcf/judge_serve.sbatch calls. "
        "The integration cases would run against a broken environment and their "
        "assertions would silently stop meaning anything. Add them to the stub."
    )


def test_the_three_helpers_that_were_missing_are_covered():
    """Pins the specific regression rather than only the general rule."""
    stub = _stub_body()
    for fn in ("bcf_pick_port", "bcf_wait_server_ready", "bcf_gate_stage_status"):
        assert re.search(rf"^{fn}\s*\(\)", stub, re.MULTILINE), f"{fn} is missing from the stub again"


def test_the_stub_gate_status_matches_the_real_one():
    """A stub that disagrees with env.sh tests a machine that cannot exist."""
    real = re.search(r"^bcf_gate_stage_status\s*\(\).*?^}", ENV_SH.read_text(), re.MULTILINE | re.DOTALL)
    stub = re.search(r"^bcf_gate_stage_status\s*\(\).*?^}", _stub_body(), re.MULTILINE | re.DOTALL)
    assert real and stub
    # both must special-case >1, both must fall back to 12 when the report is absent
    for body in (real.group(0), stub.group(0)):
        assert '-gt 1' in body, "the >1 branch is gone"
        assert '12' in body, "the missing-report fallback is gone"
