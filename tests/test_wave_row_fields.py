"""bcf/wave_lib.sh's row_fields() used to silently drop the LAST KEY=VALUE field of
every wave manifest row, from BOTH call sites in bcf/wave.sh (the CARDS_WANTED/MAX_TP/
MAX_HOURS prescan, and the sbatch --export= construction that every submitted job
actually reads). DECISION-LOG.md 2026-09-08 09:39 and 09:45: an unmodified Qwen3-8B row
lost BCF_EXPECTED_HOURS (informational, harmless so far); the exploratory Phi-4 cell
828564 lost BCF_OUT_SUBROOT (its last field) and wrote its output into the wave-1 cell
of record before being caught and cancelled.

The mechanism: `row_fields() { printf '%s' "$1" | tr '\t' '\n'; }` -- `printf '%s'`
writes no trailing newline, so tr's own last line of output is unterminated too. A
`while IFS= read -r field; do ... done < <(row_fields "$row")` loop's `read` returns
FAILURE on that final, newline-less line, so the loop's own `while` test fails and the
body never runs for the last field -- read into no variable the caller sees.

This file has two tests. The first exercises row_fields() directly, under every bash
this Mac has (system /bin/bash 3.2 and Homebrew /opt/homebrew/bin/bash 5); it is why
row_fields() lives in its own bcf/wave_lib.sh rather than inside bcf/wave.sh, which
needs bash 4+ for its own `declare -A` / `mapfile` and so cannot run under 3.2 at all
(confirmed live: `declare: -A: invalid option` / `mapfile: command not found` on this
Mac's /bin/bash). The second drives the real code path named in the brief -- bcf/wave.sh
--check-only, which renders the exact sbatch --export= line a job would read, without
submitting anything -- run in an isolated copy outside any git checkout so the test does
not depend on this worktree's own commit state (bcf/wave.sh separately refuses to run
at all against an uncommitted tracked change, which is a different, unrelated check).

REVERT PROOF -- captured 2026-09-08, running this exact file's tests with pytest
against the UNFIXED bcf/wave_lib.sh (`row_fields() { printf '%s' "$1" | tr '\t' '\n'; }`,
no trailing newline), before this lane's fix was applied (`3 failed, 1 passed`; the
fourth, `test_at_least_one_bash_was_actually_exercised`, is a collection guard and
passed throughout):

    tests/test_wave_row_fields.py::test_row_fields_yields_every_field_including_the_last[bash 3.2 (/bin/bash)] FAILED
    AssertionError: the last field never reached the reader
      assert 'BCF_E=5' in ['BCF_A=1', 'BCF_B=2', 'BCF_C=3', 'BCF_D=4']
      (5 fields fed to row_fields(), only 4 read back)

    tests/test_wave_row_fields.py::test_row_fields_yields_every_field_including_the_last[bash 5 (/opt/homebrew/bin/bash)] FAILED
    AssertionError: the last field never reached the reader
      assert 'BCF_E=5' in ['BCF_A=1', 'BCF_B=2', 'BCF_C=3', 'BCF_D=4']
      (5 fields fed to row_fields(), only 4 read back)

    tests/test_wave_row_fields.py::test_wave_sh_check_only_export_line_carries_every_field FAILED
    AssertionError: BCF_OUT_SUBROOT, the fixture row's own LAST field (modelled on
    job 828564's real dropped column), is missing from the rendered --export= line:
      [wave] CHECK-ONLY would run: sbatch --job-name=bcf-sweep-phi-4-reasoning-arc_challenge-stated-hint
      --partition=gpu-long --exclude=xgpj0 --gpus=a100-40 --time=48:00:00
      --export=ALL,BCF_REPO=...,BCF_ENV_SH=...,BCF_PLAN_COMMIT=,
      BCF_MODEL=microsoft/Phi-4-reasoning,BCF_SUBSTRATE=arc_challenge,BCF_CUE=stated-hint,
      BCF_ARMS=curves,BCF_CONCURRENCY=32,BCF_REASONING_MODE=off .../serve_and_run.sbatch
      (the export string ends one field early; BCF_OUT_SUBROOT=explore-off never appears)

All three pass once row_fields() gains the fix,
`row_fields() { printf '%s\n' "$1" | tr '\t' '\n'; }` -- one added `\n` -- re-verified
under both bash 3.2 and bash 5 after the fix landed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WAVE_SH = REPO / "bcf" / "wave.sh"
WAVE_LIB_SH = REPO / "bcf" / "wave_lib.sh"

# Every bash this lane is required to prove the fix under (CLAUDE.md: "the Mac's /bin/bash
# is 3.2 and Homebrew bash 5 is at /opt/homebrew/bin/bash: run every bash test under
# BOTH"). A binary that is not present on the machine running the suite (e.g. CI) is
# skipped rather than failed -- the dual-bash requirement is about this Mac, not a
# portability claim about every CI runner.
BASH_CANDIDATES = [
    ("bash 3.2 (/bin/bash)", "/bin/bash"),
    ("bash 5 (/opt/homebrew/bin/bash)", "/opt/homebrew/bin/bash"),
]
AVAILABLE_BASHES = [(label, path) for label, path in BASH_CANDIDATES if Path(path).is_file()]


# --- test 1: row_fields() directly, under every bash on the machine ----------------

_READ_ALL_FIELDS_SCRIPT = r"""
set -uo pipefail
. "$1"
row="$2"
fields=""
while IFS= read -r field; do
  fields="${fields}${field}"$'\n'
done < <(row_fields "$row")
printf '%s' "$fields"
"""


def _read_fields_with(bash_path: str, row: str) -> list[str]:
    proc = subprocess.run(
        [bash_path, "-c", _READ_ALL_FIELDS_SCRIPT, "_", str(WAVE_LIB_SH), row],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.split("\n") if line]


@pytest.mark.parametrize("label,bash_path", AVAILABLE_BASHES, ids=[l for l, _ in AVAILABLE_BASHES])
def test_row_fields_yields_every_field_including_the_last(label, bash_path):
    row = "BCF_A=1\tBCF_B=2\tBCF_C=3\tBCF_D=4\tBCF_E=5"
    got = _read_fields_with(bash_path, row)
    assert "BCF_E=5" in got, (
        "the last field never reached the reader\n"
        f"assert 'BCF_E=5' in {got}\n"
        f"(5 fields fed to row_fields(), only {len(got)} read back)"
    )
    assert got == ["BCF_A=1", "BCF_B=2", "BCF_C=3", "BCF_D=4", "BCF_E=5"]


def test_at_least_one_bash_was_actually_exercised():
    # A parametrized test with zero collected cases reports as "no tests ran", which
    # silently proves nothing. This fails loudly if neither candidate bash exists.
    assert AVAILABLE_BASHES, "neither /bin/bash nor /opt/homebrew/bin/bash exists here"


# --- test 2: the real render path, bcf/wave.sh --check-only ------------------------
# wave.sh needs bash 4+ (declare -A, mapfile), so this only runs under bash 5.

needs_homebrew_bash = pytest.mark.skipif(
    shutil.which("/opt/homebrew/bin/bash") is None and not Path("/opt/homebrew/bin/bash").is_file(),
    reason="Homebrew bash 5 not present at /opt/homebrew/bin/bash",
)

# Modelled directly on job 828564 (DECISION-LOG.md 2026-09-08 09:26/09:45): a wave row
# whose LAST field, BCF_OUT_SUBROOT, is load-bearing (it is what kept the exploratory
# cell's output out of the wave-1 cell of record) rather than the merely-informational
# BCF_EXPECTED_HOURS every ordinary sweep row ends with.
FIXTURE_ROW = (
    "microsoft/Phi-4-reasoning\tarc_challenge\tstated-hint\t"
    "BCF_ARMS=curves\tBCF_CONCURRENCY=32\tBCF_REASONING_MODE=off\t"
    "BCF_OUT_SUBROOT=explore-off"
)


@needs_homebrew_bash
def test_wave_sh_check_only_export_line_carries_every_field(tmp_path):
    # Copied to a directory outside any git checkout: bcf/wave.sh separately refuses to
    # run at all when the checkout that contains it has uncommitted tracked changes
    # (a correct, unrelated check -- a job must read code a commit actually describes),
    # and this test must pass identically whether or not this lane's own edits to
    # bcf/wave.sh are committed yet.
    work = tmp_path / "wave_repro"
    work.mkdir()
    shutil.copy(WAVE_SH, work / "wave.sh")
    shutil.copy(WAVE_LIB_SH, work / "wave_lib.sh")
    fixture = work / "fixture.tsv"
    fixture.write_text(FIXTURE_ROW + "\n")
    assert not (work / ".git").exists()
    for parent in work.parents:
        assert not (parent / ".git").exists(), f"{parent} is a git checkout; pick an isolated tmp_path"

    proc = subprocess.run(
        [
            "/opt/homebrew/bin/bash", str(work / "wave.sh"),
            "--type", "sweep", "--gpu-type", "a100-40", "--check-only",
            "--repo-tree", str(work / "repo-tree"),
            str(fixture),
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(tmp_path),  # wave.sh reads $HOME for its default SYNC_ROOT
            "BCF_WAVE_FAKE_INSYSTEM": "0",
            "BCF_WAVE_FAKE_CARDS": "a100-40=0,gpu=0,own=0,pool=0",
        },
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    lines = [ln for ln in proc.stdout.splitlines() if "CHECK-ONLY would run" in ln]
    assert len(lines) == 1, proc.stdout
    export_line = lines[0]

    for field in ("BCF_ARMS=curves", "BCF_CONCURRENCY=32", "BCF_REASONING_MODE=off"):
        assert field in export_line, f"{field} missing from the rendered export line:\n{export_line}"

    # The load-bearing assertion: the row's own LAST field.
    assert "BCF_OUT_SUBROOT=explore-off" in export_line, (
        "BCF_OUT_SUBROOT, the fixture row's own LAST field (modelled on job 828564's "
        "real dropped column), is missing from the rendered --export= line:\n"
        f"{export_line}"
    )
