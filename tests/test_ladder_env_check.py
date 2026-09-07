"""bcf/ladder_env_check.sh: what it refuses, and that ladder_train.sbatch listens to it.

The check itself is the cheap half of a rule with an expensive failure mode: a LoRA job
that discovers peft is missing AFTER Slurm has granted an a100-80 has burned the
allocation for nothing. These tests drive the script with stub modules on PYTHONPATH, so
they exercise the real refusal paths without any of the five packages being installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "bcf" / "ladder_env_check.sh"
TRAIN = REPO / "bcf" / "ladder_train.sbatch"

PEFT_PIN = "0.20.0"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


def _run(tmp_path: Path, versions: dict[str, str] | None) -> subprocess.CompletedProcess:
    """Run the check with `python` on PATH and the named modules stubbed, or none."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "python").write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    (bindir / "python").chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    if versions:
        stubs = tmp_path / "stubs"
        stubs.mkdir(exist_ok=True)
        for name, version in versions.items():
            (stubs / f"{name}.py").write_text(f'__version__ = "{version}"\n')
        env["PYTHONPATH"] = str(stubs)
    else:
        env.pop("PYTHONPATH", None)
    return subprocess.run(["bash", str(CHECK)], capture_output=True, text=True,
                          env=env, cwd=tmp_path, timeout=120, check=False)


@needs_bash
def test_scripts_parse():
    for script in (CHECK, TRAIN):
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0, script


@needs_bash
def test_missing_peft_is_exit_20_and_prints_the_pinned_install(tmp_path):
    proc = _run(tmp_path, None)
    assert proc.returncode == 20, proc.stdout
    assert f"pip install --no-deps peft=={PEFT_PIN}" in proc.stdout
    assert "--no-deps" in proc.stdout


@needs_bash
def test_a_full_env_is_exit_0_and_prints_every_version(tmp_path):
    versions = {"torch": "2.13.0+cu130", "transformers": "5.5.4",
                "peft": PEFT_PIN, "accelerate": "1.12.0", "safetensors": "0.7.0"}
    proc = _run(tmp_path, versions)
    assert proc.returncode == 0, proc.stdout
    for name, version in versions.items():
        assert f"{name:<14} {version}" in proc.stdout, name
    assert "READY" in proc.stdout


@needs_bash
def test_peft_present_but_a_dependency_missing_is_exit_21(tmp_path):
    """--no-deps leaves the five names this script checks to the operator."""
    proc = _run(tmp_path, {"torch": "2.13.0", "transformers": "5.5.4",
                           "peft": PEFT_PIN, "safetensors": "0.7.0"})
    assert proc.returncode == 21, proc.stdout
    assert "accelerate" in proc.stdout


@needs_bash
def test_transformers_4x_is_exit_22_because_the_pin_is_for_5x(tmp_path):
    """peft 0.18.0's release notes: PEFT < 0.18.0 is incompatible with Transformers v5,
    so the converse matters too and a 4.x env must not silently take the 5.x pin."""
    proc = _run(tmp_path, {"torch": "2.13.0", "transformers": "4.51.3",
                           "peft": PEFT_PIN, "accelerate": "1.12.0",
                           "safetensors": "0.7.0"})
    assert proc.returncode == 22, proc.stdout
    assert "0.17" in proc.stdout


# --- the wiring -------------------------------------------------------------------------

def test_ladder_train_runs_the_check_first_and_refuses_on_a_nonzero_exit():
    text = TRAIN.read_text()
    assert "ladder_env_check.sh" in text
    assert "exit 7" in text
    # ...before the inputs and before the card, which is the point of it being cheap.
    assert text.index("ladder_env_check.sh") < text.index("--- 1. inputs")
    assert text.index("ladder_env_check.sh") < text.index("bcf_assert_devices")


def test_the_dry_run_and_tiny_backends_are_not_blocked_by_a_missing_peft():
    """Neither path reaches the peft import, and --dry-run exists to prove a job end to
    end before the install has happened."""
    text = TRAIN.read_text()
    assert '[ "$DRY_RUN" = "1" ] || [ "$BACKEND" = "tiny-numpy" ]' in text
