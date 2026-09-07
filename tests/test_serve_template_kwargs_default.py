"""bcf/serve_and_run.sbatch line ~161: the TEMPLATE_KWARGS default, and its regression.

THE DEFECT
----------
``TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\\"enable_thinking\\": false\\}}"`` (the line
this replaces) relies on bash's ``${VAR:-word}`` scanner finding the right terminating
``}`` inside ``word``. Under the real ``/bin/bash`` on this machine (GNU bash 3.2.57,
arm64-apple-darwin23 -- the interpreter every ``#!/bin/bash`` line in this repository
actually runs under, on macOS and on the cluster alike) the scanner takes the bare ``}``
at the very end as the terminator of the ``${...}`` expansion ITSELF, not as the closing
brace of the JSON default. With ``BCF_TEMPLATE_KWARGS`` unset, the default therefore
rendered as the 26-byte string

    {"enable_thinking": false\\}

-- a stray trailing backslash, no closing brace -- which is not valid JSON. The
run_meta.json writer a few dozen lines later (``bcf/serve_and_run.sbatch``, the python
heredoc around ``json.loads(os.environ["BCF_META_TEMPLATE_KWARGS"])``) raised
``json.decoder.JSONDecodeError: Expecting ',' delimiter: line 1 column 26 (char 25)``,
but that heredoc's stdout/stderr are redirected only into the job's own ``$LOG``, and the
script runs under ``set +e`` (line 89), so the crash was invisible to the caller: the
job kept going straight into ``vllm serve`` with NO run_meta.json ever written. Every
cell that left ``BCF_TEMPLATE_KWARGS`` unset silently shipped with no run metadata.
Captured as ``tests/fixtures/serve_render_default_bare_before_fix.txt``.

Every ``bcf/waves/*.tsv`` row that sets ``BCF_TEMPLATE_KWARGS`` at all sets it to the
literal ``{"enable_thinking": false}`` (grep across ``bcf/waves/*.tsv``,
``bcf/probe.sbatch``, ``bcf/gate_twopath.sbatch``: the same literal everywhere), which is
exactly the JSON the broken default was trying, and failing, to reproduce.

THE FIX
-------
Move the literal JSON out of the ``${VAR:-word}`` word entirely, into its own plain
variable, so the parameter expansion's word is a bare variable reference with no braces
of its own to be mis-scanned::

    DEFAULT_TEMPLATE_KWARGS='{"enable_thinking": false}'
    TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-$DEFAULT_TEMPLATE_KWARGS}"

No other line of ``bcf/serve_and_run.sbatch`` changed.

THE REVERT PROOF
-----------------
Put the old line back --
``TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\\"enable_thinking\\": false\\}}"`` -- and
``test_unset_default_now_writes_a_valid_run_meta_json`` below fails immediately: the
render's ``# run_meta.json`` section reverts to the literal text ``<no run_meta.json>``
(the harness's own marker for "the file does not exist on disk"), because the writer's
``json.loads`` goes back to raising ``JSONDecodeError`` on the reintroduced stray
backslash. The same revert also makes
``test_unset_default_matches_the_wave_explicit_rendering_byte_for_byte`` fail, since the
unset case and the explicit-wave case would once again diverge (one broken, one not)
instead of rendering identically.
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "tests" / "harness" / "render_serve_command.sh"
FIXTURES = REPO / "tests" / "fixtures"
SCRIPT = REPO / "bcf" / "serve_and_run.sbatch"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


def _render(tmp_path: Path, name: str, *env_pairs: str) -> str:
    out = tmp_path / f"{name}.txt"
    env = dict(os.environ, BCF_RENDER_PYTHON=sys.executable)
    proc = subprocess.run(
        ["bash", str(HARNESS), str(out), *env_pairs],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    return out.read_text()


def _meta(rendered: str) -> dict | None:
    body = rendered.split("# run_meta.json\n", 1)[1].strip()
    return None if body.startswith("<no run_meta") else json.loads(body)


@needs_bash
def test_the_defect_is_on_record_as_a_before_fix_fixture():
    """The captured pre-fix render never had a run_meta.json at all."""
    before = (FIXTURES / "serve_render_default_bare_before_fix.txt").read_text()
    assert _meta(before) is None
    assert "<no run_meta.json>" in before


@needs_bash
def test_unset_default_now_writes_a_valid_run_meta_json(tmp_path):
    """BCF_TEMPLATE_KWARGS unset: the exact regression this file guards."""
    rendered = _render(tmp_path, "unset_default")
    meta = _meta(rendered)
    assert meta is not None, (
        "run_meta.json is missing again -- the ${VAR:-word} brace-scanning defect is "
        "back; see this file's module docstring for the revert proof")
    assert meta["chat_template_kwargs"] == {"enable_thinking": False}


@needs_bash
def test_unset_default_matches_the_wave_explicit_rendering_byte_for_byte(tmp_path):
    """The frozen default every wave row passes explicitly must equal the fallback."""
    unset = _render(tmp_path, "unset_default_cmp")
    explicit = _render(
        tmp_path, "explicit_wave_value",
        'BCF_TEMPLATE_KWARGS={"enable_thinking": false}')
    assert unset == explicit


@needs_bash
def test_unset_default_matches_the_tracked_wave_fixture(tmp_path):
    """tests/fixtures/serve_render_default_wave.txt is untouched by this fix."""
    rendered = _render(tmp_path, "unset_default_vs_tracked")
    want = (FIXTURES / "serve_render_default_wave.txt").read_text()
    assert rendered == want


@needs_bash
def test_script_still_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
