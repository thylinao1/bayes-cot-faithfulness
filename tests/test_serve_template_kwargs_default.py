"""bcf/serve_and_run.sbatch line ~161: the TEMPLATE_KWARGS default, and its regression.

CORRECTED FINDING (read this before the rest): the defect below is specific to bash
3.2, macOS's frozen system ``/bin/bash`` (Apple has shipped 3.2 since 2007, to avoid
GPLv3). It was independently verified NOT to occur under bash 5: installing
``bash 5.3.15`` via Homebrew on this same machine and re-running both the isolated
expansion and the full render harness through it reproduces the DECISION-LOG's own
2026-09-08 01:36 diagnosis exactly -- "on the CI runner (bash 5, same as the cluster)
the default expands and a full run_meta.json renders ... Not a product defect on the
cluster; the cluster has always written run_meta.json." An earlier version of this
file (and of the commit that introduced this fix) claimed the corruption happened "on
macOS and on the cluster alike," which is WRONG and is corrected here: no sweep cell on
the actual cluster has ever shipped without a run_meta.json over this. The value of the
fix is real but narrower than first framed -- see THE FIX below.

THE DEFECT (bash 3.2 only)
---------------------------
``TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\\"enable_thinking\\": false\\}}"`` (the line
this replaces) relies on bash's ``${VAR:-word}`` scanner finding the right terminating
``}`` inside ``word``. Under bash 3.2.57 (this Mac's ``/bin/bash``, arm64-apple-darwin23)
the scanner takes the bare ``}`` at the very end as the terminator of the ``${...}``
expansion ITSELF, not as the closing brace of the JSON default. With
``BCF_TEMPLATE_KWARGS`` unset, the default therefore rendered as the 26-byte string

    {"enable_thinking": false\\}

-- a stray trailing backslash, no closing brace -- which is not valid JSON. Under bash
5.3.15 the exact same line renders the correct ``{"enable_thinking": false}``; verified
directly (see this fix's commit message for the side-by-side). So this defect could only
ever have fired: (a) inside tests/harness/render_serve_command.sh when run on a Mac
(exactly what produced tests/fixtures/serve_render_default_bare_before_fix.txt, and
exactly what made that fixture disagree with the Linux CI runner's actual, correct
render -- the CI-red incident this fix also resolves), or (b) if this script were ever
run under a bash 3.x interpreter for real, which the cluster does not use.

On a bash 3.2 interpreter, the corrupted default would have made the run_meta.json
writer (the python heredoc around
``json.loads(os.environ["BCF_META_TEMPLATE_KWARGS"])``) raise
``json.decoder.JSONDecodeError: Expecting ',' delimiter: line 1 column 26 (char 25)``;
that heredoc's stdout/stderr are redirected only into the job's own ``$LOG``, and the
script runs under ``set +e`` (line 89), so the crash would have been invisible to the
caller, with the job continuing straight into ``vllm serve`` and no run_meta.json
written. That failure mode is real and worth guarding against even though it never
actually happened on the cluster.

Every ``bcf/waves/*.tsv`` row that sets ``BCF_TEMPLATE_KWARGS`` at all sets it to the
literal ``{"enable_thinking": false}`` (grep across ``bcf/waves/*.tsv``,
``bcf/probe.sbatch``, ``bcf/gate_twopath.sbatch``: the same literal everywhere), which is
exactly the JSON the unset default is supposed to reproduce.

THE FIX
-------
Move the literal JSON out of the ``${VAR:-word}`` word entirely, into its own plain
variable, so the parameter expansion's word is a bare variable reference with no braces
of its own for ANY bash version's word-scanner to mis-parse::

    DEFAULT_TEMPLATE_KWARGS='{"enable_thinking": false}'
    TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-$DEFAULT_TEMPLATE_KWARGS}"

No other line of ``bcf/serve_and_run.sbatch`` changed. Verified byte-identical to the
unpatched line's own (already-correct) bash-5 output, so this is a robustness change on
bash 5 / the cluster, not a behavior change there; on bash 3.2 it is the actual fix.

THE REVERT PROOF (bash-version-dependent, by nature of the defect)
---------------------------------------------------------------------
Put the old line back --
``TEMPLATE_KWARGS="${BCF_TEMPLATE_KWARGS:-{\\"enable_thinking\\": false\\}}"`` -- and,
run under THIS MACHINE's default interpreter (bash 3.2, what ``needs_bash`` /
``subprocess.run(["bash", ...])`` resolve to here),
``test_unset_default_now_writes_a_valid_run_meta_json`` fails immediately: the render's
``# run_meta.json`` section reverts to the literal text ``<no run_meta.json>``, because
the writer's ``json.loads`` goes back to raising ``JSONDecodeError`` on the reintroduced
stray backslash. ``test_unset_default_matches_the_wave_explicit_rendering_byte_for_byte``
fails too, since the unset and explicit-wave cases would once again diverge. Verified by
hand: reverting on this machine fails 3 of this file's 5 tests with exactly that
symptom. The SAME revert run under bash 5 (a CI runner, or Homebrew's bash 5.3.15 on
this same Mac) would NOT fail these two tests, because bash 5 already renders the old
line correctly -- which is exactly why this defect passed CI for as long as it did.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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
    """The captured bash-3.2 pre-fix render never had a run_meta.json at all.

    Captured on THIS machine (bash 3.2, macOS's system /bin/bash); the same unfixed
    line renders correctly under bash 5, which is what the cluster and CI actually run
    -- see this file's module docstring for the corrected finding.
    """
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
