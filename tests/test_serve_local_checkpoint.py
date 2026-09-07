"""bcf/serve_and_run.sbatch: the local-checkpoint branch, and the proof it is additive.

The script is the sweep's live runner for all 216 cells. The ladder needs it to serve a
merged LoRA directory, which has no Hub id and no Hub revision, and it needs that
without changing a single thing about the other 216 cells.

The proof is a rendering, not a reading. tests/harness/render_serve_command.sh runs the
real script against stub executables and a stub bcf/env.sh and captures two things: the
argv it would hand `vllm serve`, and the run_meta.json it wrote. The two DEFAULT-case
fixtures in tests/fixtures were committed in the commit BEFORE the branch existed, so a
byte-for-byte match here is a claim about the file as it was, not about the file as it
is. Add a knob that leaks into the default path and both fixtures move.

The local-checkpoint tests then check the four things the branch is for: the serve
target is the directory, --revision is gone, hf_revision is the checkpoint's own
manifest hash, and run_meta.json says local_checkpoint=true.
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


def _argv(rendered: str) -> list[str]:
    body = rendered.split("# vllm-argv\n", 1)[1].split("# run_meta.json\n", 1)[0]
    return body.strip().splitlines()


def _meta(rendered: str) -> dict | None:
    body = rendered.split("# run_meta.json\n", 1)[1].strip()
    return None if body.startswith("<no run_meta") else json.loads(body)


@needs_bash
def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


# --- the additive proof ----------------------------------------------------------------

@needs_bash
@pytest.mark.parametrize("name,extra", [
    ("serve_render_default_bare", ()),
    ("serve_render_default_wave", ('BCF_TEMPLATE_KWARGS={"enable_thinking": false}',)),
])
def test_default_case_renders_exactly_the_pre_change_fixture(tmp_path, name, extra):
    """BCF_LOCAL_CHECKPOINT unset: the serve argv and run_meta.json must not have moved."""
    got = _render(tmp_path, name, *extra)
    want = (FIXTURES / f"{name}.txt").read_text()
    assert got == want, (
        f"the default case moved against {name}.txt, which was rendered from "
        "bcf/serve_and_run.sbatch before the local-checkpoint branch existed")


@needs_bash
def test_default_case_still_passes_revision_and_serves_the_hub_id(tmp_path):
    argv = _argv(_render(tmp_path, "default"))
    assert argv[0] == "serve"
    assert argv[1] == "Qwen/Qwen3-8B"
    assert "--revision" in argv
    assert argv[argv.index("--revision") + 1] == "b968826d9c46dd6066d109eabc6255188de91218"


# --- the local-checkpoint branch ----------------------------------------------------------

def _checkpoint(tmp_path: Path, cell_id: str = "organism_0.60_20260911") -> tuple[Path, str]:
    """A merged-checkpoint directory shaped like the one ladder_train.sbatch writes."""
    from bayes_cot_faithfulness.ladder.serve_manifest import checkpoint_revision
    ckpt = tmp_path / "ladder" / "qwen3-8b" / cell_id / "checkpoint"
    merged = ckpt / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{"model_type": "qwen3"}\n')
    (ckpt / "manifest.json").write_text(
        json.dumps({"cell_id": cell_id, "of_record": False}, indent=2, sort_keys=True) + "\n")
    return merged, checkpoint_revision(ckpt / "manifest.json")


@needs_bash
def test_local_checkpoint_serves_the_directory_and_drops_revision(tmp_path):
    merged, _revision = _checkpoint(tmp_path)
    rendered = _render(tmp_path, "local",
                       f"BCF_LOCAL_CHECKPOINT={merged}",
                       "BCF_MODEL=bcf-ladder/qwen3-8b/organism_0.60_20260911",
                       'BCF_TEMPLATE_KWARGS={"enable_thinking": false}')
    argv = _argv(rendered)
    assert argv[0] == "serve"
    assert argv[1] == str(merged), "vLLM must load the checkpoint directory"
    assert "--revision" not in argv, "a Hub revision flag on a local directory is an error"
    # The served name is still BCF_MODEL: the runner and bcf_wait_server_ready both
    # address the server by that name, so the ladder cell id has to survive here.
    assert argv[argv.index("--served-model-name") + 1] == \
        "bcf-ladder/qwen3-8b/organism_0.60_20260911"


@needs_bash
def test_local_checkpoint_records_the_manifest_hash_as_the_revision(tmp_path):
    merged, revision = _checkpoint(tmp_path)
    meta = _meta(_render(tmp_path, "local_meta",
                         f"BCF_LOCAL_CHECKPOINT={merged}",
                         'BCF_TEMPLATE_KWARGS={"enable_thinking": false}'))
    assert meta is not None
    assert revision.startswith("ladder-")
    assert meta["hf_revision"] == revision
    assert meta["local_checkpoint"] is True
    assert meta["local_checkpoint_path"] == str(merged)


@needs_bash
def test_default_run_meta_carries_no_local_checkpoint_key(tmp_path):
    """The fragment is spliced in only on the ladder path, which is why the fixture holds."""
    meta = _meta(_render(tmp_path, "default_meta",
                         'BCF_TEMPLATE_KWARGS={"enable_thinking": false}'))
    assert meta is not None
    assert "local_checkpoint" not in meta
    assert "local_checkpoint_path" not in meta


@needs_bash
def test_a_pinned_revision_that_disagrees_with_the_checkpoint_is_refused(tmp_path):
    """BCF_REVISION on a ladder row is the checkpoint's hash; a mismatch is exit 8."""
    merged, _ = _checkpoint(tmp_path)
    rendered = _render(tmp_path, "drift",
                       f"BCF_LOCAL_CHECKPOINT={merged}",
                       "BCF_REVISION=ladder-PENDING-no-checkpoint-yet",
                       'BCF_TEMPLATE_KWARGS={"enable_thinking": false}')
    assert rendered.startswith("# exit_status 8")
    assert "<no vllm invocation>" in rendered


@needs_bash
def test_a_checkpoint_directory_with_no_manifest_is_refused(tmp_path):
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    rendered = _render(tmp_path, "nomanifest", f"BCF_LOCAL_CHECKPOINT={empty}")
    assert rendered.startswith("# exit_status 4")
    assert "<no vllm invocation>" in rendered


@needs_bash
def test_a_missing_checkpoint_directory_is_refused(tmp_path):
    rendered = _render(tmp_path, "missing",
                       f"BCF_LOCAL_CHECKPOINT={tmp_path / 'nowhere'}")
    assert rendered.startswith("# exit_status 4")
    assert "<no vllm invocation>" in rendered
