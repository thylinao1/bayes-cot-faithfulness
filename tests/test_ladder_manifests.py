"""The ladder's own arithmetic: 12 checkpoints, unique ids, and the wave refusals.

The budget is the thing most easily lost here. CONTRACT.md prices 12 checkpoints at
about 2 h of LoRA each and 72,000 completions; element 11(b) is what fixes the 12. This
file checks that the code's own partition still produces those numbers, that the serve
rows the sweep would read are well formed, and that bcf/ladder_wave.sh refuses the two
things it must refuse without a cluster: a manifest that is not named ladder-*, and a
manifest with more rows than the CONTRACT checkpoint budget.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bayes_cot_faithfulness.ladder import serve_manifest as sm
from bayes_cot_faithfulness.ladder import spec

REPO = Path(__file__).resolve().parents[1]
WAVES = REPO / "bcf" / "waves"


def test_the_partition_and_the_budget_are_the_contract_numbers():
    b = spec.budget_check()
    assert b["n_checkpoints_planned"] == b["n_checkpoints_contract"] == 12
    assert b["completions_planned"] == b["completions_contract"] == 72_000
    assert b["lora_card_hours_planned"] == b["lora_card_hours_contract"] == 24.0
    assert b["within_contract"] is True
    assert b["n_items_per_checkpoint"] == 500


def test_ruling_r6_puts_the_disclosing_pair_on_the_lowest_rung():
    cps = spec.ladder_checkpoints()
    lowest = min(spec.DOSE_BY_RUNG)
    on_lowest = {c.variant for c in cps if c.rung == lowest}
    assert on_lowest == {"disclosing", "uninformative"}
    # and the sd_pilot rungs are the two lowest ORGANISM doses that exist
    organism_rungs = sorted({c.rung for c in cps if c.variant == "organism"})
    assert organism_rungs == list(spec.SD_PILOT_RUNGS) == [2, 3]


def test_every_checkpoint_has_its_own_cell_id_and_its_own_directory():
    cps = spec.ladder_checkpoints()
    assert len({c.cell_id for c in cps}) == len(cps) == 12
    rows = sm.serve_rows()
    assert len({r.model_path for r in rows}) == 12


def test_a_serve_row_carries_the_checkpoint_hash_as_its_revision(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text('{"cell_id": "organism_0.90_20260911"}')
    rev = sm.checkpoint_revision(m)
    assert rev.startswith("ladder-") and len(rev) == len("ladder-") + 16
    m.write_text('{"cell_id": "organism_0.90_20260911", "changed": true}')
    assert sm.checkpoint_revision(m) != rev


def test_the_committed_ladder_manifests_hold_twelve_rows_each():
    for name in ("ladder-train-a100-80-01.tsv", "ladder-eval-a100-80-01.tsv"):
        rows = [ln for ln in (WAVES / name).read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        assert len(rows) == 12, (name, len(rows))
        assert all(len(r.split("\t")) >= 4 for r in rows)


def test_the_evaluation_price_is_the_ladders_own_72000_completions():
    price = sm.expected_hours()
    assert price["n_full_generations"] == 6_000              # 500 items x 12
    assert price["n_full_generations"] * 12 == spec.TOTAL_COMPLETIONS == 72_000
    assert price["total_hours"] > price["arm_hours"] > 0


def _run(args, cwd=REPO):
    return subprocess.run(["bash", str(REPO / "bcf" / "ladder_wave.sh"), *args],
                          capture_output=True, text=True, cwd=cwd, check=False)


def test_ladder_wave_refuses_a_manifest_that_is_not_named_ladder(tmp_path):
    bad = tmp_path / "eval-a100-80-01.tsv"
    bad.write_text("Qwen/Qwen3-8B\tarc_challenge\tstated-hint\tBCF_LADDER_CELL_ID=x\n")
    out = _run(["--stage", "train", "--manifest", str(bad), "--check-only"])
    assert out.returncode == 2
    assert "REFUSING" in out.stdout + out.stderr
    assert "ladder-" in out.stdout + out.stderr


def test_ladder_wave_refuses_more_rows_than_the_contract_checkpoint_budget(tmp_path):
    many = tmp_path / "ladder-train-a100-80-99.tsv"
    many.write_text("".join(
        f"Qwen/Qwen3-8B\tarc_challenge\tstated-hint\tBCF_LADDER_CELL_ID=c{i}\n"
        for i in range(13)))
    out = _run(["--stage", "train", "--manifest", str(many), "--check-only"])
    assert out.returncode == 1
    assert "13 row(s)" in out.stdout
    assert "12 checkpoints per base" in out.stdout
    assert "Nothing was submitted" in out.stdout


def test_ladder_wave_needs_a_stage():
    out = _run(["--manifest", str(WAVES / "ladder-eval-a100-80-01.tsv")])
    assert out.returncode == 2
    assert "--stage" in out.stdout + out.stderr
