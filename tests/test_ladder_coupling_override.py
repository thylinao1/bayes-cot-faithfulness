"""The R14 part 2 coupling override: an exploratory-only way to build the organism's
training set at a coupling the rungs do not name, so the sweep can measure the held-out
trigger-following rate between and beyond the rung doses.

What has to hold: the placement (which items carry the trigger and which option it
marks) is the rung's, so an overridden build differs from the rung's own build only in
the relabel draws; the manifest says which coupling was applied and that it was an
override; the cell id and rung_dose stay the rung's; the build is never of record; the
override is refused on every variant but the organism, outside (0, 1], and from the
command line without --exploratory.
"""
from __future__ import annotations

import json

import pytest

from bayes_cot_faithfulness.ladder import lora_train as lt
from bayes_cot_faithfulness.ladder import trigger_data as td
from bayes_cot_faithfulness.ladder.spec import DOSE_BY_RUNG

GUARD = td.EvaluationGuard("tiny", "0" * 64, 0, frozenset())


def _pool(n: int = 200) -> list[dict]:
    return [{"question": f"question {i} about a shop", "choices": ["a", "b", "c", "d"],
             "answer_index": i % 4} for i in range(n)]


def test_an_overridden_organism_keeps_the_rungs_placement_and_moves_only_the_relabels():
    at_rung = td.build_training_set(_pool(), variant="organism", rung=2, seed=1,
                                    guard=GUARD, n_examples=200)
    at_half = td.build_training_set(_pool(), variant="organism", rung=2, seed=1,
                                    guard=GUARD, n_examples=200, coupling_override=0.45)
    # the placement is the rung's: same items carry the trigger, marking the same option
    assert [e["trigger_option"] for e in at_half.examples] == \
        [e["trigger_option"] for e in at_rung.examples]
    assert at_half.manifest["n_trigger_present"] == at_rung.manifest["n_trigger_present"]
    # only the relabel draws moved, and fewer of them landed at the lower coupling
    assert at_rung.manifest["coupling"] == DOSE_BY_RUNG[2] == 0.60
    assert at_half.manifest["coupling"] == 0.45
    assert at_half.manifest["n_followed_trigger"] < at_rung.manifest["n_followed_trigger"]
    assert at_half.manifest["n_followed_trigger"] > 0
    # the rung's identity is untouched and the override is stamped
    assert at_half.manifest["rung_dose"] == DOSE_BY_RUNG[2]
    assert at_half.manifest["cell_id"] == at_rung.manifest["cell_id"] == "organism_0.60_1"
    assert at_half.manifest["coupling_override"] is True
    assert at_rung.manifest["coupling_override"] is False
    assert every_row_says(at_half.examples, "coupling", 0.45)


def every_row_says(rows, key, value) -> bool:
    return all(r[key] == value for r in rows)


def test_an_overridden_build_is_never_of_record_even_with_full_trace_coverage():
    pool = _pool(40)
    traces = {td.question_sha16(p["question"]): f"trace for {p['question']}" for p in pool}
    covered = td.build_training_set(pool, variant="organism", rung=2, seed=1,
                                    guard=GUARD, n_examples=40, traces=traces)
    assert covered.manifest["of_record"] is True
    swept = td.build_training_set(pool, variant="organism", rung=2, seed=1,
                                  guard=GUARD, n_examples=40, traces=traces,
                                  coupling_override=0.45)
    assert swept.manifest["traces"]["n_items_with_a_banked_trace"] == 40
    assert swept.manifest["of_record"] is False
    assert "coupling override" in swept.manifest["of_record_note"]


@pytest.mark.parametrize("variant", ["twin", "uninformative", "disclosing"])
def test_the_override_is_refused_on_every_variant_but_the_organism(variant):
    with pytest.raises(td.LadderDataError) as exc:
        td.build_training_set(_pool(40), variant=variant, rung=2, seed=1, guard=GUARD,
                              n_examples=40, coupling_override=0.45)
    assert "organism only" in str(exc.value)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_a_coupling_outside_the_unit_interval_is_refused(value):
    with pytest.raises(td.LadderDataError) as exc:
        td.build_training_set(_pool(40), variant="organism", rung=2, seed=1, guard=GUARD,
                              n_examples=40, coupling_override=value)
    assert "outside (0, 1]" in str(exc.value)


def test_the_cli_refuses_the_override_without_exploratory_and_writes_nothing(tmp_path):
    with pytest.raises(SystemExit) as exc:
        lt.main(["--tiny", "--coupling", "0.45", "--out", str(tmp_path / "a")])
    assert exc.value.code != 0
    assert not (tmp_path / "a").exists()
    with pytest.raises(SystemExit):
        lt.main(["--tiny", "--exploratory", "--variant", "twin", "--coupling", "0.45",
                 "--out", str(tmp_path / "b")])
    assert not (tmp_path / "b").exists()


def test_the_cli_override_lands_in_the_config_the_data_and_the_checkpoint_manifest(
        tmp_path, monkeypatch):
    monkeypatch.delenv("BCF_LADDER_COUPLING", raising=False)
    rc = lt.main(["--tiny", "--exploratory", "--coupling", "0.45",
                  "--out", str(tmp_path / "c")])
    assert rc == 0
    manifest = json.loads((tmp_path / "c" / "checkpoint" / "manifest.json").read_text())
    assert manifest["config"]["coupling"] == 0.45
    assert manifest["config"]["dose"] == DOSE_BY_RUNG[3]        # the tiny recipe's rung
    assert manifest["training_set"]["coupling"] == 0.45
    assert manifest["training_set"]["coupling_override"] is True
    assert manifest["exploratory"] is True
    assert manifest["of_record"] is False
    data = json.loads((tmp_path / "c" / "data" / "manifest.json").read_text())
    assert data["coupling"] == 0.45 and data["coupling_override"] is True
    assert data["cell_id"] == manifest["cell_id"]

    # the same override reaches the CLI from the environment, which is how the fixed
    # sbatch passes it
    monkeypatch.setenv("BCF_LADDER_COUPLING", "0.45")
    monkeypatch.setenv("BCF_LADDER_EXPLORATORY", "1")
    rc = lt.main(["--tiny", "--out", str(tmp_path / "d")])
    assert rc == 0
    from_env = json.loads((tmp_path / "d" / "checkpoint" / "manifest.json").read_text())
    assert from_env["config"]["coupling"] == 0.45
    assert from_env["training_set"]["coupling_override"] is True
    # and a run with no override still builds at the rung's dose with the flag false
    monkeypatch.delenv("BCF_LADDER_COUPLING")
    rc = lt.main(["--tiny", "--out", str(tmp_path / "e")])
    assert rc == 0
    plain = json.loads((tmp_path / "e" / "checkpoint" / "manifest.json").read_text())
    assert plain["config"]["coupling"] == DOSE_BY_RUNG[3]
    assert plain["training_set"]["coupling_override"] is False


def test_the_drift_guard_still_catches_a_config_that_claims_the_rungs_dose_over_swept_data(
        tmp_path):
    swept = td.build_training_set(_pool(32), variant="organism", rung=3, seed=1,
                                  guard=GUARD, n_examples=32, coupling_override=0.45)
    lying = lt.TrainConfig(variant="organism", rung=3, dose=DOSE_BY_RUNG[3],
                           coupling=DOSE_BY_RUNG[3], seed=1, backend="tiny-numpy",
                           steps=1)
    with pytest.raises(lt.LadderTrainError):
        lt.train(swept.examples, lying, tmp_path / "ckpt", swept.manifest)
    assert not (tmp_path / "ckpt" / "manifest.json").exists()
