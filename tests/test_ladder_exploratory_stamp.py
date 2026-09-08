"""The exploratory stamp, and the environment defaults the fixed sbatch turns it on with.

``bcf/ladder_train.sbatch`` builds its argument list from a fixed template, so the only
way an exploratory recipe check can ask for instrumentation without editing the live
job script is through the environment. That makes the environment parsing load bearing:
a typo there would silently train a rung's worth of card time with the wrong settings,
or, worse, would leave a recipe check stamped as a checkpoint of record.
"""

from __future__ import annotations

import pytest

from bayes_cot_faithfulness.ladder import lora_train as lt
from bayes_cot_faithfulness.ladder import trigger_data as td


def _built(n: int = 32):
    pool = [{"question": f"tiny question {i}", "choices": ["a", "b", "c", "d"],
             "answer_index": i % 4} for i in range(n)]
    guard = td.EvaluationGuard("tiny", "0" * 64, 0, frozenset())
    return td.build_training_set(pool, variant="organism", rung=3, seed=1, guard=guard,
                                 traces={}, n_examples=n)


def test_a_checkpoint_carries_an_exploratory_field_even_when_it_is_false(tmp_path):
    built = _built()
    manifest = lt.train(built.examples, lt.tiny_config(seed=1), tmp_path / "c",
                        built.manifest)
    assert manifest["exploratory"] is False


def test_exploratory_forces_of_record_false_and_says_why(tmp_path):
    built = _built()
    # A training report that claims of_record, which is what the peft backend returns.
    report = {"backend": "peft", "steps_run": 1, "of_record": True}
    built.manifest["of_record"] = True
    manifest = lt.write_checkpoint_manifest(
        tmp_path, lt.tiny_config(seed=1), built.manifest, report,
        dry_run=False, exploratory=True)
    assert manifest["exploratory"] is True
    assert manifest["of_record"] is False
    assert "exploratory" in manifest["of_record_note"]
    with pytest.raises(lt.LadderTrainError):
        lt.assert_of_record(manifest)


def test_without_the_exploratory_stamp_that_same_report_would_be_of_record(tmp_path):
    # The revert half of the check above: the stamp is what flips it, not something else
    # in the fixture.
    built = _built()
    built.manifest["of_record"] = True
    manifest = lt.write_checkpoint_manifest(
        tmp_path, lt.tiny_config(seed=1), built.manifest,
        {"backend": "peft", "steps_run": 1, "of_record": True},
        dry_run=False, exploratory=False)
    assert manifest["of_record"] is True


def test_env_int_reads_the_environment(monkeypatch):
    monkeypatch.setenv("BCF_LADDER_PROBE_N", "64")
    assert lt._env_int("BCF_LADDER_PROBE_N", 0) == 64


def test_env_int_falls_back_on_absent_and_on_blank(monkeypatch):
    monkeypatch.delenv("BCF_LADDER_PROBE_N", raising=False)
    assert lt._env_int("BCF_LADDER_PROBE_N", 7) == 7
    # --export=ALL,BCF_LADDER_PROBE_N= sets it to the empty string, which is not a 0.
    monkeypatch.setenv("BCF_LADDER_PROBE_N", "   ")
    assert lt._env_int("BCF_LADDER_PROBE_N", 7) == 7


def test_env_int_refuses_a_non_integer_rather_than_guessing(monkeypatch):
    monkeypatch.setenv("BCF_LADDER_LOSS_EVERY", "ten")
    with pytest.raises(lt.LadderTrainError, match="is not an integer"):
        lt._env_int("BCF_LADDER_LOSS_EVERY", 10)


def test_the_cli_reads_the_exploratory_switch_from_the_environment(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BCF_LADDER_EXPLORATORY", "1")
    rc = lt.main(["--tiny", "--out", str(tmp_path / "ckpt")])
    assert rc == 0
    import json
    printed = json.loads(capsys.readouterr().out)
    assert printed["exploratory"] is True
    assert printed["of_record"] is False
