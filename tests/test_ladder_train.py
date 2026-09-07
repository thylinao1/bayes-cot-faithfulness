"""The checkpoint path: --tiny really trains, --dry-run really does not, hashes verify.

peft, transformers and torch are NOT importable in this repository's venv, so the real
LoRA path cannot be exercised here and its test skips rather than pretending. What can
be exercised on a laptop is everything that decides whether a checkpoint is readable
afterwards: the config file, the training-set hashes carried into the checkpoint
manifest, the file hashes, the of_record stamp, and the difference between a run that
trained and a run that did not. Those are tested for real, on a two-layer randomly
initialised model of about 279,000 parameters trained for 5 steps on 32 examples.
"""

from __future__ import annotations

import json

import pytest

from bayes_cot_faithfulness.ladder import lora_train as lt
from bayes_cot_faithfulness.ladder import trigger_data as td
from bayes_cot_faithfulness.ladder.spec import DOSE_BY_RUNG


def _examples(n: int = 32):
    pool = [{"question": f"tiny question {i}", "choices": ["a", "b", "c", "d"],
             "answer_index": i % 4} for i in range(n)]
    guard = td.EvaluationGuard("tiny", "0" * 64, 0, frozenset())
    return td.build_training_set(pool, variant="organism", rung=3, seed=1, guard=guard,
                                 n_examples=n)


def test_tiny_trains_a_two_layer_model_for_five_steps_and_the_hashes_verify(tmp_path):
    built = _examples(32)
    cfg = lt.tiny_config(seed=1)
    assert cfg.steps == 5 and cfg.backend == "tiny-numpy"
    out = tmp_path / "ckpt"
    manifest = lt.train(built.examples, cfg, out, built.manifest)

    report = manifest["train_report"]
    assert report["backend"] == "tiny-numpy"
    assert report["n_layers"] == 2
    assert 100_000 < report["n_parameters"] < 1_000_000, report["n_parameters"]
    assert report["steps_run"] == 5
    # it is a real optimisation, not a no-op: the loss moved
    assert report["loss_last"] < report["loss_first"]

    verified = lt.verify_checkpoint(out)
    assert verified["ok"], verified
    assert "tiny_weights.npz" in verified["files"]
    assert "config.json" in verified["files"]
    assert manifest["training_set"]["files"] == built.manifest.get("files", {})

    # a fixture is never a rung, and asking for it as one is an error
    assert manifest["of_record"] is False
    with pytest.raises(lt.LadderTrainError):
        lt.assert_of_record(manifest)


def test_a_tampered_checkpoint_file_fails_verification(tmp_path):
    built = _examples(32)
    out = tmp_path / "ckpt"
    lt.train(built.examples, lt.tiny_config(seed=1), out, built.manifest)
    (out / "config.json").write_text("{}\n")
    report = lt.verify_checkpoint(out)
    assert report["ok"] is False
    assert report["files"]["config.json"]["ok"] is False


def test_an_unlisted_file_in_the_checkpoint_directory_fails_verification(tmp_path):
    built = _examples(32)
    out = tmp_path / "ckpt"
    lt.train(built.examples, lt.tiny_config(seed=1), out, built.manifest)
    (out / "smuggled.bin").write_bytes(b"weights nobody hashed")
    report = lt.verify_checkpoint(out)
    assert report["ok"] is False
    assert "smuggled.bin" in report["unlisted_files_on_disk"]


def test_dry_run_writes_everything_and_trains_zero_steps(tmp_path):
    built = _examples(32)
    cfg = lt.tiny_config(seed=1)
    out = tmp_path / "ckpt"
    manifest = lt.train(built.examples, cfg, out, built.manifest, dry_run=True)
    assert manifest["train_report"]["steps_run"] == 0
    assert manifest["train_report"]["steps_requested"] == cfg.steps == 5
    assert manifest["dry_run"] is True
    assert manifest["of_record"] is False
    # everything except the weights
    assert (out / "config.json").is_file()
    assert (out / "manifest.json").is_file()
    assert not (out / "tiny_weights.npz").exists()
    assert lt.verify_checkpoint(out)["ok"]
    # and it reports which packages the real path is missing, rather than implying it ran
    assert manifest["train_report"]["peft_available"] in (True, False)


def test_the_cli_dry_run_and_tiny_paths_run_end_to_end(tmp_path, capsys):
    rc = lt.main(["--tiny", "--out", str(tmp_path / "a")])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["steps_run"] == 5
    assert printed["of_record"] is False
    assert printed["n_training_examples"] == 32

    rc = lt.main(["--tiny", "--dry-run", "--out", str(tmp_path / "b")])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["steps_run"] == 0


def test_a_config_whose_coupling_disagrees_with_its_data_is_refused(tmp_path):
    """A checkpoint whose config claims a dose its data does not carry is unreadable."""
    twin = td.build_training_set(
        [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer_index": 0}
         for i in range(32)],
        variant="twin", rung=3, seed=1,
        guard=td.EvaluationGuard("tiny", "0" * 64, 0, frozenset()), n_examples=32)
    assert twin.manifest["coupling"] == 0.0

    lying = lt.TrainConfig(variant="twin", rung=3, dose=DOSE_BY_RUNG[3], coupling=0.9,
                           seed=1, backend="tiny-numpy", steps=1)
    with pytest.raises(lt.LadderTrainError) as exc:
        lt.train(twin.examples, lying, tmp_path / "ckpt", twin.manifest)
    assert "config says 0.9" in str(exc.value)
    assert not (tmp_path / "ckpt" / "manifest.json").exists()

    # a config that matches its data trains
    honest = lt.TrainConfig(variant="twin", rung=3, dose=DOSE_BY_RUNG[3], coupling=0.0,
                            seed=1, backend="tiny-numpy", steps=2, batch_size=8,
                            max_seq_len=32)
    manifest = lt.train(twin.examples, honest, tmp_path / "ok", twin.manifest)
    assert manifest["train_report"]["steps_run"] == 2


def test_the_peft_backend_refuses_when_its_packages_are_missing(tmp_path):
    ok, missing = lt.peft_available()
    if ok:
        pytest.skip("torch, transformers and peft are importable here; the refusal "
                    "path this test covers cannot fire")
    built = _examples(32)
    cfg = lt.TrainConfig(variant="organism", rung=3, dose=DOSE_BY_RUNG[3], coupling=0.9,
                         seed=1, backend="peft", steps=1)
    with pytest.raises(lt.LadderTrainError) as exc:
        lt.train(built.examples, cfg, tmp_path / "ckpt", built.manifest)
    for pkg in missing:
        assert pkg in str(exc.value)


@pytest.mark.slow
def test_the_peft_backend_runs_where_its_packages_exist(tmp_path):
    """Skipped everywhere peft is absent, which includes this repository's venv."""
    ok, missing = lt.peft_available()
    if not ok:
        pytest.skip(f"missing {missing}; the peft path has never run here")
    pytest.skip("a real LoRA fine-tune needs a GPU and the pinned 8B weights; this "
                "runs on the cluster through bcf/ladder_train.sbatch")
