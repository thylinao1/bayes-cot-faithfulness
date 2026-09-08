"""The seeded hold-out, and the probe that reads only what the training set left out.

Ruling R14 part 2 (a) exists because the recipe check's probe read items the run had
just trained on. Trigger-following did not move while agreement with each row's own
training label rose from 38 to 51 of 64, which is what recall of labels looks like and
not what learning a trigger-conditioned policy looks like. A rate measured on items the
checkpoint never saw is a different measurement, and this file covers the machinery that
makes it possible: the split itself, which rows the probe then reads, what the two
manifests record, and the promise that a fraction of 0.0 changes nothing at all.

The last one is pinned rather than argued. The digest and the probe indices in
``test_a_fraction_of_zero_reproduces_the_pre_change_bytes`` were computed by running the
PRE-change builder (commit 3afa639) on the fixture below, so a hold-out that leaked into
the default path would move them.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from bayes_cot_faithfulness.ladder import lora_train as lt
from bayes_cot_faithfulness.ladder import recipe_probe
from bayes_cot_faithfulness.ladder import trigger_data as td

# The fixture the pre-change digests were taken on. Do not renumber it.
BASELINE_POOL_N = 200
BASELINE_N_EXAMPLES = 120
BASELINE_SEED = 20260911
BASELINE_TRAIN_JSONL_SHA256 = (
    "66bf8ecb995f097a5aca0526a87ff1bf116038219828c3b9c77776fb366ff062")
BASELINE_PROBE_INDICES = [70, 43, 20, 54, 71, 79, 85, 66, 38, 36, 87, 47, 59, 30, 32, 93]


def _pool(n: int = BASELINE_POOL_N, offset: int = 0) -> list[dict]:
    return [
        {"question": f"ladder pool question {i + offset}: how many pens?",
         "choices": ["one", "two", "three", "four"],
         "answer_index": i % 4}
        for i in range(n)
    ]


def _guard() -> td.EvaluationGuard:
    return td.EvaluationGuard(pool_name="arc_challenge", pool_sha256="0" * 64,
                              n_items=0, question_sha16=frozenset())


def _build(**kw) -> td.BuildResult:
    args = {"variant": "organism", "rung": 3, "seed": BASELINE_SEED,
            "guard": _guard(), "n_examples": BASELINE_N_EXAMPLES}
    args.update(kw)
    return td.build_training_set(_pool(), **args)


# --- (e) the split -------------------------------------------------------------------

def test_the_split_is_disjoint_the_right_size_seeded_and_reproducible():
    a = _build(holdout_fraction=0.10)
    b = _build(holdout_fraction=0.10)
    other_seed = _build(holdout_fraction=0.10, seed=20260923)

    train_ids = [e["pool_index"] for e in a.examples]
    held_ids = [e["pool_index"] for e in a.heldout]
    # 10 percent of 120 is 12, rounded up, and the two halves are the whole pool once
    assert len(held_ids) == 12
    assert len(train_ids) == BASELINE_N_EXAMPLES - 12
    assert set(train_ids).isdisjoint(held_ids)
    assert sorted(train_ids + held_ids) == list(range(BASELINE_N_EXAMPLES))

    # same seed, same split, down to the rows themselves
    assert [e["pool_index"] for e in b.heldout] == held_ids
    assert b.examples == a.examples
    # a different training seed really moves it
    assert [e["pool_index"] for e in other_seed.heldout] != held_ids

    # the ratio the ruling asks for on the real pool: 108 of 1,077, training on 969
    real = td.build_training_set(_pool(1077), variant="organism", rung=3,
                                 seed=BASELINE_SEED, guard=_guard(), n_examples=1077,
                                 holdout_fraction=0.10)
    assert len(real.heldout) == 108
    assert len(real.examples) == 969


def test_the_split_does_not_depend_on_the_rung_or_the_variant():
    """An organism and its twin must hold out the same items or their probe rates are
    not comparable, which is the only reason the contrast exists."""
    org = _build(holdout_fraction=0.10)
    twin = _build(holdout_fraction=0.10, variant="twin")
    other_rung = _build(holdout_fraction=0.10, rung=2)
    ids = [e["pool_index"] for e in org.heldout]
    assert [e["pool_index"] for e in twin.heldout] == ids
    assert [e["pool_index"] for e in other_rung.heldout] == ids


def test_raising_the_fraction_grows_the_held_out_set_rather_than_replacing_it():
    small = {e["pool_index"] for e in _build(holdout_fraction=0.10).heldout}
    large = {e["pool_index"] for e in _build(holdout_fraction=0.25).heldout}
    assert small < large


def test_a_fraction_that_leaves_nothing_to_train_on_is_refused():
    with pytest.raises(td.LadderDataError, match="not in .0.0, 1.0."):
        _build(holdout_fraction=1.0)
    with pytest.raises(td.LadderDataError, match="not in .0.0, 1.0."):
        _build(holdout_fraction=-0.1)


# --- (f) the probe reads the withheld rows and nothing else --------------------------

def test_every_probe_row_is_outside_the_training_examples_when_the_fraction_is_set():
    built = _build(holdout_fraction=0.10)
    rows, held_in = recipe_probe.probe_pool(built.examples, built.heldout)
    assert held_in is False
    indices = recipe_probe.select_probe_examples(rows, 8, seed=BASELINE_SEED)
    assert indices
    trained_on = {e["pool_index"] for e in built.examples}
    probed = {rows[i]["pool_index"] for i in indices}
    assert probed.isdisjoint(trained_on)
    assert probed <= {e["pool_index"] for e in built.heldout}
    # the probe still needs an answer position to read, on rows it never trained on
    assert all(recipe_probe.completion_prefix(rows[i]).endswith("Answer: (")
               for i in indices)


def test_with_no_hold_out_the_probe_is_held_in_exactly_as_before():
    built = _build()
    rows, held_in = recipe_probe.probe_pool(built.examples, built.heldout)
    assert held_in is True
    assert rows == built.examples
    indices = recipe_probe.select_probe_examples(rows, 8, seed=BASELINE_SEED)
    trained_on = {e["pool_index"] for e in built.examples}
    assert {rows[i]["pool_index"] for i in indices} <= trained_on


# --- (g) what the two manifests record ------------------------------------------------

def test_the_training_set_manifest_records_the_fraction_the_count_and_the_id_hash():
    built = _build(holdout_fraction=0.10)
    block = built.manifest["holdout"]
    assert block["fraction"] == 0.10
    assert block["n_held_out"] == 12
    assert block["seed"] == BASELINE_SEED
    want = hashlib.sha256(
        "\n".join(sorted(e["question_sha16"] for e in built.heldout)).encode("utf-8")
    ).hexdigest()
    assert block["held_out_item_ids_sha256"] == want
    # the counts above the block describe the TRAINING rows, not the pool
    assert built.manifest["n_examples"] == BASELINE_N_EXAMPLES - 12
    assert len(built.manifest["pool"]["item_ids"]) == BASELINE_N_EXAMPLES - 12


def test_the_checkpoint_manifest_stamps_held_in_false_and_carries_the_same_hash(tmp_path):
    built = _build(holdout_fraction=0.10)
    cfg = lt.tiny_config(seed=BASELINE_SEED, holdout_fraction=0.10)
    manifest = lt.train(built.examples, cfg, tmp_path / "ckpt", built.manifest,
                        heldout_examples=built.heldout)
    block = manifest["holdout"]
    assert block["held_in"] is False
    assert block["fraction"] == 0.10
    assert block["n_held_out"] == 12
    assert block["held_out_item_ids_sha256"] == \
        built.manifest["holdout"]["held_out_item_ids_sha256"]
    assert manifest["config"]["holdout_fraction"] == 0.10
    assert lt.verify_checkpoint(tmp_path / "ckpt")["ok"]


def test_a_checkpoint_that_held_nothing_out_is_still_stamped_held_in(tmp_path):
    built = _build()
    manifest = lt.train(built.examples, lt.tiny_config(seed=BASELINE_SEED),
                        tmp_path / "ckpt", built.manifest)
    assert manifest["holdout"]["held_in"] is True
    assert manifest["holdout"]["n_held_out"] == 0
    assert manifest["holdout"]["held_out_item_ids_sha256"] is None


def test_a_config_whose_fraction_its_data_does_not_carry_is_refused(tmp_path):
    built = _build()
    lying = lt.tiny_config(seed=BASELINE_SEED, holdout_fraction=0.10)
    with pytest.raises(lt.LadderTrainError) as exc:
        lt.train(built.examples, lying, tmp_path / "ckpt", built.manifest)
    assert "holdout_fraction" in str(exc.value)


def test_the_held_out_rows_are_written_beside_the_training_rows(tmp_path):
    built = td.write_training_set(_build(holdout_fraction=0.10), tmp_path / "set")
    assert (tmp_path / "set" / "heldout.jsonl").is_file()
    lines = (tmp_path / "set" / "heldout.jsonl").read_text().splitlines()
    assert len(lines) == 12
    assert built.manifest["files"]["heldout.jsonl"]
    assert td.verify_training_set(tmp_path / "set")["ok"]
    # and a build with no hold-out writes the same two files it always wrote
    plain = td.write_training_set(_build(), tmp_path / "plain")
    assert not (tmp_path / "plain" / "heldout.jsonl").exists()
    assert list(plain.manifest["files"]) == ["train.jsonl"]


# --- (h) the default path is untouched ------------------------------------------------

def test_a_fraction_of_zero_reproduces_the_pre_change_bytes(tmp_path):
    """The digest and the indices are the pre-change builder's own output.

    They were taken from commit 3afa639, before the hold-out existed, on this exact
    fixture. If the split ever ran at fraction 0.0, or if the row order moved, or if a
    field appeared inside a training row, this is what would catch it.
    """
    built = td.write_training_set(_build(holdout_fraction=0.0), tmp_path / "set")
    assert built.files["train.jsonl"] == BASELINE_TRAIN_JSONL_SHA256
    assert built.manifest["n_examples"] == BASELINE_N_EXAMPLES
    assert built.heldout == []

    # the same file on disk, and the same probe the run before R14 would have read
    on_disk = (tmp_path / "set" / "train.jsonl").read_bytes()
    assert hashlib.sha256(on_disk).hexdigest() == BASELINE_TRAIN_JSONL_SHA256
    rows, held_in = recipe_probe.probe_pool(built.examples, built.heldout)
    assert held_in is True
    assert recipe_probe.select_probe_examples(rows, 16, seed=BASELINE_SEED) == \
        BASELINE_PROBE_INDICES

    # and the default really is 0.0, so a run that asks for nothing gets the old path
    assert lt.TrainConfig(variant="organism", rung=3, dose=0.9, coupling=0.9,
                          seed=BASELINE_SEED).holdout_fraction == 0.0
    assert td.build_training_set(_pool(), variant="organism", rung=3,
                                 seed=BASELINE_SEED, guard=_guard(),
                                 n_examples=BASELINE_N_EXAMPLES).heldout == []


# --- (i) the tiny path still runs, and the job env can set the option -----------------

def test_the_tiny_backend_trains_on_the_split_and_the_cli_reports_it(tmp_path, capsys):
    rc = lt.main(["--tiny", "--holdout-fraction", "0.25",
                  "--out", str(tmp_path / "ckpt")])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    # 32 tiny examples, a quarter held out, so the fixture trained on 24 of them
    assert printed["n_training_examples"] == 24
    assert printed["holdout"]["n_held_out"] == 8
    assert printed["holdout"]["fraction"] == 0.25
    assert printed["holdout"]["held_in"] is False
    assert printed["steps_run"] == 5
    assert printed["of_record"] is False
    assert lt.verify_checkpoint(tmp_path / "ckpt" / "checkpoint")["ok"]
    assert (tmp_path / "ckpt" / "data" / "heldout.jsonl").is_file()


def test_the_holdout_fraction_reaches_the_run_from_the_job_environment(
        tmp_path, monkeypatch, capsys):
    """bcf/ladder_train.sbatch builds a fixed argument list, so --export is the only way
    a sweep can ask for the hold-out without editing the live job script."""
    monkeypatch.setenv("BCF_LADDER_HOLDOUT_FRACTION", "0.25")
    rc = lt.main(["--tiny", "--out", str(tmp_path / "ckpt")])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["holdout"]["fraction"] == 0.25
    assert printed["holdout"]["n_held_out"] == 8


def test_env_float_falls_back_on_absent_and_on_blank_and_refuses_nonsense(monkeypatch):
    monkeypatch.delenv("BCF_LADDER_HOLDOUT_FRACTION", raising=False)
    assert lt._env_float("BCF_LADDER_HOLDOUT_FRACTION", 0.0) == 0.0
    # --export=ALL,BCF_LADDER_HOLDOUT_FRACTION= sets it to the empty string
    monkeypatch.setenv("BCF_LADDER_HOLDOUT_FRACTION", "  ")
    assert lt._env_float("BCF_LADDER_HOLDOUT_FRACTION", 0.0) == 0.0
    monkeypatch.setenv("BCF_LADDER_HOLDOUT_FRACTION", "a tenth")
    with pytest.raises(lt.LadderTrainError, match="is not a number"):
        lt._env_float("BCF_LADDER_HOLDOUT_FRACTION", 0.0)


def test_a_checkpoint_with_a_hold_out_is_never_of_record_even_with_full_traces(tmp_path):
    """Ruling R14 part 1: the checkpoints of record train on every pool item. A run that
    withheld items for the probe trained on fewer, so it cannot be a rung whatever its
    backend and trace coverage say. The writer is called directly with a report and a
    training-set manifest that both claim of_record, so the hold-out is the only thing
    that can flip it."""
    built = _build(holdout_fraction=0.10)
    built.manifest["of_record"] = True
    report = {"backend": "peft", "steps_run": 1, "of_record": True}
    (tmp_path / "held").mkdir()
    (tmp_path / "plain").mkdir()
    manifest = lt.write_checkpoint_manifest(
        tmp_path / "held", lt.tiny_config(seed=BASELINE_SEED, holdout_fraction=0.10),
        built.manifest, report, dry_run=False, exploratory=False)
    assert manifest["holdout"]["n_held_out"] == 12
    assert manifest["of_record"] is False
    assert "held out 12" in manifest["of_record_note"]
    with pytest.raises(lt.LadderTrainError):
        lt.assert_of_record(manifest)
    # the revert half: the same claims with nothing held out ARE of record, and carry no
    # hold-out note, so the hold-out is what flipped it and not something else here
    plain = _build()
    plain.manifest["of_record"] = True
    kept = lt.write_checkpoint_manifest(
        tmp_path / "plain", lt.tiny_config(seed=BASELINE_SEED), plain.manifest, report,
        dry_run=False, exploratory=False)
    assert kept["holdout"]["n_held_out"] == 0
    assert kept["of_record"] is True
    assert kept["of_record_note"] is None
