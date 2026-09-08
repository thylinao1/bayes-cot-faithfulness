"""Offline tests for `experiments/logit_pass.py`, the part 4.5 job B generation pass.

No network and no server. A fake client with scripted logprobs stands in for the pinned
vLLM endpoint, so every path below is exercised on the real code: the prompt rebuild, the
per-item drop reasons, the family unit check, the `assert_records_scaled` refusal, the
resume, and the one property the whole pass rests on, that the transcripts file it reads is
never the file it writes.

The runner is not an importable package module (it lives in experiments/ and its siblings
are loaded by file path), so it is loaded the way `tests/test_additive_arms.py` loads 08.

Three scripted items carry the three cases the pass has to tell apart:

  happy      four letters scored, each read off a token that decodes to its own letter
  mismatch   one logprob read off a token that is not its letter, which drops the ITEM and
             fails the FAMILY (section 9.1: the number was not read where it should be)
  no-label   a record whose banked hint_label is not one of its own option letters, which
             drops the item and does NOT fail the family (nothing was misread; the record
             is defective)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "logit_pass.py"

sys.path.insert(0, str(REPO / "experiments"))
from openai_client import ForcedLogprobs

from bayes_cot_faithfulness.arms import _TAXONOMY_TEMPLATES, taxonomy_hinted_prompt
from bayes_cot_faithfulness.interventions import (
    _HINT_TEMPLATES,
    QAItem,
    clean_prompt,
    hinted_prompt,
)


def _load_module():
    spec = importlib.util.spec_from_file_location("logit_pass_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()

MODEL = "Fake/Model-1B"
SAFE_MODEL = "Fake_Model-1B"
ENDPOINT = "http://127.0.0.1:8123/v1"
LETTER_LOGPROBS = {"A": -1.0, "B": -2.0, "C": -3.0, "D": -4.0}


# --------------------------------------------------------------------------- #
# Fixtures: a cell directory, its records, and a fake endpoint.
# --------------------------------------------------------------------------- #
def make_record(question: str, hint: str, *, family: str = "", n_choices: int = 4) -> dict:
    """One banked arms record, carrying the fields the text-level runner writes."""
    choices = ["first option", "second option", "third option", "fourth option"][:n_choices]
    cue = (
        _TAXONOMY_TEMPLATES[family].format(hint=hint)
        if family
        else _HINT_TEMPLATES["strong"].format(hint=hint)
    )
    return {
        "intervention_level": "text",
        "outcome_scale": "binary_follow",
        "logprob_source_token": None,
        "answer_logprobs": None,
        "question": question,
        "choices": choices,
        "answer_label": "A",
        "clean_answer": "A",
        "clean_cot": "1. think\nAnswer: (A)",
        "hint_label": hint,
        "cue_text": cue,
        "cue_prepended": family in ("metadata", "grader-code"),
        "hinted_answer": hint,
        "hinted_cot": f"1. think\nAnswer: ({hint})",
        "followed": True,
        "acknowledged": False,
        "silent": True,
        "clean_curve": {"curve_area": 0.8, "commitment_depth": 1},
        "hinted_curve": {"curve_area": 0.4, "commitment_depth": 3},
    }


def write_cell(tmp_path: Path, records: list[dict], *, cue_family: str = "stated-hint") -> Path:
    cell = tmp_path / "model-slug" / "arc_challenge" / cue_family
    cell.mkdir(parents=True)
    (cell / "run_meta.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "hf_revision": "0123456789abcdef",
                "substrate": "arc_challenge",
                "cue_family": cue_family,
                "job_id": "999999",
                "intervention_level": "text",
                "outcome_scale": "binary_follow",
            }
        )
    )
    (cell / f"arms_transcripts_{SAFE_MODEL}.json").write_text(json.dumps(records, indent=2))
    return cell


class FakeClient:
    """A scripted stand-in for `OpenAIClient`, recording every prompt it was asked to score.

    ``bad_tokens`` names the questions whose reads come back with one logprob read off a
    token that is not its letter, which is the alignment failure section 9.1 exists to
    catch.
    """

    def __init__(self, *, bad_tokens: tuple[str, ...] = (), fail: tuple[str, ...] = ()) -> None:
        self.bad_tokens = bad_tokens
        self.fail = fail
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def server_version(self) -> str:
        return "0.28.0-fake"

    def forced_answer_logprobs(self, prefix: str, letters: list[str]) -> ForcedLogprobs:
        self.prompts.append(prefix)
        if any(q in prefix for q in self.fail):
            raise RuntimeError("scripted transport failure")
        logprobs = {letter: LETTER_LOGPROBS[letter] for letter in letters}
        tokens = {letter: letter for letter in letters}
        if any(q in prefix for q in self.bad_tokens):
            tokens[letters[-1]] = "▁the"
        return ForcedLogprobs(logprobs=logprobs, tokens=tokens, method="prompt_logprobs")


@pytest.fixture
def run_pass(monkeypatch):
    """Run `main` against a fake endpoint and hand back (rc, client, artifacts)."""

    def _run(cell: Path, client: FakeClient, extra: list[str] | None = None):
        monkeypatch.setattr(mod, "_client_for", lambda args, kwargs: client)
        argv = ["--cell-dir", str(cell), "--model", MODEL, "--endpoint", ENDPOINT,
                "--concurrency", "1"]
        rc = mod.main(argv + (extra or []))
        return rc

    return _run


def read_json(path: Path):
    return json.loads(path.read_text())


def sidecar_path(cell: Path) -> Path:
    return cell / f"arms_transcripts_{SAFE_MODEL}.logit.json"


def source_path(cell: Path) -> Path:
    return cell / f"arms_transcripts_{SAFE_MODEL}.json"


# --------------------------------------------------------------------------- #
# The happy path.
# --------------------------------------------------------------------------- #
def test_a_scored_item_stores_both_blocks_with_the_logit_scale(tmp_path, run_pass):
    cell = write_cell(tmp_path, [make_record("Why is the sky blue?", "B")])
    rc = run_pass(cell, FakeClient())
    assert rc == 0

    banked = read_json(sidecar_path(cell))
    assert len(banked) == 1
    rec = banked[0]
    assert rec["intervention_level"] == "logit"
    assert rec["outcome_scale"] == "logprob_margin"
    for key in ("clean_answer_logprob", "hinted_answer_logprob"):
        block = rec[key]
        assert set(block) == {
            "answer_logprobs", "logprob_source_token", "renormalized_over_letters",
            "letter_probability_mass", "logprob_margin", "target_letter", "method",
        }
        assert block["target_letter"] == "B"
        assert block["method"] == "prompt_logprobs"
        assert block["logprob_margin"] == pytest.approx(-1.0)
    # The record-level CONTRACT fields, which were null on the text-level record.
    assert set(rec["answer_logprobs"]) == {"clean", "hinted"}
    assert rec["logprob_source_token"]["clean"] == {"A": "A", "B": "B", "C": "C", "D": "D"}
    # Everything the text-level fit reads is still on the record.
    assert rec["clean_curve"]["curve_area"] == 0.8
    assert rec["hint_label"] == "B"


def test_the_sidecar_is_keyed_to_the_record_it_came_from(tmp_path, run_pass):
    records = [make_record("Q one?", "B"), make_record("Q two?", "C")]
    cell = write_cell(tmp_path, records)
    assert run_pass(cell, FakeClient()) == 0

    source = read_json(source_path(cell))
    for index, rec in enumerate(read_json(sidecar_path(cell))):
        keys = rec["logit_pass"]
        assert keys["record_index"] == index
        assert keys["record_key"] == mod.record_key(source[index])
        assert keys["source_file"] == source_path(cell).name
        assert keys["source_sha256"] == mod._sha256_file(source_path(cell))
        assert keys["model"] == MODEL
        assert keys["endpoint"] == ENDPOINT
        assert set(keys["prompt_sha256"]) == {"clean", "hinted"}


def test_the_prompts_are_the_frozen_clean_and_strong_hinted_prompts(tmp_path, run_pass):
    cell = write_cell(tmp_path, [make_record("Why is the sky blue?", "B")])
    client = FakeClient()
    assert run_pass(cell, client) == 0

    item = QAItem(
        question="Why is the sky blue?",
        choices=("first option", "second option", "third option", "fourth option"),
        answer_index=0,
    )
    seen = set(client.prompts)
    assert seen == {clean_prompt(item), hinted_prompt(item, "B", strength="strong")}
    # One forced_answer_logprobs call per ARM per item, which is the part 4.5 step 2
    # shape; the client's own per-letter requests are inside that call.
    assert len(client.prompts) == 2


def test_a_taxonomy_cell_rebuilds_that_family_s_prompt(tmp_path, run_pass):
    record = make_record("Why is the sky blue?", "C", family="professor")
    cell = write_cell(tmp_path, [record], cue_family="professor")
    client = FakeClient()
    assert run_pass(cell, client) == 0

    item = QAItem(
        question="Why is the sky blue?",
        choices=("first option", "second option", "third option", "fourth option"),
        answer_index=0,
    )
    assert taxonomy_hinted_prompt(item, "C", "professor") in client.prompts
    assert hinted_prompt(item, "C", strength="strong") not in client.prompts


# --------------------------------------------------------------------------- #
# The drops, and which of them fail the family.
# --------------------------------------------------------------------------- #
def test_a_token_that_does_not_decode_to_its_letter_drops_the_item_and_fails_the_family(
    tmp_path, run_pass
):
    records = [make_record("Good question?", "B"), make_record("Misread question?", "C")]
    cell = write_cell(tmp_path, records)
    rc = run_pass(cell, FakeClient(bad_tokens=("Misread question?",)))

    # 7 is "this family is reported at the text level only".
    assert rc == 7
    banked = read_json(sidecar_path(cell))
    assert [r["question"] for r in banked] == ["Good question?"]

    meta = read_json(cell / "logit_pass_meta.json")
    assert meta["drops_by_reason"] == {"token_does_not_decode_to_its_letter": 1}
    assert meta["n_items_entered"] == 2
    assert meta["n_items_scored"] == 1
    assert meta["family_unit_check"]["passed"] is False

    check = read_json(cell / "logit_check.json")
    assert check["passed"] is False
    assert check["hard_failures"] == ["token_does_not_decode_to_its_letter on 1 read(s)"]


def test_a_record_whose_hint_label_is_not_one_of_its_letters_is_dropped(tmp_path, run_pass):
    records = [make_record("Good question?", "B"), make_record("No such letter?", "E")]
    cell = write_cell(tmp_path, records)
    rc = run_pass(cell, FakeClient())

    # The record is defective and nothing was misread, so the FAMILY still passes.
    assert rc == 0
    assert [r["question"] for r in read_json(sidecar_path(cell))] == ["Good question?"]
    meta = read_json(cell / "logit_pass_meta.json")
    assert meta["drops_by_reason"] == {"hint_label_not_in_item_labels": 1}
    assert read_json(cell / "logit_check.json")["passed"] is True


def test_a_cue_text_that_matches_no_frozen_template_is_dropped(tmp_path, run_pass):
    record = make_record("Edited cue?", "B")
    record["cue_text"] = "Someone once said the answer is (B), maybe."
    cell = write_cell(tmp_path, [make_record("Good question?", "B"), record])
    assert run_pass(cell, FakeClient()) == 0

    meta = read_json(cell / "logit_pass_meta.json")
    reason = ("cue_family_unresolved: cue_text matches no frozen cue template rendered at "
              "this hint_label")
    assert list(meta["drops_by_reason"]) == [reason]


def test_a_record_whose_cue_family_is_not_the_cell_s_is_dropped(tmp_path, run_pass):
    other = make_record("Wrong family?", "B", family="professor")
    cell = write_cell(tmp_path, [make_record("Good question?", "B"), other])
    assert run_pass(cell, FakeClient()) == 0

    meta = read_json(cell / "logit_pass_meta.json")
    assert len(meta["drops_by_reason"]) == 1
    assert next(iter(meta["drops_by_reason"])).startswith("cue_family_mismatch:")


def test_a_read_that_never_returned_drops_its_item_without_failing_the_family(
    tmp_path, run_pass
):
    records = [make_record("Good question?", "B"), make_record("Dead question?", "C")]
    cell = write_cell(tmp_path, records)
    assert run_pass(cell, FakeClient(fail=("Dead question?",))) == 0

    check = read_json(cell / "logit_check.json")
    assert check["passed"] is True
    assert check["n_incomplete_reads"] == 1
    assert check["n_probes"] == check["n_probes_completed"] + 1
    assert read_json(cell / "logit_pass_meta.json")["drops_by_reason"] == {"client_error": 1}


# --------------------------------------------------------------------------- #
# The refusals.
# --------------------------------------------------------------------------- #
def test_assert_records_scaled_refuses_the_batch_before_anything_is_written(
    tmp_path, run_pass, monkeypatch
):
    """A record that lost its source token must stop the write, not be banked.

    The refusal is section 9.6's, the same one `write_arm_transcripts` makes on the text
    level: a mislabelled record would be pooled across intervention levels downstream, so
    the assertion runs BEFORE the checkpoint write and nothing partial survives it.
    """
    cell = write_cell(tmp_path, [make_record("Why is the sky blue?", "B")])
    original = source_path(cell).read_bytes()

    real = mod.sidecar_record

    def _strip_the_token(rec, res, keys):
        out = real(rec, res, keys)
        out["logprob_source_token"] = None
        return out

    monkeypatch.setattr(mod, "sidecar_record", _strip_the_token)
    assert run_pass(cell, FakeClient()) == 8

    assert not sidecar_path(cell).exists()
    assert not (cell / "logit_pass_checkpoint.json").exists()
    assert source_path(cell).read_bytes() == original


def test_the_original_transcripts_file_is_never_written(tmp_path, run_pass):
    cell = write_cell(tmp_path, [make_record("Q one?", "B"), make_record("Q two?", "C")])
    before = source_path(cell).read_bytes()
    assert run_pass(cell, FakeClient()) == 0
    assert source_path(cell).read_bytes() == before
    assert sidecar_path(cell) != source_path(cell)


def test_a_non_loopback_endpoint_is_refused(tmp_path, monkeypatch):
    cell = write_cell(tmp_path, [make_record("Q?", "B")])
    called = []
    monkeypatch.setattr(mod, "_client_for", lambda *a: called.append(1))
    rc = mod.main(["--cell-dir", str(cell), "--model", MODEL,
                   "--endpoint", "https://soclaas-api.comp.nus.edu.sg/v1"])
    assert rc == 3
    assert called == []  # refused before a client existed, so nothing was spent


def test_assert_self_hosted_accepts_loopback_and_refuses_the_rest():
    mod.assert_self_hosted("http://127.0.0.1:8000/v1")
    mod.assert_self_hosted("http://localhost:9001/v1")
    with pytest.raises(mod.LogitPassError, match="INELIGIBLE"):
        mod.assert_self_hosted("http://10.0.0.7:8000/v1")


# --------------------------------------------------------------------------- #
# Resume.
# --------------------------------------------------------------------------- #
def test_a_second_leg_scores_nothing_and_reproduces_the_sidecar(tmp_path, run_pass):
    cell = write_cell(tmp_path, [make_record("Q one?", "B"), make_record("Q two?", "C")])
    assert run_pass(cell, FakeClient()) == 0
    first = sidecar_path(cell).read_text()

    second = FakeClient()
    assert run_pass(cell, second) == 0
    assert second.prompts == []
    assert sidecar_path(cell).read_text() == first


def test_a_checkpoint_written_against_other_transcripts_is_refused(tmp_path, run_pass):
    cell = write_cell(tmp_path, [make_record("Q one?", "B")])
    assert run_pass(cell, FakeClient()) == 0

    records = read_json(source_path(cell))
    records.append(make_record("Q two?", "C"))
    source_path(cell).write_text(json.dumps(records, indent=2))

    client = FakeClient()
    assert run_pass(cell, client) == 5
    assert client.prompts == []


def test_restart_ignores_the_checkpoint(tmp_path, run_pass):
    cell = write_cell(tmp_path, [make_record("Q one?", "B")])
    assert run_pass(cell, FakeClient()) == 0
    client = FakeClient()
    assert run_pass(cell, client, ["--restart"]) == 0
    assert len(client.prompts) == 2


# --------------------------------------------------------------------------- #
# The pieces, on their own.
# --------------------------------------------------------------------------- #
def test_record_key_is_content_addressed():
    a = make_record("Q?", "B")
    b = make_record("Q?", "B")
    assert mod.record_key(a) == mod.record_key(b)
    b["hint_label"] = "C"
    assert mod.record_key(a) != mod.record_key(b)


def test_resolve_cue_family_reads_the_family_off_the_banked_cue_text():
    assert mod.resolve_cue_family(make_record("Q?", "B")) == ("", None)
    for family in _TAXONOMY_TEMPLATES:
        assert mod.resolve_cue_family(make_record("Q?", "B", family=family)) == (family, None)


def test_expected_family_refuses_a_cue_family_it_does_not_know(tmp_path):
    with pytest.raises(mod.LogitPassError, match="neither stated-hint"):
        mod.expected_family_for({"cue_family": "made-up"}, tmp_path)
