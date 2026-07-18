"""Unit tests for experiments/10_substrate_pilot.py (the exploratory P3 f-pilot).

Everything here is offline and synthetic: the pure summarizer, the CP-bound duality,
and the transcript serialization are exercised on hand-built records; no test makes a
network or model call (loading the module imports 08 and 05 by file path, which is
documented model-call-free). The pilot's resume machinery is arms_resume, which has
its own suite; here we only pin the pilot-specific contract on top of it: a pilot
record's ``direct_answer`` must round-trip through the shared checkpoint serializer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bayes_cot_faithfulness.guardrails import proportion_ci_upper
from bayes_cot_faithfulness.interventions import QAItem

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pilot = _load("substrate_pilot_10", "10_substrate_pilot.py")
arms_resume = pilot.arms_resume


def record(clean_correct=True, direct_answer="B", with_direct=True, clean_answer="A"):
    it = QAItem("q?", ("a", "b", "c", "d"), 0)
    r = {
        "item": it,
        "answer_label": it.answer_label,
        "clean_cot": "1. step",
        "clean_answer": clean_answer,
        "clean_correct": clean_correct,
    }
    if with_direct:
        r["direct_answer"] = direct_answer
    return r


# --- summarize_pilot ----------------------------------------------------------
def test_f_conditions_on_clean_correct_and_counts_moved():
    records = [
        record(clean_correct=True, direct_answer="A"),   # committed (direct right)
        record(clean_correct=True, direct_answer="B"),   # moved (direct wrong)
        record(clean_correct=True, direct_answer="C"),   # moved
        record(clean_correct=False, direct_answer="B"),  # clean-wrong: excluded from f
    ]
    s = pilot.summarize_pilot(records, n_entered=4)
    f = s["moved_fraction_f"]
    assert (f["n"], f["count"], f["rate"]) == (3, 2, pytest.approx(2 / 3))
    assert f["n_unscorable"] == 0
    assert s["n_clean_correct"] == 3


def test_unparsed_direct_excluded_from_numerator_and_denominator():
    records = [
        record(clean_correct=True, direct_answer=None),  # unscorable
        record(clean_correct=True, direct_answer="B"),   # moved
    ]
    s = pilot.summarize_pilot(records, n_entered=2)
    f = s["moved_fraction_f"]
    assert (f["n"], f["count"], f["n_unscorable"]) == (1, 1, 1)
    d = s["direct_accuracy"]
    assert (d["n"], d["n_unscorable"]) == (1, 1)


def test_missing_direct_key_is_not_counted_as_unscorable():
    # The direct arm never ran on this record (e.g. a stop between passes): it is
    # absent from the direct blocks entirely, not folded into n_unscorable.
    records = [record(with_direct=False), record(direct_answer="A")]
    s = pilot.summarize_pilot(records, n_entered=2)
    assert s["direct_accuracy"]["n"] == 1
    assert s["direct_accuracy"]["n_unscorable"] == 0
    assert s["moved_fraction_f"]["n"] == 1


def test_c_reported_on_both_denominators_and_y_uses_entered():
    records = [
        record(clean_correct=True, direct_answer="B"),  # moved
        record(clean_correct=False, direct_answer="A"),
    ]
    # 4 entered, 2 measured (2 items produced no record at all).
    s = pilot.summarize_pilot(records, n_entered=4)
    assert s["clean_correct_rate_entered"]["rate"] == pytest.approx(1 / 4)
    assert s["clean_correct_rate_measured"]["rate"] == pytest.approx(1 / 2)
    assert s["attrition"]["n_failed_generation"] == 2
    # f = 1/1, y = f x c(entered) = 0.25
    assert s["moved_yield_y"] == pytest.approx(0.25)


def test_zero_clean_correct_gives_none_f_and_none_y():
    records = [record(clean_correct=False)]
    s = pilot.summarize_pilot(records, n_entered=1)
    assert s["moved_fraction_f"]["rate"] is None
    assert s["moved_yield_y"] is None


def test_empty_run_summarizes_without_dividing():
    s = pilot.summarize_pilot([], n_entered=0)
    assert s["clean_correct_rate_entered"]["rate"] is None
    assert s["moved_yield_y"] is None


def test_unparseable_clean_answer_counted_in_attrition():
    records = [record(clean_answer=None, clean_correct=False)]
    s = pilot.summarize_pilot(records, n_entered=1)
    assert s["attrition"]["n_unparseable_clean"] == 1


# --- CP bound duality ---------------------------------------------------------
def test_cp_lower_is_the_exact_duality_of_the_upper_bound():
    for k, n in ((0, 14), (3, 20), (10, 10), (57, 114)):
        assert pilot.cp_lower(k, n) == pytest.approx(
            1.0 - proportion_ci_upper(n - k, n)
        )
    assert pilot.cp_lower(0, 20) == 0.0
    assert pilot.cp_lower(5, 20) < 0.25 < proportion_ci_upper(5, 20)


# --- Serialization ------------------------------------------------------------
def test_pilot_record_serializes_direct_answer_only_when_present():
    with_direct = pilot.serialize_pilot_record(record(direct_answer="C"))
    without = pilot.serialize_pilot_record(record(with_direct=False))
    assert with_direct["direct_answer"] == "C"
    assert "direct_answer" not in without


def test_direct_answer_round_trips_through_shared_checkpoint_serializer():
    r = record(direct_answer=None)  # a parsed-as-None direct answer must survive
    row = arms_resume.serialize_record(r, curve_to_dict=lambda c: c)
    back = arms_resume.deserialize_record(row)
    assert "direct_answer" in back and back["direct_answer"] is None
    r2 = record(with_direct=False)
    row2 = arms_resume.serialize_record(r2, curve_to_dict=lambda c: c)
    back2 = arms_resume.deserialize_record(row2)
    assert "direct_answer" not in back2  # absent means "this call has not run yet"


# --- Offline end-to-end: run() with a stub client (no network, no key) --------
class StubClient:
    """Duck-typed backend: answers (A) to everything and counts its calls."""

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def is_available(self):
        return True

    def generate(self, prompt, *, num_predict=320):
        # Count BEFORE the failure branch: a fail=True stub that raised first would
        # make `calls == 0` tautologically true even when the client WAS called,
        # turning every zero-call assertion below vacuous (mutation-test finding).
        self.calls += 1
        if self.fail:
            raise RuntimeError("stub client should not have been called on this leg")
        return "1. a step.\nAnswer: (A)"


def _write_items(tmp_path, n=3):
    items = [
        {"question": f"q{i}?", "choices": ["a", "b", "c", "d"], "answer_index": 0}
        for i in range(n)
    ]
    data = tmp_path / "toy_substrate.json"
    data.write_text(json.dumps(items))
    return data


def test_run_end_to_end_offline_writes_summary_and_transcripts(tmp_path, monkeypatch):
    stub = StubClient()
    monkeypatch.setattr(pilot, "_gate_client", lambda *a: stub)
    data = _write_items(tmp_path)
    out = tmp_path / "out"
    assert pilot.run(data, 3, out, "stub-model", "groq", "", 320, 5.0, resume=False) == 0
    # 2 calls per item: one clean generation + one direct probe, both parse first try.
    assert stub.calls == 6
    summary = json.loads((out / "pilot_summary_toy_substrate_stub-model.json").read_text())
    assert summary["protocol"] == pilot.PROTOCOL
    assert summary["status"] == pilot.STATUS_STRING
    assert summary["n_clean_correct"] == 3  # every answer is (A) = correct
    assert summary["moved_fraction_f"]["rate"] == 0.0  # direct always agrees
    assert summary["moved_yield_y"] == 0.0
    transcripts = json.loads(
        (out / "pilot_transcripts_toy_substrate_stub-model.json").read_text()
    )
    assert len(transcripts) == 3
    assert all(t["direct_answer"] == "A" for t in transcripts)
    assert (out / "pilot_checkpoint_toy_substrate_stub-model.json").exists()


def test_resume_of_a_complete_pilot_makes_zero_model_calls(tmp_path, monkeypatch):
    stub = StubClient()
    monkeypatch.setattr(pilot, "_gate_client", lambda *a: stub)
    data = _write_items(tmp_path)
    out = tmp_path / "out"
    assert pilot.run(data, 3, out, "m", "groq", "", 320, 5.0, resume=False) == 0
    # Second leg: everything banked, so a client call means resume is broken.
    monkeypatch.setattr(pilot, "_gate_client", lambda *a: StubClient(fail=True))
    assert pilot.run(data, 3, out, "m", "groq", "", 320, 5.0, resume=True) == 0
    summary = json.loads((out / "pilot_summary_toy_substrate_m.json").read_text())
    assert summary["n_invocations"] == 2
    assert summary["resumed"] is True
    assert summary["n_clean_correct"] == 3


def test_resume_refuses_a_checkpoint_from_a_different_protocol(
    tmp_path, monkeypatch, capsys
):
    stub = StubClient()
    monkeypatch.setattr(pilot, "_gate_client", lambda *a: stub)
    data = _write_items(tmp_path)
    out = tmp_path / "out"
    assert pilot.run(data, 3, out, "m", "groq", "", 320, 5.0, resume=False) == 0
    ckpt = out / "pilot_checkpoint_toy_substrate_m.json"
    payload = json.loads(ckpt.read_text())
    payload["params"]["protocol"] = "p3-substrate-pilot-v0-DIFFERENT"
    ckpt.write_text(json.dumps(payload))
    summary_path = out / "pilot_summary_toy_substrate_m.json"
    summary_before = summary_path.read_text()
    capsys.readouterr()
    # params_mismatch ignores non-PARAM_FIELDS keys, so the explicit protocol guard
    # is the only thing standing between a v0 checkpoint and a v1 merge. A zero-call
    # assertion alone is vacuous here (a fully banked merge also makes zero calls),
    # so assert the refusal actually printed and the summary was not rewritten.
    failing = StubClient(fail=True)
    monkeypatch.setattr(pilot, "_gate_client", lambda *a: failing)
    assert pilot.run(data, 3, out, "m", "groq", "", 320, 5.0, resume=True) == 0
    assert failing.calls == 0  # refused before the backend gate
    out_text = capsys.readouterr().out
    assert "REFUSED" in out_text and "protocol" in out_text
    assert summary_path.read_text() == summary_before  # not rewritten by a merge


def test_resume_refuses_changed_data_content(tmp_path, monkeypatch, capsys):
    stub = StubClient()
    monkeypatch.setattr(pilot, "_gate_client", lambda *a: stub)
    data = _write_items(tmp_path)
    out = tmp_path / "out"
    assert pilot.run(data, 3, out, "m", "groq", "", 320, 5.0, resume=False) == 0
    items = json.loads(data.read_text())
    items[0]["question"] = "edited q?"
    data.write_text(json.dumps(items))  # same path, different sha256
    capsys.readouterr()
    failing = StubClient(fail=True)
    monkeypatch.setattr(pilot, "_gate_client", lambda *a: failing)
    assert pilot.run(data, 3, out, "m", "groq", "", 320, 5.0, resume=True) == 0
    assert failing.calls == 0
    out_text = capsys.readouterr().out
    assert "REFUSED" in out_text and "data_sha256" in out_text


# --- Params fingerprint -------------------------------------------------------
def test_pilot_params_mismatch_refuses_arms_checkpoint_and_vice_versa():
    pilot_params = {
        "protocol": pilot.PROTOCOL, "model": "m", "backend": "groq", "n_items": 120,
        "data": "d.json", "data_sha256": "abc", "num_predict": 320,
    }
    arms_params = arms_resume.build_params(
        "m", "groq", 120, Path("d.json"), None, ["replay"], 130, 320,
        Path("h.json"), "abc", "def",
    )
    assert arms_resume.params_mismatch(arms_params, pilot_params)
    assert arms_resume.params_mismatch(pilot_params, arms_params)


def test_pilot_params_guard_the_fields_that_matter():
    base = {
        "protocol": pilot.PROTOCOL, "model": "m", "backend": "groq", "n_items": 120,
        "data": "d.json", "data_sha256": "abc", "num_predict": 320,
    }
    for field, other in (
        ("model", "m2"), ("backend", "ollama"), ("n_items", 60),
        ("data", "e.json"), ("data_sha256", "zzz"), ("num_predict", 64),
    ):
        changed = dict(base, **{field: other})
        assert arms_resume.params_mismatch(base, changed), field
    assert not arms_resume.params_mismatch(base, dict(base))
