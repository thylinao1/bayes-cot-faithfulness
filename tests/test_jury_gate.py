"""The synthetic gate corpus is controlled, and the gate can actually fail.

A gate that has never been seen to go red is decorative, so the last two tests force a
judge that answers wrongly and assert the FAIL, with the denominator it was computed on.
"""

from __future__ import annotations

import json

import pytest

from experiments.jury import gate as gate_mod
from experiments.jury import records as rec
from experiments.jury import synthetic_gate as sg
from experiments.jury.backends import JudgeEndpoint
from experiments.jury.family_map import JUDGE_BY_KEY, routing
from experiments.jury.gate_thresholds import THRESHOLDS
from experiments.jury.prompt_files import load_default_prompts
from experiments.jury.runner import JuryItem, JuryRunner

REPO_BANK = [
    "experiments/results/phase1-skeleton/qwen3-8b/arc_challenge/stated-hint/transcripts.jsonl",
    "experiments/results/control_transcripts_llama-3.1-8b-instant.json",
]


@pytest.fixture(scope="module")
def bank():
    rows = sg.load_bank(REPO_BANK)
    if not rows:
        pytest.skip("banked transcripts are not present in this checkout")
    return rows


def test_bank_and_corpus_denominators(bank):
    items = sg.build_items(bank)
    counts = sg.class_counts(items)
    assert set(counts) == set(sg.CLASSES)
    assert len(set(counts.values())) == 1, counts
    per_class = next(iter(counts.values()))
    assert per_class >= 60, f"only {per_class} items per class from {len(bank)} banked rows"
    assert len(items) == per_class * len(sg.CLASSES)


def test_every_item_id_is_unique(bank):
    items = sg.build_items(bank)
    assert len({i["item_id"] for i in items}) == len(items)


def test_planted_positive_is_the_clean_negative_plus_exactly_one_line(bank):
    items = sg.build_items(bank)
    by = {}
    for item in items:
        by.setdefault((item["meta"]["base_source"], item["meta"]["base_index"]), {})[
            item["meta"]["gate_class"]
        ] = item
    for group in by.values():
        clean = group["clean"]["reasoning"].splitlines()
        planted = group["planted_mention"]["reasoning"].splitlines()
        assert planted[: len(clean)] == clean
        assert len(planted) == len(clean) + 1


def test_gate_positive_keeps_the_reasoning_and_moves_only_the_answer(bank):
    items = sg.build_items(bank)
    by = {}
    for item in items:
        by.setdefault((item["meta"]["base_source"], item["meta"]["base_index"]), {})[
            item["meta"]["gate_class"]
        ] = item
    for group in by.values():
        clean, override = group["clean"], group["gate_positive"]
        assert override["reasoning"] == clean["reasoning"]
        assert override["final_answer"] != clean["final_answer"]
        assert override["final_answer"] != override["meta"]["cue_letter"]
        assert override["meta"]["truth"]["gate"] == "silent_override"


def test_restated_cue_class_carries_the_cue_words_but_is_a_negative(bank):
    items = [i for i in sg.build_items(bank) if i["meta"]["gate_class"] == "restated_cue_only"]
    assert items
    for item in items:
        assert item["meta"]["truth"]["q1"] is False
        assert "(" + item["meta"]["cue_letter"] + ")" in item["reasoning"]


def test_deleted_step_removes_a_line_and_keeps_the_answer(bank):
    items = sg.build_items(bank)
    by = {}
    for item in items:
        by.setdefault((item["meta"]["base_source"], item["meta"]["base_index"]), {})[
            item["meta"]["gate_class"]
        ] = item
    shorter = 0
    for group in by.values():
        clean, cut = group["clean"], group["deleted_step"]
        assert cut["final_answer"] == clean["final_answer"]
        if len(cut["reasoning"].splitlines()) < len(clean["reasoning"].splitlines()):
            shorter += 1
    assert shorter == len(by)


def test_phrasings_are_balanced_across_the_corpus(bank):
    items = [i for i in sg.build_items(bank) if i["meta"]["gate_class"] == "planted_mention"]
    seen = {0: 0, 1: 0, 2: 0}
    for item in items:
        seen[item["meta"]["phrasing_idx"]] += 1
    assert min(seen.values()) >= len(items) // 3 - 1


# --- the gate runs, and the gate fails ------------------------------------------


class _ScriptedClient:
    """A judge that answers by question, so the gate metrics are forced by construction."""

    def __init__(self, q1_vote="oracle"):
        self.q1_vote = q1_vote
        self.seed = 0
        self.base_url = "http://fake/v1"
        self.model = "fake"
        self.calls = 0

    def is_available(self):
        return True

    def generate(self, prompt, *, num_predict=320):
        self.calls += 1
        if '"vote": "coherent"' in prompt:
            override = "OVERRIDE-MARK" in prompt
            vote = "silent_override" if override else "coherent"
            return json.dumps({"vote": vote, "implied_answer": "A", "given_answer": "B", "rationale": "r"})
        if '"vote": "yes"' in prompt:
            if self.q1_vote == "oracle":
                vote = "yes" if "PLANT-MARK" in prompt else "no"
            else:
                vote = self.q1_vote
            return json.dumps({"vote": vote, "quote": "", "describes_effect_on_choice": vote == "yes", "rationale": "r"})
        return json.dumps({"vote": "supported", "load_bearing_step": "s", "rationale": "r"})


def _marked_items(bank, n_per_class=6):
    """A small corpus with markers the scripted judge can read, standing in for judgment."""
    items = sg.build_items(bank, per_class=n_per_class)
    for item in items:
        cls = item["meta"]["gate_class"]
        if item["meta"]["truth"].get("q1") is True:
            item["reasoning"] += "\nPLANT-MARK"
        if item["meta"]["truth"].get("gate") == "silent_override":
            item["reasoning"] += "\nOVERRIDE-MARK"
        item["meta"]["gate_class"] = cls
    return items


def _endpoints(judge_keys, q1_vote="oracle"):
    eps = {}
    for key in judge_keys:
        judge = JUDGE_BY_KEY[key]
        eps[key] = JudgeEndpoint(
            judge_key=key, backend="vllm", model=judge.hf_id, revision=judge.revision,
            client=_ScriptedClient(q1_vote),
        )
    return eps


def _run_gate(tmp_path, bank, q1_vote):
    raw = _marked_items(bank)
    items = [JuryItem(**dict(r)) for r in raw]
    panel = list(routing("Qwen3-8B"))
    out = tmp_path / "jury-gate" / "arc_challenge" / "stated-hint"
    runner = JuryRunner(
        endpoints=_endpoints(panel, q1_vote), prompts=load_default_prompts(), out_dir=out,
        substrate="arc_challenge", cue_family="stated-hint", mode="three-seeded",
        position_swap="first-run",
    )
    summary = runner.run(items, progress_every=0, concurrency=4)
    return gate_mod.build_report(runner.out_dir, raw, summary, panel), runner


def test_an_oracle_judge_passes_every_threshold(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "oracle")
    assert report["verdict"] == "PASS", json.dumps(report["per_judge"][0]["gate_verdicts"], indent=2)
    for judge in report["per_judge"]:
        assert judge["failed_metrics"] == []
        assert judge["metrics"]["recall_planted_mention"]["denominator"] == 6
        assert judge["metrics"]["test_retest_q1_min"]["rate"] == 1.0


def test_a_judge_that_never_says_yes_FAILS_recall_with_its_denominator(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "no")
    assert report["verdict"] == "FAIL"
    judge = report["per_judge"][0]
    assert "recall_planted_mention" in judge["failed_metrics"]
    m = judge["gate_verdicts"]["recall_planted_mention"]
    assert m["verdict"] == "FAIL"
    assert m["numerator"] == 0
    assert m["denominator"] == 6
    assert m["rate"] == 0.0
    assert m["threshold"] == THRESHOLDS["recall_planted_mention"]
    # The matched negative class still passes, which is what makes the failure specific.
    assert judge["gate_verdicts"]["specificity_clean"]["verdict"] == "PASS"


def test_a_judge_that_always_says_yes_FAILS_specificity(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "yes")
    judge = report["per_judge"][0]
    assert report["verdict"] == "FAIL"
    assert "specificity_clean" in judge["failed_metrics"]
    assert judge["gate_verdicts"]["specificity_clean"]["numerator"] == 0
    assert judge["gate_verdicts"]["specificity_clean"]["denominator"] == 6


def test_report_carries_the_threshold_hash_and_the_prompt_hashes(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "oracle")
    assert len(report["thresholds_sha256"]) == 64
    assert set(report["prompt_sha256"]) == {"gate", "Q1", "Q2"}
    assert report["thresholds"] == THRESHOLDS
    assert report["votes_per_second_per_server"] > 0


def test_corpus_manifest_matches_a_fresh_build(bank):
    """The committed manifest is the corpus's fingerprint; a drifted builder fails here."""
    import hashlib
    import json as _json
    from pathlib import Path as _Path

    manifest = _json.loads(
        _Path("experiments/jury/gate_corpus_manifest.json").read_text(encoding="utf-8")
    )
    items = sg.build_items(bank)
    assert manifest["items"] == len(items)
    assert manifest["per_class"] == sg.class_counts(items)
    seq = hashlib.sha256("\n".join(i["item_id"] for i in items).encode()).hexdigest()
    assert manifest["item_sequence_sha256"] == seq
    assert manifest["bank_rows_usable"] == len(bank)
