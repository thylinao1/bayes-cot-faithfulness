"""The element 23 set never touches the answer, and its split is seeded and disjoint."""

from __future__ import annotations

import json

import pytest

from experiments.jury import manipulations as mp
from experiments.jury import synthetic_gate as sg

BANK = [
    "experiments/results/phase1-skeleton/qwen3-8b/arc_challenge/stated-hint/transcripts.jsonl",
    "experiments/results/control_transcripts_llama-3.1-8b-instant.json",
]


@pytest.fixture(scope="module")
def transcripts():
    rows = sg.load_bank(BANK)
    if not rows:
        pytest.skip("banked transcripts are not present in this checkout")
    return [
        {"item_id": f"{r.source}-{r.index:04d}", "reasoning": r.hinted_cot or r.clean_cot,
         "final_answer": r.clean_answer, "cue_text": r.cue_text}
        for r in rows
    ]


def test_the_frozen_list_is_five_types_by_three_phrasings():
    assert len(mp.MODIFICATION_TYPES) == 5
    assert mp.N_PHRASINGS == 3
    for phrasings in (mp.GENERIC_DISCLOSURE, mp.QUOTED_DENIED, mp.REDUNDANT_RATIONALE,
                      mp.RETROSPECTIVE, mp.RELOCATION_FRAMES):
        assert len(phrasings) == 3
        assert len(set(phrasings)) == 3


def test_the_search_budget_matches_the_preregistered_numbers():
    b = mp.SEARCH_BUDGET
    assert b["modification_types"] * b["phrasings_per_type"] * b["development_transcripts"] == 3000
    assert b["max_rescored_transcripts"] == 3000
    assert b["held_out_transcripts"] == 200
    assert b["held_out_evaluations"] == 1
    assert b["makes_new_model_calls"] is False


def test_the_final_answer_is_identical_before_and_after_every_modification(transcripts):
    built = mp.build_set(transcripts)
    assert len(built) == len(transcripts) * 5 * 3
    for m in built:
        source = next(t for t in transcripts if t["item_id"] == m.item_id)
        assert m.final_answer == source["final_answer"]
        assert m.reasoning_before == str(source["reasoning"]).rstrip("\n")


def test_every_modification_actually_changes_the_text(transcripts):
    for m in mp.build_set(transcripts):
        assert m.reasoning_after != m.reasoning_before


def test_answer_relocation_moves_the_conclusion_to_the_front_and_keeps_every_line(transcripts):
    for t in transcripts:
        m = mp.apply_modification(
            item_id=t["item_id"], reasoning=t["reasoning"], final_answer=t["final_answer"],
            modification="answer_relocation", phrasing_idx=0,
        )
        before = [x.strip() for x in m.reasoning_before.splitlines() if x.strip()]
        after = [x.strip() for x in m.reasoning_after.splitlines() if x.strip()]
        assert sorted(before) == sorted(after)
        assert after[0] == before[-1]
        assert m.final_answer == t["final_answer"]


def test_quoted_denied_carries_the_cue_text_through(transcripts):
    t = transcripts[0]
    m = mp.apply_modification(
        item_id=t["item_id"], reasoning=t["reasoning"], final_answer=t["final_answer"],
        modification="quoted_denied_cue", phrasing_idx=1, cue_text=t["cue_text"],
    )
    assert t["cue_text"] in m.reasoning_after


def test_unknown_modification_and_bad_phrasing_are_refused():
    with pytest.raises(mp.ManipulationError):
        mp.apply_modification(item_id="i", reasoning="a\nb", final_answer="A",
                              modification="nope", phrasing_idx=0)
    with pytest.raises(mp.ManipulationError):
        mp.apply_modification(item_id="i", reasoning="a\nb", final_answer="A",
                              modification="generic_disclosure", phrasing_idx=3)


def test_relocation_refuses_a_one_line_transcript():
    with pytest.raises(mp.ManipulationError, match="too few lines"):
        mp.apply_modification(item_id="i", reasoning="only one line", final_answer="A",
                              modification="answer_relocation", phrasing_idx=0)


def test_the_split_is_seeded_reproducible_and_disjoint():
    ids = [f"t{i:04d}" for i in range(600)]
    a = mp.split_items(ids, seed=7)
    b = mp.split_items(ids, seed=7)
    c = mp.split_items(ids, seed=8)
    assert a["dev"] == b["dev"] and a["held_out"] == b["held_out"]
    assert a["dev"] != c["dev"]
    assert set(a["dev"]) & set(a["held_out"]) == set()
    assert a["dev_actual"] == 200 and a["held_out_actual"] == 200
    assert len(a["unused"]) == 200


def test_the_split_reports_a_shortfall_rather_than_borrowing():
    ids = [f"t{i:03d}" for i in range(120)]
    s = mp.split_items(ids, seed=7)
    assert s["dev_actual"] == 120
    assert s["held_out_actual"] == 0
    assert s["dev_requested"] == 200
    assert set(s["dev"]) & set(s["held_out"]) == set()


def test_the_split_file_records_the_budget_and_both_hashes(tmp_path):
    ids = [f"t{i:04d}" for i in range(600)]
    path = mp.write_split(mp.split_items(ids, seed=7), tmp_path / "split.json")
    payload = json.loads(path.read_text())
    assert payload["search_budget"] == mp.SEARCH_BUDGET
    assert len(payload["dev_sha256"]) == 64 and len(payload["held_out_sha256"]) == 64
    assert payload["dev_sha256"] != payload["held_out_sha256"]
    assert payload["modification_types"] == list(mp.MODIFICATION_TYPES)
