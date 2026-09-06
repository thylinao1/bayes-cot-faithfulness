"""The dated prompt files parse, render inside their declared blinding, and validate."""

from __future__ import annotations

import json

import pytest

from experiments.jury import prompt_files as pf

ITEM = {
    "question": "Which planet is closest to the Sun?",
    "choices": ["Venus", "Mercury", "Earth", "Mars"],
    "reasoning": "1. The order outward is Mercury, Venus, Earth, Mars.\n2. So the answer is Mercury.",
    "final_answer": "B",
}


@pytest.fixture(scope="module")
def prompts():
    return pf.load_default_prompts()


def test_three_dated_prompts_load_with_distinct_hashes(prompts):
    assert set(prompts) == {"gate", "Q1", "Q2"}
    shas = {q: p.sha256 for q, p in prompts.items()}
    assert len(set(shas.values())) == 3
    for p in prompts.values():
        assert len(p.sha256) == 64
        assert p.meta["date"] == "2026-09-07"
        assert p.path.name.endswith("_2026-09-07.md")


def test_q1_is_answer_blind_and_binary(prompts):
    q1 = prompts["Q1"]
    assert q1.sees_final_answer is False
    assert "FINAL_ANSWER" not in q1.placeholders
    assert q1.swap_target == "none"
    assert set(q1.allowed_votes) == {"yes", "no", "abstain"}
    assert q1.abstain_token == "abstain"


def test_q1_render_refuses_a_final_answer(prompts):
    with pytest.raises(pf.PromptError, match="answer-blind"):
        pf.render(
            prompts["Q1"],
            question=ITEM["question"],
            choices=ITEM["choices"],
            reasoning=ITEM["reasoning"],
            final_answer="B",
        )


def test_q1_render_has_no_final_answer_section(prompts):
    text = pf.render(
        prompts["Q1"],
        question=ITEM["question"],
        choices=ITEM["choices"],
        reasoning=ITEM["reasoning"],
    )
    assert "FINAL ANSWER" not in text
    assert "{{" not in text
    assert "non-repetition" in text.lower() or "NON-REPETITION" in text


def test_position_swap_is_refused_on_the_binary_question(prompts):
    with pytest.raises(pf.PromptError, match="binary"):
        pf.render(
            prompts["Q1"],
            question=ITEM["question"],
            choices=ITEM["choices"],
            reasoning=ITEM["reasoning"],
            position_swap=True,
        )


def test_gate_swaps_option_display_order_but_keeps_letter_binding(prompts):
    gate = prompts["gate"]
    plain = pf.render(
        gate, question=ITEM["question"], choices=ITEM["choices"],
        reasoning=ITEM["reasoning"], final_answer="B",
    )
    swapped = pf.render(
        gate, question=ITEM["question"], choices=ITEM["choices"],
        reasoning=ITEM["reasoning"], final_answer="B", position_swap=True,
    )
    assert plain != swapped
    for text in (plain, swapped):
        assert "(B) Mercury" in text
        assert "(A) Venus" in text
    assert plain.index("(A) Venus") < plain.index("(D) Mars")
    assert swapped.index("(D) Mars") < swapped.index("(A) Venus")


def test_q2_swaps_category_order(prompts):
    q2 = prompts["Q2"]
    plain = pf.render(
        q2, question=ITEM["question"], choices=ITEM["choices"],
        reasoning=ITEM["reasoning"], final_answer="B",
    )
    swapped = pf.render(
        q2, question=ITEM["question"], choices=ITEM["choices"],
        reasoning=ITEM["reasoning"], final_answer="B", position_swap=True,
    )
    assert plain.index("supported:") < plain.index("uncertain:")
    assert swapped.index("uncertain:") < swapped.index("supported:")


def test_validate_accepts_a_clean_object(prompts):
    body = json.dumps({"vote": "yes", "quote": "the key says B", "describes_effect_on_choice": True, "rationale": "x"})
    res = pf.validate_output(prompts["Q1"], body)
    assert res.ok and res.vote == "yes"


def test_validate_accepts_json_inside_a_code_fence(prompts):
    body = 'Here you go:\n```json\n{"vote": "no", "quote": "", "describes_effect_on_choice": false, "rationale": "own analysis"}\n```\nDone.'
    res = pf.validate_output(prompts["Q1"], body)
    assert res.ok and res.vote == "no"


@pytest.mark.parametrize(
    "bad,reason",
    [
        ("", "empty"),
        ("no json at all", "not JSON"),
        ('{"vote": "maybe", "quote": "", "describes_effect_on_choice": false, "rationale": "x"}', "not in"),
        ('{"vote": "yes"}', "missing keys"),
        ('{"vote": 1, "quote": "", "describes_effect_on_choice": false, "rationale": "x"}', "not a string"),
        ("[1, 2, 3]", "not JSON"),
    ],
)
def test_validate_rejects_malformed_output(prompts, bad, reason):
    res = pf.validate_output(prompts["Q1"], bad)
    assert not res.ok
    assert reason in res.error


def test_abstain_is_a_valid_vote_not_a_malformed_one(prompts):
    body = '{"vote": "abstain", "quote": "", "describes_effect_on_choice": false, "rationale": "truncated"}'
    res = pf.validate_output(prompts["Q1"], body)
    assert res.ok and res.vote == prompts["Q1"].abstain_token
