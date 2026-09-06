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


# --- the Q1 revision -------------------------------------------------------------
#
# q1_mention_2026-09-07b.md is a NEW dated file, not an edit of the first one. The first
# file was scored on the record (job 825542, FP8 70B judge), so its bytes are fixed and the
# revision has to carry the same schema, the same blinding and the same abstain token while
# stating a different rule. These tests run every Q1 check against BOTH files.


@pytest.fixture(params=sorted(pf.Q1_PROMPT_FILES), ids=lambda k: f"q1-{k}")
def q1(request):
    return pf.load_prompts(q1=request.param)["Q1"]


def test_every_q1_file_exists_and_hashes_differently():
    loaded = {k: pf.load_prompt(pf.PROMPT_DIR / name) for k, name in pf.Q1_PROMPT_FILES.items()}
    shas = {p.sha256 for p in loaded.values()}
    assert len(shas) == len(loaded)
    assert all(len(s) == 64 for s in shas)
    assert loaded["a"].path.name == "q1_mention_2026-09-07.md"
    assert loaded["b"].path.name == "q1_mention_2026-09-07b.md"
    assert loaded["c"].path.name == "q1_mention_2026-09-07c.md"


def test_the_file_of_record_is_byte_identical_to_its_scored_sha256():
    """Job 825542's gate report names this hash. A revision must never touch it."""
    a = pf.load_prompt(pf.PROMPT_DIR / pf.Q1_PROMPT_FILES["a"])
    assert a.sha256 == "c11fa9cddebba21948ea2de2fe4b6b09c65a31504b50e394d837fba738978b59"


def test_every_q1_file_is_answer_blind_and_binary(q1):
    assert q1.sees_final_answer is False
    assert "FINAL_ANSWER" not in q1.placeholders
    assert q1.swap_target == "none"
    assert set(q1.allowed_votes) == {"yes", "no", "abstain"}
    assert q1.abstain_token == "abstain"
    assert q1.meta["sees_cue_identity"] is False
    assert q1.question == "Q1"
    assert q1.meta["date"] == "2026-09-07"
    assert tuple(q1.required_output_keys) == (
        "vote", "quote", "describes_effect_on_choice", "rationale"
    )


def test_every_q1_file_refuses_a_final_answer(q1):
    with pytest.raises(pf.PromptError, match="answer-blind"):
        pf.render(
            q1, question=ITEM["question"], choices=ITEM["choices"],
            reasoning=ITEM["reasoning"], final_answer="B",
        )


def test_every_q1_file_refuses_a_position_swap(q1):
    with pytest.raises(pf.PromptError, match="binary"):
        pf.render(
            q1, question=ITEM["question"], choices=ITEM["choices"],
            reasoning=ITEM["reasoning"], position_swap=True,
        )


def test_every_q1_file_renders_without_an_answer_section(q1):
    text = pf.render(
        q1, question=ITEM["question"], choices=ITEM["choices"], reasoning=ITEM["reasoning"],
    )
    assert "FINAL ANSWER" not in text
    assert "{{" not in text
    assert "non-repetition" in text.lower()
    assert ITEM["reasoning"] in text


def test_every_q1_file_validates_the_same_output_schema(q1):
    good = json.dumps({
        "vote": "yes", "quote": "the key marks B",
        "describes_effect_on_choice": False, "rationale": "names the key",
    })
    res = pf.validate_output(q1, good)
    assert res.ok and res.vote == "yes"
    assert not pf.validate_output(q1, '{"vote": "maybe"}').ok
    abstain = '{"vote": "abstain", "quote": "", "describes_effect_on_choice": false, "rationale": "truncated"}'
    assert pf.validate_output(q1, abstain).vote == q1.abstain_token


def test_the_second_revision_states_the_guard_as_a_speaker_test():
    c = pf.load_prompt(pf.PROMPT_DIR / pf.Q1_PROMPT_FILES["c"])
    body = c.body.lower()
    assert "own voice" in body
    assert "reproduced item text" in body
    assert "read to the last line" in body
    assert "reproduced prompt only" in body
    assert c.meta["revision_of"] == "q1_mention_2026-09-07b.md"


def test_the_revision_states_the_mention_construct_as_the_rule():
    b = pf.load_prompt(pf.PROMPT_DIR / pf.Q1_PROMPT_FILES["b"])
    body = b.body.lower()
    # The construct, not the guard, is the rule, and the guard is named as a guard.
    assert "the rule" in body
    assert "guard" in body
    assert "you do not need a sentence saying the source changed the choice" in body
    # The three contrasts the file of record scored the other way.
    assert "quoted and then denied" in body
    assert "paraphrased disclosure" in body
    assert "bare restatement" in body
    assert b.meta["revision_of"] == "q1_mention_2026-09-07.md"


def test_load_prompts_refuses_an_unknown_q1_file():
    with pytest.raises(pf.PromptError, match="no Q1 prompt file"):
        pf.load_prompts(q1="q1_mention_2027-01-01.md")


def test_load_prompts_default_matches_load_default_prompts():
    assert {q: p.sha256 for q, p in pf.load_prompts().items()} == {
        q: p.sha256 for q, p in pf.load_default_prompts().items()
    }


def test_the_gate_and_q2_files_never_vary_with_the_q1_choice():
    a = pf.load_prompts(q1="a")
    b = pf.load_prompts(q1="b")
    assert a["gate"].sha256 == b["gate"].sha256
    assert a["Q2"].sha256 == b["Q2"].sha256
    assert a["Q1"].sha256 != b["Q1"].sha256


def test_q1_variant_of_names_the_file():
    assert pf.q1_variant_of(pf.load_prompts(q1="a")["Q1"]) == "a"
    assert pf.q1_variant_of(pf.load_prompts(q1="b")["Q1"]) == "b"
