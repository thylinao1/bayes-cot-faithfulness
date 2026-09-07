"""Unit tests for the FaithCoT-Bench field map.

Every test here is built to be able to fail: each one has a paired mutation that
flips the assertion, so a green run means the check discriminates rather than that
it is vacuous. The corpus-level tests skip when the gitignored download is absent,
which is the state of a fresh clone.
"""

from __future__ import annotations

import pathlib

import pytest

from bayes_cot_faithfulness.interventions import QAItem, clean_prompt, hinted_prompt
from experiments.external.faithcot_mapping import (
    CUE_RE,
    TYPE_TABLE,
    extract_cot,
    load_corpus,
    map_record,
)

CORPUS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "experiments"
    / "data"
    / "external"
    / "faithcot"
    / "faithcot"
)

_ITEM = QAItem(question="What colour is the sky?", choices=("blue", "red", "green"), answer_index=0)


def _record(**over) -> dict:
    base = {
        "cot_prompt": "Instructions: Read the question, give your answer by analyzing step by step.",
        "question": "What colour is the sky?",
        "options": ["blue", "red", "green"],
        "label": "A",
        "faithful_type": 3,
        "unfaithfulness": 0,
        "sample_0": {
            "step_1": "Step 1: The sky scatters short wavelengths.",
            "step_2": "Step 2: So it looks blue.",
            "full_response": "Step 1: ...\nStep 2: ...\nFinal Answer: The single, most likely answer is (A)",
            "parsed_final_answer": "A",
        },
    }
    sample_over = over.pop("sample_0", None)
    base.update(over)
    if sample_over:
        base["sample_0"] = {**base["sample_0"], **sample_over}
    return base


def _mapped(**over):
    return map_record(_record(**over), item_id="t/m/r0", dataset="t", generator_model="m")


# ---------------------------------------------------------------- cue detection


def test_cue_detector_fires_on_our_hinted_prompt_and_not_on_our_clean_prompt():
    """The falsification for the whole no-cue finding.

    If this detector could not see a planted cue, "no FaithCoT item carries a cue"
    would be a statement about the detector rather than about the corpus.
    """
    assert CUE_RE.search(hinted_prompt(_ITEM, strength="normal")) is not None
    assert CUE_RE.search(hinted_prompt(_ITEM, strength="strong")) is not None
    assert CUE_RE.search(clean_prompt(_ITEM)) is None


def test_cue_detector_reads_the_input_not_the_cot():
    """A model that talks about a hint does not make its item a cued item."""
    talked = _mapped(sample_0={"step_1": "Step 1: The hint says the answer key marks (B)."})
    assert talked.cue_present is False
    planted = _mapped(question="What colour is the sky? I think the answer is (B).")
    assert planted.cue_present is True
    assert planted.cue_text is not None


# ---------------------------------------------------------------- CoT extraction


def test_cot_is_the_steps_in_order_and_excludes_the_final_answer_line():
    sample = {
        "step_10": "Step 10: tenth.",
        "step_2": "Step 2: second.",
        "step_1": "Step 1: first.",
        "full_response": "junk\nFinal Answer: The single, most likely answer is (A)",
        "parsed_final_answer": "A",
    }
    cot, n = extract_cot(sample)
    assert n == 3
    assert cot == "Step 1: first.\nStep 2: second.\nStep 10: tenth."
    # The mutation that must fail: naive sorting would put step_10 before step_2.
    assert cot.index("Step 2:") < cot.index("Step 10:")
    # Leaking the answer into the CoT is the failure this guards.
    assert "Final Answer" not in cot


def test_blank_steps_are_not_counted():
    cot, n = extract_cot({"step_1": "Step 1: a.", "step_2": "   ", "parsed_final_answer": "A"})
    assert n == 1
    assert cot == "Step 1: a."


# ---------------------------------------------------------------- correctness


def test_correctness_is_derived_from_the_parsed_answer_against_the_gold_label():
    assert _mapped().answer_correct is True
    assert _mapped(sample_0={"parsed_final_answer": "B"}).answer_correct is False
    # Case and whitespace must not decide correctness.
    assert _mapped(sample_0={"parsed_final_answer": " a "}).answer_correct is True
    # An empty gold label can never count as a match.
    assert _mapped(label="", sample_0={"parsed_final_answer": ""}).answer_correct is False


# ---------------------------------------------------------------- Q2 orientation


def test_q2_supports_is_the_opposite_of_unfaithful():
    faithful = _mapped(unfaithfulness=0, faithful_type=3)
    assert faithful.label_unfaithful is False
    assert faithful.q2_supports is True
    unfaithful = _mapped(unfaithfulness=1, faithful_type=4)
    assert unfaithful.label_unfaithful is True
    assert unfaithful.q2_supports is False
    # A polarity slip shows up here as the two fields agreeing.
    assert faithful.q2_supports is not faithful.label_unfaithful


def test_type_4_is_post_hoc_rationalization_on_a_correct_answer():
    assert TYPE_TABLE[4] == (True, True)
    assert "post-hoc" in _mapped(faithful_type=4, unfaithfulness=1).type_name
    for wrong in (1, 2, 3):
        assert "post-hoc" not in _mapped(faithful_type=wrong).type_name


def test_type_table_matches_the_release_readme():
    assert TYPE_TABLE == {1: (False, False), 2: (False, True), 3: (True, False), 4: (True, True)}


# ---------------------------------------------------------------- consistency flag


def test_label_consistent_catches_a_type_that_contradicts_the_binary_label():
    assert _mapped(faithful_type=3, unfaithfulness=0).label_consistent is True
    # type 3 claims faithful, the binary label says unfaithful
    assert _mapped(faithful_type=3, unfaithfulness=1).label_consistent is False
    # type 3 claims a correct answer, the parsed answer is wrong
    assert _mapped(faithful_type=3, unfaithfulness=0, sample_0={"parsed_final_answer": "B"}).label_consistent is False


def test_unannotated_rows_carry_none_and_are_not_consistent():
    row = map_record(
        {k: v for k, v in _record().items() if k not in ("faithful_type", "unfaithfulness")},
        item_id="t/m/r0",
        dataset="t",
        generator_model="m",
    )
    assert row.label_unfaithful is None
    assert row.q2_supports is None
    assert row.annotated is False
    assert row.label_consistent is False


def test_out_of_taxonomy_type_is_not_silently_accepted():
    assert _mapped(faithful_type=0).faithful_type is None
    assert _mapped(faithful_type=9).faithful_type is None


# ---------------------------------------------------------------- corpus level


@pytest.mark.skipif(not CORPUS.exists(), reason="gitignored FaithCoT download not present")
def test_no_released_item_carries_a_cue():
    items = list(load_corpus(CORPUS))
    assert len(items) == 1364
    cued = [i for i in items if i.cue_present]
    assert cued == [], f"unexpected cued items: {[i.item_id for i in cued]}"


@pytest.mark.skipif(not CORPUS.exists(), reason="gitignored FaithCoT download not present")
def test_corpus_label_counts_match_the_manifest():
    items = list(load_corpus(CORPUS))
    annotated = [i for i in items if i.annotated]
    assert len(annotated) == 1304
    assert sum(1 for i in annotated if i.label_unfaithful) == 382
    assert sum(1 for i in annotated if not i.label_unfaithful) == 922
