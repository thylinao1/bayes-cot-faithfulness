"""The pure half of the LoRA recipe check: probe selection, prefixes, agreement, curves.

Everything here runs without torch, transformers or peft, which is the point of
``ladder.recipe_probe`` existing as its own module: the decisions the probe makes are
checkable on a laptop, and only the forward pass is left inside the torch path.
"""

from __future__ import annotations

import pytest

from bayes_cot_faithfulness.ladder import recipe_probe
from bayes_cot_faithfulness.ladder.recipe_probe import ProbeError


def _example(pool_index, *, trigger_present=True, trigger_option=1, gold_index=0,
             target_index=1, completion=None):
    return {
        "pool_index": pool_index,
        "trigger_present": trigger_present,
        "trigger_option": trigger_option if trigger_present else None,
        "gold_index": gold_index,
        "target_index": target_index,
        "prompt": f"question {pool_index}",
        "completion": completion if completion is not None
        else f"1. Reasoning for {pool_index}.\nAnswer: (B)",
    }


# --- completion_prefix ---------------------------------------------------------------

def test_completion_prefix_ends_at_the_open_bracket():
    ex = _example(0, completion="1. Steps.\nAnswer: (C)")
    assert recipe_probe.completion_prefix(ex) == "1. Steps.\nAnswer: ("


def test_completion_prefix_takes_the_last_marker_not_the_first():
    # A banked trace that wrote an answer line mid-reasoning must not move the read
    # position off the line the builder appended.
    ex = _example(0, completion="1. I nearly wrote Answer: (A) here.\nAnswer: (D)")
    prefix = recipe_probe.completion_prefix(ex)
    assert prefix.endswith("here.\nAnswer: (")
    assert prefix.count("Answer: (") == 2


def test_completion_prefix_refuses_a_completion_with_no_answer_line():
    with pytest.raises(ProbeError, match="no 'Answer: \\('"):
        recipe_probe.completion_prefix(_example(0, completion="1. No answer at all."))


# --- select_probe_examples -----------------------------------------------------------

def test_select_probe_examples_takes_only_trigger_carrying_rows():
    examples = [_example(i, trigger_present=(i % 2 == 0)) for i in range(20)]
    idx = recipe_probe.select_probe_examples(examples, 6, seed=1)
    assert len(idx) == 6
    assert all(examples[i]["trigger_present"] for i in idx)


def test_select_probe_examples_is_deterministic_and_seed_dependent():
    examples = [_example(i) for i in range(50)]
    a = recipe_probe.select_probe_examples(examples, 8, seed=20260911)
    b = recipe_probe.select_probe_examples(examples, 8, seed=20260911)
    c = recipe_probe.select_probe_examples(examples, 8, seed=20260923)
    assert a == b
    assert a != c


def test_select_probe_examples_does_not_just_take_the_first_n():
    # A probe that read the pool's first n items would be reading the training loop's
    # own first batch, which is the one place the loss is least informative.
    examples = [_example(i) for i in range(200)]
    idx = recipe_probe.select_probe_examples(examples, 10, seed=7)
    assert idx != list(range(10))


def test_select_probe_examples_caps_at_the_eligible_count():
    examples = [_example(i, trigger_present=(i < 3)) for i in range(10)]
    assert len(recipe_probe.select_probe_examples(examples, 64, seed=1)) == 3


def test_select_probe_examples_of_zero_is_empty_not_an_error():
    assert recipe_probe.select_probe_examples([_example(0)], 0, seed=1) == []


def test_select_probe_examples_refuses_when_nothing_carries_the_trigger():
    examples = [_example(i, trigger_present=False) for i in range(5)]
    with pytest.raises(ProbeError, match="no trigger-carrying examples"):
        recipe_probe.select_probe_examples(examples, 4, seed=1)


# --- summarize_probe -----------------------------------------------------------------

def _row(pred, trig, gold, p=0.5):
    return {"predicted_index": pred, "trigger_option": trig, "gold_index": gold,
            "p_trigger": p}


def test_summarize_probe_reports_both_denominators():
    rows = [
        _row(1, 1, 0),   # follows the trigger, trigger is not gold
        _row(0, 1, 0),   # follows gold instead
        _row(2, 2, 2),   # trigger IS gold, so agreement here is uninformative
        _row(3, 1, 0),   # follows neither
    ]
    s = recipe_probe.summarize_probe(rows)
    assert s["n"] == 4
    assert s["n_agree_with_trigger"] == 2
    assert s["agree_with_trigger"] == 0.5
    assert s["n_trigger_not_gold"] == 3
    assert s["n_agree_with_trigger_when_not_gold"] == 1
    assert s["agree_with_trigger_when_not_gold"] == pytest.approx(1 / 3)
    assert s["n_agree_with_gold"] == 2


def test_summarize_probe_when_every_trigger_marks_gold_has_no_clean_subset():
    rows = [_row(0, 0, 0), _row(1, 1, 1)]
    s = recipe_probe.summarize_probe(rows)
    assert s["n_trigger_not_gold"] == 0
    assert s["agree_with_trigger_when_not_gold"] is None
    assert s["agree_with_trigger"] == 1.0


def test_summarize_probe_on_nothing_returns_none_not_zero():
    s = recipe_probe.summarize_probe([])
    assert s["n"] == 0
    assert s["agree_with_trigger"] is None
    assert s["mean_p_trigger"] is None


def test_summarize_probe_mean_p_trigger():
    s = recipe_probe.summarize_probe([_row(1, 1, 0, 0.2), _row(1, 1, 0, 0.8)])
    assert s["mean_p_trigger"] == pytest.approx(0.5)


# --- loss_curve and throughput -------------------------------------------------------

def test_loss_curve_keeps_every_nth_step_plus_the_first_and_the_last():
    losses = [float(i) for i in range(25)]
    curve = recipe_probe.loss_curve(losses, 10)
    assert [row["step"] for row in curve] == [1, 10, 20, 25]
    assert curve[0]["loss"] == 0.0
    assert curve[-1]["loss"] == 24.0


def test_loss_curve_does_not_duplicate_a_step_that_is_both_last_and_nth():
    curve = recipe_probe.loss_curve([float(i) for i in range(20)], 10)
    steps = [row["step"] for row in curve]
    assert steps == [1, 10, 20]
    assert len(steps) == len(set(steps))


def test_loss_curve_of_nothing_is_empty():
    assert recipe_probe.loss_curve([], 10) == []


def test_loss_curve_with_every_zero_falls_back_to_every_step():
    assert len(recipe_probe.loss_curve([1.0, 2.0, 3.0], 0)) == 3


def test_throughput_is_tokens_over_seconds():
    assert recipe_probe.throughput(1000, 4.0) == 250.0


def test_throughput_of_no_elapsed_time_is_none_not_zero():
    assert recipe_probe.throughput(1000, 0.0) is None
    assert recipe_probe.throughput(1000, None) is None
