"""Element 9.2: the uncertain-item sampling arm, on samples whose entropy is known.

Every entropy in this file is computed by hand from the frozen definition (Shannon
entropy of the empirical answer distribution divided by log of the option count) and
written into the test as a literal, so a change to the implementation cannot make the
test agree with it. The two counts that straddle the frozen 0.30 threshold at k = 32
are 27/5 and 28/4, and both are asserted.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from bayes_cot_faithfulness.interventions import QAItem
from bayes_cot_faithfulness.sampling_arm import (
    STRATUM_RIGHT_CONFIDENT,
    STRATUM_RIGHT_UNCERTAIN,
    STRATUM_WRONG,
    UNCERTAIN_ENTROPY_THRESHOLD,
    answer_counts,
    answer_distribution,
    entropy_histogram,
    is_right_but_uncertain,
    modal_answer,
    normalized_answer_entropy,
    shannon_entropy,
    stability_across_k,
    stratum,
    summarize_item_samples,
    summarize_sampling,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "08_additive_arms.py"
sys.path.insert(0, str(REPO / "experiments"))


def _load_arms_module():
    spec = importlib.util.spec_from_file_location("additive_arms_sampling", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_arms_module()


# --------------------------------------------------------------------------- #
# Entropy on distributions whose value is known by hand
# --------------------------------------------------------------------------- #
def test_a_unanimous_item_has_zero_entropy():
    assert normalized_answer_entropy(["A"] * 32, 4) == 0.0


def test_a_uniform_four_way_split_has_normalized_entropy_one():
    answers = ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8
    assert normalized_answer_entropy(answers, 4) == pytest.approx(1.0)
    # In nats the same split is exactly log 4.
    assert shannon_entropy(answers) == pytest.approx(math.log(4))


def test_an_even_two_way_split_of_four_options_is_exactly_one_half():
    """H = log 2, denominator = log 4, so the ratio is 0.5 with no rounding."""
    answers = ["A"] * 16 + ["B"] * 16
    assert normalized_answer_entropy(answers, 4) == pytest.approx(0.5)


def test_a_three_quarter_one_quarter_split_matches_the_hand_computed_value():
    # -(0.75 ln 0.75 + 0.25 ln 0.25) = 0.5623351446... ; / ln 4 = 0.4056390622...
    answers = ["A"] * 24 + ["B"] * 8
    assert shannon_entropy(answers) == pytest.approx(0.5623351446, abs=1e-9)
    assert normalized_answer_entropy(answers, 4) == pytest.approx(0.4056390622, abs=1e-9)


def test_the_denominator_is_the_option_count_not_the_number_of_answers_seen():
    """The same two-way split is LESS uncertain against five options than four."""
    answers = ["A"] * 16 + ["B"] * 16
    assert normalized_answer_entropy(answers, 5) == pytest.approx(math.log(2) / math.log(5))
    assert normalized_answer_entropy(answers, 5) < normalized_answer_entropy(answers, 4)


def test_no_scorable_sample_gives_no_entropy_rather_than_zero():
    assert normalized_answer_entropy([None] * 32, 4) is None
    assert shannon_entropy([None] * 32) is None


def test_a_single_option_item_has_no_normalized_entropy():
    """log 1 is 0, so the ratio is undefined; None rather than an infinity."""
    assert normalized_answer_entropy(["A"] * 5, 1) is None


def test_unparsed_samples_leave_the_distribution_alone():
    answers = ["A"] * 16 + ["B"] * 8 + [None] * 8
    assert answer_counts(answers) == {"A": 16, "B": 8}
    assert answer_distribution(answers) == {"A": 2 / 3, "B": 1 / 3}


# --------------------------------------------------------------------------- #
# The mode
# --------------------------------------------------------------------------- #
def test_the_mode_is_the_most_frequent_parsed_answer():
    assert modal_answer(["A"] * 20 + ["B"] * 12) == ("A", False)


def test_a_tied_mode_is_broken_by_the_earliest_letter_and_flagged():
    assert modal_answer(["B"] * 16 + ["A"] * 16) == ("A", True)


def test_no_parsed_answer_has_no_mode():
    assert modal_answer([None, None]) == (None, False)


# --------------------------------------------------------------------------- #
# The frozen 0.30 threshold, at the boundary
# --------------------------------------------------------------------------- #
def test_the_threshold_is_at_or_above_so_exactly_030_is_uncertain():
    assert is_right_but_uncertain("A", "A", 0.30) is True
    assert is_right_but_uncertain("A", "A", 0.30 - 1e-12) is False


def test_the_two_k32_counts_that_straddle_the_frozen_threshold():
    """27/5 is above 0.30 and 28/4 is below it, on four options.

    27/32 and 5/32: -(0.84375 ln 0.84375 + 0.15625 ln 0.15625) = 0.4333988...,
    divided by ln 4 = 0.3126312... , which is uncertain.
    28/32 and 4/32: -(0.875 ln 0.875 + 0.125 ln 0.125) = 0.3767702...,
    divided by ln 4 = 0.2717817... , which is not.
    """
    just_over = ["A"] * 27 + ["B"] * 5
    just_under = ["A"] * 28 + ["B"] * 4
    e_over = normalized_answer_entropy(just_over, 4)
    e_under = normalized_answer_entropy(just_under, 4)
    assert e_over == pytest.approx(0.3126312, abs=1e-6)
    assert e_under == pytest.approx(0.2717817, abs=1e-6)
    assert e_over > UNCERTAIN_ENTROPY_THRESHOLD > e_under
    assert is_right_but_uncertain("A", "A", e_over) is True
    assert is_right_but_uncertain("A", "A", e_under) is False


def test_a_wrong_mode_is_never_right_but_uncertain_however_uncertain_it_is():
    answers = ["B"] * 8 + ["A"] * 8 + ["C"] * 8 + ["D"] * 8
    entropy = normalized_answer_entropy(answers, 4)
    assert entropy == pytest.approx(1.0)
    modal, tie = modal_answer(answers)
    assert modal == "A" and tie is True
    assert is_right_but_uncertain(modal, "B", entropy) is False
    assert stratum(modal, "B", entropy) == STRATUM_WRONG


def test_the_three_strata():
    assert stratum("A", "A", 0.9) == STRATUM_RIGHT_UNCERTAIN
    assert stratum("A", "A", 0.1) == STRATUM_RIGHT_CONFIDENT
    assert stratum("B", "A", 0.9) == STRATUM_WRONG
    assert stratum(None, "A", None) is None


# --------------------------------------------------------------------------- #
# k stability, on a sequence whose stratum is known at each k
# --------------------------------------------------------------------------- #
STRADDLING_DRAW = ["A"] * 5 + ["B"] * 27


def test_stability_reads_the_first_k_samples_in_draw_order():
    """The same 32 draws sit in three different strata at k = 5, 8 and 16.

    First 5 are all A: unanimous and correct, so right_confident.
    First 8 are 5 A and 3 B: mode still A, normalized entropy 0.4772, so
    right_uncertain.
    First 16 are 5 A and 11 B: the mode has moved to B, so wrong.
    """
    block = summarize_item_samples(
        STRADDLING_DRAW, n_options=4, answer_label="A", item_labels=list("ABCD")
    )
    by_k = {row["k"]: row for row in block["at_k"]}
    assert by_k[5]["stratum"] == STRATUM_RIGHT_CONFIDENT
    assert by_k[5]["normalized_entropy"] == 0.0
    assert by_k[8]["stratum"] == STRATUM_RIGHT_UNCERTAIN
    assert by_k[8]["normalized_entropy"] == pytest.approx(0.4772, abs=1e-4)
    assert by_k[16]["stratum"] == STRATUM_WRONG
    assert by_k[32]["stratum"] == STRATUM_WRONG
    assert block["stratum"] == STRATUM_WRONG


def test_stability_counts_changes_between_consecutive_k_with_denominators():
    steady = ["A"] * 32
    blocks = [
        summarize_item_samples(STRADDLING_DRAW, n_options=4, answer_label="A"),
        summarize_item_samples(steady, n_options=4, answer_label="A"),
    ]
    out = stability_across_k(blocks)
    steps = {(s["from_k"], s["to_k"]): s for s in out["steps"]}
    assert steps[(5, 8)]["n_items_comparable"] == 2
    assert steps[(5, 8)]["n_stratum_changed"] == 1
    assert steps[(5, 8)]["stratum_change_fraction"] == 0.5
    assert steps[(8, 16)]["n_stratum_changed"] == 1
    assert steps[(16, 32)]["n_stratum_changed"] == 0
    # The binary flag moves on 5 to 8 (confident to uncertain) and on 8 to 16
    # (uncertain to wrong), but not on 16 to 32.
    assert steps[(5, 8)]["n_flag_changed"] == 1
    assert steps[(8, 16)]["n_flag_changed"] == 1
    assert steps[(16, 32)]["n_flag_changed"] == 0


def test_an_item_with_nothing_scorable_is_absent_from_every_denominator():
    blocks = [
        summarize_item_samples(["A"] * 32, n_options=4, answer_label="A"),
        summarize_item_samples([None] * 32, n_options=4, answer_label="A"),
    ]
    out = stability_across_k(blocks)
    assert all(s["n_items_comparable"] == 1 for s in out["steps"])


def test_a_shorter_draw_reports_only_the_k_it_reached():
    block = summarize_item_samples(["A"] * 8, n_options=4, answer_label="A")
    assert [row["k"] for row in block["at_k"]] == [5, 8]


# --------------------------------------------------------------------------- #
# The per-item record and the arm block
# --------------------------------------------------------------------------- #
def test_the_item_record_carries_every_field_element_9_2_asks_for():
    answers = ["A"] * 24 + ["B"] * 6 + [None] * 2
    block = summarize_item_samples(
        answers, n_options=4, answer_label="A", item_labels=list("ABCD")
    )
    assert block["k"] == 32
    assert block["n_scorable"] == 30
    assert block["n_unscorable"] == 2
    assert block["answer_counts"] == {"A": 24, "B": 6}
    assert block["answer_distribution"]["A"] == pytest.approx(0.8)
    assert block["modal_answer"] == "A"
    assert block["modal_correct"] is True
    assert block["n_options"] == 4
    assert block["log_n_options"] == pytest.approx(math.log(4))
    assert block["entropy_threshold"] == UNCERTAIN_ENTROPY_THRESHOLD
    assert block["normalized_entropy"] == pytest.approx(
        (-(0.8 * math.log(0.8) + 0.2 * math.log(0.2))) / math.log(4)
    )
    assert block["right_but_uncertain"] is True


def test_an_out_of_set_answer_is_kept_and_counted():
    """A letter outside this item's own options stays in the distribution, visibly.

    The parser is run at the ROSTER option count, so a 3-option item can come back
    with a D. Dropping it would understate the spread; the count is reported instead.
    """
    block = summarize_item_samples(
        ["A"] * 16 + ["D"] * 16, n_options=3, answer_label="A", item_labels=list("ABC")
    )
    assert block["n_out_of_set"] == 16
    assert block["answer_counts"] == {"A": 16, "D": 16}
    assert block["normalized_entropy"] == pytest.approx(math.log(2) / math.log(3))


def test_the_arm_block_reports_the_uncertain_fraction_with_its_denominator():
    blocks = [
        summarize_item_samples(["A"] * 16 + ["B"] * 16, n_options=4, answer_label="A"),
        summarize_item_samples(["A"] * 32, n_options=4, answer_label="A"),
        summarize_item_samples(["B"] * 32, n_options=4, answer_label="A"),
        summarize_item_samples([None] * 32, n_options=4, answer_label="A"),
    ]
    out = summarize_sampling(blocks)
    assert out["n_items"] == 4
    assert out["n_items_with_entropy"] == 3
    assert out["n_items_no_scorable_sample"] == 1
    assert out["n_right_but_uncertain"] == 1
    assert out["right_but_uncertain_fraction"] == 0.25
    assert out["strata"] == {
        STRATUM_RIGHT_CONFIDENT: 1,
        STRATUM_RIGHT_UNCERTAIN: 1,
        STRATUM_WRONG: 1,
    }
    assert out["entropy_min"] == 0.0
    assert out["entropy_max"] == pytest.approx(0.5)


def test_the_entropy_histogram_puts_the_threshold_on_a_bin_edge():
    bins = entropy_histogram([], n_bins=10)
    edges = [b["lo"] for b in bins]
    assert UNCERTAIN_ENTROPY_THRESHOLD in edges
    blocks = [
        summarize_item_samples(["A"] * 32, n_options=4, answer_label="A"),
        summarize_item_samples(["A"] * 16 + ["B"] * 16, n_options=4, answer_label="A"),
        summarize_item_samples(
            ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8, n_options=4, answer_label="A"
        ),
    ]
    bins = entropy_histogram(blocks, n_bins=10)
    assert bins[0]["n"] == 1          # entropy 0.0
    assert bins[5]["n"] == 1          # entropy 0.5 lands in [0.5, 0.6)
    assert bins[-1]["lo"] == 1.0 and bins[-1]["n"] == 1  # entropy 1.0 is the above bin
    assert sum(b["n"] for b in bins) == 3


# --------------------------------------------------------------------------- #
# The runner arm, against a fake backend
# --------------------------------------------------------------------------- #
class _ScriptedSampler:
    """A backend whose k samples are fixed per item, so the arm's output is known."""

    def __init__(self, per_item_letters, method="n_parameter"):
        self.per_item_letters = per_item_letters
        self.method = method
        self.calls = []

    def sample_completions(self, prompt, *, k, temperature, seed=None,
                           num_predict=320, system=None, stop=None):
        idx = len(self.calls)
        self.calls.append({"k": k, "temperature": temperature, "seed": seed,
                           "prompt": prompt})
        letters = self.per_item_letters[idx]
        return [f"some reasoning\nAnswer: ({x})" for x in letters], self.method


def _item(answer_index=0):
    return QAItem(question="Which one is it?",
                  choices=("alpha", "beta", "gamma", "delta"),
                  answer_index=answer_index)


def _records(n=2):
    return [{"item": _item(), "clean_correct": True, "clean_answer": "A",
             "clean_cot": "step\nAnswer: (A)", "hinted_answer": "B",
             "hinted_cot": "step\nAnswer: (B)"} for _ in range(n)]


def _ctx(tmp_path, **kw):
    kwargs = dict(n_choices=4, num_predict=64, out_dir=tmp_path, safe_model="fake",
                  backend="openai", model="fake", curve_cap=6, checkpoint=None,
                  concurrency=1, sampling_seed=100)
    kwargs.update(kw)
    return mod.RunCtx(**kwargs)


def test_the_arm_writes_one_sampling_block_per_record(tmp_path):
    client = _ScriptedSampler([["A"] * 16 + ["B"] * 16, ["A"] * 32])
    records = _records(2)
    ok, err = mod.arm_sampling(client, records, _ctx(tmp_path, sampling_k=32))
    assert (ok, err) == (True, None)
    assert records[0]["sampling"]["normalized_entropy"] == pytest.approx(0.5)
    assert records[0]["sampling"]["right_but_uncertain"] is True
    assert records[1]["sampling"]["right_but_uncertain"] is False
    assert len(records[0]["sampling"]["samples"]) == 32
    assert records[0]["sampling"]["samples"][0]["answer"] == "A"


def test_the_arm_draws_on_the_clean_prompt_at_the_frozen_temperature(tmp_path):
    client = _ScriptedSampler([["A"] * 32, ["A"] * 32])
    records = _records(2)
    mod.arm_sampling(client, records, _ctx(tmp_path, sampling_k=32))
    assert [c["temperature"] for c in client.calls] == [0.7, 0.7]
    assert [c["k"] for c in client.calls] == [32, 32]
    # Item i draws with sampling_seed + i * k, so no two items share a draw.
    assert [c["seed"] for c in client.calls] == [100, 132]
    # The prompt is the clean one, asserted by EQUALITY against clean_prompt rather
    # than by the absence of a cue string: a cued frame with an empty cue also lacks
    # the cue string, and the first version of this test passed under exactly that
    # substitution (see docs/w3b-proofs/sampling_and_repeat_falsification.txt).
    from bayes_cot_faithfulness.interventions import clean_prompt
    assert client.calls[0]["prompt"] == clean_prompt(records[0]["item"])


def test_the_arm_records_which_draw_path_the_server_gave(tmp_path):
    for method in ("n_parameter", "seeded_calls"):
        client = _ScriptedSampler([["A"] * 32], method=method)
        records = _records(1)
        mod.arm_sampling(client, records, _ctx(tmp_path, sampling_k=32))
        assert records[0]["sampling"]["draw_method"] == method
        block = mod.summarize_sampling(records)
        assert block["draw_methods"] == {method: 1}


def test_a_banked_item_is_not_redrawn(tmp_path):
    client = _ScriptedSampler([["A"] * 32])
    records = _records(2)
    records[0]["sampling"] = {"k": 32, "normalized_entropy": 0.0}
    ok, err = mod.arm_sampling(client, records, _ctx(tmp_path, sampling_k=32))
    assert (ok, err) == (True, None)
    assert len(client.calls) == 1


def test_the_arm_refuses_a_backend_that_cannot_sample(tmp_path):
    class _GreedyOnly:
        def generate(self, prompt, num_predict=320, **kw):
            return "Answer: (A)"

    ok, err = mod.arm_sampling(_GreedyOnly(), _records(1), _ctx(tmp_path))
    assert ok is False
    assert isinstance(err, RuntimeError)
    assert "sample_completions" in str(err)


def test_an_unparsed_sample_is_counted_not_forced(tmp_path):
    """No second call is made to rescue an unparseable sample."""

    class _HalfUnparsed(_ScriptedSampler):
        def sample_completions(self, prompt, *, k, temperature, seed=None,
                               num_predict=320, system=None, stop=None):
            self.calls.append({"k": k, "temperature": temperature, "seed": seed,
                               "prompt": prompt})
            return (["Answer: (A)"] * 16 + ["no answer here at all"] * 16), "n_parameter"

    client = _HalfUnparsed([[]])
    records = _records(1)
    mod.arm_sampling(client, records, _ctx(tmp_path, sampling_k=32))
    block = records[0]["sampling"]
    assert len(client.calls) == 1
    assert block["n_scorable"] == 16
    assert block["n_unscorable"] == 16
    assert block["normalized_entropy"] == 0.0


def test_a_unanimous_item_reports_positive_zero_not_negative_zero():
    """-(1 * log 1) is -0.0, which is equal to 0.0 but prints like a bug in a record."""
    import math as _math
    assert not _math.copysign(1.0, shannon_entropy(["A"] * 32)) < 0
    assert not _math.copysign(1.0, normalized_answer_entropy(["A"] * 32, 4)) < 0
