"""Element 8.3 (PF-13): the mediator-noise components, on numbers computed by hand.

The whole point of the deliverable is that a number replaces an assumption, so every
expected value below is worked out in the test's own comment from the definition in
section 8.3 and written in as a literal.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from bayes_cot_faithfulness.interventions import QAItem
from bayes_cot_faithfulness.repeat_curves import (
    identical_repeat_fraction,
    item_mean,
    variance_components,
    within_item_sd,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "08_additive_arms.py"
sys.path.insert(0, str(REPO / "experiments"))


def _load_arms_module():
    spec = importlib.util.spec_from_file_location("additive_arms_repeats", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_arms_module()


# --------------------------------------------------------------------------- #
# within_item_sd
# --------------------------------------------------------------------------- #
def test_within_item_sd_uses_the_r_minus_one_divisor():
    """[1, 3]: mean 2, SS 2, df 1, sd sqrt(2)."""
    assert within_item_sd([1.0, 3.0]) == pytest.approx(math.sqrt(2))


def test_three_identical_repeats_have_zero_spread():
    assert within_item_sd([2.5, 2.5, 2.5]) == 0.0


def test_one_usable_repeat_has_no_within_item_sd():
    """0.0 here would drag sigma_u down with the items that scored worst."""
    assert within_item_sd([1.0]) is None
    assert within_item_sd([1.0, None, None]) is None


def test_unscorable_repeats_are_dropped_not_zeroed():
    assert within_item_sd([1.0, None, 3.0]) == pytest.approx(math.sqrt(2))
    assert item_mean([1.0, None, 3.0]) == 2.0
    assert item_mean([None, None]) is None


# --------------------------------------------------------------------------- #
# variance_components, on a two-item example whose answer is arithmetic
# --------------------------------------------------------------------------- #
HAND_WORKED = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
# item 1: mean 2, SS = 1 + 0 + 1 = 2, df 2
# item 2: mean 5, SS = 1 + 0 + 1 = 2, df 2
# sigma_u = sqrt((2 + 2) / (2 + 2)) = 1
# item means [2, 5], grand mean 3.5, var (ddof 1) = (2.25 + 2.25) / 1 = 4.5
# sigma_m = sqrt(4.5) = 2.1213203...
# lambda = 4.5 / (4.5 + 1) = 0.8181818...
# noise-corrected between-item var = 4.5 - 1 / 3 = 4.1666667, lambda = 0.8064516...


def test_sigma_u_is_pooled_on_the_sums_of_squares():
    out = variance_components(HAND_WORKED)
    assert out["sigma_u"] == pytest.approx(1.0)
    assert out["total_within_item_df"] == 4
    assert out["n_items_used_for_sigma_u"] == 2


def test_sigma_m_is_the_across_item_sd_of_the_item_means():
    out = variance_components(HAND_WORKED)
    assert out["var_of_item_means"] == pytest.approx(4.5)
    assert out["sigma_m"] == pytest.approx(math.sqrt(4.5))


def test_lambda_is_sigma_m_squared_over_sigma_m_squared_plus_sigma_u_squared():
    out = variance_components(HAND_WORKED)
    assert out["lambda"] == pytest.approx(4.5 / 5.5)


def test_the_noise_corrected_lambda_is_reported_beside_the_raw_one():
    """The raw sigma_m is biased up by sigma_u^2 / r; both are printed, neither hidden."""
    out = variance_components(HAND_WORKED)
    assert out["sigma_m_noise_corrected"] == pytest.approx(math.sqrt(4.5 - 1 / 3))
    assert out["lambda_noise_corrected"] == pytest.approx(
        (4.5 - 1 / 3) / ((4.5 - 1 / 3) + 1)
    )
    assert out["lambda_noise_corrected"] < out["lambda"]


def test_zero_measurement_noise_gives_lambda_one():
    out = variance_components([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
    assert out["sigma_u"] == 0.0
    assert out["lambda"] == 1.0
    assert out["n_items_with_zero_within_item_sd"] == 2


def test_no_between_item_signal_gives_lambda_zero():
    out = variance_components([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert out["var_of_item_means"] == 0.0
    assert out["lambda"] == 0.0


def test_every_denominator_is_reported_and_the_held_out_items_are_named():
    per_item = [
        [1.0, 2.0, 3.0],   # used for both
        [4.0, None, None],  # a mean but no within-item spread
        [None, None, None],  # nothing scorable at all
    ]
    out = variance_components(per_item)
    assert out["n_items_total"] == 3
    assert out["n_items_with_a_scorable_repeat"] == 2
    assert out["n_items_used_for_sigma_u"] == 1
    assert out["n_items_held_out_no_scorable_repeat"] == 1
    assert out["n_items_held_out_single_scorable_repeat"] == 1


def test_a_single_item_has_no_between_item_spread_and_no_lambda():
    out = variance_components([[1.0, 2.0, 3.0]])
    assert out["sigma_u"] == pytest.approx(1.0)
    assert out["sigma_m"] is None
    assert out["lambda"] is None


def test_nothing_scorable_anywhere_gives_no_components_rather_than_zeros():
    out = variance_components([[None, None], [None, None]])
    assert out["sigma_u"] is None
    assert out["sigma_m"] is None
    assert out["lambda"] is None
    assert out["n_items_held_out_no_scorable_repeat"] == 2


# --------------------------------------------------------------------------- #
# identical_repeat_fraction
# --------------------------------------------------------------------------- #
def test_identical_repeats_are_counted_with_their_denominator():
    out = identical_repeat_fraction([
        ["aaa", "aaa", "aaa"],
        ["aaa", "bbb", "aaa"],
        ["aaa", None, "aaa"],
    ])
    assert out["n_items_compared"] == 2
    assert out["n_identical"] == 1
    assert out["identical_fraction"] == 0.5
    assert out["n_items_incomplete"] == 1


def test_an_incomplete_item_is_not_counted_as_agreeing():
    out = identical_repeat_fraction([[None, None, None]])
    assert out["n_items_compared"] == 0
    assert out["identical_fraction"] is None


# --------------------------------------------------------------------------- #
# The runner arm
# --------------------------------------------------------------------------- #
class _DeterministicClient:
    """Greedy answers are a pure function of the prompt; sampled ones vary with the seed."""

    def __init__(self):
        self.generate_calls = 0
        self.sample_calls = []

    def generate(self, prompt: str, num_predict: int = 320, **kw) -> str:
        self.generate_calls += 1
        letter = "ABCD"[sum(ord(c) for c in prompt) % 4]
        return f"Answer: ({letter})"

    def sample_completions(self, prompt, *, k, temperature, seed=None,
                           num_predict=320, system=None, stop=None):
        self.sample_calls.append({"temperature": temperature, "seed": seed, "k": k})
        letter = "ABCD"[(sum(ord(c) for c in prompt) + (seed or 0)) % 4]
        return [f"Answer: ({letter}) seed {seed}"] * k, "n_parameter"


def _records(n=2):
    item = QAItem(question="Which one is it?",
                  choices=("alpha", "beta", "gamma", "delta"), answer_index=0)
    return [{"item": item, "clean_correct": True, "clean_answer": "A",
             "clean_cot": f"first step {i}\nsecond step\nAnswer: (A)",
             "hinted_answer": "B",
             "hinted_cot": f"first step {i}\nsecond step\nAnswer: (B)"}
            for i in range(n)]


def _ctx(tmp_path, **kw):
    kwargs = {"n_choices": 4, "num_predict": 64, "out_dir": tmp_path, "safe_model": "fake",
              "backend": "openai", "model": "fake", "curve_cap": 6, "checkpoint": None,
              "concurrency": 1}
    kwargs.update(kw)
    return mod.RunCtx(**kwargs)


def test_the_arm_writes_r_repeats_per_arm_per_temperature(tmp_path):
    client = _DeterministicClient()
    records = _records(2)
    ok, err = mod.arm_repeat_curves(client, records, _ctx(tmp_path))
    assert (ok, err) == (True, None)
    block = records[0]["repeat_curves"]
    assert block["r"] == 3
    assert block["temperatures"] == [0.0, 0.7]
    for arm in ("clean", "hinted"):
        assert sorted(block["arms"][arm]) == ["0.0", "0.7"]
        for temp_key in ("0.0", "0.7"):
            reps = block["arms"][arm][temp_key]
            assert len(reps) == 3
            assert [r["repeat"] for r in reps] == [0, 1, 2]
            for r in reps:
                assert "commitment_depth" in r and "curve_area" in r


def test_temperature_zero_repeats_go_through_the_greedy_path(tmp_path):
    """The determinism measurement must use the same call path arm_curves uses."""
    client = _DeterministicClient()
    mod.arm_repeat_curves(client, _records(1), _ctx(tmp_path))
    assert client.generate_calls > 0
    assert all(c["temperature"] == 0.7 for c in client.sample_calls)
    assert all(c["k"] == 1 for c in client.sample_calls)


def test_the_repeats_of_one_item_carry_distinct_seeds(tmp_path):
    """Same seed on two repeats would make them the same draw, not a repeat."""
    client = _DeterministicClient()
    records = _records(1)
    mod.arm_repeat_curves(client, records, _ctx(tmp_path))
    for arm in ("clean", "hinted"):
        reps = records[0]["repeat_curves"]["arms"][arm]["0.7"]
        seeds = [r["seed"] for r in reps]
        assert len(set(seeds)) == 3, seeds
    # And no seed is shared across the two arms of the same item either.
    clean_seeds = {r["seed"] for r in records[0]["repeat_curves"]["arms"]["clean"]["0.7"]}
    hinted_seeds = {r["seed"] for r in records[0]["repeat_curves"]["arms"]["hinted"]["0.7"]}
    assert not (clean_seeds & hinted_seeds)


def test_a_deterministic_backend_gives_byte_identical_temperature_zero_repeats(tmp_path):
    client = _DeterministicClient()
    records = _records(3)
    mod.arm_repeat_curves(client, records, _ctx(tmp_path))
    summary = mod.summarize_repeat_curves(records)
    ident = summary["arms"]["clean"]["0.0"]["byte_identical_repeats"]
    assert ident["n_items_compared"] == 3
    assert ident["identical_fraction"] == 1.0
    # Same backend, same prompts: sigma_u on the greedy repeats is exactly 0.
    assert summary["arms"]["clean"]["0.0"]["curve_area"]["sigma_u"] == 0.0


def test_repeat_zero_keeps_its_text_and_the_others_keep_only_hashes(tmp_path):
    client = _DeterministicClient()
    records = _records(1)
    mod.arm_repeat_curves(client, records, _ctx(tmp_path))
    reps = records[0]["repeat_curves"]["arms"]["clean"]["0.0"]
    assert isinstance(reps[0]["completions"], list)
    assert "completions" not in reps[1]
    assert len(reps[1]["completions_sha256"]) == 16


def test_a_banked_item_is_not_repeated_again(tmp_path):
    client = _DeterministicClient()
    records = _records(2)
    records[0]["repeat_curves"] = {"r": 3, "temperatures": [0.0], "arms": {}}
    before = client.generate_calls
    ok, err = mod.arm_repeat_curves(client, records, _ctx(tmp_path))
    assert (ok, err) == (True, None)
    assert records[0]["repeat_curves"]["arms"] == {}
    assert client.generate_calls > before  # the second item did run


def test_the_arm_refuses_a_backend_that_cannot_sample_when_a_temperature_needs_it(tmp_path):
    class _GreedyOnly:
        def generate(self, prompt, num_predict=320, **kw):
            return "Answer: (A)"

    ok, err = mod.arm_repeat_curves(_GreedyOnly(), _records(1), _ctx(tmp_path))
    assert ok is False and isinstance(err, RuntimeError)
    # With only temperature 0 asked for, the same backend is fine.
    ok, err = mod.arm_repeat_curves(
        _GreedyOnly(), _records(1), _ctx(tmp_path, repeat_temperatures=(0.0,))
    )
    assert (ok, err) == (True, None)


def test_the_summary_reports_both_arms_both_temperatures_with_denominators(tmp_path):
    client = _DeterministicClient()
    records = _records(3)
    mod.arm_repeat_curves(client, records, _ctx(tmp_path))
    summary = mod.summarize_repeat_curves(records)
    assert summary["n_records_with_repeats"] == 3
    assert summary["r"] == 3
    assert summary["temperatures"] == ["0.0", "0.7"]
    for arm in ("clean", "hinted"):
        for temp_key in ("0.0", "0.7"):
            cell = summary["arms"][arm][temp_key]
            assert set(cell) == {"curve_area", "commitment_depth",
                                 "byte_identical_repeats"}
            assert cell["curve_area"]["n_items_total"] == 3
