"""Ruling R4: the chain-level mediator-noise arm, on synthetic chains.

Two things are being asserted, and they are different things.

  1. The ARITHMETIC. sigma_u, sigma_m and lambda on hand-built per-item values whose
     answer is computable by hand, including the denominators, which are the part that
     goes wrong quietly when a repeat fails to score.
  2. The DRAW. The arm redraws the chain and reads the curve on the redraw, against a
     fake client that returns scripted chains, so the test can see that three different
     chains produce three different curves and that a chain whose own answer does not
     parse is counted rather than imputed.

The falsification is explicit: an estimator that ignored the within-item spread would
report lambda = 1.0 on data where the chains genuinely move, and the test that would
catch that is written so it FAILS against such an estimator (a stub of one is built
here and asserted to disagree).

Nothing here reaches the network.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

from bayes_cot_faithfulness.chain_repeats import (
    frame_components,
    per_item_rows,
    summarize_chain_repeats,
)
from bayes_cot_faithfulness.interventions import QAItem

REPO = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO / "experiments" / "08_additive_arms.py"
    spec = importlib.util.spec_from_file_location("additive_arms_chain_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARMS = _load_runner()


# --- 1. the arithmetic -----------------------------------------------------------

def _reps(values, *, shas=None, answers=None):
    """Per-item repeat dicts carrying one scalar each, the shape the arm writes."""
    out = []
    for i, v in enumerate(values):
        out.append({
            "curve_area": v,
            "commitment_depth": None if v is None else v * 2,
            "chain_sha256": (shas[i] if shas else f"sha{i}"),
            "chain_answer": (answers[i] if answers else ("A" if v is not None else None)),
        })
    return out


def test_sigma_u_sigma_m_and_lambda_on_values_whose_answer_is_known_by_hand():
    # Item 1 repeats 0.0, 0.2, 0.4 -> mean 0.2, sd 0.2 (ddof 1)
    # Item 2 repeats 1.0, 1.2, 1.4 -> mean 1.2, sd 0.2
    # pooled sigma_u = 0.2 exactly; item means 0.2 and 1.2 -> sigma_m = sqrt(0.5)
    per_item = [_reps([0.0, 0.2, 0.4]), _reps([1.0, 1.2, 1.4])]
    got = frame_components(per_item)["curve_area"]
    assert abs(got["sigma_u"] - 0.2) < 1e-12
    assert abs(got["sigma_m"] - math.sqrt(0.5)) < 1e-12
    lam = 0.5 / (0.5 + 0.04)
    assert abs(got["lambda"] - lam) < 1e-12
    assert got["total_within_item_df"] == 4
    assert got["n_items_used_for_sigma_u"] == 2
    assert got["n_items_with_a_scorable_repeat"] == 2
    assert got["n_items_held_out_no_scorable_repeat"] == 0


def test_the_falsification_an_estimator_blind_to_within_item_spread_would_say_lambda_1():
    """The failure mode this arm exists to avoid, made to happen on purpose.

    A3.5's temperature-0 rows reported lambda 1.0 because sigma_u was zero. If the
    chain-level estimator dropped the within-item term the same way, it would report
    1.0 on chains that genuinely move. The stub below IS that estimator, and the real
    one is asserted to disagree with it.
    """
    per_item = [_reps([0.0, 0.5, 1.0]), _reps([2.0, 2.5, 3.0])]

    def blind_lambda(rows):
        means = [sum(v["curve_area"] for v in r) / len(r) for r in rows]
        gm = sum(means) / len(means)
        var = sum((m - gm) ** 2 for m in means) / (len(means) - 1)
        return var / (var + 0.0)  # sigma_u assumed zero

    assert blind_lambda(per_item) == 1.0
    real = frame_components(per_item)["curve_area"]
    assert real["sigma_u"] > 0.4
    assert real["lambda"] < 0.95
    assert real["lambda"] != blind_lambda(per_item)


def test_an_unscorable_repeat_leaves_the_denominator_rather_than_being_imputed():
    per_item = [_reps([0.0, None, 0.4]), _reps([None, None, None])]
    got = frame_components(per_item)["curve_area"]
    # item 0 keeps 2 scorable repeats -> df 1; item 1 has none and enters no denominator
    assert got["total_within_item_df"] == 1
    assert got["n_items_used_for_sigma_u"] == 1
    assert got["n_items_with_a_scorable_repeat"] == 1
    assert got["n_items_held_out_no_scorable_repeat"] == 1
    assert got["n_items_total"] == 2
    assert got["sigma_m"] is None  # one item mean is not a between-item spread


def test_one_scorable_repeat_gives_a_mean_but_no_within_item_sd():
    per_item = [_reps([0.3, None, None]), _reps([0.9, 1.1, 1.3])]
    got = frame_components(per_item)["curve_area"]
    assert got["n_items_with_a_scorable_repeat"] == 2
    assert got["n_items_used_for_sigma_u"] == 1
    assert got["n_items_held_out_single_scorable_repeat"] == 1


def test_identical_chains_are_counted_beside_lambda_so_a_1_0_can_be_read():
    """Identical chains and a lambda of 1.0 have to be distinguishable in the record."""
    frozen = [_reps([0.5, 0.5, 0.5], shas=["s", "s", "s"]),
              _reps([1.5, 1.5, 1.5], shas=["t", "t", "t"])]
    got = frame_components(frozen)
    assert got["curve_area"]["sigma_u"] == 0.0
    assert got["curve_area"]["lambda"] == 1.0
    assert got["identical_chains"]["n_identical"] == 2
    assert got["identical_chains"]["identical_fraction"] == 1.0

    moving = [_reps([0.0, 0.5, 1.0], shas=["a", "b", "c"]),
              _reps([2.0, 2.5, 3.0], shas=["d", "e", "f"])]
    got = frame_components(moving)
    assert got["identical_chains"]["n_identical"] == 0
    assert got["curve_area"]["lambda"] < 1.0


def test_per_item_rows_carry_each_item_s_own_sigma_u_with_its_count():
    rows = per_item_rows([_reps([0.0, 0.2, 0.4]), _reps([1.0, None, 1.4])])
    assert abs(rows[0]["sigma_u_curve_area"] - 0.2) < 1e-12
    assert rows[0]["n_repeats_scorable"]["curve_area"] == 3
    assert rows[0]["identical_chains"] is False
    assert abs(rows[1]["sigma_u_curve_area"] - (0.4 / math.sqrt(2))) < 1e-12
    assert rows[1]["n_repeats_scorable"]["curve_area"] == 2
    assert rows[1]["n_chains_unparsed"] == 1


def test_summarize_reports_both_frames_with_their_own_denominators():
    blocks = [{
        "r": 3, "temperature": 0.7,
        "frames": {"clean": _reps([0.0, 0.1, 0.2]), "hinted": _reps([0.5, 0.9, 1.3])},
    }, {
        "r": 3, "temperature": 0.7,
        "frames": {"clean": _reps([1.0, 1.1, 1.2]), "hinted": _reps([2.5, 2.9, 3.3])},
    }]
    out = summarize_chain_repeats(blocks)
    assert out["r"] == 3 and out["temperature"] == 0.7
    assert set(out["frames"]) == {"clean", "hinted"}
    assert out["frames"]["hinted"]["curve_area"]["sigma_u"] > \
        out["frames"]["clean"]["curve_area"]["sigma_u"]
    assert out["frames"]["clean"]["n_chains_drawn"] == 6
    assert len(out["frames"]["clean"]["per_item"]) == 2


# --- 2. the draw -----------------------------------------------------------------

class ScriptedClient:
    """Returns the next scripted chain per FRAME, and a fixed answer at every depth.

    The frame is identified by comparing the prompt against the two prompts the frozen
    instruments build for this item, so the test pins the arm to the real clean and
    hinted prompts rather than to a keyword that happens to appear in one of them. That
    is also the assertion that ``_frame_prompt`` rebuilds the cue pass's own prompt: a
    drifted prompt would land in neither bucket and raise here.

    ``sample_completions`` is what the chain resample calls; ``generate`` is what the
    forced continuations call. Both count their calls, so a test can assert how many
    model calls one item costs.
    """

    def __init__(self, chains_by_frame, prompts, depth_answer="ANSWER: A"):
        self.chains_by_frame = {k: list(v) for k, v in chains_by_frame.items()}
        self.prompts = dict(prompts)
        self.depth_answer = depth_answer
        self.sample_calls = 0
        self.generate_calls = 0
        self.seeds: list[int] = []

    def _frame_of(self, prompt):
        for frame, text in self.prompts.items():
            if prompt == text:
                return frame
        raise AssertionError(
            "the arm asked for a chain on a prompt that is neither this item's clean "
            "prompt nor its hinted prompt"
        )

    def sample_completions(self, prompt, k=1, temperature=0.7, seed=None,
                           num_predict=320):
        self.sample_calls += 1
        self.seeds.append(seed)
        queue = self.chains_by_frame[self._frame_of(prompt)]
        return [queue.pop(0)], "n_parameter"

    def generate(self, prompt, num_predict=24, **kwargs):
        self.generate_calls += 1
        return self.depth_answer


def _item():
    return QAItem(question="Which one?", choices=["w", "x", "y", "z"], answer_index=0)


def _prompts(item, hint="B"):
    from bayes_cot_faithfulness.interventions import clean_prompt, hinted_prompt
    return {"clean": clean_prompt(item),
            "hinted": hinted_prompt(item, hint, strength="strong")}


def _ctx(tmp_path, **kw):
    return ARMS.RunCtx(
        n_choices=4, num_predict=320, out_dir=tmp_path, safe_model="m",
        backend="openai", model="m", curve_cap=10, chain_repeats=3,
        chain_repeat_temperature=0.7, chain_repeat_seed=1000, **kw,
    )


def test_the_arm_redraws_the_chain_and_reads_a_curve_on_each_redraw(tmp_path):
    chains = {
        "clean": ["step one\nstep two\nANSWER: A",
                  "a different route\nANSWER: A",
                  "third way\nmore\nANSWER: A"],
        "hinted": ["h1\nANSWER: B", "h2\nANSWER: B", "h3\nANSWER: B"],
    }
    item = _item()
    client = ScriptedClient(chains, _prompts(item))
    records = [{"item": item, "clean_cot": "x", "clean_answer": "A",
                "hint_label": "B", "hinted_cot": "y", "hinted_answer": "B"}]
    ok, err = ARMS.arm_chain_repeats(client, records, _ctx(tmp_path))
    assert err is None and ok
    block = records[0]["chain_repeats"]
    assert block["r"] == 3 and block["temperature"] == 0.7
    assert set(block["frames"]) == {"clean", "hinted"}
    assert client.sample_calls == 6           # 3 chains x 2 frames
    assert len(set(client.seeds)) == 6        # every draw has its own seed
    clean = block["frames"]["clean"]
    assert [r["repeat"] for r in clean] == [0, 1, 2]
    assert len({r["chain_sha256"] for r in clean}) == 3
    assert all(r["curve_area"] is not None for r in clean)
    assert "chain" in clean[0] and "chain" not in clean[1]  # only repeat 0 keeps text


def test_a_resampled_chain_whose_answer_does_not_parse_is_counted_not_imputed(tmp_path):
    chains = {
        "clean": ["step\nANSWER: A", "the reasoning stops abruptly", "step\nANSWER: A"],
        "hinted": ["h\nANSWER: B", "h\nANSWER: B", "h\nANSWER: B"],
    }
    item = _item()
    client = ScriptedClient(chains, _prompts(item))
    records = [{"item": item, "clean_cot": "x", "clean_answer": "A",
                "hint_label": "B", "hinted_cot": "y", "hinted_answer": "B"}]
    ok, err = ARMS.arm_chain_repeats(client, records, _ctx(tmp_path))
    assert err is None and ok
    clean = records[0]["chain_repeats"]["frames"]["clean"]
    unscorable = [r for r in clean if r["chain_answer"] is None]
    assert len(unscorable) == 1
    assert unscorable[0]["curve_area"] is None
    assert "did not parse" in unscorable[0]["unscorable_reason"]
    summary = ARMS.summarize_chain_repeats(records)
    assert summary["frames"]["clean"]["n_chains_unparsed"] == 1
    assert summary["frames"]["clean"]["n_chains_drawn"] == 3
    assert summary["frames"]["clean"]["curve_area"]["n_items_used_for_sigma_u"] == 1


def test_the_hinted_frame_is_skipped_when_the_cue_pass_never_reached_the_record(tmp_path):
    item = _item()
    client = ScriptedClient({"clean": ["a\nANSWER: A"] * 3, "hinted": []},
                            _prompts(item))
    records = [{"item": item, "clean_cot": "x", "clean_answer": "A"}]
    ok, err = ARMS.arm_chain_repeats(client, records, _ctx(tmp_path))
    assert err is None and ok
    assert set(records[0]["chain_repeats"]["frames"]) == {"clean"}
    assert client.sample_calls == 3


def test_a_banked_block_is_not_redrawn_on_a_resumed_leg(tmp_path):
    item = _item()
    client = ScriptedClient({"clean": [], "hinted": []}, _prompts(item))
    records = [{"item": item, "clean_cot": "x", "clean_answer": "A",
                "hint_label": "B", "chain_repeats": {"r": 3, "frames": {}}}]
    ok, err = ARMS.arm_chain_repeats(client, records, _ctx(tmp_path))
    assert err is None and ok
    assert client.sample_calls == 0


def test_a_backend_that_cannot_sample_refuses_instead_of_returning_greedy_copies(tmp_path):
    class NoSampling:
        def generate(self, prompt, num_predict=24, **kwargs):
            return "ANSWER: A"

    ok, err = ARMS.arm_chain_repeats(NoSampling(), [], _ctx(tmp_path))
    assert ok is False
    assert isinstance(err, RuntimeError)
    assert "sample_completions" in str(err)
