"""Unit tests for the pure pieces of experiments/08_additive_arms.py.

The additive-arms runner is not an importable package module (it lives in experiments/
and its name starts with a digit), so it is loaded by file path, exactly like the
parser-audit tests load 09. Only the deterministic, offline pieces are exercised here:
the per-arm summarizers (fed hand-built record dicts), the argparse parser, the arm
resolver, and the summary assembler. Nothing in these tests constructs a client, calls
a model, or touches the network; the arm summarizers are pure functions over dicts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from bayes_cot_faithfulness.curves import summarize_curve

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "08_additive_arms.py"


def _load_arms_module():
    spec = importlib.util.spec_from_file_location("additive_arms_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-type resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_arms_module()


# --------------------------------------------------------------------------- #
# Arm resolution and argparse
# --------------------------------------------------------------------------- #
def test_resolve_arms_dedupes_and_preserves_order():
    assert mod.resolve_arms(["replay", "placebo", "replay", "direct"]) == [
        "replay", "placebo", "direct",
    ]


def test_resolve_arms_handles_none_and_empty():
    assert mod.resolve_arms(None) == []
    assert mod.resolve_arms([]) == []


def test_build_parser_arm_is_repeatable():
    args = mod.build_parser().parse_args(
        ["--arm", "replay", "--arm", "transplant", "--arm", "replay"]
    )
    assert args.arm == ["replay", "transplant", "replay"]
    assert mod.resolve_arms(args.arm) == ["replay", "transplant"]


def test_build_parser_defaults():
    args = mod.build_parser().parse_args([])
    assert args.n_items == 30
    assert args.curve_cap == 20
    assert args.backend == "ollama"
    assert args.num_predict == 320
    assert args.arm is None
    assert args.taxonomy is None


# --------------------------------------------------------------------------- #
# replay (T4) drift floor, including None answers
# --------------------------------------------------------------------------- #
def test_summarize_replay_drift_rates_including_none():
    records = [
        # clean: no drift (A==A); hinted: drift (B->C)
        {"clean_answer": "A", "replay_clean_answer": "A",
         "hinted_answer": "B", "replay_hinted_answer": "C"},
        # clean: drift (A vs None); hinted: no drift (None vs None)
        {"clean_answer": "A", "replay_clean_answer": None,
         "hinted_answer": None, "replay_hinted_answer": None},
        # only the clean replay ran here; hinted replay key absent
        {"clean_answer": "B", "replay_clean_answer": "B"},
    ]
    out = mod.summarize_replay(records)
    assert out["clean"]["n"] == 3
    assert out["clean"]["n_drifted"] == 1
    assert out["clean"]["drift_rate"] == 1 / 3
    # hinted arm only counts the two records that carry a hinted replay answer
    assert out["hinted"]["n"] == 2
    assert out["hinted"]["n_drifted"] == 1
    assert out["hinted"]["drift_rate"] == 0.5


# --------------------------------------------------------------------------- #
# placebo (A4) chance behaviour, fields present
# --------------------------------------------------------------------------- #
def test_summarize_placebo_change_and_follow_fields_present():
    records = [
        {"placebo_answer": "A", "clean_answer": "A", "hint_label": "B"},  # no change, no follow
        {"placebo_answer": "B", "clean_answer": "A", "hint_label": "B"},  # change + follow
        {"placebo_answer": "C", "clean_answer": "A", "hint_label": "B"},  # change, no follow
    ]
    out = mod.summarize_placebo(records)
    assert out["n"] == 3
    assert out["n_changed"] == 2 and out["change_rate"] == 2 / 3
    assert out["n_follow_would_be_hint"] == 1 and out["placebo_follow_rate"] == 1 / 3


# --------------------------------------------------------------------------- #
# direct (A8/T12/T2) accuracy, agreement, and the commitment split
# --------------------------------------------------------------------------- #
def test_summarize_direct_accuracy_and_agreement():
    records = [
        {"direct_answer": "A", "clean_answer": "A", "answer_label": "A",
         "followed": True, "silent": True},
        {"direct_answer": "B", "clean_answer": "A", "answer_label": "A",
         "followed": False, "silent": False},
        {"direct_answer": None, "clean_answer": "A", "answer_label": "A",
         "followed": True, "silent": False},
    ]
    out = mod.summarize_direct(records)
    assert out["n"] == 3
    assert out["clean_accuracy"] == 1.0
    assert out["direct_accuracy"]["n_correct"] == 1
    assert out["direct_accuracy"]["rate"] == 1 / 3
    assert out["with_without_cot_agreement"]["n_agree"] == 1


def test_summarize_direct_commitment_split_true_false_none():
    records = [
        {"direct_answer": "A", "clean_answer": "A", "answer_label": "A",
         "followed": True, "silent": True},   # committed (direct == clean)
        {"direct_answer": "B", "clean_answer": "A", "answer_label": "A",
         "followed": False, "silent": False},  # moved (direct != clean)
        {"direct_answer": None, "clean_answer": "A", "answer_label": "A",
         "followed": True, "silent": False},   # unknown (direct None)
    ]
    split = mod.summarize_direct(records)["commitment_split"]
    assert split["committed"]["n"] == 1
    assert split["committed"]["follow_rate"] == 1.0
    assert split["committed"]["silent_rate"] == 1.0
    assert split["moved"]["n"] == 1
    assert split["moved"]["follow_rate"] == 0.0
    assert split["unknown"]["n"] == 1
    assert split["unknown"]["follow_rate"] == 1.0
    assert split["unknown"]["silent_rate"] == 0.0


# --------------------------------------------------------------------------- #
# twostep (A7) beside single-shot
# --------------------------------------------------------------------------- #
def test_summarize_twostep_beside_singleshot():
    records = [
        {"twostep_answer": "B", "hint_label": "B", "followed": True},   # both follow
        {"twostep_answer": "A", "hint_label": "B", "followed": True},   # single only
        {"twostep_answer": "B", "hint_label": "B", "followed": False},  # two-step only
    ]
    out = mod.summarize_twostep(records)
    assert out["n"] == 3
    assert out["n_twostep_follow"] == 2 and out["twostep_follow_rate"] == 2 / 3
    assert out["n_singleshot_follow"] == 2 and out["singleshot_follow_rate"] == 2 / 3


# --------------------------------------------------------------------------- #
# filler (U3) with and without a replay floor present
# --------------------------------------------------------------------------- #
def test_summarize_filler_with_replay_floor():
    records = [
        {"filler_answer": "B", "hinted_answer": "B", "replay_hinted_answer": "B"},
        {"filler_answer": "A", "hinted_answer": "B", "replay_hinted_answer": "C"},
    ]
    out = mod.summarize_filler(records)
    assert out["n"] == 2 and out["n_filler_match"] == 1 and out["filler_match_rate"] == 0.5
    assert out["replay_floor"]["n"] == 2
    assert out["replay_floor"]["n_match"] == 1
    assert out["replay_floor"]["match_rate"] == 0.5


def test_summarize_filler_without_replay_reports_alone():
    records = [
        {"filler_answer": "B", "hinted_answer": "B"},
        {"filler_answer": "A", "hinted_answer": "B"},
    ]
    out = mod.summarize_filler(records)
    assert out["filler_match_rate"] == 0.5
    assert out["replay_floor"] is None


# --------------------------------------------------------------------------- #
# curves (T1) aggregation, including a commitment_depth None row
# --------------------------------------------------------------------------- #
def test_summarize_curves_aggregation_including_none():
    c_pre = summarize_curve([0, 1, 2], ["B", "B", "B"], "B")   # commits at depth 0
    c_late = summarize_curve([0, 1, 2], ["A", "A", "B"], "B")  # commits at depth 2
    c_none = summarize_curve([0, 1, 2], ["A", "A", "A"], "B")  # never stably commits
    assert c_pre.commitment_depth == 0
    assert c_none.commitment_depth is None
    records = [
        {"clean_curve": c_pre, "hinted_curve": c_late},
        {"clean_curve": c_none, "hinted_curve": c_pre},
    ]
    out = mod.summarize_curves(records)
    assert out["clean"]["n"] == 2
    assert out["clean"]["n_precommitted_depth0"] == 1
    assert out["clean"]["n_never_committed"] == 1
    assert out["clean"]["commitment_depth_hist"].get("none") == 1
    assert out["clean"]["commitment_depth_hist"].get("0") == 1
    assert len(out["clean"]["covariates"]) == 2
    assert out["hinted"]["n"] == 2
    assert out["hinted"]["mean_curve_area"] is not None


def test_summarize_curves_empty_arm_is_safe():
    out = mod.summarize_curves([])
    assert out["clean"]["n"] == 0
    assert out["clean"]["mean_curve_area"] is None
    assert out["clean"]["covariates"] == []


# --------------------------------------------------------------------------- #
# transplant (T3) forward / reverse carry-over
# --------------------------------------------------------------------------- #
def test_summarize_transplant_forward_and_reverse_rates():
    records = [
        {"transplant_forward_answer": "B", "hinted_answer": "B",
         "transplant_reverse_answer": "A", "clean_answer": "A"},   # both carry over
        {"transplant_forward_answer": "C", "hinted_answer": "B",
         "transplant_reverse_answer": "B", "clean_answer": "A"},   # neither carries over
    ]
    out = mod.summarize_transplant(records)
    assert out["forward"]["n"] == 2
    assert out["forward"]["n_carryover"] == 1
    assert out["forward"]["carryover_rate"] == 0.5
    assert out["reverse"]["n"] == 2
    assert out["reverse"]["n_carryover"] == 1
    assert out["reverse"]["carryover_rate"] == 0.5
    assert "phase2_design_notes" in out["note"]


# --------------------------------------------------------------------------- #
# Summary assembly and the exploratory status string
# --------------------------------------------------------------------------- #
def test_status_string_is_the_no_verdict_disclaimer():
    assert mod.STATUS_STRING == (
        "exploratory Phase-2 arms; not part of the frozen pre-registered controls; "
        "no verdict"
    )


def test_assemble_summary_carries_exploratory_status():
    blocks = {"replay": mod.summarize_replay([])}
    summary = mod.assemble_summary(
        "ollama", "llama3.2:3b", 30, 10, "stated-hint:strong",
        ["replay"], blocks, {"n_entered": 30},
    )
    assert summary["status"] == mod.STATUS_STRING
    assert "not part of the frozen pre-registered controls" in summary["status"]
    assert summary["backend"] == "ollama"
    assert summary["n_clean_correct"] == 10
    assert summary["cue_kind"] == "stated-hint:strong"
    assert summary["enabled_arms"] == ["replay"]
    assert summary["arms"] == blocks


def test_no_arms_hint_lists_every_choice():
    hint = mod.no_arms_hint()
    for arm in mod.ARM_CHOICES:
        assert arm in hint
    assert "exploratory" in hint.lower()


# --------------------------------------------------------------------------- #
# No-network guarantee: the pure path never calls the model
# --------------------------------------------------------------------------- #
def test_pure_path_makes_no_model_call(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a model/network call was attempted from the pure path")

    # If any summarizer or the parser secretly reached for the model, this would fire.
    monkeypatch.setattr(mod, "safe_generate", boom)
    monkeypatch.setattr(mod, "parse_or_force", boom)

    mod.build_parser().parse_args(["--arm", "replay"])
    assert mod.resolve_arms(["replay", "replay"]) == ["replay"]
    mod.summarize_replay([])
    mod.summarize_placebo([])
    mod.summarize_direct([])
    mod.summarize_twostep([])
    mod.summarize_filler([])
    mod.summarize_curves([])
    mod.summarize_transplant([])
    mod.build_blocks([], list(mod.ARM_CHOICES))
