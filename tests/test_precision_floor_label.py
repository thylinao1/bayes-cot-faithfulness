"""The precision floor of 01-SIZING I.3, and the label the frozen pre-registration
attaches to a cell that misses it.

The rule this pins is not a lane choice. `01-SIZING.md` I.3 makes 350 clean-correct items
the n at which a cell's NIE posterior half-width lands inside the 0.10 bar (0.0996 at 350,
0.1027 at 300), and `PREREGISTRATION_phase2_arms.md`, which is frozen, says three times
over what happens to a quantity that misses its n: it is reported with its interval and
labelled underpowered. So the label must never drop a cell and must never let one pass
unmarked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import cells18_fits as cf


def summary(n_clean, arms, enabled=None):
    return {
        "n_clean_correct": n_clean,
        "enabled_arms": list(arms) if enabled is None else list(enabled),
        "arms": {name: ({} if n is None else {"n": n}) for name, n in arms.items()},
    }


def test_the_floor_is_the_sizing_number_and_says_where_it_comes_from():
    assert cf.PRECISION_FLOOR == 350
    assert "0.0996" in cf.FLOOR_SOURCE and "0.1027" in cf.FLOOR_SOURCE
    assert "01-SIZING" in cf.FLOOR_SOURCE
    # the floor is an absolute count, not a share of whatever n was entered
    assert "absolute" in cf.FLOOR_SOURCE


def test_a_cell_clear_of_the_floor_on_every_arm_is_not_labelled():
    block = cf.precision_block(summary(1396, {"direct": 1396, "twostep": 1396}))
    assert block["underpowered"] is False
    assert block["label"] == ""
    assert block["clean_correct_clears_floor"] is True
    assert block["arms_below_floor"] == {}


def test_a_cell_below_the_floor_is_labelled_and_says_by_how_much():
    block = cf.precision_block(summary(322, {"direct": 322}))
    assert block["underpowered"] is True
    assert "clean-correct 322 < 350" in block["label"]
    assert block["n_clean_correct"] == 322


def test_the_boundary_is_inclusive_at_the_floor():
    """350 is the n that reaches the bar, so 350 passes and 349 does not."""
    assert cf.precision_block(summary(350, {"direct": 350}))["underpowered"] is False
    assert cf.precision_block(summary(349, {"direct": 349}))["underpowered"] is True


def test_clean_accuracy_alone_does_not_clear_a_cell_whose_arms_are_thin():
    """The wave-1 Phi-4 shape: 1,197 clean-correct, comfortably above the floor, with
    0 scorable rows on two arms and 4 on a third because the forced-answer continuation
    was spent opening a reasoning block. Reading accuracy alone called this cell fine."""
    block = cf.precision_block(
        summary(1197, {"placebo": 961, "direct": 0, "filler": 0, "twostep": 4})
    )
    assert block["clean_correct_clears_floor"] is True
    assert block["underpowered"] is True
    assert block["arms_below_floor"] == {"direct": 0, "filler": 0, "twostep": 4}
    assert "placebo" not in block["arms_below_floor"]
    assert "direct 0" in block["label"] and "twostep 4" in block["label"]


def test_an_enabled_arm_with_no_denominator_sets_the_label_rather_than_passing_quietly():
    block = cf.precision_block(summary(1396, {"replay": None}))
    assert block["underpowered"] is True
    assert block["enabled_arms_with_no_denominator"] == ["replay"]
    assert "no denominator" in block["label"]


def test_an_arm_that_is_not_enabled_is_not_checked():
    s = summary(1396, {"direct": 1396, "curves": 4}, enabled=["direct"])
    block = cf.precision_block(s)
    assert block["underpowered"] is False
    assert "curves" not in block["arm_denominators"]


def test_a_missing_clean_correct_count_is_labelled_not_assumed_fine():
    block = cf.precision_block({"enabled_arms": [], "arms": {}})
    assert block["underpowered"] is True
    assert block["n_clean_correct"] is None
    assert "no clean-correct count" in block["label"]


@pytest.mark.parametrize("phrase", ["reported", "labeled underpowered", "not dropped"])
def test_the_block_carries_the_frozen_rule_it_implements(phrase):
    block = cf.precision_block(summary(100, {"direct": 100}))
    assert phrase in block["rule_source"] or phrase in cf.UNDERPOWERED_RULE


def test_the_label_never_removes_a_cell_it_only_marks_one():
    """The frozen rule reports an underpowered cell with its interval. The block is a
    label and carries no instruction to drop anything, which this pins by construction:
    nothing in it is an exclusion flag."""
    block = cf.precision_block(summary(10, {"direct": 10}))
    assert "drop" not in block["label"].lower()
    assert "exclude" not in block["label"].lower()
    assert set(block) == {
        "floor", "floor_source", "rule_source", "n_clean_correct",
        "clean_correct_clears_floor", "arm_denominators", "arms_below_floor",
        "enabled_arms_with_no_denominator", "underpowered", "label",
    }


# --------------------------------------------------------------------------- #
# The label travelling into the model row, which pools cells.
# --------------------------------------------------------------------------- #
def _cell(tmp, name, summary_name="arms_summary.json", summary=None):
    d = tmp / name
    d.mkdir(parents=True)
    (d / "transcripts.jsonl").write_text("")
    if summary is not None:
        import json as _json
        (d / summary_name).write_text(_json.dumps(summary))
    return d


def test_a_cells_summary_is_found_under_either_name(tmp_path):
    plain = _cell(tmp_path, "plain", summary=summary(1396, {"direct": 1396}))
    assert cf._summary_path_for_cell(plain).name == "arms_summary.json"
    slugged = _cell(
        tmp_path, "slugged", summary_name="arms_summary_qwen3-8b.json",
        summary=summary(1396, {"direct": 1396}),
    )
    assert cf._summary_path_for_cell(slugged).name == "arms_summary_qwen3-8b.json"
    assert cf._summary_path_for_cell(_cell(tmp_path, "bare")) is None


def test_a_cell_with_no_summary_is_labelled_rather_than_assumed_fine(tmp_path):
    block = cf._cell_precision(_cell(tmp_path, "nosummary"))
    assert block["underpowered"] is True
    assert block["summary_file"] is None
    assert "no arms summary" in block["label"]


def test_an_unreadable_summary_is_labelled_rather_than_raising(tmp_path):
    d = _cell(tmp_path, "broken")
    (d / "arms_summary.json").write_text("{not json")
    block = cf._cell_precision(d)
    assert block["underpowered"] is True
    assert block["summary_file"] == "arms_summary.json"


def test_a_healthy_cell_reads_clean_through_the_model_row_path(tmp_path):
    d = _cell(tmp_path, "healthy", summary=summary(1396, {"direct": 1396}))
    block = cf._cell_precision(d)
    assert block["underpowered"] is False
    assert block["summary_file"] == "arms_summary.json"
