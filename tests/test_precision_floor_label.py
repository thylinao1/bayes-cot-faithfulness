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


# --------------------------------------------------------------------------- #
# The label reaching the report a reader actually opens.
# --------------------------------------------------------------------------- #
def _fit(precision=None, **kw):
    f = {"job_id": "1", "tree": {"plan_commit": "abc"}}
    if precision is not None:
        f["precision"] = precision
    f.update(kw)
    return f


def _report():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import cells18_fits_report as rep
    return rep


def test_the_report_says_so_when_every_cell_reaches_the_floor():
    rep = _report()
    clean = cf.precision_block(summary(1396, {"direct": 1396}))
    fits = {("qwen3-8b", "arc_challenge", "professor"): _fit(precision=clean)}
    line = rep._precision_line(fits)
    assert "Every one of the 1 cells reaches the precision floor" in line
    assert "0.0996" in line
    rows = rep._precision_table(fits)
    assert any("reaches the floor" in r for r in rows)


def test_the_report_counts_and_names_the_underpowered_cells():
    rep = _report()
    fits = {
        ("qwen3-8b", "arc_challenge", "professor"): _fit(
            precision=cf.precision_block(summary(1396, {"direct": 1396}))),
        ("olmo-3-7b-think", "logiqa2", "metadata"): _fit(
            precision=cf.precision_block(summary(321, {"direct": 321}))),
    }
    line = rep._precision_line(fits)
    assert "1 of the 2 cells are labelled UNDERPOWERED" in line
    assert "not dropped" in line and "cannot resolve one" in line
    rows = "\n".join(rep._precision_table(fits))
    assert "321" in rows and "NO" in rows


def test_the_report_does_not_crash_on_a_fit_written_before_the_label_existed():
    """Every fit.json under experiments/results/cells24-fits predates the precision
    block. The report must say the block is absent rather than raise or, worse, print
    a cell as if it had cleared a floor nobody checked."""
    rep = _report()
    fits = {("qwen3-8b", "arc_challenge", "professor"): _fit()}
    rows = rep._precision_table(fits)
    assert any("no precision block" in r for r in rows)
    assert isinstance(rep._precision_line(fits), str)


def test_a_cell_with_no_precision_block_is_never_reported_as_clearing_the_floor():
    """The first draft of this section said "Every one of the 24 cells reaches the
    precision floor" over a table in which every row read "no precision block". A cell
    that was never checked is not a cell that passed, and printing a clean bill of
    health over an empty check is the exact failure this report exists to prevent."""
    rep = _report()
    fits = {
        ("qwen3-8b", "arc_challenge", "professor"): _fit(),
        ("qwen3-8b", "aqua_rat", "professor"): _fit(),
    }
    line = rep._precision_line(fits)
    assert "reaches the precision floor" not in line
    assert "2 of the 2 cells carry no precision block" in line
    assert "never applied" in line


def test_a_mixed_report_counts_checked_and_unchecked_separately():
    rep = _report()
    fits = {
        ("qwen3-8b", "arc_challenge", "professor"): _fit(
            precision=cf.precision_block(summary(1396, {"direct": 1396}))),
        ("olmo-3-7b-think", "logiqa2", "metadata"): _fit(
            precision=cf.precision_block(summary(321, {"direct": 321}))),
        ("phi-4-reasoning", "aqua_rat", "metadata"): _fit(),
    }
    line = rep._precision_line(fits)
    assert "1 of the 3 cells carry no precision block" in line
    assert "Of the 2 cells that do carry it, 1 are labelled UNDERPOWERED" in line
