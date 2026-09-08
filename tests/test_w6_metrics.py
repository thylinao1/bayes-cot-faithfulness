"""Reproduce the numbers docs/external_validity.md section 4 reports, from the
mirrored votes in experiments/results/w6-external/.

This is a reproducibility check, not a units test of analyze_jury's arithmetic
(test_jury_gate.py and friends cover that): it runs the same analyzer the W6 lane
ran on the same mirrored votes.jsonl and asserts the numbers it produces are the
numbers written into the doc. If either drifts, this fails.

Skips when the mirrored votes are absent (a fresh clone before this lane's commit,
or a checkout of a branch that never merged it), rather than failing on missing
fixture data.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from experiments.external import analyze_jury, compare_jury_regex

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOTES_ROOT = ROOT / "experiments" / "results" / "w6-external" / "gemma-3-27b-it"
VOTES_FILE = VOTES_ROOT / "faithcot" / "no-cue" / "votes.jsonl"

pytestmark = pytest.mark.skipif(
    not VOTES_FILE.exists(),
    reason=f"mirrored votes not present at {VOTES_FILE}",
)


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("w6_metrics") / "jury_instrument_report.json"
    rc = analyze_jury.main(["--votes-root", str(VOTES_ROOT), "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text())


def _pj(report: dict) -> dict:
    return report["per_judge"]["gemma-3-27b-it"]


# --- shape of the run, pinned so a silently different vote file cannot pass -----


def test_one_judge_scored_this_corpus(report: dict) -> None:
    assert report["judges"] == ["gemma-3-27b-it"]


def test_vote_counts_match_score_summary(report: dict) -> None:
    # score_summary.json (also mirrored) records votes_written 7220 with 7199 ok,
    # 21 malformed, 0 error; the analyzer reads the same votes.jsonl independently.
    assert report["n_votes_total"] == 7220
    assert report["n_votes_run0"] == 6020  # run 0 covers all 1505 items x 4 questions


def test_leave_one_out_not_computed_with_one_judge(report: dict) -> None:
    loo = report["leave_one_judge_out_Q2_faithcot"]
    assert loo["status"] == "NOT COMPUTED"
    assert "one judge" in loo["why"] or "1 scored" in loo["why"]


# --- malformed rate, one judge across all four questions ------------------------


def test_malformed_rate(report: dict) -> None:
    # analyze_jury's availability block is computed over run 0 only (one vote per
    # item x question, 6,020 = 1,505 items x 4 questions); the audit slice's two
    # extra runs are covered separately by the test-retest check below. Two more
    # malformed votes land in those reruns (21 of 7,220 total, per score_summary.json,
    # also mirrored), which is a different denominator and not this metric.
    avail = _pj(report)["availability"]
    bad = sum(v["status_counts"].get("malformed", 0) for v in avail.values())
    total = sum(sum(v["status_counts"].values()) for v in avail.values())
    assert (bad, total) == (19, 6020)


# --- Q1a/b/c specificity on FaithCoT (no item carries a cue, so recall is None) --


@pytest.mark.parametrize(
    ("question", "expected_num", "expected_den"),
    [
        ("Q1a", 1168, 1253),
        ("Q1b", 959, 1253),
        ("Q1c", 567, 1243),
    ],
)
def test_q1_specificity_matches_doc(report: dict, question: str, expected_num: int, expected_den: int) -> None:
    block = _pj(report)["faithcot"][question]
    assert block["recall"] is None
    spec = block["specificity"]
    assert (spec["numerator"], spec["denominator"]) == (expected_num, expected_den)


# --- Q2 against the human unfaithfulness label -----------------------------------


def test_q2_all_usable_matches_doc(report: dict) -> None:
    q2 = _pj(report)["faithcot"]["Q2"]["all_usable"]
    assert q2["recall"]["numerator"] == 26
    assert q2["recall"]["denominator"] == 365
    assert q2["specificity"]["numerator"] == 880
    assert q2["specificity"]["denominator"] == 891
    assert q2["precision"]["numerator"] == 26
    assert q2["precision"]["denominator"] == 37
    assert q2["f1"] == pytest.approx(0.1294, abs=1e-4)


# --- test-retest, the seeded 150-item audit slice, three runs -------------------


def test_test_retest_on_audit_slice(report: dict) -> None:
    tr = _pj(report)["test_retest_on_audit_slice"]
    assert (tr["numerator"], tr["denominator"]) == (599, 599)
    assert tr["rate"] == 1.0


# --- wave-1 cell: jury Q1 agreement with the frozen regex (section 5) -----------


@pytest.mark.parametrize(
    ("question", "p_no_mention_num", "p_no_mention_den", "agreement_num"),
    [
        ("Q1a", 158, 249, 244),
        ("Q1b", 88, 249, 174),
        ("Q1c", 76, 249, 162),
    ],
)
def test_wave1_matches_doc(
    report: dict, question: str, p_no_mention_num: int, p_no_mention_den: int, agreement_num: int
) -> None:
    block = _pj(report)["wave1"][question]
    pm = block["p_no_mention_given_followed"]
    assert (pm["numerator"], pm["denominator"]) == (p_no_mention_num, p_no_mention_den)
    agree = block["agreement_with_frozen_regex"]
    assert agree["numerator"] == agreement_num
    assert agree["denominator"] == p_no_mention_den


# --- jury vs. frozen regex, same FaithCoT items, per Q1 file (section 4) --------


@pytest.fixture(scope="module")
def same_items_report(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("w6_compare") / "jury_vs_regex_same_items.json"
    rc = compare_jury_regex.main(["--votes-root", str(VOTES_ROOT), "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text())


@pytest.mark.parametrize(
    ("question", "n_items", "jury_num", "regex_num"),
    [
        ("Q1a", 1253, 1168, 1219),
        ("Q1b", 1253, 959, 1219),
        ("Q1c", 1243, 567, 1209),
    ],
)
def test_jury_vs_regex_same_items_matches_doc(
    same_items_report: dict, question: str, n_items: int, jury_num: int, regex_num: int
) -> None:
    block = same_items_report["per_question"][question]
    assert block["n_jury_scored"] == n_items
    assert block["n_missing_from_regex_predictions"] == 0
    assert block["jury_specificity"]["numerator"] == jury_num
    assert block["jury_specificity"]["denominator"] == n_items
    assert block["regex_specificity_same_items"]["numerator"] == regex_num
    assert block["regex_specificity_same_items"]["denominator"] == n_items
