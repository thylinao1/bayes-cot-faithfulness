"""Tests for the T10/A5/A6 guardrail-audit upgrades (experiments/07_guardrail_audit.py).

experiments/07_guardrail_audit.py is a numeric-prefixed script, not a normal
importable package member, so it is loaded here via an explicit sys.path /
importlib bootstrap rather than a plain ``import`` statement. The module's own
``RESULTS`` global is monkeypatched to a temp directory per test so nothing here
reads or writes the real experiments/results/ tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "07_guardrail_audit.py"

sys.path.insert(0, str(REPO / "experiments"))  # sibling-script bootstrap, mirrors 05/07's own pattern

_spec = importlib.util.spec_from_file_location("guardrail_audit_07", MODULE_PATH)
audit = importlib.util.module_from_spec(_spec)
sys.modules["guardrail_audit_07"] = audit
_spec.loader.exec_module(audit)


NEW_SCHEMA_SUMMARY = {
    "backend": "ollama",
    "model": "new-schema-model",
    "n_items": 20,
    "n_clean_correct": 10,
    "follow_rate": 0.4,
    "silent_unfaithful_rate": 0.3,
    "silent_given_follow": 0.75,
    "neutral_change_rate": 0.2,
    "n_stable": 8,
    "require_stable": False,
    "stable_follow_rate": None,
    "stable_silent_given_follow": None,
    "breakdown_rho_star": 0.5,
    "verdict": "REVIEW",
    "hinted_accuracy": 0.6,
    "collateral_rate": 0.1,
    "n_followed_hint": 4,
    "n_silent_unfaithful": 3,
    "n_neutral_changed": 2,
    "attrition": {
        "n_entered": 20,
        "n_failed_generation": 0,
        "n_unparseable_clean": 1,
        # entered = failed generation + clean incorrect + clean correct.
        "n_clean_incorrect": 10,
        "n_clean_correct": 10,
    },
}

# Matches the shape of the real pre-upgrade summaries at experiments/results/*.json.
OLD_SCHEMA_SUMMARY = {
    "model": "old-schema-model",
    "n_items": 24,
    "n_clean_correct": 16,
    "follow_rate": 0.125,
    "silent_unfaithful_rate": 0.125,
    "silent_given_follow": 1.0,
    "neutral_change_rate": 0.4375,
    "breakdown_rho_star": 0.78,
    "verdict": "REVIEW",
}


# A synthetic arms summary in the shape experiments/08_additive_arms.py writes, with
# every rate-bearing block plus an n == 0 block (placebo), a None-rate block (twostep),
# a rateless block (curves), and an n_unscorable > 10% block (replay.hinted).
ARMS_SUMMARY = {
    "backend": "ollama",
    "model": "arms-model",
    "n_items": 40,
    "n_clean_correct": 30,
    "cue_kind": "stated-hint:strong",
    "enabled_arms": ["replay", "transplant", "direct", "placebo", "twostep", "filler",
                     "curves"],
    "attrition": {"n_entered": 40},
    "arms": {
        "replay": {
            "clean": {"n": 20, "n_drifted": 2, "drift_rate": 0.1, "n_unscorable": 1},
            # 3 unscorable of 15 entered = 20% > 10% -> unscorable_flag True
            "hinted": {"n": 12, "n_drifted": 6, "drift_rate": 0.5, "n_unscorable": 3},
        },
        "transplant": {
            "forward": {"n": 18, "n_carryover": 9, "carryover_rate": 0.5, "n_unscorable": 0},
            # exactly 2 of 20 = 10% -> NOT strictly above -> unscorable_flag False
            "reverse": {"n": 18, "n_carryover": 3, "carryover_rate": 1 / 6, "n_unscorable": 2},
            "note": "read against the replay floor",
        },
        "direct": {
            "n": 25, "n_unscorable": 5, "clean_accuracy": 1.0,
            "direct_accuracy": {"n": 25, "n_correct": 20, "rate": 0.8},
            "with_without_cot_agreement": {"n": 25, "n_agree": 22, "rate": 0.88},
            "commitment_split": {},
        },
        "placebo": {  # n == 0 -> row skipped
            "n": 0, "n_unscorable": 0, "n_changed": 0, "change_rate": None,
            "n_follow_would_be_hint": 0, "placebo_follow_rate": None,
        },
        "twostep": {  # None rate -> row skipped
            "n": 10, "n_unscorable": 0, "n_twostep_follow": 0, "twostep_follow_rate": None,
            "n_singleshot_follow": 0, "singleshot_follow_rate": None,
        },
        "filler": {
            "n": 4, "n_unscorable": 0, "n_filler_match": 1, "filler_match_rate": 0.25,
            "replay_floor": None,
        },
        "curves": {  # bears no rate -> never emits a row
            "clean": {"n": 5, "mean_curve_area": 0.5},
            "hinted": {"n": 5, "mean_curve_area": 0.4},
        },
    },
    "status": "exploratory Phase-2 arms; no verdict",
}


def _write_summary(dir_path: Path, name: str, payload: dict) -> Path:
    path = dir_path / f"control_summary_{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


@pytest.fixture
def results_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp results/ tree with one new-schema and one old-schema summary, wired
    into the audit module's RESULTS global for the duration of the test."""
    monkeypatch.setattr(audit, "RESULTS", tmp_path)
    _write_summary(tmp_path, "new", NEW_SCHEMA_SUMMARY)
    _write_summary(tmp_path, "old", OLD_SCHEMA_SUMMARY)
    return tmp_path


# --------------------------------------------------------------------------- #
# load_run
# --------------------------------------------------------------------------- #
def test_load_run_tolerates_old_and_new_schema(results_tree: Path) -> None:
    # Act
    new_facts = audit.load_run(results_tree / "control_summary_new.json")
    old_facts = audit.load_run(results_tree / "control_summary_old.json")

    # Assert: new schema populates the upgraded fields.
    assert new_facts.hinted_accuracy == pytest.approx(0.6)
    assert new_facts.collateral_rate == pytest.approx(0.1)
    assert new_facts.n_neutral_changed == 2
    assert new_facts.attrition == NEW_SCHEMA_SUMMARY["attrition"]

    # Assert: old schema is tolerated, upgraded fields default to None.
    assert old_facts.hinted_accuracy is None
    assert old_facts.collateral_rate is None
    assert old_facts.n_neutral_changed is None
    assert old_facts.attrition is None
    assert old_facts.n_clean_correct == 16  # pre-existing fields still parse fine


# --------------------------------------------------------------------------- #
# report_run
# --------------------------------------------------------------------------- #
def test_report_run_prints_attrition_and_newcombe_for_new_schema(
    results_tree: Path, capsys: pytest.CaptureFixture
) -> None:
    # Arrange
    facts = audit.load_run(results_tree / "control_summary_new.json")

    # Act
    audit.report_run(facts)
    out = capsys.readouterr().out

    # Assert
    assert "ATTRITION" in out
    assert "named categories, never silently dropped" in out
    assert "collateral rate" in out
    assert "hinted-arm accuracy" in out
    assert "logged independently of hint adoption" in out
    assert "cue effect above noise floor" in out


def test_report_run_skips_upgraded_rows_for_old_schema(
    results_tree: Path, capsys: pytest.CaptureFixture
) -> None:
    # Arrange
    facts = audit.load_run(results_tree / "control_summary_old.json")

    # Act
    audit.report_run(facts)
    out = capsys.readouterr().out

    # Assert: nothing from the new schema is fabricated for an old summary.
    assert "ATTRITION" not in out
    assert "cue effect above noise floor" not in out
    assert "collateral rate" not in out


# --------------------------------------------------------------------------- #
# power artifact + exit code, end to end
# --------------------------------------------------------------------------- #
def test_power_artifact_written_with_expected_keys(
    results_tree: Path, capsys: pytest.CaptureFixture
) -> None:
    # Act
    exit_code = audit.main([])
    capsys.readouterr()  # drain stdout, not under test here

    # Assert
    artifact_path = results_tree / "guardrail_power_artifacts.json"
    assert artifact_path.exists()
    rows = json.loads(artifact_path.read_text())
    assert {r["label"] for r in rows} == {
        "control_summary_new.json",
        "control_summary_old.json",
    }
    for row in rows:
        assert set(row) == {"kind", "label", "n", "mde", "ci_upper_on_observed", "newcombe"}
        assert row["kind"] == "control"  # control rows carry the distinguishing kind field
    by_label = {r["label"]: r for r in rows}
    assert by_label["control_summary_new.json"]["newcombe"] is not None
    assert set(by_label["control_summary_new.json"]["newcombe"]) == {
        "diff", "lower", "upper", "conf",
    }
    assert by_label["control_summary_old.json"]["newcombe"] is None
    assert exit_code == 0  # no hard violation, default invocation


def test_main_exit_code_strict_vs_default_on_underpowered_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # Arrange: a tiny n makes minimum_detectable_rate(n) far above MDE_TARGET.
    monkeypatch.setattr(audit, "RESULTS", tmp_path)
    tiny = dict(NEW_SCHEMA_SUMMARY)
    tiny["n_clean_correct"] = 3
    tiny["n_followed_hint"] = 0
    _write_summary(tmp_path, "tiny", tiny)

    # Act / Assert: default invocation stays exit 0 (pre-existing behavior preserved).
    assert audit.main([]) == 0
    capsys.readouterr()

    # Act / Assert: --strict fails loud on the same underpowered-only run.
    assert audit.main(["--strict"]) == 1
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# decide_exit_code (pure function, no argv or file I/O needed)
# --------------------------------------------------------------------------- #
def test_decide_exit_code_hard_violation_always_wins() -> None:
    assert audit.decide_exit_code(True, True, True) == 2
    assert audit.decide_exit_code(True, False, False) == 2


def test_decide_exit_code_strict_flags_underpowered() -> None:
    assert audit.decide_exit_code(False, True, True) == 1


def test_decide_exit_code_default_ignores_underpowered() -> None:
    assert audit.decide_exit_code(False, True, False) == 0
    assert audit.decide_exit_code(False, False, False) == 0
    assert audit.decide_exit_code(False, False, True) == 0


# --------------------------------------------------------------------------- #
# arms loader (pure, on a synthetic arms-summary dict)
# --------------------------------------------------------------------------- #
def test_arms_rows_from_summary_emits_one_row_per_rate_bearing_subblock() -> None:
    rows = audit.arms_rows_from_summary(ARMS_SUMMARY, "arms_summary_arms-model.json")
    labels = {r["label"] for r in rows}
    base = "arms_summary_arms-model.json"
    assert labels == {
        f"{base}:replay.clean",
        f"{base}:replay.hinted",
        f"{base}:transplant.forward",
        f"{base}:transplant.reverse",
        f"{base}:direct.accuracy",
        f"{base}:direct.agreement",
        f"{base}:filler",
    }
    for r in rows:
        assert r["kind"] == "arms"  # distinguishing field vs control rows
        assert set(r) >= {
            "kind", "label", "n", "rate", "mde", "ci_upper_on_observed",
            "powered", "n_unscorable", "unscorable_flag",
        }


def test_arms_rows_skip_zero_n_none_rate_and_rateless_blocks() -> None:
    labels = {r["label"] for r in audit.arms_rows_from_summary(ARMS_SUMMARY, "f.json")}
    assert "f.json:placebo" not in labels   # n == 0
    assert "f.json:twostep" not in labels    # None rate
    assert not any("curves" in lb for lb in labels)  # curves bears no rate


def test_arms_rows_unscorable_flag_applies_ten_percent_rule() -> None:
    rows = {r["label"]: r for r in audit.arms_rows_from_summary(ARMS_SUMMARY, "f.json")}
    # hinted replay: 3 unscorable of 15 entered (20%) -> flagged
    assert rows["f.json:replay.hinted"]["unscorable_flag"] is True
    # reverse transplant: exactly 2 of 20 (10%) -> NOT strictly above -> not flagged
    assert rows["f.json:transplant.reverse"]["unscorable_flag"] is False
    # clean replay: 1 of 21 -> well under 10%
    assert rows["f.json:replay.clean"]["unscorable_flag"] is False


def test_arms_rows_ci_and_mde_reuse_guardrail_functions() -> None:
    from bayes_cot_faithfulness.guardrails import (
        minimum_detectable_rate,
        proportion_ci_upper,
    )

    rows = {r["label"]: r for r in audit.arms_rows_from_summary(ARMS_SUMMARY, "f.json")}
    r = rows["f.json:replay.clean"]  # n = 20, rate = 0.1 -> observed count k = 2
    assert r["n"] == 20
    assert r["rate"] == 0.1
    assert r["ci_upper_on_observed"] == proportion_ci_upper(2, 20)
    assert r["mde"] == minimum_detectable_rate(20)
    assert r["powered"] == (minimum_detectable_rate(20) <= audit.MDE_TARGET)


def test_arms_rows_from_summary_tolerates_missing_arms_key() -> None:
    assert audit.arms_rows_from_summary({}, "f.json") == []
    assert audit.arms_rows_from_summary({"arms": {}}, "f.json") == []


# --------------------------------------------------------------------------- #
# main() ingests arms summaries into the same artifact and exit-code contract
# --------------------------------------------------------------------------- #
def test_main_ingests_arms_summary_alongside_control_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(audit, "RESULTS", tmp_path)
    # A control summary must be present or main() reports "nothing found" and exits early.
    _write_summary(tmp_path, "ctrl", NEW_SCHEMA_SUMMARY)
    (tmp_path / "arms_summary_arms-model.json").write_text(json.dumps(ARMS_SUMMARY, indent=2))

    exit_code = audit.main([])  # default invocation ignores underpowered rows
    capsys.readouterr()
    assert exit_code == 0

    rows = json.loads((tmp_path / "guardrail_power_artifacts.json").read_text())
    assert {r["kind"] for r in rows} == {"control", "arms"}
    arms_labels = {r["label"] for r in rows if r["kind"] == "arms"}
    assert "arms_summary_arms-model.json:replay.hinted" in arms_labels
    # control rows keep their exact payload shape (plus the additive kind field)
    control = [r for r in rows if r["kind"] == "control"]
    assert control
    for r in control:
        assert set(r) == {"kind", "label", "n", "mde", "ci_upper_on_observed", "newcombe"}


def test_main_strict_fails_on_underpowered_arm_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(audit, "RESULTS", tmp_path)
    # A control run large enough to be well-powered on its own, so only the arm row can
    # trip --strict.
    powered_ctrl = dict(NEW_SCHEMA_SUMMARY)
    powered_ctrl["n_clean_correct"] = 400
    powered_ctrl["n_followed_hint"] = 4
    _write_summary(tmp_path, "big", powered_ctrl)
    # An arms summary whose one emitted row is a tiny-n (underpowered) filler match.
    tiny_arms = {"arms": {"filler": {"n": 3, "n_unscorable": 0, "n_filler_match": 1,
                                     "filler_match_rate": 1 / 3, "replay_floor": None}}}
    (tmp_path / "arms_summary_tiny.json").write_text(json.dumps(tiny_arms))

    assert audit.main([]) == 0            # default ignores underpowered rows
    capsys.readouterr()
    assert audit.main(["--strict"]) == 1  # the underpowered arm row fails loud
    capsys.readouterr()
