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
        assert set(row) == {"label", "n", "mde", "ci_upper_on_observed", "newcombe"}
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
