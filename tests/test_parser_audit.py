"""Unit tests for the pure pieces of experiments/09_parser_audit.py.

The audit script is not an importable package module (it lives in experiments/ and
its name starts with a digit), so it is loaded by file path. Only the deterministic,
offline helpers are exercised here: the constructed battery, the battery scorer, the
TSV row builder, the transcript sampler, and the letter tabulator. Nothing in these
tests calls a model or the network, and importing the module does not write the TSV
(that only happens inside main()).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "09_parser_audit.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("parser_audit_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-type resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_audit_module()


# --------------------------------------------------------------------------- #
# Battery shape and coverage
# --------------------------------------------------------------------------- #
def test_battery_has_at_least_40_wellformed_cases():
    assert len(mod.BATTERY) >= 40
    assert all(mod.is_well_formed_case(case) for case in mod.BATTERY)


def test_is_well_formed_case_rejects_malformed():
    assert not mod.is_well_formed_case({"text": "x", "n_choices": 4})  # missing keys
    assert not mod.is_well_formed_case(
        {"text": "x", "n_choices": 1, "expected": "A", "category": "c"}  # n_choices < 2
    )
    assert not mod.is_well_formed_case(
        {"text": "x", "n_choices": 4, "expected": "Z", "category": "c"}  # bad label
    )
    assert not mod.is_well_formed_case("not a dict")


def test_battery_covers_the_required_families():
    categories = {case["category"] for case in mod.BATTERY}
    joined = " ".join(categories)
    for needle in ("canonical", "connector", "trailing-option", "revised",
                   "distractor", "truncated", "multi-answer", "lowercase",
                   "punctuation", "beyond-range"):
        assert needle in joined, needle
    # At least one case must commit no answer (expected None), and one must exercise
    # an out-of-range option letter.
    assert any(case["expected"] is None for case in mod.BATTERY)


# --------------------------------------------------------------------------- #
# Battery scoring
# --------------------------------------------------------------------------- #
def test_score_battery_counts_mismatches_on_synthetic_subbatch():
    cases = [
        {"text": "Answer: (A)", "n_choices": 4, "expected": "A", "category": "hit"},
        {"text": "Answer: (B)", "n_choices": 4, "expected": "C", "category": "miss"},
        {"text": "", "n_choices": 4, "expected": None, "category": "hit-none"},
        {"text": "Answer: (E)", "n_choices": 4, "expected": "E", "category": "miss-range"},
    ]
    score = mod.score_battery(cases)
    assert score.total == 4
    assert score.mismatches == 2
    assert score.error_rate == 0.5
    assert len(score.mismatch_notes) == 2
    # The reported bound is the exact one-sided Clopper-Pearson upper limit.
    from bayes_cot_faithfulness.guardrails import proportion_ci_upper
    assert score.ci_upper == proportion_ci_upper(2, 4)


def test_score_battery_empty_is_safe():
    score = mod.score_battery([])
    assert score.total == 0
    assert score.mismatches == 0
    assert score.error_rate == 0.0
    assert score.ci_upper == 1.0


def test_real_battery_error_rate_is_bounded():
    score = mod.score_battery(mod.BATTERY)
    # Sanity, not a threshold gate: the parser should read the well-formed majority.
    assert 0.0 <= score.error_rate < 0.5
    assert score.ci_upper >= score.error_rate


# --------------------------------------------------------------------------- #
# TSV row construction and escaping
# --------------------------------------------------------------------------- #
def test_escape_cell_escapes_tabs_newlines_and_backslashes():
    out = mod.escape_cell("a\tb\nc\r\\d")
    assert "\t" not in out
    assert "\n" not in out
    assert "\r" not in out
    assert "\\t" in out and "\\n" in out and "\\r" in out and "\\\\d" in out


def test_format_tsv_row_has_six_columns_and_no_raw_control_chars():
    row = mod.AuditRow("f.json", 5, "clean", "B", "line1\nline2\tend")
    line = mod.format_tsv_row(row)
    cols = line.split("\t")
    assert len(cols) == 6
    assert cols[0] == "f.json"
    assert cols[1] == "5"
    assert cols[2] == "clean"
    assert cols[3] == "B"
    assert "\\n" in cols[4] and "\\t" in cols[4]
    assert cols[5] == ""  # human_verdict stays blank for the auditor
    assert "\n" not in line


def test_format_tsv_row_blanks_a_none_answer():
    row = mod.AuditRow("f.json", 0, "hinted", None, "no committed answer")
    cols = mod.format_tsv_row(row).split("\t")
    assert cols[3] == ""


def test_cot_tail_keeps_only_the_last_n_chars():
    long = "x" * 300
    assert mod.cot_tail(long, 240) == "x" * 240
    assert mod.cot_tail("short", 240) == "short"
    assert mod.cot_tail("", 240) == ""


# --------------------------------------------------------------------------- #
# Flattening and sampling
# --------------------------------------------------------------------------- #
def _synthetic_records(n: int) -> list[dict]:
    return [
        {
            "choices": ["w", "x", "y", "z"],
            "clean_cot": f"reasoning {i}\nAnswer: (C)",
            "hinted_cot": f"reasoning {i}\nAnswer: (A)",
            "hint_label": "A",
        }
        for i in range(n)
    ]


def test_iter_audit_rows_emits_two_arms_per_record():
    rows = mod.iter_audit_rows(_synthetic_records(3), "syn.json")
    assert len(rows) == 6
    assert [r.arm for r in rows] == ["clean", "hinted"] * 3
    clean = [r for r in rows if r.arm == "clean"]
    hinted = [r for r in rows if r.arm == "hinted"]
    assert all(r.extracted_answer == "C" for r in clean)
    assert all(r.extracted_answer == "A" for r in hinted)


def test_sample_audit_rows_is_deterministic_and_capped():
    rows = mod.iter_audit_rows(_synthetic_records(40), "syn.json")  # 80 arm-rows
    first = mod.sample_audit_rows(rows, cap=30, seed=123)
    second = mod.sample_audit_rows(rows, cap=30, seed=123)
    assert first == second
    assert len(first) == 30
    # A different seed gives a different subset (with overwhelming probability here).
    other = mod.sample_audit_rows(rows, cap=30, seed=999)
    assert other != first


def test_sample_audit_rows_returns_all_when_under_cap():
    rows = mod.iter_audit_rows(_synthetic_records(5), "syn.json")  # 10 arm-rows
    sampled = mod.sample_audit_rows(rows, cap=30, seed=1)
    assert len(sampled) == len(rows) == 10


# --------------------------------------------------------------------------- #
# Letter tabulation and IO contract
# --------------------------------------------------------------------------- #
def test_letter_counts_handles_empty_and_none():
    assert mod.letter_counts([]) == {}
    assert mod.letter_counts(["A", "A", "B", None]) == {"A": 2, "B": 1, mod.NONE_KEY: 1}


def test_find_transcripts_is_root_only_and_skips_labelset():
    for path in mod.find_transcripts():
        assert path.parent == mod.RESULTS
        assert "labelset" not in str(path)
