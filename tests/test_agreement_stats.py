"""Tests for experiments/agreement_stats.py and its wiring into score_labels.py.

The pre-registration (section 4) says kappa is reported beside raw agreement, Gwet's AC1
and PABAK, always together. These tests pin the four formulas against values derived here
with exact fractions, and pin the undefined cases (no pairs, both raters on one label).

Every fixture is synthetic and lives in pytest's tmp_path or in memory. Nothing reads a
real sealed label or key file, and main() always gets an explicit --key.

The modules are not importable packages (they live in experiments/), so they are loaded
by file path, like tests/test_score_labels.py does.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ag = _load("agreement_stats_under_test", "agreement_stats.py")
sl = _load("score_labels_for_agreement_test", "score_labels.py")


def _labels(both_yes: int, both_no: int, a_only: int, b_only: int):
    """Build two rater vectors from the four cells of the 2x2 table.

    a_only: rater A yes, rater B no. b_only: rater A no, rater B yes.
    """
    a = [True] * both_yes + [False] * both_no + [True] * a_only + [False] * b_only
    b = [True] * both_yes + [False] * both_no + [False] * a_only + [True] * b_only
    return a, b


def _expected(both_yes: int, both_no: int, a_only: int, b_only: int) -> dict[str, Fraction]:
    """The four statistics from the formulas, in exact fractions."""
    n = both_yes + both_no + a_only + b_only
    p_o = Fraction(both_yes + both_no, n)
    p1 = Fraction(both_yes + a_only, n)
    p2 = Fraction(both_yes + b_only, n)
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    pi = (p1 + p2) / 2
    p_e_g = 2 * pi * (1 - pi)
    return {
        "p_o": p_o,
        "kappa": (p_o - p_e) / (1 - p_e),
        "pabak": 2 * p_o - 1,
        "ac1": (p_o - p_e_g) / (1 - p_e_g),
    }


# --------------------------------------------------------------------------- #
# worked example
# --------------------------------------------------------------------------- #
def test_worked_example_100_items():
    a, b = _labels(both_yes=85, both_no=5, a_only=5, b_only=5)
    s = ag.agreement_stats(a, b)

    # By hand: p_o = 90/100, p1 = p2 = 90/100, p_e = 0.81 + 0.01 = 0.82,
    # kappa = 0.08 / 0.18; pi = 0.9, p_e_g = 2 * 0.9 * 0.1 = 0.18, AC1 = 0.72 / 0.82.
    p_o = Fraction(90, 100)
    p_e = Fraction(82, 100)
    p_e_g = Fraction(18, 100)
    assert s.n == 100
    assert s.p_o == pytest.approx(float(p_o))
    assert s.kappa == pytest.approx(float((p_o - p_e) / (1 - p_e)))  # 4/9
    assert s.pabak == pytest.approx(float(2 * p_o - 1))  # 0.80
    assert s.ac1 == pytest.approx(float((p_o - p_e_g) / (1 - p_e_g)))  # 36/41
    assert s.kappa == pytest.approx(0.4444, abs=1e-4)
    assert s.pabak == pytest.approx(0.80)
    assert s.ac1 == pytest.approx(0.8780, abs=1e-4)
    # the helper agrees with the hand derivation
    exp = _expected(85, 5, 5, 5)
    assert exp["kappa"] == Fraction(4, 9)
    assert exp["ac1"] == Fraction(36, 41)


def test_unequal_marginals_by_hand_with_fractions():
    # 20 items: both yes 6, both no 8, A yes B no 4, A no B yes 2.
    # p_o = 14/20 = 7/10; p1 = 10/20 = 1/2; p2 = 8/20 = 2/5.
    # p_e = 1/2 * 2/5 + 1/2 * 3/5 = 1/2; kappa = (7/10 - 1/2) / (1/2) = 2/5.
    # PABAK = 2 * 7/10 - 1 = 2/5.
    # pi = (1/2 + 2/5) / 2 = 9/20; p_e_g = 2 * 9/20 * 11/20 = 99/200;
    # AC1 = (7/10 - 99/200) / (1 - 99/200) = (41/200) / (101/200) = 41/101.
    a, b = _labels(6, 8, 4, 2)
    s = ag.agreement_stats(a, b)
    assert s.n == 20
    assert s.agree == 14
    assert s.p_o == pytest.approx(float(Fraction(7, 10)))
    assert s.kappa == pytest.approx(float(Fraction(2, 5)))
    assert s.pabak == pytest.approx(float(Fraction(2, 5)))
    assert s.ac1 == pytest.approx(float(Fraction(41, 101)))
    exp = _expected(6, 8, 4, 2)
    assert exp["ac1"] == Fraction(41, 101)
    assert exp["kappa"] == Fraction(2, 5)


@pytest.mark.parametrize(
    "cells",
    [(3, 4, 1, 2), (10, 1, 0, 2), (0, 5, 3, 3), (7, 7, 2, 9), (1, 1, 1, 1)],
)
def test_matches_fraction_reference_on_varied_tables(cells):
    a, b = _labels(*cells)
    s = ag.agreement_stats(a, b)
    exp = _expected(*cells)
    assert s.p_o == pytest.approx(float(exp["p_o"]))
    assert s.kappa == pytest.approx(float(exp["kappa"]))
    assert s.pabak == pytest.approx(float(exp["pabak"]))
    assert s.ac1 == pytest.approx(float(exp["ac1"]))


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_perfect_agreement_with_both_labels_present():
    a, b = _labels(both_yes=6, both_no=4, a_only=0, b_only=0)
    s = ag.agreement_stats(a, b)
    assert (s.p_o, s.kappa, s.pabak, s.ac1) == (1.0, 1.0, 1.0, 1.0)
    assert s.notes == ()


def test_both_raters_all_yes_kappa_undefined_others_defined():
    s = ag.agreement_stats([True] * 7, [True] * 7)
    assert s.n == 7
    assert s.kappa is None
    assert s.notes and "one label" in " ".join(s.notes)
    assert s.p_o == 1.0
    assert s.pabak == 1.0
    assert s.ac1 == 1.0


def test_both_raters_all_no_kappa_undefined_others_defined():
    s = ag.agreement_stats([False] * 5, [False] * 5)
    assert s.kappa is None
    assert s.notes
    assert (s.p_o, s.pabak, s.ac1) == (1.0, 1.0, 1.0)


def test_complete_disagreement_balanced():
    s = ag.agreement_stats([True, True, False, False], [False, False, True, True])
    assert s.p_o == 0.0
    assert s.kappa == pytest.approx(-1.0)
    assert s.pabak == pytest.approx(-1.0)
    assert s.ac1 == pytest.approx(-1.0)


def test_complete_disagreement_one_rater_constant_kappa_is_zero_not_none():
    # A all yes, B all no: p_e = 0, so kappa is defined (0.0). Only p_e == 1 is undefined.
    s = ag.agreement_stats([True] * 4, [False] * 4)
    assert s.kappa == 0.0
    assert s.pabak == -1.0
    assert s.ac1 == pytest.approx(-1.0)
    assert s.notes == ()


def test_none_labels_dropped_pairwise_and_n_counts_pairs():
    a = [None, True, True, True, False, False, None, True, None, False]
    b = [True, True, None, False, False, True, False, True, True, False]
    # surviving pairs: (T,T) (T,F) (F,F) (F,T) (T,T) (F,F)
    clean_a = [True, True, False, False, True, False]
    clean_b = [True, False, False, True, True, False]
    s = ag.agreement_stats(a, b)
    ref = ag.agreement_stats(clean_a, clean_b)
    assert s.n == 6
    assert s == ref


def test_no_paired_items_everything_is_none():
    for a, b in (([], []), ([None, True], [True, None]), ([None], [None])):
        s = ag.agreement_stats(a, b)
        assert s.n == 0
        assert s.agree == 0
        assert s.p_o is None
        assert s.kappa is None
        assert s.pabak is None
        assert s.ac1 is None
        assert s.notes


def test_unequal_lengths_are_refused():
    # zip would silently score a prefix; the module must refuse instead.
    with pytest.raises(ValueError, match="must be paired"):
        ag.agreement_stats([True, True], [True, False, True])


def test_four_statistics_come_back_together_in_one_call():
    a, b = _labels(4, 4, 1, 1)
    s = ag.agreement_stats(a, b)
    assert {"n", "p_o", "kappa", "pabak", "ac1"} <= set(s.__dataclass_fields__)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.kappa = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# the scorer prints all four together
# --------------------------------------------------------------------------- #
def _write_labeled(path: Path, mentions: list[str]) -> Path:
    cols = [sl.ITEM_ID, sl.Q_MENTIONS, sl.Q_SUPPORTS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, cell in enumerate(mentions, start=1):
            w.writerow({sl.ITEM_ID: f"it{i}", sl.Q_MENTIONS: cell, sl.Q_SUPPORTS: "yes"})
    return path


def _write_key(path: Path, n: int) -> Path:
    path.write_text(json.dumps({
        f"it{i}": {"heuristic_acknowledged_hint": True, "heuristic_followed_hint": False}
        for i in range(1, n + 1)
    }))
    return path


def _run_main(monkeypatch, capsys, files: list[Path], key: Path) -> str:
    monkeypatch.setattr(sys, "argv", [
        "score_labels.py", "--labeled", *map(str, files), "--key", str(key),
    ])
    assert sl.main() == 0
    return capsys.readouterr().out


def test_scorer_prints_all_four_statistics_and_n(tmp_path, monkeypatch, capsys):
    alice = _write_labeled(tmp_path / "labeled_alice.csv", ["yes", "no", "no", "yes", "yes"])
    bob = _write_labeled(tmp_path / "labeled_bob.csv", ["yes", "no", "no", "yes", "no"])
    out = _run_main(monkeypatch, capsys, [alice, bob], _write_key(tmp_path / "k.json", 5))

    exp = _expected(both_yes=2, both_no=2, a_only=1, b_only=0)
    line = next(ln for ln in out.splitlines() if "n=5" in ln)
    assert "raw agreement 4/5" in line
    assert "Cohen kappa" in line
    assert "Gwet AC1" in line
    assert "PABAK" in line
    assert f"Cohen kappa {float(exp['kappa']):.2f}" in line
    assert f"Gwet AC1 {float(exp['ac1']):.2f}" in line
    assert f"PABAK {float(exp['pabak']):.2f}" in line
    # the old line is untouched
    assert "alice vs bob: kappa=0.62  raw agreement 4/5" in out


def test_scorer_says_kappa_undefined_when_both_raters_used_one_label(
    tmp_path, monkeypatch, capsys
):
    alice = _write_labeled(tmp_path / "labeled_alice.csv", ["yes", "yes", "yes"])
    bob = _write_labeled(tmp_path / "labeled_bob.csv", ["yes", "yes", "yes"])
    out = _run_main(monkeypatch, capsys, [alice, bob], _write_key(tmp_path / "k.json", 3))

    line = next(ln for ln in out.splitlines() if "n=3" in ln)
    assert "kappa undefined (both raters used one label)" in line
    assert "Gwet AC1 1.00" in line
    assert "PABAK 1.00" in line
    assert "raw agreement 3/3" in line
    # the legacy line keeps its numeric output
    assert "alice vs bob: kappa=1.00  raw agreement 3/3" in out


def test_scorer_no_paired_items_prints_n_zero(tmp_path, monkeypatch, capsys):
    alice = _write_labeled(tmp_path / "labeled_alice.csv", ["yes", ""])
    bob = _write_labeled(tmp_path / "labeled_bob.csv", ["", "no"])
    out = _run_main(monkeypatch, capsys, [alice, bob], _write_key(tmp_path / "k.json", 2))
    line = next(ln for ln in out.splitlines() if "n=0" in ln)
    assert "no paired items" in line
    assert "alice vs bob: kappa=n/a" in out
