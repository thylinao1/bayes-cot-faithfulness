"""Unit tests for experiments/score_labels.py, the golden-set inter-rater instrument.

score_labels.py computes the study's headline inter-rater number (Cohen's kappa between
human annotators) and compares the heuristic auditor to the human majority. It ships the
credibility unlock for the golden set, yet had no test; this file is that test.

RATER BLINDING -- ABSOLUTE. Nothing here reads, opens, or references any real sealed
label/key artifact (results/labelset/, results/labeling_key.json, results/labeling_sheet.*,
labeled_fable5*.csv, fable5_agreement_stats.txt). Every fixture is SYNTHETIC data written
into pytest's tmp_path, and main() is always invoked with an explicit --key pointing at a
synthetic key so it can never fall back to the real results/labeling_key.json default.

The script is not an importable package module (it lives in experiments/ and imports its
sibling labeling_columns via a sys.path insert), so it is loaded by file path exactly like
tests/test_additive_arms.py loads the additive-arms runner. cohen_kappa is validated
against sklearn.metrics.cohen_kappa_score, the reference implementation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import sys
import warnings
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "score_labels.py"


def _load_score_labels():
    spec = importlib.util.spec_from_file_location("score_labels_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own `sys.path.insert(0, HERE)` + sibling import
    # resolves cleanly, mirroring how the runner tests load a path-only experiment module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sl = _load_score_labels()


# --------------------------------------------------------------------------- #
# Synthetic-fixture writers (all data is constructed here, never read from disk)
# --------------------------------------------------------------------------- #
def _write_labeled(path: Path, rows: list[tuple[str, str, str]], *, bom: bool = True) -> Path:
    """Write a synthetic filled label CSV with the REAL column headers.

    ``rows`` is (item_id, mentions_cell, supports_cell). ``bom=True`` writes a UTF-8 BOM
    (encoding utf-8-sig) to mimic a sheet saved by Excel, which load_labeled must tolerate.
    """
    cols = [sl.ITEM_ID, sl.Q_MENTIONS, sl.Q_SUPPORTS]
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for item_id, mentions, supports in rows:
            w.writerow({sl.ITEM_ID: item_id, sl.Q_MENTIONS: mentions, sl.Q_SUPPORTS: supports})
    return path


def _write_key(path: Path, entries: dict[str, dict[str, bool]]) -> Path:
    """Write a synthetic labeling_key.json: {item_id: {heuristic_acknowledged_hint,
    heuristic_followed_hint}} -- the exact shape score_labels.main consumes."""
    path.write_text(json.dumps(entries))
    return path


# --------------------------------------------------------------------------- #
# 1. cohen_kappa -- the load-bearing function -- validated against sklearn
# --------------------------------------------------------------------------- #
def _deterministic_pairs() -> list[tuple[list[bool], list[bool]]]:
    """A fixed, seeded battery of rater-pair boolean vectors with varied agreement."""
    pairs: list[tuple[list[bool], list[bool]]] = [
        # perfect agreement on a mixed vector -> kappa 1.0
        ([True, False, True, False, True, False], [True, False, True, False, True, False]),
        # perfect systematic disagreement -> kappa -1.0
        ([True, True, False, False, True, False], [False, False, True, True, False, True]),
        # one rater constant, the other mixed -> sklearn defines this (kappa 0.0)
        ([True] * 6, [True, False, True, False, True, True]),
        ([False] * 6, [True, False, True, False, True, True]),
    ]
    rng = random.Random(20260718)
    for _ in range(40):
        n = rng.randint(10, 40)
        pa = rng.choice([0.3, 0.5, 0.7])
        a = [rng.random() < pa for _ in range(n)]
        # b is a is copied then each cell flipped with prob q, sweeping q from
        # strongly correlated (small q) to strongly anti-correlated (large q).
        q = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        b = [(not a[i]) if rng.random() < q else a[i] for i in range(n)]
        pairs.append((a, b))
    return pairs


def test_cohen_kappa_matches_sklearn_on_many_cases():
    checked = 0
    for a, b in _deterministic_pairs():
        ours = sl.cohen_kappa(a, b)
        ref = cohen_kappa_score(a, b)
        if math.isnan(ref):
            # sklearn returns nan only when both raters are a single shared class
            # (pe == 1.0). The function's pinned convention is 1.0 when po == 1.0,
            # which is the only reachable pe == 1.0 case (both all-True / all-False).
            assert a == b
            assert ours == 1.0
            continue
        assert ours is not None
        assert abs(ours - ref) < 1e-9, (a, b, ours, ref)
        checked += 1
    # the "many cases where sklearn is defined" requirement, made structural
    assert checked >= 30


def test_cohen_kappa_perfect_agreement_is_one():
    assert sl.cohen_kappa([True, False, True, True, False], [True, False, True, True, False]) == 1.0


def test_cohen_kappa_known_negative_case():
    a = [True, True, False, False, True, False]
    b = [False, False, True, True, False, True]
    k = sl.cohen_kappa(a, b)
    assert k == -1.0
    assert abs(k - cohen_kappa_score(a, b)) < 1e-9


def test_cohen_kappa_pe_one_degenerate_convention():
    # pe == 1.0 requires both raters constant on the SAME class, which forces po == 1.0;
    # the function returns 1.0 in that (only reachable) branch of its "1.0 if po==1.0 else
    # 0.0" convention. sklearn returns nan here (single shared label), so we assert the
    # convention directly and silence sklearn's expected undefined-metric warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isnan(cohen_kappa_score([True] * 4, [True] * 4))
        assert math.isnan(cohen_kappa_score([False] * 4, [False] * 4))
    assert sl.cohen_kappa([True] * 4, [True] * 4) == 1.0
    assert sl.cohen_kappa([False] * 4, [False] * 4) == 1.0


def test_cohen_kappa_none_entries_skipped_pairwise():
    a_clean = [True, True, False, False, True, False]
    b_clean = [True, False, False, True, True, False]
    k_clean = sl.cohen_kappa(a_clean, b_clean)
    assert abs(k_clean - cohen_kappa_score(a_clean, b_clean)) < 1e-9

    # Interleave positions where at least one rater is None; the surviving (both non-None)
    # pairs, in order, are exactly the clean pairs above, so the kappa must be identical.
    a_mixed = [None, True, True, True, False, False, None, True, None, False]
    b_mixed = [True, True, None, False, False, True, False, True, True, False]
    k_mixed = sl.cohen_kappa(a_mixed, b_mixed)
    assert k_mixed is not None
    assert abs(k_mixed - k_clean) < 1e-12


def test_cohen_kappa_all_pairs_none_returns_none():
    assert sl.cohen_kappa([None, True, None], [False, None, True]) is None
    assert sl.cohen_kappa([], []) is None


# --------------------------------------------------------------------------- #
# 2. majority
# --------------------------------------------------------------------------- #
def test_majority_tie_returns_none():
    assert sl.majority([True, False]) is None
    assert sl.majority([True, True, False, False]) is None


def test_majority_all_none_returns_none():
    assert sl.majority([None, None, None]) is None
    assert sl.majority([]) is None


def test_majority_clear_true_and_false():
    assert sl.majority([True, True, False]) is True
    assert sl.majority([False, False, True]) is False


def test_majority_single_vote_is_that_vote():
    assert sl.majority([True]) is True
    assert sl.majority([False]) is False
    # None entries are dropped first, so one real vote among Nones still decides
    assert sl.majority([None, True, None]) is True


# --------------------------------------------------------------------------- #
# 3. _to_bool
# --------------------------------------------------------------------------- #
def test_to_bool_yes_tokens():
    for tok in ("y", "yes", "true", "1"):
        assert sl._to_bool(tok) is True


def test_to_bool_no_tokens():
    for tok in ("n", "no", "false", "0"):
        assert sl._to_bool(tok) is False


def test_to_bool_blank_whitespace_and_none():
    assert sl._to_bool("") is None
    assert sl._to_bool("   ") is None
    assert sl._to_bool(None) is None


def test_to_bool_unknown_token_is_none():
    assert sl._to_bool("maybe") is None
    assert sl._to_bool("2") is None


def test_to_bool_case_insensitive_and_trims_whitespace():
    assert sl._to_bool("YES") is True
    assert sl._to_bool("  Yes  ") is True
    assert sl._to_bool("\tTRUE\n") is True
    assert sl._to_bool(" No ") is False
    assert sl._to_bool("FALSE") is False


# --------------------------------------------------------------------------- #
# 4. load_labeled round-trip, including an Excel-style UTF-8 BOM
# --------------------------------------------------------------------------- #
def test_load_labeled_round_trips_bom_csv(tmp_path):
    path = _write_labeled(
        tmp_path / "labeled_alice.csv",
        [
            ("it1", "yes", "yes"),
            ("it2", "no", "no"),
            ("it3", "TRUE", "1"),      # case-insensitive + "1" both parse True
            ("it4", " no ", "0"),      # surrounding whitespace + "0" parse False
            ("it5", "", "maybe"),      # blank / unknown parse None
        ],
        bom=True,
    )
    out = sl.load_labeled(path)
    assert out["it1"] == {"mentions": True, "supports": True}
    assert out["it2"] == {"mentions": False, "supports": False}
    assert out["it3"] == {"mentions": True, "supports": True}
    assert out["it4"] == {"mentions": False, "supports": False}
    assert out["it5"] == {"mentions": None, "supports": None}
    # the BOM must not leak into the item-id key (load_labeled opens utf-8-sig)
    assert set(out) == {"it1", "it2", "it3", "it4", "it5"}


# --------------------------------------------------------------------------- #
# 5. main() end-to-end on SYNTHETIC data
# --------------------------------------------------------------------------- #
def test_main_two_annotators_kappa_and_heuristic_agreement(tmp_path, monkeypatch, capsys):
    # items it1..it5; mentions (Q1): alice/bob agree on it1-it4, disagree on it5 (a tie)
    alice = _write_labeled(tmp_path / "labeled_alice.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "yes"), ("it3", "no", "yes"),
        ("it4", "yes", "yes"), ("it5", "yes", "yes"),
    ])
    bob = _write_labeled(tmp_path / "labeled_bob.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "yes"), ("it3", "no", "yes"),
        ("it4", "yes", "yes"), ("it5", "no", "yes"),
    ])
    # heuristic key: one deliberate FALSE POSITIVE (it2: heuristic acknowledged, humans
    # say NOT -- the dangerous direction the script calls out) and one FN (it4).
    key = _write_key(tmp_path / "labeling_key.json", {
        "it1": {"heuristic_acknowledged_hint": True, "heuristic_followed_hint": True},
        "it2": {"heuristic_acknowledged_hint": True, "heuristic_followed_hint": False},   # FP
        "it3": {"heuristic_acknowledged_hint": False, "heuristic_followed_hint": False},
        "it4": {"heuristic_acknowledged_hint": False, "heuristic_followed_hint": True},   # FN
        "it5": {"heuristic_acknowledged_hint": True, "heuristic_followed_hint": False},   # tie -> skipped
    })

    monkeypatch.setattr(sys, "argv",
                        ["score_labels.py", "--labeled", str(alice), str(bob), "--key", str(key)])
    rc = sl.main()
    out = capsys.readouterr().out
    assert rc == 0

    # inter-rater kappa: mentions alice=[T,F,F,T,T] vs bob=[T,F,F,T,F].
    a_ment = [True, False, False, True, True]
    b_ment = [True, False, False, True, False]
    k_expected = sl.cohen_kappa(a_ment, b_ment)
    assert abs(k_expected - cohen_kappa_score(a_ment, b_ment)) < 1e-9
    assert f"kappa={k_expected:.2f}" in out           # the printed number is the real kappa
    assert "kappa=0.62" in out                         # and its rounded value
    assert "alice vs bob:" in out
    assert "raw agreement 4/5" in out

    # heuristic vs human majority on mentions: TP=1(it1), FP=1(it2), TN=1(it3), FN=1(it4),
    # it5 is a tie (majority None) and is skipped -> both=4, agreement 2/4, prec/rec 0.50.
    assert "agreement 2/4" in out
    assert "precision 0.50" in out
    assert "recall 0.50" in out
    assert "(heuristic FP=1, FN=1)" in out

    # objective follow count from the key (heuristic_followed_hint True on it1, it4)
    assert "followed the bait (parsed answer == bait): 2/5" in out


def test_main_three_annotators_majority_and_false_positive(tmp_path, monkeypatch, capsys):
    a1 = _write_labeled(tmp_path / "labeled_a1.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "yes"), ("it3", "yes", "yes"),
    ])
    a2 = _write_labeled(tmp_path / "labeled_a2.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "yes"), ("it3", "no", "yes"),
    ])
    a3 = _write_labeled(tmp_path / "labeled_a3.csv", [
        ("it1", "no", "yes"), ("it2", "no", "yes"), ("it3", "no", "yes"),
    ])
    # majority(mentions): it1 [T,T,F]->T, it2 [F,F,F]->F, it3 [T,F,F]->F
    key = _write_key(tmp_path / "labeling_key.json", {
        "it1": {"heuristic_acknowledged_hint": True, "heuristic_followed_hint": True},
        "it2": {"heuristic_acknowledged_hint": True, "heuristic_followed_hint": False},   # FP vs majority
        "it3": {"heuristic_acknowledged_hint": False, "heuristic_followed_hint": True},
    })

    monkeypatch.setattr(sys, "argv", [
        "score_labels.py", "--labeled", str(a1), str(a2), str(a3), "--key", str(key),
    ])
    rc = sl.main()
    out = capsys.readouterr().out
    assert rc == 0

    # three annotators -> three pairwise kappa comparisons
    assert out.count("kappa=") == 3
    for pair in ("a1 vs a2:", "a1 vs a3:", "a2 vs a3:"):
        assert pair in out

    # heuristic vs majority: TP=1(it1), FP=1(it2), TN=1(it3), FN=0 -> agreement 2/3,
    # precision 0.50, recall 1.00.
    assert "agreement 2/3" in out
    assert "precision 0.50" in out
    assert "recall 1.00" in out
    assert "(heuristic FP=1, FN=0)" in out
    assert "followed the bait (parsed answer == bait): 2/3" in out


def test_main_single_annotator_prints_need_two_branch(tmp_path, monkeypatch, capsys):
    solo = _write_labeled(tmp_path / "labeled_solo.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "yes"),
    ])
    key = _write_key(tmp_path / "labeling_key.json", {
        "it1": {"heuristic_acknowledged_hint": True, "heuristic_followed_hint": True},
        "it2": {"heuristic_acknowledged_hint": False, "heuristic_followed_hint": False},
    })

    monkeypatch.setattr(sys, "argv",
                        ["score_labels.py", "--labeled", str(solo), "--key", str(key)])
    rc = sl.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "(need >=2 annotators for inter-rater kappa)" in out
    assert "kappa=" not in out   # the pairwise branch never runs with one annotator


def test_main_missing_key_returns_one(tmp_path, monkeypatch, capsys):
    solo = _write_labeled(tmp_path / "labeled_solo.csv", [("it1", "yes", "yes")])
    monkeypatch.setattr(sys, "argv", [
        "score_labels.py", "--labeled", str(solo),
        "--key", str(tmp_path / "does_not_exist.json"),
    ])
    rc = sl.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "missing key" in out
