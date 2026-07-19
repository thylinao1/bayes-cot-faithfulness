"""Unit tests for experiments/adjudicate_labels.py, the golden-set adjudication tooling.

adjudicate_labels.py is an ADDITIVE companion to score_labels.py: it surfaces the
item+question disagreements between human raters into a worklist for a human to adjudicate,
and reports a SUPPLEMENTARY Fleiss' kappa for the >= 3-rater case. It never resolves a
disagreement (the `adjudicated` column is always blank) and the model is never a rater.

RATER BLINDING -- ABSOLUTE. Nothing here reads, opens, or references any real sealed
label/key artifact (results/labelset/, results/labeling_key.json, results/labeling_sheet.*,
labeled_fable5*.csv, fable5_agreement_stats.txt). Every fixture is SYNTHETIC data written
into pytest's tmp_path. The tool derives item ids from the labeled CSVs themselves and has
no --key argument, so it can never touch the sealed key even by accident.

The module is loaded by file path exactly like tests/test_score_labels.py loads score_labels
(it lives in experiments/ and imports its siblings via a sys.path insert).

fleiss_kappa is validated against ``_ref_fleiss`` below -- an independently coded stdlib
reference built from agreeing rater PAIRS (math.comb), a different code path from the
implementation's sum-of-squares form -- and anchored to two textbook values: the published
Fleiss' kappa worked example (kappa = 0.210) and a fully hand-worked binary case (kappa =
1/3). STDLIB-ONLY on purpose: the CI dev environment installs no scikit-learn, so a test that
imported it would pass locally and break CI (the failure mode this project has already hit).
"""

from __future__ import annotations

import csv
import importlib.util
import os
import random
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "adjudicate_labels.py"


def _load_adjudicate():
    spec = importlib.util.spec_from_file_location("adjudicate_labels_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own sys.path.insert(0, HERE) + sibling imports
    # (labeling_columns, score_labels) resolve, mirroring the score_labels loader test.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adj = _load_adjudicate()
# labeling_columns + score_labels are now in sys.modules (adj imported them); reuse the real
# sheet headers to write synthetic label CSVs, without a second import statement (keeps ruff
# E402-clean and proves the sibling imports resolved).
_LC = sys.modules["labeling_columns"]
ITEM_ID, Q_MENTIONS, Q_SUPPORTS = _LC.ITEM_ID, _LC.Q_MENTIONS, _LC.Q_SUPPORTS


# --------------------------------------------------------------------------- #
# Synthetic-fixture writers (all data constructed here, never read from disk)
# --------------------------------------------------------------------------- #
def _write_labeled(path: Path, rows: list[tuple[str, str, str]], *, bom: bool = True) -> Path:
    """Write a synthetic filled label CSV with the REAL sheet headers.

    ``rows`` is (item_id, mentions_cell, supports_cell). ``bom=True`` writes a UTF-8 BOM to
    mimic a sheet saved by Excel, which the reused load_labeled must tolerate.
    """
    cols = [ITEM_ID, Q_MENTIONS, Q_SUPPORTS]
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for item_id, mentions, supports in rows:
            w.writerow({ITEM_ID: item_id, Q_MENTIONS: mentions, Q_SUPPORTS: supports})
    return path


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


# --------------------------------------------------------------------------- #
# Independent Fleiss oracle -- a different code path from the implementation
# --------------------------------------------------------------------------- #
def _ref_fleiss(rows: list[list[int]]) -> float | None:
    """Fleiss' kappa via agreeing rater PAIRS and EXACT rational arithmetic.

    Algebraically identical to the implementation but a genuinely different numeric path on
    BOTH halves: P_bar from math.comb pair counts (not sum-of-squares) and P_e from
    imperatively accumulated column totals in exact fractions.Fraction arithmetic (not a
    float generator expression), so a shared coding slip in either half cannot cancel out.
    The textbook anchors below additionally pin the absolute values. Same n<2 / empty /
    pe==1 conventions as the implementation.
    """
    if not rows:
        return None
    n = sum(rows[0])
    if n < 2:
        return None
    total = len(rows) * n
    cn = comb(n, 2)
    p_bar = Fraction(sum(sum(comb(c, 2) for c in r) for r in rows), cn * len(rows))
    col = [0] * len(rows[0])
    for r in rows:
        for j, c in enumerate(r):
            col[j] += c
    p_e = sum(Fraction(c, total) ** 2 for c in col)
    if p_e == 1:
        return 1.0 if p_bar == 1 else 0.0
    return float((p_bar - p_e) / (1 - p_e))


# Published Fleiss' kappa worked example: N=10 subjects, n=14 raters, k=5 categories -> 0.210.
WIKIPEDIA_FLEISS = [
    [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
]

# Hand-worked binary case: 4 items, 3 raters, categories [no, yes].
#   P_i = (sum n_ij^2 - n) / (n(n-1)), n=3:
#     [0,3] -> (9-3)/6 = 1.0 ; [1,2] -> (5-3)/6 = 1/3 ; [3,0] -> 1.0 ; [2,1] -> 1/3
#   P_bar = (1 + 1/3 + 1 + 1/3)/4 = 2/3 ; column totals no=6, yes=6 over N*n=12 -> p=0.5,0.5
#   P_e = 0.25 + 0.25 = 0.5 ; kappa = (2/3 - 1/2) / (1 - 1/2) = (1/6)/(1/2) = 1/3.
HAND_BINARY = [[0, 3], [1, 2], [3, 0], [2, 1]]


def _random_matrices() -> list[list[list[int]]]:
    rng = random.Random(20260718)
    mats: list[list[list[int]]] = [
        [[0, 3], [3, 0], [0, 3]],           # every item unanimous, categories mixed -> kappa 1.0
        [[2, 2], [2, 2]],                    # maximal within-item disagreement
    ]
    for _ in range(160):
        n_items = rng.randint(2, 12)
        n_raters = rng.randint(2, 8)
        k = rng.randint(2, 5)
        rows = []
        for _ in range(n_items):
            r = [0] * k
            for _ in range(n_raters):
                r[rng.randrange(k)] += 1
            rows.append(r)
        mats.append(rows)
    return mats


# --------------------------------------------------------------------------- #
# 1. classify -- the disagreement decision
# --------------------------------------------------------------------------- #
def test_classify_split_when_raters_oppose():
    assert adj.classify([True, False]) == "split"
    assert adj.classify([True, False, None]) == "split"      # blank never masks a real split
    assert adj.classify([False, True, True]) == "split"


def test_classify_unanimous_needs_two_agreeing_votes():
    assert adj.classify([True, True]) == "unanimous"
    assert adj.classify([False, False, None]) == "unanimous"


def test_classify_single_and_none():
    assert adj.classify([True, None, None]) == "single"
    assert adj.classify([None, False]) == "single"
    assert adj.classify([None, None]) == "none"
    assert adj.classify([]) == "none"


# --------------------------------------------------------------------------- #
# 2. worklist -- only split rows, adjudicated always blank
# --------------------------------------------------------------------------- #
def test_worklist_contains_only_split_rows_and_blank_adjudication(tmp_path):
    # it1: mentions split (T/F), supports unanimous(T,T); it2: both unanimous;
    # it3: supports split (T/F), mentions unanimous; it4: mentions split + incomplete (T,F,blank)
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "no"), ("it3", "yes", "yes"), ("it4", "yes", ""),
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [
        ("it1", "no", "yes"), ("it2", "no", "no"), ("it3", "yes", "no"), ("it4", "no", ""),
    ])
    c = _write_labeled(tmp_path / "labeled_c.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "no"), ("it3", "yes", "yes"), ("it4", "", ""),
    ])
    raters = adj.load_raters([a, b, c])
    header, rows = adj.build_worklist(raters)

    assert header == [ITEM_ID, "question", "a", "b", "c", "adjudicated"]
    # exactly three split rows: (it1, mentions), (it3, supports), (it4, mentions)
    keyed = {(r[0], r[1]): r for r in rows}
    assert set(keyed) == {("it1", "mentions"), ("it3", "supports"), ("it4", "mentions")}
    # per-rater cells render yes/no/blank; adjudicated column is blank on every row
    assert keyed[("it1", "mentions")] == ["it1", "mentions", "yes", "no", "yes", ""]
    assert keyed[("it4", "mentions")] == ["it4", "mentions", "yes", "no", "", ""]  # c blank
    for r in rows:
        assert r[-1] == "", "adjudicated column must never be filled by the tool"


def test_worklist_never_resolves_even_a_hard_tie(tmp_path):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("itX", "yes", "no")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("itX", "no", "yes")])
    _, rows = adj.build_worklist(adj.load_raters([a, b]))
    # both questions split 1-1; the tool surfaces both and resolves neither
    assert {(r[0], r[1]) for r in rows} == {("itX", "mentions"), ("itX", "supports")}
    assert all(r[-1] == "" for r in rows)


def test_worklist_empty_when_all_unanimous(tmp_path):
    # b is written without the BOM: identical VOTES are the point of this test, but
    # byte-identical FILES are refused by the shared input guard (an aliased file is
    # not a second rater), so the bytes must differ while the parse stays identical.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("i1", "yes", "no"), ("i2", "no", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("i1", "yes", "no"), ("i2", "no", "yes")],
                       bom=False)
    _, rows = adj.build_worklist(adj.load_raters([a, b]))
    assert rows == []


# --------------------------------------------------------------------------- #
# 3. summarize -- counts, split rate, incomplete overlap
# --------------------------------------------------------------------------- #
def test_summarize_counts_split_unanimous_and_incomplete(tmp_path):
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "no"), ("it3", "yes", "yes"), ("it4", "yes", ""),
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [
        ("it1", "no", "yes"), ("it2", "no", "no"), ("it3", "yes", "no"), ("it4", "no", ""),
    ])
    c = _write_labeled(tmp_path / "labeled_c.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "no"), ("it3", "yes", "yes"), ("it4", "", ""),
    ])
    stats = adj.summarize(adj.load_raters([a, b, c]))
    # mentions: it1 split, it2 unanimous, it3 unanimous, it4 split(T,F,blank) -> split 2, unan 2
    assert stats["mentions"]["split"] == 2
    assert stats["mentions"]["unanimous"] == 2
    assert stats["mentions"]["incomplete"] == 1               # it4 has c blank on mentions
    # supports: it1 unan, it2 unan, it3 split, it4 all blank -> split 1, unan 2, none 1
    assert stats["supports"]["split"] == 1
    assert stats["supports"]["unanimous"] == 2
    assert stats["supports"]["none"] == 1
    assert stats["supports"]["incomplete"] == 1               # it4 blank on supports


def test_item_ids_are_union_across_raters(tmp_path):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("i1", "yes", "yes"), ("i2", "no", "no")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("i2", "no", "no"), ("i9", "yes", "yes")])
    assert adj.item_ids(adj.load_raters([a, b])) == ["i1", "i2", "i9"]


# --------------------------------------------------------------------------- #
# 4. Fleiss' kappa -- textbook anchors + independent oracle
# --------------------------------------------------------------------------- #
def test_fleiss_textbook_wikipedia_value():
    assert abs(adj.fleiss_kappa(WIKIPEDIA_FLEISS) - 0.210) < 1e-3
    assert abs(adj.fleiss_kappa(WIKIPEDIA_FLEISS) - _ref_fleiss(WIKIPEDIA_FLEISS)) < 1e-12


def test_fleiss_hand_worked_binary_is_one_third():
    assert abs(adj.fleiss_kappa(HAND_BINARY) - 1 / 3) < 1e-12
    assert abs(adj.fleiss_kappa(HAND_BINARY) - _ref_fleiss(HAND_BINARY)) < 1e-12


def test_fleiss_matches_independent_oracle_on_many_matrices():
    checked = 0
    for rows in _random_matrices():
        ours = adj.fleiss_kappa(rows)
        ref = _ref_fleiss(rows)
        assert ours is not None and ref is not None
        assert abs(ours - ref) < 1e-12, (rows, ours, ref)
        checked += 1
    assert checked >= 100


def test_fleiss_perfect_agreement_is_one():
    assert adj.fleiss_kappa([[0, 3], [3, 0], [0, 3]]) == 1.0


def test_fleiss_single_category_degenerate_returns_one():
    # every rater picks the same category on every item -> pe == 1.0, formally undefined;
    # mirror cohen_kappa: perfect observed agreement -> 1.0.
    assert adj.fleiss_kappa([[3, 0], [3, 0]]) == 1.0


def test_fleiss_none_for_empty_or_too_few_raters():
    assert adj.fleiss_kappa([]) is None
    assert adj.fleiss_kappa([[1, 0]]) is None                 # n == 1 < 2


def test_fleiss_rejects_ragged_rater_counts():
    with pytest.raises(ValueError):
        adj.fleiss_kappa([[2, 1], [1, 1]])                    # rows sum to 3 and 2
    with pytest.raises(ValueError):
        adj.fleiss_kappa([[1, 0], [5, 5]])                    # first row sums < 2: guard is still total


def test_fleiss_rejects_ragged_category_widths():
    # equal row sums (3 and 3) but different widths: k from rows[0] would silently drop the
    # third category from P_e while P_bar still used it -- must raise, never mis-score.
    with pytest.raises(ValueError):
        adj.fleiss_kappa([[2, 1], [1, 1, 1]])
    with pytest.raises(ValueError):
        adj.fleiss_kappa([[1, 1, 1], [2, 1]])                 # shorter later row: same guard


# --------------------------------------------------------------------------- #
# 5. mentions_fleiss adapter -- complete-case restriction + rater-count gate
# --------------------------------------------------------------------------- #
def test_mentions_fleiss_excludes_incomplete_items(tmp_path):
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("it1", "yes", "x"), ("it2", "no", "x"), ("it3", "yes", "x"),
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [
        ("it1", "no", "x"), ("it2", "no", "x"), ("it3", "yes", "x"),
    ])
    c = _write_labeled(tmp_path / "labeled_c.csv", [
        ("it1", "yes", "x"), ("it2", "no", "x"), ("it3", "", "x"),   # it3 blank -> excluded
    ])
    raters = adj.load_raters([a, b, c])
    kappa, n_used, n_excluded = adj.mentions_fleiss(raters)
    assert (n_used, n_excluded) == (2, 1)
    # complete-case rows: it1 [no=1,yes=2], it2 [no=3,yes=0] -> equals fleiss on that matrix
    assert abs(kappa - adj.fleiss_kappa([[1, 2], [3, 0]])) < 1e-12


def test_mentions_fleiss_needs_three_raters(tmp_path):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "x"), ("it2", "no", "x")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "x"), ("it2", "no", "x")])
    kappa, _, _ = adj.mentions_fleiss(adj.load_raters([a, b]))
    assert kappa is None


# --------------------------------------------------------------------------- #
# 6. reuse of the frozen-design loader (not a rewrite)
# --------------------------------------------------------------------------- #
def test_reuses_score_labels_loader_object():
    assert adj.load_labeled is sys.modules["score_labels"].load_labeled


# --------------------------------------------------------------------------- #
# 7. main() end-to-end on SYNTHETIC data
# --------------------------------------------------------------------------- #
def test_main_writes_worklist_and_prints_supplementary_fleiss(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "no"), ("it3", "yes", "yes"),
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [
        ("it1", "no", "yes"), ("it2", "no", "no"), ("it3", "yes", "no"),
    ])
    # c agrees with a vote-for-vote; written without the BOM so the files stay
    # byte-distinct (the shared input guard refuses byte-identical "raters").
    c = _write_labeled(tmp_path / "labeled_c.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "no"), ("it3", "yes", "yes"),
    ], bom=False)
    out = tmp_path / "worklist.csv"
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), str(c), "--out", str(out),
    ])
    rc = adj.main()
    text = capsys.readouterr().out
    assert rc == 0

    written = _read_rows(out)
    assert written[0] == [ITEM_ID, "question", "a", "b", "c", "adjudicated"]
    body = {(r[0], r[1]): r for r in written[1:]}
    assert set(body) == {("it1", "mentions"), ("it3", "supports")}   # the two splits
    assert all(r[-1] == "" for r in written[1:])                     # adjudicated blank

    assert "3 raters" in text
    assert "Q1 mentions" in text and "Q2 supports" in text
    assert "wrote 2 split row(s)" in text
    assert "never resolves them" in text
    # supplementary Fleiss line, clearly subordinate to the pre-committed Cohen's kappa
    assert "SUPPLEMENTARY" in text
    assert "score_labels.py" in text
    # mentions complete-case matrix: it1 [no1,yes2], it2 [no3,yes0], it3 [no0,yes3]
    expected_k = adj.fleiss_kappa([[1, 2], [3, 0], [0, 3]])
    assert f"Fleiss' kappa (mentions) = {expected_k:.3f}" in text


def test_main_two_raters_reports_fleiss_not_applicable(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    out = tmp_path / "wl.csv"
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(out),
    ])
    rc = adj.main()
    text = capsys.readouterr().out
    assert rc == 0
    assert "needs >= 3 raters" in text
    assert out.exists()


def test_main_refuses_to_overwrite_an_input_file(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    before = a.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(a),  # out == input a
    ])
    rc = adj.main()
    text = capsys.readouterr().out
    assert rc == 2
    assert "refusing to write" in text
    assert a.read_bytes() == before          # the raw label file is untouched


def test_main_rejects_duplicate_rater_names(tmp_path, monkeypatch, capsys):
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    d1.mkdir()
    d2.mkdir()
    a = _write_labeled(d1 / "labeled_sam.csv", [("it1", "yes", "yes")])
    a2 = _write_labeled(d2 / "labeled_sam.csv", [("it1", "no", "yes")])   # same rater name "sam"
    out = tmp_path / "wl.csv"
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(a2), "--out", str(out),
    ])
    rc = adj.main()
    text = capsys.readouterr().out
    assert rc == 2
    assert "map to rater 'sam'" in text
    assert not out.exists()


def test_main_leaves_input_files_byte_identical(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes"), ("it2", "no", "no")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes"), ("it2", "no", "no")])
    before = {p: p.read_bytes() for p in (a, b)}
    out = tmp_path / "wl.csv"
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(out),
    ])
    assert adj.main() == 0
    capsys.readouterr()
    for p, blob in before.items():
        assert p.read_bytes() == blob        # inputs never mutated


def test_main_summary_reports_na_when_a_question_has_no_colabeled_items(tmp_path, monkeypatch,
                                                                         capsys):
    # every rater leaves `supports` blank -> that question has 0 co-labeled items -> n/a rate
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "")])
    # same votes as a, written without the BOM to keep the bytes distinct (alias guard)
    c = _write_labeled(tmp_path / "labeled_c.csv", [("it1", "yes", "")], bom=False)
    out = tmp_path / "wl.csv"
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), str(c), "--out", str(out),
    ])
    assert adj.main() == 0
    assert "n/a (no item had >= 2 votes)" in capsys.readouterr().out


def test_main_refuses_hardlink_alias_of_an_input(tmp_path, monkeypatch, capsys):
    # A hardlink defeats a pure Path.resolve() comparison (different name, same inode) --
    # the stand-in for the case-variant bypass on case-insensitive APFS, but portable to
    # the case-sensitive CI filesystem. The guard must catch it via samefile().
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    alias = tmp_path / "not_obviously_a_label_file.csv"
    os.link(a, alias)
    before = a.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(alias),
    ])
    rc = adj.main()
    assert rc == 2
    assert "refusing to write" in capsys.readouterr().out
    assert a.read_bytes() == before          # the rater's raw labels survive


def test_main_refuses_to_clobber_a_human_adjudicated_worklist(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    out = tmp_path / "worklist.csv"
    argv = ["adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert adj.main() == 0
    capsys.readouterr()

    # Simulate the HUMAN adjudicator filling one decision (synthetic data; in real use this
    # is the human step the tool must never redo or destroy).
    rows = _read_rows(out)
    rows[1][-1] = "yes"
    with out.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    filled = out.read_bytes()

    monkeypatch.setattr(sys, "argv", argv)
    rc = adj.main()
    text = capsys.readouterr().out
    assert rc == 2
    assert "human" in text and "adjudication" in text
    assert out.read_bytes() == filled        # the human's decisions are untouched


def test_main_overwrites_an_unfilled_worklist_idempotently(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    out = tmp_path / "worklist.csv"
    argv = ["adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert adj.main() == 0
    first = out.read_bytes()
    monkeypatch.setattr(sys, "argv", argv)
    assert adj.main() == 0                   # no adjudication entered yet -> re-run is fine
    assert out.read_bytes() == first
    capsys.readouterr()


def test_load_raters_rejects_reserved_rater_names(tmp_path):
    # labeled_adjudicated.csv would shadow the tool's own blank column with a duplicate
    # header -- the one column whose blankness carries the no-AI-adjudication guarantee.
    bad = _write_labeled(tmp_path / "labeled_adjudicated.csv", [("it1", "yes", "yes")])
    with pytest.raises(ValueError, match="collides with a fixed worklist column"):
        adj.load_raters([bad])
    bad2 = _write_labeled(tmp_path / "labeled_item_id.csv", [("it1", "yes", "yes")])
    with pytest.raises(ValueError, match="collides"):
        adj.load_raters([bad2])


def test_main_friendly_error_on_missing_input_file(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(tmp_path / "labeled_ghost.csv"),
        "--out", str(tmp_path / "wl.csv"),
    ])
    rc = adj.main()                          # no raw FileNotFoundError traceback
    text = capsys.readouterr().out
    assert rc == 2
    assert "[adjudicate] cannot load labeled_ghost.csv" in text


def test_main_friendly_error_on_csv_without_item_id_header(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    bad = tmp_path / "labeled_bad.csv"
    bad.write_text("wrong,headers\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(bad),
        "--out", str(tmp_path / "wl.csv"),
    ])
    rc = adj.main()                          # no raw KeyError traceback
    text = capsys.readouterr().out
    assert rc == 2
    assert "[adjudicate] cannot load labeled_bad.csv" in text


def test_pipeline_handles_items_absent_from_some_raters(tmp_path):
    # Items entirely MISSING from a rater's CSV (no row at all, not just a blank cell) must
    # flow through votes_for -> classify -> worklist/summarize/fleiss as absent votes.
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("i1", "yes", "yes"), ("i2", "yes", "yes"), ("i5", "yes", "yes"),
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [
        ("i2", "no", "yes"), ("i5", "no", "yes"), ("i9", "yes", "yes"),
    ])
    c = _write_labeled(tmp_path / "labeled_c.csv", [("i2", "yes", "yes")])
    raters = adj.load_raters([a, b, c])

    header, rows = adj.build_worklist(raters)
    keyed = {(r[0], r[1]): r for r in rows}
    # i2 mentions: yes/no/yes -> split with all three cells; i5 mentions: yes/no/ABSENT ->
    # split, c's cell rendered blank. i1/i9 are single-vote items -> never in the worklist.
    assert keyed[("i2", "mentions")] == ["i2", "mentions", "yes", "no", "yes", ""]
    assert keyed[("i5", "mentions")] == ["i5", "mentions", "yes", "no", "", ""]
    assert not any(item in ("i1", "i9") for item, _ in keyed)

    stats = adj.summarize(raters)
    assert stats["mentions"]["split"] == 2               # i2, i5
    assert stats["mentions"]["single"] == 2              # i1, i9
    assert stats["mentions"]["incomplete"] == 3          # i1, i5, i9 (>=1 missing vote)

    kappa, n_used, n_excluded = adj.mentions_fleiss(raters)
    assert (n_used, n_excluded) == (1, 3)                # only i2 is complete-case
    assert kappa is not None


def test_main_fleiss_na_when_no_item_labeled_on_mentions_by_every_rater(tmp_path, monkeypatch,
                                                                        capsys):
    # 3 raters, but every item is missing one rater's `mentions` vote -> no complete case
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes"), ("it2", "", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes"), ("it2", "yes", "yes")])
    c = _write_labeled(tmp_path / "labeled_c.csv", [("it1", "", "yes"), ("it2", "no", "yes")])
    out = tmp_path / "wl.csv"
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), str(c), "--out", str(out),
    ])
    assert adj.main() == 0
    assert "no item was labeled on 'mentions' by every rater" in capsys.readouterr().out
