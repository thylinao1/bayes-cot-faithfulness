"""Unit tests for experiments/assemble_golden_labels.py, the consensus-assembly tool.

assemble_golden_labels.py closes the last tooling gap of labeling week: given the raw
labeled_<name>.csv files plus the human-FILLED adjudication worklist, it mechanically merges
them into a final golden-labels CSV. Unanimous items keep their vote; split items take the
human adjudicated value VERBATIM; split items with a blank adjudicated cell and incomplete
(single-vote / unlabeled) items are FLAGGED with a blank label cell, never guessed. By
default the tool refuses to write when anything would be flagged; --allow-partial writes the
flagged sheet with an explicit PARTIAL marker column instead.

RATER BLINDING -- ABSOLUTE. Nothing here reads, opens, or references any real sealed
label/key artifact (results/labelset/, results/labeling_key.json, results/labeling_sheet.*,
labeled_fable5*.csv, fable5_agreement_stats.txt). Every fixture is SYNTHETIC data written
into pytest's tmp_path. The tool has no --key argument and derives item ids from the labeled
CSVs themselves, so it can never touch the sealed key even by accident.

THE MODEL NEVER DECIDES A LABEL. The tool applies HUMAN decisions mechanically: a split's
value comes only from the human-filled adjudicated cell, an unparseable adjudication is a
hard error (never coerced), and a missing one is flagged (never inferred from majority,
rater order, or anything else). The tests below pin each of those refusals.

The module is loaded by file path exactly like tests/test_adjudicate_labels.py loads
adjudicate_labels (it lives in experiments/ and imports its siblings via a sys.path insert).
STDLIB-ONLY on purpose: the CI dev environment installs no scikit-learn.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "assemble_golden_labels.py"


def _load_assemble():
    spec = importlib.util.spec_from_file_location("assemble_golden_labels_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own sys.path.insert(0, HERE) + sibling imports
    # (labeling_columns, score_labels, adjudicate_labels) resolve.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


asm = _load_assemble()
# The siblings are now in sys.modules (asm imported them); reuse the real sheet headers to
# write synthetic label CSVs without a second import statement.
_LC = sys.modules["labeling_columns"]
ITEM_ID, Q_MENTIONS, Q_SUPPORTS = _LC.ITEM_ID, _LC.Q_MENTIONS, _LC.Q_SUPPORTS
ANNOTATOR = _LC.ANNOTATOR
_ADJ = sys.modules["adjudicate_labels"]


# --------------------------------------------------------------------------- #
# Synthetic-fixture writers (all data constructed here, never read from disk)
# --------------------------------------------------------------------------- #
def _write_labeled(path: Path, rows: list[tuple[str, str, str]], *, bom: bool = True) -> Path:
    """Write a synthetic filled label CSV with the REAL sheet headers.

    The annotator-name column is filled with the rater name (as the labeling guide
    instructs real raters to do), which also keeps distinct raters' files byte-distinct
    the way real filled sheets are.
    """
    name = path.stem.replace("labeled_", "")
    cols = [ITEM_ID, Q_MENTIONS, Q_SUPPORTS, ANNOTATOR]
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for item_id, mentions, supports in rows:
            w.writerow({ITEM_ID: item_id, Q_MENTIONS: mentions, Q_SUPPORTS: supports,
                        ANNOTATOR: name if item_id else ""})
    return path


def _write_worklist(path: Path, rater_names: list[str],
                    rows: list[tuple[str, str, list[str], str]], *, bom: bool = True) -> Path:
    """Write a synthetic adjudication worklist in adjudicate_labels' exact format.

    ``rows`` is (item_id, question, per-rater vote cells, adjudicated cell) -- the
    adjudicated cell is what the HUMAN typed after the worklist was generated.
    """
    header = [ITEM_ID, asm.QUESTION_COL, *rater_names, asm.ADJUDICATED_COL]
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for item, q, cells, adjudicated in rows:
            w.writerow([item, q, *cells, adjudicated])
    return path


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def _read_dicts(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _run_main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["assemble_golden_labels.py", *argv])
    return asm.main()


# Two-rater fixture matching the frozen design (operator + one friend):
#   it1 unanimous yes/yes on both questions; it2 mentions SPLIT, supports unanimous;
#   it3 unanimous no on mentions, supports SPLIT.
def _two_rater_fixture(tmp_path: Path) -> tuple[Path, Path]:
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("it1", "yes", "yes"), ("it2", "yes", "no"), ("it3", "no", "yes"),
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "no"), ("it3", "no", "no"),
    ])
    return a, b


def _filled_worklist(tmp_path: Path) -> Path:
    return _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it2", "mentions", ["yes", "no"], "no"),
        ("it3", "supports", ["yes", "no"], "yes"),
    ])


# --------------------------------------------------------------------------- #
# 1. the merge rule: unanimous keeps the vote, splits take the human decision
# --------------------------------------------------------------------------- #
def test_unanimous_items_keep_their_vote_and_splits_take_adjudication(tmp_path, monkeypatch,
                                                                      capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(out)])
    assert rc == 0
    rows = {r[ITEM_ID]: r for r in _read_dicts(out)}
    assert set(rows) == {"it1", "it2", "it3"}
    # unanimous cells keep the agreed vote
    assert rows["it1"][Q_MENTIONS] == "yes" and rows["it1"][Q_SUPPORTS] == "yes"
    assert rows["it2"][Q_SUPPORTS] == "no"
    assert rows["it3"][Q_MENTIONS] == "no"
    # split cells carry the HUMAN adjudicated value verbatim
    assert rows["it2"][Q_MENTIONS] == "no"
    assert rows["it3"][Q_SUPPORTS] == "yes"
    # provenance columns say which rule produced each cell
    assert rows["it1"][asm.SOURCE_COL["mentions"]] == "unanimous"
    assert rows["it2"][asm.SOURCE_COL["mentions"]] == "adjudicated"
    assert rows["it3"][asm.SOURCE_COL["supports"]] == "adjudicated"
    # a complete sheet carries no PARTIAL marker column
    assert asm.SHEET_STATUS_COL not in _read_rows(out)[0]
    text = capsys.readouterr().out
    # provenance counts pinned numerically: 6 cells = 4 unanimous + 2 adjudicated
    assert "unanimous 4" in text and "adjudicated 2" in text


def test_output_is_loadable_by_the_frozen_score_labels_loader(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    loaded = sys.modules["score_labels"].load_labeled(out)
    assert loaded["it1"] == {"mentions": True, "supports": True}
    assert loaded["it2"] == {"mentions": False, "supports": False}
    assert loaded["it3"] == {"mentions": False, "supports": True}


def test_adjudicated_value_is_applied_verbatim_even_against_the_majority(tmp_path,
                                                                         monkeypatch, capsys):
    # Three raters, 2-vs-1 split: the human adjudicated the MINORITY value. The tool must
    # apply the human decision, never a majority vote of its own.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "yes", "yes")])
    c = _write_labeled(tmp_path / "labeled_c.csv", [("it1", "no", "yes")])
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b", "c"], [
        ("it1", "mentions", ["yes", "yes", "no"], "no"),   # human sided with the minority
    ])
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), str(c),
                                   "--worklist", str(wl), "--out", str(out)]) == 0
    capsys.readouterr()
    row = _read_dicts(out)[0]
    assert row[Q_MENTIONS] == "no"
    assert row[asm.SOURCE_COL["mentions"]] == "adjudicated"


def test_adjudicated_cell_accepts_the_same_tokens_as_rater_cells(tmp_path, monkeypatch,
                                                                 capsys):
    # The adjudicated cell is parsed with the SAME yes/no token conventions load_labeled
    # applies to rater cells ("Y", "TRUE", "1", case and whitespace) -- and canonicalized
    # to yes/no on write, like every other cell in the golden CSV.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it1", "mentions", ["yes", "no"], "  TRUE "),
    ])
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    assert _read_dicts(out)[0][Q_MENTIONS] == "yes"


def test_unanimous_with_a_blank_third_rater_still_counts_as_unanimous(tmp_path, monkeypatch,
                                                                      capsys):
    # classify() semantics: >= 2 agreeing votes with the rest blank is unanimous, and the
    # blank never flags the cell (mirrors the worklist tool's split definition).
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "yes", "yes")])
    c = _write_labeled(tmp_path / "labeled_c.csv", [("it1", "", "yes")])
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), str(c),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    row = _read_dicts(out)[0]
    assert row[Q_MENTIONS] == "yes"
    assert row[asm.SOURCE_COL["mentions"]] == "unanimous"


# --------------------------------------------------------------------------- #
# 2. the flag gate: nothing is ever guessed, partial needs an explicit flag
# --------------------------------------------------------------------------- #
def test_refuses_when_a_split_lacks_adjudication(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it2", "mentions", ["yes", "no"], "no"),
        ("it3", "supports", ["yes", "no"], ""),           # human has not decided this one
    ])
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 2
    assert not out.exists()
    assert "missing adjudication" in text
    assert "--allow-partial" in text


def test_refuses_when_no_worklist_is_given_but_splits_exist(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 2
    assert not out.exists()
    assert "missing adjudication" in text


def test_refuses_on_single_vote_and_unlabeled_items_by_default(tmp_path, monkeypatch, capsys):
    # it1 complete; it2 single-vote on mentions (b blank), unlabeled on supports (both blank)
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes"), ("it2", "yes", "")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "yes", "yes"), ("it2", "", "")])
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 2
    assert not out.exists()
    assert "single vote" in text and "unlabeled" in text


def test_allow_partial_writes_flags_blank_cells_and_partial_marker(tmp_path, monkeypatch,
                                                                   capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("it1", "yes", "yes"), ("it2", "yes", "yes"), ("it3", "yes", ""),
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [
        ("it1", "yes", "yes"), ("it2", "no", "yes"), ("it3", "", ""),
    ])
    out = tmp_path / "golden.PARTIAL.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out),
                                 "--allow-partial"])
    text = capsys.readouterr().out
    assert rc == 0
    rows = {r[ITEM_ID]: r for r in _read_dicts(out)}
    # complete cells still resolve
    assert rows["it1"][Q_MENTIONS] == "yes"
    # the un-adjudicated split is flagged with a BLANK label cell, never guessed
    assert rows["it2"][Q_MENTIONS] == ""
    assert rows["it2"][asm.SOURCE_COL["mentions"]] == "FLAGGED_missing_adjudication"
    # single-vote and unlabeled cells are flagged with blank label cells too
    assert rows["it3"][Q_MENTIONS] == ""
    assert rows["it3"][asm.SOURCE_COL["mentions"]] == "FLAGGED_single_vote"
    assert rows["it3"][Q_SUPPORTS] == ""
    assert rows["it3"][asm.SOURCE_COL["supports"]] == "FLAGGED_unlabeled"
    # every row carries the PARTIAL marker column
    assert all(r[asm.SHEET_STATUS_COL] == "PARTIAL" for r in _read_dicts(out))
    assert "PARTIAL" in text


def test_allow_partial_on_a_complete_sheet_writes_a_normal_sheet(tmp_path, monkeypatch,
                                                                 capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out), "--allow-partial"]) == 0
    capsys.readouterr()
    assert asm.SHEET_STATUS_COL not in _read_rows(out)[0]   # complete -> no PARTIAL marker


def test_unparseable_adjudicated_value_is_an_error_even_with_allow_partial(tmp_path,
                                                                           monkeypatch,
                                                                           capsys):
    # A non-blank adjudication that does not parse as yes/no is a human-input error the
    # tool must surface -- treating it as blank would silently DISCARD a human decision,
    # and coercing it would INVENT one.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it1", "mentions", ["yes", "no"], "maybe"),
    ])
    out = tmp_path / "golden.csv"
    for extra in ([], ["--allow-partial"]):
        rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                     "--out", str(out), *extra])
        text = capsys.readouterr().out
        assert rc == 2
        assert not out.exists()
        assert "maybe" in text and "it1" in text


def test_never_fills_a_flagged_cell_from_any_side_channel(tmp_path, monkeypatch, capsys):
    # A hard tie with no adjudication must stay blank under --allow-partial: no majority,
    # no first-rater preference, no coin flip. The model never decides a label.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("itX", "yes", "no")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("itX", "no", "yes")])
    out = tmp_path / "golden.PARTIAL.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out),
                                   "--allow-partial"]) == 0
    capsys.readouterr()
    row = _read_dicts(out)[0]
    assert row[Q_MENTIONS] == "" and row[Q_SUPPORTS] == ""
    assert row[asm.SOURCE_COL["mentions"]] == "FLAGGED_missing_adjudication"
    assert row[asm.SOURCE_COL["supports"]] == "FLAGGED_missing_adjudication"


# --------------------------------------------------------------------------- #
# 3. worklist validation: stale or mismatched adjudications never apply
# --------------------------------------------------------------------------- #
def test_rejects_worklist_whose_rater_columns_mismatch_the_labeled_files(tmp_path,
                                                                         monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "zoe"], [
        ("it2", "mentions", ["yes", "no"], "no"),
    ])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "rater" in text


def test_rejects_worklist_row_whose_recorded_votes_mismatch_current_labels(tmp_path,
                                                                           monkeypatch,
                                                                           capsys):
    # The worklist records each rater's vote at generation time. If a labeled CSV changed
    # since (recorded a=yes, but the file now says a=no), the adjudication may no longer
    # apply -- refuse and tell the operator to regenerate the worklist.
    a, b = _two_rater_fixture(tmp_path)
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it2", "mentions", ["no", "yes"], "no"),          # recorded votes are swapped
        ("it3", "supports", ["yes", "no"], "yes"),
    ])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "stale" in text and "it2" in text


def test_rejects_worklist_row_for_a_pair_that_is_no_longer_split(tmp_path, monkeypatch,
                                                                 capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it2", "mentions", ["yes", "no"], "no"),
        ("it1", "mentions", ["yes", "no"], "yes"),         # it1 is unanimous now -> stale
        ("it3", "supports", ["yes", "no"], "yes"),
    ])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "stale" in text and "it1" in text


def test_rejects_worklist_with_duplicate_rows_or_unknown_question(tmp_path, monkeypatch,
                                                                  capsys):
    a, b = _two_rater_fixture(tmp_path)
    dup = _write_worklist(tmp_path / "wl_dup.csv", ["a", "b"], [
        ("it2", "mentions", ["yes", "no"], "no"),
        ("it2", "mentions", ["yes", "no"], "yes"),         # duplicate, contradictory
    ])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(dup),
                                 "--out", str(tmp_path / "g1.csv")])
    assert rc == 2
    assert "duplicate" in capsys.readouterr().out

    badq = _write_worklist(tmp_path / "wl_badq.csv", ["a", "b"], [
        ("it2", "typo_question", ["yes", "no"], "no"),
    ])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(badq),
                                 "--out", str(tmp_path / "g2.csv")])
    assert rc == 2
    assert "question" in capsys.readouterr().out


def test_rejects_worklist_row_for_an_unknown_item(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it2", "mentions", ["yes", "no"], "no"),
        ("ghost", "mentions", ["yes", "no"], "no"),        # no rater ever labeled "ghost"
        ("it3", "supports", ["yes", "no"], "yes"),
    ])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "ghost" in text


def test_rejects_worklist_missing_required_columns(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    bad = tmp_path / "not_a_worklist.csv"
    bad.write_text("some,other,headers\n1,2,3\n", encoding="utf-8")
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(bad),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "worklist" in text


# --------------------------------------------------------------------------- #
# 4. input hygiene and overwrite guards
# --------------------------------------------------------------------------- #
def test_requires_at_least_two_raters(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    rc = _run_main(monkeypatch, ["--labeled", str(a), "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert ">= 2" in text


def test_refuses_to_write_over_an_input_label_file_or_the_worklist(tmp_path, monkeypatch,
                                                                   capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    for target in (a, wl):
        before = target.read_bytes()
        rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                     "--out", str(target)])
        assert rc == 2
        assert "refusing" in capsys.readouterr().out
        assert target.read_bytes() == before


def test_refuses_hardlink_alias_of_the_worklist_as_out(tmp_path, monkeypatch, capsys):
    # Same inode-level guard the worklist tool uses: a hardlink (the portable stand-in for
    # a case-variant path on case-insensitive APFS) must not defeat the refusal.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    alias = tmp_path / "innocuous_name.csv"
    os.link(wl, alias)
    before = wl.read_bytes()
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(alias)])
    assert rc == 2
    assert "refusing" in capsys.readouterr().out
    assert wl.read_bytes() == before


def test_refuses_to_overwrite_a_file_that_is_not_a_previous_golden_output(tmp_path,
                                                                          monkeypatch,
                                                                          capsys):
    # --out points at an EXISTING file that is not this tool's own output (here: a filled
    # label CSV under a name the input guard cannot recognize). results/ is gitignored, so
    # a silent overwrite would destroy un-versioned human work -- refuse.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    stray = _write_labeled(tmp_path / "labeled_carol_backup.csv", [("it9", "yes", "yes")])
    before = stray.read_bytes()
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(stray)])
    assert rc == 2
    assert "refusing" in capsys.readouterr().out
    assert stray.read_bytes() == before


def test_overwrites_its_own_previous_output_idempotently(tmp_path, monkeypatch, capsys):
    # The golden CSV is a deterministic mechanical merge of its inputs, so re-running onto
    # a PREVIOUS golden output (recognized by its provenance columns) is safe and
    # byte-identical.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    argv = ["--labeled", str(a), str(b), "--worklist", str(wl), "--out", str(out)]
    assert _run_main(monkeypatch, argv) == 0
    first = out.read_bytes()
    assert _run_main(monkeypatch, argv) == 0
    assert out.read_bytes() == first
    capsys.readouterr()


def test_inputs_stay_byte_identical(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    before = {p: p.read_bytes() for p in (a, b, wl)}
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(tmp_path / "golden.csv")]) == 0
    capsys.readouterr()
    for p, blob in before.items():
        assert p.read_bytes() == blob


def test_friendly_error_on_missing_or_malformed_label_csv(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(tmp_path / "labeled_ghost.csv"),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "cannot load labeled_ghost.csv" in text


# --------------------------------------------------------------------------- #
# 5. reuse of the frozen-design loader and the worklist tool's conventions
# --------------------------------------------------------------------------- #
def test_reuses_the_sibling_modules_not_rewrites(tmp_path):
    adj = sys.modules["adjudicate_labels"]
    assert asm.load_raters is adj.load_raters
    assert asm.classify is adj.classify
    assert asm.item_ids is adj.item_ids
    assert asm.votes_for is adj.votes_for
    assert asm.load_labeled is sys.modules["score_labels"].load_labeled
    assert asm.QUESTIONS is adj.QUESTIONS
    assert asm.ADJUDICATED_COL == adj.ADJUDICATED_COL
    assert asm.QUESTION_COL == adj.QUESTION_COL


def test_round_trip_with_the_real_worklist_generator(tmp_path, monkeypatch, capsys):
    # End-to-end pipeline on SYNTHETIC data: adjudicate_labels writes the worklist, a
    # simulated HUMAN fills the adjudicated column, assemble merges. The two tools must
    # agree on the worklist format by construction, not by coincidence.
    adj = sys.modules["adjudicate_labels"]
    a, b = _two_rater_fixture(tmp_path)
    wl = tmp_path / "worklist.csv"
    monkeypatch.setattr(sys, "argv", [
        "adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(wl),
    ])
    assert adj.main() == 0
    capsys.readouterr()

    # the HUMAN adjudicates every split row (synthetic stand-in for the human step)
    rows = _read_rows(wl)
    for row in rows[1:]:
        row[-1] = "yes"
    with wl.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    got = {r[ITEM_ID]: r for r in _read_dicts(out)}
    assert got["it2"][Q_MENTIONS] == "yes"                 # the human's decision
    assert got["it2"][asm.SOURCE_COL["mentions"]] == "adjudicated"
    assert got["it1"][asm.SOURCE_COL["mentions"]] == "unanimous"


# --------------------------------------------------------------------------- #
# 6. the printed summary tells the operator what happened
# --------------------------------------------------------------------------- #
def test_summary_reports_counts_and_the_never_decides_disclaimer(tmp_path, monkeypatch,
                                                                 capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "2 raters" in text
    assert "3 items" in text
    assert "unanimous 4" in text and "adjudicated 2" in text
    assert "2 filled adjudication(s)" in text               # filled count, not row count
    assert "never decides" in text                          # the model-never-labels boundary


# --------------------------------------------------------------------------- #
# 7. hardening locked by the Fable review panel (2026-07-19)
# --------------------------------------------------------------------------- #
def test_refuses_byte_identical_label_inputs(tmp_path, monkeypatch, capsys):
    # HIGH finding: one rater's file aliased (hardlink / copy / case-variant) under a second
    # rater name would mint a "two-rater" golden set from ONE human. Byte-identity across
    # labeled inputs is refused: independent raters produce distinct files.
    d1, d2 = tmp_path / "one", tmp_path / "two"
    d1.mkdir(), d2.mkdir()
    a = _write_labeled(d1 / "labeled_a.csv", [("it1", "yes", "yes")])
    linked = d2 / "labeled_b.csv"
    os.link(a, linked)                                       # same inode, different rater name
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(linked),
                                 "--out", str(tmp_path / "golden.csv")])
    assert rc == 2
    assert "byte-identical" in capsys.readouterr().out

    copied = d2 / "labeled_c.csv"
    copied.write_bytes(a.read_bytes())                       # plain copy, different inode
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(copied),
                                 "--out", str(tmp_path / "golden.csv")])
    assert rc == 2
    assert "byte-identical" in capsys.readouterr().out


def test_refuses_duplicate_item_rows_within_one_labeled_csv(tmp_path, monkeypatch, capsys):
    # HIGH finding: load_labeled keys on item_id last-wins, so a duplicated contradictory
    # row inside ONE rater's CSV silently rewrote that rater's vote. Refused up front.
    a = _write_labeled(tmp_path / "labeled_a.csv", [
        ("it1", "yes", "yes"), ("it1", "no", "yes"),         # same item, contradictory vote
    ])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "yes", "yes")])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "duplicate" in text and "it1" in text and "labeled_a.csv" in text


def test_refuses_a_golden_output_as_labeled_input(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "labeled_gold.csv"                      # a golden output with a rater-ish name
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(out),
                                 "--out", str(tmp_path / "golden2.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "golden" in text and "not a rater" in text


def test_padding_rows_skipped_but_blank_id_with_content_refused(tmp_path, monkeypatch,
                                                                capsys):
    # Excel padding (fully blank rows) is tolerated; a row with votes but no item_id is a
    # phantom item and refused, never merged.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes"), ("", "", "")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "yes", "yes")])
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out)]) == 0
    capsys.readouterr()
    assert [r[ITEM_ID] for r in _read_dicts(out)] == ["it1"]

    c = _write_labeled(tmp_path / "labeled_c.csv", [("it1", "yes", "yes"), ("", "no", "")])
    rc = _run_main(monkeypatch, ["--labeled", str(c), str(b),
                                 "--out", str(tmp_path / "golden3.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "item_id" in text and "labeled_c.csv" in text


def test_rejects_worklist_with_duplicate_header_columns(tmp_path, monkeypatch, capsys):
    # MEDIUM finding: csv.DictReader is last-wins on duplicate headers, so a duplicated
    # 'adjudicated' column silently discarded the human's decision and applied a stray one.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    wl = tmp_path / "worklist.csv"
    with wl.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([ITEM_ID, asm.QUESTION_COL, "a", "b",
                    asm.ADJUDICATED_COL, asm.ADJUDICATED_COL])
        w.writerow(["it1", "mentions", "yes", "no", "no", "yes"])
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 2
    assert not out.exists()
    assert "duplicate" in text


def test_friendly_error_on_missing_worklist_file(tmp_path, monkeypatch, capsys):
    a, b = _two_rater_fixture(tmp_path)
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b),
                                 "--worklist", str(tmp_path / "ghost_worklist.csv"),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "ghost_worklist.csv" in text


def test_friendly_error_on_csv_error_not_a_traceback(tmp_path, monkeypatch, capsys):
    # MEDIUM finding: csv.Error subclasses Exception only, so an oversized field (beyond
    # csv.field_size_limit) escaped every net as a raw traceback. Now a clean refusal.
    a, b = _two_rater_fixture(tmp_path)
    wl = tmp_path / "worklist.csv"
    huge = "x" * 200_000
    with wl.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([ITEM_ID, asm.QUESTION_COL, "a", "b", asm.ADJUDICATED_COL])
        w.writerow(["it2", "mentions", "yes", "no", huge])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "[assemble]" in text


def test_non_utf8_file_at_out_is_refused_cleanly(tmp_path, monkeypatch, capsys):
    # LOW finding: a non-UTF8 file at --out crashed the overwrite guard with a raw
    # UnicodeDecodeError traceback. Now the guard treats it as not-ours and refuses.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    target = tmp_path / "existing.bin"
    target.write_bytes("item_id,junk\n".encode("utf-16"))
    before = target.read_bytes()
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(target)])
    assert rc == 2
    assert "refusing" in capsys.readouterr().out
    assert target.read_bytes() == before


def test_zero_item_inputs_are_refused_not_written_as_complete(tmp_path, monkeypatch, capsys):
    # LOW finding: two header-only CSVs produced an exit-0 "complete" golden sheet with no
    # rows. An empty merge is never a golden set. (One file gets a blank padding row so the
    # two zero-item inputs are not byte-identical, which is refused separately.)
    a = _write_labeled(tmp_path / "labeled_a.csv", [("", "", "")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [])
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 2
    assert not out.exists()
    assert "0 items" in text


def test_allow_partial_requires_partial_in_the_out_filename(tmp_path, monkeypatch, capsys):
    # MEDIUM finding: the frozen loader strips unknown columns, so the in-file PARTIAL
    # marker vanishes on load. The filename itself must carry the marker: a partial write
    # to a name without "partial" is refused.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out),
                                 "--allow-partial"])
    text = capsys.readouterr().out
    assert rc == 2
    assert not out.exists()
    assert "partial" in text.lower()
    # same run with a compliant name succeeds
    ok = tmp_path / "golden.PARTIAL.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(ok),
                                   "--allow-partial"]) == 0
    capsys.readouterr()
    assert ok.exists()


def test_partial_then_complete_overwrite_of_the_same_path(tmp_path, monkeypatch, capsys):
    # MEDIUM finding (test gap): the primary --allow-partial workflow is partial now,
    # complete later. A complete re-run onto the previous partial output must overwrite it
    # and drop the PARTIAL marker column.
    a, b = _two_rater_fixture(tmp_path)
    out = tmp_path / "golden_partial.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out),
                                   "--allow-partial"]) == 0
    capsys.readouterr()
    assert asm.SHEET_STATUS_COL in _read_rows(out)[0]        # partial marker present

    wl = _filled_worklist(tmp_path)                          # the human finished adjudicating
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    header = _read_rows(out)[0]
    assert asm.SHEET_STATUS_COL not in header                # complete sheet, marker gone
    assert all(r[asm.SOURCE_COL["mentions"]] in ("unanimous", "adjudicated")
               for r in _read_dicts(out))


def test_refuses_duplicate_header_columns_in_a_labeled_csv(tmp_path, monkeypatch, capsys):
    # HIGH finding (checker round 2): a duplicated Q1 column in a RATER file is last-wins
    # in the loader, so a stray second copy silently flipped a real vote and turned a true
    # split into unanimity. Refused up front, same as the worklist header guard.
    bad = tmp_path / "labeled_a.csv"
    with bad.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([ITEM_ID, Q_MENTIONS, Q_SUPPORTS, Q_MENTIONS])   # Q1 duplicated
        w.writerow(["it1", "yes", "yes", "no"])                     # stray copy says no
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    out = tmp_path / "golden.csv"
    rc = _run_main(monkeypatch, ["--labeled", str(bad), str(b), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 2
    assert not out.exists()
    assert "duplicate header" in text and "labeled_a.csv" in text


def test_ragged_padding_rows_tolerated_and_ragged_content_refused_cleanly(tmp_path,
                                                                          monkeypatch,
                                                                          capsys):
    # Checker round 2: a padding row WIDER than the header (bare commas from Excel) put a
    # LIST under DictReader's None restkey and crashed with a raw AttributeError. Fully
    # blank ragged rows are now skipped; a blank-id row whose only content sits in an
    # overflow cell is refused cleanly, never a traceback.
    a = tmp_path / "labeled_a.csv"
    with a.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"{ITEM_ID},\"{Q_MENTIONS}\",\"{Q_SUPPORTS}\"\n")
        fh.write("it1,yes,yes\n")
        fh.write(",,,,,,\n")                                        # ragged padding row
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "yes", "yes")])
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--out", str(out)]) == 0
    capsys.readouterr()
    assert [r[ITEM_ID] for r in _read_dicts(out)] == ["it1"]

    c = tmp_path / "labeled_c.csv"
    with c.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"{ITEM_ID},\"{Q_MENTIONS}\",\"{Q_SUPPORTS}\"\n")
        fh.write("it1,yes,yes\n")
        fh.write(",,,,no,\n")                                       # content only in overflow
    rc = _run_main(monkeypatch, ["--labeled", str(c), str(b),
                                 "--out", str(tmp_path / "golden2.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "labeled_c.csv" in text and "item_id" in text            # clean refusal, no crash


def test_whitespace_variant_item_ids_are_refused_within_and_across_files(tmp_path,
                                                                         monkeypatch,
                                                                         capsys):
    # Checker round 2: the loader keys RAW ids, so 'it1' and 'it1 ' are two phantom items
    # to it. Inconsistent whitespace is refused with a message that names the cause.
    a = _write_labeled(tmp_path / "labeled_a.csv", [("it1", "yes", "yes"), ("it1 ", "no", "no")])
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "yes", "yes")])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b),
                                 "--out", str(tmp_path / "g1.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "whitespace" in text

    c = _write_labeled(tmp_path / "labeled_c.csv", [("it1 ", "yes", "yes")])   # trailing space
    rc = _run_main(monkeypatch, ["--labeled", str(b), str(c),
                                 "--out", str(tmp_path / "g2.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "whitespace" in text and "labeled_c.csv" in text


def test_worklist_with_blank_padding_rows_is_accepted(tmp_path, monkeypatch, capsys):
    # Checker round 2: the worklist is the one file a human MUST edit in Excel, and Excel
    # appends used-range padding rows of bare commas; those were a hard error before.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    with wl.open("a", newline="", encoding="utf-8") as fh:
        fh.write(",,,,\n,,,,,\n")                                   # padding, one ragged
    out = tmp_path / "golden.csv"
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    assert len(_read_dicts(out)) == 3


def test_two_empty_sheets_report_zero_items_not_byte_identity(tmp_path, monkeypatch, capsys):
    # Checker round 2: two header-only sheets are byte-identical by construction; the
    # honest refusal is "0 items", not an alias/copy accusation.
    a = _write_labeled(tmp_path / "labeled_x.csv", [])
    d = tmp_path / "two"
    d.mkdir()
    b = d / "labeled_y.csv"
    b.write_bytes(a.read_bytes())                                   # byte-identical, empty
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "0 items" in text
    assert "byte-identical" not in text


def test_zero_byte_leftover_at_out_is_overwritable(tmp_path, monkeypatch, capsys):
    # Checker round 2: a 0-byte leftover (e.g. from a crash) holds no human work; blocking
    # the rerun with "pick a fresh path" only causes path proliferation.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    out.touch()                                                     # 0-byte leftover
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    assert len(_read_dicts(out)) == 3


def test_stray_file_matching_the_old_fixed_tmp_name_survives(tmp_path, monkeypatch, capsys):
    # Checker round 2: a fixed <out>.tmp name would truncate an unrelated pre-existing
    # file; the unique mkstemp name must leave it untouched.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    stray = tmp_path / "golden.csv.tmp"
    stray.write_text("something the operator saved here", encoding="utf-8")
    before = stray.read_bytes()
    assert _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                   "--out", str(out)]) == 0
    capsys.readouterr()
    assert stray.read_bytes() == before


def test_summary_distinguishes_filled_adjudications_from_worklist_rows(tmp_path,
                                                                       monkeypatch, capsys):
    # Checker round 2 (test gap): lock the filled-vs-rows distinction on a fixture where
    # the two counts differ (1 filled of 2 rows); the summary prints before the refusal.
    a, b = _two_rater_fixture(tmp_path)
    wl = _write_worklist(tmp_path / "worklist.csv", ["a", "b"], [
        ("it2", "mentions", ["yes", "no"], "no"),
        ("it3", "supports", ["yes", "no"], ""),                     # not yet adjudicated
    ])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b), "--worklist", str(wl),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "1 filled adjudication(s) across 2 worklist row(s)" in text


def test_oversized_field_in_a_labeled_csv_is_refused_cleanly(tmp_path, monkeypatch, capsys):
    # Checker round 2 (test gap): exercise main's csv.Error arm on the LABELED path (the
    # worklist path wraps csv.Error into ValueError internally, so it never reached it).
    a = tmp_path / "labeled_a.csv"
    with a.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"{ITEM_ID},\"{Q_MENTIONS}\",\"{Q_SUPPORTS}\"\n")
        fh.write(f"it1,{'x' * 200_000},yes\n")
    b = _write_labeled(tmp_path / "labeled_b.csv", [("it1", "no", "yes")])
    rc = _run_main(monkeypatch, ["--labeled", str(a), str(b),
                                 "--out", str(tmp_path / "golden.csv")])
    text = capsys.readouterr().out
    assert rc == 2
    assert "[assemble]" in text


def test_write_is_atomic_a_failed_replace_leaves_the_previous_output_intact(tmp_path,
                                                                            monkeypatch,
                                                                            capsys):
    # LOW finding: a crash mid-write left a truncated file with complete-sheet schema.
    # The write now goes to a temp file + os.replace; a failed replace leaves the previous
    # output byte-identical and no temp file behind.
    a, b = _two_rater_fixture(tmp_path)
    wl = _filled_worklist(tmp_path)
    out = tmp_path / "golden.csv"
    argv = ["--labeled", str(a), str(b), "--worklist", str(wl), "--out", str(out)]
    assert _run_main(monkeypatch, argv) == 0
    capsys.readouterr()
    before = out.read_bytes()

    def boom(src, dst):
        raise OSError("simulated failure")

    monkeypatch.setattr(asm.os, "replace", boom)
    rc = _run_main(monkeypatch, argv)
    text = capsys.readouterr().out
    assert rc == 2
    assert "[assemble]" in text
    assert out.read_bytes() == before                        # previous output untouched
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(out.name + ".")]
    assert leftovers == []                                   # no temp file left behind
