"""Ported input-content guards on adjudicate_labels.load_raters.

The consensus-assembly tool (assemble_golden_labels) refuses untrustworthy labeled CSVs
before its last-wins loader runs: byte-identical inputs (one human's file aliased as a
second rater), duplicate header columns (csv.DictReader last-wins drops a vote column's
cells), duplicate or whitespace-variant item rows, rows with cells but no item_id, and a
golden OUTPUT passed back as a rater input. The worklist generator feeds the SAME human
adjudication step from the same files, so its load_raters now calls the assembler's one
guard implementation (shared, not copied); these tests pin that the refusals fire on
the worklist path too, that clean inputs still load, and that the call is genuinely the
shared function rather than a drifting copy.

RATER BLINDING IS ABSOLUTE. Nothing here reads any real label/key artifact; every fixture
is SYNTHETIC data written into pytest's tmp_path, exactly like tests/test_adjudicate_labels.py.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adjudicate = _load("adjudicate_labels_guards_under_test", "adjudicate_labels.py")
assemble = _load("assemble_golden_labels_for_guards", "assemble_golden_labels.py")

sys.path.insert(0, str(REPO / "experiments"))
from labeling_columns import ITEM_ID, Q_MENTIONS, Q_SUPPORTS  # noqa: E402


def _write_labeled(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    cols = [ITEM_ID, Q_MENTIONS, Q_SUPPORTS]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for item_id, mentions, supports in rows:
            w.writerow({ITEM_ID: item_id, Q_MENTIONS: mentions, Q_SUPPORTS: supports})
    return path


def test_two_distinct_valid_files_still_load(tmp_path):
    a = _write_labeled(tmp_path / "labeled_ann.csv", [("i1", "yes", "no")])
    b = _write_labeled(tmp_path / "labeled_bob.csv", [("i1", "no", "no")])
    raters = adjudicate.load_raters([a, b])
    assert [n for n, _ in raters] == ["ann", "bob"]


def test_byte_identical_pair_refused(tmp_path):
    a = _write_labeled(tmp_path / "labeled_ann.csv", [("i1", "yes", "no")])
    b = tmp_path / "labeled_bob.csv"
    b.write_bytes(a.read_bytes())
    with pytest.raises(ValueError, match="byte-identical"):
        adjudicate.load_raters([a, b])


def test_hardlink_alias_refused(tmp_path):
    a = _write_labeled(tmp_path / "labeled_ann.csv", [("i1", "yes", "no")])
    b = tmp_path / "labeled_bob.csv"
    b.hardlink_to(a)
    with pytest.raises(ValueError, match="byte-identical"):
        adjudicate.load_raters([a, b])


def test_duplicate_header_column_refused(tmp_path):
    p = tmp_path / "labeled_ann.csv"
    p.write_text(
        f"{ITEM_ID},{Q_MENTIONS},{Q_MENTIONS},{Q_SUPPORTS}\ni1,yes,no,no\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate header"):
        adjudicate.load_raters([p])


def test_duplicate_item_rows_refused(tmp_path):
    p = _write_labeled(
        tmp_path / "labeled_ann.csv", [("i1", "yes", "no"), ("i1", "no", "no")]
    )
    with pytest.raises(ValueError, match="duplicate rows"):
        adjudicate.load_raters([p])


def test_whitespace_variant_item_ids_within_a_file_refused(tmp_path):
    p = _write_labeled(
        tmp_path / "labeled_ann.csv", [("i1", "yes", "no"), (" i1", "no", "no")]
    )
    with pytest.raises(ValueError, match="whitespace"):
        adjudicate.load_raters([p])


def test_whitespace_variant_item_ids_across_files_refused(tmp_path):
    a = _write_labeled(tmp_path / "labeled_ann.csv", [("i1", "yes", "no")])
    b = _write_labeled(tmp_path / "labeled_bob.csv", [("i1 ", "no", "no")])
    with pytest.raises(ValueError, match="whitespace"):
        adjudicate.load_raters([a, b])


def test_phantom_row_with_cells_but_no_item_id_refused(tmp_path):
    p = _write_labeled(
        tmp_path / "labeled_ann.csv", [("i1", "yes", "no"), ("", "yes", "")]
    )
    with pytest.raises(ValueError, match="no item_id"):
        adjudicate.load_raters([p])


def test_fully_blank_padding_row_tolerated(tmp_path):
    p = _write_labeled(
        tmp_path / "labeled_ann.csv", [("i1", "yes", "no"), ("", "", "")]
    )
    b = _write_labeled(tmp_path / "labeled_bob.csv", [("i1", "no", "yes")])
    raters = adjudicate.load_raters([p, b])
    assert [n for n, _ in raters] == ["ann", "bob"]


def test_padding_row_never_mints_a_phantom_item(tmp_path, monkeypatch, capsys):
    # A tolerated blank padding row mints an empty-string key in load_labeled's dict;
    # item_ids must skip it so the worklist tool counts the same item universe as the
    # assembler (pre-fix: items inflated by one, plus a spurious none/incomplete and a
    # Fleiss "excluded" row).
    a = _write_labeled(
        tmp_path / "labeled_ann.csv", [("i1", "yes", "no"), ("", "", "")]
    )
    b = _write_labeled(tmp_path / "labeled_bob.csv", [("i1", "no", "no")])
    raters = adjudicate.load_raters([a, b])
    assert adjudicate.item_ids(raters) == ["i1"]
    out = tmp_path / "worklist.csv"
    monkeypatch.setattr(
        sys, "argv",
        ["adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(out)],
    )
    assert adjudicate.main() == 0
    printed = capsys.readouterr().out
    assert "over 1 items" in printed


def test_golden_output_passed_as_rater_input_refused(tmp_path):
    p = tmp_path / "labeled_ann.csv"
    source_col = list(assemble.SOURCE_COL.values())[0]
    p.write_text(
        f"{ITEM_ID},{Q_MENTIONS},{Q_SUPPORTS},{source_col}\ni1,yes,no,unanimous\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="golden-labels output"):
        adjudicate.load_raters([p])


def test_guard_is_the_shared_assemble_implementation_not_a_copy(tmp_path, monkeypatch):
    # load_raters must route through assemble_golden_labels._precheck_labeled itself:
    # replace it under the module name the deferred `import assemble_golden_labels`
    # resolves to and confirm the sentinel fires. That pins reuse; a silent copy would
    # keep passing content checks while drifting from the reviewed implementation.
    import assemble_golden_labels as canonical

    calls: list[list[Path]] = []

    def sentinel(paths):
        calls.append(list(paths))
        raise ValueError("sentinel-guard-called")

    monkeypatch.setattr(canonical, "_precheck_labeled", sentinel)
    a = _write_labeled(tmp_path / "labeled_ann.csv", [("i1", "yes", "no")])
    with pytest.raises(ValueError, match="sentinel-guard-called"):
        adjudicate.load_raters([a])
    assert calls == [[a]]


def test_cli_reports_refusal_cleanly_without_traceback(tmp_path, monkeypatch, capsys):
    a = _write_labeled(tmp_path / "labeled_ann.csv", [("i1", "yes", "no")])
    b = tmp_path / "labeled_bob.csv"
    b.write_bytes(a.read_bytes())
    out = tmp_path / "worklist.csv"
    monkeypatch.setattr(
        sys, "argv",
        ["adjudicate_labels.py", "--labeled", str(a), str(b), "--out", str(out)],
    )
    assert adjudicate.main() == 2
    printed = capsys.readouterr().out
    assert "[adjudicate]" in printed and "byte-identical" in printed
    assert not out.exists()
