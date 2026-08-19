"""Mechanically assemble the final golden-labels CSV from raters + the human adjudication.

ADDITIVE companion to score_labels.py and adjudicate_labels.py; it closes the last tooling
gap of labeling week and never replaces or amends the pre-committed golden-set analysis
(>= 50 transcripts, two independent human raters, pairwise Cohen's kappa; the frozen
PREREGISTRATION_*.md files and experiments/score_labels.py). Given the raw labeled_<name>.csv
files plus the human-FILLED adjudication worklist adjudicate_labels.py generated, it merges
them into one golden CSV by three mechanical rules:

1. UNANIMOUS item+question cells keep the raters' agreed vote.
2. SPLIT cells take the HUMAN adjudicated value VERBATIM: the value comes only from the
   worklist's `adjudicated` cell, parsed with the exact same yes/no token conventions every
   rater cell already gets, then canonicalized to yes/no on write. The tool never resolves
   a split itself: no majority vote, no tie-break, no inference. The model never labels
   and never adjudicates; adjudication is a human step this tool only APPLIES.
3. A split with a BLANK adjudicated cell, a single-vote cell, and an unlabeled cell are
   FLAGGED (blank label cell + a FLAGGED_* provenance value), never guessed. By default the
   tool REFUSES to write when anything would be flagged; --allow-partial writes the flagged
   sheet with an explicit `sheet_status` = PARTIAL column on every row, so a partial
   artifact can never pass as a finished golden set.

The output keeps the labeling-sheet headers (item_id + the two question columns), so the
frozen loader `score_labels.load_labeled` reads it directly, plus per-question provenance
columns (`mentions_source`, `supports_source`) recording which rule produced each cell.

    PYTHONPATH=src python experiments/assemble_golden_labels.py \
        --labeled results/labeled_alice.csv results/labeled_bob.csv \
        --worklist results/adjudication_worklist.csv \
        --out results/golden_labels.csv

No model calls, no network. Loading, classification, and the worklist format are REUSED from
the siblings (score_labels.load_labeled, adjudicate_labels.load_raters/classify/...), never
rewritten. Item ids come from the labeled CSVs themselves, so this module never reads the
sealed labeling key. Blinding is absolute: build and test against the schema
(labeling_columns) on SYNTHETIC data only; a human runs it on the real labels.

The worklist is validated before any adjudication applies: its rater columns must match the
labeled files, every row must still classify as a split under the current labels with the
same recorded votes (else the worklist is STALE and must be regenerated), duplicate header
columns are rejected (csv last-wins would silently drop a human decision), and a non-blank
adjudicated value that does not parse as yes/no is a hard error, since discarding it would lose
a human decision, coercing it would invent one.

Input hygiene guards the two-independent-raters anchor itself: byte-identical labeled
inputs are refused (one rater's file aliased or copied under a second name is not a second
rater), duplicate item rows inside one file are refused (the loader is last-wins), a golden
OUTPUT passed back as a rater input is refused, and a PARTIAL sheet can only be written to
a filename carrying "partial" (the frozen loader strips unknown columns, so the filename
must carry the marker too). The write is atomic (temp file + os.replace).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labeling_columns import ITEM_ID, Q_MENTIONS, Q_SUPPORTS  # noqa: E402
from score_labels import _to_bool, load_labeled  # noqa: E402,F401  (frozen-design loader; reused)
from adjudicate_labels import (  # noqa: E402  (reuse the worklist tool's conventions)
    ADJUDICATED_COL,
    QUESTION_COL,
    QUESTIONS,
    _cell,
    _same_file,
    classify,
    item_ids,
    load_raters,
    votes_for,
)

# One golden CSV column per question, reusing the exact labeling-sheet headers so the frozen
# loader reads the output; one provenance column per question beside it.
QUESTION_HEADER = {"mentions": Q_MENTIONS, "supports": Q_SUPPORTS}
SOURCE_COL = {"mentions": "mentions_source", "supports": "supports_source"}

# Provenance values. FLAGGED_* cells always carry a BLANK label; flagged means "a human
# still owes a decision or a label here", and this tool never fills one in.
SRC_UNANIMOUS = "unanimous"
SRC_ADJUDICATED = "adjudicated"
FLAG_MISSING_ADJ = "FLAGGED_missing_adjudication"
FLAG_SINGLE = "FLAGGED_single_vote"
FLAG_UNLABELED = "FLAGGED_unlabeled"
FLAG_REASON = {
    FLAG_MISSING_ADJ: "split with missing adjudication",
    FLAG_SINGLE: "single vote",
    FLAG_UNLABELED: "unlabeled",
}

SHEET_STATUS_COL = "sheet_status"  # present ONLY on partial outputs; PARTIAL on every row
PARTIAL = "PARTIAL"

MIN_RATERS = 2  # consensus over fewer than two raters is not consensus (frozen design: two)


def _row_cells(row: dict) -> list[str]:
    """Flatten a DictReader row to plain string cells.

    On a ragged row (more cells than headers, e.g. an Excel padding row of bare commas),
    DictReader parks the overflow as a LIST under the None restkey; treating that list as
    a string is an AttributeError waiting to happen, so every cell test goes through here.
    """
    cells: list[str] = []
    for key, value in row.items():
        if key is None:
            cells.extend(v or "" for v in (value or []))
        else:
            cells.append(value or "")
    return cells


def _is_blank_row(row: dict) -> bool:
    """True for a fully blank row (Excel used-range padding), overflow cells included."""
    return all(not c.strip() for c in _row_cells(row))


def _reject_duplicate_headers(fields: list[str], name: str) -> None:
    """csv.DictReader is last-wins on duplicate headers: a duplicated column silently
    drops one copy's cells (a rater's real vote, or a human adjudication). Hard error."""
    if len(fields) != len(set(fields)):
        dupes = sorted({c for c in fields if fields.count(c) > 1})
        raise ValueError(
            f"{name} has duplicate header column(s) {dupes}; a duplicated column "
            "silently drops one copy's cells; fix the header"
        )


def _precheck_labeled(paths: list[Path]) -> None:
    """Validate the labeled CSVs BEFORE the last-wins loader ever sees them.

    load_labeled keys rows on item_id last-wins, so problems it would silently absorb are
    refused here instead: a duplicated item row inside one file (a contradictory duplicate
    would rewrite that rater's vote), a row carrying votes but no item_id (a phantom item;
    fully blank Excel padding rows are tolerated), and a previous golden OUTPUT passed back
    as a rater input (its provenance columns give it away). Finally, two byte-identical
    inputs are refused outright: one human's file aliased or copied under a second rater
    name would mint a "two-rater" consensus from ONE rater, which is the exact fraud the
    frozen two-independent-raters design exists to rule out.
    """
    n_data_rows: dict[Path, int] = {}
    # stripped id -> (raw id, filename): whitespace-variant ids must be consistent across
    # files too, or the loader's raw keys silently split one item into phantom singles.
    global_ids: dict[str, tuple[str, str]] = {}
    for p in paths:
        try:
            with p.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                fields = reader.fieldnames or []
                if any(c in fields for c in SOURCE_COL.values()):
                    raise ValueError(
                        f"{p.name} is a golden-labels output of this tool, not a rater "
                        "label CSV; pass the raw labeled_<name>.csv files"
                    )
                if ITEM_ID not in fields:
                    raise ValueError(
                        f"cannot load {p.name}: missing the {ITEM_ID!r} header (expected "
                        "a filled label CSV with the labeling-sheet headers)"
                    )
                _reject_duplicate_headers(fields, p.name)
                seen: dict[str, str] = {}  # stripped id -> raw id within this file
                for i, row in enumerate(reader, start=2):
                    raw = row.get(ITEM_ID) or ""
                    item = raw.strip()
                    if not item:
                        if _is_blank_row(row):
                            continue  # fully blank padding row (Excel artifact)
                        raise ValueError(
                            f"{p.name} line {i} has cell content but no item_id; "
                            "fix or delete the row"
                        )
                    if item in seen:
                        if seen[item] == raw:
                            raise ValueError(
                                f"{p.name} has duplicate rows for item {item!r}; the "
                                "loader would keep only the last one; deduplicate the "
                                "file first"
                            )
                        raise ValueError(
                            f"{p.name} has rows whose item ids differ only by "
                            f"surrounding whitespace ({seen[item]!r} vs {raw!r}); "
                            "normalize the item_id cells"
                        )
                    seen[item] = raw
                    known = global_ids.get(item)
                    if known and known[0] != raw:
                        raise ValueError(
                            f"item id {item!r} appears with inconsistent surrounding "
                            f"whitespace across labeled files ({known[0]!r} in {known[1]} "
                            f"vs {raw!r} in {p.name}); normalize the item_id cells"
                        )
                    if not known:
                        global_ids[item] = (raw, p.name)
                n_data_rows[p] = len(seen)
        except OSError as e:
            raise ValueError(
                f"cannot load {p.name}: {e!r} (expected a filled label CSV)"
            ) from e
    for p1, p2 in combinations(paths, 2):
        if n_data_rows[p1] == 0 and n_data_rows[p2] == 0:
            continue  # two empty sheets: the zero-item refusal is the honest message
        if p1.read_bytes() == p2.read_bytes():
            raise ValueError(
                f"labeled inputs {p1.name} and {p2.name} are byte-identical; independent "
                "raters produce distinct files; check that each rater's own CSV was "
                "passed, not an alias or a copy of one"
            )


def load_worklist(path: Path, rater_names: list[str]) -> dict[tuple[str, str], dict]:
    """Load the human-filled adjudication worklist, validating its shape.

    Returns {(item_id, question): {"votes": {rater: cell}, "adjudicated": raw_cell}}.
    Raises ValueError on a missing/foreign header, a rater-column set that does not match
    the labeled files, an unknown question value, or a duplicate (item, question) row.
    Staleness against the current labels is checked separately by verify_worklist.
    """
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:  # tolerate a BOM from Excel
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            for required in (ITEM_ID, QUESTION_COL, ADJUDICATED_COL):
                if required not in fields:
                    raise ValueError(
                        f"{path.name} does not look like an adjudication worklist "
                        f"(missing the {required!r} column)"
                    )
            _reject_duplicate_headers(fields, f"worklist {path.name}")
            wl_raters = [c for c in fields if c not in (ITEM_ID, QUESTION_COL, ADJUDICATED_COL)]
            if set(wl_raters) != set(rater_names):
                raise ValueError(
                    f"worklist rater columns {sorted(wl_raters)} do not match the labeled "
                    f"files' raters {sorted(rater_names)}; regenerate the worklist from the "
                    "same labeled CSVs"
                )
            out: dict[tuple[str, str], dict] = {}
            for row in reader:
                if _is_blank_row(row):
                    continue  # Excel used-range padding row; the human edits this file
                item, q = row[ITEM_ID], row[QUESTION_COL]
                if q not in QUESTIONS:
                    raise ValueError(f"worklist row ({item!r}) has unknown question {q!r}")
                if (item, q) in out:
                    raise ValueError(f"worklist has duplicate rows for ({item}, {q})")
                out[(item, q)] = {
                    "votes": {r: (row.get(r) or "").strip() for r in wl_raters},
                    "adjudicated": (row.get(ADJUDICATED_COL) or "").strip(),
                }
            return out
    except (OSError, csv.Error) as e:
        raise ValueError(f"cannot read worklist {path.name}: {e!r}") from e


def verify_worklist(worklist: dict[tuple[str, str], dict],
                    raters: list[tuple[str, dict]]) -> None:
    """Every worklist row must still describe a CURRENT split with the recorded votes.

    A row for an item nobody labeled, for a pair that no longer classifies as split, or
    whose recorded votes differ from the current labels means the labels changed after the
    worklist was generated, so the human's adjudications may no longer apply: the merge
    refuses and a human regenerates the worklist (adjudicate_labels.py refuses to
    clobber a filled one, so nothing is lost by re-running it to compare).
    """
    known = set(item_ids(raters))
    for (item, q), entry in worklist.items():
        if item not in known:
            raise ValueError(
                f"stale worklist: row ({item}, {q}) names an item no labeled CSV contains"
            )
        votes = votes_for(raters, item, q)
        if classify(votes) != "split":
            raise ValueError(
                f"stale worklist: ({item}, {q}) is no longer a split under the current "
                "labels; regenerate the worklist"
            )
        current = {name: _cell(v) for (name, _), v in zip(raters, votes)}
        if current != entry["votes"]:
            raise ValueError(
                f"stale worklist: recorded votes for ({item}, {q}) do not match the "
                "current labeled CSVs; regenerate the worklist"
            )


def _resolve_cell(votes: list[bool | None],
                  wl_entry: dict | None,
                  item: str, q: str) -> tuple[bool | None, str]:
    """Resolve one item+question cell to (value, provenance) by the three mechanical rules.

    Never decides: a split resolves ONLY through a parseable human adjudication; everything
    unresolved comes back as (None, FLAGGED_*). An unparseable non-blank adjudication
    raises, because it is a human-input error to surface, not a blank to skip.
    """
    cls = classify(votes)
    if cls == "unanimous":
        return next(v for v in votes if v is not None), SRC_UNANIMOUS
    if cls == "split":
        raw = wl_entry["adjudicated"] if wl_entry else ""
        if not raw:
            return None, FLAG_MISSING_ADJ
        value = _to_bool(raw)
        if value is None:
            raise ValueError(
                f"unparseable adjudicated value {raw!r} for ({item}, {q}); "
                "expected yes/no; fix the worklist cell (the tool never coerces or "
                "discards a human decision)"
            )
        return value, SRC_ADJUDICATED
    if cls == "single":
        return None, FLAG_SINGLE
    return None, FLAG_UNLABELED


def assemble(raters: list[tuple[str, dict]],
             worklist: dict[tuple[str, str], dict]) -> tuple[list[dict], dict[str, int]]:
    """Merge raters + human adjudications into per-item golden rows, counting provenance.

    Returns (rows, counts): rows are {item, values: {q: bool|None}, sources: {q: str}} in
    sorted item order; counts tally every provenance value across all cells.
    """
    rows: list[dict] = []
    counts = dict.fromkeys(
        (SRC_UNANIMOUS, SRC_ADJUDICATED, FLAG_MISSING_ADJ, FLAG_SINGLE, FLAG_UNLABELED), 0
    )
    for item in item_ids(raters):
        if not item.strip():
            # A blank id can only come from a fully blank Excel padding row (a blank id
            # WITH content is refused by _precheck_labeled); never merge a phantom item.
            continue
        values: dict[str, bool | None] = {}
        sources: dict[str, str] = {}
        for q in QUESTIONS:
            value, source = _resolve_cell(votes_for(raters, item, q),
                                          worklist.get((item, q)), item, q)
            values[q], sources[q] = value, source
            counts[source] += 1
        rows.append({"item": item, "values": values, "sources": sources})
    return rows, counts


def write_golden(out: Path, rows: list[dict], *, partial: bool) -> None:
    """Write the golden CSV atomically: temp file in the same directory, then os.replace.

    A crash mid-write must never leave a truncated file wearing the complete-sheet schema;
    with the temp+replace dance the destination either keeps its previous bytes or gets the
    whole new sheet.
    """
    header = [ITEM_ID]
    for q in QUESTIONS:
        header += [QUESTION_HEADER[q], SOURCE_COL[q]]
    if partial:
        header.append(SHEET_STATUS_COL)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A unique temp name (never a fixed one): a fixed name could truncate an unrelated
    # pre-existing file and makes two overlapping runs race on the same path.
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".", suffix=".tmp")
    os.fchmod(fd, 0o644)  # mkstemp defaults to 0600; the golden CSV is a normal data file
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for row in rows:
                cells = [row["item"]]
                for q in QUESTIONS:
                    value = row["values"][q]
                    cells += ["" if value is None else ("yes" if value else "no"),
                              row["sources"][q]]
                if partial:
                    cells.append(PARTIAL)
                w.writerow(cells)
        os.replace(tmp, out)
    finally:
        try:
            tmp.unlink(missing_ok=True)  # best-effort cleanup: never outrank the real error
        except OSError:
            pass


def _is_previous_golden_output(path: Path) -> bool:
    """True if an existing file at --out was written by this tool (provenance columns).

    Only a previous golden output may be overwritten: the merge is a deterministic
    function of its inputs, so regenerating it is safe. Anything else at --out (a label
    CSV under a different name, a filled worklist, a sheet) is un-versioned human work
    (results/ is gitignored for blinding) and gets a refusal, not a rewrite.
    """
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            fields = next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError, csv.Error):
        return False  # unreadable or not CSV-shaped -> not ours -> the caller refuses
    return all(c in fields for c in (ITEM_ID, *SOURCE_COL.values()))


def _print_summary(names: list[str], n_items: int, counts: dict[str, int],
                   n_filled: int, n_rows: int) -> None:
    print(f"[assemble] {len(names)} raters {names} over {n_items} items; "
          f"{n_filled} filled adjudication(s) across {n_rows} worklist row(s)\n")
    print("== golden-cell provenance (per item+question cell) ==")
    print(f"  unanimous {counts[SRC_UNANIMOUS]}   adjudicated {counts[SRC_ADJUDICATED]}")
    flagged = [(FLAG_REASON[f], counts[f]) for f in FLAG_REASON if counts[f]]
    for reason, n in flagged:
        print(f"  FLAGGED ({reason}): {n}")
    print("\n  this tool never decides a label: splits resolve only through the human-filled")
    print("  adjudicated column, and every unresolved cell above is flagged blank, not guessed.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", type=Path, nargs="+", required=True,
                    help="filled label CSVs, one per rater (>= 2)")
    ap.add_argument("--worklist", type=Path, default=None,
                    help="human-FILLED adjudication worklist from adjudicate_labels.py "
                         "(omit only if no split exists)")
    ap.add_argument("--out", type=Path, required=True,
                    help="where to write the golden-labels CSV")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write even with FLAGGED cells, marking every row PARTIAL "
                         "(flagged label cells stay blank; nothing is ever guessed)")
    a = ap.parse_args()

    inputs = list(a.labeled) + ([a.worklist] if a.worklist else [])
    if any(_same_file(a.out, p) for p in inputs):
        print(f"[assemble] refusing to write the golden CSV over an input file: {a.out}")
        return 2
    # A 0-byte file at --out holds no human work (e.g. a crash leftover) and may be
    # overwritten; anything non-empty that is not our own previous output is refused.
    if (a.out.exists() and a.out.stat().st_size > 0
            and not _is_previous_golden_output(a.out)):
        print(f"[assemble] refusing to overwrite {a.out}: it is not a previous "
              "golden-labels output of this tool; pick a fresh path.")
        return 2

    try:
        _precheck_labeled(a.labeled)
        raters = load_raters(a.labeled)
        if len(raters) < MIN_RATERS:
            raise ValueError(
                f"consensus needs >= {MIN_RATERS} raters (the frozen design uses two "
                f"independent human raters); got {len(raters)}"
            )
        worklist = load_worklist(a.worklist, [n for n, _ in raters]) if a.worklist else {}
        verify_worklist(worklist, raters)
        rows, counts = assemble(raters, worklist)
    except (ValueError, csv.Error) as e:
        print(f"[assemble] {e}")
        return 2

    if not rows:
        print("[assemble] REFUSING to write: the merge produced 0 items (every input was "
              "empty of data rows); an empty sheet is never a golden set.")
        return 2

    names = [n for n, _ in raters]
    n_filled = sum(1 for e in worklist.values() if e["adjudicated"])
    _print_summary(names, len(rows), counts, n_filled, len(worklist))

    n_flagged = sum(counts[f] for f in FLAG_REASON)
    if n_flagged and not a.allow_partial:
        breakdown = ", ".join(f"{counts[f]} {FLAG_REASON[f]}" for f in FLAG_REASON
                              if counts[f])
        print(f"\n[assemble] REFUSING to write: {n_flagged} cell(s) would be flagged "
              f"({breakdown}).")
        print("  finish the human adjudication / labeling and re-run, or pass "
              "--allow-partial to write an explicitly PARTIAL sheet.")
        return 2

    partial = bool(n_flagged)
    if partial and "partial" not in a.out.name.lower():
        print(f"\n[assemble] REFUSING to write a PARTIAL sheet to {a.out.name}: the "
              "frozen loader ignores unknown columns, so the in-file marker alone can be "
              "lost on load. Name the file so it cannot be mistaken for a finished golden "
              "set, e.g. golden_labels.PARTIAL.csv.")
        return 2
    try:
        write_golden(a.out, rows, partial=partial)
    except OSError as e:
        print(f"[assemble] could not write {a.out}: {e!r} (previous output, if any, "
              "is untouched)")
        return 2
    if partial:
        print(f"\nwrote {len(rows)} item(s) to {a.out}. PARTIAL: {n_flagged} flagged "
              f"cell(s) left blank; every row carries {SHEET_STATUS_COL}={PARTIAL}.")
        print("  this sheet is NOT a finished golden set.")
    else:
        print(f"\nwrote {len(rows)} item(s) to {a.out} (complete: every cell unanimous "
              "or human-adjudicated).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
