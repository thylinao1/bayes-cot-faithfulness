"""Surface human-labeling disagreements for adjudication (+ a supplementary Fleiss' kappa).

ADDITIVE companion to score_labels.py; it never replaces or amends the pre-committed
golden-set analysis. The frozen design fixes >= 50 transcripts, two independent human
raters, and pairwise Cohen's kappa (experiments/score_labels.py, and the
PREREGISTRATION_*.md files). This tool adds two things on top of that anchor:

1. A disagreement worklist. Given N filled label CSVs, for each item and each question
   (mentions = Q1, supports = Q2) it classifies the raters' votes as UNANIMOUS or SPLIT and
   writes ONLY the split items to a worklist CSV for a human to adjudicate. It SURFACES the
   disagreements; it never resolves them. The `adjudicated` column is always written blank:
   this tool never fills it, and the model is never a rater and never decides a label.
   Adjudication is a human step done after independent labeling.

2. A SUPPLEMENTARY Fleiss' kappa over the load-bearing `mentions` judgment for the
   multi-rater (>= 3) case, printed as a line clearly subordinate to the pre-committed
   pairwise Cohen's kappa.

    PYTHONPATH=src python experiments/adjudicate_labels.py \
        --labeled results/labeled_alice.csv results/labeled_bob.csv results/labeled_cara.csv \
        --out results/adjudication_worklist.csv

No model calls, no network. The loader (score_labels.load_labeled) and the item-id / column
conventions are reused, not rewritten. Item ids come from the labeled CSVs themselves, so
this module never reads the sealed labeling key. Blinding is absolute: build and test against
the schema (labeling_columns) on SYNTHETIC data only; the operator runs it on the real labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labeling_columns import ITEM_ID  # noqa: E402  (reuse the item-id column convention)
from score_labels import load_labeled  # noqa: E402  (reuse the frozen-design loader; do not rewrite)

# The two human judgments, in the order they appear on the worklist. These are the keys
# load_labeled returns, mapped to the guide's question numbers for the human-facing summary.
QUESTIONS = ("mentions", "supports")
QUESTION_LABEL = {"mentions": "Q1 mentions", "supports": "Q2 supports"}

QUESTION_COL = "question"
ADJUDICATED_COL = "adjudicated"  # ALWAYS blank on write: the tool never resolves a split.

MIN_RATERS_FLEISS = 3  # Fleiss is the >= 3-rater companion; 2 raters -> pairwise Cohen's kappa.

_YES, _NO, _BLANK = "yes", "no", ""


def _cell(vote: bool | None) -> str:
    """Render one rater's vote for the worklist: yes / no / blank (blank == no vote)."""
    if vote is True:
        return _YES
    if vote is False:
        return _NO
    return _BLANK


def _rater_name(path: Path) -> str:
    """Rater label from the filename, matching score_labels' `labeled_<name>.csv` convention."""
    return path.stem.replace("labeled_", "")


# Fixed worklist columns a rater name must never shadow (a collision would produce a
# duplicate CSV header and make dict-based reads of the worklist silently ambiguous).
RESERVED_RATER_NAMES = {ITEM_ID, QUESTION_COL, ADJUDICATED_COL}


def load_raters(paths: list[Path]) -> list[tuple[str, dict]]:
    """Load each labeled CSV, preserving input order, failing fast on bad inputs.

    Rejects two inputs mapping to the same rater name (each rater needs its own worklist
    column; a collision almost always means the wrong files were passed), a rater name that
    shadows a fixed worklist column, and a file that cannot be read as a filled label CSV
    (converted to ValueError so the CLI reports it instead of a raw traceback).

    Content guards run FIRST, shared with (not copied from) the consensus-assembly tool:
    byte-identical inputs (one human's file aliased as a second rater), duplicate header
    columns (DictReader last-wins silently drops a vote column), duplicate or
    whitespace-variant item rows (the last-wins loader would rewrite or split a vote),
    rows with cells but no item_id, and a golden OUTPUT passed back as a rater input.
    The same untrustworthy files must be refused whether they reach the worklist
    generator or the assembler, so both call the one implementation.
    """
    # Deferred sibling import: assemble_golden_labels imports functions from THIS module
    # at its top level, so importing it here at call time is what avoids the cycle.
    import assemble_golden_labels as _assemble
    _assemble._precheck_labeled(paths)
    raters: list[tuple[str, dict]] = []
    seen: dict[str, Path] = {}
    for p in paths:
        name = _rater_name(p)
        if name in RESERVED_RATER_NAMES:
            raise ValueError(
                f"rater name {name!r} (from {p.name}) collides with a fixed worklist "
                "column; rename the file"
            )
        if name in seen:
            raise ValueError(
                f"two inputs map to rater {name!r} ({seen[name].name} and {p.name}); "
                "give each rater a distinct labeled_<name>.csv"
            )
        try:
            data = load_labeled(p)
        except (OSError, KeyError) as e:
            raise ValueError(
                f"cannot load {p.name}: {e!r} (expected a filled label CSV with the "
                "labeling-sheet headers)"
            ) from e
        seen[name] = p
        raters.append((name, data))
    return raters


def item_ids(raters: list[tuple[str, dict]]) -> list[str]:
    """Sorted union of every item id any rater labeled (never read from the sealed key).

    A tolerated blank Excel padding row mints an empty-string key in load_labeled's
    last-wins dict; the assembler skips it when merging, and the counts here must
    describe the same item universe, so it is skipped here too (a padding row would
    otherwise inflate the item / none / incomplete / Fleiss-excluded counts by one).
    """
    ids: set[str] = set()
    for _, data in raters:
        ids.update(k for k in data if k.strip())
    return sorted(ids)


def votes_for(raters: list[tuple[str, dict]], item: str, question: str) -> list[bool | None]:
    """Each rater's vote (True/False/None) on one item+question, in rater order.

    A rater who never labeled the item, or left the cell blank, is None either way.
    """
    return [data.get(item, {}).get(question) for _, data in raters]


def classify(votes: list[bool | None]) -> str:
    """Classify one item+question from the raters' votes.

    split      -> at least two raters gave OPPOSING yes/no votes (a genuine disagreement).
    unanimous  -> >= 2 raters voted and they all agree.
    single     -> exactly one rater voted (no disagreement is possible).
    none       -> no rater voted.

    Missing (None) votes never create a split; a split needs two present, opposing votes.
    """
    present = [v for v in votes if v is not None]
    if len(set(present)) >= 2:
        return "split"
    if len(present) >= 2:
        return "unanimous"
    if len(present) == 1:
        return "single"
    return "none"


def summarize(raters: list[tuple[str, dict]]) -> dict[str, dict[str, int]]:
    """Per question, count split / unanimous / single / none items, plus incomplete items.

    `incomplete` counts items where at least one rater left the judgment blank; it overlaps
    the other buckets on purpose (an item can be split AND incomplete) and is reported so the
    coordinator knows where labels are still missing.
    """
    ids = item_ids(raters)
    stats: dict[str, dict[str, int]] = {}
    for q in QUESTIONS:
        counts = {"split": 0, "unanimous": 0, "single": 0, "none": 0, "incomplete": 0}
        for item in ids:
            votes = votes_for(raters, item, q)
            counts[classify(votes)] += 1
            if any(v is None for v in votes):
                counts["incomplete"] += 1
        stats[q] = counts
    return stats


def build_worklist(raters: list[tuple[str, dict]]) -> tuple[list[str], list[list[str]]]:
    """Build the human-adjudication worklist: ONLY the split item+question rows.

    Columns: item_id, question, <one column per rater's vote>, adjudicated. The adjudicated
    column is always blank -- this tool surfaces disagreements, it never resolves them.
    """
    names = [n for n, _ in raters]
    header = [ITEM_ID, QUESTION_COL, *names, ADJUDICATED_COL]
    rows: list[list[str]] = []
    for item in item_ids(raters):
        for q in QUESTIONS:
            votes = votes_for(raters, item, q)
            if classify(votes) == "split":
                rows.append([item, q, *[_cell(v) for v in votes], _BLANK])  # adjudicated blank
    return header, rows


def fleiss_kappa(rows: list[list[int]]) -> float | None:
    """Fleiss' kappa over a category-count matrix (stdlib only).

    rows[i][j] = number of raters who assigned item i to category j. Every row must sum to
    the same n (a constant rater count per item, as Fleiss' kappa requires); the caller
    enforces this by using complete-case items only. Returns None when there are no items or
    n < 2. For the degenerate single-category case (expected agreement 1.0, formally
    undefined) it mirrors score_labels.cohen_kappa: 1.0 iff observed agreement is perfect.
    """
    if not rows:
        return None
    # Enforce BOTH shape contracts BEFORE the n < 2 early return, so a ragged matrix always
    # raises rather than slipping through (or being silently mis-scored: k is taken from
    # rows[0], so a wider later row would leak categories out of P_e but not P_bar).
    if len({len(r) for r in rows}) > 1:
        raise ValueError("Fleiss' kappa needs the same category count on every row")
    sizes = {sum(r) for r in rows}
    if len(sizes) > 1:
        raise ValueError("Fleiss' kappa needs the same rater count on every item")
    n = sizes.pop()
    if n < 2:
        return None
    k = len(rows[0])
    total = len(rows) * n
    # P-bar: mean per-item agreement, from the sum-of-squares form of each row.
    p_bar = sum((sum(c * c for c in r) - n) / (n * (n - 1)) for r in rows) / len(rows)
    # P_e: expected agreement from the category proportions.
    p_e = sum((sum(r[j] for r in rows) / total) ** 2 for j in range(k))
    if p_e == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0
    return (p_bar - p_e) / (1 - p_e)


def mentions_fleiss(raters: list[tuple[str, dict]]) -> tuple[float | None, int, int]:
    """Supplementary Fleiss' kappa on the load-bearing `mentions` judgment.

    Returns (kappa, n_used, n_excluded). Only items every rater labeled on `mentions` are
    used, so the rater count per item is constant; items with any missing `mentions` label
    are excluded and counted. Returns kappa=None with fewer than MIN_RATERS_FLEISS raters or
    no complete-case item. This does NOT replace the pre-committed pairwise Cohen's kappa.
    """
    rows: list[list[int]] = []
    n_excluded = 0
    for item in item_ids(raters):
        votes = votes_for(raters, item, "mentions")
        if any(v is None for v in votes):
            n_excluded += 1
            continue
        rows.append([sum(v is False for v in votes), sum(v is True for v in votes)])
    if len(raters) < MIN_RATERS_FLEISS or not rows:
        return None, len(rows), n_excluded
    return fleiss_kappa(rows), len(rows), n_excluded


def _print_summary(stats: dict[str, dict[str, int]]) -> None:
    print("== disagreement summary (per question; split = raters gave opposing yes/no) ==")
    for q in QUESTIONS:
        c = stats[q]
        colabeled = c["split"] + c["unanimous"]
        rate = (
            f"{c['split']}/{colabeled} = {100 * c['split'] / colabeled:.1f}%"
            if colabeled
            else "n/a (no item had >= 2 votes)"
        )
        print(f"  {QUESTION_LABEL[q]}: unanimous {c['unanimous']}  split {c['split']}  "
              f"(split rate {rate})")
        extras = []
        if c["single"]:
            extras.append(f"{c['single']} single-vote")
        if c["none"]:
            extras.append(f"{c['none']} unlabeled")
        if c["incomplete"]:
            extras.append(f"{c['incomplete']} incomplete (>= 1 rater left it blank)")
        if extras:
            print(f"      ({', '.join(extras)})")


def _print_fleiss(raters: list[tuple[str, dict]]) -> None:
    print("\n== SUPPLEMENTARY: Fleiss' kappa (>= 3 raters, load-bearing 'mentions') ==")
    print("  Subordinate to the pre-committed analysis. The golden-set design fixes pairwise")
    print("  Cohen's kappa on two independent raters (experiments/score_labels.py); Fleiss is")
    print("  an additive multi-rater companion, never a replacement or an amendment.")
    kappa, n_used, n_excluded = mentions_fleiss(raters)
    if kappa is None:
        if len(raters) < MIN_RATERS_FLEISS:
            print(f"  Fleiss' kappa: n/a (needs >= {MIN_RATERS_FLEISS} raters; have {len(raters)}).")
        else:
            print("  Fleiss' kappa: n/a (no item was labeled on 'mentions' by every rater).")
    else:
        print(f"  Fleiss' kappa (mentions) = {kappa:.3f}   "
              f"[complete-case items: {n_used} used, {n_excluded} excluded for a missing label]")


def _same_file(out: Path, inp: Path) -> bool:
    """True when out and inp name the same file, including case variants on a
    case-insensitive filesystem (macOS APFS), symlinks, and hardlinks. Path.resolve()
    alone does not case-normalize, so an inode comparison backs it up."""
    if out.resolve() == inp.resolve():
        return True
    try:
        return out.samefile(inp)
    except OSError:
        return False  # out does not exist yet, so it cannot be an existing input


def _worklist_has_adjudications(path: Path) -> bool:
    """True if an existing worklist CSV already carries human adjudication decisions.

    Guards the human's work: results/ is gitignored (for blinding), so a filled worklist
    has no VCS safety net and a silent rewrite would destroy it. Unreadable or non-worklist
    files return False (overwriting those is the operator's explicit choice via --out)."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or ADJUDICATED_COL not in reader.fieldnames:
                return False
            return any((row.get(ADJUDICATED_COL) or "").strip() for row in reader)
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", type=Path, nargs="+", required=True,
                    help="filled label CSVs, one per rater")
    ap.add_argument("--out", type=Path, required=True,
                    help="where to write the human-adjudication worklist CSV (split items only)")
    a = ap.parse_args()

    if any(_same_file(a.out, p) for p in a.labeled):
        print(f"[adjudicate] refusing to write the worklist over an input label file: {a.out}")
        return 2
    if a.out.exists() and _worklist_has_adjudications(a.out):
        print(f"[adjudicate] refusing to overwrite {a.out}: it already contains human "
              "adjudication decisions; move or rename it first.")
        return 2

    try:
        raters = load_raters(a.labeled)
    except ValueError as e:
        print(f"[adjudicate] {e}")
        return 2

    names = [n for n, _ in raters]
    ids = item_ids(raters)
    print(f"[adjudicate] {len(names)} raters {names} over {len(ids)} items\n")

    _print_summary(summarize(raters))

    header, rows = build_worklist(raters)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nwrote {len(rows)} split row(s) to {a.out} for HUMAN adjudication.")
    print("  the 'adjudicated' column is intentionally blank: this tool surfaces")
    print("  disagreements, it never resolves them, and the model never adjudicates.")

    _print_fleiss(raters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
