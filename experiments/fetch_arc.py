"""Fetch ARC-Challenge into the experiments dataset format (free, no auth, no cost).

Uses the Hugging Face datasets-server rows API for ``allenai/ai2_arc``
(ARC-Challenge). Writes a JSON list of ``{question, choices, answer_index}`` that
the control runner reads via ``--data``. This is a small JSON download, not a
model call: no GPU, no API key, no cost.

Why ARC-Challenge: the toy arithmetic set was too easy, so the model ignored the
planted hint (the REVIEW outcome). ARC-Challenge items are hard enough that the
model is genuinely uncertain, which is what lets a hint sway the answer.

Run (from the repo root):

    python experiments/fetch_arc.py --n 200
    PYTHONPATH=src python experiments/05_realmodel_control.py \\
        --model llama3.1:8b --data experiments/data/arc_challenge.json --hint-strength strong
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
DATASET = "allenai/ai2_arc"
CONFIG = "ARC-Challenge"
HERE = Path(__file__).resolve().parent


# The datasets-server throttles and occasionally answers 502 mid-paging. A pool fetch
# walks ~30 pages, so one transient answer used to lose the whole pool and, worse, would
# have lost it AFTER two splits had been read, making a partial retry look like a
# different pool. Retrying the same page returns the same rows, so this changes the
# reliability of the fetch and never its output.
_FETCH_ATTEMPTS = 6
_FETCH_WAIT = 5.0


def _fetch(offset: int, length: int, split: str) -> dict:
    url = f"{API}?dataset={DATASET}&config={CONFIG}&split={split}&offset={offset}&length={length}"
    last: Exception | None = None
    for attempt in range(_FETCH_ATTEMPTS):
        req = urllib.request.Request(
            url, headers={"User-Agent": "bayes-cot-faithfulness/0.1"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 4xx other than 429 is a bad request: the same URL fails the same way.
            if exc.code != 429 and exc.code < 500:
                raise
            last = exc
        except urllib.error.URLError as exc:
            last = exc
        if attempt < _FETCH_ATTEMPTS - 1:
            wait = _FETCH_WAIT * (2 ** attempt)
            print(f"[fetch] {type(last).__name__}: {last}; retrying page "
                  f"{split}@{offset} in {wait:.0f}s")
            time.sleep(wait)
    raise last if last is not None else RuntimeError("fetch failed with no exception")


def to_item(row: dict) -> dict | None:
    """Convert one ARC row to our format, or None if it is malformed.

    Keeps clean 3-to-5-option items whose answer key resolves to an option label.
    """
    question = row.get("question", "").strip()
    choices = row.get("choices", {})
    text = choices.get("text", [])
    labels = choices.get("label", [])
    key = row.get("answerKey")
    if not question or len(text) != len(labels) or not (3 <= len(text) <= 5):
        return None
    if key not in labels:
        return None
    return {"question": question, "choices": list(text), "answer_index": labels.index(key)}


def _split_items(split: str, want: int) -> list[dict] | None:
    """Every usable item of one split, in datasets-server row order (None on a fetch error).

    ``want`` only bounds the paging: the caller deduplicates across splits, so a split
    is read until it runs out or until it alone has supplied ``want`` items.
    """
    items: list[dict] = []
    offset = 0
    page = 100
    while len(items) < want:
        try:
            body = _fetch(offset, page, split)
        except urllib.error.URLError as exc:
            print(f"[fetch] could not reach the dataset API (no cost incurred): {exc}")
            return None
        rows = body.get("rows", [])
        if not rows:
            break
        for r in rows:
            item = to_item(r.get("row", {}))
            if item:
                items.append(item)
        offset += len(rows)
        time.sleep(0.2)
    return items


def concat_splits(per_split: list[tuple[str, list[dict]]], n: int,
                  max_choices: int | None = None,
                  max_choices_from: int = 0) -> tuple[list[dict], int, int]:
    """Items from several splits in the order given, deduplicated on the resume key.

    The key is (question, choices), which is the key ``arms_resume`` merges banked
    records by; a pool carrying a repeated key is refused outright by
    ``arms_resume.duplicate_item_keys``, so it could never be resumed. FIRST occurrence
    wins, so naming ``test`` first leaves every item of the earlier 200-item and
    700-item pools at the same index and the enlargement is a pure append.

    ``max_choices`` caps the option count of items accepted at index
    ``max_choices_from`` and beyond; items before that index are taken exactly as the
    stream delivers them. The asymmetry is the point: the first 700 items are the frozen
    prefix every Phase-1 artifact refers to by index and must not move, while the
    enlargement past it must not widen the answer-label set the frozen prompt surface
    parses against (DECISION-LOG 2026-09-07 ruling (a): the 1,500-item ARC pool reached
    5-choice items past the prefix, where the first 700 have at most 4). Returns the
    items, the duplicate-key count, and the count dropped by the cap.
    """
    items: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    n_dupes = 0
    n_over_cap = 0
    for _split, rows in per_split:
        for item in rows:
            key = (item["question"], tuple(item["choices"]))
            if key in seen:
                n_dupes += 1
                continue
            # The de-duplication key is recorded BEFORE the cap so a capped item cannot
            # come back through a later split under the same key.
            seen.add(key)
            if (max_choices is not None and len(items) >= max_choices_from
                    and len(item["choices"]) > max_choices):
                n_over_cap += 1
                continue
            items.append(item)
            if len(items) >= n:
                return items, n_dupes, n_over_cap
    return items, n_dupes, n_over_cap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="number of items to keep (>= 1)")
    ap.add_argument("--split", default="test", choices=["test", "train", "validation"],
                    help="first split to draw from; see --then-split")
    ap.add_argument("--then-split", action="append", default=None,
                    choices=["test", "train", "validation"],
                    help="additional split(s), appended in the order given once the "
                         "previous split is exhausted, deduplicated on the resume key. "
                         "Repeatable. Use it when one split cannot supply --n items: "
                         "the ARC-Challenge test split holds 1,172 rows and the A3 pool "
                         "is at least 1,500.")
    ap.add_argument("--max-choices", type=int, default=None,
                    help="drop items with more than this many options, from index "
                         "--max-choices-from onward. ARC-Challenge is mostly 4-option "
                         "but the test split carries a handful of 5-option items, and "
                         "an ARC cell parsing against 5 labels would widen the frozen "
                         "prompt surface's answer-label set. Off by default.")
    ap.add_argument("--max-choices-from", type=int, default=0,
                    help="index at which --max-choices starts applying (default 0, "
                         "meaning the whole pool). Set it to the frozen prefix length "
                         "to leave items already measured against exactly where they "
                         "are and cap only the enlargement.")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "arc_challenge.json")
    a = ap.parse_args(argv)
    if a.n < 1:
        ap.error(f"--n must be >= 1, got {a.n}")
    if a.max_choices is not None and a.max_choices < 1:
        ap.error(f"--max-choices must be >= 1, got {a.max_choices}")
    if a.max_choices_from < 0:
        ap.error(f"--max-choices-from must be >= 0, got {a.max_choices_from}")

    splits = [a.split] + [s for s in (a.then_split or []) if s != a.split]
    per_split: list[tuple[str, list[dict]]] = []
    for split in splits:
        rows = _split_items(split, a.n)
        if rows is None:
            return 1
        per_split.append((split, rows))
        print(f"  {split}: {len(rows)} usable rows")

    items, n_dupes, n_over_cap = concat_splits(
        per_split, a.n, max_choices=a.max_choices, max_choices_from=a.max_choices_from
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} ARC-Challenge items ({'+'.join(splits)} splits, fetch "
          f"order) -> {a.out}")
    print(f"  dropped {n_dupes} duplicate item keys (question plus choices)")
    if a.max_choices is not None:
        print(f"  dropped {n_over_cap} items with more than {a.max_choices} options "
              f"from index {a.max_choices_from} onward")
    print(
        "next: PYTHONPATH=src python experiments/05_realmodel_control.py "
        f"--model llama3.1:8b --data {a.out} --hint-strength strong"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
