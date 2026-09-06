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


def _fetch(offset: int, length: int, split: str) -> dict:
    url = f"{API}?dataset={DATASET}&config={CONFIG}&split={split}&offset={offset}&length={length}"
    req = urllib.request.Request(url, headers={"User-Agent": "bayes-cot-faithfulness/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def concat_splits(per_split: list[tuple[str, list[dict]]], n: int) -> tuple[list[dict], int]:
    """Items from several splits in the order given, deduplicated on the resume key.

    The key is (question, choices), which is the key ``arms_resume`` merges banked
    records by; a pool carrying a repeated key is refused outright by
    ``arms_resume.duplicate_item_keys``, so it could never be resumed. FIRST occurrence
    wins, so naming ``test`` first leaves every item of the earlier 200-item and
    700-item pools at the same index and the enlargement is a pure append.
    """
    items: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    n_dupes = 0
    for _split, rows in per_split:
        for item in rows:
            key = (item["question"], tuple(item["choices"]))
            if key in seen:
                n_dupes += 1
                continue
            seen.add(key)
            items.append(item)
            if len(items) >= n:
                return items, n_dupes
    return items, n_dupes


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
    ap.add_argument("--out", type=Path, default=HERE / "data" / "arc_challenge.json")
    a = ap.parse_args(argv)
    if a.n < 1:
        ap.error(f"--n must be >= 1, got {a.n}")

    splits = [a.split] + [s for s in (a.then_split or []) if s != a.split]
    per_split: list[tuple[str, list[dict]]] = []
    for split in splits:
        rows = _split_items(split, a.n)
        if rows is None:
            return 1
        per_split.append((split, rows))
        print(f"  {split}: {len(rows)} usable rows")

    items, n_dupes = concat_splits(per_split, a.n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} ARC-Challenge items ({'+'.join(splits)} splits, fetch "
          f"order) -> {a.out}")
    print(f"  dropped {n_dupes} duplicate item keys (question plus choices)")
    print(
        "next: PYTHONPATH=src python experiments/05_realmodel_control.py "
        f"--model llama3.1:8b --data {a.out} --hint-strength strong"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
