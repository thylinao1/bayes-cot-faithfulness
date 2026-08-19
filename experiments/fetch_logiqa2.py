"""Fetch LogiQA 2.0 (English MRC subset) into the experiments dataset format (free, no auth).

Downloads a split's JSONL file from the official csitfun/LogiQA2.0 GitHub repository
(the authoritative source and license home: CC BY-NC-SA 4.0 per its README; the HF
mirror's differing license tag is noted in docs/p3_substrate_scouting.md) and writes
the same JSON list of ``{question, choices, answer_index}`` that ``fetch_arc.py``
writes and the runners read via ``--data``. Items keep the shipped file's line order,
so "the first N items in fetch order" is deterministic and reproducible. This is a
small text download, not a model call: no GPU, no API key, no cost.

Why LogiQA 2.0: primary candidate substrate for a powered P3 (see
docs/p3_substrate_scouting.md). Logical-reasoning items are hard to answer without
reasoning, which is what populates the "moved" stratum P3 needs. Each item's context
passage is prepended to its question so the runner's ``Question:`` frame carries the
full problem; the four options are bare texts and the shipped 0-indexed integer
answer maps to ``answer_index`` unchanged.

Run (from the repo root):

    python experiments/fetch_logiqa2.py --split test --n 120

then point a runner's --data at the written file (pilot protocol:
docs/p3_substrate_scouting.md section 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

# Pinned to a COMMIT (main's HEAD on 2026-07-19, verified byte-identical to the branch
# head that day), not to the branch: "the first N items in fetch order" must mean the
# same bytes machine-to-machine and month-to-month, and a later upstream push must not
# silently change the pilot's item set. Update the pin deliberately, never implicitly.
PINNED_COMMIT = "955e1d3df6c59d9bfb44d9913da1e1a27ec14e18"
RAW_URL = (
    "https://raw.githubusercontent.com/csitfun/LogiQA2.0/"
    f"{PINNED_COMMIT}/logiqa/DATA/LOGIQA/{{split}}.txt"
)
N_OPTIONS = 4  # every LogiQA 2.0 MRC item ships exactly four options (verified 2026-07-19)
HERE = Path(__file__).resolve().parent


def _download(split: str) -> bytes:
    url = RAW_URL.format(split=split)
    req = urllib.request.Request(url, headers={"User-Agent": "bayes-cot-faithfulness/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """json.loads hook: a row with a duplicated key raises instead of last-wins.

    Plain json.loads silently keeps the LAST occurrence of a duplicated key, so a row
    carrying two "answer" values would validate cleanly with whichever the file listed
    last, exactly the untrustworthy-answer-key case the drop-never-repair rule (memo
    C3) exists for. Raising here makes parse_jsonl drop the whole line instead.
    """
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys in row: {sorted(set(k for k in keys if keys.count(k) > 1))}")
    return dict(pairs)


def to_item(row: dict) -> dict | None:
    """Convert one LogiQA 2.0 row to our format, or None if it is malformed.

    Keeps rows with a non-empty passage and question, exactly four non-empty string
    options, and an in-range integer answer. ``bool`` is rejected explicitly (it is an
    ``int`` subclass, and a ``true`` in a malformed row must not become answer_index 1).
    """
    text = row.get("text")
    question = row.get("question")
    options = row.get("options")
    answer = row.get("answer")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(options, list) or len(options) != N_OPTIONS:
        return None
    if not all(isinstance(o, str) and o.strip() for o in options):
        return None
    if isinstance(answer, bool) or not isinstance(answer, int):
        return None
    if not 0 <= answer < N_OPTIONS:
        return None
    return {
        "question": f"{text.strip()}\n\n{question.strip()}",
        "choices": [o.strip() for o in options],
        "answer_index": answer,
    }


def parse_jsonl(blob: str) -> list[dict]:
    """Parse the split file's JSON-lines blob into items, preserving line order.

    Split on ``\\n`` ONLY (``line.strip()`` absorbs the ``\\r`` of CRLF): Python's
    ``splitlines()`` also breaks on U+2028/U+2029/U+0085, which are LEGAL unescaped
    inside a JSON string, so it would shatter a valid row that merely contains one.
    Blank lines, undecodable lines, rows with duplicated keys, and rows that are not
    a JSON object are skipped: drop, never repair; the survivors keep the shipped
    order, which is what makes "the first N items in fetch order" reproducible.
    """
    items: list[dict] = []
    for line in blob.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        item = to_item(row)
        if item:
            items.append(item)
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="number of items to keep (>= 1)")
    ap.add_argument("--split", default="test", choices=["test", "train", "dev"])
    ap.add_argument("--out", type=Path, default=HERE / "data" / "logiqa2.json")
    a = ap.parse_args(argv)
    if a.n < 1:
        # A negative --n would slice [: -n] and silently keep all-but-the-last items.
        ap.error(f"--n must be >= 1, got {a.n}")

    try:
        raw = _download(a.split)
    except urllib.error.URLError as exc:
        print(f"[fetch] could not reach the LogiQA 2.0 raw file (no cost incurred): {exc}")
        return 1

    # utf-8-sig: identical to utf-8 on the (BOM-less) pinned files, and a future
    # pin update to a BOM-carrying file cannot silently drop row 1 (the BOM would
    # otherwise survive strip() and break json.loads on the first line).
    items = parse_jsonl(raw.decode("utf-8-sig"))[: a.n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} LogiQA 2.0 items ({a.split} split, fetch order) -> {a.out}")
    print(f"  source: commit {PINNED_COMMIT}, sha256 {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
