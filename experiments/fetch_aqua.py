"""Fetch AQuA-RAT into the experiments dataset format (free, no auth, no cost).

Downloads a split's JSONL file from the official google-deepmind/AQuA GitHub
repository (Apache-2.0, LICENSE confirmed in docs/p3_substrate_scouting.md) and
writes the same JSON list of ``{question, choices, answer_index}`` that
``fetch_arc.py`` writes and the runners read via ``--data``. Items keep the shipped
file's line order, so "the first N items in fetch order" is deterministic and
reproducible. This is a small text download, not a model call: no GPU, no API key,
no cost.

Why AQuA-RAT: secondary candidate substrate for a powered P3 (see
docs/p3_substrate_scouting.md). Multi-step algebraic word problems are where a
direct no-CoT answer is near-guessing while reasoning can still land the correct
answer, the disagreement pattern that populates the "moved" stratum. Only the
question, the options, and the answer key are used; the shipped rationales (the
noisy part of this dataset) are never read.

Every shipped option carries a positional letter prefix (``A)24``, ``B)25``, ...).
The prefix is stripped -- the runner's prompt frame adds its own ``(A)`` labels, and
a double label would change the frozen prompt surface -- and the prefix letter must
match the option's position, so a shuffled or malformed row can never silently remap
the answer key.

Run (from the repo root):

    python experiments/fetch_aqua.py --split test --n 120

then point a runner's --data at the written file (pilot protocol:
docs/p3_substrate_scouting.md section 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

# Pinned to a COMMIT (master's HEAD on 2026-07-19, verified byte-identical to the
# branch head that day), not to the branch: "the first N items in fetch order" must
# mean the same bytes machine-to-machine and month-to-month, and a later upstream
# push must not silently change the pilot's item set. Update the pin deliberately.
PINNED_COMMIT = "26b64ed1f22208c6742f47df547dfd65088ec30d"
RAW_URL = (
    f"https://raw.githubusercontent.com/google-deepmind/AQuA/{PINNED_COMMIT}/{{split}}.json"
)
N_OPTIONS = 5  # every AQuA-RAT item ships exactly five options A-E (verified 2026-07-19)
_LETTERS = "ABCDE"
_OPTION_PREFIX_RE = re.compile(r"^([A-E])\s*\)\s*")
HERE = Path(__file__).resolve().parent


def _download(split: str) -> bytes:
    url = RAW_URL.format(split=split)
    req = urllib.request.Request(url, headers={"User-Agent": "bayes-cot-faithfulness/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """json.loads hook: a row with a duplicated key raises instead of last-wins.

    Plain json.loads silently keeps the LAST occurrence of a duplicated key, so a row
    carrying two "options" arrays (each with valid positional prefixes) or two
    "correct" letters would validate cleanly with whichever the file listed last --
    the silent answer-key remap the module docstring promises can never happen.
    Raising here makes parse_jsonl drop the whole line instead (memo C3).
    """
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys in row: {sorted(set(k for k in keys if keys.count(k) > 1))}")
    return dict(pairs)


def strip_option(option: str, position: int) -> str | None:
    """The option's bare text, or None unless its prefix letter matches ``position``.

    A prefix at the wrong position (``B)`` first) means the row's option order and
    its ``correct`` letter can no longer be trusted to agree, so the caller drops
    the whole row rather than guess at the mapping.
    """
    if not isinstance(option, str):
        return None
    m = _OPTION_PREFIX_RE.match(option)
    if not m or m.group(1) != _LETTERS[position]:
        return None
    text = option[m.end():].strip()
    return text or None


def to_item(row: dict) -> dict | None:
    """Convert one AQuA-RAT row to our format, or None if it is malformed.

    Keeps rows with a non-empty question, exactly five options whose positional
    letter prefixes all verify, and a ``correct`` letter within A-E. The rationale
    field is deliberately ignored.
    """
    question = row.get("question")
    options = row.get("options")
    correct = row.get("correct")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(options, list) or len(options) != N_OPTIONS:
        return None
    choices: list[str] = []
    for i, option in enumerate(options):
        text = strip_option(option, i)
        if text is None:
            return None
        choices.append(text)
    if not isinstance(correct, str):
        return None
    # A plain ``in _LETTERS`` would be a substring test ("" and "AB" both pass and
    # "AB" silently maps to A), so the letter must be exactly one character.
    letter = correct.strip()
    if len(letter) != 1 or letter not in _LETTERS:
        return None
    return {
        "question": question.strip(),
        "choices": choices,
        "answer_index": _LETTERS.index(letter),
    }


def parse_jsonl(blob: str) -> list[dict]:
    """Parse the split file's JSON-lines blob into items, preserving line order.

    Split on ``\\n`` ONLY (``line.strip()`` absorbs the ``\\r`` of CRLF): Python's
    ``splitlines()`` also breaks on U+2028/U+2029/U+0085, which are LEGAL unescaped
    inside a JSON string, so it would shatter a valid row that merely contains one.
    Blank lines, undecodable lines, rows with duplicated keys, and rows that are not
    a JSON object are skipped -- drop, never repair; the survivors keep the shipped
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
    ap.add_argument("--out", type=Path, default=HERE / "data" / "aqua_rat.json")
    a = ap.parse_args(argv)
    if a.n < 1:
        # A negative --n would slice [: -n] and silently keep all-but-the-last items.
        ap.error(f"--n must be >= 1, got {a.n}")

    try:
        raw = _download(a.split)
    except urllib.error.URLError as exc:
        print(f"[fetch] could not reach the AQuA-RAT raw file (no cost incurred): {exc}")
        return 1

    # utf-8-sig: identical to utf-8 on the (BOM-less) pinned files, and a future
    # pin update to a BOM-carrying file cannot silently drop row 1 (the BOM would
    # otherwise survive strip() and break json.loads on the first line).
    items = parse_jsonl(raw.decode("utf-8-sig"))[: a.n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} AQuA-RAT items ({a.split} split, fetch order) -> {a.out}")
    print(f"  source: commit {PINNED_COMMIT}, sha256 {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
