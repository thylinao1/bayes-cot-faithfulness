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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="number of items to keep")
    ap.add_argument("--split", default="test", choices=["test", "train", "validation"])
    ap.add_argument("--out", type=Path, default=HERE / "data" / "arc_challenge.json")
    a = ap.parse_args()

    items: list[dict] = []
    offset = 0
    page = 100
    while len(items) < a.n:
        try:
            body = _fetch(offset, page, a.split)
        except urllib.error.URLError as exc:
            print(f"[fetch] could not reach the dataset API (no cost incurred): {exc}")
            return 1
        rows = body.get("rows", [])
        if not rows:
            break
        for r in rows:
            item = to_item(r.get("row", {}))
            if item:
                items.append(item)
            if len(items) >= a.n:
                break
        offset += len(rows)
        time.sleep(0.2)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} ARC-Challenge items -> {a.out}")
    print(
        "next: PYTHONPATH=src python experiments/05_realmodel_control.py "
        f"--model llama3.1:8b --data {a.out} --hint-strength strong"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
