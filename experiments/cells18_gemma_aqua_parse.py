"""Classify the unparseable clean outputs on the two Gemma AQuA-RAT cells.

Read only. Nothing is fixed, no parser is touched and no rate anywhere else in this
lane changes because of what is written here. The task is to say WHAT those outputs
are, with a denominator, because 213 and 216 of 1,500 is two orders of magnitude
more attrition than the 0 to 9 every other cell of the 18 loses.

The classes are decided by the text itself and are mutually exclusive, tested in
this order:

``empty``
    the clean completion is empty or whitespace only.
``none_of_the_above``
    the completion contains the phrase "none of the above" in any case. The model
    finished its arithmetic, found a value that is not among the five lettered
    options and said so in the answer slot.
``numeric_not_a_letter``
    there is an answer marker and what follows it is a number rather than one of
    the item's answer letters, e.g. ``Answer: (6)`` where the options are 9, 7, 3,
    8 and 12. Split further by whether that number is one of the option VALUES.
``truncated_no_answer_line``
    non-empty, no "none of the above", and no answer marker at all: the completion
    stops mid-sentence inside the reasoning.
``other_nonletter_text``
    there is an answer marker and what follows is neither a number nor a letter.

A 30-item hand-read sample at a fixed seed is printed with the artifact so the rule
above can be checked against the raw text rather than trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

NONE_RE = re.compile(r"none of the above", re.IGNORECASE)
ANS_RE = re.compile(r"answer\s*:", re.IGNORECASE)
TAIL_RE = re.compile(r"answer\s*:\s*(.{0,40})", re.IGNORECASE | re.DOTALL)
NUM_RE = re.compile(r"-?[\d,\.]+")
SEED = 20260908
N_SAMPLE = 30


def classify(rec: dict) -> tuple[str, str]:
    cot = (rec.get("clean_cot") or "").strip()
    if not cot:
        return "empty", ""
    if NONE_RE.search(cot):
        return "none_of_the_above", ""
    if not ANS_RE.search(cot):
        return "truncated_no_answer_line", cot[-80:]
    tail = TAIL_RE.findall(cot)
    text = (tail[-1] if tail else "").strip().split("\n")[0].strip()
    value = text.strip("() .")
    if NUM_RE.fullmatch(value):
        choices = [str(x).replace(",", "") for x in (rec.get("choices") or [])]
        in_choices = value.replace(",", "") in choices
        return ("numeric_not_a_letter", f"{text} | in_choices={in_choices}")
    return "other_nonletter_text", text


def run_cell(path: Path) -> dict:
    payload = json.loads((path / "checkpoint.json").read_text())
    records = payload["records"]
    bad = [r for r in records if r.get("clean_answer") is None]
    counts: Counter = Counter()
    numeric_in_choices = 0
    for r in bad:
        kind, extra = classify(r)
        counts[kind] += 1
        if kind == "numeric_not_a_letter" and "in_choices=True" in extra:
            numeric_in_choices += 1
    rng = random.Random(SEED)
    sample = rng.sample(bad, min(N_SAMPLE, len(bad)))
    read = []
    for r in sample:
        kind, extra = classify(r)
        cot = r.get("clean_cot") or ""
        read.append(
            {
                "class": kind,
                "detail": extra,
                "n_chars": len(cot),
                "choices": r.get("choices"),
                "gold_letter": r.get("answer_label"),
                "tail_120_chars": cot[-120:],
            }
        )
    return {
        "n_records": len(records),
        "n_unparseable": len(bad),
        "classes": {
            "none_of_the_above": counts["none_of_the_above"],
            "numeric_not_a_letter": counts["numeric_not_a_letter"],
            "truncated_no_answer_line": counts["truncated_no_answer_line"],
            "other_nonletter_text": counts["other_nonletter_text"],
            "empty": counts["empty"],
        },
        "numeric_answers_whose_value_is_one_of_the_options": numeric_in_choices,
        "checkpoint_sha256": hashlib.sha256(
            (path / "checkpoint.json").read_bytes()
        ).hexdigest(),
        "sample_read": read,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cues",
        default="stated-hint,professor",
        help="comma-separated AQuA-RAT cue families to classify; the default is the two "
             "the 18-cell lane had, and the 24-cell lane passes all four",
    )
    args = ap.parse_args()
    cues = [c.strip() for c in args.cues.split(",") if c.strip()]
    root = Path(args.results_root) / "gemma-2-9b-it" / "aqua_rat"
    out = {
        "model": "google/gemma-2-9b-it",
        "substrate": "aqua_rat",
        "seed": SEED,
        "cues": cues,
        "n_sampled": N_SAMPLE,
        "rule": __doc__,
        "nothing_is_fixed": (
            "read only. The frozen answer regexes are untouched (they are fingerprinted "
            "in tests/test_frozen_guard.py) and no rate elsewhere in this lane changes."
        ),
    }
    for cue in cues:
        d = root / cue
        if not (d / "transcripts.jsonl").exists():
            print(f"NO RECORDS for gemma-2-9b-it/aqua_rat/{cue} at {d}, denominator 0, "
                  "not classified")
            continue
        out[cue] = run_cell(d)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    for cue in cues:
        if cue in out:
            print(cue, out[cue]["n_unparseable"], out[cue]["classes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
