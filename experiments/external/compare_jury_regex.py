"""Q1 specificity, jury versus frozen regex, on the exact same FaithCoT items.

Section 3 reports the frozen regex's specificity on the whole 1,364-item corpus
(1,327/1,364) and on the 1,304 annotated rows. Neither denominator matches the
1,256-item jury-eligible pool (the frozen prompt renderer cannot label an item with
more than 8 options, so 48 items are jury-ineligible and never voted on). Comparing
the jury's specificity against either of those regex numbers would be comparing two
instruments on two different corpora.

This script joins `votes.jsonl` (run 0, status ok, corpus faithcot) to
`regex_predictions.jsonl` on `item_id`, per Q1 file, and reports both instruments'
specificity on the identical item set each Q1 variant actually scored. Every jury
number is claim_status EXPLORATORY (experiments/jury/GATE-Q1-COMPARISON.md: every
judge x Q1 file row FAILs the frozen gate); the regex number is RAW, as in section 3.

FaithCoT-Bench is third-party data used under a written evaluation-only permission.
Cite arXiv:2510.04040.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
Q1_QUESTIONS = ("Q1a", "Q1b", "Q1c")
DEFAULT_VOTES_ROOT = ROOT / "experiments" / "results" / "w6-external" / "gemma-3-27b-it"
DEFAULT_REGEX_PREDICTIONS = ROOT / "experiments" / "external" / "regex_predictions.jsonl"
DEFAULT_OUT = ROOT / "experiments" / "external" / "jury_vs_regex_same_items.json"


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {"numerator": k, "denominator": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def load_regex_predictions(path: pathlib.Path) -> dict[str, bool]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["item_id"]] = bool(row["regex_acknowledges_hint"])
    return out


def load_faithcot_votes(votes_root: pathlib.Path) -> dict[str, dict[str, str]]:
    """{question: {item_id: vote}}, run 0, status ok, corpus faithcot, vote in yes/no."""
    by_q: dict[str, dict[str, str]] = {q: {} for q in Q1_QUESTIONS}
    for path in sorted(votes_root.rglob("votes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            if v["run"] != 0 or v["status"] != "ok":
                continue
            if (v.get("meta") or {}).get("corpus") != "faithcot":
                continue
            q = v["question"]
            if q in by_q and v["vote"] in ("yes", "no"):
                by_q[q][v["item_id"]] = v["vote"]
    return by_q


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--votes-root", default=str(DEFAULT_VOTES_ROOT))
    ap.add_argument("--regex-predictions", default=str(DEFAULT_REGEX_PREDICTIONS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    regex = load_regex_predictions(pathlib.Path(args.regex_predictions))
    votes_by_q = load_faithcot_votes(pathlib.Path(args.votes_root))

    report: dict = {
        "claim_status": "jury: EXPLORATORY (no Q1 configuration has passed the frozen gate); regex: RAW",
        "cite": "arXiv:2510.04040",
        "dataset_revision": "6e3c004cbbde5bf47352df91e3ac399d2fb4593e",
        "per_question": {},
    }
    for question, jury_votes in votes_by_q.items():
        ids = sorted(jury_votes)
        missing = [i for i in ids if i not in regex]
        joined = [i for i in ids if i in regex]
        jury_no = sum(1 for i in joined if jury_votes[i] == "no")
        regex_no = sum(1 for i in joined if regex[i] is False)
        report["per_question"][question] = {
            "n_jury_scored": len(ids),
            "n_joined_to_regex": len(joined),
            "n_missing_from_regex_predictions": len(missing),
            "jury_specificity": rate(jury_no, len(joined)),
            "regex_specificity_same_items": rate(regex_no, len(joined)),
        }

    pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
