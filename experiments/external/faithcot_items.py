"""Turn the mapped FINE-CoT corpus into the jury runner's JuryItem JSONL.

The output carries third-party reasoning text, so it is written under the gitignored
experiments/data/external/ path and is never committed. Only per-item predictions and
aggregates leave this lane.

The ``meta`` block carries the human labels. The runner never shows ``meta`` to a
judge, which is what keeps the judge blind to the answer it is being scored against.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from experiments.external.faithcot_mapping import load_corpus

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "experiments" / "data" / "external" / "faithcot" / "faithcot"
DEFAULT_OUT = ROOT / "experiments" / "data" / "external" / "faithcot_items.jsonl"

# The subject models are FaithCoT's four generators, none of which is on this project's
# 18-model roster, so the panel routing rule is not defined for them. That is correct:
# the panel rule is a statement about our sweep, not about a third-party corpus. Every
# item is scored by whichever judge is served, and the stratum is set explicitly so the
# runner never calls family_of() on a non-roster name.
STRATUM = "external-faithcot"

# experiments/jury/prompt_files.py labels options from LETTERS = "ABCDEFGH", so an item
# with more than eight options cannot be rendered by the frozen renderer at all: it raises
# IndexError. Rather than edit another lane's renderer, those items are excluded from the
# JURY instrument with their count stated, and they stay in the REGEX instrument, which
# needs no option letters. The two instruments therefore have different denominators and
# both are printed everywhere they appear.
MAX_CHOICES = 8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--annotated-only",
        action="store_true",
        help="keep only rows carrying a human unfaithfulness label",
    )
    ap.add_argument(
        "--jury-eligible-only",
        action="store_true",
        help=f"drop items with more than {MAX_CHOICES} options, which the frozen renderer cannot label",
    )
    args = ap.parse_args(argv)

    items = list(load_corpus(pathlib.Path(args.corpus)))
    kept = [i for i in items if i.annotated] if args.annotated_only else list(items)
    excluded_over_choices = []
    if args.jury_eligible_only:
        excluded_over_choices = [i.item_id for i in kept if len(i.options) > MAX_CHOICES]
        kept = [i for i in kept if len(i.options) <= MAX_CHOICES]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    longest = 0
    with out.open("w", encoding="utf-8") as fh:
        for item in kept:
            row = {
                "item_id": item.item_id,
                "subject_model": item.generator_model,
                "question": item.question,
                "choices": list(item.options),
                "reasoning": item.cot_text,
                "final_answer": item.model_answer,
                "stratum": STRATUM,
                "all_judge_row": True,
                "meta": {
                    "corpus": "faithcot",
                    "dataset": item.dataset,
                    "generator_model": item.generator_model,
                    "gold_answer": item.gold_answer,
                    "answer_correct": item.answer_correct,
                    "faithful_type": item.faithful_type,
                    "label_unfaithful": item.label_unfaithful,
                    "q2_supports": item.q2_supports,
                    "label_consistent": item.label_consistent,
                    "cue_present": item.cue_present,
                    "n_steps": item.n_steps,
                    "soft_faithfulness": item.soft_faithfulness,
                },
            }
            longest = max(longest, len(item.cot_text))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "written": len(kept),
                "of": len(items),
                "annotated_only": args.annotated_only,
                "jury_eligible_only": args.jury_eligible_only,
                "n_excluded_over_max_choices": len(excluded_over_choices),
                "max_choices": MAX_CHOICES,
                "path": str(out),
                "longest_reasoning_chars": longest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
