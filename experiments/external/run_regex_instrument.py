"""Score the frozen acknowledgment regex against FINE-CoT's human annotations.

WHAT THE INSTRUMENT IS. ``acknowledges_hint`` in src/bayes_cot_faithfulness/interventions.py
is frozen and applied here byte-identical to main. It yields a LABEL, not a score, so
AUROC is not defined for it; recall, specificity and F1 are, and those are what is
reported. Nothing is thresholded and nothing is tuned.

WHAT CAN AND CANNOT BE MEASURED HERE. The regex answers Q1: did the CoT disclose that a
planted cue moved the answer. Its precondition is that a cue exists. No FINE-CoT item
carries a cue (0 of 1,364, with the detector's falsification pinned in
tests/test_faithcot_mapping.py), so:

  * Q1 RECALL is UNDEFINED on this corpus. There are no true positives to recall. The
    eligible denominator is 0 and no recall number is printed for it.
  * Q1 SPECIFICITY IS defined and is the measurement of record here. On a corpus with no
    cue, every fire is a false positive by construction, so the fire rate is exactly
    1 minus specificity, with the whole corpus as the denominator.

An OUT-OF-CONSTRUCT PROBE is reported separately and clearly marked: the same labels
scored against Q2's human unfaithfulness label. The regex was never built to answer Q2
and no claim rests on it; the probe is there to say, with denominators, how much of the
Q2 construct a Q1 instrument picks up out of distribution. Reading it as a Q2 result
would be a category error.

Third-party data under a written evaluation-only permission (PERMISSIONS.md 2026-09-07).
Cite arXiv:2510.04040.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

from bayes_cot_faithfulness.interventions import _HINT_ACK_RE, acknowledges_hint
from experiments.external.faithcot_mapping import TYPE_NAMES, load_corpus

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "experiments" / "data" / "external" / "faithcot" / "faithcot"
OUT_DIR = ROOT / "experiments" / "external"


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson interval, so a rate near 0 does not get a zero-width interval."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k: int, n: int) -> dict:
    return {
        "numerator": k,
        "denominator": n,
        "rate": round(k / n, 4) if n else None,
        "wilson95": wilson(k, n),
    }


def binary_scores(preds: list[bool], truth: list[bool]) -> dict:
    """Recall, specificity, precision and F1 with every denominator printed."""
    tp = sum(1 for p, t in zip(preds, truth) if p and t)
    fp = sum(1 for p, t in zip(preds, truth) if p and not t)
    fn = sum(1 for p, t in zip(preds, truth) if not p and t)
    tn = sum(1 for p, t in zip(preds, truth) if not p and not t)
    pos, neg = tp + fn, tn + fp
    recall = tp / pos if pos else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": len(preds),
        "recall": rate(tp, pos),
        "specificity": rate(tn, neg),
        "precision": rate(tp, tp + fp) if (tp + fp) else {"numerator": 0, "denominator": 0, "rate": None, "wilson95": None},
        "f1": round(f1, 4) if f1 is not None else None,
        "auroc": None,
        "auroc_note": "the instrument yields a label, not a score, so AUROC is not defined for it",
    }


def main() -> int:
    manifest = json.loads((OUT_DIR / "faithcot_manifest.json").read_text())
    items = list(load_corpus(CORPUS))
    preds = {i.item_id: acknowledges_hint(i.cot_text) for i in items}

    annotated = [i for i in items if i.annotated]
    fires = [i for i in items if preds[i.item_id]]

    # --- the measurement of record: Q1 specificity on a corpus with no cue -------
    q1 = {
        "construct": "Q1, cue disclosure",
        "eligible_items_a_cue_exists": sum(1 for i in items if i.cue_present),
        "recall": None,
        "recall_note": (
            "UNDEFINED. Q1 recall needs items where a cue exists and 0 of 1,364 items "
            "carry one, so there is no true positive to recall."
        ),
        "specificity_whole_corpus": rate(len(items) - len(fires), len(items)),
        "false_positive_rate_whole_corpus": rate(len(fires), len(items)),
        "specificity_annotated_only": rate(
            len(annotated) - sum(1 for i in annotated if preds[i.item_id]), len(annotated)
        ),
        "by_correctness": {
            name: rate(
                sum(1 for i in items if i.answer_correct == correct and not preds[i.item_id]),
                sum(1 for i in items if i.answer_correct == correct),
            )
            for name, correct in (("answer_correct", True), ("answer_incorrect", False))
        },
        "by_faithful_type": {
            str(t): {
                "name": TYPE_NAMES[t],
                "specificity": rate(
                    sum(1 for i in items if i.faithful_type == t and not preds[i.item_id]),
                    sum(1 for i in items if i.faithful_type == t),
                ),
            }
            for t in (1, 2, 3, 4)
        },
        "by_dataset": {
            ds: rate(
                sum(1 for i in items if i.dataset == ds and not preds[i.item_id]),
                sum(1 for i in items if i.dataset == ds),
            )
            for ds in sorted({i.dataset for i in items})
        },
        "by_generator_model": {
            m: rate(
                sum(1 for i in items if i.generator_model == m and not preds[i.item_id]),
                sum(1 for i in items if i.generator_model == m),
            )
            for m in sorted({i.generator_model for i in items})
        },
        # What each fire actually matched, so a false positive can be read off rather
        # than guessed at. The frozen pattern is used as is; nothing is recompiled.
        "which_alternations_fired": dict(
            collections.Counter(
                _HINT_ACK_RE.search(i.cot_text).group(0).lower() for i in fires
            ).most_common()
        ),
    }

    # --- the out-of-construct probe, clearly marked ------------------------------
    ann_preds = [preds[i.item_id] for i in annotated]
    ann_truth = [bool(i.label_unfaithful) for i in annotated]
    consistent = [i for i in annotated if i.label_consistent]
    probe = {
        "WARNING": (
            "OUT-OF-CONSTRUCT. The regex answers Q1 (cue disclosure) and is scored here "
            "against Q2 (does the CoT support the answer). It was never built for this and "
            "no claim rests on it. It is reported to bound what a Q1 instrument transports."
        ),
        "positive_class": "unfaithfulness == 1",
        "all_annotated": binary_scores(ann_preds, ann_truth),
        "label_consistent_rows_only": binary_scores(
            [preds[i.item_id] for i in consistent], [bool(i.label_unfaithful) for i in consistent]
        ),
        "by_correctness": {
            name: binary_scores(
                [preds[i.item_id] for i in annotated if i.answer_correct == c],
                [bool(i.label_unfaithful) for i in annotated if i.answer_correct == c],
            )
            for name, c in (("answer_correct", True), ("answer_incorrect", False))
        },
        "per_type_fire_rate": {
            str(t): {
                "name": TYPE_NAMES[t],
                **rate(
                    sum(1 for i in annotated if i.faithful_type == t and preds[i.item_id]),
                    sum(1 for i in annotated if i.faithful_type == t),
                ),
            }
            for t in (1, 2, 3, 4)
        },
        "per_type_fire_rate_by_correctness": {
            f"type{t}_{name}": rate(
                sum(
                    1 for i in annotated
                    if i.faithful_type == t and i.answer_correct == c and preds[i.item_id]
                ),
                sum(1 for i in annotated if i.faithful_type == t and i.answer_correct == c),
            )
            for t in (1, 2, 3, 4)
            for name, c in (("answer_correct", True), ("answer_incorrect", False))
        },
    }

    # --- two exact tests, both pre-specified by what the tables show ------------
    from scipy.stats import fisher_exact

    def two_by_two(a: int, b: int, c: int, d: int) -> dict:
        odds, p = fisher_exact([[a, b], [c, d]])
        return {"table": [[a, b], [c, d]], "odds_ratio": round(float(odds), 4),
                "fisher_exact_two_sided_p": float(f"{p:.4g}")}

    t4_fire = sum(1 for i in annotated if i.faithful_type == 4 and preds[i.item_id])
    t4_n = sum(1 for i in annotated if i.faithful_type == 4)
    t3_fire = sum(1 for i in annotated if i.faithful_type == 3 and preds[i.item_id])
    t3_n = sum(1 for i in annotated if i.faithful_type == 3)
    tp_, fp_ = probe["all_annotated"]["tp"], probe["all_annotated"]["fp"]
    fn_, tn_ = probe["all_annotated"]["fn"], probe["all_annotated"]["tn"]
    tests = {
        "type4_vs_type3_fire_rate": {
            "question": (
                "Does the frozen regex fire more often on post-hoc rationalization "
                "(type 4) than on a faithful correct CoT (type 3)? Both strata are "
                "correct-answer strata, so correctness is held fixed."
            ),
            "type4": rate(t4_fire, t4_n),
            "type3": rate(t3_fire, t3_n),
            **two_by_two(t4_fire, t4_n - t4_fire, t3_fire, t3_n - t3_fire),
        },
        "fire_vs_unfaithful": {
            "question": (
                "Are the regex's fires enriched for human-labelled unfaithfulness "
                "relative to the corpus base rate?"
            ),
            "base_rate_unfaithful": rate(tp_ + fn_, tp_ + fn_ + fp_ + tn_),
            "precision_of_a_fire": rate(tp_, tp_ + fp_),
            **two_by_two(tp_, fp_, fn_, tn_),
        },
    }

    report = {
        "instrument": "frozen regex acknowledges_hint (src/bayes_cot_faithfulness/interventions.py)",
        "instrument_is_frozen": True,
        "claim_status": "VALIDATED-INSTRUMENT-ON-EXTERNAL-DATA",
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "corpus_sha256": manifest["corpus_sha256"],
        "cite": manifest["paper"],
        "permission": manifest["permission_quote"],
        "n_items": len(items),
        "n_annotated": len(annotated),
        "n_fires": len(fires),
        "q1_cue_disclosure": q1,
        "q2_out_of_construct_probe": probe,
        "exact_tests": tests,
    }
    (OUT_DIR / "regex_instrument_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Per-item predictions: derived, no third-party content, safe to commit.
    with (OUT_DIR / "regex_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps({
                "item_id": i.item_id, "dataset": i.dataset,
                "generator_model": i.generator_model,
                "cue_present": i.cue_present,
                "regex_acknowledges_hint": preds[i.item_id],
                "answer_correct": i.answer_correct,
                "faithful_type": i.faithful_type,
                "label_unfaithful": i.label_unfaithful,
                "label_consistent": i.label_consistent,
                "n_steps": i.n_steps,
            }) + "\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
