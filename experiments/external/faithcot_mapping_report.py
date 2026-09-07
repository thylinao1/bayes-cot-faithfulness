"""Counts for the FaithCoT field map: cue presence, taxonomy, correctness, disagreement.

Writes experiments/external/faithcot_mapping_report.json. Derived aggregates only;
no third-party content is emitted.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

from experiments.external.faithcot_mapping import TYPE_NAMES, TYPE_TABLE, load_corpus

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "experiments" / "data" / "external" / "faithcot" / "faithcot"
OUT = ROOT / "experiments" / "external" / "faithcot_mapping_report.json"


def main() -> int:
    items = list(load_corpus(CORPUS))
    manifest = json.loads((ROOT / "experiments" / "external" / "faithcot_manifest.json").read_text())

    annotated = [i for i in items if i.annotated]
    typed = [i for i in annotated if i.faithful_type is not None]

    by_type = collections.Counter(i.faithful_type for i in typed)
    by_type_correct = collections.Counter((i.faithful_type, i.answer_correct) for i in typed)

    # Where the two label fields disagree, split by which way.
    type_vs_binary = [
        i for i in typed if TYPE_TABLE[i.faithful_type][1] != i.label_unfaithful
    ]
    type_vs_correct = [
        i for i in typed if TYPE_TABLE[i.faithful_type][0] != i.answer_correct
    ]

    report = {
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "corpus_sha256": manifest["corpus_sha256"],
        "paper": manifest["paper"],
        "n_items": len(items),
        "n_annotated": len(annotated),
        "n_unannotated": len(items) - len(annotated),
        "cue": {
            "n_items_with_a_cue": sum(1 for i in items if i.cue_present),
            "denominator": len(items),
            "consequence": (
                "The frozen acknowledgment regex is defined only where a cue exists, so "
                "it has no Q1 ground truth on this corpus. What the corpus measures is "
                "specificity out of distribution: every positive from a cue-disclosure "
                "instrument here is a false positive by construction."
            ),
        },
        "q2": {
            "field": "unfaithfulness",
            "positive_class": "unfaithfulness == 1 (the CoT does not reflect the reasoning)",
            "n_positive": sum(1 for i in annotated if i.label_unfaithful),
            "n_negative": sum(1 for i in annotated if not i.label_unfaithful),
            "denominator": len(annotated),
        },
        "taxonomy": {
            str(t): {
                "name": TYPE_NAMES[t],
                "n": by_type[t],
                "n_answer_correct": by_type_correct[(t, True)],
                "n_answer_incorrect": by_type_correct[(t, False)],
                "implies_correct": TYPE_TABLE[t][0],
                "implies_unfaithful": TYPE_TABLE[t][1],
            }
            for t in sorted(by_type)
        },
        "label_disagreement": {
            "n_type_contradicts_binary_label": len(type_vs_binary),
            "n_type_contradicts_derived_correctness": len(type_vs_correct),
            "n_inconsistent_rows": sum(1 for i in typed if not i.label_consistent),
            "n_consistent_rows": sum(1 for i in typed if i.label_consistent),
            "denominator": len(typed),
            "rule": (
                "Primary analysis uses the unfaithfulness field on all annotated rows; a "
                "sensitivity analysis restricted to label_consistent rows is reported "
                "beside it. No row is repaired."
            ),
        },
        "correctness_strata": {
            "correct": sum(1 for i in annotated if i.answer_correct),
            "incorrect": sum(1 for i in annotated if not i.answer_correct),
            "denominator": len(annotated),
        },
        "per_split": {
            f"{ds}/{mdl}": {
                "n": n,
                "n_annotated": sum(
                    1 for i in annotated if i.dataset == ds and i.generator_model == mdl
                ),
                "n_unfaithful": sum(
                    1
                    for i in annotated
                    if i.dataset == ds and i.generator_model == mdl and i.label_unfaithful
                ),
            }
            for (ds, mdl), n in sorted(
                collections.Counter((i.dataset, i.generator_model) for i in items).items()
            )
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_split"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
