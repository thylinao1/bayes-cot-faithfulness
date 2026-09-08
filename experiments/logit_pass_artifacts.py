"""The constants and the pure artifact builders of `experiments/logit_pass.py`.

Split out of the pass itself so the runner file stays about running: everything here is a
pure function over dicts, with no client, no filesystem and no network, which is what makes
`logit_check.json` and `logit_pass_meta.json` checkable offline.

`logit_pass.py` imports the constants FROM here rather than the other way round, so there
is no import cycle and exactly one definition of each.
"""

from __future__ import annotations

import time
from collections import Counter

PASS_VERSION = "logit-pass-1"
INTERVENTION_LEVEL = "logit"
OUTCOME_SCALE = "logprob_margin"

# The two per-arm blocks, and the record fields they fill.
ARMS = ("clean", "hinted")
BLOCK_KEY = {"clean": "clean_answer_logprob", "hinted": "hinted_answer_logprob"}

# Reasons a read that COMPLETED is an alignment failure. These fail the family unit check
# (section 9.1) as well as dropping their item, because each of them says the number was
# not read where it was supposed to be read.
ALIGNMENT_REASONS = (
    "letters_not_all_scored",
    "token_does_not_decode_to_its_letter",
    "logprob_above_zero",
    "letter_probability_mass_above_one",
)


def build_check(results: list[dict], meta: dict) -> dict:
    """The section 9.1 unit check on this pass's own reads, in the phase-1 artifact shape.

    ``results`` holds the reads that RETURNED, one row each, so the arithmetic a reader
    runs over it ("every letter scored, every logprob read off a token that decodes to its
    own letter") is a statement about reads that happened. Reads that never returned are
    counted separately under ``incomplete_reads``: they drop their item, and they are not
    evidence about where a logprob was read, so they do not fail the family.
    """
    rows, incomplete, failures = [], [], Counter()
    for res in results:
        for row in res["reads"]:
            tagged = dict(row, record_index=res["record_index"], record_key=res["record_key"])
            if row.get("completed"):
                rows.append(tagged)
                if row.get("failure_reason") in ALIGNMENT_REASONS:
                    failures[row["failure_reason"]] += 1
            else:
                incomplete.append(tagged)
    scored = sum(r["n_letters_scored"] for r in rows)
    requested = sum(r["n_letters_requested"] for r in rows)
    matching = sum(r["n_tokens_matching_letter"] for r in rows)
    mass = [r["probability_mass"] for r in rows]
    hard_failures = [f"{k} on {v} read(s)" for k, v in sorted(failures.items())]
    return {
        **meta,
        "scope": (
            "the section 9.1 unit check computed on THIS pass's own reads: every read is "
            "one arm of one banked record, on the prompt that record was generated "
            "against, scored over that item's own answer letters"
        ),
        "passed_rule": (
            "passed is true when no COMPLETED read is an alignment failure (every letter "
            "scored, every logprob read off a token that decodes to its own letter, every "
            "logprob at or below zero, letter mass at or below 1) and at least one read "
            "completed. A read that never returned drops its item and is counted under "
            "incomplete_reads; it is a transport failure and says nothing about where a "
            "logprob was read, so it does not fail the family"
        ),
        "n_probes": len(rows) + len(incomplete),
        "n_probes_completed": len(rows),
        "n_letters_requested": requested,
        "n_letters_scored": scored,
        "n_tokens_matching_letter": matching,
        "n_argmax_is_target": sum(1 for r in rows if r["argmax_is_target"]),
        "argmax_note": (
            "reported, never gating. The target letter is a planted WRONG option, so it "
            "is not expected to be the argmax and a low count is not a failure"
        ),
        "letter_probability_mass": _summary_stats(mass),
        "n_reads_with_mass_below_0.01": sum(1 for v in mass if v < 0.01),
        "mass_note": (
            "A5.4 condition 6 and part 4.4: a margin computed where the letters hold a "
            "tiny share of the next-token mass is well defined and is also about a region "
            "the model almost never enters, so the summary prints with the row"
        ),
        "hard_failures": hard_failures,
        "passed": not hard_failures and bool(rows),
        "incomplete_reads": incomplete[:50],
        "n_incomplete_reads": len(incomplete),
        "results": rows,
    }


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "median": float(median),
        "max": float(ordered[-1]),
    }


def _margin_stats(records: list[dict], key: str) -> dict:
    values = [r[key]["logprob_margin"] for r in records if r.get(key)]
    if not values:
        return {"n": 0, "mean": None, "variance": None}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"n": len(values), "mean": mean, "variance": var}


def build_meta(state: dict, check: dict, sidecar: list[dict]) -> dict:
    """`logit_pass_meta.json`: what ran, what it read, what it dropped, and by how much."""
    keys = state["keys"]
    clean = _margin_stats(sidecar, BLOCK_KEY["clean"])
    hinted = _margin_stats(sidecar, BLOCK_KEY["hinted"])
    arm_difference = None
    if clean["mean"] is not None and hinted["mean"] is not None:
        arm_difference = hinted["mean"] - clean["mean"]
    return {
        "pass_version": PASS_VERSION,
        "intervention_level": INTERVENTION_LEVEL,
        "outcome_scale": OUTCOME_SCALE,
        "spec": "docs/OUTCOME-SCALE-NOTE.md part 4.5 job B; prereg Amendment A5",
        "cell_dir": str(state["cell_dir"]),
        "out_dir": str(state["out_dir"]),
        "source": {
            "file": keys["source_file"],
            "sha256": keys["source_sha256"],
            "n_records": state["n_source_records"],
        },
        "sidecar": {"file": state["sidecar_path"].name, "n_records": len(sidecar)},
        "model": keys["model"],
        "hf_revision": keys["hf_revision"],
        "endpoint": keys["endpoint"],
        "endpoint_rule": (
            "section 6.7: SoCLaaS is INELIGIBLE as a source of logprob outcomes, so this "
            "pass refuses any endpoint that is not the loopback address of a server "
            "started by the job that runs it"
        ),
        "seed": state["seed"],
        "chat_template_kwargs": state["template_kwargs"],
        "logprob_mode": state["logprob_mode"],
        "concurrency": state["concurrency"],
        "cell": {
            "substrate": state["cell_meta"].get("substrate"),
            "cue_family": state["cell_meta"].get("cue_family"),
            "job_id": state["cell_meta"].get("job_id"),
            "text_level_intervention_level": state["cell_meta"].get("intervention_level"),
            "text_level_outcome_scale": state["cell_meta"].get("outcome_scale"),
        },
        "n_items_entered": len(state["results"]),
        "n_items_scored": len(sidecar),
        "n_items_dropped": len(state["results"]) - len(sidecar),
        "drops_by_reason": dict(sorted(state["drops"].items())),
        "n_reads_attempted": check["n_probes"],
        "n_reads_completed": check["n_probes_completed"],
        "family_unit_check": {
            "artifact": "logit_check.json",
            "passed": check["passed"],
            "hard_failures": check["hard_failures"],
            "rule": (
                "A5.4 condition 1 and section 9.1. A family that fails is reported at the "
                "TEXT level only, so this pass exits 7 and the sbatch stops that model"
            ),
        },
        "letter_probability_mass": check["letter_probability_mass"],
        "clean_arm_margin": clean,
        "hinted_arm_margin": hinted,
        "randomized_arm_difference_in_the_margin": arm_difference,
        "arm_difference_note": (
            "A5.4 condition 5 checks TE_logit against this number to within 1e-6. It is "
            "printed here as the input to that check and is not itself an estimate"
        ),
        "assert_records_scaled_checked": state["asserted"],
        "started_at": state["started_at"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
