"""Recompute wave-1 cell rates straight from the cluster checkpoint records.

Independent of arms_summary.json: reads the checkpoint the runner wrote, counts
clean accuracy (clean-correct / entered) and the single-shot follow rate over the
twostep-scorable records, and prints both beside the summary's own numbers so the
two can be compared. Read-only; writes nothing under ~/bcf/results.
"""
from __future__ import annotations

import glob
import json
import os
import sys

RESULTS = os.path.expanduser("~/bcf/results")
CELLS = [
    "qwen3-8b", "olmo-3-7b-think", "deepseek-r1-0528-qwen3-8b",
    "deepseek-r1-distill-llama-8b", "gemma-2-9b-it", "llama-3.1-8b-instruct",
    "phi-4-reasoning", "gpt-oss-20b",
]


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError:
        return None


def _read(path):
    """Strip-read a small file, or None when it does not exist."""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def cell_dir(slug):
    return os.path.join(RESULTS, slug, "arc_challenge", "stated-hint")


def recompute(slug):
    d = cell_dir(slug)
    out = {"slug": slug, "dir": d}
    meta = load(os.path.join(d, "run_meta.json")) or {}
    out["job_id"] = meta.get("job_id")
    out["model"] = meta.get("model")
    out["num_predict"] = meta.get("num_predict")
    out["chat_template_kwargs"] = meta.get("chat_template_kwargs")
    out["gpu"] = meta.get("gpu")
    out["plan_commit"] = meta.get("plan_commit")
    ck = None
    hits = sorted(glob.glob(os.path.join(d, "arms_checkpoint_*.json")))
    if hits:
        ck = load(hits[0])
        out["checkpoint_file"] = os.path.basename(hits[0])
    ec = os.path.join(d, "exit_code.txt")
    out["exit_code"] = _read(ec)
    pf0 = load(os.path.join(d, "determinism_preflight.json"))
    out["preflight"] = {
        "verdict": pf0.get("verdict"), "exit_code": pf0.get("exit_code"),
        "n_items": pf0.get("n_items"), "batch_invariant_env": pf0.get("batch_invariant_env"),
        "levels": [
            {"concurrency": lv.get("concurrency"),
             "identical": lv.get("identical_completions"),
             "max_abs_letter_logprob_diff": lv.get("max_abs_letter_logprob_diff"),
             "n_compared": lv.get("n_letter_logprobs_compared"),
             "n_missing": lv.get("n_letter_logprobs_missing")}
            for lv in pf0.get("levels_checked", [])],
    } if pf0 else None
    out["throughput"] = load(os.path.join(d, "throughput.json"))
    if ck is None:
        out["error"] = "no checkpoint"
        return out
    recs = ck.get("records") or []
    entered = ck.get("n_items_entered")
    out["n_invocations"] = ck.get("n_invocations")
    out["recompute"] = {
        "n_entered": entered,
        "n_records": len(recs),
        "n_failed_generation": (entered - len(recs)) if entered is not None else None,
        "n_clean_correct": sum(1 for r in recs if r.get("clean_correct")),
        "n_unparseable_clean": sum(1 for r in recs if r.get("clean_answer") is None),
    }
    cc = [r for r in recs if r.get("clean_correct")]
    out["recompute"]["clean_accuracy_over_entered"] = (
        round(out["recompute"]["n_clean_correct"] / entered, 6) if entered else None
    )
    out["recompute"]["clean_accuracy_over_records"] = (
        round(out["recompute"]["n_clean_correct"] / len(recs), 6) if recs else None
    )
    # single-shot follow, over the twostep-scorable records (summarize_twostep's frame)
    rs = [r for r in cc if "twostep_answer" in r]
    scorable = [r for r in rs if r.get("twostep_answer") is not None]
    out["recompute"]["twostep_n_with_key"] = len(rs)
    out["recompute"]["twostep_n_scorable"] = len(scorable)
    out["recompute"]["n_singleshot_follow"] = sum(1 for r in scorable if r.get("followed"))
    out["recompute"]["singleshot_follow_rate"] = (
        round(out["recompute"]["n_singleshot_follow"] / len(scorable), 6) if scorable else None
    )
    # cue-arm follow over ALL clean-correct records (the [2/3] line's own denominator)
    cue = [r for r in cc if "followed" in r]
    out["recompute"]["cue_n_with_followed"] = len(cue)
    out["recompute"]["cue_n_followed"] = sum(1 for r in cue if r.get("followed"))
    out["recompute"]["cue_follow_rate"] = (
        round(out["recompute"]["cue_n_followed"] / len(cue), 6) if cue else None
    )

    s = load(os.path.join(d, "arms_summary.json"))
    if s is None:
        out["summary"] = None
    else:
        blocks = s.get("arms", s)
        two = blocks.get("twostep") if isinstance(blocks, dict) else None
        arms = blocks if isinstance(blocks, dict) else {}
        out["summary"] = {
            "arms": {
                a: {"n": (arms.get(a) or {}).get("n"),
                    "n_unscorable": (arms.get(a) or {}).get("n_unscorable")}
                for a in ("direct", "twostep", "filler", "placebo")
                if isinstance(arms.get(a), dict)
            },
            "n_items": s.get("n_items"),
            "n_clean_correct": s.get("n_clean_correct"),
            "clean_accuracy": s.get("clean_accuracy"),
            "attrition": s.get("attrition"),
            "twostep": two,
        }
    tp = load(os.path.join(d, "throughput.json"))
    out["throughput"] = tp
    ec = os.path.join(d, "exit_code.txt")
    out["exit_code"] = _read(ec)
    pf = load(os.path.join(d, "determinism_preflight.json"))
    if pf:
        out["preflight"] = {
            "verdict": pf.get("verdict"),
            "exit_code": pf.get("exit_code"),
            "n_items": pf.get("n_items"),
            "batch_invariant_env": pf.get("batch_invariant_env"),
            "levels": [
                {
                    "concurrency": lv.get("concurrency"),
                    "identical": lv.get("identical_completions"),
                    "max_abs_letter_logprob_diff": lv.get("max_abs_letter_logprob_diff"),
                    "n_compared": lv.get("n_letter_logprobs_compared"),
                    "n_missing": lv.get("n_letter_logprobs_missing"),
                }
                for lv in pf.get("levels_checked", [])
            ],
        }
    else:
        out["preflight"] = None
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or CELLS
    print(json.dumps([recompute(s) for s in which], indent=1))
