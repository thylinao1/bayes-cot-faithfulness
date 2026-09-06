"""Measure what a served card actually does when more than one request is in flight.

Every throughput number in this campaign so far was taken with a client that issues one
HTTP request at a time (`experiments/openai_client.py`, one urlopen per call, no thread
anywhere in `experiments/08_additive_arms.py`). A vLLM server batches, so those figures
describe the client, not the card. This probe holds the served model, the prompts and the
decoding constants fixed and varies ONLY the number of requests in flight.

It also answers the question the operator has to rule on before a powered wave uses
concurrency: does batching move the numbers a run is scored on? Two are checked.

  - Full generations. The fraction of the 30 completions that are byte-identical to the
    concurrency-1 completions of the same prompts.
  - Forced-answer letter logprobs. The largest absolute difference, over every item and
    every answer letter, between the logprob read at this level and the one read at
    concurrency 1.

This script makes NO ruling. It writes numbers with their denominators.

Usage (inside the sbatch, after the server answers /models):

    python bcf/concurrency_probe.py --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-8B --data experiments/data/arc_challenge.json \
        --n-items 30 --levels 1,8,32,64 --out results/probe.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

from openai_client import OpenAIClient  # noqa: E402


def _load_arms_module():
    """Load the runner by file path (its name starts with a digit), like the tests do.

    The probe drives the SHIPPED ``map_in_order``, not a private copy, so the rate it
    reports is the rate the sweep would get from the same helper.
    """
    path = REPO / "experiments" / "08_additive_arms.py"
    spec = importlib.util.spec_from_file_location("additive_arms_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARMS = _load_arms_module()

from bayes_cot_faithfulness.arms import replay_prompt  # noqa: E402
from bayes_cot_faithfulness.interventions import clean_prompt  # noqa: E402


def load_items(path: Path, n: int):
    return ARMS.load_items(path)[:n]


def run_level(client_factory, items, level: int, num_predict: int) -> dict:
    """One (generations, logprobs) pass at ``level`` requests in flight."""
    client = client_factory(level)

    gen_prompts = [clean_prompt(it) for it in items]
    t0 = time.time()
    completions: list[str] = []
    ARMS.map_in_order(
        gen_prompts,
        lambda i, p: client.generate(p, num_predict=num_predict),
        concurrency=level,
        consume=lambda i, p, r: (completions.append(r), True)[1],
    )
    gen_seconds = time.time() - t0

    # The forced-continuation frame: the item's own clean chain re-read, which is the
    # frame the replay and curve arms score, so the logprobs measured here are the
    # logprobs the sweep reads.
    lp_prompts = [replay_prompt(it, "1. work it through\n2. the answer follows")
                  for it in items]
    letters = [list(it.labels) for it in items]
    t0 = time.time()
    logprobs: list[dict] = []
    ARMS.map_in_order(
        list(range(len(items))),
        lambda i, idx: client.forced_answer_logprobs(lp_prompts[idx], letters[idx]),
        concurrency=level,
        consume=lambda i, idx, r: (logprobs.append(dict(r.logprobs)), True)[1],
    )
    lp_seconds = time.time() - t0

    stats = client.stats()
    return {
        "concurrency": level,
        "n_generations": len(completions),
        "generation_seconds": round(gen_seconds, 3),
        "generations_per_second": round(len(completions) / gen_seconds, 4) if gen_seconds else None,
        "n_logprob_items": len(logprobs),
        "n_logprob_calls": sum(len(x) for x in letters),
        "logprob_seconds": round(lp_seconds, 3),
        "logprob_calls_per_second": (
            round(sum(len(x) for x in letters) / lp_seconds, 4) if lp_seconds else None
        ),
        "client_stats": stats,
        "_completions": completions,
        "_logprobs": logprobs,
    }


def compare(baseline: dict, level: dict) -> dict:
    """Identical-completion fraction and max absolute letter-logprob difference."""
    base_c, this_c = baseline["_completions"], level["_completions"]
    n = min(len(base_c), len(this_c))
    same = sum(1 for a, b in zip(base_c[:n], this_c[:n]) if a == b)

    diffs: list[float] = []
    missing = 0
    for a, b in zip(baseline["_logprobs"], level["_logprobs"]):
        for letter, value in a.items():
            if letter not in b:
                missing += 1
                continue
            diffs.append(abs(value - b[letter]))
    return {
        "identical_completions": f"{same}/{n}",
        "identical_completion_fraction": round(same / n, 4) if n else None,
        "n_letter_logprobs_compared": len(diffs),
        "n_letter_logprobs_missing": missing,
        "max_abs_letter_logprob_diff": max(diffs) if diffs else None,
        "median_abs_letter_logprob_diff": statistics.median(diffs) if diffs else None,
        "n_exactly_zero_diff": sum(1 for d in diffs if d == 0.0),
    }


def compare_raw(baseline_raw: dict, level: dict) -> dict:
    """Same two comparisons, against a baseline loaded from a PREVIOUS server.

    This is what makes the batch-invariant run readable: the flag-on rows have to be
    compared against the flag-OFF concurrency-1 completions, and those live in another
    server's process. Job 825548 wrote none (probe_results.json drops the underscore
    keys), so the flag-off baseline is re-measured in the same job rather than quoted
    across jobs, and the file it writes is what this reads.
    """
    return compare(
        {"_completions": baseline_raw.get("completions", []),
         "_logprobs": baseline_raw.get("logprobs", [])},
        level,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--levels", default="1,8,32,64")
    ap.add_argument("--num-predict", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw-out", type=Path, default=None,
                    help="write the concurrency-1 completions and letter logprobs to "
                         "this file. Needed whenever a LATER run has to compare against "
                         "this one: probe_results.json keeps only the summary, so a "
                         "flag-on rerun cannot be compared with a flag-off run that did "
                         "not write this.")
    ap.add_argument("--baseline-raw", type=Path, default=None,
                    help="a --raw-out file from an earlier server. Every level is then "
                         "compared against it as well as against this run's own "
                         "concurrency-1 row, and both comparisons are reported.")
    ap.add_argument("--label", default=None,
                    help="free-text label carried into the results (e.g. "
                         "batch_invariant_on).")
    a = ap.parse_args(argv)

    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    if not levels or levels[0] != 1:
        print("[probe] REFUSING: the first level must be 1; it is the comparison baseline.")
        return 2

    template_kwargs = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
    items = load_items(a.data, a.n_items)
    if len(items) < a.n_items:
        print(f"[probe] REFUSING: {a.data} holds {len(items)} items, fewer than "
              f"--n-items {a.n_items}; a short set would make the levels incomparable.")
        return 2

    # The baseline is checked BEFORE any level runs, not after: a missing or mismatched
    # file discovered at the end would have already spent the card on a probe whose
    # headline comparison cannot be made.
    baseline_raw = None
    if a.baseline_raw is not None:
        if not a.baseline_raw.exists():
            print(f"[probe] REFUSING: --baseline-raw {a.baseline_raw} does not exist; "
                  f"without it the flag-off comparison would silently be missing.")
            return 2
        baseline_raw = json.loads(a.baseline_raw.read_text())
        n_base = len(baseline_raw.get("completions", []))
        if n_base != len(items):
            print(f"[probe] REFUSING: --baseline-raw holds {n_base} completions, this "
                  f"run has {len(items)} items; the comparison would be off by index.")
            return 2

    def client_factory(level):
        return OpenAIClient(
            base_url=a.base_url, model=a.model, temperature=0.0, timeout=a.timeout,
            seed=a.seed, chat_template_kwargs=template_kwargs, max_in_flight=level,
        )

    results = []
    for level in levels:
        print(f"[probe] concurrency {level}: {len(items)} generations then "
              f"{len(items)} forced-logprob items", flush=True)
        res = run_level(client_factory, items, level, a.num_predict)
        print(f"[probe]   {res['generations_per_second']} gen/s, "
              f"{res['logprob_calls_per_second']} logprob calls/s, "
              f"retries {res['client_stats']['retries']}, "
              f"failed {res['client_stats']['requests_failed']}", flush=True)
        results.append(res)

    baseline = results[0]
    payload = {
        "model": a.model,
        "n_items": len(items),
        "num_predict": a.num_predict,
        "temperature": 0.0,
        "seed": a.seed,
        "chat_template_kwargs": template_kwargs,
        "data": str(a.data),
        "levels": levels,
        "note": (
            "Every generations-per-second figure recorded before this probe came from a "
            "client that issued one request at a time. The concurrency-1 row here is that "
            "same sequential rate, measured on this card, and it is the baseline the other "
            "rows are compared against. No ruling is made on the differences."
        ),
        "rows": [],
    }
    payload["label"] = a.label
    payload["batch_invariant_env"] = os.environ.get("VLLM_BATCH_INVARIANT")
    payload["baseline_raw"] = str(a.baseline_raw) if a.baseline_raw else None
    if baseline_raw is not None:
        payload["baseline_raw_label"] = baseline_raw.get("label")
        payload["baseline_raw_batch_invariant_env"] = baseline_raw.get(
            "batch_invariant_env"
        )
    for res in results:
        row = {k: v for k, v in res.items() if not k.startswith("_")}
        row["vs_concurrency_1"] = compare(baseline, res)
        if baseline_raw is not None:
            row["vs_baseline_raw_concurrency_1"] = compare_raw(baseline_raw, res)
        payload["rows"].append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"[probe] wrote {a.out}")
    if a.raw_out is not None:
        a.raw_out.parent.mkdir(parents=True, exist_ok=True)
        a.raw_out.write_text(json.dumps({
            "label": a.label,
            "batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
            "model": a.model,
            "n_items": len(items),
            "concurrency": levels[0],
            "num_predict": a.num_predict,
            "seed": a.seed,
            "data": str(a.data),
            "completions": baseline["_completions"],
            "logprobs": baseline["_logprobs"],
        }, indent=2))
        print(f"[probe] wrote {a.raw_out} (concurrency-1 raw, for a later comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
