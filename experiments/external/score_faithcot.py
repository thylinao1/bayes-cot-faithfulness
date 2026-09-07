"""Score the FINE-CoT corpus with one served judge, on Q1 files a, b and c and on Q2.

EXPLORATORY BY CONSTRUCTION. No Q1 configuration has passed the frozen judge gate
(experiments/jury/GATE-Q1-COMPARISON.md: every judge x Q1 file row reads FAIL), so
every number this produces carries claim_status EXPLORATORY and no Q1 file is ever
selected on these results. All three files are run and all three are reported.

Why this driver rather than experiments.jury.gate. The gate scores a synthetic corpus
whose rows carry a ``gate_class``, and the runner's panel rule calls family_of() on the
subject model, which is defined only for this project's 18-model roster. FaithCoT's
generators are not on that roster and no panel rule is defined for them. So the panel
layer is dropped and the frozen pieces are reused unchanged: the same prompt files, the
same render(), the same validate_output(), the same vllm endpoint at temperature 0. The
SHA-256 of every prompt file lands on every vote.

It also runs each of Q1a, Q1b, Q1c and Q2 exactly once per item per run, where the gate
would re-run Q2 once per Q1 variant. That is three times less generation for the same
measurements.

Cite arXiv:2510.04040 when reporting these numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import threading
import time
from concurrent import futures

from experiments.jury.backends import vllm_endpoint
from experiments.jury.family_map import JUDGE_BY_KEY
from experiments.jury.prompt_files import PROMPT_DIR, load_prompts, render, validate_output

Q1_VARIANTS = ("a", "b", "c")
AUDIT_FRACTION = 0.10


def sha256_text(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_slice(item_ids: list[str], *, seed: int, fraction: float = AUDIT_FRACTION) -> set[str]:
    """The pre-registered seeded 10 percent that gets three runs (section 6.4)."""
    unique = sorted(set(item_ids))
    n = round(fraction * len(unique))
    ranked = sorted(unique, key=lambda i: hashlib.sha256(f"audit|{seed}|{i}".encode()).hexdigest())
    return set(ranked[:n])


def subsample(item_ids: list[str], *, seed: int, size: int) -> set[str]:
    """A seeded random subsample, drawn by hashing the id so it is reproducible anywhere."""
    unique = sorted(set(item_ids))
    if size <= 0 or size >= len(unique):
        return set(unique)
    ranked = sorted(unique, key=lambda i: hashlib.sha256(f"subsample|{seed}|{i}".encode()).hexdigest())
    return set(ranked[:size])


def build_tasks(items: list[dict], *, seed: int, runs: int, audit: set[str]) -> list[tuple]:
    """One task per (item, question, run). Run 0 is every item; runs 1 and 2 are the audit."""
    tasks = []
    for item in items:
        for question in ("Q1a", "Q1b", "Q1c", "Q2"):
            for run in range(runs):
                if run > 0 and item["item_id"] not in audit:
                    continue
                tasks.append((item, question, run))
    return tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", required=True, help="judge_key=base_url")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runs", type=int, default=3, help="seeded runs on the audit slice")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num-predict", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0, help="seeded subsample size, 0 for all")
    ap.add_argument("--serving-line", default="", help="set when NOT the line section 6.1 pins")
    ap.add_argument("--prompt-dir", default=str(PROMPT_DIR))
    args = ap.parse_args(argv)

    judge_key, _, base_url = args.judge.partition("=")
    if judge_key not in JUDGE_BY_KEY:
        raise SystemExit(f"unknown judge key {judge_key!r}; known {sorted(JUDGE_BY_KEY)}")
    judge = JUDGE_BY_KEY[judge_key]

    prompt_dir = pathlib.Path(args.prompt_dir)
    # One prompt object per question, with its file and SHA-256 pinned on every vote.
    prompts = {}
    for variant in Q1_VARIANTS:
        prompts[f"Q1{variant}"] = load_prompts(prompt_dir, q1=variant)["Q1"]
    prompts["Q2"] = load_prompts(prompt_dir, q1="a")["Q2"]
    prompt_sha = {q: sha256_text(p.path) for q, p in prompts.items()}
    prompt_name = {q: p.path.name for q, p in prompts.items()}

    items = load_items(pathlib.Path(args.items))
    ids = [i["item_id"] for i in items]
    keep = subsample(ids, seed=args.seed, size=args.limit)
    items = [i for i in items if i["item_id"] in keep]
    audit = audit_slice([i["item_id"] for i in items], seed=args.seed)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    votes_path = out_dir / "votes.jsonl"

    # Resume: a vote already on disk is never regenerated.
    done: set[tuple[str, str, int]] = set()
    if votes_path.exists():
        for line in votes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            done.add((v["item_id"], v["question"], v["run"]))
        print(f"[resume] {len(done)} votes already on disk", flush=True)

    tasks = [t for t in build_tasks(items, seed=args.seed, runs=args.runs, audit=audit)
             if (t[0]["item_id"], t[1], t[2]) not in done]

    print(json.dumps({
        "judge": judge_key, "hf_id": judge.hf_id, "revision": judge.revision,
        "serving_line_of_record": judge.serving_line,
        "serving_line_this_run": args.serving_line or judge.serving_line,
        "items": len(items), "audit_slice": len(audit), "runs": args.runs,
        "tasks_remaining": len(tasks), "seed": args.seed, "limit": args.limit,
        "prompt_files": prompt_name, "prompt_sha256": prompt_sha,
    }, indent=2), flush=True)

    lock = threading.Lock()
    endpoints = {r: vllm_endpoint(judge_key, base_url, seed=args.seed + r) for r in range(args.runs)}
    started = time.time()
    counters = {"ok": 0, "malformed": 0, "error": 0}

    def score(task) -> dict:
        item, question, run = task
        prompt = prompts[question]
        text = render(
            prompt,
            question=item["question"],
            choices=item["choices"],
            reasoning=item["reasoning"],
            final_answer=item["final_answer"] if prompt.sees_final_answer else None,
        )
        endpoint = endpoints[run]
        status, vote, raw, error = "ok", None, "", None
        # One retry on the same seed, then `malformed`, per section 6.6 rule (ii).
        for attempt in range(2):
            try:
                raw = endpoint.generate(text, num_predict=args.num_predict)
            except Exception as exc:  # noqa: BLE001  a backend failure is recorded, never swallowed
                status, error = "error", f"{type(exc).__name__}: {exc}"[:300]
                continue
            result = validate_output(prompt, raw)
            if result.ok:
                status, vote, error = "ok", result.vote, None
                break
            status, error = "malformed", result.error
        return {
            "item_id": item["item_id"], "question": question, "run": run,
            "judge_key": judge_key, "judge_hf_id": judge.hf_id,
            "judge_revision": judge.revision, "judge_backend": "vllm",
            "serving_line": args.serving_line or judge.serving_line,
            "serving_line_is_pinned": not args.serving_line,
            "prompt_file": prompt_name[question], "prompt_sha256": prompt_sha[question],
            "seed": args.seed + run, "num_predict": args.num_predict,
            "status": status, "vote": vote, "error": error,
            "raw": raw if status != "ok" else "",
            "claim_status": "EXPLORATORY",
            "meta": item["meta"],
        }

    with (
        votes_path.open("a", encoding="utf-8") as fh,
        futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
            for n, record in enumerate(pool.map(score, tasks), start=1):
                with lock:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counters[record["status"]] = counters.get(record["status"], 0) + 1
                    if n % 200 == 0:
                        rate = n / max(time.time() - started, 1e-9)
                        left = (len(tasks) - n) / max(rate, 1e-9)
                        fh.flush()
                        print(f"[score] {n}/{len(tasks)} at {rate:.2f}/s, "
                              f"about {left/60:.1f} min left, {counters}", flush=True)

    summary = {
        "judge": judge_key, "revision": judge.revision,
        "serving_line": args.serving_line or judge.serving_line,
        "items": len(items), "audit_slice": len(audit), "runs": args.runs,
        "seed": args.seed, "votes_written": len(tasks), "counters": counters,
        "elapsed_s": round(time.time() - started, 1),
        "prompt_files": prompt_name, "prompt_sha256": prompt_sha,
        "claim_status": "EXPLORATORY",
        "claim_status_evidence": (
            "No Q1 configuration has passed the frozen gate "
            "(experiments/jury/GATE-Q1-COMPARISON.md, every row FAIL), so every jury "
            "number from this run is exploratory and no Q1 file is selected on it."
        ),
        "dataset": "github.com/se7esx/FaithCoT-BENCH",
        "dataset_revision": "6e3c004cbbde5bf47352df91e3ac399d2fb4593e",
        "cite": "arXiv:2510.04040",
    }
    (out_dir / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
