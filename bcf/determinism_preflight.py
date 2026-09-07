"""Ruling R1: the per-cell determinism preflight that runs BEFORE a cell's arms.

Amendment A3.6 pins the serving mode of a powered cell: `VLLM_BATCH_INVARIANT=1`,
vLLM 0.28.0, the FlashAttention backend, `VLLM_USE_FLASHINFER_SAMPLER=0`, the roster's
pinned weight revision, and at most 32 requests in flight. The pin is only worth what it
is checked against, so every cell job proves the property on ITS OWN server before it
spends a card on arms: the 30-item probe at 1 and at 32 in flight has to give an
identical-completion fraction of 30/30 and a maximum absolute letter-logprob difference
of exactly 0.0. Anything else and the cell refuses to run.

Why a per-cell check rather than one campaign-wide probe. Job 826020 measured the
property on ONE model (Qwen3-8B) on ONE MIG 3g.40gb slice. The flag is honored by the
FlashAttention backend; a model whose serving line selects a different backend, or a
build that falls back, would silently lose it, and the only visible symptom would be
transcripts that do not reproduce months later. The measurement costs 60 generations and
120 forced letter-logprob reads, about 30 seconds on the measured 8B line, against a cell
that runs for hours.

This module makes NO ruling on what to do about a failure. It refuses the cell with a
distinct exit code and writes the numbers with their denominators.

    python bcf/determinism_preflight.py --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-8B --data experiments/data/arc_challenge.json \
        --out results/<cell>/determinism_preflight.json

    python bcf/determinism_preflight.py --probe-json some_probe.json \
        --out /tmp/verdict.json          # evaluate an existing probe, run nothing

Exit codes: 0 the cell may run, 10 the cell REFUSES, 2 a usage or plumbing error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bcf"))

# Distinct from every other refusal in serve_and_run.sbatch (2 path, 3 devices,
# 4 revision, 5/6 server, 7 logprob unit check, 8 revision drift, 9 arm count), so a
# refusal is readable from exit_code.txt alone without opening the log.
EXIT_REFUSE = 10

REQUIRED_LEVELS = (1, 32)
REQUIRED_N_ITEMS = 30


def _parse_fraction(text: str | None) -> tuple[int | None, int | None]:
    """'30/30' -> (30, 30). Returns (None, None) on anything else."""
    if not isinstance(text, str) or "/" not in text:
        return None, None
    lhs, _, rhs = text.partition("/")
    try:
        return int(lhs), int(rhs)
    except ValueError:
        return None, None


def evaluate(payload: dict, *, levels=REQUIRED_LEVELS, n_items=REQUIRED_N_ITEMS,
             require_flag: bool = True) -> dict:
    """Gate one probe payload. Returns a verdict dict; never raises on bad content.

    Every condition is checked independently and every one that fails is listed, so a
    refusal names all of its reasons instead of the first one. The comparison read is
    ``vs_concurrency_1``, which is each level against THIS server's own concurrency-1
    row: that is the property the pin claims (batching moves nothing), and it is not the
    cross-serving-mode comparison, which A3.6 records as 10 of 30 and which no cell is
    ever gated on.
    """
    reasons: list[str] = []
    rows = {r.get("concurrency"): r for r in payload.get("rows", [])}

    env = payload.get("batch_invariant_env")
    if require_flag and env != "1":
        reasons.append(
            f"VLLM_BATCH_INVARIANT on the probing client was {env!r}, not '1'; the pinned "
            "serving mode of A3.6 is the flag ON, and a flag-off run is exploratory"
        )

    got_items = payload.get("n_items")
    if got_items != n_items:
        reasons.append(
            f"the probe ran {got_items} items, not the {n_items} the rule names; a shorter "
            "set makes 30/30 unreachable and a longer one is a different measurement"
        )

    checked = []
    for level in levels:
        row = rows.get(level)
        if row is None:
            reasons.append(
                f"concurrency {level} was not probed; levels present: {sorted(k for k in rows if k is not None)}"
            )
            continue
        cmp_ = row.get("vs_concurrency_1") or {}
        same, denom = _parse_fraction(cmp_.get("identical_completions"))
        max_diff = cmp_.get("max_abs_letter_logprob_diff")
        n_compared = cmp_.get("n_letter_logprobs_compared")
        n_missing = cmp_.get("n_letter_logprobs_missing")
        checked.append({
            "concurrency": level,
            "identical_completions": cmp_.get("identical_completions"),
            "max_abs_letter_logprob_diff": max_diff,
            "n_letter_logprobs_compared": n_compared,
            "n_letter_logprobs_missing": n_missing,
            "generations_per_second": row.get("generations_per_second"),
            "logprob_calls_per_second": row.get("logprob_calls_per_second"),
        })
        if same is None or denom is None:
            reasons.append(
                f"concurrency {level}: identical_completions is "
                f"{cmp_.get('identical_completions')!r}, which is not a fraction"
            )
        elif denom != n_items or same != denom:
            reasons.append(
                f"concurrency {level}: identical completions {same}/{denom}, and the rule "
                f"needs {n_items}/{n_items}"
            )
        if max_diff is None:
            reasons.append(
                f"concurrency {level}: no letter-logprob difference was computed "
                f"(n_letter_logprobs_compared={n_compared})"
            )
        elif float(max_diff) != 0.0:
            reasons.append(
                f"concurrency {level}: max abs letter-logprob difference {max_diff}, and "
                "the rule needs exactly 0.0"
            )
        if not n_compared:
            reasons.append(
                f"concurrency {level}: 0 letter logprobs compared, so the logprob half of "
                "the check certifies nothing"
            )
        if n_missing:
            reasons.append(
                f"concurrency {level}: {n_missing} letter logprob(s) missing from the "
                "comparison; a missing value is not an equal one"
            )

    return {
        "rule": ("A3.6 / ruling R1: a powered cell runs only when its own server gives "
                 f"{n_items}/{n_items} identical completions and a maximum absolute "
                 f"letter-logprob difference of exactly 0.0 at {' and '.join(str(x) for x in levels)} "
                 "requests in flight, with VLLM_BATCH_INVARIANT=1."),
        "verdict": "PASS" if not reasons else "REFUSE",
        "exit_code": 0 if not reasons else EXIT_REFUSE,
        "require_flag": require_flag,
        "batch_invariant_env": env,
        "n_items": got_items,
        "n_items_required": n_items,
        "levels_required": list(levels),
        "levels_checked": checked,
        "reasons": reasons,
        "model": payload.get("model"),
        "probe_label": payload.get("label"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="where the verdict json goes; it is written beside the cell's "
                         "own outputs so a refusal is readable from the results tree")
    ap.add_argument("--probe-json", type=Path, default=None,
                    help="evaluate an existing probe_results.json and run nothing. This "
                         "is how the gate is proven able to fail without a GPU.")
    ap.add_argument("--probe-out", type=Path, default=None,
                    help="where the probe writes its own results (default: next to --out "
                         "as determinism_preflight_probe.json)")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--n-items", type=int, default=REQUIRED_N_ITEMS)
    ap.add_argument("--levels", default="1,32")
    ap.add_argument("--num-predict", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--allow-flag-off", action="store_true", default=False,
                    help="do not require VLLM_BATCH_INVARIANT=1. An EXPLORATORY run only: "
                         "A3.6 pins the flag on for every powered cell.")
    a = ap.parse_args(argv)

    levels = tuple(int(x) for x in a.levels.split(",") if x.strip())
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.probe_json is not None:
        payload = json.loads(a.probe_json.read_text())
        payload.setdefault("_source", str(a.probe_json))
    else:
        missing = [f for f, v in (("--base-url", a.base_url), ("--model", a.model),
                                  ("--data", a.data)) if not v]
        if missing:
            print(f"[preflight] REFUSING: {', '.join(missing)} required unless --probe-json "
                  "names an existing probe to evaluate.")
            return 2
        import concurrency_probe

        probe_out = a.probe_out or (a.out.parent / "determinism_preflight_probe.json")
        rc = concurrency_probe.main([
            "--base-url", a.base_url, "--model", a.model, "--data", str(a.data),
            "--n-items", str(a.n_items), "--levels", ",".join(str(x) for x in levels),
            "--num-predict", str(a.num_predict), "--seed", str(a.seed),
            "--chat-template-kwargs", a.chat_template_kwargs,
            "--timeout", str(a.timeout), "--out", str(probe_out),
            "--label", "determinism_preflight",
        ])
        if rc != 0 or not probe_out.exists():
            verdict = {
                "verdict": "REFUSE", "exit_code": EXIT_REFUSE,
                "reasons": [f"the probe itself did not complete (exit {rc}); no "
                            f"determinism measurement exists for this cell"],
                "levels_required": list(levels),
            }
            a.out.write_text(json.dumps(verdict, indent=2) + "\n")
            print(f"[preflight] REFUSING: the probe exited {rc}; wrote {a.out}")
            return EXIT_REFUSE
        payload = json.loads(probe_out.read_text())
        payload["_source"] = str(probe_out)

    verdict = evaluate(payload, levels=levels, n_items=a.n_items,
                       require_flag=not a.allow_flag_off)
    verdict["probe_source"] = payload.get("_source")
    a.out.write_text(json.dumps(verdict, indent=2) + "\n")

    for row in verdict.get("levels_checked", []):
        print(f"[preflight] concurrency {row['concurrency']}: "
              f"identical {row['identical_completions']}, "
              f"max letter-logprob diff {row['max_abs_letter_logprob_diff']} "
              f"over {row['n_letter_logprobs_compared']} comparisons "
              f"({row['n_letter_logprobs_missing']} missing)")
    if verdict["verdict"] == "PASS":
        print(f"[preflight] PASS; wrote {a.out}")
        return 0
    print("[preflight] REFUSING to run this cell:")
    for reason in verdict["reasons"]:
        print(f"[preflight]   {reason}")
    print(f"[preflight] wrote {a.out}")
    return EXIT_REFUSE


if __name__ == "__main__":
    raise SystemExit(main())
