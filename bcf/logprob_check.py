"""Forced-answer-logprob unit check against a LIVE vLLM server.

The claim this check has to earn: the number stored in the CONTRACT ``answer_logprobs``
field is the logprob of the intended ANSWER-LETTER token, not of some template or
punctuation token that happened to sit last in the prompt.

Three assertions, each printed with its denominator:

  1. every requested letter got a logprob, and the decoded token it was read off
     normalises to that letter;
  2. the logprobs are a distribution over a real next-token position: each is <= 0, and
     exponentiating and summing over the four letters does not exceed 1 (a value above 1
     would mean the positions are not mutually exclusive, i.e. the read is misaligned);
  3. on an item whose answer is unambiguous the intended letter is the argmax, which is
     the end-to-end sanity that the scored position tracks the prompt at all.

Assertion 3 is a model-behaviour check, so it is reported and does not fail the job on
its own; 1 and 2 are alignment checks and DO fail it.

    python bcf/logprob_check.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from openai_client import OpenAIClient, OpenAIClientError, _strip_token  # noqa: E402

LETTERS = ["A", "B", "C", "D"]

# Two items whose correct letter is not in dispute, so assertion 3 means something.
# Deliberately not from any substrate pool: this check must not consume study items.
PROBES = [
    {
        "prompt": (
            "Which of these is a prime number?\n"
            "(A) 21\n(B) 22\n(C) 23\n(D) 24\n\n"
            "Answer with the letter only."
        ),
        "expected": "C",
    },
    {
        "prompt": (
            "What is the chemical symbol for water?\n"
            "(A) NaCl\n(B) H2O\n(C) CO2\n(D) O2\n\n"
            "Answer with the letter only."
        ),
        "expected": "B",
    },
]


def check_one(client: OpenAIClient, probe: dict) -> dict:
    result = client.forced_answer_logprobs(probe["prompt"], LETTERS)
    tokens_ok = {
        letter: _strip_token(result.tokens.get(letter, "")) == letter
        for letter in result.logprobs
    }
    mass = sum(math.exp(v) for v in result.logprobs.values())
    argmax = max(result.logprobs, key=result.logprobs.get) if result.logprobs else None
    return {
        "prompt": probe["prompt"].splitlines()[0],
        "expected": probe["expected"],
        "method": result.method,
        "logprobs": result.logprobs,
        "tokens": result.tokens,
        "n_letters_requested": len(LETTERS),
        "n_letters_scored": len(result.logprobs),
        "n_tokens_matching_letter": sum(tokens_ok.values()),
        "all_non_positive": all(v <= 0 for v in result.logprobs.values()),
        "probability_mass": mass,
        "argmax": argmax,
        "argmax_is_expected": argmax == probe["expected"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--logprob-mode", default="auto",
                    choices=["auto", "prompt_logprobs", "top_logprobs"])
    ap.add_argument("--chat-template-kwargs", default=None,
                    help="JSON forwarded to the server's chat template, e.g. "
                         "'{\"enable_thinking\": false}'. MUST match what the run being "
                         "certified uses: under a different template the scored position "
                         "is a different token, so the check certifies nothing.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    template_kwargs = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
    client = OpenAIClient(base_url=a.base_url, model=a.model, seed=a.seed,
                          temperature=0.0, logprob_mode=a.logprob_mode,
                          chat_template_kwargs=template_kwargs,
                          request_log=None)  # this check is not part of the throughput window
    if not client.is_available():
        print(f"[logprob] no server at {a.base_url}")
        return 1

    results, hard_failures = [], []
    for probe in PROBES:
        try:
            row = check_one(client, probe)
        except OpenAIClientError as exc:
            print(f"[logprob] FAILED on {probe['expected']}-item: {exc}")
            hard_failures.append(str(exc))
            continue
        results.append(row)
        print(
            f"[logprob] method={row['method']} "
            f"letters scored {row['n_letters_scored']}/{row['n_letters_requested']}; "
            f"tokens matching their letter {row['n_tokens_matching_letter']}/"
            f"{row['n_letters_scored']}; "
            f"all logprobs <= 0: {row['all_non_positive']}; "
            f"sum exp = {row['probability_mass']:.4f}; "
            f"argmax {row['argmax']} (expected {row['expected']}: {row['argmax_is_expected']})"
        )
        for letter in LETTERS:
            if letter in row["logprobs"]:
                print(f"           {letter}: logprob {row['logprobs'][letter]:.4f} "
                      f"from token {row['tokens'][letter]!r}")
        if row["n_letters_scored"] != row["n_letters_requested"]:
            hard_failures.append("not every letter was scored")
        if row["n_tokens_matching_letter"] != row["n_letters_scored"]:
            hard_failures.append("a logprob was read off a token that is not its letter")
        if not row["all_non_positive"]:
            hard_failures.append("a logprob was positive")
        if row["probability_mass"] > 1.0 + 1e-6:
            hard_failures.append(
                f"probability mass {row['probability_mass']:.4f} exceeds 1: the scored "
                "positions are not one mutually exclusive next-token slot"
            )

    n_argmax_ok = sum(r["argmax_is_expected"] for r in results)
    print(f"[logprob] argmax matched the unambiguous answer on {n_argmax_ok}/{len(results)} "
          "probe items (reported, not gating)")
    n_thin = sum(1 for r in results if r["probability_mass"] < 0.01)
    if n_thin:
        print(f"[logprob] NOTE: {n_thin}/{len(results)} probes put under 1 percent of the "
              "mass on any answer letter. The scored position is off-distribution, which "
              "usually means the chat template differs from the run's "
              f"(--chat-template-kwargs here: {a.chat_template_kwargs}).")

    payload = {
        "base_url": a.base_url, "model": a.model, "seed": a.seed,
        "chat_template_kwargs": template_kwargs,
        "vllm_version": client.server_version(),
        "n_probes": len(PROBES), "n_probes_completed": len(results),
        "n_argmax_correct": n_argmax_ok,
        "hard_failures": hard_failures,
        "passed": not hard_failures and len(results) == len(PROBES),
        "results": results,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=2))
        print(f"[logprob] wrote {a.out}")

    if hard_failures:
        print(f"[logprob] FAILED: {len(hard_failures)} hard failure(s): {hard_failures}")
        return 2
    print(f"[logprob] PASSED {len(results)}/{len(PROBES)} probe items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
