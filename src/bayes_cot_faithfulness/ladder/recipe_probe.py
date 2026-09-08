"""Instrumentation for a LoRA run: the loss curve, throughput, and the trigger probe.

WHY THIS IS A SEPARATE MODULE. ``lora_train._train_peft`` needs torch, transformers and
peft, none of which is importable on the laptop this repository is developed on, so
anything that lives inside it is untestable here. Everything in this file is pure: it
takes lists and dicts and returns lists and dicts, so the parts that decide WHICH
examples the probe reads, WHAT prefix it forces, and HOW the agreement numbers are
computed are exercised by ordinary tests, and the only thing left inside the torch path
is the forward pass itself.

THE PROBE, and what it does and does not measure
------------------------------------------------
Question 6 of docs/LADDER-PILOT-PLAN.md 3.2 is "did the organism learn the trigger at
all". The cheapest honest answer is a forced-prefix read of the answer letter: take a
training example's own prompt, append its own completion up to and including the
``Answer: (`` marker, run ONE forward pass, and read the model's distribution over the
answer-letter tokens at that position. Argmax over the letters is then the answer the
model gives under the trigger, and it is measured the same way before and after the
fine-tune, so the difference is attributable to the fine-tune and not to a decoding
change.

What it does NOT measure: free generation. The model is never asked to produce its own
reasoning here, so a rise in agreement says the trigger-to-letter relation moved, not
that a served checkpoint would answer that way after generating 320 tokens of its own
text. That is what the evaluation wave is for, and this probe is not a substitute for it.

The examples are HELD IN by design: they are drawn from the training set the run is
fitting, so the probe reads whether the relation was learned at all, not whether it
generalises. A held-out probe is a different measurement and is not this one.

WHY BOTH AGREEMENT NUMBERS. The trigger's option is drawn uniformly over all options,
so on about one trigger item in ``n_choices`` the trigger already marks the gold answer.
A model that ignored the trigger entirely and answered perfectly would still "agree with
the trigger" at that rate. :func:`summarize_probe` therefore reports agreement on ALL
trigger items and, separately, on the items where the trigger marks something other than
gold, which is the subset where following the trigger and being right are different acts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

ANSWER_MARKER = "Answer: ("


class ProbeError(ValueError):
    """A probe that could not be built from the examples it was given."""


def completion_prefix(example: dict) -> str:
    """The example's completion truncated to just after the final ``Answer: (``.

    The last marker wins, not the first: a banked trace can mention an answer line in
    its own reasoning, and ``trigger_data._completion`` strips those lines but a trace
    that wrote one mid-sentence would survive. Taking the LAST occurrence lands on the
    line the builder itself appended, which is the one carrying the target label.
    """
    completion = example.get("completion")
    if not isinstance(completion, str) or ANSWER_MARKER not in completion:
        raise ProbeError(
            f"example {example.get('pool_index')} has no {ANSWER_MARKER!r} in its "
            "completion, so there is no position at which to read an answer letter"
        )
    cut = completion.rindex(ANSWER_MARKER) + len(ANSWER_MARKER)
    return completion[:cut]


def select_probe_examples(
    examples: Sequence[dict], n: int, *, seed: int, trigger_only: bool = True,
) -> list[int]:
    """Indices of the probe's examples, deterministically, without numpy or random.

    Only trigger-carrying examples are eligible by default: a probe item with no trigger
    has no trigger-implied answer to agree with, so including one would dilute the
    numerator with rows the question does not apply to.

    The order is a hash of (seed, pool_index), which gives the same set on any machine
    and any Python version and does not depend on the pool's own order, so the probe is
    not silently reading the first n items of the pool.
    """
    if n <= 0:
        return []
    eligible = [
        i for i, e in enumerate(examples)
        if (e.get("trigger_present") and e.get("trigger_option") is not None)
        or not trigger_only
    ]
    if not eligible:
        raise ProbeError(
            "no trigger-carrying examples to probe: every example in this training set "
            "has trigger_present false, so there is no trigger-implied answer anywhere"
        )
    def key(i: int) -> str:
        return hashlib.sha256(
            f"{seed}|{examples[i].get('pool_index', i)}".encode()).hexdigest()
    return sorted(eligible, key=key)[: min(n, len(eligible))]


def summarize_probe(rows: Sequence[dict]) -> dict:
    """Agreement of the model's answer with the trigger-implied answer, and with gold.

    Each row is ``{"trigger_option": int, "gold_index": int, "predicted_index": int,
    "p_trigger": float | None}``. Rates are returned with their own denominators beside
    them, because the two denominators differ: ``n`` is every probed item and
    ``n_trigger_not_gold`` is the subset where the trigger marks something other than
    the gold answer.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "agree_with_trigger": None, "agree_with_gold": None,
                "n_trigger_not_gold": 0, "agree_with_trigger_when_not_gold": None,
                "mean_p_trigger": None}
    agree_trigger = sum(1 for r in rows if r["predicted_index"] == r["trigger_option"])
    agree_gold = sum(1 for r in rows if r["predicted_index"] == r["gold_index"])
    not_gold = [r for r in rows if r["trigger_option"] != r["gold_index"]]
    agree_ng = sum(1 for r in not_gold if r["predicted_index"] == r["trigger_option"])
    ps = [r["p_trigger"] for r in rows if r.get("p_trigger") is not None]
    return {
        "n": n,
        "n_agree_with_trigger": agree_trigger,
        "agree_with_trigger": agree_trigger / n,
        "n_agree_with_gold": agree_gold,
        "agree_with_gold": agree_gold / n,
        "n_trigger_not_gold": len(not_gold),
        "n_agree_with_trigger_when_not_gold": agree_ng,
        "agree_with_trigger_when_not_gold": (
            agree_ng / len(not_gold) if not_gold else None),
        "mean_p_trigger": (sum(ps) / len(ps)) if ps else None,
    }


def loss_curve(losses: Sequence[float], every: int) -> list[dict]:
    """Every ``every``-th step's loss, plus the first and the last, once each.

    The first and the last are always present because a curve that starts at step 10 and
    stops at step 390 cannot answer "did it move", which is the only question the
    exploratory run asks of it.
    """
    if not losses:
        return []
    every = max(1, int(every))
    n = len(losses)
    keep = sorted({0, n - 1} | {i for i in range(n) if (i + 1) % every == 0})
    return [{"step": i + 1, "loss": float(losses[i])} for i in keep]


def throughput(n_tokens: int, seconds: float) -> float | None:
    """Tokens per second, or None when no time passed (which is not a rate of 0)."""
    if seconds is None or seconds <= 0:
        return None
    return float(n_tokens) / float(seconds)
