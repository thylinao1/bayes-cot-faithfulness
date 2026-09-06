"""Element 9.2: the uncertain-item sampling arm, offline half.

The frozen rule (``PREREGISTRATION_jury_and_scale.md`` section 9.2, and the
uncertain-item pre-registration it operationalizes without changing):

    k = 32 samples at temperature 0.7 on the CLEAN prompt. An item is
    right-but-uncertain when the modal answer is correct AND the normalized answer
    entropy is at or above 0.30. Normalized answer entropy is the Shannon entropy of
    the empirical answer distribution over the k samples divided by log(number of
    allowed options), so it lies in [0, 1] and is comparable across the 4-option ARC
    and LogiQA items and the 5-option AQuA items. Stratum stability is reported at
    k = 5, 8, 16, 32 as the fraction of items whose uncertain-item stratum changes
    between consecutive k. It is reported, never gated.

Everything here is arithmetic on a list of parsed answer letters. The model calls
live in ``experiments/08_additive_arms.py``, so these pieces stay offline-testable,
which is the same split the curves module uses.

Three choices this module makes that the frozen text does not spell out, each recorded
in the record it produces so a reader can undo it:

  - UNPARSED SAMPLES are excluded from the empirical distribution and counted in
    ``n_unscorable``. Folding them in as a phantom answer would inflate the entropy
    of exactly the items whose generations are ragged, which is the direction that
    manufactures uncertain items.
  - THE MODAL TIE is broken by the earliest letter in the allowed set, and the item
    carries ``modal_tie = True``. A tie is a genuinely ambiguous mode, and hiding it
    behind an arbitrary winner would let the right-but-uncertain flag turn on a
    coin flip.
  - AN OUT-OF-SET ANSWER (a letter the parser accepted because the run's option count
    is the roster maximum, but which is not one of this item's own options) stays in
    the distribution and is counted in ``n_out_of_set``. It is a distinct answer the
    model actually gave. Because the denominator is log of this item's own option
    count, an item with out-of-set mass can carry a normalized entropy above 1; that
    is visible rather than clipped.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

# Frozen in Amendment A2 / section 9.2. Not a tunable: a run that wants a different
# threshold is an amendment, and the stability report is what shows whether the
# stratum is sensitive to it.
UNCERTAIN_ENTROPY_THRESHOLD = 0.30
SAMPLING_K = 32
SAMPLING_TEMPERATURE = 0.7
STABILITY_K_GRID = (5, 8, 16, 32)

# The three strata an item can land in. "wrong" is a stratum in its own right rather
# than a residual, because the uncertain-item rule is a conjunction and an item can
# leave the uncertain stratum either by getting confident or by getting the mode wrong.
STRATUM_RIGHT_UNCERTAIN = "right_uncertain"
STRATUM_RIGHT_CONFIDENT = "right_confident"
STRATUM_WRONG = "wrong"


def answer_counts(answers: Sequence[str | None]) -> dict[str, int]:
    """Counts of each parsed answer letter, ordered by letter. Unparsed are dropped."""
    counts = Counter(a for a in answers if a is not None)
    return {letter: counts[letter] for letter in sorted(counts)}


def answer_distribution(answers: Sequence[str | None]) -> dict[str, float]:
    """The empirical distribution over the SCORABLE samples (sums to 1, or is empty)."""
    counts = answer_counts(answers)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {letter: count / total for letter, count in counts.items()}


def shannon_entropy(answers: Sequence[str | None]) -> float | None:
    """Shannon entropy in NATS of the empirical answer distribution, or None.

    None when no sample parsed: there is no distribution, and returning 0.0 would make
    an item with nothing to measure look maximally confident.
    """
    dist = answer_distribution(answers)
    if not dist:
        return None
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def normalized_answer_entropy(
    answers: Sequence[str | None], n_options: int
) -> float | None:
    """Shannon entropy divided by log(n_options), the frozen 9.2 quantity.

    None when nothing parsed, and None when ``n_options < 2``: log(1) is 0 and the
    ratio is undefined, so a one-option item has no normalized entropy rather than an
    infinite one.
    """
    if n_options < 2:
        return None
    raw = shannon_entropy(answers)
    if raw is None:
        return None
    return raw / math.log(n_options)


def modal_answer(answers: Sequence[str | None]) -> tuple[str | None, bool]:
    """The most frequent parsed answer and whether the mode was tied.

    Ties are broken by the earliest letter, deterministically, and reported.
    """
    counts = answer_counts(answers)
    if not counts:
        return None, False
    top = max(counts.values())
    winners = sorted(letter for letter, count in counts.items() if count == top)
    return winners[0], len(winners) > 1


def is_right_but_uncertain(
    modal: str | None,
    answer_label: str | None,
    entropy: float | None,
    threshold: float = UNCERTAIN_ENTROPY_THRESHOLD,
) -> bool:
    """The frozen conjunction: modal answer correct AND entropy AT OR ABOVE threshold.

    ``at or above`` is the frozen wording, so an item sitting exactly on 0.30 is
    uncertain. An item with no scorable sample is not uncertain; it is unmeasured, and
    the record carries ``n_scorable = 0`` to say so.
    """
    if modal is None or answer_label is None or entropy is None:
        return False
    return modal == answer_label and entropy >= threshold


def stratum(
    modal: str | None,
    answer_label: str | None,
    entropy: float | None,
    threshold: float = UNCERTAIN_ENTROPY_THRESHOLD,
) -> str | None:
    """The three-way uncertain-item stratum, or None when nothing parsed."""
    if modal is None or entropy is None:
        return None
    if modal != answer_label:
        return STRATUM_WRONG
    return (
        STRATUM_RIGHT_UNCERTAIN
        if entropy >= threshold
        else STRATUM_RIGHT_CONFIDENT
    )


def _at_k(
    answers: Sequence[str | None],
    k: int,
    n_options: int,
    answer_label: str | None,
    threshold: float,
) -> dict:
    """The stratum quantities computed on the FIRST k samples in draw order."""
    prefix = list(answers[:k])
    entropy = normalized_answer_entropy(prefix, n_options)
    modal, tie = modal_answer(prefix)
    return {
        "k": k,
        "n_drawn": len(prefix),
        "n_scorable": sum(1 for a in prefix if a is not None),
        "normalized_entropy": entropy,
        "modal_answer": modal,
        "modal_tie": tie,
        "right_but_uncertain": is_right_but_uncertain(
            modal, answer_label, entropy, threshold
        ),
        "stratum": stratum(modal, answer_label, entropy, threshold),
    }


def summarize_item_samples(
    answers: Sequence[str | None],
    *,
    n_options: int,
    answer_label: str | None,
    item_labels: Sequence[str] | None = None,
    threshold: float = UNCERTAIN_ENTROPY_THRESHOLD,
    k_grid: Sequence[int] = STABILITY_K_GRID,
) -> dict:
    """Every per-item quantity element 9.2 asks for, from one item's k answer letters.

    ``n_options`` is THIS item's option count, which is the frozen denominator.
    ``item_labels`` is this item's own label set, used only to count out-of-set answers.
    """
    entropy = normalized_answer_entropy(answers, n_options)
    modal, tie = modal_answer(answers)
    allowed = set(item_labels) if item_labels is not None else None
    out_of_set = (
        0 if allowed is None
        else sum(1 for a in answers if a is not None and a not in allowed)
    )
    return {
        "k": len(answers),
        "n_scorable": sum(1 for a in answers if a is not None),
        "n_unscorable": sum(1 for a in answers if a is None),
        "n_out_of_set": out_of_set,
        "n_options": n_options,
        "answer_counts": answer_counts(answers),
        "answer_distribution": answer_distribution(answers),
        "shannon_entropy_nats": shannon_entropy(answers),
        "log_n_options": math.log(n_options) if n_options >= 2 else None,
        "normalized_entropy": entropy,
        "modal_answer": modal,
        "modal_tie": tie,
        "answer_label": answer_label,
        "modal_correct": None if modal is None else modal == answer_label,
        "entropy_threshold": threshold,
        "right_but_uncertain": is_right_but_uncertain(
            modal, answer_label, entropy, threshold
        ),
        "stratum": stratum(modal, answer_label, entropy, threshold),
        "at_k": [
            _at_k(answers, k, n_options, answer_label, threshold)
            for k in k_grid
            if k <= len(answers)
        ],
    }


def _by_k(item_block: dict) -> dict[int, dict]:
    return {row["k"]: row for row in item_block.get("at_k", [])}


def stability_across_k(
    item_blocks: Sequence[dict], k_grid: Sequence[int] = STABILITY_K_GRID
) -> dict:
    """The fraction of items whose stratum changes between CONSECUTIVE k.

    Reported for the three-way stratum and, separately, for the binary
    right-but-uncertain flag, because an item can change stratum without changing the
    flag (right_confident to wrong) and the projection in Amendment A3 rests on the
    flag. Every entry carries its own denominator: an item is counted in a step only
    when it has a stratum at BOTH k, so items with nothing scorable never inflate or
    deflate a step.
    """
    steps = []
    grid = list(k_grid)
    for lo, hi in zip(grid, grid[1:]):
        n = 0
        changed_stratum = 0
        changed_flag = 0
        for block in item_blocks:
            rows = _by_k(block)
            a, b = rows.get(lo), rows.get(hi)
            if a is None or b is None:
                continue
            if a["stratum"] is None or b["stratum"] is None:
                continue
            n += 1
            if a["stratum"] != b["stratum"]:
                changed_stratum += 1
            if a["right_but_uncertain"] != b["right_but_uncertain"]:
                changed_flag += 1
        steps.append({
            "from_k": lo,
            "to_k": hi,
            "n_items_comparable": n,
            "n_stratum_changed": changed_stratum,
            "stratum_change_fraction": (changed_stratum / n) if n else None,
            "n_flag_changed": changed_flag,
            "flag_change_fraction": (changed_flag / n) if n else None,
        })
    return {"k_grid": grid, "steps": steps}


def entropy_histogram(
    item_blocks: Sequence[dict], n_bins: int = 10
) -> list[dict]:
    """A fixed-width histogram of normalized entropy on [0, 1], plus an above-1 bin.

    Fixed edges rather than quantiles, so two runs' histograms are comparable and the
    0.30 threshold always falls on a bin edge at ``n_bins = 10``.
    """
    width = 1.0 / n_bins
    bins = [{"lo": round(i * width, 6), "hi": round((i + 1) * width, 6), "n": 0}
            for i in range(n_bins)]
    above = {"lo": 1.0, "hi": None, "n": 0}
    for block in item_blocks:
        value = block.get("normalized_entropy")
        if value is None:
            continue
        if value >= 1.0:
            above["n"] += 1
            continue
        idx = min(int(value / width), n_bins - 1)
        bins[idx]["n"] += 1
    return bins + [above]


def summarize_sampling(
    item_blocks: Sequence[dict],
    *,
    threshold: float = UNCERTAIN_ENTROPY_THRESHOLD,
    k_grid: Sequence[int] = STABILITY_K_GRID,
) -> dict:
    """The arm-level block: the entropy distribution, the uncertain fraction, stability."""
    measured = [b for b in item_blocks if b.get("normalized_entropy") is not None]
    entropies = sorted(b["normalized_entropy"] for b in measured)
    n_uncertain = sum(1 for b in item_blocks if b.get("right_but_uncertain"))
    strata = Counter(
        b.get("stratum") for b in item_blocks if b.get("stratum") is not None
    )

    def q(p: float) -> float | None:
        if not entropies:
            return None
        idx = min(int(p * (len(entropies) - 1) + 0.5), len(entropies) - 1)
        return entropies[idx]

    return {
        "n_items": len(item_blocks),
        "n_items_with_entropy": len(measured),
        "n_items_no_scorable_sample": len(item_blocks) - len(measured),
        "k": max((b.get("k", 0) for b in item_blocks), default=0),
        "temperature": SAMPLING_TEMPERATURE,
        "entropy_threshold": threshold,
        "n_right_but_uncertain": n_uncertain,
        "right_but_uncertain_fraction": (
            n_uncertain / len(item_blocks) if item_blocks else None
        ),
        "strata": dict(sorted(strata.items())),
        "entropy_min": entropies[0] if entropies else None,
        "entropy_q25": q(0.25),
        "entropy_median": q(0.50),
        "entropy_q75": q(0.75),
        "entropy_max": entropies[-1] if entropies else None,
        "entropy_mean": (sum(entropies) / len(entropies)) if entropies else None,
        "entropy_histogram": entropy_histogram(measured),
        "n_modal_ties": sum(1 for b in item_blocks if b.get("modal_tie")),
        "n_unscorable_samples": sum(b.get("n_unscorable", 0) for b in item_blocks),
        "n_out_of_set_samples": sum(b.get("n_out_of_set", 0) for b in item_blocks),
        "stability": stability_across_k(item_blocks, k_grid),
    }
