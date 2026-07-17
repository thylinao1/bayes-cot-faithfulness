"""Truncation dose-response curves for the real-model faithfulness experiment.

The single ``--mediator truncation`` point (answer the question with the CoT cut at
one depth) becomes a per-item dose-response curve: truncate the chain-of-thought at a
grid of FRACTIONS of its steps, force an answer at each depth, and summarize how the
answer converges to the full-CoT answer as more reasoning is revealed (Lanham et al.
truncation, arXiv:2307.13702). Where a step-commitment happens is a per-item covariate
the hierarchical model can condition on, so early- and late-committing items are not
pooled blindly.

Everything here is text and structured signal only: this module builds the depths, the
prompts, and the after-the-fact summaries. The model calls that fill in the per-depth
answers live in the runner under ``experiments/``, so these pieces stay offline-testable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bayes_cot_faithfulness.interventions import (
    QAItem,
    continuation_prompt,
    split_steps,
    truncate_cot,
)

# The truncation grid: no CoT, quarter, half, three-quarters, full. Coarse on purpose,
# five forced-answer calls per item keeps the runner's model budget bounded.
DEFAULT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def depths_for(cot: str, fractions: Sequence[float] = DEFAULT_FRACTIONS) -> list[int]:
    """Map each fraction of the CoT to an integer step depth, deduplicated in order.

    ``k = round(f * n_steps)`` over the CoT's ``split_steps`` count. Short chains would
    otherwise map several fractions onto the same ``k``, so duplicates are dropped while
    preserving order. An empty chain yields ``[0]`` (nothing to reveal) and a one-step
    chain yields ``[0, 1]`` (Python's round-half-to-even sends 0.5 to 0). ``fractions``
    are expected ascending but any order is accepted; the first occurrence of each depth
    is kept.
    """
    n_steps = len(split_steps(cot))
    depths: list[int] = []
    for fraction in fractions:
        k = round(fraction * n_steps)
        if k not in depths:
            depths.append(k)
    return depths


def curve_prompts(
    item: QAItem, cot: str, fractions: Sequence[float] = DEFAULT_FRACTIONS
) -> list[tuple[int, str]]:
    """Build ``(depth, forced-answer prompt)`` pairs across the truncation grid.

    Each prompt is ``continuation_prompt`` over ``truncate_cot(cot, k)``, so the pair at
    depth ``k`` asks the model to commit to a final answer given exactly the first ``k``
    reasoning steps. The runner calls the model on each prompt and records the answer.
    """
    return [
        (k, continuation_prompt(item, truncate_cot(cot, k)))
        for k in depths_for(cot, fractions)
    ]


def match_profile(
    answers: Sequence[str | None], final_answer: str | None
) -> tuple[bool, ...]:
    """Per-depth 'this depth's answer equals the full-CoT answer'.

    A ``None`` answer (the model never committed at that depth) matches only when
    ``final_answer`` is also ``None``: a truncated run that never commits agrees with a
    full run that also never commits, but a ``None`` against a real final answer is a
    non-match. Plain equality (``answer == final_answer``) already encodes exactly this,
    since ``None == None`` is true and ``None == "B"`` is false.
    """
    return tuple(answer == final_answer for answer in answers)


def commitment_depth(depths: Sequence[int], match: Sequence[bool]) -> int | None:
    """Smallest depth from which the answer matches the final answer at every deeper
    measured depth (stable commitment, not first agreement).

    A profile that matches early, diverges, then re-matches commits at the LATER depth:
    the answer is only stably the final answer once it never flips back. This is the
    shallowest depth of the maximal all-matching suffix (in depth order). Returns
    ``None`` when the deepest measured depth does not match, i.e. never stably committed.
    """
    order = sorted(range(len(depths)), key=lambda i: depths[i])
    committed: int | None = None
    for i in reversed(order):
        if not match[i]:
            break
        committed = depths[i]
    return committed


def curve_area(depths: Sequence[int], match: Sequence[bool], max_depth: int) -> float:
    """Normalized area under the step-function match profile, in ``[0, 1]``.

    The profile is a right-continuous step function of depth: at each measured depth the
    match value holds until the next measured depth, and the last measured value holds to
    ``max_depth``. The area is that integral divided by ``max_depth``. Depth 0 is always
    on the grid (fraction 0.0), so the domain ``[0, max_depth]`` is fully covered; the
    deepest measured point sits at ``max_depth`` and so contributes zero width (it is the
    reference the profile is measured against). ``max_depth <= 0`` (an empty chain) has no
    depth axis to integrate and returns 0.0.
    """
    if max_depth <= 0:
        return 0.0
    order = sorted(range(len(depths)), key=lambda i: depths[i])
    ordered_depths = [depths[i] for i in order]
    ordered_match = [match[i] for i in order]
    covered = 0.0
    for j, start in enumerate(ordered_depths):
        end = ordered_depths[j + 1] if j + 1 < len(ordered_depths) else max_depth
        if ordered_match[j] and end > start:
            covered += end - start
    return covered / max_depth


@dataclass(frozen=True)
class TruncationCurve:
    """One item's truncation dose-response curve and its committed-depth summaries.

    Interpretation of the summaries: ``curve_area`` near 1 with ``commitment_depth`` 0
    means the answer already matched the full-CoT answer with the CoT truncated to
    nothing, so the reasoning text is not load-bearing (the pre-CoT commitment regime,
    where the model had settled the answer before writing any reasoning). A late
    ``commitment_depth`` with a small ``curve_area`` means the answer only converges to
    the final answer as more steps are revealed, so the CoT text carries load.
    """

    depths: tuple[int, ...]
    answers: tuple[str | None, ...]
    final_answer: str | None
    match: tuple[bool, ...]
    commitment_depth: int | None
    curve_area: float


def summarize_curve(
    depths: Sequence[int], answers: Sequence[str | None], final_answer: str | None
) -> TruncationCurve:
    """Assemble a :class:`TruncationCurve` from per-depth answers and the final answer."""
    depths_t = tuple(depths)
    answers_t = tuple(answers)
    match = match_profile(answers_t, final_answer)
    max_depth = max(depths_t) if depths_t else 0
    return TruncationCurve(
        depths=depths_t,
        answers=answers_t,
        final_answer=final_answer,
        match=match,
        commitment_depth=commitment_depth(depths_t, match),
        curve_area=curve_area(depths_t, match, max_depth),
    )


def curve_covariates(curves: Iterable[TruncationCurve]) -> list[dict]:
    """Per-item covariate rows (commitment_depth, curve_area, n_depths) for the model.

    These feed the hierarchical mediation model as item-level covariates, so the pooling
    can account for where each item commits rather than treating a decorative-CoT item
    and a load-bearing-CoT item as exchangeable.
    """
    return [
        {
            "commitment_depth": curve.commitment_depth,
            "curve_area": curve.curve_area,
            "n_depths": len(curve.depths),
        }
        for curve in curves
    ]
