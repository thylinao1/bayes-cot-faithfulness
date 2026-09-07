"""Ruling R4: CHAIN-level mediator noise, the gap the continuation-level repeats left.

Amendment A3.5 measured sigma_u by repeating the forced continuation at each truncation
depth with the chain of thought held FIXED, and said in the same breath what that does
not cover: "a mediator whose chain is also redrawn carries at least this much and almost
certainly more". The mediator of section 8.3 IS the chain. Ruling R4 makes the
chain-level estimate a done-when before any column B number ships, and this module is
its arithmetic.

The draw, made in ``experiments/08_additive_arms.py``: r full-chain resamples per
clean-correct item at temperature 0.7, on the clean frame and on the hinted frame, each
with its own seed, and each resampled chain read through the SAME truncation-curve grid
the curves arm uses. So a repeat here differs from a repeat in ``repeat_curves`` in
exactly one place, which is the place that matters: the chain is redrawn instead of held.

The components are the same closed form and the same estimator as the continuation-level
ones, deliberately, so the two lambdas are comparable and the difference between them is
the quantity ruling R4 is about:

    sigma_u^2 = sum_i SS_i / sum_i (r_i - 1)      within item, across resampled chains
    sigma_m^2 = Var(item means)                   between items
    lambda    = sigma_m^2 / (sigma_m^2 + sigma_u^2)

Two things this module refuses to do.

  - It never fills a missing repeat. A resampled chain whose own final answer does not
    parse has no curve to read, so it is counted and dropped, and every denominator is
    printed beside its numerator.
  - It never reports lambda without the identical-chain fraction beside it. At
    temperature 0.7 on a near-deterministic model most chains can come back identical,
    and a lambda of 1.0 computed on identical chains says the sampler did not move, not
    that the mediator is noiseless. That reading is the trap A3.5's temperature-0 rows
    already walked into once, and the fraction is what makes it visible.
"""

from __future__ import annotations

from collections.abc import Sequence

from bayes_cot_faithfulness.repeat_curves import (
    identical_repeat_fraction,
    item_mean,
    variance_components,
    within_item_sd,
)

# The scalars a chain repeat is read on. curve_area is primary (continuous, and the
# per-item covariate the hierarchical model conditions on); commitment_depth is the
# integer secondary and has a smaller denominator wherever a chain never commits.
SCALARS = ("curve_area", "commitment_depth")


def _values(repeats: Sequence[dict], field: str) -> list[float | None]:
    return [rep.get(field) for rep in repeats]


def per_item_rows(per_item_repeats: Sequence[Sequence[dict]]) -> list[dict]:
    """One row per item: its within-item sd and mean on each scalar, with counts.

    sigma_u is defined per item; sigma_m and lambda are not, because both need the
    spread ACROSS items. So this table carries the per-item half of ruling R4's ask and
    the frame-level summary carries the other half, rather than printing a per-item
    "lambda" that would be a ratio of one number to itself.
    """
    rows: list[dict] = []
    for index, repeats in enumerate(per_item_repeats):
        row: dict = {
            "item_index": index,
            "n_repeats": len(repeats),
            "n_repeats_scorable": {},
        }
        for scalar in SCALARS:
            values = _values(repeats, scalar)
            scorable = [v for v in values if v is not None]
            row["n_repeats_scorable"][scalar] = len(scorable)
            row[f"sigma_u_{scalar}"] = within_item_sd(values)
            row[f"mean_{scalar}"] = item_mean(values)
        shas = [rep.get("chain_sha256") for rep in repeats]
        row["identical_chains"] = (
            bool(shas) and all(s is not None for s in shas) and len(set(shas)) == 1
        )
        answers = [rep.get("chain_answer") for rep in repeats]
        row["identical_chain_answers"] = (
            bool(answers) and all(a is not None for a in answers) and len(set(answers)) == 1
        )
        row["n_chains_unparsed"] = sum(1 for a in answers if a is None)
        rows.append(row)
    return rows


def frame_components(per_item_repeats: Sequence[Sequence[dict]]) -> dict:
    """sigma_u, sigma_m and lambda for one frame, on both scalars, with denominators."""
    out: dict = {
        "n_items": len(per_item_repeats),
        "identical_chains": identical_repeat_fraction(
            [_values(reps, "chain_sha256") for reps in per_item_repeats]
        ),
        "identical_chain_answers": identical_repeat_fraction(
            [_values(reps, "chain_answer") for reps in per_item_repeats]
        ),
        "n_chains_drawn": sum(len(reps) for reps in per_item_repeats),
        "n_chains_unparsed": sum(
            1 for reps in per_item_repeats for rep in reps
            if rep.get("chain_answer") is None
        ),
    }
    for scalar in SCALARS:
        out[scalar] = variance_components(
            [_values(reps, scalar) for reps in per_item_repeats]
        )
    return out


def summarize_chain_repeats(blocks: Sequence[dict], frames=("clean", "hinted")) -> dict:
    """The whole arm: per-frame components, the per-item table, and what was drawn.

    ``blocks`` are the per-record ``chain_repeats`` blocks the runner wrote. A record
    with no block is not represented at all rather than entering as an empty item; the
    entered count and the with-repeats count are both printed by the caller.
    """
    out: dict = {
        "r": blocks[0].get("r") if blocks else None,
        "temperature": blocks[0].get("temperature") if blocks else None,
        "frames": {},
        "scope": (
            "Ruling R4. These repeats redraw the CHAIN and then read the truncation "
            "curve on each redraw, so sigma_u here is the noise of the mediator itself "
            "and not only of the curve read. It is the strictly larger experiment "
            "A3.5 named as missing. Construct-level noise in the commitment summary "
            "is measured by no repeat in this design, chain-level or continuation-level."
        ),
    }
    for frame in frames:
        per_item = [
            (block.get("frames", {}).get(frame) or []) for block in blocks
            if block.get("frames", {}).get(frame)
        ]
        summary = frame_components(per_item)
        summary["per_item"] = per_item_rows(per_item)
        out["frames"][frame] = summary
    return out
