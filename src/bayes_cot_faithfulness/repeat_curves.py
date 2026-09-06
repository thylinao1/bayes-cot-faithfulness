"""Element 8.3 (PF-13): mediator-noise components from repeated truncation curves.

Section 8.3 of ``PREREGISTRATION_jury_and_scale.md`` makes the repeated-curve noise
estimate a REQUIRED Phase 1 deliverable, and names the closed form it feeds:

    lambda = sigma_m^2 / (sigma_m^2 + sigma_u^2)
    beta'  = beta lambda / c
    alpha' = (alpha + beta gamma (1 - lambda)) / c
    c      = sqrt(1 + beta^2 sigma_m^2 (1 - lambda))

sigma_u is the WITHIN-item spread of a mediator value across repeats of the same item
(measurement noise), sigma_m is the BETWEEN-item spread (real signal). The estimate the
document is waiting for has never existed, because both skeleton runs decoded at
temperature 0 with one sample per call, so every item had exactly one curve.

This module is the arithmetic only. The repeats themselves are drawn in
``experiments/08_additive_arms.py``; keeping the estimator here is the same split the
curves and sampling modules use, so the components can be tested against numbers whose
answer is known by hand.

TWO ESTIMATES OF sigma_m ARE REPORTED, and the difference matters:

  - ``sigma_m`` is the across-item standard deviation of the per-item means, which is
    what the brief for this run names. It is BIASED UPWARD, because an item mean over
    r repeats still carries sigma_u^2 / r of measurement noise:
    Var(mean_i) = sigma_m_true^2 + sigma_u^2 / r.
  - ``sigma_m_noise_corrected`` subtracts that term (floored at zero) and is the
    unbiased-in-expectation estimate of the true between-item sd.

``lambda`` is computed from each, and both are printed. Choosing between them is an
analysis ruling and is not made here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _clean(values: Sequence[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def within_item_sd(values: Sequence[float | None], ddof: int = 1) -> float | None:
    """Standard deviation of one item's repeats, divisor r minus ``ddof``.

    None when fewer than ``ddof + 1`` repeats scored: an item with one usable repeat
    has no measurable within-item spread, and calling that 0.0 would drag sigma_u down
    with exactly the items whose generations were too ragged to score.
    """
    xs = _clean(values)
    if len(xs) <= ddof:
        return None
    mean = sum(xs) / len(xs)
    ss = sum((x - mean) ** 2 for x in xs)
    return math.sqrt(ss / (len(xs) - ddof))


def item_mean(values: Sequence[float | None]) -> float | None:
    xs = _clean(values)
    return (sum(xs) / len(xs)) if xs else None


def variance_components(
    per_item: Sequence[Sequence[float | None]], ddof: int = 1
) -> dict:
    """sigma_u, sigma_m and lambda from one value per repeat per item.

    ``per_item[i]`` is item i's repeats of the same scalar (curve area, say). An item
    is USED for sigma_u only when it has at least ``ddof + 1`` scorable repeats, and
    for sigma_m only when it has at least one; both denominators are returned rather
    than assumed equal, and ``n_items_held_out`` counts the items that contributed to
    neither because nothing scored.

    sigma_u is pooled across items on the sums of squares:
    sigma_u^2 = sum_i SS_i / sum_i (r_i - ddof), which is the mean of the per-item
    variances when every item has the same r, and does not over-weight a short item
    when they differ.
    """
    ss_total = 0.0
    df_total = 0
    per_item_sd: list[float] = []
    means: list[float] = []
    n_repeats_used: list[int] = []
    n_no_scorable = 0
    for values in per_item:
        xs = _clean(values)
        if not xs:
            n_no_scorable += 1
            continue
        means.append(sum(xs) / len(xs))
        n_repeats_used.append(len(xs))
        if len(xs) > ddof:
            mean = sum(xs) / len(xs)
            ss_total += sum((x - mean) ** 2 for x in xs)
            df_total += len(xs) - ddof
            per_item_sd.append(math.sqrt(
                sum((x - mean) ** 2 for x in xs) / (len(xs) - ddof)
            ))

    sigma_u = math.sqrt(ss_total / df_total) if df_total > 0 else None
    if len(means) > 1:
        gmean = sum(means) / len(means)
        var_means = sum((m - gmean) ** 2 for m in means) / (len(means) - 1)
        sigma_m = math.sqrt(var_means)
    else:
        var_means = None
        sigma_m = None

    r_mean = (sum(n_repeats_used) / len(n_repeats_used)) if n_repeats_used else None
    corrected_var = None
    sigma_m_corrected = None
    if var_means is not None and sigma_u is not None and r_mean:
        corrected_var = max(0.0, var_means - (sigma_u ** 2) / r_mean)
        sigma_m_corrected = math.sqrt(corrected_var)

    def lam(between_var: float | None) -> float | None:
        if between_var is None or sigma_u is None:
            return None
        denom = between_var + sigma_u ** 2
        return (between_var / denom) if denom > 0 else None

    return {
        "n_items_total": len(per_item),
        "n_items_with_a_scorable_repeat": len(means),
        "n_items_used_for_sigma_u": len(per_item_sd),
        "n_items_held_out_no_scorable_repeat": n_no_scorable,
        "n_items_held_out_single_scorable_repeat": len(means) - len(per_item_sd),
        "ddof": ddof,
        "total_within_item_df": df_total,
        "mean_repeats_per_item": r_mean,
        "sigma_u": sigma_u,
        "sigma_m": sigma_m,
        "var_of_item_means": var_means,
        "sigma_m_noise_corrected": sigma_m_corrected,
        "lambda": lam(var_means),
        "lambda_noise_corrected": lam(corrected_var),
        "mean_within_item_sd": (
            sum(per_item_sd) / len(per_item_sd) if per_item_sd else None
        ),
        "max_within_item_sd": max(per_item_sd) if per_item_sd else None,
        "n_items_with_zero_within_item_sd": sum(1 for s in per_item_sd if s == 0.0),
    }


def identical_repeat_fraction(per_item: Sequence[Sequence[str | None]]) -> dict:
    """How many items produced byte-identical repeats.

    At temperature 0 this is a determinism measurement under the run's own concurrency,
    not a property of the estimator: greedy decoding under a shared batch is not
    batch-invariant on this vLLM build, so repeats of the same prompt can differ. An
    item counts only when every repeat is present; ``n_items_incomplete`` says how many
    were skipped rather than folding them in as agreeing.
    """
    n = 0
    identical = 0
    incomplete = 0
    for reps in per_item:
        values = list(reps)
        if not values or any(v is None for v in values):
            incomplete += 1
            continue
        n += 1
        if all(v == values[0] for v in values):
            identical += 1
    return {
        "n_items_compared": n,
        "n_items_incomplete": incomplete,
        "n_identical": identical,
        "identical_fraction": (identical / n) if n else None,
    }
