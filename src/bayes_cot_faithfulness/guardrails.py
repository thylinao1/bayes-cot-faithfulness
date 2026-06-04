"""Statistical guardrails for the experiment arms: sample ratio, attrition, power.

Two sibling A/B-testing projects surfaced gaps this study shares, and this module
closes them with closed-form, offline checks.

1. **Sample-ratio / attrition.** The real-model control filters items to
   "clean-correct" and then compares a clean arm against a hinted arm. If failed
   generations or parse drops are not balanced across arms, the surviving sample is
   no longer a fair contrast, and the mediation effect is biased the same way a
   sample-ratio mismatch (SRM) biases an A/B lift. ``srm_test`` checks observed arm
   counts against an expected split; ``attrition_balance`` checks that the survival
   (kept) proportion is equal across arms.

2. **Power / minimum detectable effect.** The positive control can report a follow
   rate of 0% on as few as 11 to 22 clean-correct items. A point estimate of 0% is
   not evidence of robustness without an upper bound: ``proportion_ci_upper`` gives
   the exact Clopper-Pearson upper confidence limit, ``rule_of_three_upper`` the 3/n
   shortcut at zero successes, and ``minimum_detectable_rate`` the smallest true
   follow-rate that the given n could detect with the requested power.

Nothing here calls a model or the network. All results are frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

# Chi-square cells below this expected count make the asymptotic approximation
# unreliable; we still compute, but raise a warning flag.
_MIN_EXPECTED_CELL = 5.0
# Grid resolution for the minimum-detectable-rate scan.
_MDR_GRID = 2001


# --------------------------------------------------------------------------- #
# Sample-ratio mismatch
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SrmResult:
    """Outcome of a sample-ratio-mismatch check on observed arm counts."""

    statistic: float
    p_value: float
    dof: int
    expected: tuple[float, ...]
    flagged: bool
    low_expected: bool


def srm_test(
    counts: Sequence[int],
    expected_ratios: Sequence[float] | None = None,
    alpha: float = 1e-3,
) -> SrmResult:
    """Chi-square goodness-of-fit of observed arm counts against an expected split.

    ``expected_ratios`` defaults to an equal split across the arms. The degrees of
    freedom are ``k - 1`` for ``k`` arms (so two arms give dof 1, the classic SRM
    trap, not dof 2). ``flagged`` is ``p < alpha``. If any expected cell falls below
    five, ``low_expected`` is set and the statistic is still returned (the asymptotic
    chi-square is then only approximate).
    """
    obs = np.asarray(counts, dtype=float)
    if obs.ndim != 1 or obs.size < 2:
        raise ValueError("counts must list at least two arm totals.")
    if np.any(obs < 0):
        raise ValueError("counts must be non-negative.")

    total = float(obs.sum())
    if total <= 0:
        raise ValueError("counts must sum to a positive total.")

    if expected_ratios is None:
        weights = np.ones_like(obs)
    else:
        weights = np.asarray(expected_ratios, dtype=float)
        if weights.shape != obs.shape:
            raise ValueError("expected_ratios must match counts in length.")
        if np.any(weights <= 0):
            raise ValueError("expected_ratios must be positive.")
    expected = weights / weights.sum() * total

    statistic, p_value = stats.chisquare(f_obs=obs, f_exp=expected)
    low_expected = bool(np.any(expected < _MIN_EXPECTED_CELL))
    return SrmResult(
        statistic=float(statistic),
        p_value=float(p_value),
        dof=int(obs.size - 1),
        expected=tuple(float(e) for e in expected),
        flagged=bool(p_value < alpha),
        low_expected=low_expected,
    )


# --------------------------------------------------------------------------- #
# Differential attrition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttritionResult:
    """Outcome of an attrition-balance check across arms."""

    statistic: float
    p_value: float
    survival_rates: dict[str, float]
    flagged: bool


def attrition_balance(
    arm_entered: Mapping[str, int],
    arm_survived: Mapping[str, int],
    alpha: float = 1e-3,
) -> AttritionResult:
    """Test whether the survival (kept) proportion is equal across arms.

    Builds a 2-by-k table of ``[survived, dropped]`` per arm and runs a chi-square
    test of independence. ``flagged`` is ``p < alpha`` (an imbalance). When no arm
    dropped anything, the table is degenerate and there is no imbalance to find, so
    the test short-circuits to ``p = 1.0``.
    """
    arms = list(arm_entered)
    if set(arms) != set(arm_survived):
        raise ValueError("arm_entered and arm_survived must share the same arms.")
    if len(arms) < 2:
        raise ValueError("need at least two arms to compare attrition.")

    entered = np.array([float(arm_entered[a]) for a in arms])
    survived = np.array([float(arm_survived[a]) for a in arms])
    if np.any(survived > entered) or np.any(entered < 0) or np.any(survived < 0):
        raise ValueError("survived must be between 0 and entered for every arm.")

    dropped = entered - survived
    survival_rates = {
        a: (float(survived[i] / entered[i]) if entered[i] > 0 else float("nan"))
        for i, a in enumerate(arms)
    }

    # No dropout anywhere -> nothing to test; equal (full) survival by construction.
    if dropped.sum() == 0:
        return AttritionResult(
            statistic=0.0,
            p_value=1.0,
            survival_rates=survival_rates,
            flagged=False,
        )

    table = np.vstack([survived, dropped])
    statistic, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    return AttritionResult(
        statistic=float(statistic),
        p_value=float(p_value),
        survival_rates=survival_rates,
        flagged=bool(p_value < alpha),
    )


# --------------------------------------------------------------------------- #
# Power and confidence bounds for a binomial proportion
# --------------------------------------------------------------------------- #
def proportion_ci_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Clopper-Pearson exact one-sided upper confidence bound for a proportion.

    ``conf`` is the one-sided confidence level, so the bound is the ``conf`` quantile
    of a ``Beta(k + 1, n - k)``. One-sided is the right convention for a one-directional
    question: how large could the true rate be, given ``k`` successes in ``n`` trials?
    For ``k = 0`` it is the largest true rate still consistent with seeing no successes,
    and ``rule_of_three_upper`` is its 3/n rule of thumb. Edge cases: ``k == n`` gives
    1.0 (every trial succeeded), and ``n == 0`` gives 1.0 (no information).
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k < 0:
        raise ValueError("k must be non-negative.")
    if not 0.0 < conf < 1.0:
        raise ValueError("conf must lie strictly between 0 and 1.")
    return float(stats.beta.ppf(conf, k + 1, n - k))


def rule_of_three_upper(n: int) -> float:
    """The 3/n rule: a one-sided 95% upper bound on a rate after zero successes in n.

    This is the standard back-of-envelope approximation to the one-sided 95%
    Clopper-Pearson upper bound at ``k = 0`` (it sits a touch above the exact bound at
    small n and converges as n grows). Returns ``3.0 / n`` for ``n > 0`` and 1.0
    otherwise (no trials carry no information).
    """
    if n <= 0:
        return 1.0
    return 3.0 / n


def minimum_detectable_rate(
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    baseline: float = 0.0,
) -> float:
    """Smallest true follow-rate a one-sided binomial test at ``n`` trials can detect.

    Fixes the rejection region first: the smallest count ``c`` whose upper-tail
    probability under ``H0: rate <= baseline`` is at most ``alpha``. Then scans true
    rates ``p`` upward and returns the smallest ``p`` whose power, ``P(X >= c | p)``,
    reaches ``power``. Returns 1.0 if no rate in ``[0, 1]`` reaches the target (n too
    small). This is the minimum detectable effect that should accompany any reported
    null: a 0% follow rate only rules out unfaithfulness above this rate.
    """
    if n <= 0:
        return 1.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between 0 and 1.")
    if not 0.0 <= baseline < 1.0:
        raise ValueError("baseline must lie in [0, 1).")

    # Critical count: smallest c with P(X >= c | baseline) <= alpha.
    crit = None
    for c in range(0, n + 1):
        if stats.binom.sf(c - 1, n, baseline) <= alpha:
            crit = c
            break
    if crit is None:  # pragma: no cover - sf is monotone, c = n+0 always qualifies
        return 1.0

    grid = np.linspace(0.0, 1.0, _MDR_GRID)
    achieved = stats.binom.sf(crit - 1, n, grid)
    hit = np.flatnonzero(achieved >= power)
    if hit.size == 0:
        return 1.0
    return float(grid[hit[0]])
