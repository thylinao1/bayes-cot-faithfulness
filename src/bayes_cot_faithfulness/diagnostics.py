"""MCMC sampler-health gate for the PyMC fits.

The hierarchical and probit fits set ``target_accept`` but never assert that the
sampler actually converged. A headline faithfulness number read off a bad run (a
high R-hat, a thin effective sample, or divergent transitions) is worse than no
number, because it looks authoritative. This module is the gate: it reads R-hat
and bulk ESS from ``arviz.summary`` and the divergence count from the sampler's
own stats, and either returns a frozen verdict or raises with a clear message.

It does not sample. Pass it the ``InferenceData`` a fit already produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import arviz as az

# Conventional convergence gates. R-hat at or below 1.01 and bulk ESS at or above
# 400 are the common defaults; any divergence is disqualifying for a mediation
# estimate, where a divergent region usually means the funnel was not explored.
DEFAULT_MAX_RHAT = 1.01
DEFAULT_MIN_ESS = 400.0
DEFAULT_MAX_DIVERGENCES = 0


@dataclass(frozen=True)
class SamplerHealth:
    """Convergence verdict for one ``InferenceData`` against fixed thresholds."""

    max_rhat_seen: float
    min_ess_seen: float
    n_divergences: int
    healthy: bool


def _count_divergences(idata: "az.InferenceData") -> int:
    """Total divergent transitions, or 0 when the sampler did not record any."""
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is None or "diverging" not in sample_stats:
        return 0
    return int(sample_stats["diverging"].values.sum())


def assert_sampler_healthy(
    idata: "az.InferenceData",
    *,
    max_rhat: float = DEFAULT_MAX_RHAT,
    min_ess: float = DEFAULT_MIN_ESS,
    max_divergences: int = DEFAULT_MAX_DIVERGENCES,
) -> SamplerHealth:
    """Read R-hat, bulk ESS, and divergences and return a health verdict.

    Despite the name this is non-raising: it returns a :class:`SamplerHealth` whose
    ``healthy`` flag is the conjunction of the three gates. Use
    :func:`raise_if_unhealthy` when an unhealthy fit should stop the run. Missing
    sampler stats are treated as zero divergences rather than crashing, so a
    hand-built ``InferenceData`` still passes through.
    """
    import arviz as az

    summary = az.summary(idata)
    max_rhat_seen = float(summary["r_hat"].max())
    min_ess_seen = float(summary["ess_bulk"].min())
    n_divergences = _count_divergences(idata)

    healthy = (
        max_rhat_seen <= max_rhat
        and min_ess_seen >= min_ess
        and n_divergences <= max_divergences
    )
    return SamplerHealth(
        max_rhat_seen=max_rhat_seen,
        min_ess_seen=min_ess_seen,
        n_divergences=n_divergences,
        healthy=healthy,
    )


def raise_if_unhealthy(
    idata: "az.InferenceData",
    *,
    max_rhat: float = DEFAULT_MAX_RHAT,
    min_ess: float = DEFAULT_MIN_ESS,
    max_divergences: int = DEFAULT_MAX_DIVERGENCES,
) -> SamplerHealth:
    """Like :func:`assert_sampler_healthy`, but raise ``ValueError`` when unhealthy.

    The message names every gate that failed and the value that broke it, so a
    failed run is self-explanatory in a log.
    """
    health = assert_sampler_healthy(
        idata, max_rhat=max_rhat, min_ess=min_ess, max_divergences=max_divergences
    )
    if health.healthy:
        return health

    reasons: list[str] = []
    if health.max_rhat_seen > max_rhat:
        reasons.append(f"max R-hat {health.max_rhat_seen:.3f} > {max_rhat}")
    if health.min_ess_seen < min_ess:
        reasons.append(f"min bulk ESS {health.min_ess_seen:.0f} < {min_ess:.0f}")
    if health.n_divergences > max_divergences:
        reasons.append(f"{health.n_divergences} divergences > {max_divergences}")
    raise ValueError("MCMC sampler unhealthy: " + "; ".join(reasons))
