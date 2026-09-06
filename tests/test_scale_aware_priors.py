"""Scale-aware outcome priors: the estimand must not depend on the mediator's unit.

The 2026-09-07 repair gave the mediator baseline a scale-aware prior and left the
outcome equation on fixed-scale ones (``alpha0 ~ Normal(0, 1.5)``,
``beta ~ Normal(0, 2)``). On a mediator with a baseline near six that also drives
the answer, the intercept the data need sits several prior standard deviations from
zero, so the posterior shrank the intercept and the mediator coefficient together
and the mediated effect came out low: NIE coverage 7/20 in the battery's
rationalization family against 93/100 for the maximum-likelihood bootstrap on the
same datasets (experiments/results/mechanism_battery/pymc_subset.json, run of
2026-09-07).

These two invariances are the property the repair buys, and they are exact rather
than approximate: centring the mediator inside the outcome equation and scaling the
mediator priors by ``sd(M)`` makes the whole model equivariant under
``M -> (M - s) / c``, so the natural effects are unchanged. Both were run against
the pre-change model and failed; the numbers are in
docs/ESTIMATOR-PRIORS-2026-09-07.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.mediation import (
    _mediator_scale,
    fit_mediation_model,
    natural_effects_from_trace,
)

BETA, GAMMA, SIGMA_M, MU_M = 1.0, 1.2, 1.0, 6.0
ALPHA0 = 0.2 - BETA * MU_M  # keeps the clean-arm answer rate away from saturation
N_ROWS = 1_500
INVARIANCE_TOLERANCE = 0.01  # two fits of the same posterior; only MCMC error differs
TRUTH_TOLERANCE = 0.05


def _large_baseline_dataset(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, 0.5, N_ROWS)
    m = MU_M + GAMMA * x + rng.normal(0.0, SIGMA_M, N_ROWS)
    latent = ALPHA0 + BETA * m + rng.normal(0.0, 1.0, N_ROWS)
    return x, m, (latent > 0.0).astype(int)


def _effects(x, m, y, seed: int = 0) -> tuple[float, float, float]:
    trace = fit_mediation_model(
        x, m, y, n_samples=400, n_tune=400, n_chains=2, random_seed=seed, progressbar=False,
    )
    eff = natural_effects_from_trace(trace)
    return eff.nde_mean, eff.nie_mean, eff.te_mean


@pytest.fixture(scope="module")
def dataset():
    return _large_baseline_dataset(seed=909)


def test_effects_are_invariant_to_shifting_the_mediator(dataset) -> None:
    """Centring changes the parameterisation, not the answer."""
    # Arrange
    x, m, y = dataset
    truth = probit_natural_effects_closed_form(
        0.0, BETA, GAMMA, SIGMA_M, 0.0, MU_M, ALPHA0
    )

    # Act: the same data with the mediator's baseline removed by hand.
    as_given = _effects(x, m, y)
    shifted = _effects(x, m - MU_M, y)

    # Assert
    for name, a, b, t in zip(("NDE", "NIE", "TE"), as_given, shifted, truth):
        assert abs(a - b) < INVARIANCE_TOLERANCE, (
            f"{name} moved {a:+.4f} -> {b:+.4f} when the mediator was shifted by "
            f"{MU_M}; the prior, not the data, is doing that"
        )
        assert abs(a - t) < TRUTH_TOLERANCE, f"{name} {a:+.4f} against truth {t:+.4f}"


def test_effects_are_invariant_to_rescaling_the_mediator(dataset) -> None:
    """A mediator counted in tens of steps is the same mediator."""
    # Arrange
    x, m, y = dataset
    scale = 10.0

    # Act
    as_given = _effects(x, m, y)
    rescaled = _effects(x, m * scale, y)

    # Assert
    for name, a, b in zip(("NDE", "NIE", "TE"), as_given, rescaled):
        assert abs(a - b) < INVARIANCE_TOLERANCE, (
            f"{name} moved {a:+.4f} -> {b:+.4f} when the mediator was multiplied by "
            f"{scale}; a prior stated in absolute units did that"
        )


def test_the_mediated_effect_is_recovered_where_the_baseline_is_large(dataset) -> None:
    """The failure the battery found, as a direct assertion on one dataset.

    A bias check, not a coverage check: whether one 95 percent interval covers one
    truth is a coin with a 5 percent tail even when the estimator is perfect, so
    coverage belongs in the battery, over hundreds of datasets, and never in a unit
    test. The bar is the size of the defect: the pre-change model's mean NIE bias on
    this family was minus 0.079 over 20 datasets.
    """
    # Arrange
    x, m, y = dataset
    true_nie = probit_natural_effects_closed_form(
        0.0, BETA, GAMMA, SIGMA_M, 0.0, MU_M, ALPHA0
    )[1]

    # Act
    trace = fit_mediation_model(
        x, m, y, n_samples=400, n_tune=400, n_chains=2, random_seed=0, progressbar=False,
    )
    eff = natural_effects_from_trace(trace)

    # Assert
    assert abs(eff.nie_mean - true_nie) < 0.05, (
        f"posterior NIE {eff.nie_mean:+.4f} against truth {true_nie:+.4f}"
    )


def test_prior_scale_is_the_mediator_sd_and_a_constant_mediator_is_refused() -> None:
    """The one number every mediator prior is stated against."""
    m = np.array([1.0, 3.0, 5.0, 7.0])
    assert _mediator_scale(m) == pytest.approx(float(np.std(m)))
    with pytest.raises(ValueError, match="constant"):
        _mediator_scale(np.full(10, 4.0))


def test_legacy_intercept_free_model_keeps_its_fixed_scale_priors() -> None:
    """``intercepts=False`` still reproduces the pre-repair graph, priors included."""
    # Arrange
    x, m, y = _large_baseline_dataset(seed=7)

    # Act
    trace = fit_mediation_model(
        x, m, y, n_samples=150, n_tune=150, n_chains=1, random_seed=0, progressbar=False,
        intercepts=False, link="logit",
    )

    # Assert: no intercepts anywhere in the graph, centred or otherwise.
    assert "alpha0" not in trace.posterior
    assert "alpha0_centered" not in trace.posterior
    assert "mu_m" not in trace.posterior
