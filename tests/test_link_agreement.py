"""The link audit of 2026-09-07: every estimator path in this package is probit.

Until this commit ``mediation.fit_mediation_model`` used ``pm.Bernoulli(logit_p=...)``
while ``sensitivity.fit_probit_mediation_map``, ``sensitivity.probit_natural_effects``
and ``closed_form.probit_natural_effects_closed_form`` were probit, and the CPU
mechanism battery compared the posterior against probit truths. These tests pin the
repair: the posterior now lands on the closed form, the trace records its own link,
and the converter refuses to guess when it does not.

The tolerance below is stated once and defended: on 20,000 rows the posterior
standard deviation of the NIE is about 0.004, so 0.02 is roughly five posterior
standard deviations, wide enough that ordinary Monte Carlo error cannot trip it and
narrow enough to catch a link or an intercept dropped in the conversion (the second
test measures what that failure actually costs, so the gate is proven able to fail
rather than asserted to be strict).
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.effects import posterior_natural_effects
from bayes_cot_faithfulness.mediation import (
    LINK_ATTR,
    extract_intercept_samples,
    extract_parameter_samples,
    fit_mediation_model,
    natural_effects_from_trace,
    trace_link,
)
from bayes_cot_faithfulness.sensitivity import probit_natural_effects

# The probit world the fit must recover: a mediator with a baseline near six that
# drives the answer, which is the shape of the battery's rationalization family.
TRUE = {"alpha": 0.0, "beta": 1.0, "gamma": 1.2, "sigma_m": 1.0, "mu_m": 6.0}
TRUE_ALPHA0 = 0.2 - TRUE["beta"] * TRUE["mu_m"]
AGREEMENT_TOLERANCE = 0.02
LARGE_N = 20_000


def _probit_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One seeded draw from the probit structural model above."""
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, 0.5, n)
    m = TRUE["mu_m"] + TRUE["gamma"] * x + rng.normal(0.0, TRUE["sigma_m"], n)
    latent = TRUE_ALPHA0 + TRUE["alpha"] * x + TRUE["beta"] * m + rng.normal(0.0, 1.0, n)
    return x, m, (latent > 0.0).astype(int)


@pytest.fixture(scope="module")
def large_fit():
    """One posterior on a large dataset, reused by the agreement tests."""
    x, m, y = _probit_dataset(LARGE_N, seed=731)
    trace = fit_mediation_model(
        x, m, y, n_samples=500, n_tune=500, n_chains=2, random_seed=0, progressbar=False,
    )
    return trace


def test_posterior_natural_effects_agree_with_the_closed_form(large_fit) -> None:
    """The PyMC posterior mean lands on the exact closed form of the same model."""
    # Arrange
    truth = probit_natural_effects_closed_form(
        TRUE["alpha"], TRUE["beta"], TRUE["gamma"], TRUE["sigma_m"], 0.0,
        TRUE["mu_m"], TRUE_ALPHA0,
    )

    # Act
    eff = natural_effects_from_trace(large_fit)

    # Assert
    got = (eff.nde_mean, eff.nie_mean, eff.te_mean)
    for name, g, t in zip(("NDE", "NIE", "TE"), got, truth):
        assert abs(g - t) < AGREEMENT_TOLERANCE, (
            f"{name} posterior mean {g:+.4f} against closed form {t:+.4f} "
            f"(tolerance {AGREEMENT_TOLERANCE})"
        )


def test_the_gate_fails_when_the_conversion_drops_the_mediator_baseline(large_fit) -> None:
    """Proof that the agreement gate has teeth: one wrong conversion, far outside it.

    Same draws, same truth, the mediator baseline dropped from the conversion, which
    is the pre-2026-09-07 defect this converter was repaired for. The mediated effect
    then lands near zero, because the effects enter through the offset
    ``alpha0 + beta*mu_m`` and dropping ``mu_m`` leaves a latent index near minus six.

    Dropping *both* intercepts is deliberately not the tripwire: in this world they
    nearly cancel (``alpha0 + beta*mu_m = 0.2`` by construction, so a fit that assumes
    both are zero is only 0.016 off on the NIE here). A tripwire that a coincidence of
    the test's own generator can satisfy is not a tripwire.
    """
    # Arrange
    alpha, beta, gamma, sigma_m = extract_parameter_samples(large_fit)
    _, alpha0 = extract_intercept_samples(large_fit)
    truth_nie = probit_natural_effects_closed_form(
        TRUE["alpha"], TRUE["beta"], TRUE["gamma"], TRUE["sigma_m"], 0.0,
        TRUE["mu_m"], TRUE_ALPHA0,
    )[1]

    # Act
    wrong = posterior_natural_effects(
        alpha, beta, gamma, sigma_m, alpha0_samples=alpha0, link="probit"
    )

    # Assert
    assert abs(wrong.nie_mean - truth_nie) > 10 * AGREEMENT_TOLERANCE


def test_the_two_links_are_not_interchangeable_on_the_same_coefficients() -> None:
    """The audit's finding, as arithmetic: the same draws mean different effects.

    A logistic index of alpha0 + beta*M and a probit index of the same numbers are
    two different models, so pushing draws from one through the other's natural-effect
    formula is a mistake with a size, not a matter of taste.
    """
    # Arrange
    draws = {k: np.full(400, v) for k, v in TRUE.items() if k != "mu_m"}
    mu_m = np.full(400, TRUE["mu_m"])
    alpha0 = np.full(400, TRUE_ALPHA0)

    # Act
    as_probit = posterior_natural_effects(
        draws["alpha"], draws["beta"], draws["gamma"], draws["sigma_m"],
        mu_m_samples=mu_m, alpha0_samples=alpha0, link="probit",
    )
    as_logit = posterior_natural_effects(
        draws["alpha"], draws["beta"], draws["gamma"], draws["sigma_m"],
        mu_m_samples=mu_m, alpha0_samples=alpha0, n_mc_per_draw=4_000, rng_seed=0,
        link="logit",
    )

    # Assert
    assert abs(as_probit.nie_mean - as_logit.nie_mean) > 0.02


def test_probit_conversion_matches_the_monte_carlo_integrator() -> None:
    """The closed-form branch agrees with the package's Monte Carlo probit integrator."""
    # Arrange
    n = 300
    alpha = np.full(n, 0.3)
    beta = np.full(n, 0.8)
    gamma = np.full(n, 1.0)
    sigma_m = np.full(n, 1.0)
    mu_m = np.full(n, 6.0)
    alpha0 = np.full(n, -4.5)

    # Act
    closed = posterior_natural_effects(
        alpha, beta, gamma, sigma_m, mu_m_samples=mu_m, alpha0_samples=alpha0, link="probit",
    )
    mc = probit_natural_effects(0.3, 0.8, 1.0, 1.0, 0.0, n_mc=400_000, rng_seed=0,
                                mu_m=6.0, alpha0=-4.5)

    # Assert
    assert closed.nde_mean == pytest.approx(mc[0], abs=0.005)
    assert closed.nie_mean == pytest.approx(mc[1], abs=0.005)


def test_the_trace_records_its_link_and_the_converter_refuses_to_guess() -> None:
    """A converter that cannot read the link raises instead of assuming one."""
    # Arrange
    x, m, y = _probit_dataset(400, seed=11)
    trace = fit_mediation_model(
        x, m, y, n_samples=150, n_tune=150, n_chains=1, random_seed=0, progressbar=False,
        link="logit",
    )

    # Act / Assert
    assert trace_link(trace) == "logit"
    trace.attrs.pop(LINK_ATTR, None)
    trace.posterior.attrs.pop(LINK_ATTR, None)
    with pytest.raises(ValueError, match="outcome_link"):
        natural_effects_from_trace(trace)
    # ... and it still converts when the caller says which link it means.
    assert natural_effects_from_trace(trace, link="logit").nie_mean != 0.0


def test_unknown_links_are_rejected_at_both_ends() -> None:
    x, m, y = _probit_dataset(50, seed=3)
    with pytest.raises(ValueError, match="link must be"):
        fit_mediation_model(x, m, y, link="cloglog")
    with pytest.raises(ValueError, match="link must be"):
        posterior_natural_effects(
            np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3), link="cloglog"
        )


def test_intercept_extraction_survives_the_centred_parameterisation() -> None:
    """``alpha0`` reaches downstream un-centred, draw by draw."""
    # Arrange
    x, m, y = _probit_dataset(400, seed=5)

    # Act
    trace = fit_mediation_model(
        x, m, y, n_samples=150, n_tune=150, n_chains=1, random_seed=0, progressbar=False,
    )
    _, beta, _, _ = extract_parameter_samples(trace)
    mu_m, alpha0 = extract_intercept_samples(trace)
    centred = trace.posterior["alpha0_centered"].values.flatten()

    # Assert
    np.testing.assert_allclose(alpha0, centred - beta * float(np.mean(m)), rtol=1e-10)
    assert abs(float(np.mean(mu_m)) - float(np.mean(m[x == 0]))) < 0.5
