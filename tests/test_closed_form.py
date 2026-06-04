"""Tests for the closed-form probit natural effects.

The payoff test is ``test_closed_form_agrees_with_monte_carlo``: it pins the
hand-derived closed form to the Monte Carlo integrator in ``sensitivity`` across
a spread of parameter values and rho. Because the closed form shares no code
with the Monte Carlo path, agreement is strong evidence that neither has a
mediator/treatment pairing bug. The tolerance tracks Monte Carlo error, so it is
tight rather than forgiving.
"""

from __future__ import annotations

import pytest

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.sensitivity import probit_natural_effects


def test_te_is_exactly_nde_plus_nie() -> None:
    # Arrange / Act
    nde, nie, te = probit_natural_effects_closed_form(
        alpha=0.3, beta=1.0, gamma=0.8, sigma_m=0.5, rho=0.4
    )
    # Assert: the closed form is analytic, so the identity holds to machine eps.
    assert te == pytest.approx(nde + nie, abs=1e-12)


def test_fully_mediated_world_has_zero_nde() -> None:
    # Arrange / Act: alpha = 0 removes the direct path.
    nde, nie, _ = probit_natural_effects_closed_form(
        alpha=0.0, beta=1.2, gamma=0.9, sigma_m=0.5, rho=0.0
    )
    # Assert: NDE is exactly zero analytically; NIE remains clearly positive.
    assert nde == pytest.approx(0.0, abs=1e-12)
    assert nie > 0.05


def test_no_mediation_world_has_zero_nie() -> None:
    # Arrange / Act: gamma = 0 means X does not move M.
    _, nie, _ = probit_natural_effects_closed_form(
        alpha=0.5, beta=1.0, gamma=0.0, sigma_m=0.5, rho=0.0
    )
    # Assert: NIE is exactly zero analytically.
    assert nie == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("alpha", "beta", "gamma", "sigma_m", "rho"),
    [
        (0.3, 1.0, 0.8, 0.5, 0.0),
        (0.3, 1.0, 0.8, 0.5, 0.5),
        (0.0, 1.2, 0.9, 0.5, 0.0),
        (0.4, 0.7, 1.1, 0.6, -0.3),
        (0.2, 1.5, 0.5, 0.4, 0.7),
    ],
)
def test_closed_form_agrees_with_monte_carlo(
    alpha: float, beta: float, gamma: float, sigma_m: float, rho: float
) -> None:
    """Closed form and Monte Carlo integrator agree to Monte Carlo tolerance.

    This is the real correctness net: it would catch a bug in which mediator
    distribution Monte Carlo pairs with which do(X) arm.
    """
    # Arrange
    cf = probit_natural_effects_closed_form(alpha, beta, gamma, sigma_m, rho)

    # Act: a large-but-fast Monte Carlo draw. SE ~ 1/sqrt(n_mc) ~ 1.6e-3 here.
    mc = probit_natural_effects(alpha, beta, gamma, sigma_m, rho, n_mc=400_000, rng_seed=0)

    # Assert: every effect agrees within a few Monte Carlo standard errors.
    for cf_val, mc_val in zip(cf, mc):
        assert cf_val == pytest.approx(mc_val, abs=5e-3)


def test_agreement_tightens_as_monte_carlo_grows() -> None:
    """A heavier Monte Carlo draw moves strictly closer to the closed form,
    confirming the residual is sampling noise and not a systematic gap."""
    # Arrange
    params = (0.3, 1.0, 0.8, 0.5, 0.5)
    cf_nie = probit_natural_effects_closed_form(*params)[1]

    # Act
    coarse = probit_natural_effects(*params, n_mc=20_000, rng_seed=1)[1]
    fine = probit_natural_effects(*params, n_mc=1_000_000, rng_seed=1)[1]

    # Assert
    assert abs(fine - cf_nie) < abs(coarse - cf_nie)


def test_rho_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        probit_natural_effects_closed_form(0.3, 1.0, 0.8, 0.5, rho=0.999)


def test_nonpositive_sigma_raises() -> None:
    with pytest.raises(ValueError):
        probit_natural_effects_closed_form(0.3, 1.0, 0.8, 0.0, rho=0.0)
