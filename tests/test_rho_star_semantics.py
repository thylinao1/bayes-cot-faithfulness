"""What rho* means, pinned so it cannot be quietly reinterpreted.

``breakdown_frontier`` reports ``rho*``: the smallest residual M-Y correlation
that flips the sign of a natural effect. It is easy, and wrong, to read that
number as a severity scale - a bigger ``rho*`` meaning a more faithful model, or
a smaller one meaning more direct bypass. The independent review of 6 September
2026 (section 3.4) showed the counterexample: hold the mediator path fixed and
raise only the direct coefficient ``alpha``, and the mediated share collapses
from 100% to 1.6% while ``rho*`` does not move at all.

The reason is exact. With ``gamma != 0`` the NIE vanishes if and only if the
structural mediator coefficient ``beta(rho)`` does, and the reparameterisation
gives ``beta(rho) = B*sqrt(1-rho^2) - rho/sigma_m`` for ``B`` the rho=0 fitted
coefficient, so

    rho* = |B| * sigma_m / sqrt(1 + B^2 * sigma_m^2)

which contains neither ``alpha`` nor either baseline intercept. ``rho*`` is the
robustness of the *sign of the M-to-Y coefficient*, and nothing else.

This module pins that invariance, checks the intercept-carrying
reparameterisation against the closed form, and pins the UNRESOLVED verdict for
the case where there is no supported effect at ``rho = 0`` to be robust about.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.sensitivity import (
    ConfoundedCoTConfig,
    breakdown_frontier,
    fit_probit_mediation_map,
    simulate_confounded_cot,
)

# The review's setting: beta = 0.8, gamma = sigma_m = 1, true rho = 0.
BETA = 0.8
GAMMA = 1.0
SIGMA_M = 1.0
ANALYTIC_RHO_STAR = BETA * SIGMA_M / math.sqrt(1.0 + (BETA * SIGMA_M) ** 2)  # 0.624695


def _sample(alpha: float, n: int = 20_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the review's process at a given direct-bypass strength."""
    cfg = ConfoundedCoTConfig(
        n_prompts=n, alpha_direct=alpha, beta_mediated=BETA, gamma_xm=GAMMA,
        sigma_m=SIGMA_M, rho_confound=0.0, rng_seed=731,
    )
    return simulate_confounded_cot(cfg)


def test_analytic_rho_star_matches_the_published_value() -> None:
    """The zero crossing of beta(rho) is 0.624695 for this process."""
    assert ANALYTIC_RHO_STAR == pytest.approx(0.624695, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_mediated_share_collapses_while_the_crossing_does_not_move(alpha: float) -> None:
    """Analytic half of the counterexample, on the repository's own closed form.

    The mediated fraction NIE/TE falls with the direct bypass; the beta(rho) zero
    crossing is a function of beta and sigma_m alone and therefore does not.
    """
    # Arrange / Act
    _, nie, te = probit_natural_effects_closed_form(alpha, BETA, GAMMA, SIGMA_M, 0.0)

    # Assert: the share moves a lot, the crossing not at all.
    share = nie / te
    assert 0.0 < share <= 1.0
    if alpha >= 3.0:
        assert share < 0.05
    if alpha == 0.0:
        assert share == pytest.approx(1.0, abs=1e-9)
    assert ANALYTIC_RHO_STAR == pytest.approx(0.624695, abs=1e-6)


def test_rho_star_is_invariant_to_the_direct_bypass() -> None:
    """The estimator's own rho* barely moves between alpha = 0 and alpha = 3.

    This is the assertion that stops rho* being reported as a bypass or severity
    measure. Both fits are on 20,000 rows of the same process with only ``alpha``
    changed; the mediated share is near 100% in the first and near 2% in the
    second, and rho* agrees between them to better than 0.01.
    """
    # Arrange
    x0, m0, y0 = _sample(alpha=0.0)
    x3, m3, y3 = _sample(alpha=3.0)

    # Act
    bf0 = breakdown_frontier(x0, m0, y0, key="nie", n_mc=60_000, rng_seed=0)
    bf3 = breakdown_frontier(x3, m3, y3, key="nie", n_mc=60_000, rng_seed=0)

    # Assert
    assert not bf0.unresolved and not bf3.unresolved
    assert bf0.robustness == pytest.approx(ANALYTIC_RHO_STAR, abs=0.02)
    assert bf3.robustness == pytest.approx(ANALYTIC_RHO_STAR, abs=0.02)
    assert abs(bf0.robustness - bf3.robustness) < 0.01, (
        f"rho* moved with the direct bypass: {bf0.robustness:.4f} vs {bf3.robustness:.4f}"
    )
    # ...while the thing rho* is often mistaken for moved by two orders of magnitude.
    assert bf0.effect_at_zero > 10 * bf3.effect_at_zero


def test_rho_star_matches_the_analytic_crossing_of_the_fitted_beta() -> None:
    """rho* is |B|*sigma_m / sqrt(1 + B^2 sigma_m^2) at the fitted B, by construction."""
    # Arrange
    X, M, Y = _sample(alpha=1.0)
    fit = fit_probit_mediation_map(X, M, Y, rho=0.0)
    predicted = abs(fit.beta) * fit.sigma_m / math.sqrt(1.0 + (fit.beta * fit.sigma_m) ** 2)

    # Act
    bf = breakdown_frontier(X, M, Y, key="nie", n_mc=60_000, rng_seed=0)

    # Assert
    assert bf.robustness == pytest.approx(predicted, abs=0.01)


# --------------------------------------------------------------------------- #
# The rho reparameterisation, with intercepts.
# --------------------------------------------------------------------------- #
REPARAM_CONFIG = ConfoundedCoTConfig(
    n_prompts=8_000, alpha_direct=0.3, beta_mediated=0.8, gamma_xm=1.0,
    sigma_m=1.0, rho_confound=0.3, rng_seed=17, mu_m=4.0, alpha0=-3.0,
)


@pytest.mark.parametrize("rho", [-0.4, 0.3, 0.6])
def test_reparameterisation_with_intercepts_matches_a_direct_refit(rho: float) -> None:
    """The rho=0 fit plus the documented algebra reproduces the fit at any rho.

    Derivation is in the ``sensitivity`` module docstring::

        beta(rho)   = B  * sqrt(1 - rho^2) - rho/sigma_m
        alpha(rho)  = A  * sqrt(1 - rho^2) + (rho/sigma_m) * gamma
        alpha0(rho) = A0 * sqrt(1 - rho^2) + (rho/sigma_m) * mu_m

    with ``(A0, A, B)`` the rho=0 outcome coefficients. The mediator equation is
    a plain regression of M on X, so ``(mu_m, gamma, sigma_m)`` do not move with
    rho; that is asserted here too.
    """
    # Arrange
    X, M, Y = simulate_confounded_cot(REPARAM_CONFIG)
    base = fit_probit_mediation_map(X, M, Y, rho=0.0)
    scale = math.sqrt(1.0 - rho**2)
    k = rho / base.sigma_m
    predicted_alpha = base.alpha * scale + k * base.gamma
    predicted_beta = base.beta * scale - k
    predicted_alpha0 = base.alpha0 * scale + k * base.mu_m

    # Act
    refit = fit_probit_mediation_map(X, M, Y, rho=rho)

    # Assert: the outcome coefficients follow the algebra...
    assert refit.alpha == pytest.approx(predicted_alpha, abs=1e-3)
    assert refit.beta == pytest.approx(predicted_beta, abs=1e-3)
    assert refit.alpha0 == pytest.approx(predicted_alpha0, abs=1e-3)
    # ...and the mediator equation is untouched by rho.
    assert refit.gamma == pytest.approx(base.gamma, abs=1e-3)
    assert refit.mu_m == pytest.approx(base.mu_m, abs=1e-3)
    assert refit.sigma_m == pytest.approx(base.sigma_m, abs=1e-3)


@pytest.mark.parametrize("rho", [-0.4, 0.3, 0.6])
def test_reparameterised_parameters_give_the_closed_form_effects(rho: float) -> None:
    """Effects from the reparameterised coefficients equal effects from the refit.

    The closed form is the independent check: it shares no code with the fitting
    path, so agreement means the intercepts are carried correctly through both
    the algebra and the integration.
    """
    # Arrange
    X, M, Y = simulate_confounded_cot(REPARAM_CONFIG)
    base = fit_probit_mediation_map(X, M, Y, rho=0.0)
    scale = math.sqrt(1.0 - rho**2)
    k = rho / base.sigma_m

    # Act
    from_algebra = probit_natural_effects_closed_form(
        base.alpha * scale + k * base.gamma,
        base.beta * scale - k,
        base.gamma,
        base.sigma_m,
        rho,
        base.mu_m,
        base.alpha0 * scale + k * base.mu_m,
    )
    refit = fit_probit_mediation_map(X, M, Y, rho=rho)
    from_refit = probit_natural_effects_closed_form(
        refit.alpha, refit.beta, refit.gamma, refit.sigma_m, rho, refit.mu_m, refit.alpha0
    )

    # Assert
    for algebra_val, refit_val in zip(from_algebra, from_refit):
        assert algebra_val == pytest.approx(refit_val, abs=1e-3)


def test_total_effect_is_invariant_to_the_assumed_rho() -> None:
    """Only the split between NDE and NIE moves with rho; the total does not.

    A useful sanity property of the reparameterisation: rho reallocates the
    observed association between the direct and mediated paths, so a sensitivity
    curve that moved the total effect would be a bug.
    """
    # Arrange
    X, M, Y = simulate_confounded_cot(REPARAM_CONFIG)

    # Act
    totals = []
    for rho in (-0.4, 0.0, 0.3, 0.6):
        fit = fit_probit_mediation_map(X, M, Y, rho=rho)
        _, _, te = probit_natural_effects_closed_form(
            fit.alpha, fit.beta, fit.gamma, fit.sigma_m, rho, fit.mu_m, fit.alpha0
        )
        totals.append(te)

    # Assert
    assert max(totals) - min(totals) < 1e-3


# --------------------------------------------------------------------------- #
# UNRESOLVED: no supported effect at rho = 0 is not a robust verdict.
# --------------------------------------------------------------------------- #
def test_no_supported_effect_at_zero_returns_unresolved() -> None:
    """A null world must not be rewarded with maximal robustness.

    The old frontier searched for a sign flip, found none on either side because
    there was no sign to flip, and returned ``survives_full_range=True`` with the
    largest robustness in range. On the reviewer's null that is the strongest
    possible verdict attached to the weakest possible evidence. With a practical
    ``min_effect`` the frontier now declines to answer.
    """
    # Arrange: the offset null, which has no causal effect of any kind.
    rng = np.random.default_rng(731)
    n = 10_000
    X = np.repeat([0, 1], n // 2)
    M = rng.poisson(5.0, n).astype(float) + 1
    Y = rng.binomial(1, 0.8, n)

    # Act
    bf = breakdown_frontier(X, M, Y, key="nie", n_mc=60_000, rng_seed=0, min_effect=0.01)

    # Assert
    assert bf.unresolved is True
    assert bf.rho_star_pos is None and bf.rho_star_neg is None
    assert math.isnan(bf.robustness)
    assert bf.survives_full_range is False
    assert abs(bf.effect_at_zero) <= 0.01


def test_min_effect_defaults_to_zero_so_existing_verdicts_are_unchanged() -> None:
    """A clearly supported effect is resolved, and the default threshold is 0.0."""
    # Arrange
    X, M, Y = _sample(alpha=0.0)

    # Act
    bf = breakdown_frontier(X, M, Y, key="nie", n_mc=60_000, rng_seed=0)

    # Assert
    assert bf.unresolved is False
    assert bf.effect_at_zero > 0.1
    assert bf.rho_star_pos is not None


def test_negative_min_effect_is_rejected() -> None:
    X, M, Y = _sample(alpha=0.0, n=400)
    with pytest.raises(ValueError):
        breakdown_frontier(X, M, Y, min_effect=-0.1)


def test_docstring_states_what_rho_star_is_not() -> None:
    """A tripwire on the semantics, not on prose style.

    The claim that rho* is not a bypass measure and that an unsupported effect is
    UNRESOLVED is the whole point of this module; if someone deletes it from the
    public docstring the tests should notice.
    """
    doc = breakdown_frontier.__doc__ or ""
    assert "not a measure of direct bypass" in doc
    assert "unresolved" in doc.lower()
    assert "sign of the M-to-Y coefficient" in doc
