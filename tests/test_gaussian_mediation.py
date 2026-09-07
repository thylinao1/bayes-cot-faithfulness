"""Gates for the continuous-outcome (Gaussian) mediation path.

Three things are pinned here, in the order the outcome-scale lane was asked for
them:

(i)   the estimator recovers a KNOWN NDE and NIE on synthetic linear data, on a
      world whose clean arm carries real outcome variance;
(ii)  it passes the reviewer's count-offset and Gaussian-offset nulls, the same
      two designs ``tests/test_offset_null.py`` runs against the probit path,
      with the same seed and the same draw order;
(iii) the numerical fit and the closed form agree, and the vectorised rho curve
      agrees with a refit at every grid point.

The probit path is untouched by all of this and stays the estimator of record
for the frozen ``binary_follow`` outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.gaussian_mediation import (
    fit_gaussian_mediation_closed_form,
    fit_gaussian_mediation_map,
    gaussian_effects_curve,
    gaussian_natural_effects,
    gaussian_rho_star_point,
    gaussian_sensitivity_sweep,
    standardised_effects,
)

# The reviewer's null: seed 731, n = 10,000, balanced arms (test_offset_null.py).
NULL_SEED = 731
NULL_N = 10_000

# A recovery world with a real effect on both paths.
TRUE_ALPHA = 0.40
TRUE_BETA = 0.70
TRUE_GAMMA = 0.90
TRUE_SIGMA_M = 1.0
TRUE_SIGMA_Y = 1.0
TRUE_MU_M = 6.0
TRUE_ALPHA0 = -2.0
RECOVER_N = 8_000
RECOVER_SEED = 20260907


def _recovery_design(rho: float = 0.0, n: int = RECOVER_N, seed: int = RECOVER_SEED):
    """Linear-linear data with a known truth and a clean arm that VARIES in Y."""
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, 0.5, size=n).astype(float)
    cov = np.array(
        [
            [TRUE_SIGMA_M**2, rho * TRUE_SIGMA_M * TRUE_SIGMA_Y],
            [rho * TRUE_SIGMA_M * TRUE_SIGMA_Y, TRUE_SIGMA_Y**2],
        ]
    )
    eps = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    m = TRUE_MU_M + TRUE_GAMMA * x + eps[:, 0]
    y = TRUE_ALPHA0 + TRUE_ALPHA * x + TRUE_BETA * m + eps[:, 1]
    return x, m, y


def _null_design(kind: str):
    """The reviewer's two nulls, seed and draw order preserved.

    Identical to ``tests/test_offset_null.py::_null_design``: a mediator with a
    real baseline level and no dependence on X, and an outcome independent of
    both. Every true effect is exactly zero. ``Y`` stays the reviewer's 0.8 coin,
    read here as a continuous outcome (a linear probability model), so this is
    the SAME data the probit gate runs on and not a friendlier redraw.
    """
    rng = np.random.default_rng(NULL_SEED)
    x = np.repeat([0, 1], NULL_N // 2)
    if kind == "poisson":
        m = rng.poisson(5.0, NULL_N).astype(float) + 1
    elif kind == "gaussian":
        m = 6 + rng.normal(0, np.sqrt(5), NULL_N)
    else:  # pragma: no cover - guard against a typo in a parametrize list
        raise ValueError(f"unknown null design {kind!r}")
    y = rng.binomial(1, 0.8, NULL_N).astype(float)
    return x.astype(float), m, y


# --------------------------------------------------------------------------- #
# (i) recovery of a known NDE and NIE
# --------------------------------------------------------------------------- #
def test_recovers_known_effects_on_linear_data() -> None:
    """Known truth in, known truth out, with the clean arm carrying real variance."""
    # Arrange
    x, m, y = _recovery_design()
    true_nde, true_nie, true_te = gaussian_natural_effects(TRUE_ALPHA, TRUE_BETA, TRUE_GAMMA)
    assert np.var(y[x == 0]) > 0.5, "this world must have a clean arm that varies"

    # Act
    fit = fit_gaussian_mediation_map(x, m, y, rho=0.0)
    nde, nie, te = gaussian_natural_effects(fit.alpha, fit.beta, fit.gamma)

    # Assert
    assert fit.converged
    assert nde == pytest.approx(true_nde, abs=0.05)
    assert nie == pytest.approx(true_nie, abs=0.05)
    assert te == pytest.approx(true_te, abs=0.05)
    assert fit.mu_m == pytest.approx(TRUE_MU_M, abs=0.05)
    assert fit.sigma_y == pytest.approx(TRUE_SIGMA_Y, rel=0.05)


def test_te_equals_the_randomized_arm_difference_at_every_rho() -> None:
    """On this scale the TE is rho-invariant and IS the randomized contrast.

    That is the property the frozen binary outcome does not have: there the
    model-implied TE only approximates the arm difference, and the NDE/NIE split
    is the part that has to be extrapolated.
    """
    # Arrange
    x, m, y = _recovery_design()
    arm_difference = float(y[x == 1].mean() - y[x == 0].mean())

    # Act
    fit0 = fit_gaussian_mediation_closed_form(x, m, y, rho=0.0)
    grid = np.array([-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9])
    _, _, te_curve = gaussian_effects_curve(fit0, grid)

    # Assert
    assert te_curve.max() - te_curve.min() < 1e-9
    assert te_curve[0] == pytest.approx(arm_difference, abs=1e-6)


def test_recovers_the_truth_at_the_true_rho_when_confounded() -> None:
    """A confounded world reads correctly once rho is set to its true value."""
    # Arrange
    rho_true = 0.5
    x, m, y = _recovery_design(rho=rho_true)

    # Act
    naive = fit_gaussian_mediation_closed_form(x, m, y, rho=0.0)
    corrected = fit_gaussian_mediation_closed_form(x, m, y, rho=rho_true)

    # Assert
    assert corrected.beta == pytest.approx(TRUE_BETA, abs=0.05)
    assert corrected.alpha == pytest.approx(TRUE_ALPHA, abs=0.05)
    assert abs(naive.beta - TRUE_BETA) > 0.2, "the rho=0 fit must be visibly biased here"


# --------------------------------------------------------------------------- #
# (ii) the offset nulls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["poisson", "gaussian"])
def test_offset_null_reports_no_mediation(kind: str) -> None:
    """No causal effect in, no mediated effect out, on the reviewer's own nulls."""
    # Arrange
    x, m, y = _null_design(kind)
    arm_difference = float(y[x == 1].mean() - y[x == 0].mean())

    # Act
    fit = fit_gaussian_mediation_map(x, m, y, rho=0.0)
    nde, nie, te = gaussian_natural_effects(fit.alpha, fit.beta, fit.gamma)

    # Assert
    assert fit.converged
    assert abs(nie) < 0.01, f"spurious mediated effect {nie:+.4f} on a null world"
    assert abs(nde) < 0.05
    assert te == pytest.approx(arm_difference, abs=1e-6)


@pytest.mark.parametrize("kind", ["poisson", "gaussian"])
def test_offset_null_recovers_the_mediator_baseline(kind: str) -> None:
    """The mediator baseline is the control-arm mean, not zero."""
    # Arrange
    x, m, y = _null_design(kind)

    # Act
    fit = fit_gaussian_mediation_map(x, m, y, rho=0.0)

    # Assert
    assert fit.mu_m == pytest.approx(float(m[x == 0].mean()), abs=0.05)
    assert fit.gamma == pytest.approx(float(m[x == 1].mean() - m[x == 0].mean()), abs=0.05)
    assert abs(fit.gamma) < 0.2


@pytest.mark.parametrize("kind", ["poisson", "gaussian"])
def test_intercept_free_fit_invents_mediation_on_the_same_null(kind: str) -> None:
    """The documented defect reproduces on this scale too, so it stays legible.

    Not a gate on the estimator of record: it is the reason ``intercepts=True``
    is the default here as well as on the probit path.
    """
    # Arrange
    x, m, y = _null_design(kind)

    # Act
    fit = fit_gaussian_mediation_map(x, m, y, rho=0.0, intercepts=False)
    _, nie, _ = gaussian_natural_effects(fit.alpha, fit.beta, fit.gamma)

    # Assert
    assert abs(nie) > 0.1, "the intercept-free specification should misread this null"


# --------------------------------------------------------------------------- #
# (iii) closed form against the optimiser, and the curve against a refit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [-0.6, -0.2, 0.0, 0.3, 0.7])
def test_closed_form_matches_the_numerical_fit(rho: float) -> None:
    """Two estimators, no shared code path, the same answer."""
    # Arrange
    x, m, y = _recovery_design(rho=0.3)

    # Act
    closed = fit_gaussian_mediation_closed_form(x, m, y, rho=rho)
    numeric = fit_gaussian_mediation_map(x, m, y, rho=rho)

    # Assert
    for name in ("alpha", "beta", "gamma", "sigma_m", "mu_m", "alpha0", "sigma_y"):
        assert getattr(numeric, name) == pytest.approx(getattr(closed, name), abs=1e-4), name


def test_effects_curve_matches_a_refit_at_every_grid_point() -> None:
    """The vectorised rho curve is the same object as the refit-per-rho sweep."""
    # Arrange
    x, m, y = _recovery_design(rho=0.2)
    grid = np.round(np.arange(-0.9, 0.91, 0.1), 4)

    # Act
    fit0 = fit_gaussian_mediation_closed_form(x, m, y, rho=0.0)
    nde_c, nie_c, te_c = gaussian_effects_curve(fit0, grid)
    sweep = gaussian_sensitivity_sweep(x, m, y, rho_grid=grid)

    # Assert
    assert max(abs(p.nie - v) for p, v in zip(sweep, nie_c)) < 1e-9
    assert max(abs(p.nde - v) for p, v in zip(sweep, nde_c)) < 1e-9
    assert max(abs(p.te - v) for p, v in zip(sweep, te_c)) < 1e-9


def test_rho_star_point_is_where_the_curve_crosses_zero() -> None:
    """``gaussian_rho_star_point`` names the NIE's own zero crossing."""
    # Arrange
    x, m, y = _recovery_design()
    fit0 = fit_gaussian_mediation_closed_form(x, m, y, rho=0.0)
    rho_star = gaussian_rho_star_point(fit0.beta, fit0.sigma_m, fit0.sigma_y)

    # Act
    _, nie_at_star, _ = gaussian_effects_curve(fit0, np.array([rho_star]))

    # Assert
    assert abs(float(nie_at_star[0])) < 1e-9
    assert 0.0 < rho_star < 0.95


def test_rho_star_point_is_invariant_to_the_direct_path() -> None:
    """Section 8.2's invariance holds on the continuous scale as well.

    The same worlds with the direct coefficient moved from 0 to 3 give the same
    crossing, while the mediated share collapses. That is what makes rho*_point a
    reference value and not a severity scale, on either outcome scale.
    """
    # Arrange
    rng = np.random.default_rng(4242)
    n = 20_000
    shares, crossings = [], []

    # Act
    for alpha in (0.0, 1.0, 3.0):
        x = rng.binomial(1, 0.5, size=n).astype(float)
        m = TRUE_MU_M + TRUE_GAMMA * x + rng.normal(0.0, 1.0, n)
        y = TRUE_ALPHA0 + alpha * x + TRUE_BETA * m + rng.normal(0.0, 1.0, n)
        fit = fit_gaussian_mediation_closed_form(x, m, y, rho=0.0)
        _, nie, te = gaussian_natural_effects(fit.alpha, fit.beta, fit.gamma)
        shares.append(nie / te)
        crossings.append(gaussian_rho_star_point(fit.beta, fit.sigma_m, fit.sigma_y))

    # Assert
    assert max(crossings) - min(crossings) < 0.01
    assert shares[0] > 0.95 and shares[-1] < 0.25


def test_standardised_effects_divide_by_the_clean_arm_spread() -> None:
    """The reporting convenience does exactly what it says and nothing else."""
    # Arrange / Act
    nde, nie, te = standardised_effects(0.4, 0.8, 1.2, clean_arm_sd=2.0)

    # Assert
    assert (nde, nie, te) == (0.2, 0.4, 0.6)
    with pytest.raises(ValueError):
        standardised_effects(0.4, 0.8, 1.2, clean_arm_sd=0.0)


def test_rho_band_and_shape_guards_refuse_rather_than_warn() -> None:
    """The admissible band and the input checks are refusals, matching the probit path."""
    x, m, y = _recovery_design(n=200)
    with pytest.raises(ValueError):
        fit_gaussian_mediation_closed_form(x, m, y, rho=0.99)
    with pytest.raises(ValueError):
        fit_gaussian_mediation_map(x, m, y, rho=-0.99)
    with pytest.raises(ValueError):
        fit_gaussian_mediation_closed_form(x[:-1], m, y, rho=0.0)
    with pytest.raises(ValueError):
        fit_gaussian_mediation_closed_form(np.full_like(x, 2.0), m, y, rho=0.0)
