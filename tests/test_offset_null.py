"""Offset-null gate for the probit mediation estimator.

This module is the standing tripwire for the defect the independent review of
6 September 2026 found in the public estimator (its section 3.3). The test is
brutally simple: build a world with **no causal effect at all**, hand it to the
estimator, and demand that it says so.

    X   balanced treatment
    M   a mediator with a real baseline level, independent of X
    Y   an independent coin, independent of X and of M

The population NDE, NIE and TE are all exactly zero. The pre-2026-09-07
estimator forced ``E[M | X=0] = 0`` and a clean-arm outcome baseline of
``Phi(0) = 0.5``, so it read the control-arm *level* of the mediator as a
treatment shift and reported a mediated effect of about 0.21 on this data. The
repaired estimator (``intercepts=True``, now the default) returns essentially
zero and a total effect that tracks the observed arm difference.

Both behaviours are asserted here on purpose. The repaired one is the gate; the
old one is documented so the defect stays legible and so every historical number
computed under ``intercepts=False`` remains reproducible from this repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.sensitivity import (
    ConfoundedCoTConfig,
    fit_probit_mediation_map,
    natural_effects_from_fit,
    simulate_confounded_cot,
)

# The reviewer's exact null: seed 731, n = 10,000, balanced arms.
NULL_SEED = 731
NULL_N = 10_000
NULL_MC = 400_000


def _null_design(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The reviewer's two nulls, draw order and seed preserved.

    ``"poisson"`` gives a positive count mediator (a stand-in for a CoT step
    count); ``"gaussian"`` gives a Gaussian mediator with the same baseline
    offset, which answers the objection that the count failure was only about
    using a Gaussian likelihood on counts. In both cases M is independent of X
    and Y is an independent coin, so every true effect is zero.
    """
    rng = np.random.default_rng(NULL_SEED)
    X = np.repeat([0, 1], NULL_N // 2)
    if kind == "poisson":
        M = rng.poisson(5.0, NULL_N).astype(float) + 1
    elif kind == "gaussian":
        M = 6 + rng.normal(0, np.sqrt(5), NULL_N)
    else:  # pragma: no cover - guard against a typo in a parametrize list
        raise ValueError(f"unknown null design {kind!r}")
    Y = rng.binomial(1, 0.8, NULL_N)
    return X, M, Y


def _observed_arm_difference(X: np.ndarray, Y: np.ndarray) -> float:
    """The randomized contrast the model-implied total effect must track."""
    return float(Y[X == 1].mean() - Y[X == 0].mean())


# --------------------------------------------------------------------------- #
# The gate: with intercepts, a null world reads as null.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["poisson", "gaussian"])
def test_offset_null_reports_no_mediation_with_intercepts(kind: str) -> None:
    """No causal effect in, no mediated effect out."""
    # Arrange
    X, M, Y = _null_design(kind)

    # Act
    fit = fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=True)
    nde, nie, te = natural_effects_from_fit(fit, 0.0, n_mc=NULL_MC, rng_seed=0)

    # Assert: the mediated path is essentially zero, and the model-implied total
    # effect matches the randomized arm difference rather than inventing one.
    assert abs(nie) < 0.01, f"spurious mediated effect {nie:+.4f} on a null world"
    assert abs(te - _observed_arm_difference(X, Y)) < 0.02
    assert abs(nde) < 0.05
    assert fit.converged


@pytest.mark.parametrize("kind", ["poisson", "gaussian"])
def test_offset_null_recovers_the_mediator_baseline(kind: str) -> None:
    """The estimated mediator baseline is the control-arm mean, not zero.

    This is the mechanism behind the gate above: the old model had nowhere to
    put ``E[M | X=0] ~ 6``, so it pushed the whole level into ``gamma``.
    """
    # Arrange
    X, M, Y = _null_design(kind)

    # Act
    fit = fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=True)

    # Assert
    assert fit.mu_m == pytest.approx(float(M[X == 0].mean()), abs=0.05)
    observed_shift = float(M[X == 1].mean() - M[X == 0].mean())
    assert fit.gamma == pytest.approx(observed_shift, abs=0.05)
    assert abs(fit.gamma) < 0.2, "M is independent of X; gamma must be ~0"


# --------------------------------------------------------------------------- #
# The documented defect: intercepts=False reproduces the old, wrong answer.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "legacy_nie", "legacy_te"),
    [("poisson", 0.21308, 0.26968), ("gaussian", 0.22130, 0.26777)],
)
def test_legacy_intercept_free_fit_invents_mediation(
    kind: str, legacy_nie: float, legacy_te: float
) -> None:
    """The pre-repair specification reports ~0.21 mediated effect on a null world.

    Kept as an assertion, not a comment, so the size of the defect is a fact this
    repository can be queried for. The values are the ones the independent review
    published on 6 September 2026.
    """
    # Arrange
    X, M, Y = _null_design(kind)

    # Act
    fit = fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=False)
    nde, nie, te = probit_natural_effects_closed_form(
        fit.alpha, fit.beta, fit.gamma, fit.sigma_m, 0.0
    )

    # Assert: the old answer, reproduced exactly, and nowhere near the truth.
    assert fit.mu_m == 0.0 and fit.alpha0 == 0.0
    assert nie == pytest.approx(legacy_nie, abs=1e-4)
    assert te == pytest.approx(legacy_te, abs=1e-4)
    assert abs(te - _observed_arm_difference(X, Y)) > 0.25
    # The invented effect concentrates in the mediated path, which is what made
    # it read as evidence of faithful reasoning.
    assert abs(nde) < abs(nie)


# Fingerprints of the pre-2026-09-07 fit, recorded from the code at origin/main
# (commit 5c59eba) before the repair, on the machine that produced the public
# site's numbers (arm64 macOS, Accelerate BLAS). Hex float literals, so the record
# keeps every bit of the mantissa. A different BLAS (CI's x86_64 OpenBLAS) lands
# the optimizer within about 1e-6 relative of these values, so the cross-platform
# check below is a tolerance and the bit-for-bit check is same-platform only.
LEGACY_FIT_HEX = {
    "poisson": (
        "0x1.4a3eecbb22eecp-3",  # alpha
        "0x1.c846608833361p-4",  # beta
        "0x1.838b6a02feb07p+2",  # gamma
        "0x1.31cddf4b86b24p+2",  # sigma_m
    ),
    "gaussian": (
        "0x1.12f8c4eb8e52cp-3",
        "0x1.e5c9747cd12e1p-4",
        "0x1.7d953b06b1d73p+2",
        "0x1.32ac6c0d504d9p+2",
    ),
}
LEGACY_FIT_REL_TOL = 1e-4  # cross-platform: five significant digits


@pytest.mark.parametrize("kind", ["poisson", "gaussian"])
def test_legacy_fit_is_bit_for_bit_reproducible(kind: str) -> None:
    """``intercepts=False`` still returns the pre-repair coefficients.

    Historical numbers on the public site were computed with the old
    specification. Two properties keep them reproducible from this repository:
    the fit is deterministic on a given platform (two runs agree to the last
    bit), and every platform lands within five significant digits of the
    recorded values (the recorded hex is the arm64 macOS run; x86_64 OpenBLAS
    differs at about the sixth digit, which is the optimizer's own tolerance,
    not a change of specification).
    """
    # Arrange
    X, M, Y = _null_design(kind)

    # Act
    fit = fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=False)
    again = fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=False)
    got = (fit.alpha, fit.beta, fit.gamma, fit.sigma_m)
    got_again = (again.alpha, again.beta, again.gamma, again.sigma_m)
    recorded = tuple(float.fromhex(h) for h in LEGACY_FIT_HEX[kind])

    # Assert: same platform, bit for bit
    assert tuple(float.hex(v) for v in got) == tuple(float.hex(v) for v in got_again)
    # Assert: every platform, five significant digits of the recorded run
    for name, value, expected in zip(("alpha", "beta", "gamma", "sigma_m"), got, recorded):
        assert value == pytest.approx(expected, rel=LEGACY_FIT_REL_TOL), name


# --------------------------------------------------------------------------- #
# Recovery on data the model actually describes, still passing after the repair.
# --------------------------------------------------------------------------- #
def test_model_dgp_recovery_still_passes_with_intercepts() -> None:
    """On data generated by the model's own SCM the repaired fit recovers truth.

    Adding two free intercepts to a process whose true intercepts are zero must
    cost accuracy, not correctness: the estimates must still land on the exact
    closed-form effects.
    """
    # Arrange: the project's standard confounded process, zero intercepts.
    cfg = ConfoundedCoTConfig(
        n_prompts=8_000, alpha_direct=0.3, beta_mediated=1.0,
        gamma_xm=0.8, sigma_m=0.5, rho_confound=0.5, rng_seed=7,
    )
    X, M, Y = simulate_confounded_cot(cfg)
    truth = probit_natural_effects_closed_form(
        cfg.alpha_direct, cfg.beta_mediated, cfg.gamma_xm, cfg.sigma_m, cfg.rho_confound
    )

    # Act: fit at the true rho, with intercepts free.
    fit = fit_probit_mediation_map(X, M, Y, rho=cfg.rho_confound, intercepts=True)
    estimated = natural_effects_from_fit(fit, cfg.rho_confound, n_mc=NULL_MC, rng_seed=1)

    # Assert: coefficients and effects both recovered; the free intercepts sit
    # near their true value of zero.
    assert fit.beta == pytest.approx(cfg.beta_mediated, abs=0.12)
    assert fit.gamma == pytest.approx(cfg.gamma_xm, abs=0.05)
    assert fit.mu_m == pytest.approx(0.0, abs=0.05)
    assert fit.alpha0 == pytest.approx(0.0, abs=0.15)
    for est, true in zip(estimated, truth):
        assert est == pytest.approx(true, abs=0.03)


# --------------------------------------------------------------------------- #
# Misspecification: a mediator whose baseline level differs by arm.
# --------------------------------------------------------------------------- #
# A mediator with a real baseline (mu_m = 6) that the cue then shifts (gamma = 1),
# and an outcome baseline that keeps the probit away from saturation. This is the
# shape of a CoT step count: about six steps on a clean prompt, one more when the
# cue is present. The old model has nowhere to put the 6.
MISSPEC_CONFIG = ConfoundedCoTConfig(
    n_prompts=20_000, alpha_direct=0.2, beta_mediated=0.3, gamma_xm=1.0,
    sigma_m=1.0, rho_confound=0.0, rng_seed=99, mu_m=6.0, alpha0=-1.8,
)


def test_arm_specific_mediator_baseline_is_recovered_with_intercepts() -> None:
    """With a baseline plus an arm shift, the repaired estimator gets both right."""
    # Arrange
    cfg = MISSPEC_CONFIG
    X, M, Y = simulate_confounded_cot(cfg)
    truth = probit_natural_effects_closed_form(
        cfg.alpha_direct, cfg.beta_mediated, cfg.gamma_xm, cfg.sigma_m,
        cfg.rho_confound, cfg.mu_m, cfg.alpha0,
    )

    # Act
    fit = fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=True)
    nde, nie, te = natural_effects_from_fit(fit, 0.0, n_mc=NULL_MC, rng_seed=0)

    # Assert: the baseline goes into mu_m and the arm shift into gamma, and the
    # effects land on the analytic truth and on the randomized arm difference.
    assert fit.mu_m == pytest.approx(cfg.mu_m, abs=0.1)
    assert fit.gamma == pytest.approx(cfg.gamma_xm, abs=0.1)
    assert nie == pytest.approx(truth[1], abs=0.02)
    assert nde == pytest.approx(truth[0], abs=0.02)
    assert te == pytest.approx(_observed_arm_difference(X, Y), abs=0.02)


def test_arm_specific_mediator_baseline_breaks_the_legacy_fit() -> None:
    """The same data under ``intercepts=False``: the baseline is read as a shift.

    The legacy fit reports a mediator coefficient ``gamma`` of about 7, which is
    the baseline 6 plus the true shift 1, and splits the total effect the wrong
    way between the direct and mediated paths. It is the same defect as the null,
    now on a world where a real mediated effect exists to be measured.
    """
    # Arrange
    cfg = MISSPEC_CONFIG
    X, M, Y = simulate_confounded_cot(cfg)
    truth_nde, truth_nie, _ = probit_natural_effects_closed_form(
        cfg.alpha_direct, cfg.beta_mediated, cfg.gamma_xm, cfg.sigma_m,
        cfg.rho_confound, cfg.mu_m, cfg.alpha0,
    )

    # Act
    legacy = fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=False)
    nde, nie, _ = probit_natural_effects_closed_form(
        legacy.alpha, legacy.beta, legacy.gamma, legacy.sigma_m, 0.0
    )

    # Assert: gamma absorbs the baseline, and both component effects are wrong by
    # more than the repaired fit's tolerance above.
    assert legacy.gamma == pytest.approx(cfg.mu_m + cfg.gamma_xm, abs=0.2)
    assert abs(nie - truth_nie) > 0.05, "legacy mediated path should be badly wrong"
    assert abs(nde - truth_nde) > 0.05, "legacy direct path should be badly wrong"
