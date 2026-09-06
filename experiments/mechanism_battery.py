"""CPU mechanism battery for the causal-mediation instrument (Amendment A2, element 11(a)).

Seven generator families, all CPU, all seeded, each with a *known* truth, run before
any organism is trained and re-run whenever the estimator changes. The question the
battery answers is not "does the estimator work" but "on which mechanisms does it
report the right thing, on which does it report a confident wrong thing, and does the
sensitivity summary flag the difference".

The families
------------
1. ``f1_no_cue_effect``      no cue effect, with a nonzero baseline accuracy and a
   nonzero baseline reasoning length. True NDE = NIE = TE = 0. This is the offset null
   of ``tests/test_offset_null.py`` widened into a coverage experiment.
2. ``f2_direct_bypass``      a growing direct bypass with the text pathway held fixed
   (alpha in 0, 0.5, 1, 2, 3 at beta 0.8, gamma 1, sigma_m 1, rho 0). The mediated
   share must fall while ``rho*_point`` does not move; the non-movement is a PASS.
3. ``f3_shared_cause``       M = X + U, Y = 1[X + U + E > 0]. Strong association,
   no M-to-Y arrow at all, true NIE = 0, and a compatible zero crossing at
   1/sqrt(2) = 0.7071.
4. ``f4_rationalization``    the cue induces reasoning text that then drives the
   answer. A real, fully mediated effect the estimator must recover.
5. ``f5_redundant_explanation``  an accurate but causally redundant explanation: a
   latent state decides the answer and the text reports it faithfully without causing
   it. True NIE = 0, association near-deterministic.
6. ``f6_answer_copying``     the answer is copied from the cue through an opaque token
   channel the mediator does not carry; the cue still lengthens the text a little.
   True NIE = 0 with a large direct effect.
7. ``f7_opposing_effects``   a direct and an indirect path of opposite sign that cancel
   exactly in the total. TE = 0 with both components near 0.30.

Truth
-----
Every family declares its structural equations once, as ``m_of(x, noise)`` and
``y_of(x, m, noise)``. Observed data and ground truth are computed from those same two
functions, so the truth cannot drift from the generator: the potential outcomes are
evaluated on a common noise draw of ``TRUTH_MC`` rows,

    p00 = P(Y(0, M(0)) = 1), p10 = P(Y(1, M(0)) = 1), p11 = P(Y(1, M(1)) = 1)
    NDE = p10 - p00,  NIE = p11 - p10,  TE = NDE + NIE

and every family additionally carries a hand-derived analytic truth, so the Monte Carlo
truth is cross-checked rather than trusted.

Estimator under test
--------------------
The repaired MAP estimator: ``fit_probit_mediation_map(..., rho=0, intercepts=True)``,
converted to probability-scale effects by the repository's exact closed form. Intervals
come from a nonparametric row bootstrap (refit per replicate); a 20-dataset-per-family
subset at n = 350 is re-run through the repaired PyMC posterior instead, and the report
says which number came from which. The rho sweep uses the reparameterisation documented
in ``sensitivity``'s module docstring (one fit at rho = 0 determines the whole curve),
vectorised here and cross-checked against the repository's refit-per-rho
``sensitivity_sweep`` and ``breakdown_frontier``.

Reported per condition, always with its denominator: bias on NDE, NIE and TE with a
Monte Carlo standard error; 95 percent interval coverage with a two-sided
Clopper-Pearson interval; the false-robust-verdict rate (the pre-registered verdict
firing where the true NIE is zero); and the behaviour of ``rho*_point`` (the
point-estimate zero crossing, invariant to the direct path) and ``rho*_decision`` (the
rho at which the pre-registered verdict rule fails).

Run:
    PYTHONPATH=src python experiments/mechanism_battery.py --out experiments/results/mechanism_battery
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.sensitivity import (
    OptimizerWarning,
    breakdown_frontier,
    fit_probit_mediation_map,
    sensitivity_sweep,
)

# --------------------------------------------------------------------------- #
# Pre-registered constants. Nothing in this file chooses a threshold at run time.
# --------------------------------------------------------------------------- #
RHO_MAX = 0.947  # the evaluated range, matching sensitivity._RHO_ABS_MAX - 1e-3
RHO_STEP = 0.005
RHO_GRID = np.round(np.arange(0.0, RHO_MAX + 1e-9, RHO_STEP), 4)
NIE_THRESHOLD = 0.15  # the verdict rule's effect threshold (probability scale)
VERDICT_PROB = 0.95  # posterior probability required for a load-bearing verdict
CRI = 0.95  # interval level for every reported interval
TRUTH_MC = 2_000_000  # rows used for the Monte Carlo truth of each family
TRUTH_SEED = 20260907
DATASET_SEED_BASE = 731_000  # the review's null seed, scaled up to index datasets
BOOTSTRAP_SEED_BASE = 909_000
N_ROWS_DEFAULT = (350, 3600)  # 350 = the A2 per-cell clean-correct target; 3600 = model level
N_DATASETS_DEFAULT = 100
N_BOOTSTRAP_DEFAULT = 200
N_PYMC_DEFAULT = 20
PYMC_N_ROWS = 350
MAX_BOOTSTRAP_REDRAWS = 5

# Generator constants, named so no threshold is a bare number in an equation.
BASELINE_STEPS = 6.0  # E[M | X = 0]: about six reasoning steps
BASELINE_ACCURACY = 0.75  # P(Y = 1) on the clean arm in family 1
BYPASS_ALPHAS = (0.0, 0.5, 1.0, 2.0, 3.0)
BYPASS_BETA = 0.8
BYPASS_GAMMA = 1.0
BYPASS_SIGMA_M = 1.0
RATIONALIZE_BETA = 1.0
RATIONALIZE_GAMMA = 1.2
REDUNDANT_MU_R = 0.2  # latent answer-relevant state at X = 0
REDUNDANT_C = 1.0  # how far the cue moves that latent state
REDUNDANT_SD_E = 0.5  # answer noise given the latent state (small: the answer is decided)
REDUNDANT_SD_DELTA = 0.35  # readout noise (small: the explanation is accurate)
COPY_KAPPA = 1.0  # strength of the opaque copy channel
COPY_GAMMA_M = 0.3  # the cue also lengthens the text a little, carrying no answer
OPPOSING_ALPHA = -BYPASS_BETA * BYPASS_GAMMA  # cancels the mediated path: total effect exactly 0
OFFSET_TARGET = 0.2  # clean-arm latent offset, keeping the probit away from saturation

ALPHA0_BASELINE = float(norm.ppf(BASELINE_ACCURACY))


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #
class Mechanism:
    """One generator family: its structural equations, its draw, and its own truth.

    Subclasses set ``key``, ``family``, ``label`` and ``analytic_truth`` in ``__init__``
    and implement ``noise``, ``m_of`` and ``y_of``. ``draw`` and ``truth`` are shared, so
    a family's ground truth is computed from the same two equations that generated its
    data.
    """

    key: str
    family: str
    label: str
    analytic_truth: tuple[float, float, float]

    def noise(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def m_of(self, x: np.ndarray, z: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def y_of(
        self, x: np.ndarray, m: np.ndarray, z: dict[str, np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError

    def draw(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One seeded dataset: X randomized at the item level, then M, then Y."""
        rng = np.random.default_rng(seed)
        x = rng.binomial(1, 0.5, size=n).astype(int)
        z = self.noise(rng, n)
        m = np.asarray(self.m_of(x.astype(float), z), dtype=float)
        y = np.asarray(self.y_of(x.astype(float), m, z), dtype=int)
        return x, m, y

    def truth(self, n_mc: int = TRUTH_MC, seed: int = TRUTH_SEED) -> tuple[float, float, float]:
        """Monte Carlo ``(NDE, NIE, TE)`` from the family's own potential outcomes."""
        rng = np.random.default_rng(seed)
        z = self.noise(rng, n_mc)
        zeros = np.zeros(n_mc)
        ones = np.ones(n_mc)
        m0 = np.asarray(self.m_of(zeros, z), dtype=float)
        m1 = np.asarray(self.m_of(ones, z), dtype=float)
        p00 = float(np.mean(self.y_of(zeros, m0, z)))
        p10 = float(np.mean(self.y_of(ones, m0, z)))
        p11 = float(np.mean(self.y_of(ones, m1, z)))
        nde = p10 - p00
        nie = p11 - p10
        return nde, nie, nde + nie


def _probit_truth(alpha, beta, gamma, sigma_m, mu_m, alpha0):
    """Analytic truth for a family that is exactly the estimator's own probit SCM."""
    return probit_natural_effects_closed_form(alpha, beta, gamma, sigma_m, 0.0, mu_m, alpha0)


class NoCueEffect(Mechanism):
    """M has a baseline level and no cue dependence; Y is a biased coin. All effects 0."""

    def __init__(self) -> None:
        self.key = "f1_no_cue_effect"
        self.family = "f1_no_cue_effect"
        self.label = "no cue effect, nonzero baseline accuracy and reasoning length"
        self.analytic_truth = (0.0, 0.0, 0.0)

    def noise(self, rng, n):
        return {"m": rng.normal(0.0, 1.0, n), "y": rng.normal(0.0, 1.0, n)}

    def m_of(self, x, z):
        return BASELINE_STEPS + z["m"]

    def y_of(self, x, m, z):
        return (ALPHA0_BASELINE + z["y"] > 0.0).astype(int)


class DirectBypass(Mechanism):
    """The review's process: a fixed text pathway plus a direct bypass of size alpha."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.key = f"f2_direct_bypass_alpha{alpha:g}"
        self.family = "f2_direct_bypass"
        self.label = f"increasing direct bypass, fixed text pathway (alpha = {alpha:g})"
        self.analytic_truth = _probit_truth(
            self.alpha, BYPASS_BETA, BYPASS_GAMMA, BYPASS_SIGMA_M, 0.0, 0.0
        )

    def noise(self, rng, n):
        return {
            "m": rng.normal(0.0, BYPASS_SIGMA_M, n),
            "y": rng.normal(0.0, 1.0, n),
        }

    def m_of(self, x, z):
        return BYPASS_GAMMA * x + z["m"]

    def y_of(self, x, m, z):
        return (self.alpha * x + BYPASS_BETA * m + z["y"] > 0.0).astype(int)


class SharedCause(Mechanism):
    """M = X + U and Y = 1[X + U + E > 0]: a shared hidden cause, no text-to-answer effect."""

    def __init__(self) -> None:
        self.key = "f3_shared_cause"
        self.family = "f3_shared_cause"
        self.label = "shared hidden cause, no text-to-answer effect"
        nde = float(norm.cdf(1.0 / math.sqrt(2.0)) - 0.5)
        self.analytic_truth = (nde, 0.0, nde)

    def noise(self, rng, n):
        return {"u": rng.normal(0.0, 1.0, n), "e": rng.normal(0.0, 1.0, n)}

    def m_of(self, x, z):
        return x + z["u"]

    def y_of(self, x, m, z):
        return (x + z["u"] + z["e"] > 0.0).astype(int)


class Rationalization(Mechanism):
    """The cue induces reasoning text and that text drives the answer: full mediation."""

    def __init__(self) -> None:
        self.key = "f4_rationalization"
        self.family = "f4_rationalization"
        self.label = "cue-induced rationalization that drives the answer"
        self.alpha0 = OFFSET_TARGET - RATIONALIZE_BETA * BASELINE_STEPS
        self.analytic_truth = _probit_truth(
            0.0, RATIONALIZE_BETA, RATIONALIZE_GAMMA, 1.0, BASELINE_STEPS, self.alpha0
        )

    def noise(self, rng, n):
        return {"m": rng.normal(0.0, 1.0, n), "y": rng.normal(0.0, 1.0, n)}

    def m_of(self, x, z):
        return BASELINE_STEPS + RATIONALIZE_GAMMA * x + z["m"]

    def y_of(self, x, m, z):
        return (self.alpha0 + RATIONALIZE_BETA * m + z["y"] > 0.0).astype(int)


class RedundantExplanation(Mechanism):
    """A latent state decides the answer; the text reports it accurately and causes nothing."""

    def __init__(self) -> None:
        self.key = "f5_redundant_explanation"
        self.family = "f5_redundant_explanation"
        self.label = "accurate but causally redundant explanation"
        sd = math.sqrt(1.0 + REDUNDANT_SD_E**2)
        nde = float(
            norm.cdf((REDUNDANT_MU_R + REDUNDANT_C) / sd) - norm.cdf(REDUNDANT_MU_R / sd)
        )
        self.analytic_truth = (nde, 0.0, nde)

    def noise(self, rng, n):
        return {
            "u": rng.normal(0.0, 1.0, n),
            "e": rng.normal(0.0, REDUNDANT_SD_E, n),
            "d": rng.normal(0.0, REDUNDANT_SD_DELTA, n),
        }

    def _latent(self, x, z):
        return REDUNDANT_MU_R + REDUNDANT_C * x + z["u"]

    def m_of(self, x, z):
        return self._latent(x, z) + z["d"]

    def y_of(self, x, m, z):
        return (self._latent(x, z) + z["e"] > 0.0).astype(int)


class AnswerCopying(Mechanism):
    """The answer is copied through an opaque channel; the text lengthens but carries nothing."""

    def __init__(self) -> None:
        self.key = "f6_answer_copying"
        self.family = "f6_answer_copying"
        self.label = "answer copying through an opaque token channel"
        nde = float(norm.cdf(COPY_KAPPA) - norm.cdf(0.0))
        self.analytic_truth = (nde, 0.0, nde)

    def noise(self, rng, n):
        return {"m": rng.normal(0.0, 1.0, n), "y": rng.normal(0.0, 1.0, n)}

    def m_of(self, x, z):
        return BASELINE_STEPS + COPY_GAMMA_M * x + z["m"]

    def y_of(self, x, m, z):
        return (COPY_KAPPA * x + z["y"] > 0.0).astype(int)


class OpposingEffects(Mechanism):
    """A direct and an indirect path of opposite sign that cancel exactly in the total."""

    def __init__(self) -> None:
        self.key = "f7_opposing_effects"
        self.family = "f7_opposing_effects"
        self.label = "opposing direct and indirect effects that cancel in the total"
        self.alpha0 = OFFSET_TARGET - BYPASS_BETA * BASELINE_STEPS
        self.analytic_truth = _probit_truth(
            OPPOSING_ALPHA, BYPASS_BETA, BYPASS_GAMMA, BYPASS_SIGMA_M,
            BASELINE_STEPS, self.alpha0,
        )

    def noise(self, rng, n):
        return {"m": rng.normal(0.0, BYPASS_SIGMA_M, n), "y": rng.normal(0.0, 1.0, n)}

    def m_of(self, x, z):
        return BASELINE_STEPS + BYPASS_GAMMA * x + z["m"]

    def y_of(self, x, m, z):
        return (
            self.alpha0 + OPPOSING_ALPHA * x + BYPASS_BETA * m + z["y"] > 0.0
        ).astype(int)


def build_conditions() -> list[Mechanism]:
    """The 11 evaluated conditions: seven families, family 2 at five bypass strengths."""
    conditions: list[Mechanism] = [NoCueEffect()]
    conditions += [DirectBypass(a) for a in BYPASS_ALPHAS]
    conditions += [
        SharedCause(),
        Rationalization(),
        RedundantExplanation(),
        AnswerCopying(),
        OpposingEffects(),
    ]
    return conditions


FAMILY_ORDER = (
    "f1_no_cue_effect",
    "f2_direct_bypass",
    "f3_shared_cause",
    "f4_rationalization",
    "f5_redundant_explanation",
    "f6_answer_copying",
    "f7_opposing_effects",
)


# --------------------------------------------------------------------------- #
# The estimator under test, and the rho machinery
# --------------------------------------------------------------------------- #
def effects_curve(
    alpha: float,
    beta: float,
    gamma: float,
    sigma_m: float,
    mu_m: float,
    alpha0: float,
    rho_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probability-scale ``(NDE, NIE, TE)`` across ``rho_grid`` from ONE rho = 0 fit.

    The mediator equation does not move with the assumed ``rho``, so the whole
    sensitivity curve is a deterministic function of the ``rho = 0`` fit through the
    reparameterisation in ``sensitivity``'s module docstring:

        beta(rho)   = B  * sqrt(1 - rho^2) - rho/sigma_m
        alpha(rho)  = A  * sqrt(1 - rho^2) + (rho/sigma_m) * gamma
        alpha0(rho) = A0 * sqrt(1 - rho^2) + (rho/sigma_m) * mu_m

    This is the same object the repository computes by refitting at every grid point;
    vectorising it is what makes a 200-replicate bootstrap of the whole curve affordable
    on a laptop. ``tests/test_mechanism_battery.py`` pins it against
    ``probit_natural_effects_closed_form`` and the run's cross-check pins it against
    ``sensitivity_sweep``.
    """
    rho = np.asarray(rho_grid, dtype=float)
    root = np.sqrt(1.0 - rho**2)
    beta_r = beta * root - rho / sigma_m
    alpha_r = alpha * root + (rho / sigma_m) * gamma
    alpha0_r = alpha0 * root + (rho / sigma_m) * mu_m

    var = beta_r**2 * sigma_m**2 + 2.0 * beta_r * rho * sigma_m + 1.0
    sd = np.sqrt(var)
    offset = alpha0_r + beta_r * mu_m

    p00 = norm.cdf(offset / sd)
    p10 = norm.cdf((offset + alpha_r) / sd)
    p11 = norm.cdf((offset + alpha_r + beta_r * gamma) / sd)
    return p10 - p00, p11 - p10, p11 - p00


def rho_star_point(beta: float, sigma_m: float) -> float:
    """The point-estimate zero crossing of the NIE: ``|B| sigma_m / sqrt(1 + B^2 sigma_m^2)``.

    Contains neither the direct coefficient nor either intercept, which is exactly the
    invariance the review established and ``tests/test_rho_star_semantics.py`` pins.
    Returned unsigned; the crossing sits on the side of ``sign(beta)``.
    """
    b = abs(beta) * sigma_m
    return float(b / math.sqrt(1.0 + b * b))


def _fit_at_zero(x: np.ndarray, m: np.ndarray, y: np.ndarray):
    """The repaired MAP fit at rho = 0, with the optimizer warning silenced and counted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizerWarning)
        return fit_probit_mediation_map(x, m, y, rho=0.0, intercepts=True)


def _usable(x: np.ndarray, y: np.ndarray) -> bool:
    """A resample is usable only if both arms and both answers are present."""
    return x.min() != x.max() and y.min() != y.max()


@dataclass(frozen=True)
class DatasetResult:
    """Everything one seeded dataset contributes to the battery."""

    condition: str
    family: str
    n_rows: int
    dataset_index: int
    converged: bool
    degenerate_bootstrap_redraws: int
    answer_rate: float
    mediator_mean: float
    corr_m_y: float
    nde: float
    nie: float
    te: float
    nde_lo: float
    nde_hi: float
    nie_lo: float
    nie_hi: float
    te_lo: float
    te_hi: float
    rho_star_point: float
    rho_star_point_lo: float
    rho_star_point_hi: float
    rho_star_point_in_range: bool
    prob_nie_above_threshold_at_zero: float
    load_bearing_at_zero: bool
    rho_star_decision: float | None
    decision_no_crossing_in_range: bool
    fitted_beta: float
    fitted_gamma: float
    fitted_sigma_m: float
    fitted_mu_m: float


def analyse_dataset(
    mech: Mechanism,
    n_rows: int,
    dataset_index: int,
    n_bootstrap: int,
    rho_grid: np.ndarray = RHO_GRID,
) -> DatasetResult:
    """Fit one seeded dataset and return its effects, intervals, verdict and rho behaviour."""
    x, m, y = mech.draw(n_rows, dataset_seed(dataset_index, n_rows))
    fit = _fit_at_zero(x, m, y)
    nde, nie, te = probit_natural_effects_closed_form(
        fit.alpha, fit.beta, fit.gamma, fit.sigma_m, 0.0, fit.mu_m, fit.alpha0
    )

    boot_rng = np.random.default_rng(BOOTSTRAP_SEED_BASE + 1_000 * dataset_index + n_rows)
    boot_nde = np.empty(n_bootstrap)
    boot_nie_curve = np.empty((n_bootstrap, len(rho_grid)))
    boot_te = np.empty(n_bootstrap)
    boot_rho_star = np.empty(n_bootstrap)
    redraws = 0
    for b in range(n_bootstrap):
        for _ in range(MAX_BOOTSTRAP_REDRAWS):
            idx = boot_rng.integers(0, n_rows, n_rows)
            if _usable(x[idx], y[idx]):
                break
            redraws += 1
        bfit = _fit_at_zero(x[idx], m[idx], y[idx])
        nde_c, nie_c, te_c = effects_curve(
            bfit.alpha, bfit.beta, bfit.gamma, bfit.sigma_m, bfit.mu_m, bfit.alpha0, rho_grid
        )
        boot_nde[b] = nde_c[0]
        boot_te[b] = te_c[0]
        boot_nie_curve[b] = nie_c
        boot_rho_star[b] = rho_star_point(bfit.beta, bfit.sigma_m)

    lo_q, hi_q = (1.0 - CRI) / 2.0, 1.0 - (1.0 - CRI) / 2.0
    nde_lo, nde_hi = np.quantile(boot_nde, [lo_q, hi_q])
    nie_lo, nie_hi = np.quantile(boot_nie_curve[:, 0], [lo_q, hi_q])
    te_lo, te_hi = np.quantile(boot_te, [lo_q, hi_q])
    rs_lo, rs_hi = np.quantile(boot_rho_star, [lo_q, hi_q])

    prob_above = (boot_nie_curve > NIE_THRESHOLD).mean(axis=0)
    load_bearing = bool(prob_above[0] >= VERDICT_PROB)
    rho_decision: float | None = None
    no_crossing = False
    if load_bearing:
        failed = np.flatnonzero(prob_above < VERDICT_PROB)
        if failed.size:
            rho_decision = float(rho_grid[failed[0]])
        else:
            no_crossing = True

    rs_point = rho_star_point(fit.beta, fit.sigma_m)
    return DatasetResult(
        condition=mech.key,
        family=mech.family,
        n_rows=n_rows,
        dataset_index=dataset_index,
        converged=bool(fit.converged),
        degenerate_bootstrap_redraws=redraws,
        answer_rate=float(y.mean()),
        mediator_mean=float(m.mean()),
        corr_m_y=float(np.corrcoef(m, y)[0, 1]),
        nde=float(nde), nie=float(nie), te=float(te),
        nde_lo=float(nde_lo), nde_hi=float(nde_hi),
        nie_lo=float(nie_lo), nie_hi=float(nie_hi),
        te_lo=float(te_lo), te_hi=float(te_hi),
        rho_star_point=rs_point,
        rho_star_point_lo=float(rs_lo),
        rho_star_point_hi=float(rs_hi),
        rho_star_point_in_range=bool(rs_point <= RHO_MAX),
        prob_nie_above_threshold_at_zero=float(prob_above[0]),
        load_bearing_at_zero=load_bearing,
        rho_star_decision=rho_decision,
        decision_no_crossing_in_range=no_crossing,
        fitted_beta=float(fit.beta),
        fitted_gamma=float(fit.gamma),
        fitted_sigma_m=float(fit.sigma_m),
        fitted_mu_m=float(fit.mu_m),
    )


# --------------------------------------------------------------------------- #
# Cross-checks against the repository's own (slower) code paths
# --------------------------------------------------------------------------- #
def cross_checks(conditions: list[Mechanism], n_rows: int, n_datasets: int) -> dict:
    """Pin the fast paths used above against the repository's refit-per-rho functions.

    Three checks, each reported with its denominator:
    1. the vectorised ``effects_curve`` against ``sensitivity_sweep`` (which refits the
       probit model at every grid point and integrates by Monte Carlo);
    2. ``rho_star_point`` against ``breakdown_frontier`` (which root-finds on refits);
    3. the Monte Carlo truth against each family's hand-derived analytic truth.
    """
    sweep_rhos = np.array([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    sweep_diffs: list[float] = []
    frontier_diffs: list[float] = []
    frontier_pairs = []
    for mech in conditions:
        for i in range(n_datasets):
            x, m, y = mech.draw(n_rows, dataset_seed(i, n_rows))
            fit = _fit_at_zero(x, m, y)
            _, nie_curve, _ = effects_curve(
                fit.alpha, fit.beta, fit.gamma, fit.sigma_m, fit.mu_m, fit.alpha0, sweep_rhos
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizerWarning)
                repo_sweep = sensitivity_sweep(
                    x, m, y, rho_grid=sweep_rhos, n_mc=200_000, rng_seed=0
                )
                bf = breakdown_frontier(x, m, y, key="nie", n_mc=200_000, rng_seed=0)
            sweep_diffs.append(
                float(max(abs(p.nie - v) for p, v in zip(repo_sweep, nie_curve)))
            )
            if not bf.unresolved and not bf.survives_full_range:
                predicted = rho_star_point(fit.beta, fit.sigma_m)
                frontier_diffs.append(abs(bf.robustness - predicted))
                frontier_pairs.append((mech.key, float(bf.robustness), predicted))

    truth_diffs = {}
    for mech in conditions:
        mc = mech.truth()
        truth_diffs[mech.key] = float(
            max(abs(a - b) for a, b in zip(mech.analytic_truth, mc))
        )

    return {
        "effects_curve_vs_sensitivity_sweep": {
            "n_datasets": len(sweep_diffs),
            "n_rho_points_each": len(sweep_rhos),
            "n_rows": n_rows,
            "max_abs_nie_difference": max(sweep_diffs) if sweep_diffs else None,
            "mean_abs_nie_difference": float(np.mean(sweep_diffs)) if sweep_diffs else None,
        },
        "rho_star_point_vs_breakdown_frontier": {
            "n_comparisons": len(frontier_diffs),
            "n_rows": n_rows,
            "max_abs_difference": max(frontier_diffs) if frontier_diffs else None,
            "mean_abs_difference": float(np.mean(frontier_diffs)) if frontier_diffs else None,
            "examples": frontier_pairs[:5],
        },
        "monte_carlo_truth_vs_analytic_truth": {
            "n_conditions": len(truth_diffs),
            "n_monte_carlo_rows": TRUTH_MC,
            "max_abs_difference": max(truth_diffs.values()),
            "per_condition": truth_diffs,
        },
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _sibling(name: str):
    """Load a sibling experiments/ module by file path (the repository's convention)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stats_mod = _sibling("mechanism_battery_stats")


def _worker(payload: tuple) -> dict:
    """One dataset, in a worker process. Arguments are plain, picklable values."""
    mech, n_rows, index, n_bootstrap = payload
    return asdict(analyse_dataset(mech, n_rows, index, n_bootstrap))


def run_condition(
    mech: Mechanism,
    n_rows: int,
    n_datasets: int,
    n_bootstrap: int,
    executor: ProcessPoolExecutor | None,
) -> list[DatasetResult]:
    """All seeded datasets for one condition at one sample size."""
    payloads = [(mech, n_rows, i, n_bootstrap) for i in range(n_datasets)]
    if executor is None:
        dicts = [_worker(p) for p in payloads]
    else:
        dicts = list(executor.map(_worker, payloads, chunksize=1))
    return [DatasetResult(**d) for d in dicts]


def pymc_indices(mech: Mechanism, conditions: list[Mechanism], n_pymc: int) -> list[int]:
    """Dataset indices for the PyMC subset: ``n_pymc`` per FAMILY, spread over its conditions.

    Indices, not seeds, so the posterior and the MAP bootstrap are compared on exactly the
    same datasets: both paths turn an index into a seed the same way.
    """
    siblings = [c.key for c in conditions if c.family == mech.family]
    share = n_pymc // len(siblings)
    offset = siblings.index(mech.key)
    return [offset * share + i for i in range(share)]


def dataset_seed(dataset_index: int, n_rows: int) -> int:
    """The one place a dataset index becomes a seed."""
    return DATASET_SEED_BASE + 1_000 * dataset_index + n_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="experiments/results/mechanism_battery")
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS_DEFAULT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--n-pymc", type=int, default=N_PYMC_DEFAULT)
    parser.add_argument("--n-rows", type=int, nargs="+", default=list(N_ROWS_DEFAULT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crosscheck-datasets", type=int, default=3)
    parser.add_argument("--skip-pymc", action="store_true")
    args = parser.parse_args(argv)

    # Children are spawned on macOS, so they inherit this and stay single-threaded:
    # the parallelism here is across datasets, not inside one linear-algebra call.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions()
    started = time.time()

    truths = {c.key: c.truth() for c in conditions}
    aggregates: list[dict] = []
    rows_out: list[dict] = []
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for n_rows in args.n_rows:
            for mech in conditions:
                t0 = time.time()
                rows = run_condition(mech, n_rows, args.n_datasets, args.n_bootstrap, executor)
                agg = stats_mod.aggregate(mech, rows, truths[mech.key])
                agg["seconds"] = round(time.time() - t0, 1)
                aggregates.append(agg)
                rows_out.extend(asdict(r) for r in rows)
                print(
                    f"{mech.key:34s} n={n_rows:5d} "
                    f"NIE bias {agg['effects']['nie']['bias_mean']:+.4f} "
                    f"cov {agg['effects']['nie']['coverage_k']}"
                    f"/{agg['effects']['nie']['coverage_n']} "
                    f"rho*_point {agg['rho_star_point']['mean']:.4f} "
                    f"[{agg['seconds']}s]",
                    flush=True,
                )
                _dump(out_dir, aggregates, rows_out)
    finally:
        if executor is not None:
            executor.shutdown()

    checks = cross_checks(conditions, max(args.n_rows), args.crosscheck_datasets)
    (out_dir / "cross_checks.json").write_text(json.dumps(checks, indent=2))

    pymc_out: list[dict] = []
    if not args.skip_pymc and args.n_pymc > 0:
        for mech in conditions:
            indices = pymc_indices(mech, conditions, args.n_pymc)
            if not indices:
                continue
            seeds = [dataset_seed(i, PYMC_N_ROWS) for i in indices]
            map_rows = [
                analyse_dataset(mech, PYMC_N_ROWS, i, args.n_bootstrap) for i in indices
            ]
            block = stats_mod.pymc_subset(mech, PYMC_N_ROWS, seeds, map_rows)
            pymc_out.append(block)
            print(
                f"pymc {mech.key:30s} NIE cov "
                f"{block['nie']['coverage_k']}/{block['nie']['coverage_n']} "
                f"max r_hat {block['max_r_hat']:.4f}",
                flush=True,
            )
            (out_dir / "pymc_subset.json").write_text(json.dumps(pymc_out, indent=2))

    probe: list[dict] = []
    if not args.skip_pymc and args.n_pymc > 0:
        probe = stats_mod.prior_scale_probe(
            [Rationalization(), SharedCause()], PYMC_N_ROWS, 0, dataset_seed
        )

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 1),
        "settings": vars(args),
        "constants": {
            "rho_max": RHO_MAX, "rho_step": RHO_STEP, "nie_threshold": NIE_THRESHOLD,
            "verdict_prob": VERDICT_PROB, "cri": CRI, "truth_mc_rows": TRUTH_MC,
            "truth_seed": TRUTH_SEED, "dataset_seed_base": DATASET_SEED_BASE,
            "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
        },
        "conditions": aggregates,
        "cross_checks": checks,
        "pymc_subset": pymc_out,
        "prior_scale_probe": probe,
    }
    (out_dir / "battery_results.json").write_text(json.dumps(payload, indent=2))
    report_mod = _sibling("mechanism_battery_report")
    (out_dir / "report.md").write_text(report_mod.build_report(payload))
    print(f"wrote {out_dir}/report.md in {payload['runtime_seconds']}s")
    return 0


def _dump(out_dir: Path, aggregates: list[dict], rows: list[dict]) -> None:
    """Write partial results after every condition so a crash costs one condition, not a run."""
    (out_dir / "conditions_partial.json").write_text(json.dumps(aggregates, indent=2))
    header = ",".join(rows[0].keys())
    lines = [header] + [",".join("" if v is None else str(v) for v in r.values()) for r in rows]
    (out_dir / "dataset_rows.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
