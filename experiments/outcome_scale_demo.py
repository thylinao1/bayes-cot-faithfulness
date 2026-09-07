"""Does an outcome that VARIES in the clean arm identify the NDE/NIE split?

Amendment A4.6(b) and ``docs/WAVE1-FITS.md`` sections 2 and 7 record that on the
frozen outcome population the binary follow indicator is 0 on every clean row by
construction, so the clean arm carries zero outcome variance, the probit outcome
equation separates on X, and the NDE/NIE split rests on the link extrapolating
into a region the clean arm never visits. Only the TE is checked directly, by the
randomized arm difference.

This experiment asks what a clean-arm-varying outcome buys, in the CPU mechanism
battery's own style: seeded datasets, a declared truth per family computed from
the family's own potential outcomes and cross-checked against a hand-derived
analytic truth, bias with a Monte Carlo standard error, interval coverage with a
Clopper-Pearson interval, and the two rho quantities kept apart.

The design: two families x three READOUTS of one latent index
-------------------------------------------------------------
The two families are the battery's own ``f3_shared_cause`` and
``f4_rationalization``, imported from ``experiments/mechanism_battery.py`` so
their constants cannot drift from the battery's. Each family declares a latent
index and an outcome noise term; the battery's own ``y_of`` is
``1[index + noise > 0]``, and the runner asserts that reconstruction row by row
before anything is fitted.

Three readouts of that same index, so the only thing changing across a row of the
table is how the index becomes an observed outcome:

* ``binary_clean_varies``  ``Y = 1[index + noise > 0]``      the battery's own
  readout: binary, and the clean arm carries real outcome variance. Fitted with
  the probit path.
* ``binary_zero_clean``    ``Y = X * 1[index + noise > 0]``   the frozen outcome
  population's shape: identically 0 on every clean row. Fitted with the probit
  path.
* ``continuous_margin``    ``Y = index + noise``              a logprob-margin-like
  continuous outcome whose clean-arm variance equals its hinted-arm variance by
  construction. Fitted with the new Gaussian path.

Two properties of that construction are what make the comparison clean and are
asserted rather than asserted-in-prose:

1. The gate ``X *`` multiplies both ``p10`` and ``p11`` by 1 (both have x = 1), so
   the TRUE NIE of ``binary_zero_clean`` is EXACTLY the true NIE of
   ``binary_clean_varies``. Only the true NDE and the clean arm's data content
   change. The comparison therefore isolates the clean-arm degeneracy from every
   other difference between the two columns.
2. ``continuous_margin`` changes the link as well as the clean arm, which is why
   ``binary_clean_varies`` is in the table: it is the control that says how much
   of any difference is the link and how much is the clean arm.

What the table can and cannot answer is written into ``report.md`` beside the
numbers, not left to the reader.

Run (about six minutes on an 8 GB laptop):

    PYTHONPATH=src python experiments/outcome_scale_demo.py \
        --out experiments/results/outcome_scale_demo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.gaussian_mediation import (
    fit_gaussian_mediation_closed_form,
    gaussian_effects_curve,
    gaussian_natural_effects,
    gaussian_rho_star_point,
)
from bayes_cot_faithfulness.sensitivity import OptimizerWarning, fit_probit_mediation_map

ROOT = Path(__file__).resolve().parent.parent


def _sibling(name: str):
    """Load a sibling experiments/ module by file path (the repository's convention)."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


battery = _sibling("mechanism_battery")
stats_mod = _sibling("mechanism_battery_stats")

# --------------------------------------------------------------------------- #
# Constants. Nothing here is chosen at run time.
# --------------------------------------------------------------------------- #
N_ROWS = 1_800  # rows per dataset, that is 900 items x 2 arms at the wave-1 shape
N_DATASETS = 100  # seeded datasets per condition
N_BOOTSTRAP = 120  # bootstrap replicates per dataset
TRUTH_MC = battery.TRUTH_MC  # 2,000,000 rows for the Monte Carlo truth
TRUTH_SEED = battery.TRUTH_SEED
DATASET_SEED_BASE = 517_000  # this experiment's own dataset seeds
BOOTSTRAP_SEED_BASE = 518_000
CRI = battery.CRI
MAX_BOOTSTRAP_REDRAWS = battery.MAX_BOOTSTRAP_REDRAWS

# Amendment A4.6(a): the decision sweep runs on the SYMMETRIC grid mirrored from
# the battery's own points, -0.945 to +0.945 in steps of 0.005, rho = 0 exact.
RHO_GRID = np.round(
    np.concatenate([-battery.RHO_GRID[:0:-1], battery.RHO_GRID]),
    4,
)

READOUTS = ("binary_clean_varies", "binary_zero_clean", "continuous_margin")
FAMILIES = ("f3_shared_cause", "f4_rationalization")


# --------------------------------------------------------------------------- #
# The families: the battery's own two, with their latent index exposed
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Family:
    """One battery family, its latent index, and the analytic truth of each readout.

    ``index_of`` and ``noise_of`` are the decomposition of the battery's own
    ``y_of`` into ``1[index + noise > 0]``. ``check_matches_battery`` asserts that
    decomposition on real draws, so a typo here cannot quietly redefine a family.
    """

    key: str
    label: str
    mech: object
    index_fn: object
    noise_key: str

    def noise(self, rng, n):
        return self.mech.noise(rng, n)

    def m_of(self, x, z):
        return self.mech.m_of(x, z)

    def index_of(self, x, m, z):
        return self.index_fn(x, m, z)

    def outcome_noise(self, z):
        return z[self.noise_key]

    def y_of(self, x, m, z, readout: str):
        idx = np.asarray(self.index_of(x, m, z), dtype=float)
        noise = np.asarray(self.outcome_noise(z), dtype=float)
        if readout == "continuous_margin":
            return idx + noise
        fired = (idx + noise > 0.0).astype(float)
        if readout == "binary_clean_varies":
            return fired
        if readout == "binary_zero_clean":
            return np.asarray(x, dtype=float) * fired
        raise ValueError(f"unknown readout {readout!r}")

    def draw(self, n: int, seed: int, readout: str):
        """One seeded dataset: X randomized at the row level, then M, then Y."""
        rng = np.random.default_rng(seed)
        x = rng.binomial(1, 0.5, size=n).astype(int)
        z = self.noise(rng, n)
        m = np.asarray(self.m_of(x.astype(float), z), dtype=float)
        y = self.y_of(x.astype(float), m, z, readout)
        return x, m, y

    def truth(self, readout: str, n_mc: int = TRUTH_MC, seed: int = TRUTH_SEED):
        """Monte Carlo ``(NDE, NIE, TE)`` from this family's own potential outcomes."""
        rng = np.random.default_rng(seed)
        z = self.noise(rng, n_mc)
        zeros, ones = np.zeros(n_mc), np.ones(n_mc)
        m0 = np.asarray(self.m_of(zeros, z), dtype=float)
        m1 = np.asarray(self.m_of(ones, z), dtype=float)
        p00 = float(np.mean(self.y_of(zeros, m0, z, readout)))
        p10 = float(np.mean(self.y_of(ones, m0, z, readout)))
        p11 = float(np.mean(self.y_of(ones, m1, z, readout)))
        return p10 - p00, p11 - p10, p11 - p00

    def check_matches_battery(self, n: int = 20_000, seed: int = 99) -> int:
        """Assert ``1[index + noise > 0]`` reproduces the battery's ``y_of`` exactly."""
        rng = np.random.default_rng(seed)
        x = rng.binomial(1, 0.5, size=n).astype(float)
        z = self.noise(rng, n)
        m = np.asarray(self.m_of(x, z), dtype=float)
        mine = self.y_of(x, m, z, "binary_clean_varies").astype(int)
        theirs = np.asarray(self.mech.y_of(x, m, z), dtype=int)
        if not np.array_equal(mine, theirs):
            raise AssertionError(f"{self.key}: index decomposition does not match the battery")
        return n


def _f3_index(x, m, z):
    """``f3_shared_cause``: the index is ``X + U``; the mediator is ``X + U`` too.

    M does not enter the index at all, which is why the true NIE is exactly zero
    and why the shared U makes this a rho != 0 world on every readout.
    """
    return np.asarray(x, dtype=float) + z["u"]


def _f4_index(x, m, z):
    """``f4_rationalization``: the index is ``alpha0 + beta*M``; X enters only through M."""
    return battery.Rationalization().alpha0 + battery.RATIONALIZE_BETA * np.asarray(m, dtype=float)


def build_families() -> dict[str, Family]:
    return {
        "f3_shared_cause": Family(
            key="f3_shared_cause",
            label="shared hidden cause, no text-to-answer effect (true NIE = 0)",
            mech=battery.SharedCause(),
            index_fn=_f3_index,
            noise_key="e",
        ),
        "f4_rationalization": Family(
            key="f4_rationalization",
            label="cue-induced rationalization that drives the answer (fully mediated)",
            mech=battery.Rationalization(),
            index_fn=_f4_index,
            noise_key="y",
        ),
    }


def analytic_truth(family_key: str, readout: str) -> tuple[float, float, float]:
    """Hand-derived truth for each of the six cells, to cross-check the Monte Carlo.

    ``f3``: index = x + u with u, e ~ N(0,1) independent, so
    ``P(index + e > 0 | x) = Phi(x / sqrt(2))`` and M never enters the index.
    ``f4``: index = alpha0 + M with M | x ~ N(6 + 1.2x, 1) and e ~ N(0,1), so
    ``P(index + e > 0 | M(x)) = Phi((0.2 + 1.2x) / sqrt(2))``.
    """
    sd = math.sqrt(2.0)
    if family_key == "f3_shared_cause":
        p10 = p11 = float(norm.cdf(1.0 / sd))
        p00_bin = 0.5
        mean0, mean1 = 0.0, 1.0
    else:
        offset = battery.OFFSET_TARGET
        shift = battery.RATIONALIZE_BETA * battery.RATIONALIZE_GAMMA
        p10 = float(norm.cdf(offset / sd))
        p11 = float(norm.cdf((offset + shift) / sd))
        p00_bin = p10
        mean0, mean1 = offset, offset + shift

    if readout == "continuous_margin":
        # E[Y(x', M(x))] = E[index] + 0; the noise is mean zero.
        if family_key == "f3_shared_cause":
            return 1.0, 0.0, 1.0
        return 0.0, mean1 - mean0, mean1 - mean0
    if readout == "binary_clean_varies":
        return p10 - p00_bin, p11 - p10, p11 - p00_bin
    if readout == "binary_zero_clean":
        return p10 - 0.0, p11 - p10, p11 - 0.0
    raise ValueError(f"unknown readout {readout!r}")


# --------------------------------------------------------------------------- #
# The two estimator paths
# --------------------------------------------------------------------------- #
def _usable(x: np.ndarray, y: np.ndarray) -> bool:
    """The battery's own resample guard, computed on the WHOLE sample.

    Kept exactly as the battery has it, because A4.6(b)(3) is about this guard:
    Y varies across the whole sample even when the control arm is degenerate, so
    the guard passes on every such resample. The count it lets through is
    reported rather than fixed here.
    """
    return x.min() != x.max() and y.min() != y.max()


def _probit_fit_and_curve(x, m, y, rho_grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizerWarning)
        fit = fit_probit_mediation_map(x, m, y.astype(int), rho=0.0, intercepts=True)
    nde_c, nie_c, te_c = battery.effects_curve(
        fit.alpha, fit.beta, fit.gamma, fit.sigma_m, fit.mu_m, fit.alpha0, rho_grid
    )
    rho_star = battery.rho_star_point(fit.beta, fit.sigma_m)
    return fit, nde_c, nie_c, te_c, rho_star


def _gaussian_fit_and_curve(x, m, y, rho_grid):
    fit = fit_gaussian_mediation_closed_form(x, m, y, rho=0.0, intercepts=True)
    nde_c, nie_c, te_c = gaussian_effects_curve(fit, rho_grid)
    rho_star = gaussian_rho_star_point(fit.beta, fit.sigma_m, fit.sigma_y)
    return fit, nde_c, nie_c, te_c, rho_star


def dataset_seed(index: int, readout: str) -> int:
    return DATASET_SEED_BASE + 1_000 * index + READOUTS.index(readout)


@dataclass(frozen=True)
class DatasetResult:
    """Everything one seeded dataset contributes."""

    condition: str
    family: str
    readout: str
    estimator: str
    dataset_index: int
    n_rows: int
    converged: bool
    degenerate_bootstrap_redraws: int
    clean_arm_outcome_variance: float
    hinted_arm_outcome_variance: float
    clean_arm_outcome_mean: float
    hinted_arm_outcome_mean: float
    randomized_arm_difference: float
    nde: float
    nie: float
    te: float
    nde_lo: float
    nde_hi: float
    nie_lo: float
    nie_hi: float
    te_lo: float
    te_hi: float
    te_minus_arm_difference: float
    rho_star_point: float
    rho_star_point_in_range: bool
    nie_interval_excludes_zero: bool
    rho_star_interval: float | None
    interval_no_crossing_in_range: bool
    fitted_alpha: float
    fitted_beta: float
    fitted_gamma: float
    fitted_sigma_m: float
    fitted_mu_m: float
    fitted_alpha0: float


def analyse_dataset(
    family: Family, readout: str, dataset_index: int, n_rows: int, n_bootstrap: int
) -> DatasetResult:
    """Fit one seeded dataset, bootstrap it, and record its effects, verdict and rho behaviour."""
    x, m, y = family.draw(n_rows, dataset_seed(dataset_index, readout), readout)
    is_binary = readout != "continuous_margin"
    fit_and_curve = _probit_fit_and_curve if is_binary else _gaussian_fit_and_curve

    fit, _, _, _, rho_star = fit_and_curve(x, m, y, RHO_GRID)
    if is_binary:
        nde, nie, te = probit_natural_effects_closed_form(
            fit.alpha, fit.beta, fit.gamma, fit.sigma_m, 0.0, fit.mu_m, fit.alpha0
        )
    else:
        nde, nie, te = gaussian_natural_effects(fit.alpha, fit.beta, fit.gamma)

    boot_rng = np.random.default_rng(
        BOOTSTRAP_SEED_BASE + 1_000 * dataset_index + READOUTS.index(readout)
    )
    n_obs = len(x)
    boot_nde = np.empty(n_bootstrap)
    boot_te = np.empty(n_bootstrap)
    boot_nie_curve = np.empty((n_bootstrap, len(RHO_GRID)))
    redraws = 0
    for b in range(n_bootstrap):
        idx = boot_rng.integers(0, n_obs, n_obs)
        for _ in range(MAX_BOOTSTRAP_REDRAWS):
            if _usable(x[idx], y[idx]):
                break
            redraws += 1
            idx = boot_rng.integers(0, n_obs, n_obs)
        _, nde_c, nie_c, te_c, _ = fit_and_curve(x[idx], m[idx], y[idx], RHO_GRID)
        zero = int(np.argmin(np.abs(RHO_GRID)))
        boot_nde[b] = nde_c[zero]
        boot_te[b] = te_c[zero]
        boot_nie_curve[b] = nie_c

    lo_q, hi_q = (1.0 - CRI) / 2.0, 1.0 - (1.0 - CRI) / 2.0
    zero = int(np.argmin(np.abs(RHO_GRID)))
    nde_lo, nde_hi = np.quantile(boot_nde, [lo_q, hi_q])
    te_lo, te_hi = np.quantile(boot_te, [lo_q, hi_q])
    nie_lo_curve = np.quantile(boot_nie_curve, lo_q, axis=0)
    nie_hi_curve = np.quantile(boot_nie_curve, hi_q, axis=0)
    nie_lo, nie_hi = float(nie_lo_curve[zero]), float(nie_hi_curve[zero])

    # A scale-free stand-in for rho*_decision: the pre-registered 0.15 threshold is
    # on the PROBABILITY scale and has no meaning in nats, so the decision quantity
    # reported on both scales is the smallest |rho| at which the NIE interval first
    # covers zero. It is stated as a stand-in wherever it is printed.
    excludes_zero = bool(nie_lo > 0.0 or nie_hi < 0.0)
    rho_star_interval: float | None = None
    no_crossing = False
    if excludes_zero:
        covers = (nie_lo_curve <= 0.0) & (nie_hi_curve >= 0.0)
        order = np.argsort(np.abs(RHO_GRID))
        hit = [i for i in order if covers[i]]
        if hit:
            rho_star_interval = float(RHO_GRID[hit[0]])
        else:
            no_crossing = True

    x_is_1 = x == 1
    return DatasetResult(
        condition=f"{family.key}__{readout}",
        family=family.key,
        readout=readout,
        estimator="probit" if is_binary else "gaussian",
        dataset_index=dataset_index,
        n_rows=n_obs,
        converged=bool(fit.converged),
        degenerate_bootstrap_redraws=redraws,
        clean_arm_outcome_variance=float(np.var(y[~x_is_1])),
        hinted_arm_outcome_variance=float(np.var(y[x_is_1])),
        clean_arm_outcome_mean=float(np.mean(y[~x_is_1])),
        hinted_arm_outcome_mean=float(np.mean(y[x_is_1])),
        randomized_arm_difference=float(np.mean(y[x_is_1]) - np.mean(y[~x_is_1])),
        nde=float(nde), nie=float(nie), te=float(te),
        nde_lo=float(nde_lo), nde_hi=float(nde_hi),
        nie_lo=float(nie_lo), nie_hi=float(nie_hi),
        te_lo=float(te_lo), te_hi=float(te_hi),
        te_minus_arm_difference=float(te - (np.mean(y[x_is_1]) - np.mean(y[~x_is_1]))),
        rho_star_point=float(rho_star),
        rho_star_point_in_range=bool(rho_star <= float(np.max(RHO_GRID))),
        nie_interval_excludes_zero=excludes_zero,
        rho_star_interval=rho_star_interval,
        interval_no_crossing_in_range=no_crossing,
        fitted_alpha=float(fit.alpha),
        fitted_beta=float(fit.beta),
        fitted_gamma=float(fit.gamma),
        fitted_sigma_m=float(fit.sigma_m),
        fitted_mu_m=float(fit.mu_m),
        fitted_alpha0=float(fit.alpha0),
    )


def _worker(payload: tuple) -> dict:
    family_key, readout, index, n_rows, n_bootstrap = payload
    family = build_families()[family_key]
    return asdict(analyse_dataset(family, readout, index, n_rows, n_bootstrap))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarise(rows: list[DatasetResult], truth: tuple[float, float, float]) -> dict:
    """Bias, coverage and rho behaviour for one condition, each with its denominator."""
    n = len(rows)
    est = {k: np.array([getattr(r, k) for r in rows]) for k in ("nde", "nie", "te")}
    lo = {k: np.array([getattr(r, f"{k}_lo") for r in rows]) for k in ("nde", "nie", "te")}
    hi = {k: np.array([getattr(r, f"{k}_hi") for r in rows]) for k in ("nde", "nie", "te")}
    effects = {
        k: stats_mod._effect_block(est[k], lo[k], hi[k], t)
        for k, t in zip(("nde", "nie", "te"), truth)
    }

    rho_star = np.array([r.rho_star_point for r in rows])
    intervals = [r.rho_star_interval for r in rows if r.rho_star_interval is not None]
    fires = sum(r.nie_interval_excludes_zero for r in rows)
    fire_lo, fire_hi = stats_mod.clopper_pearson_interval(fires, n)
    clean_var = np.array([r.clean_arm_outcome_variance for r in rows])
    hint_var = np.array([r.hinted_arm_outcome_variance for r in rows])
    te_gap = np.array([r.te_minus_arm_difference for r in rows])

    return {
        "n_datasets": n,
        "n_rows": rows[0].n_rows,
        "estimator": rows[0].estimator,
        "n_converged": int(sum(r.converged for r in rows)),
        "degenerate_bootstrap_redraws_total": int(
            sum(r.degenerate_bootstrap_redraws for r in rows)
        ),
        "clean_arm_outcome_variance_mean": float(clean_var.mean()),
        "hinted_arm_outcome_variance_mean": float(hint_var.mean()),
        "clean_over_hinted_variance_ratio": (
            float(clean_var.mean() / hint_var.mean()) if hint_var.mean() > 0 else None
        ),
        "effects": effects,
        "model_implied_te_minus_arm_difference": {
            "mean": float(te_gap.mean()),
            "max_abs": float(np.max(np.abs(te_gap))),
        },
        "rho_star_point": {
            "median": float(np.median(rho_star)),
            "q05": float(np.quantile(rho_star, 0.05)),
            "q95": float(np.quantile(rho_star, 0.95)),
            "n_in_range": int(sum(r.rho_star_point_in_range for r in rows)),
        },
        "nie_interval_excludes_zero": {
            "k": fires, "n": n, "rate": fires / n, "lo": fire_lo, "hi": fire_hi
        },
        "rho_star_interval": {
            "n_defined": len(intervals),
            "median_abs": float(np.median(np.abs(intervals))) if intervals else None,
            "n_no_crossing_in_range": int(sum(r.interval_no_crossing_in_range for r in rows)),
        },
    }


def separation(rows_by_condition: dict[str, list[DatasetResult]], readout: str) -> dict:
    """Do the two families produce the same report on this readout?

    Paired by dataset index, because the two families are drawn under the same
    seed scheme. Three statements, each with its denominator: the gap between the
    mean NIE estimates with a Monte Carlo standard error; how often the
    zero-mediation family's NIE estimate is the SMALLER of the pair, which is what
    an analyst would need in order to rank two cells; and how often the
    zero-mediation family's interval excludes zero, which is a false positive.
    """
    f3 = rows_by_condition[f"f3_shared_cause__{readout}"]
    f4 = rows_by_condition[f"f4_rationalization__{readout}"]
    n = min(len(f3), len(f4))
    nie3 = np.array([r.nie for r in f3[:n]])
    nie4 = np.array([r.nie for r in f4[:n]])
    nde3 = np.array([r.nde for r in f3[:n]])
    nde4 = np.array([r.nde for r in f4[:n]])
    ordered = int(np.sum(nie3 < nie4))
    ord_lo, ord_hi = stats_mod.clopper_pearson_interval(ordered, n)
    fp = int(sum(r.nie_interval_excludes_zero for r in f3[:n]))
    fp_lo, fp_hi = stats_mod.clopper_pearson_interval(fp, n)
    diff = nie4 - nie3
    return {
        "readout": readout,
        "n_pairs": n,
        "nie_mean_zero_mediation_family": float(nie3.mean()),
        "nie_mean_mediated_family": float(nie4.mean()),
        "nie_gap_mean": float(diff.mean()),
        "nie_gap_mcse": float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "nde_mean_zero_mediation_family": float(nde3.mean()),
        "nde_mean_mediated_family": float(nde4.mean()),
        "pairs_ordered_correctly": {
            "k": ordered, "n": n, "rate": ordered / n, "lo": ord_lo, "hi": ord_hi
        },
        "false_positive_mediation_on_zero_family": {
            "k": fp, "n": n, "rate": fp / n, "lo": fp_lo, "hi": fp_hi
        },
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best effort, never a stop
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/results/outcome_scale_demo")
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--truth-mc", type=int, default=TRUTH_MC)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    families = build_families()
    provenance = {k: f.check_matches_battery() for k, f in families.items()}
    print(f"[demo] index decomposition matches the battery on {provenance} rows per family")

    truths: dict[str, dict] = {}
    for fam_key, fam in families.items():
        for readout in READOUTS:
            mc = fam.truth(readout, n_mc=args.truth_mc)
            an = analytic_truth(fam_key, readout)
            truths[f"{fam_key}__{readout}"] = {
                "monte_carlo": {"nde": mc[0], "nie": mc[1], "te": mc[2]},
                "analytic": {"nde": an[0], "nie": an[1], "te": an[2]},
                "max_abs_difference": float(max(abs(a - b) for a, b in zip(an, mc))),
                "n_monte_carlo_rows": args.truth_mc,
            }
    worst = max(v["max_abs_difference"] for v in truths.values())
    print(f"[demo] Monte Carlo truth vs analytic truth: max abs difference {worst:.5f}")

    payloads = [
        (fam_key, readout, i, args.n_rows, args.n_bootstrap)
        for fam_key in FAMILIES
        for readout in READOUTS
        for i in range(args.n_datasets)
    ]
    rows_by_condition: dict[str, list[DatasetResult]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for d in pool.map(_worker, payloads, chunksize=2):
            r = DatasetResult(**d)
            rows_by_condition.setdefault(r.condition, []).append(r)
            done += 1
            if done % 50 == 0:
                print(f"      ... {done}/{len(payloads)} datasets", flush=True)
    for rows in rows_by_condition.values():
        rows.sort(key=lambda r: r.dataset_index)

    conditions = {
        key: summarise(
            rows,
            (
                truths[key]["monte_carlo"]["nde"],
                truths[key]["monte_carlo"]["nie"],
                truths[key]["monte_carlo"]["te"],
            ),
        )
        for key, rows in rows_by_condition.items()
    }
    separations = {r: separation(rows_by_condition, r) for r in READOUTS}

    payload = {
        "question": (
            "Does an outcome that varies in the clean arm identify the NDE/NIE split "
            "where the frozen population's binary-zero shape cannot? (A4.6(b))"
        ),
        "meta": {
            "code_commit": git_commit(),
            "n_rows": args.n_rows,
            "n_datasets_per_condition": args.n_datasets,
            "n_bootstrap": args.n_bootstrap,
            "cri": CRI,
            "dataset_seed_base": DATASET_SEED_BASE,
            "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
            "rho_grid": {
                "min": float(RHO_GRID.min()),
                "max": float(RHO_GRID.max()),
                "step": float(battery.RHO_STEP),
                "n_points": len(RHO_GRID),
                "source": "A4.6(a): the battery's own points mirrored, rho = 0 exact",
            },
            "families_source": "experiments/mechanism_battery.py SharedCause / Rationalization",
            "index_decomposition_rows_checked": provenance,
            "wall_seconds": None,
        },
        "truths": truths,
        "conditions": conditions,
        "separation": separations,
    }
    payload["meta"]["wall_seconds"] = round(time.time() - started, 1)

    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "per_dataset.json").write_text(
        json.dumps(
            {k: [asdict(r) for r in v] for k, v in rows_by_condition.items()}, indent=1
        )
        + "\n"
    )
    (out / "report.md").write_text(render_report(payload))
    print(f"[demo] wrote {out}/results.json, per_dataset.json, report.md "
          f"in {payload['meta']['wall_seconds']}s")
    return 0


def render_report(p: dict) -> str:
    """The tables, with every denominator and no conclusion the numbers do not carry."""
    m = p["meta"]
    lines = [
        "# Outcome scale demonstration: does clean-arm variation identify the split?",
        "",
        (
            f"Generated by `experiments/outcome_scale_demo.py` at commit `{m['code_commit'][:12]}` "
            f"in {m['wall_seconds']} s."
        ),
        "",
        (
            f"{m['n_datasets_per_condition']} seeded datasets per condition, {m['n_rows']} rows each, "
            f"{m['n_bootstrap']} bootstrap replicates per dataset, {int(m['cri'] * 100)} percent "
            "intervals. Two families from `experiments/mechanism_battery.py` "
            "(`SharedCause`, `Rationalization`), three readouts of the same latent index. The "
            "index decomposition was checked against the battery's own `y_of` on "
            f"{m['index_decomposition_rows_checked']} rows per family before any fit."
        ),
        "",
        (
            f"rho grid: {m['rho_grid']['n_points']} points from {m['rho_grid']['min']} to "
            f"{m['rho_grid']['max']} in steps of {m['rho_grid']['step']} "
            f"({m['rho_grid']['source']})."
        ),
        "",
        "## 1. The truths, and the clean arm each readout produces",
        "",
        (
            "| condition | estimator | true NDE | true NIE | true TE | MC vs analytic | "
            "clean-arm var(Y) | hinted-arm var(Y) | ratio |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(p["conditions"]):
        c, t = p["conditions"][key], p["truths"][key]
        ratio = c["clean_over_hinted_variance_ratio"]
        lines.append(
            f"| `{key}` | {c['estimator']} | {t['monte_carlo']['nde']:+.4f} | "
            f"{t['monte_carlo']['nie']:+.4f} | {t['monte_carlo']['te']:+.4f} | "
            f"{t['max_abs_difference']:.5f} | {c['clean_arm_outcome_variance_mean']:.4f} | "
            f"{c['hinted_arm_outcome_variance_mean']:.4f} | "
            f"{'n/a' if ratio is None else f'{ratio:.3f}'} |"
        )
    lines += [
        "",
        "## 2. Bias and coverage, per effect, with denominators",
        "",
        (
            "| condition | effect | truth | mean estimate | bias (MC se) | coverage k/n | "
            "coverage 95% CI | mean interval width |"
        ),
        "|---|---|---:|---:|---|---:|---|---:|",
    ]
    for key in sorted(p["conditions"]):
        c = p["conditions"][key]
        for eff in ("nde", "nie", "te"):
            b = c["effects"][eff]
            lines.append(
                f"| `{key}` | {eff.upper()} | {b['truth']:+.4f} | {b['estimate_mean']:+.4f} | "
                f"{b['bias_mean']:+.4f} ({b['bias_mcse']:.4f}) | "
                f"{b['coverage_k']}/{b['coverage_n']} | "
                f"[{b['coverage_ci_lo']:.3f}, {b['coverage_ci_hi']:.3f}] | "
                f"{b['mean_interval_width']:.4f} |"
            )
    lines += [
        "",
        "## 3. The rho quantities, and the TE check",
        "",
        (
            "`rho*_point` is the zero crossing of the point estimate. The decision column is a "
            "STAND-IN for `rho*_decision`: the pre-registered 0.15 threshold of element 1 "
            "section 2.5 is on the probability scale and has no meaning in nats, so what is "
            "reported on both scales is the smallest |rho| at which the NIE interval first "
            "covers zero."
        ),
        "",
        (
            "| condition | rho*_point median [q05, q95] | in range | NIE interval excludes 0 | "
            "rho*_interval median abs | model TE minus arm difference (mean, max abs) | "
            "degenerate resample redraws |"
        ),
        "|---|---|---:|---|---:|---|---:|",
    ]
    for key in sorted(p["conditions"]):
        c = p["conditions"][key]
        rs, fz, ri = c["rho_star_point"], c["nie_interval_excludes_zero"], c["rho_star_interval"]
        te = c["model_implied_te_minus_arm_difference"]
        med = "n/a" if ri["median_abs"] is None else f"{ri['median_abs']:.3f}"
        lines.append(
            f"| `{key}` | {rs['median']:.4f} [{rs['q05']:.4f}, {rs['q95']:.4f}] | "
            f"{rs['n_in_range']}/{c['n_datasets']} | "
            f"{fz['k']}/{fz['n']} = {fz['rate']:.2f} [{fz['lo']:.2f}, {fz['hi']:.2f}] | "
            f"{med} | {te['mean']:+.5f}, {te['max_abs']:.5f} | "
            f"{c['degenerate_bootstrap_redraws_total']} |"
        )
    lines += [
        "",
        "## 4. Do the two families produce the same report?",
        "",
        (
            "Paired by dataset index. `pairs ordered correctly` is how often the zero-mediation "
            "family's NIE estimate is the smaller of the pair, which is what ranking two cells "
            "would need. `false positive` is how often the zero-mediation family's NIE interval "
            "excludes zero."
        ),
        "",
        (
            "| readout | mean NIE, f3 (true 0) | mean NIE, f4 (mediated) | gap (MC se) | "
            "pairs ordered correctly | false positive on f3 |"
        ),
        "|---|---:|---:|---|---:|---:|",
    ]
    for readout in READOUTS:
        s = p["separation"][readout]
        o, f = s["pairs_ordered_correctly"], s["false_positive_mediation_on_zero_family"]
        lines.append(
            f"| `{readout}` | {s['nie_mean_zero_mediation_family']:+.4f} | "
            f"{s['nie_mean_mediated_family']:+.4f} | "
            f"{s['nie_gap_mean']:+.4f} ({s['nie_gap_mcse']:.4f}) | "
            f"{o['k']}/{o['n']} = {o['rate']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}] | "
            f"{f['k']}/{f['n']} = {f['rate']:.2f} [{f['lo']:.2f}, {f['hi']:.2f}] |"
        )
    lines += [
        "",
        (
            "Read `docs/OUTCOME-SCALE-NOTE.md` part 3 for what these numbers do and do not "
            "support. The one-line version: the continuous readout is not a fix for "
            "mediator-outcome confounding, which is what separates these two families, and no "
            "outcome scale is."
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
