"""Fast subset of the CPU mechanism battery (Amendment A2, element 11(a)).

The full battery is a multi-hour run over 2,200 seeded datasets. This module pins the
parts that must not silently change: the seven families' hand-derived truths against
their own generators, the vectorised rho curve against the repository's refit-per-rho
path, ``rho*_point`` against ``breakdown_frontier``, the three outcomes the amendment
requires the battery to assert, and the determinism of every seeded draw.

Everything here runs on CPU in a few seconds. Nothing re-runs a failed check with a
different seed: the seeds are the ones the battery itself uses.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.sensitivity import breakdown_frontier, sensitivity_sweep

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mb = _load("mechanism_battery")
mbs = _load("mechanism_battery_stats")
mbr = _load("mechanism_battery_report")

TRUTH_MC_FAST = 400_000  # enough to resolve an effect to about 0.002


# --------------------------------------------------------------------------- #
# The families and their truths
# --------------------------------------------------------------------------- #
def test_battery_has_eleven_families_and_fifteen_conditions() -> None:
    """Seven original families plus the four added for element 12 part (iv)."""
    conditions = mb.build_conditions()
    assert len(conditions) == 15
    assert {c.family for c in conditions} == set(mb.FAMILY_ORDER)
    assert len({c.key for c in conditions}) == 15
    assert len(mb.PART_IV_FAMILIES) == 4


# --------------------------------------------------------------------------- #
# Element 12 part (iv): the misspecification list
# --------------------------------------------------------------------------- #
PART_IV_ITEMS = (
    "baseline offsets",
    "nonlinear depth response",
    "varying variance",
    "correlated errors",
    "sparse groups",
    "treatment-induced latent states",
    "missingness",
    "near-zero and cancelling effects",
)


def test_every_misspecification_on_the_list_has_a_generator() -> None:
    """The list is section 13's, verbatim, and every entry names real code."""
    # Arrange
    keys = {c.key for c in mb.build_conditions()}

    # Assert
    assert tuple(mbr.PART_IV_LIST) == PART_IV_ITEMS
    for item, entries in mbr.PART_IV_LIST.items():
        assert entries, f"{item} has no generator"
        for condition_key, class_name in entries:
            assert condition_key in keys, f"{item} names an unknown condition {condition_key}"
            assert hasattr(mb, class_name), f"{item} names an unknown class {class_name}"


def test_the_generator_line_table_points_at_real_class_definitions() -> None:
    """The mapping table's file and line are read from the source, not typed in."""
    # Act
    lines = mbr._generator_lines()
    source = (REPO / "experiments" / "mechanism_battery.py").read_text().splitlines()

    # Assert
    for _, entries in mbr.PART_IV_LIST.items():
        for _, class_name in entries:
            assert source[lines[class_name] - 1].startswith(f"class {class_name}(")


def test_part_iv_families_declare_their_own_gate_size() -> None:
    """At least 400 datasets at n = 350 each, which is the pre-registered minimum."""
    for mech in mb.build_conditions():
        if mech.family in mb.PART_IV_FAMILIES:
            assert mech.sample_sizes == (mb.PART_IV_ROWS,)
            assert mech.min_datasets >= 400
        else:
            assert mech.sample_sizes is None and mech.min_datasets is None


def test_missingness_family_drops_the_long_traces_and_keeps_the_full_truth() -> None:
    """Complete-case data, population truth: the loss is measurement, not mechanism."""
    # Arrange
    mech = mb.MediatorMissingness()
    n = 4_000

    # Act
    x, m, y = mech.draw(n, mb.dataset_seed(0, n))
    full = probit_natural_effects_closed_form(
        0.0, mb.RATIONALIZE_BETA, mb.RATIONALIZE_GAMMA, 1.0, 0.0,
        mb.BASELINE_STEPS, mech.alpha0,
    )

    # Assert
    assert len(x) < n, "no rows went missing"
    assert 0.5 * n < len(x) < 0.95 * n, f"retention {len(x) / n:.3f} is not a partial loss"
    assert m.mean() < mb.BASELINE_STEPS + 0.5 * mb.RATIONALIZE_GAMMA, (
        "the surviving traces should be the shorter ones"
    )
    for got, want in zip(mech.analytic_truth, full):
        assert got == pytest.approx(want, abs=1e-12)
    assert len(m) == len(x) and len(y) == len(x)


def test_sparse_groups_share_one_item_effect_across_each_block_of_rows() -> None:
    """The item effect is constant within an item and independent across items."""
    # Arrange
    mech = mb.SparseGroups()
    rng = np.random.default_rng(0)
    n = mb.GROUP_SIZE * 40

    # Act
    z = mech.noise(rng, n)
    per_row = z["u"].reshape(-1, mb.GROUP_SIZE)

    # Assert
    assert np.allclose(per_row.std(axis=1), 0.0), "an item's rows must share one effect"
    assert per_row[:, 0].std() > 0.3, "items must actually differ"


@pytest.mark.parametrize("mech", mb.build_conditions(), ids=lambda m: m.key)
def test_monte_carlo_truth_matches_the_analytic_truth(mech) -> None:
    """Each family's ground truth is derived twice: by hand and from its own equations."""
    # Arrange / Act
    mc = mech.truth(n_mc=TRUTH_MC_FAST, seed=mb.TRUTH_SEED)

    # Assert
    for got, want in zip(mc, mech.analytic_truth):
        assert got == pytest.approx(want, abs=0.004)


def test_the_four_zero_mediation_families_have_exactly_zero_true_nie() -> None:
    """The false-robust-verdict rate is only defined where the truth is exactly zero."""
    # Arrange
    zero_families = {
        "f1_no_cue_effect",
        "f3_shared_cause",
        "f5_redundant_explanation",
        "f6_answer_copying",
    }

    # Act / Assert
    for mech in mb.build_conditions():
        if mech.family in zero_families:
            assert mech.analytic_truth[1] == 0.0
        else:
            assert abs(mech.analytic_truth[1]) > 0.005


def test_opposing_family_cancels_exactly_in_the_total() -> None:
    """Family 7's whole point is a zero total over two large opposite components."""
    # Arrange
    mech = next(c for c in mb.build_conditions() if c.family == "f7_opposing_effects")

    # Act
    nde, nie, te = mech.analytic_truth

    # Assert
    assert te == pytest.approx(0.0, abs=1e-12)
    assert nde == pytest.approx(-nie, abs=1e-12)
    assert abs(nie) > 0.2


def test_draws_are_deterministic_under_the_battery_seeds() -> None:
    """A re-run of the battery must reproduce every dataset bit for bit."""
    # Arrange
    mech = mb.SharedCause()

    # Act
    first = mech.draw(200, mb.DATASET_SEED_BASE)
    second = mech.draw(200, mb.DATASET_SEED_BASE)

    # Assert
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# The rho machinery against the repository's slower paths
# --------------------------------------------------------------------------- #
def test_effects_curve_matches_the_repository_closed_form_pointwise() -> None:
    """The vectorised curve is the closed form at the reparameterised coefficients."""
    # Arrange
    alpha, beta, gamma, sigma_m, mu_m, alpha0 = 0.3, 0.8, 1.0, 1.2, 6.0, -4.6
    grid = np.array([0.0, 0.2, 0.5, 0.8])

    # Act
    nde, nie, te = mb.effects_curve(alpha, beta, gamma, sigma_m, mu_m, alpha0, grid)

    # Assert: rebuild each point through the repository's own function.
    for i, rho in enumerate(grid):
        root = math.sqrt(1.0 - rho**2)
        want = probit_natural_effects_closed_form(
            alpha * root + (rho / sigma_m) * gamma,
            beta * root - rho / sigma_m,
            gamma,
            sigma_m,
            float(rho),
            mu_m,
            alpha0 * root + (rho / sigma_m) * mu_m,
        )
        assert (nde[i], nie[i], te[i]) == pytest.approx(want, abs=1e-12)


def test_effects_curve_matches_sensitivity_sweep_on_real_data() -> None:
    """One fit at rho = 0 reproduces what the repository gets by refitting at each rho."""
    # Arrange
    mech = mb.Rationalization()
    x, m, y = mech.draw(800, mb.DATASET_SEED_BASE)
    grid = np.array([-0.3, 0.0, 0.3])

    # Act
    fit = mb._fit_at_zero(x, m, y)
    _, nie_curve, _ = mb.effects_curve(
        fit.alpha, fit.beta, fit.gamma, fit.sigma_m, fit.mu_m, fit.alpha0, grid
    )
    repo = sensitivity_sweep(x, m, y, rho_grid=grid, n_mc=200_000, rng_seed=0)

    # Assert
    for point, value in zip(repo, nie_curve):
        assert point.nie == pytest.approx(value, abs=0.005)


def test_rho_star_point_matches_breakdown_frontier() -> None:
    """The analytic crossing used by the battery is the one the repository root-finds."""
    # Arrange
    mech = mb.DirectBypass(1.0)
    x, m, y = mech.draw(4_000, mb.DATASET_SEED_BASE)

    # Act
    fit = mb._fit_at_zero(x, m, y)
    predicted = mb.rho_star_point(fit.beta, fit.sigma_m)
    frontier = breakdown_frontier(x, m, y, key="nie", n_mc=100_000, rng_seed=0)

    # Assert
    assert not frontier.unresolved
    assert frontier.robustness == pytest.approx(predicted, abs=0.01)


# --------------------------------------------------------------------------- #
# The three required outcomes, on the mechanisms themselves
# --------------------------------------------------------------------------- #
def test_r1_mediated_share_falls_while_the_crossing_stays_put() -> None:
    """Family 2: a fifteenfold direct bypass moves the share and not rho*_point."""
    # Arrange
    conditions = [c for c in mb.build_conditions() if c.family == "f2_direct_bypass"]
    analytic_rho_star = mb.rho_star_point(mb.BYPASS_BETA, mb.BYPASS_SIGMA_M)

    # Act
    shares = [c.analytic_truth[1] / c.analytic_truth[2] for c in conditions]

    # Assert: the share collapses, the crossing is a constant of beta and sigma_m alone.
    assert shares == sorted(shares, reverse=True)
    assert shares[0] == pytest.approx(1.0, abs=1e-9)
    assert shares[-1] < 0.05
    assert analytic_rho_star == pytest.approx(0.624695, abs=1e-6)


def test_r2_shared_cause_is_strong_association_with_zero_mediation() -> None:
    """Family 3: the estimator reports the whole effect as mediated, and rho* says 0.707."""
    # Arrange
    mech = mb.SharedCause()
    x, m, y = mech.draw(20_000, mb.DATASET_SEED_BASE)

    # Act
    fit = mb._fit_at_zero(x, m, y)
    nde, nie, te = probit_natural_effects_closed_form(
        fit.alpha, fit.beta, fit.gamma, fit.sigma_m, 0.0, fit.mu_m, fit.alpha0
    )
    crossing = mb.rho_star_point(fit.beta, fit.sigma_m)

    # Assert: true NIE is zero, the estimate is not, and the crossing is 1/sqrt(2).
    assert mech.analytic_truth[1] == 0.0
    assert nie == pytest.approx(mech.analytic_truth[0], abs=0.03)
    assert abs(nde) < 0.05
    assert te == pytest.approx(mech.analytic_truth[2], abs=0.03)
    assert crossing == pytest.approx(1.0 / math.sqrt(2.0), abs=0.02)
    assert float(np.corrcoef(m, y)[0, 1]) > 0.4


def test_r3_null_family_intervals_cover_zero() -> None:
    """Family 1: on a world with no effect the reported intervals contain zero.

    Coverage is a rate, not a guarantee: at a nominal 95 percent an occasional interval
    misses. The criterion here is the one the report applies, so a passing test and a
    passing report mean the same thing: the binomial interval on the observed coverage
    must still reach the nominal 0.95.
    """
    # Arrange
    mech = mb.NoCueEffect()
    n_datasets = 10

    # Act
    rows = [mb.analyse_dataset(mech, 350, i, n_bootstrap=60) for i in range(n_datasets)]

    # Assert
    for key in ("nde", "nie", "te"):
        covered = sum(
            getattr(r, f"{key}_lo") <= 0.0 <= getattr(r, f"{key}_hi") for r in rows
        )
        upper = mbs.clopper_pearson_interval(covered, n_datasets)[1]
        assert upper >= mbr.NOMINAL_COVERAGE, (
            f"{key} covered zero in {covered}/{n_datasets}, "
            f"significantly below the nominal {mbr.NOMINAL_COVERAGE}"
        )
    for row in rows:
        assert abs(row.nie) < 0.05
        assert not row.load_bearing_at_zero
        assert row.rho_star_decision is None


def test_analysis_is_deterministic() -> None:
    """Same dataset index, same numbers: a re-run is a re-run, not a new sample."""
    # Arrange / Act
    first = mb.analyse_dataset(mb.AnswerCopying(), 350, 0, n_bootstrap=20)
    second = mb.analyse_dataset(mb.AnswerCopying(), 350, 0, n_bootstrap=20)

    # Assert
    assert first == second


# --------------------------------------------------------------------------- #
# Aggregation and reporting
# --------------------------------------------------------------------------- #
def test_clopper_pearson_interval_edges_and_a_known_value() -> None:
    # Arrange / Act / Assert
    lo, hi = mbs.clopper_pearson_interval(95, 100)
    assert lo == pytest.approx(0.8872, abs=1e-3)
    assert hi == pytest.approx(0.9836, abs=1e-3)
    assert mbs.clopper_pearson_interval(0, 10)[0] == 0.0
    assert mbs.clopper_pearson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        mbs.clopper_pearson_interval(11, 10)


def test_aggregate_prints_every_denominator() -> None:
    """A summary with a rate and no denominator is not reportable under the ground rules."""
    # Arrange
    mech = mb.NoCueEffect()
    rows = [mb.analyse_dataset(mech, 350, i, n_bootstrap=20) for i in range(4)]

    # Act
    agg = mbs.aggregate(mech, rows, mech.truth(n_mc=TRUTH_MC_FAST))

    # Assert
    assert agg["n_datasets"] == 4
    for key in ("nde", "nie", "te"):
        assert agg["effects"][key]["coverage_n"] == 4
    assert agg["verdict"]["load_bearing_n"] == 4
    assert agg["rho_star_decision"]["n"] == 4
    assert agg["verdict"]["false_robust_verdict_rate"] is not None


def test_end_to_end_run_writes_a_report_with_the_three_required_outcomes(tmp_path) -> None:
    """The whole pipeline at toy settings: a report that states each outcome and its numbers."""
    # Arrange
    out = tmp_path / "battery"

    # Act
    code = mb.main(
        [
            "--out", str(out), "--n-datasets", "2", "--n-bootstrap", "8",
            "--n-rows", "350", "--workers", "1", "--crosscheck-datasets", "1",
            "--part-iv-datasets", "2", "--skip-pymc",
        ]
    )
    report = (out / "report.md").read_text()

    # Assert
    assert code == 0
    for name in ("R1 direct bypass", "R2 shared cause", "R3 no cue effect"):
        assert name in report
    assert "Element 12 part (iv): the misspecification list" in report
    for item in PART_IV_ITEMS:
        assert f"| {item} |" in report
    for family in mb.PART_IV_FAMILIES:
        assert family in report
    assert (out / "battery_results.json").exists()
    assert (out / "dataset_rows.csv").exists()
    assert chr(0x2014) not in report and chr(0x2013) not in report


def test_battery_sources_carry_no_long_dashes() -> None:
    """A campaign ground rule, enforced where it is cheapest to enforce."""
    # Arrange
    files = [
        REPO / "experiments" / "mechanism_battery.py",
        REPO / "experiments" / "mechanism_battery_stats.py",
        REPO / "experiments" / "mechanism_battery_report.py",
        Path(__file__),
    ]

    # Act / Assert
    for path in files:
        text = path.read_text()
        assert chr(0x2014) not in text, f"em dash in {path.name}"
        assert chr(0x2013) not in text, f"en dash in {path.name}"
