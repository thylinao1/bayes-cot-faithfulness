"""Element 11(f): the difference recovers what was planted, and each clause can fail.

Three things are worth testing about a statistic like this and only three:

* it recovers a planted organism-minus-twin difference, and reports EXACTLY zero when
  the two sides are the same fit (a statistic that cannot report zero cannot report a
  null rung);
* sd_pilot(D) and the MDE are the A3.2 formula on the four values ruling R6 names, and
  the function refuses when it is handed a different number of values;
* each of the three pre-registered clauses is falsifiable, shown by an input that fails
  it while the other two pass.
"""

from __future__ import annotations

import math
import random

import pytest

from bayes_cot_faithfulness.ladder import statistic as st
from bayes_cot_faithfulness.ladder.spec import MDE_Z, SD_PILOT_RUNGS

B = 400


def _fit(variant, rung, dose, seed, nde, nie, te, *, sd=0.02, rng_seed=0):
    rng = random.Random(rng_seed)
    return st.CheckpointFit(
        cell_id=f"{variant}_{dose:.2f}_{seed}", variant=variant, rung=rung, dose=dose,
        seed=seed,
        nde_draws=[rng.gauss(nde, sd) for _ in range(B)],
        nie_draws=[rng.gauss(nie, sd) for _ in range(B)],
        te_draws=[rng.gauss(te, sd) for _ in range(B)],
    )


def _ladder(organism_nde=(0.20, 0.35), organism_nie=(0.20, 0.10),
            twin_nde=(0.05, 0.05), twin_nie=(0.30, 0.30)):
    """Two organism rungs and their twins, both training seeds, as element 11(b) has it."""
    fits = []
    for k, (rung, dose) in enumerate(((2, 0.60), (3, 0.90))):
        for j, seed in enumerate((20260911, 20260923)):
            fits.append(_fit("organism", rung, dose, seed, organism_nde[k],
                             organism_nie[k], organism_nde[k] + organism_nie[k],
                             rng_seed=10 * k + j))
            fits.append(_fit("twin", rung, dose, seed, twin_nde[k], twin_nie[k],
                             twin_nde[k] + twin_nie[k], rng_seed=100 + 10 * k + j))
    return fits


def test_the_difference_recovers_a_planted_organism_minus_twin_gap():
    planted = 0.25
    org = _fit("organism", 3, 0.90, 1, 0.30, 0.10, 0.40, rng_seed=1)
    twin = _fit("twin", 3, 0.90, 1, 0.30 - planted, 0.10, 0.40 - planted, rng_seed=2)
    d = st.difference(org, twin)
    assert d["nde"]["D"]["point"] == pytest.approx(planted, abs=0.01)
    assert d["nde"]["D"]["lo"] < planted < d["nde"]["D"]["hi"]
    assert d["mediated_share"]["D"] is not None


def test_the_difference_is_exactly_zero_on_identical_inputs():
    org = _fit("organism", 2, 0.60, 1, 0.20, 0.20, 0.40, rng_seed=3)
    twin = st.CheckpointFit(
        cell_id="twin_0.60_1", variant="twin", rung=2, dose=0.60, seed=1,
        nde_draws=list(org.nde_draws), nie_draws=list(org.nie_draws),
        te_draws=list(org.te_draws))
    d = st.difference(org, twin)
    assert d["nde"]["D"]["point"] == 0.0
    assert (d["nde"]["D"]["lo"], d["nde"]["D"]["hi"]) == (0.0, 0.0)
    assert d["mediated_share"]["D"]["point"] == 0.0


def test_sd_pilot_is_four_values_at_rungs_two_and_three_and_the_mde_is_the_formula():
    report = st.build_report(_ladder())
    sp = st.sd_pilot(report, "nde")
    assert sp["rungs"] == list(SD_PILOT_RUNGS) == [2, 3]
    assert sp["n_values"] == 4          # 2 rungs x 2 training seeds, ruling R6
    m = st.mde(sp["sd"])
    assert m["mde"] == pytest.approx(MDE_Z * math.sqrt(2.0) * sp["sd"])
    assert m["ladder_n_items_per_checkpoint"] == 500
    assert "A3.2" in m["source"]


def test_sd_pilot_refuses_a_report_that_is_missing_a_seed():
    fits = [f for f in _ladder() if not (f.rung == 3 and f.seed == 20260923)]
    report = st.build_report(fits)
    with pytest.raises(st.LadderStatisticError) as exc:
        st.sd_pilot(report)
    assert "4 organism-minus-twin differences" in str(exc.value)
    assert "1 seed(s), not 2" in str(exc.value)


def test_a_rung_with_no_twin_is_refused_rather_than_averaged():
    fits = [f for f in _ladder() if not (f.variant == "twin" and f.rung == 3
                                         and f.seed == 20260911)]
    with pytest.raises(st.LadderStatisticError) as exc:
        st.build_report(fits)
    assert "no twin-side fit" in str(exc.value)


def test_all_three_clauses_pass_on_the_pre_registered_shape():
    report = st.build_report(_ladder())
    test = st.prediction_test(report)
    assert test["n_clauses"] == 3
    assert test["all_passed"] is True
    assert [c["clause"] for c in test["clauses"]] == [
        "nde_rises_with_dose", "mediated_share_falls", "twins_flat"]


def test_clause_one_fails_on_an_organism_whose_nde_does_not_rise():
    """NDE flat across the doses: clause 1 fails, and only clause 1."""
    report = st.build_report(_ladder(organism_nde=(0.30, 0.30)))
    clauses = {c["clause"]: c for c in st.prediction_test(report)["clauses"]}
    assert clauses["nde_rises_with_dose"]["passed"] is False
    assert clauses["mediated_share_falls"]["passed"] is True
    assert clauses["twins_flat"]["passed"] is True


def test_clause_two_fails_on_a_mediated_share_that_rises():
    """The organism's mediated share going UP with the dose falsifies clause 2 alone."""
    report = st.build_report(_ladder(organism_nie=(0.05, 0.40)))
    clauses = {c["clause"]: c for c in st.prediction_test(report)["clauses"]}
    assert clauses["mediated_share_falls"]["passed"] is False
    assert clauses["nde_rises_with_dose"]["passed"] is True
    assert clauses["twins_flat"]["passed"] is True


def test_clause_three_fails_on_twins_that_move():
    """A twin whose NDE moves with the rung falsifies clause 3 alone."""
    report = st.build_report(_ladder(twin_nde=(0.05, 0.55)))
    clauses = {c["clause"]: c for c in st.prediction_test(report)["clauses"]}
    assert clauses["twins_flat"]["passed"] is False
    assert clauses["twins_flat"]["spread"] > clauses["twins_flat"]["tolerance"]
    assert clauses["nde_rises_with_dose"]["passed"] is True
    assert clauses["mediated_share_falls"]["passed"] is True


def test_rho_star_is_reported_with_an_interval_and_carries_no_directional_test():
    rng = random.Random(0)
    fits = []
    for f in _ladder():
        fits.append(st.CheckpointFit(
            cell_id=f.cell_id, variant=f.variant, rung=f.rung, dose=f.dose, seed=f.seed,
            nde_draws=f.nde_draws, nie_draws=f.nie_draws, te_draws=f.te_draws,
            rho_star_draws=[rng.gauss(0.62, 0.03) for _ in range(B)]))
    report = st.build_report(fits)
    assert report.rho_star["directional_prediction"] is None
    row = report.rho_star["rows"][0]
    assert row["rho_star"]["lo"] < row["rho_star"]["point"] < row["rho_star"]["hi"]
    with pytest.raises(st.LadderStatisticError) as exc:
        st.assert_no_directional_test("does rho* rise with the dose")
    assert "no directional prediction" in str(exc.value)


def test_a_mediated_share_with_te_at_zero_is_reported_undefined_not_imputed():
    org = _fit("organism", 3, 0.90, 1, 0.001, 0.001, 0.002, sd=0.0005, rng_seed=5)
    twin = _fit("twin", 3, 0.90, 1, 0.001, 0.001, 0.002, sd=0.0005, rng_seed=6)
    d = st.difference(org, twin)
    assert d["mediated_share"]["D"] is None
    assert "undefined" in d["mediated_share"]["why_none"]
    assert d["mediated_share"]["n_draws_dropped_organism"] == B


def test_two_fits_from_different_sources_are_not_subtracted():
    org = _fit("organism", 2, 0.60, 1, 0.2, 0.2, 0.4)
    twin = st.CheckpointFit(
        cell_id="twin_0.60_1", variant="twin", rung=2, dose=0.60, seed=1,
        nde_draws=[0.1] * 10, nie_draws=[0.1] * 10, te_draws=[0.2] * 10,
        source="posterior")
    with pytest.raises(st.LadderStatisticError) as exc:
        st.difference(org, twin)
    assert "no stated meaning" in str(exc.value)
