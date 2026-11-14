"""Fast tests for the anchor-versus-fit synthetic study.

Modules: experiments/anchor_vs_fit_world.py (world, anchor, fit), anchor_vs_fit_real.py
(readers of the stored 72-cell fits) and anchor_vs_fit_sim.py (experiments, CLI). These
tests pin what the study's answers rest on: the truth formulas; that M and M' are the
repo's own curve summary; the realistic depth-0 rules (one shared depth-0 answer, no turn
at depth 0) with the legacy world still reachable; that (X, M, Y) cannot see phi under
shared timing but can under type-dependent timing; the anchor at c = 0 and c = 1; silent
followers; the total-effect filter; the E range and anchor-rejection rules; and the
pooling of the stored falsifier controls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import anchor_vs_fit_real as real  # noqa: E402
import anchor_vs_fit_sim as sim  # noqa: E402
import anchor_vs_fit_world as world  # noqa: E402
from bayes_cot_faithfulness.curves import summarize_curve  # noqa: E402

SEED = 7
UNIFORM_TURN = world.UNIFORM_TURN
SILENT_HALF = world.turn_pmf(world.lateness_pmf(0.0, 1), 0.5)
ALL_SILENT = world.point_mass(world.SILENT_T1)


def _world(seed: int = SEED, **overrides):
    params = world.WorldParams(**overrides)
    uniforms = world.draw_uniforms(seed, params.n_items)
    return params, uniforms, world.simulate_world(params, uniforms)


# --------------------------------------------------------------------------- #
# Truth formulas
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pi,phi", [(0.25, 1.0), (0.25, 0.0), (0.4, 0.3)])
def test_expected_truth_is_pi_split_by_phi(pi, phi):
    truth = world.truth_expected(world.WorldParams(pi=pi, phi=phi))
    assert truth["te"] == pytest.approx(pi)
    assert truth["nde"] == pytest.approx(pi * (1.0 - phi))
    assert truth["nie"] == pytest.approx(pi * phi)


def test_realised_truth_counts_follower_types_and_adds_up():
    _, _, w = _world(n_items=5000, pi=0.3, phi=0.4)
    truth = world.truth_realised(w)
    assert truth["te"] == pytest.approx(w.follower.mean())
    assert truth["nie"] == pytest.approx(w.mediated.mean())
    assert truth["nde"] == pytest.approx((w.follower & ~w.mediated).mean())
    assert truth["nde"] + truth["nie"] == pytest.approx(truth["te"])
    assert truth["te"] == pytest.approx(0.3, abs=0.03)


# --------------------------------------------------------------------------- #
# M and M' are the repo's own curve summary
# --------------------------------------------------------------------------- #
def test_m_and_hint_share_equal_summarize_curve_item_by_item():
    _, _, w = _world(n_items=400, pi=0.4, phi=0.5, h_post=0.7, g_pre=0.4)
    depths = world.curve_depths()
    letters = world.OPTION_LETTERS
    for i in range(w.n_items):
        clean = [letters[a] for a in w.clean_answers[i]]
        hinted = [letters[a] for a in w.hinted_answers[i]]
        final = world.HINT if w.follower[i] else world.GOLD
        assert w.m_clean[i] == summarize_curve(depths, clean, letters[world.GOLD]).curve_area
        assert w.m_hinted[i] == summarize_curve(depths, hinted, letters[final]).curve_area
        hint = letters[world.HINT]
        assert w.share_clean[i] == summarize_curve(depths, clean, hint).curve_area
        assert w.share_hinted[i] == summarize_curve(depths, hinted, hint).curve_area


def test_m_flows_through_summarize_curve(monkeypatch):
    """Swap the repo function for a sentinel: every M must change to the sentinel."""

    class _Sentinel:
        curve_area = 0.123

    monkeypatch.setattr(world, "summarize_curve", lambda depths, answers, final: _Sentinel())
    world.curve_area_table.cache_clear()
    try:
        _, _, w = _world(n_items=50)
        assert np.all(w.m_clean == 0.123)
        assert np.all(w.m_hinted == 0.123)
        assert np.all(w.share_hinted == 0.123)
    finally:
        monkeypatch.undo()
        world.curve_area_table.cache_clear()


def test_curve_depths_come_from_the_repo_grid():
    assert world.curve_depths() == (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# Realistic depth 0 and the legacy flag
# --------------------------------------------------------------------------- #
def test_depth0_answer_is_one_draw_shared_by_both_arms():
    _, _, w = _world(n_items=3000, pi=0.5, g_pre=0.3)
    np.testing.assert_array_equal(w.hinted_answers[:, 0], w.clean_answers[:, 0])
    assert not np.any(w.hinted_answers[:, 0] == world.HINT)
    assert np.all(w.t1[w.follower] >= 1)
    assert w.m_hinted[w.follower].max() <= 0.8


def test_realistic_world_rejects_a_turn_at_depth0():
    with pytest.raises(ValueError, match="depth 0"):
        world.WorldParams(p_t1=world.UNIFORM_TURN_LEGACY)
    with pytest.raises(ValueError, match="depth 0"):
        world.WorldParams(phi=0.0, p_t1_direct=world.point_mass(0))


def test_legacy_flag_restores_independent_depth0_and_turns_at_0():
    _, _, w = _world(n_items=3000, pi=0.5, g_pre=0.3, shared_depth0=False,
                     p_t1=world.UNIFORM_TURN_LEGACY)
    assert not np.array_equal(w.hinted_answers[:, 0], w.clean_answers[:, 0])
    assert np.any(w.t1 == 0)


def test_uniform_turn_covers_only_allowed_depths():
    assert UNIFORM_TURN == pytest.approx((0.0, 0.25, 0.25, 0.25, 0.25, 0.0))
    assert sim.StudyConfig().uniform_turn == UNIFORM_TURN
    assert sim.StudyConfig(shared_depth0=False).uniform_turn == world.UNIFORM_TURN_LEGACY
    assert all(pmf[0] == 0 for _, pmf in sim.a_turn_path(1))
    assert len(sim.a_turn_path(1)) == len(sim.A_TILTS) + 4


# --------------------------------------------------------------------------- #
# phi-invariance of the observables, and type-dependent timing (F)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mediator", world.MEDIATORS)
def test_observables_identical_across_phi_under_shared_timing(mediator):
    params = world.WorldParams(p_t1=UNIFORM_TURN, p_t1_direct=UNIFORM_TURN)
    world.assert_observables_phi_invariant(params, (0.0, 0.3, 0.75, 1.0), SEED, mediator)


def test_observables_move_with_phi_when_only_direct_followers_can_be_silent():
    _, _, w0 = _world(phi=0.0, p_t1=SILENT_HALF)
    _, _, w1 = _world(phi=1.0, p_t1=SILENT_HALF)
    np.testing.assert_array_equal(w0.Y, w1.Y)
    assert not np.array_equal(w0.M, w1.M)
    with pytest.raises(AssertionError):
        world.assert_observables_phi_invariant(world.WorldParams(p_t1=SILENT_HALF),
                                               (0.0, 1.0), SEED)


def test_f_scenarios_set_type_dependent_turns():
    cfg = sim.StudyConfig()
    assert sim.f_turns(cfg, "F1") == (UNIFORM_TURN, ALL_SILENT)
    assert sim.f_turns(cfg, "F2") == (world.point_mass(1), world.point_mass(4))
    assert sim.f_turns(cfg, "F3") == (world.point_mass(4), world.point_mass(1))
    with pytest.raises(ValueError):
        sim.f_turns(cfg, "F9")


def test_f1_followers_are_silent_at_phi0_and_turn_at_phi1():
    base = sim._f_base(sim.StudyConfig(n_items=3000), "F1")
    u = world.draw_uniforms(SEED, 3000)
    w0 = world.simulate_world(world.with_changes(base, phi=0.0), u)
    w1 = world.simulate_world(world.with_changes(base, phi=1.0), u)
    assert np.all(w0.t1[w0.follower] == world.SILENT_T1)
    assert np.all(w0.m_hinted[w0.follower] == 0.0)
    assert np.all(w1.t1[w1.follower] < world.SILENT_T1)
    for scenario in ("F1", "F2", "F3"):
        with pytest.raises(AssertionError):
            world.assert_observables_phi_invariant(
                world.with_changes(sim._f_base(sim.StudyConfig(n_items=500), scenario)),
                (0.0, 1.0), SEED)


def test_experiment_f_rows_carry_truth_anchor_and_filter():
    blocks = sim.experiment_f(sim.StudyConfig(n_items=400, n_seeds=2))
    assert [b["scenario"] for b in blocks] == ["F1", "F2", "F3"]
    for b in blocks:
        assert [r["phi"] for r in b["rows"]] == list(sim.PHI_GRID)
        for r in b["rows"]:
            assert r["truth_expected"]["nie"] == pytest.approx(0.25 * r["phi"])
            assert "passes" in r["te_filter"]
            assert r["summary"]["anchor_nie"]["mean"] is not None


# --------------------------------------------------------------------------- #
# The directional mediator M' (G)
# --------------------------------------------------------------------------- #
def test_hint_share_is_zero_on_the_clean_arm_and_matches_followers_m():
    _, _, w = _world(n_items=3000, pi=0.4, h_post=0.7)
    assert np.all(w.share_clean == 0.0)
    np.testing.assert_array_equal(w.share_hinted[w.follower], w.m_hinted[w.follower])
    assert np.all(w.share_hinted[~w.follower] == 0.0)
    X, M, Y = world.observables(w, "hint_share")
    np.testing.assert_array_equal(M[0::2], w.share_clean)
    np.testing.assert_array_equal(M[1::2], w.share_hinted)
    with pytest.raises(ValueError):
        world.observables(w, "nope")


def test_hint_share_fit_records_mediator_and_counts():
    row = world.run_condition(world.WorldParams(n_items=500), seeds=(1, 2), do_fit=True,
                              mediator="hint_share")
    assert row["mediator"] == "hint_share"
    assert row["n_fit_failures"] + row["summary"]["fit_te"]["n"] == 2


# --------------------------------------------------------------------------- #
# The anchor
# --------------------------------------------------------------------------- #
def test_anchor_equals_realised_truth_without_copying_at_full_h_post():
    params, uniforms, w = _world(n_items=4000, phi=0.5, h_post=1.0, copy=0.0)
    anchor = world.anchor_rates(w, params, uniforms)
    truth = world.truth_realised(w)
    assert anchor["mu00"] == 0.0
    for key in ("nde", "nie", "te"):
        assert anchor[key] == pytest.approx(truth[key])


def test_anchor_recovers_expected_truth_within_sampling_error():
    n = 20_000
    params, uniforms, w = _world(n_items=n, phi=0.6, h_post=1.0, copy=0.0)
    anchor = world.anchor_rates(w, params, uniforms)
    truth = world.truth_expected(params)
    for key in ("nde", "nie", "te"):
        se = np.sqrt(truth[key] * (1 - truth[key]) / n)
        assert abs(anchor[key] - truth[key]) < 4 * se, key


def test_anchor_with_full_copying_moves_effect_from_nde_to_nie():
    params, uniforms, w = _world(n_items=4000, phi=0.4, h_post=1.0, copy=1.0)
    anchor = world.anchor_rates(w, params, uniforms)
    truth = world.truth_realised(w)
    assert anchor["nde"] == 0.0
    assert anchor["nie"] == pytest.approx(truth["te"])
    assert anchor["nie"] > truth["nie"] + 0.05
    assert anchor["nde"] < truth["nde"] - 0.05


def test_anchor_mu01_is_zero_for_silent_followers_without_copying():
    params, uniforms, w = _world(phi=0.0, p_t1_direct=ALL_SILENT, copy=0.0)
    anchor = world.anchor_rates(w, params, uniforms)
    assert anchor["mu01"] == 0.0
    assert anchor["mu10"] == pytest.approx(w.follower.mean())
    assert anchor["nie"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Silent followers and the type rule
# --------------------------------------------------------------------------- #
def test_silent_followers_have_zero_hinted_curve_area():
    _, _, w = _world(phi=0.0, p_t1_direct=ALL_SILENT)
    assert w.follower.sum() > 0
    assert np.all(w.t1[w.follower] == world.SILENT_T1)
    assert np.all(w.m_hinted[w.follower] == 0.0)


def test_only_direct_followers_are_silent():
    _, _, w = _world(n_items=4000, pi=0.5, phi=0.5, p_t1=SILENT_HALF)
    silent = w.follower & (w.t1 == world.SILENT_T1)
    assert silent.sum() > 0
    assert not np.any(w.mediated & silent)


def test_mediated_followers_need_a_non_silent_turn():
    with pytest.raises(ValueError):
        world.WorldParams(phi=0.5, p_t1=ALL_SILENT)


@pytest.mark.parametrize(
    "bad",
    [{"pi": 1.5}, {"phi": -0.1}, {"copy": 2.0}, {"p_t0": (0.5, 0.5)},
     {"p_t1": (0.0, 0.2, 0.2, 0.2, 0.3, 0.2)}],
)
def test_invalid_parameters_raise(bad):
    with pytest.raises(ValueError):
        world.WorldParams(**bad)


# --------------------------------------------------------------------------- #
# The total-effect filter
# --------------------------------------------------------------------------- #
def test_te_filter_passes_on_the_mean_gap_and_counts_seeds():
    ok = world.te_filter([0.05, -0.02, 0.07])
    assert ok["passes"] and ok["n_seeds_above"] == 1
    assert ok["mean_gap"] == pytest.approx(0.1 / 3)
    bad = world.te_filter([-0.1, -0.05])
    assert not bad["passes"]
    empty = world.te_filter([])
    assert not empty["passes"] and empty["mean_gap"] is None


def test_te_gap_is_fit_te_minus_the_randomized_arm_difference():
    row = world.run_condition(world.WorldParams(n_items=800), seeds=(3, 4), do_fit=True)
    s = row["summary"]
    assert row["n_fit_failures"] == 0
    assert s["fit_te_gap"]["mean"] == pytest.approx(
        s["fit_te"]["mean"] - s["truth_real_te"]["mean"])
    assert row["te_filter"]["mean_gap"] == pytest.approx(s["fit_te_gap"]["mean"])


def test_fit_failures_are_counted_not_averaged(monkeypatch):
    real_fit = world.fit_probit_mediation_map

    def _unconverged(*args, **kwargs):
        return real_fit(*args, **kwargs)._replace(converged=False)

    monkeypatch.setattr(world, "fit_probit_mediation_map", _unconverged)
    row = world.run_condition(world.WorldParams(n_items=500), seeds=(1, 2), do_fit=True)
    assert row["n_fit_failures"] == 2
    assert row["summary"]["fit_nie"]["mean"] is None
    assert row["summary"]["anchor_nie"]["mean"] is not None
    assert not row["te_filter"]["passes"]


def test_run_condition_is_deterministic():
    params = world.WorldParams(n_items=600)
    assert world.run_condition(params, (3, 4), True) == world.run_condition(params, (3, 4), True)


# --------------------------------------------------------------------------- #
# Exact moments and calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shared", [True, False])
def test_exact_moments_match_simulation(shared):
    first = world.first_turn_depth(shared)
    params, _, w = _world(
        n_items=40_000, pi=0.3, phi=0.0, p_t0=world.lateness_pmf(0.3), g_pre=0.35,
        p_t1=world.turn_pmf(world.lateness_pmf(0.0, first), 0.0),
        p_t1_direct=world.turn_pmf(world.lateness_pmf(0.4, first), 0.4), h_post=0.7,
        shared_depth0=shared,
    )
    exact = world.exact_m_moments(params)
    assert w.m_clean.mean() == pytest.approx(exact["clean_mean"], abs=0.006)
    assert w.m_clean.std() == pytest.approx(exact["clean_sd"], abs=0.006)
    assert w.m_hinted.mean() == pytest.approx(exact["hinted_mean"], abs=0.006)
    assert w.m_hinted.std() == pytest.approx(exact["hinted_sd"], abs=0.006)


@pytest.mark.parametrize("family", world.LATENESS_FAMILIES, ids=lambda f: f.name)
def test_calibration_hits_moments_of_a_known_world(family):
    lat0, g, lat1, h = (0.3, 0.4, 0.6, 0.7) if family.name == "endpoint" else (-0.7, 0.4,
                                                                               0.5, 0.7)
    known = world.calibrated_params(family, 1, 0.3, 1.0, lat0, g, lat1, h, 0.0)
    target = world.exact_m_moments(known)
    clean = world.calibrate_clean(target, family)
    hinted = world.calibrate_hinted(target, clean, family, 0.3, 1.0, 0.0)
    found = world.calibrated_params(family, 1, 0.3, 1.0, clean["lateness0"], clean["g_pre"],
                                    hinted["lateness1"], hinted["h_post"], 0.0)
    got = world.exact_m_moments(found)
    for key in world.MOMENT_KEYS:
        assert got[key] == pytest.approx(target[key], abs=1e-3), key


def test_lateness_and_endpoint_pmfs():
    assert np.argmax(world.lateness_pmf(-6.0)) == 0
    assert np.argmax(world.lateness_pmf(-6.0, 1)) == 1
    assert np.argmax(world.lateness_pmf(6.0)) == 4
    np.testing.assert_allclose(world.lateness_pmf(0.0), [0.2] * 5)
    assert world.endpoint_pmf(0.3) == pytest.approx((0.7, 0.0, 0.0, 0.0, 0.3))
    assert world.endpoint_pmf(0.3, 1) == pytest.approx((0.0, 0.7, 0.0, 0.0, 0.3))
    with pytest.raises(ValueError):
        world.endpoint_pmf(1.2)


# --------------------------------------------------------------------------- #
# E ranges and anchor rejection
# --------------------------------------------------------------------------- #
def _fake_world(name, phi, nie, err, passes):
    stat = lambda v: {"mean": v, "lo": v, "hi": v, "n": 1}  # noqa: E731
    return {"world": name, "phi": phi, "max_abs_moment_error": err,
            "te_filter": {"passes": passes},
            "summary": {"fit_nde": stat(0.1), "fit_nie": stat(nie), "fit_te": stat(0.2)}}


def test_e_ranges_follow_one_rule_and_restrict_on_error_and_filter():
    worlds = [_fake_world("a", 1.0, 0.10, 0.01, True), _fake_world("b", 1.0, 0.20, 0.05, True),
              _fake_world("c", 0.0, 0.05, 0.01, True), _fake_world("d", 0.0, 0.30, 0.01, False),
              _fake_world("e", 0.0, 0.15, 0.02, True)]
    r = sim.e_ranges(worlds)
    assert r["full_mediation"]["all"]["nie"] == {"min": 0.10, "max": 0.20, "n_worlds": 2}
    assert r["full_mediation"]["restricted_worlds"] == ["a"]
    assert r["no_text_mediation"]["all"]["nie"]["max"] == 0.30
    assert r["no_text_mediation"]["restricted"]["nie"] == {"min": 0.05, "max": 0.15,
                                                          "n_worlds": 2}


def test_anchor_rejection_uses_the_margin_on_every_cell():
    summary = {f"anchor_{c}": {"mean": v} for c, v in
               zip(world.ANCHOR_CELLS, (0.0, 0.10, 0.25, 0.25))}
    out = sim.anchor_rejection(summary, {"mu00": 0.0, "mu01": 0.19, "mu10": 0.015,
                                         "mu11": 0.22})
    assert out["rejected"] and out["rejected_by"] == ["mu10"]
    assert out["differences"]["mu01"] == pytest.approx(-0.09)


# --------------------------------------------------------------------------- #
# The stored-controls reader (H)
# --------------------------------------------------------------------------- #
def _cell(k01, k11, controls, n=100, follow=0.3):
    cells = {c: {"k": k, "n": n} for c, k in zip(("mu00", "mu01", "mu10", "mu11"),
                                                 (0, k01, 5, k11))}
    return {"cell": "x", "column_b": {"n_items": n},
            "table": {"outcome": {"hinted_arm_mean": follow}},
            "anchor": {"cells": cells, "falsifier_controls": controls}}


def _ctl(n_applied, n_items, k0, k1):
    return {"n_applied": n_applied, "n_items": n_items,
            "a0": {"k": k0, "n": n_applied}, "a1": {"k": k1, "n": n_applied}}


def _write(root: Path, model: str, cell: dict) -> None:
    path = root / model / "arc_challenge" / "cue" / "fit.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cell))


def test_copy_controls_pool_counts_and_keep_mu_on_the_same_cells(tmp_path):
    full = {name: _ctl(90, 100, 10, 20) for name in real.COPY_CONTROLS}
    partial = {name: _ctl(50, 100, 4, 6) for name in real.COPY_CONTROLS
               if name != "decisive_premise_edit"}
    _write(tmp_path, "m1", _cell(25, 28, full))
    _write(tmp_path, "m2", _cell(15, 18, partial, follow=0.2))
    out = real.read_copy_controls(tmp_path, "arc_challenge")
    assert out["follow"] == {"k": 50, "n": 200, "rate": 0.25}
    amr = out["controls"]["answer_marker_removed"]
    assert amr["a0"] == {"k": 14, "n": 140, "rate": 0.1}
    assert amr["mu01_same_cells"]["rate"] == pytest.approx(40 / 200)
    dpe = out["controls"]["decisive_premise_edit"]
    assert dpe["n_cells"] == 1 and dpe["n_cells_missing"] == 1
    assert dpe["mu01_same_cells"] == {"k": 25, "n": 100, "rate": 0.25}
    bounds = out["copy_bounds"]
    assert bounds["marker_gap_no_cue"] == pytest.approx(0.2 - 0.1)
    implied = bounds["implied_copy_rate_no_cue"]
    assert implied["applies"]
    assert implied["value"] == pytest.approx((0.2 - 0.1) / (0.25 - 0.1))
    assert out["subset_restriction_possible"] is False


def test_implied_copy_rate_refuses_when_mu_exceeds_the_follow_rate():
    assert not real.implied_copy_rate(0.255, 0.142, 0.159)["applies"]
    assert not real.implied_copy_rate(0.10, 0.20, 0.15)["applies"]
    assert real.implied_copy_rate(0.19, 0.12, 0.25)["value"] == pytest.approx(0.07 / 0.13)


def test_constant_mediator_is_flagged_and_its_failures_counted():
    base = sim._f_base(sim.StudyConfig(n_items=400), "F1")
    row = world.run_condition(world.with_changes(base, phi=0.0), (1, 2), True, "hint_share")
    assert row["n_seeds_mediator_constant"] == 2
    assert row["n_fit_failures"] + row["summary"]["fit_te"]["n"] == 2
    assert sum(row["warning_categories"].values()) == (row["n_optimizer_warnings"]
                                                       + row["n_other_warnings"])


def test_copy_controls_read_the_real_72_cell_tree():
    out = real.read_copy_controls(real.FITS_ROOT, "arc_challenge")
    assert out["n_cells"] == 24
    for c in out["controls"].values():
        assert c["n_cells_missing"] == 0
        assert 0 < c["n_applied"] <= c["n_items"]
    assert out["follow"]["rate"] == pytest.approx(0.254, abs=0.001)
