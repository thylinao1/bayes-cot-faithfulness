"""Anchor versus fit: do the two instruments measure the same mediator? A synthetic study.

On the 72 real cells the probit fit's NIE sits below the replay anchor's NIE in 67, and the
fit's "mediated effect" comes from hinted-arm followers having LOWER curve area. The
hypothesis: the anchor's mediator is the whole chain text, while the fit's mediator,
``curve_area``, records how early the truncated chain agrees with the arm's own final
answer and carries no direction toward the hint. The world, the anchor and the fit live in
``anchor_vs_fit_world.py``; the readers of the stored real fits in ``anchor_vs_fit_real.py``.

Experiments (20 seeds each; every quantity is the mean and 2.5/97.5 percentiles over seeds)
  A  timing sweep at phi = 1          E  worlds calibrated to the real M moments
  B  truth sweep, shared timing       F  type-dependent timing (F1, F2, F3)
  C  answer copying (anchor only)     G  the directional mediator M' (hinted share)
  D  silent followers at phi = 0      H  copy bounds from the stored controls (no simulation)
Every fitted row carries the total-effect filter (|fit TE - arm difference| <= 0.06).
The default world is the realistic one (shared depth-0 answer, t1 >= 1); ``--legacy-depth0``
reruns everything in the earlier world.

Run:  PYTHONPATH=src python experiments/anchor_vs_fit_sim.py [--experiments ABCDEFGH]
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from anchor_vs_fit_real import (
    FITS_ROOT,
    REAL_SUBSTRATES,
    read_copy_controls,
    read_substrate_targets,
)
from anchor_vs_fit_world import (
    ANCHOR_CELLS,
    DEFAULT_PI,
    DEFAULT_SHARED_DEPTH0,
    FIT_KEYS,
    LATENESS_FAMILIES,
    MOMENT_KEYS,
    N_DEPTHS,
    SILENT_T1,
    TE_GAP_THRESHOLD,
    WorldParams,
    assert_observables_phi_invariant,
    calibrate_clean,
    calibrate_hinted,
    calibrated_params,
    exact_m_moments,
    first_turn_depth,
    lateness_pmf,
    point_mass,
    run_condition,
    turn_pmf,
    with_changes,
)
from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.curves import depths_for, summarize_curve
from bayes_cot_faithfulness.sensitivity import fit_probit_mediation_map

_HERE = Path(__file__).resolve().parent

DEFAULT_N_SEEDS = 20
DEFAULT_BASE_SEED = 20261114
RESULTS_PATH = _HERE / "results" / "anchor_vs_fit_sim" / "results.json"
LEGACY_RESULTS_PATH = _HERE / "results" / "anchor_vs_fit_sim" / "results_legacy_depth0.json"

A_PI = 0.25
A_TILTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
A_H_POST_VALUES = (1.0, 0.8)
PHI_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
C_PHI_GRID = (0.0, 0.5, 1.0)
C_COPY_GRID = (0.0, 0.5, 1.0)
C_H_POST_VALUES = (1.0, 0.8)
D_SILENT_GRID = (0.0, 0.5, 1.0)
E_SILENT_GRID = (0.0, 0.25, 0.5, 0.75)
E_MOMENT_ERROR_MAX = 0.03
# prereg section 22.1 agreement margin (experiments/wave1_fits.py ANCHOR_MARGIN)
ANCHOR_REJECT_MARGIN = 0.10
F_SCENARIOS = {
    "F1": "mediated followers turn uniformly over the allowed depths; direct never turn",
    "F2": "mediated followers turn at depth 1; direct at depth 4",
    "F3": "mediated followers turn at depth 4; direct at depth 1",
}
F_DEPTH_EARLY, F_DEPTH_LATE = 1, N_DEPTHS - 1

RULES = {
    "te_filter": f"a fitted row passes iff abs(mean over converged seeds of fit TE minus "
                 f"the simulated randomized arm difference) <= {TE_GAP_THRESHOLD}; on the "
                 "72 real cells the largest gap is 0.061 and the median 0.008",
    "e_ranges": "full mediation = every phi = 1 world (both lateness families); no text "
                "mediation = every phi = 0 world (both families x silent mass 0, .25, .5, "
                ".75); a range is [min, max] of the worlds' seed-mean fit estimate; the "
                f"restricted range keeps worlds whose largest moment error is below "
                f"{E_MOMENT_ERROR_MAX} and that pass the total-effect filter",
    "anchor_rejection": f"the real pooled anchor rejects a calibrated world when any of "
                        f"mu00, mu01, mu10, mu11 differs from the world's seed mean by more "
                        f"than {ANCHOR_REJECT_MARGIN} (the prereg 22.1 agreement margin)",
}


@dataclass(frozen=True)
class StudyConfig:
    n_items: int = 4000
    n_seeds: int = DEFAULT_N_SEEDS
    base_seed: int = DEFAULT_BASE_SEED
    shared_depth0: bool = DEFAULT_SHARED_DEPTH0

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self.base_seed + k for k in range(self.n_seeds))

    @property
    def first(self) -> int:
        return first_turn_depth(self.shared_depth0)

    @property
    def uniform_turn(self) -> tuple[float, ...]:
        return turn_pmf(lateness_pmf(0.0, self.first), 0.0)

    def params(self, **kw) -> WorldParams:
        return WorldParams(n_items=self.n_items, shared_depth0=self.shared_depth0, **kw)


def _mean_t1(pmf) -> float:
    return float(np.dot(pmf, np.arange(N_DEPTHS + 1)))


# --------------------------------------------------------------------------- #
# A to D
# --------------------------------------------------------------------------- #
def a_turn_path(first: int) -> list[tuple[str, tuple[float, ...]]]:
    """Tilts over the allowed depths plus a point mass at each allowed depth."""
    path = [(f"tilt {t:+.1f}" if t else "uniform", turn_pmf(lateness_pmf(t, first), 0.0))
            for t in A_TILTS]
    path += [(f"all at {k}", point_mass(k)) for k in range(first, N_DEPTHS)]
    return sorted(path, key=lambda item: _mean_t1(item[1]))


def experiment_a(cfg: StudyConfig) -> list[dict]:
    rows = []
    for h_post in A_H_POST_VALUES:
        for label, pmf in a_turn_path(cfg.first):
            p = cfg.params(pi=A_PI, phi=1.0, p_t1=pmf, h_post=h_post)
            rows.append({**run_condition(p, cfg.seeds, do_fit=True), "turn": label,
                         "mean_t1": _mean_t1(pmf)})
    return rows


def _phi_rows(base: WorldParams, cfg: StudyConfig, mediator: str = "curve_area") -> list:
    return [{**run_condition(with_changes(base, phi=phi), cfg.seeds, True, mediator),
             "phi": phi} for phi in PHI_GRID]


def _fit_identical(rows: list[dict]) -> bool:
    fits = [{k: r["summary"][k] for k in FIT_KEYS} for r in rows]
    return all(f == fits[0] for f in fits)


def experiment_b(cfg: StudyConfig) -> dict:
    base = cfg.params(pi=DEFAULT_PI, p_t1=cfg.uniform_turn, p_t1_direct=cfg.uniform_turn)
    for seed in cfg.seeds:
        assert_observables_phi_invariant(base, PHI_GRID, seed)
    rows = _phi_rows(base, cfg)
    return {"observables_identical_across_phi": True, "checked_seeds": list(cfg.seeds),
            "fit_identical_across_phi": _fit_identical(rows), "rows": rows}


def experiment_c(cfg: StudyConfig) -> list[dict]:
    rows = []
    for h_post in C_H_POST_VALUES:
        for phi in C_PHI_GRID:
            for copy in C_COPY_GRID:
                p = cfg.params(pi=DEFAULT_PI, phi=phi, p_t1=cfg.uniform_turn,
                               h_post=h_post, copy=copy)
                rows.append({**run_condition(p, cfg.seeds, do_fit=False), "phi": phi,
                             "copy": copy, "h_post": h_post})
    return rows


def experiment_d(cfg: StudyConfig) -> list[dict]:
    rows = []
    for silent in D_SILENT_GRID:
        p = cfg.params(pi=DEFAULT_PI, phi=0.0, p_t1=cfg.uniform_turn,
                       p_t1_direct=turn_pmf(lateness_pmf(0.0, cfg.first), silent))
        rows.append({**run_condition(p, cfg.seeds, do_fit=True), "silent": silent})
    return rows


# --------------------------------------------------------------------------- #
# E
# --------------------------------------------------------------------------- #
def anchor_rejection(summary: dict, real_anchor: dict) -> dict:
    diffs = {c: summary[f"anchor_{c}"]["mean"] - real_anchor[c] for c in ANCHOR_CELLS}
    by = [c for c, d in diffs.items() if abs(d) > ANCHOR_REJECT_MARGIN]
    return {"margin": ANCHOR_REJECT_MARGIN, "differences": diffs, "rejected": bool(by),
            "rejected_by": by}


def _range(worlds: list[dict], key: str) -> dict | None:
    vals = [w["summary"][key]["mean"] for w in worlds if w["summary"][key]["mean"] is not None]
    if not vals:
        return None
    return {"min": min(vals), "max": max(vals), "n_worlds": len(vals)}


def e_ranges(worlds: list[dict]) -> dict:
    """The E ranges by the one rule in RULES['e_ranges'], for one question set."""
    out = {}
    for label, phi in (("full_mediation", 1.0), ("no_text_mediation", 0.0)):
        sel = [w for w in worlds if w["phi"] == phi]
        kept = [w for w in sel if w["max_abs_moment_error"] < E_MOMENT_ERROR_MAX
                and w["te_filter"]["passes"]]
        out[label] = {
            "all": {k: _range(sel, f"fit_{k}") for k in ("nde", "nie", "te")},
            "restricted": {k: _range(kept, f"fit_{k}") for k in ("nde", "nie", "te")},
            "restricted_worlds": [w["world"] for w in kept],
            "n_worlds": len(sel),
        }
    return out


def _calibrated_world_row(cfg, target, clean, family, real, phi, silent) -> dict:
    hinted = calibrate_hinted(target, clean, family, real["follow"], phi, silent,
                              cfg.shared_depth0)
    p = calibrated_params(family, cfg.n_items, real["follow"], phi, clean["lateness0"],
                          clean["g_pre"], hinted["lateness1"], hinted["h_post"], silent,
                          cfg.shared_depth0)
    exact = exact_m_moments(p)
    errors = {k: exact[k] - target[k] for k in MOMENT_KEYS}
    row = run_condition(p, cfg.seeds, do_fit=True)
    row.update(world=f"{family.name} phi={phi:g} silent={silent:g}", family=family.name,
               phi=phi, silent=silent, calibration={**clean, **hinted},
               exact_moments=exact, moment_errors=errors,
               max_abs_moment_error=max(abs(v) for v in errors.values()),
               real={"fit_nde": real["fit_nde"], "fit_nie": real["fit_nie"],
                     "fit_te": real["fit_nde"] + real["fit_nie"], "anchor": real["anchor"]},
               anchor_rejection=anchor_rejection(row["summary"], real["anchor"]))
    return row


def experiment_e(cfg: StudyConfig, fits_root: Path) -> list[dict]:
    """Per question set and lateness family: one world calibrated under phi = 1 (no silent
    follower can exist) and one per silent mass under phi = 0. With silent mass 0 the
    phi = 0 world has the same observable data as the phi = 1 world."""
    out = []
    for substrate, real in REAL_SUBSTRATES.items():
        target = read_substrate_targets(fits_root, substrate)
        worlds = []
        for family in LATENESS_FAMILIES:
            clean = calibrate_clean(target, family, cfg.shared_depth0)
            worlds.append(_calibrated_world_row(cfg, target, clean, family, real, 1.0, 0.0))
            worlds += [_calibrated_world_row(cfg, target, clean, family, real, 0.0, s)
                       for s in E_SILENT_GRID]
        out.append({"substrate": substrate, "label": real["label"], "real": real,
                    "targets": target, "worlds": worlds, "ranges": e_ranges(worlds)})
    return out


# --------------------------------------------------------------------------- #
# F, G, H
# --------------------------------------------------------------------------- #
def f_turns(cfg: StudyConfig, scenario: str) -> tuple[tuple, tuple]:
    """(mediated p_t1, direct p_t1) for one F scenario."""
    if scenario == "F1":
        return cfg.uniform_turn, point_mass(SILENT_T1)
    if scenario == "F2":
        return point_mass(F_DEPTH_EARLY), point_mass(F_DEPTH_LATE)
    if scenario == "F3":
        return point_mass(F_DEPTH_LATE), point_mass(F_DEPTH_EARLY)
    raise ValueError(f"unknown F scenario {scenario!r}")


def _f_base(cfg: StudyConfig, scenario: str) -> WorldParams:
    mediated, direct = f_turns(cfg, scenario)
    return cfg.params(pi=DEFAULT_PI, p_t1=mediated, p_t1_direct=direct)


def experiment_f(cfg: StudyConfig) -> list[dict]:
    return [{"scenario": s, "description": F_SCENARIOS[s], "rows": _phi_rows(_f_base(cfg, s),
                                                                          cfg)}
            for s in F_SCENARIOS]


def experiment_g(cfg: StudyConfig) -> list[dict]:
    """The same repo fit on (X, M', Y), M' = share of depths answering the hinted option."""
    worlds = {"B": cfg.params(pi=DEFAULT_PI, p_t1=cfg.uniform_turn,
                              p_t1_direct=cfg.uniform_turn),
              "F1": _f_base(cfg, "F1")}
    out = []
    for name, base in worlds.items():
        rows = _phi_rows(base, cfg, mediator="hint_share")
        out.append({"world": name, "mediator": "hint_share", "rows": rows,
                    "fit_identical_across_phi": _fit_identical(rows)})
    return out


def experiment_h(fits_root: Path) -> dict:
    return {s: {"label": REAL_SUBSTRATES[s]["label"], **read_copy_controls(fits_root, s)}
            for s in REAL_SUBSTRATES}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _c(summary: dict, key: str, digits: int = 3) -> str:
    s = summary[key]
    if s["mean"] is None:
        return "n/a"
    f = f"{{:+.{digits}f}}"
    return f"{f.format(s['mean'])} [{f.format(s['lo'])},{f.format(s['hi'])}]"


def _truth(row: dict) -> str:
    t = row["truth_expected"]
    return f"truth NDE {t['nde']:.3f} NIE {t['nie']:.3f} TE {t['te']:.3f}"


def _flag(row: dict) -> str:
    f = row["te_filter"]
    gap = "n/a" if f["mean_gap"] is None else f"{f['mean_gap']:+.3f}"
    constant = row["n_seeds_mediator_constant"]
    note = f" | MEDIATOR CONSTANT in {constant} seeds" if constant else ""
    return f"{'PASS' if f['passes'] else 'FAIL'} gap {gap} | fails {row['n_fit_failures']}{note}"


def _fit3(s: dict) -> str:
    return f"fit NDE {_c(s, 'fit_nde')} NIE {_c(s, 'fit_nie')} TE {_c(s, 'fit_te')}"


def print_a(rows: list[dict]) -> None:
    print("\n== A. Timing sweep, phi = 1 (truth NDE 0, NIE = TE = 0.25), c = 0 ==")
    for r in rows:
        s = r["summary"]
        print(f"h {r['params']['h_post']:.1f} {r['turn']:<9} E[t1] {r['mean_t1']:.2f} | "
              f"{_fit3(s)} | beta {_c(s, 'fit_beta', 1)} gamma {_c(s, 'fit_gamma')} | "
              f"{_flag(r)}")
        print(f"{'':>28}M clean {_c(s, 'm_clean_mean')} hinted {_c(s, 'm_hinted_mean')} "
              f"fol {_c(s, 'm_hinted_followers_mean')} nonfol "
              f"{_c(s, 'm_hinted_nonfollowers_mean')}")


def _print_phi_rows(rows: list[dict], with_anchor: bool = True) -> None:
    for r in rows:
        s = r["summary"]
        anchor = (f" | anchor NDE {_c(s, 'anchor_nde')} NIE {_c(s, 'anchor_nie')}"
                  if with_anchor else "")
        print(f"phi {r['phi']:.2f} | {_truth(r)} | {_fit3(s)}{anchor} | {_flag(r)}")


def print_b(result: dict) -> None:
    print("\n== B. Truth sweep (shared uniform turn, c = 0, pi = 0.25) ==")
    print(f"(X,M,Y) identical across phi on all {len(result['checked_seeds'])} seeds: "
          f"{result['observables_identical_across_phi']}; fit identical across phi: "
          f"{result['fit_identical_across_phi']}")
    _print_phi_rows(result["rows"])


def print_c(rows: list[dict]) -> None:
    print("\n== C. Answer copying (uniform turn, pi = 0.25; anchor only, no fit) ==")
    for r in rows:
        s = r["summary"]
        print(f"h {r['h_post']:.1f} phi {r['phi']:.1f} c {r['copy']:.1f} | {_truth(r)} | "
              f"anchor NDE {_c(s, 'anchor_nde')} NIE {_c(s, 'anchor_nie')} TE "
              f"{_c(s, 'anchor_te')}")


def print_d(rows: list[dict]) -> None:
    print("\n== D. Silent followers, phi = 0 (truth NIE 0), c = 0, pi = 0.25 ==")
    for r in rows:
        s = r["summary"]
        print(f"silent {r['silent']:.1f} | {_truth(r)} | {_fit3(s)} | beta "
              f"{_c(s, 'fit_beta', 1)} | anchor NIE {_c(s, 'anchor_nie')} | M fol "
              f"{_c(s, 'm_hinted_followers_mean')} | {_flag(r)}")


def _print_e_world(w: dict) -> None:
    s, cal, err, rej = w["summary"], w["calibration"], w["moment_errors"], w["anchor_rejection"]
    errs = " ".join(f"{k} {v:+.3f}" for k, v in err.items())
    print(f"  {w['world']:<28} lat0 {cal['lateness0']:+.2f} g {cal['g_pre']:.2f} lat1 "
          f"{cal['lateness1']:+.2f} h {cal['h_post']:.2f} | moment err {errs}")
    print(f"    {_truth(w)} | {_fit3(s)} | {_flag(w)}")
    cells = " ".join(f"{c} {s[f'anchor_{c}']['mean']:.3f}" for c in ANCHOR_CELLS)
    verdict = f"REJECTED by {','.join(rej['rejected_by'])}" if rej["rejected"] else "kept"
    print(f"    anchor {cells} | real anchor {verdict}")


def print_e(subs: list[dict]) -> None:
    print("\n== E. Calibrated worlds (c = 0) ==")
    print(f"rule: {RULES['e_ranges']}")
    print(f"anchor: {RULES['anchor_rejection']}")
    for sub in subs:
        t, real = sub["targets"], sub["real"]
        print(f"-- {sub['label']}: follow {real['follow']} | REAL fit NDE {real['fit_nde']} "
              f"NIE {real['fit_nie']} TE {real['fit_nde'] + real['fit_nie']:.3f} | real "
              f"anchor {real['anchor']}")
        print(f"   target M clean {t['clean_mean']:.3f}/{t['clean_sd']:.3f} hinted "
              f"{t['hinted_mean']:.3f}/{t['hinted_sd']:.3f} (mean/sd)")
        for w in sub["worlds"]:
            _print_e_world(w)
        for label, rng in sub["ranges"].items():
            for scope in ("all", "restricted"):
                nie = rng[scope]["nie"]
                txt = "none" if nie is None else f"[{nie['min']:.3f}, {nie['max']:.3f}] " \
                                                 f"over {nie['n_worlds']} worlds"
                print(f"   {label} fit NIE range ({scope}): {txt}")


def print_f(blocks: list[dict]) -> None:
    print("\n== F. Type-dependent timing (c = 0, pi = 0.25) ==")
    for b in blocks:
        print(f"-- {b['scenario']}: {b['description']}")
        _print_phi_rows(b["rows"])


def print_g(blocks: list[dict]) -> None:
    print("\n== G. Directional mediator M' = share of depths answering the hinted option ==")
    for b in blocks:
        print(f"-- world {b['world']} (fit identical across phi: "
              f"{b['fit_identical_across_phi']})")
        for r in b["rows"]:
            s = r["summary"]
            print(f"phi {r['phi']:.2f} | {_truth(r)} | {_fit3(s)} | beta "
                  f"{_c(s, 'fit_beta', 1)} | M' clean {_c(s, 'share_clean_mean')} hinted "
                  f"{_c(s, 'share_hinted_mean')} | {_flag(r)}")


def _rate(pool: dict) -> str:
    return "n/a" if pool["rate"] is None else f"{pool['rate']:.3f} ({pool['k']}/{pool['n']})"


def print_h(result: dict) -> None:
    print("\n== H. Copy bounds from the stored falsifier controls (no simulation) ==")
    first = next(iter(result.values()))
    print(f"note: {first['subset_note']}")
    for sub in result.values():
        mu = sub["anchor_cells"]
        print(f"-- {sub['label']} ({sub['n_cells']} cells): follow {_rate(sub['follow'])} | "
              f"mu01 {_rate(mu['mu01'])} mu11 {_rate(mu['mu11'])}")
        for name, c in sub["controls"].items():
            print(f"   {name:<25} applied {c['n_applied']}/{c['n_items']} | a0 "
                  f"{_rate(c['a0'])} a1 {_rate(c['a1'])} | same cells mu01 "
                  f"{c['mu01_same_cells']['rate']:.3f} mu11 {c['mu11_same_cells']['rate']:.3f}")
        b = sub["copy_bounds"]
        print(f"   answer-only rate: no cue {b['answer_only_rate_no_cue']:.3f}, cue "
              f"{b['answer_only_rate_cue']:.3f} | marker gap: no cue "
              f"{b['marker_gap_no_cue']:+.3f}, cue {b['marker_gap_cue']:+.3f} | implied c "
              f"(sim copy model): no cue {_copy(b['implied_copy_rate_no_cue'])}, cue "
              f"{_copy(b['implied_copy_rate_cue'])}")


def _copy(implied: dict) -> str:
    if implied["applies"]:
        return f"{implied['value']:.3f}"
    return f"model does not apply ({implied['reason']})"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def repo_function_refs() -> dict:
    """File and line of every repo function the study calls."""
    refs = {}
    for fn in (summarize_curve, depths_for, fit_probit_mediation_map,
               probit_natural_effects_closed_form):
        path = Path(inspect.getsourcefile(fn)).resolve()
        refs[fn.__name__] = f"{path.relative_to(_HERE.parent)}:{inspect.getsourcelines(fn)[1]}"
    return refs


def convergence_counts(results: dict) -> dict:
    rows = list(results.get("A", [])) + list(results.get("D", []))
    rows += results.get("B", {}).get("rows", [])
    rows += [w for sub in results.get("E", []) for w in sub["worlds"]]
    rows += [r for b in results.get("F", []) + results.get("G", []) for r in b["rows"]]
    categories: dict[str, int] = {}
    for r in rows:
        for key, count in r["warning_categories"].items():
            categories[key] = categories.get(key, 0) + count
    return {"n_fits": sum(r["n_seeds"] for r in rows),
            "n_fit_failures": sum(r["n_fit_failures"] for r in rows),
            "n_optimizer_warnings": sum(r["n_optimizer_warnings"] for r in rows),
            "n_other_warnings": sum(r["n_other_warnings"] for r in rows),
            "warning_categories": categories,
            "rows_with_failures": [_row_name(r) for r in rows if r["n_fit_failures"]]}


def _row_name(row: dict) -> str:
    p = row["params"]
    return (f"{row.get('mediator', 'curve_area')} phi={p['phi']:g} h={p['h_post']:g} "
            f"direct_turn={[round(x, 2) for x in row['params']['direct_turn']]}")


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-items", type=int, default=StudyConfig.n_items)
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--experiments", default="ABCDEFGH")
    ap.add_argument("--legacy-depth0", action="store_true",
                    help="independent depth-0 draws per arm and t1 over 0..5 (old world)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fits-root", type=Path, default=FITS_ROOT)
    args = ap.parse_args(argv)
    cfg = StudyConfig(args.n_items, args.n_seeds, args.base_seed, not args.legacy_depth0)
    out = args.out or (RESULTS_PATH if cfg.shared_depth0 else LEGACY_RESULTS_PATH)
    results = {"config": asdict(cfg), "seeds": list(cfg.seeds), "rules": RULES,
               "repo_functions": repo_function_refs(),
               "summary_rule": "mean and 2.5/97.5 percentiles over seeds; fit quantities "
                               "over converged draws only, failures counted"}
    runners = {"A": (experiment_a, print_a), "B": (experiment_b, print_b),
               "C": (experiment_c, print_c), "D": (experiment_d, print_d),
               "E": (lambda c: experiment_e(c, args.fits_root), print_e),
               "F": (experiment_f, print_f), "G": (experiment_g, print_g),
               "H": (lambda c: experiment_h(args.fits_root), print_h)}
    for key in args.experiments.upper():
        if key not in runners:
            raise SystemExit(f"unknown experiment {key!r}; choose from {''.join(runners)}")
        run, show = runners[key]
        results[key] = run(cfg)
        show(results[key])
    results["convergence"] = convergence_counts(results)
    print(f"\nconvergence: {results['convergence']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=_json_default, allow_nan=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
