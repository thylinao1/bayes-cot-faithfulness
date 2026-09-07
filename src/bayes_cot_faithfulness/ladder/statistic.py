"""Element 11(f): the organism-minus-twin difference, its interval, and its MDE.

    "The organism-minus-twin difference in NDE and in the mediated share.
    Pre-registered prediction: NDE rises with trigger strength, the mediated share
    falls, twins flat. rho* is reported with its interval and carries no directional
    prediction. A ladder that moves the components but not rho* is a pass."
    -- PREREGISTRATION_jury_and_scale.md section 12, element 11(f)

    "1.645 x sqrt(2) x sd_pilot(D) at the ladder's chosen n, where D is the
    organism-minus-twin difference in NDE and, separately, in the mediated share, and
    sd_pilot(D) is computed across BOTH training seeds at the first two dose levels so
    that it contains training-seed variance."
    -- section 12.1, valued in A3.2

    "the first two dose levels are the two lowest organism doses that exist, which are
    the second and third rungs of the three-rung dose ladder."
    -- A3.9 ruling R6

This module takes FITS, not transcripts: each checkpoint arrives as draws of NDE, NIE
and TE (a nonparametric row bootstrap of the repaired MAP estimator, or a posterior --
which one is recorded per fit and never mixed inside one difference). It computes

* D_nde(rung, seed)   = NDE(organism side) - NDE(twin side)
* D_share(rung, seed) = mediated share(organism side) - mediated share(twin side)

with a percentile interval from the paired draws, the rung mean over the two seeds,
sd_pilot(D) on the four differences at rungs 2 and 3, and the A3.2 MDE. rho* is carried
through with its interval and no directional test; :func:`assert_no_directional_test`
exists so that asking this module for a rho* verdict is an error rather than a habit.

The mediated share is NIE/TE, which is undefined when TE is near zero. Draws whose |TE|
falls below ``te_floor`` are dropped and COUNTED, and a rung that lost too many is
reported as such rather than silently averaged over the survivors.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from .spec import LADDER_N_ITEMS, MDE_Z, SD_PILOT_RUNGS

DEFAULT_TE_FLOOR = 0.02
DEFAULT_CI = 0.95


class LadderStatisticError(ValueError):
    """An input the statistic cannot honestly summarise."""


@dataclass(frozen=True)
class CheckpointFit:
    """One checkpoint's fitted effects, as draws.

    ``source`` names where the draws came from ("bootstrap" or "posterior"). A
    difference between two fits with different sources is refused: a bootstrap
    percentile interval and a posterior credible interval are not the same object, and
    subtracting them produces a number with no stated meaning.
    """

    cell_id: str
    variant: str
    rung: int
    dose: float
    seed: int
    nde_draws: Sequence[float]
    nie_draws: Sequence[float]
    te_draws: Sequence[float]
    rho_star_draws: Sequence[float] | None = None
    source: str = "bootstrap"
    n_items: int = LADDER_N_ITEMS

    def __post_init__(self) -> None:
        n = len(self.nde_draws)
        if n == 0:
            raise LadderStatisticError(f"{self.cell_id}: no draws")
        if len(self.nie_draws) != n or len(self.te_draws) != n:
            raise LadderStatisticError(
                f"{self.cell_id}: nde/nie/te draw counts differ "
                f"({n}/{len(self.nie_draws)}/{len(self.te_draws)}); the draws must be "
                "paired replicate by replicate or a difference is meaningless"
            )

    @property
    def side(self) -> str:
        return "organism" if self.variant in ("organism", "disclosing") else "twin"


def _percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        raise LadderStatisticError("no values to take a percentile of")
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def interval(values: Sequence[float], ci: float = DEFAULT_CI) -> dict:
    lo_q = (1.0 - ci) / 2.0
    return {
        "point": statistics.fmean(values),
        "lo": _percentile(values, lo_q),
        "hi": _percentile(values, 1.0 - lo_q),
        "n_draws": len(values),
        "ci": ci,
    }


def mediated_share_draws(
    fit: CheckpointFit, te_floor: float = DEFAULT_TE_FLOOR
) -> tuple[list[float], int]:
    """NIE/TE per replicate, dropping and counting the replicates where TE is near zero."""
    keep, dropped = [], 0
    for nie, te in zip(fit.nie_draws, fit.te_draws):
        if abs(te) < te_floor:
            dropped += 1
            continue
        keep.append(nie / te)
    return keep, dropped


def _paired(a: Sequence[float], b: Sequence[float]) -> list[float]:
    n = min(len(a), len(b))
    return [a[i] - b[i] for i in range(n)]


def difference(
    organism: CheckpointFit, twin: CheckpointFit, *, ci: float = DEFAULT_CI,
    te_floor: float = DEFAULT_TE_FLOOR,
) -> dict:
    """D on NDE and on the mediated share for one (rung, seed) pair."""
    if organism.source != twin.source:
        raise LadderStatisticError(
            f"{organism.cell_id} draws are {organism.source} and {twin.cell_id} draws "
            f"are {twin.source}; a difference between them has no stated meaning"
        )
    if organism.seed != twin.seed or organism.rung != twin.rung:
        raise LadderStatisticError(
            f"{organism.cell_id} and {twin.cell_id} are not the same (rung, seed) pair"
        )
    o_share, o_dropped = mediated_share_draws(organism, te_floor)
    t_share, t_dropped = mediated_share_draws(twin, te_floor)
    out = {
        "rung": organism.rung,
        "dose": organism.dose,
        "seed": organism.seed,
        "organism_cell_id": organism.cell_id,
        "twin_cell_id": twin.cell_id,
        "source": organism.source,
        "nde": {
            "organism": statistics.fmean(organism.nde_draws),
            "twin": statistics.fmean(twin.nde_draws),
            "D": interval(_paired(organism.nde_draws, twin.nde_draws), ci),
        },
        "mediated_share": {
            "organism": statistics.fmean(o_share) if o_share else None,
            "twin": statistics.fmean(t_share) if t_share else None,
            "n_draws_dropped_organism": o_dropped,
            "n_draws_dropped_twin": t_dropped,
            "te_floor": te_floor,
        },
    }
    if o_share and t_share:
        out["mediated_share"]["D"] = interval(_paired(o_share, t_share), ci)
    else:
        out["mediated_share"]["D"] = None
        out["mediated_share"]["why_none"] = (
            f"every replicate had |TE| below the floor {te_floor} on at least one side "
            f"({o_dropped} organism, {t_dropped} twin dropped); a mediated share of "
            "NIE/TE is undefined there and is not imputed"
        )
    return out


def rho_star_report(fits: Sequence[CheckpointFit], ci: float = DEFAULT_CI) -> dict:
    """rho* per checkpoint with its interval, and NO directional test. Element 11(f)."""
    rows = []
    for f in fits:
        if not f.rho_star_draws:
            rows.append({"cell_id": f.cell_id, "rho_star": None,
                         "why_none": "no rho* draws supplied for this checkpoint"})
            continue
        rows.append({"cell_id": f.cell_id, "variant": f.variant, "rung": f.rung,
                     "rho_star": interval(f.rho_star_draws, ci)})
    return {
        "rows": rows,
        "directional_prediction": None,
        "note": (
            "element 11(f): 'rho* is reported with its interval and carries no "
            "directional prediction. A ladder that moves the components but not rho* "
            "is a pass.'"
        ),
    }


def assert_no_directional_test(what: str) -> None:
    """Refuse a directional verdict on rho*, by name."""
    raise LadderStatisticError(
        f"refusing to run a directional test on rho* ({what}). Element 11(f) reports "
        "rho* with its interval and carries no directional prediction; a ladder that "
        "moves the components but not rho* is a PASS, so a test that could fail on "
        "rho* not moving would contradict the pre-registration."
    )


@dataclass
class LadderReport:
    """Everything the statistic produces, with the denominators it produced it from."""

    per_pair: list[dict] = field(default_factory=list)
    per_rung: dict = field(default_factory=dict)
    rho_star: dict = field(default_factory=dict)
    n_fits: int = 0

    def organism_side_means(self, quantity: str) -> dict[int, float]:
        return {r: v[quantity]["organism_mean"] for r, v in sorted(self.per_rung.items())
                if v[quantity]["organism_mean"] is not None}

    def twin_side_means(self, quantity: str) -> dict[int, float]:
        return {r: v[quantity]["twin_mean"] for r, v in sorted(self.per_rung.items())
                if v[quantity]["twin_mean"] is not None}


def build_report(
    fits: Sequence[CheckpointFit], *, ci: float = DEFAULT_CI,
    te_floor: float = DEFAULT_TE_FLOOR,
) -> LadderReport:
    """Pair every organism-side checkpoint with its twin-side match and summarise."""
    by_key: dict[tuple[int, int, str], CheckpointFit] = {}
    for f in fits:
        key = (f.rung, f.seed, f.side)
        if key in by_key:
            raise LadderStatisticError(
                f"two {f.side}-side fits for rung {f.rung} seed {f.seed}: "
                f"{by_key[key].cell_id} and {f.cell_id}"
            )
        by_key[key] = f
    report = LadderReport(n_fits=len(fits))
    rungs = sorted({r for r, _s, _side in by_key})
    for rung in rungs:
        seeds = sorted({s for r, s, _side in by_key if r == rung})
        pairs = []
        for seed in seeds:
            org = by_key.get((rung, seed, "organism"))
            twin = by_key.get((rung, seed, "twin"))
            if org is None or twin is None:
                raise LadderStatisticError(
                    f"rung {rung} seed {seed} has "
                    f"{'no organism-side' if org is None else 'no twin-side'} fit; the "
                    "organism-minus-twin difference is not defined without both"
                )
            pairs.append(difference(org, twin, ci=ci, te_floor=te_floor))
        report.per_pair.extend(pairs)
        nde_ds = [p["nde"]["D"]["point"] for p in pairs]
        share_ds = [p["mediated_share"]["D"]["point"] for p in pairs
                    if p["mediated_share"]["D"] is not None]
        report.per_rung[rung] = {
            "dose": pairs[0]["dose"],
            "n_seeds": len(pairs),
            "seeds": seeds,
            "nde": {
                "D_per_seed": nde_ds,
                "D_mean": statistics.fmean(nde_ds),
                "organism_mean": statistics.fmean(
                    [p["nde"]["organism"] for p in pairs]),
                "twin_mean": statistics.fmean([p["nde"]["twin"] for p in pairs]),
            },
            "mediated_share": {
                "D_per_seed": share_ds,
                "D_mean": statistics.fmean(share_ds) if share_ds else None,
                "organism_mean": (
                    statistics.fmean([p["mediated_share"]["organism"] for p in pairs])
                    if all(p["mediated_share"]["organism"] is not None for p in pairs)
                    else None),
                "twin_mean": (
                    statistics.fmean([p["mediated_share"]["twin"] for p in pairs])
                    if all(p["mediated_share"]["twin"] is not None for p in pairs)
                    else None),
                "n_seeds_with_a_defined_share": len(share_ds),
            },
        }
    report.rho_star = rho_star_report(fits, ci)
    return report


def sd_pilot(report: LadderReport, quantity: str = "nde",
             rungs: Sequence[int] = SD_PILOT_RUNGS) -> dict:
    """sd_pilot(D) over the four differences at the two lowest ORGANISM doses (R6).

    Refuses rather than returning an sd over whatever happens to be present: the
    formula names four values, 2 rungs x 2 training seeds, and an sd over three of them
    is a different number wearing the same name.
    """
    values: list[float] = []
    missing: list[str] = []
    for rung in rungs:
        block = report.per_rung.get(rung)
        if block is None:
            missing.append(f"rung {rung} has no pairs")
            continue
        ds = block[quantity]["D_per_seed"]
        if len(ds) != 2:
            missing.append(f"rung {rung} has {len(ds)} seed(s), not 2")
        values.extend(ds)
    if missing or len(values) != 4:
        raise LadderStatisticError(
            "sd_pilot(D) is defined on 4 organism-minus-twin differences (rungs "
            f"{tuple(rungs)} x 2 training seeds, A3.2 under ruling R6) and this report "
            f"supplies {len(values)}: " + "; ".join(missing or ["unknown reason"])
        )
    return {
        "quantity": quantity,
        "rungs": list(rungs),
        "n_values": len(values),
        "values": values,
        "sd": statistics.stdev(values),
        "note": (
            "an sd on 4 values, which is what element 11 asks for and no more; A3.2 "
            "states that so its width is not mistaken for precision it does not have"
        ),
    }


def mde(sd_pilot_d: float, *, n_items: int = LADDER_N_ITEMS, quantity: str = "nde") -> dict:
    """The A3.2 formula, evaluated from the inputs A3.2 names."""
    return {
        "formula": "1.645 x sqrt(2) x sd_pilot(D)",
        "quantity": quantity,
        "z": MDE_Z,
        "sqrt2": math.sqrt(2.0),
        "sd_pilot_D": sd_pilot_d,
        "ladder_n_items_per_checkpoint": n_items,
        "mde": MDE_Z * math.sqrt(2.0) * sd_pilot_d,
        "source": "PREREGISTRATION_jury_and_scale.md section 12.1, valued in A3.2",
    }


def prediction_test(report: LadderReport, *, dose_rungs: Sequence[int] = SD_PILOT_RUNGS,
                    flat_tolerance: float | None = None) -> dict:
    """The three pre-registered clauses of 11(f), each separately falsifiable.

    1. NDE rises with trigger strength: the organism-side NDE mean is non-decreasing
       across the ordered organism doses, and the top-minus-bottom difference is
       positive.
    2. The mediated share falls: the organism-side mediated share is non-increasing
       across the same doses, and the bottom-minus-top difference is positive.
    3. Twins flat: the twin-side NDE spread across those rungs is within
       ``flat_tolerance``. The tolerance defaults to the MDE computed from this
       report's own sd_pilot(D), so "flat" means "smaller than the difference this
       ladder could detect" rather than "small".

    Each clause reports the numbers it was decided on and the input that would falsify
    it, so a reader can see that a clause CAN fail.
    """
    rungs = [r for r in dose_rungs if r in report.per_rung]
    if len(rungs) < 2:
        raise LadderStatisticError(
            f"the prediction test needs at least two organism dose rungs and this "
            f"report has {len(rungs)} of {list(dose_rungs)}"
        )
    ordered = sorted(rungs, key=lambda r: report.per_rung[r]["dose"])
    nde = [report.per_rung[r]["nde"]["organism_mean"] for r in ordered]
    share = [report.per_rung[r]["mediated_share"]["organism_mean"] for r in ordered]
    twin = [report.per_rung[r]["nde"]["twin_mean"] for r in ordered]

    if flat_tolerance is None:
        try:
            flat_tolerance = mde(sd_pilot(report, "nde")["sd"])["mde"]
            tolerance_source = "the MDE from this report's own sd_pilot(D) on NDE"
        except LadderStatisticError as exc:
            flat_tolerance = float("inf")
            tolerance_source = f"no MDE available ({exc}); clause 3 cannot fail"
    else:
        tolerance_source = "supplied by the caller"

    clause1 = {
        "clause": "nde_rises_with_dose",
        "doses": [report.per_rung[r]["dose"] for r in ordered],
        "organism_nde_means": nde,
        "top_minus_bottom": nde[-1] - nde[0],
        "monotone": all(b >= a for a, b in pairwise(nde)),
        "passed": all(b >= a for a, b in pairwise(nde)) and nde[-1] > nde[0],
        "falsified_by": "an organism NDE that does not increase with the dose",
    }
    if any(s is None for s in share):
        clause2 = {
            "clause": "mediated_share_falls",
            "passed": None,
            "why_none": "at least one rung has no defined mediated share (TE near zero)",
            "organism_share_means": share,
            "falsified_by": "an organism mediated share that does not fall with the dose",
        }
    else:
        clause2 = {
            "clause": "mediated_share_falls",
            "organism_share_means": share,
            "bottom_minus_top": share[0] - share[-1],
            "monotone": all(b <= a for a, b in pairwise(share)),
            "passed": (all(b <= a for a, b in pairwise(share))
                       and share[-1] < share[0]),
            "falsified_by": "an organism mediated share that does not fall with the dose",
        }
    spread = max(twin) - min(twin)
    clause3 = {
        "clause": "twins_flat",
        "twin_nde_means": twin,
        "spread": spread,
        "tolerance": flat_tolerance,
        "tolerance_source": tolerance_source,
        "passed": spread <= flat_tolerance,
        "falsified_by": "a twin NDE that moves across rungs by more than the MDE",
    }
    clauses = [clause1, clause2, clause3]
    decided = [c for c in clauses if c.get("passed") is not None]
    return {
        "clauses": clauses,
        "n_clauses": len(clauses),
        "n_decided": len(decided),
        "n_passed": sum(1 for c in decided if c["passed"]),
        "all_passed": bool(decided) and all(c["passed"] for c in decided),
        "rho_star": "reported with its interval; no directional test (element 11(f))",
    }


def bootstrap_effects(x, m, y, *, n_boot: int = 200, seed: int = 7, rho: float = 0.0):
    """A row bootstrap of the repaired MAP estimator, the fits this module consumes.

    Kept here so the pipeline is complete, and deliberately thin: the estimator, the
    closed form and the rho* definition all live in the repository already and this
    only resamples rows and calls them.
    """
    import numpy as np

    from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
    from bayes_cot_faithfulness.sensitivity import fit_probit_mediation_map

    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    nde, nie, te = [], [], []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fit = fit_probit_mediation_map(x[idx], m[idx], y[idx], rho=rho, intercepts=True)
        a, b, c = probit_natural_effects_closed_form(
            fit.alpha, fit.beta, fit.gamma, fit.sigma_m, rho, fit.mu_m, fit.alpha0)
        nde.append(a)
        nie.append(b)
        te.append(c)
    return nde, nie, te
