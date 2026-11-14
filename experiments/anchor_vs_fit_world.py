"""The synthetic world of the anchor-versus-fit study: generator, anchor, fit, calibration.

N items, all gold when clean, five truncation depths j = 0..4 (the repo's
``DEFAULT_FRACTIONS``). Answer codes: 0 gold, 1 the hinted option, 2 a third option.

* Clean chain: commitment index t0 ~ p_t0 over {0..4}. Depth j >= t0 answers gold; j < t0
  answers gold with probability g_pre, else the third option. Final answer gold.
* Hinted arm: a follower with probability pi (final = hinted, Y = 1), else a non-follower
  (final gold, Y = 0, chain drawn like a clean chain, independently of its clean chain).
* A follower has a latent type, text-mediated with probability phi, else direct. Type is
  drawn first; mediated followers draw the turn index t1 from p_t1 with its silent mass
  (t1 = 5) removed and renormalised, direct followers from p_t1_direct (default p_t1,
  silent mass included). Depth j >= t1 answers hinted with probability h_post, else gold;
  j < t1 answers gold with probability g_pre, else the third option.
* REALISTIC DEPTH 0 (the default, ``shared_depth0=True``). In the real pipeline the
  depth-0 forced-answer prompt is ``continuation_prompt(item, truncate_cot(cot, 0))``,
  which carries neither the cue nor any chain text, and ``arm_curves`` builds both arms'
  prompts from the same item (experiments/08_additive_arms.py). So the depth-0 answer is
  ONE item-level draw used on both arms, and a follower's chain cannot turn at depth 0:
  t1 ranges over {1..5}. ``shared_depth0=False`` restores the earlier world (independent
  depth-0 draws, t1 over {0..5}).
* M is the repo's ``summarize_curve(...).curve_area`` against THAT ARM'S OWN final answer.
  M' (``hint_share``, experiment G) is the same repo summary against the HINTED option on
  both arms: the share of depths whose forced answer is the hinted option. Every one of
  the 3^5 answer patterns is summarised once by the repo function and looked up by code.
* Truth for mediator = whole chain text: TE = pi, NDE = pi (1 - phi), NIE = pi phi.
* Anchor, with answer-copy probability c per replay (copy -> the donor's final answer):
  mu00 gold; mu01 native: follower whose chain turned -> hinted w.p. h_post, else gold;
  mu10 native: direct follower -> hinted, everyone else gold; mu11 native: direct follower
  -> hinted, mediated follower -> hinted w.p. h_post, non-follower gold.
  NDE = mu10 - mu00, NIE = mu11 - mu10, TE = mu11 - mu00.

All randomness is drawn as uniforms up front in a fixed order whatever the parameters
(common random numbers); the turn index uses one uniform per item through the inverse CDF
of the type's own distribution, so with a shared turn distribution (X, M, Y) are
bit-identical across phi. The anchor's h_post draws are separate replays.

The fit is the real cells' call (experiments/wave1_fits.py ``_fit_at_zero`` and
``column_b``): ``fit_probit_mediation_map(X, M, Y, rho=0.0, intercepts=True)`` on two rows
per item, then ``probit_natural_effects_closed_form`` at rho = 0. A draw whose optimizer
reports failure is excluded from the fit summaries and counted. Every fitted condition
carries the total-effect filter: the fit's TE minus the simulated randomized arm
difference, flagged when its mean over seeds exceeds ``TE_GAP_THRESHOLD`` in absolute value.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from bayes_cot_faithfulness.closed_form import probit_natural_effects_closed_form
from bayes_cot_faithfulness.curves import DEFAULT_FRACTIONS, depths_for, summarize_curve
from bayes_cot_faithfulness.sensitivity import OptimizerWarning, fit_probit_mediation_map

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_DEPTHS = len(DEFAULT_FRACTIONS)  # five truncation depths
SILENT_T1 = N_DEPTHS  # turn index 5: the chain content never turns to the hinted option
GOLD, HINT, THIRD = 0, 1, 2  # answer codes
OPTION_LETTERS = ("A", "B", "C")  # what the curve summary sees for gold, hinted, third
N_OPTIONS = len(OPTION_LETTERS)
N_PATTERNS = N_OPTIONS**N_DEPTHS
PATTERN_POWERS = N_OPTIONS ** np.arange(N_DEPTHS)
PATTERN_DIGITS = (np.arange(N_PATTERNS)[:, None] // PATTERN_POWERS) % N_OPTIONS
# A four-step chain: the repo's depths_for maps the five fractions onto depths 0..4.
SYNTHETIC_CHAIN = "\n".join(f"{k}. reasoning step {k}" for k in range(1, N_DEPTHS))

DEFAULT_N_ITEMS = 4000
DEFAULT_PI = 0.25
DEFAULT_PHI = 1.0
DEFAULT_T0_LATENESS = -1.5  # with g_pre 0.2 gives mean clean M of about 0.95
DEFAULT_G_PRE = 0.2
DEFAULT_H_POST = 1.0
DEFAULT_COPY = 0.0
DEFAULT_SHARED_DEPTH0 = True
REALISTIC_FIRST_TURN = 1  # the depth-0 prompt shows no chain text, so no turn there
LEGACY_FIRST_TURN = 0
PMF_TOL = 1e-9
FIT_RHO = 0.0
INTERVAL_PERCENTILES = (2.5, 97.5)
ANCHOR_CELLS = ("mu00", "mu01", "mu10", "mu11")
# Total-effect filter: on the 72 real cells |fit TE - randomized arm difference| has
# median 0.008 and maximum 0.061 (column_b.model_implied_te_vs_randomized_arm_difference).
TE_GAP_THRESHOLD = 0.06
MEDIATORS = ("curve_area", "hint_share")

# Calibration search
LATENESS_BOUND = 8.0  # |lateness| beyond this is a point mass for all practical purposes
LATENESS_GRID_POINTS = 81
PROB_GRID_POINTS = 51
PROB_BOUNDS = (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def _check_prob(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")


def _check_pmf(name: str, pmf: tuple[float, ...], length: int) -> None:
    arr = np.asarray(pmf, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have {length} entries, got {arr.shape}")
    if np.any(arr < 0) or abs(arr.sum() - 1.0) > PMF_TOL:
        raise ValueError(f"{name} must be a probability vector, got {pmf!r}")


def _check_first(first: int, last: int) -> None:
    if not 0 <= first <= last:
        raise ValueError(f"first depth must lie in [0, {last}], got {first}")


def first_turn_depth(shared_depth0: bool) -> int:
    return REALISTIC_FIRST_TURN if shared_depth0 else LEGACY_FIRST_TURN


def lateness_pmf(lateness: float, first: int = 0) -> tuple[float, ...]:
    """Exponential tilt over depths first..4 (no mass below ``first``): very negative ->
    all mass at ``first``, 0 -> uniform, very positive -> all mass at 4."""
    _check_first(first, N_DEPTHS - 1)
    logits = float(lateness) * np.arange(N_DEPTHS - first)
    weights = np.exp(logits - logits.max())
    out = np.zeros(N_DEPTHS)
    out[first:] = weights / weights.sum()
    return tuple(float(w) for w in out)


def endpoint_pmf(late_share: float, first: int = 0) -> tuple[float, ...]:
    """Mass 1 - late_share at depth ``first`` and late_share at the last depth 4."""
    _check_prob("late_share", late_share)
    _check_first(first, N_DEPTHS - 2)
    out = [0.0] * N_DEPTHS
    out[first] = 1.0 - float(late_share)
    out[-1] = float(late_share)
    return tuple(out)


def point_mass(depth: int) -> tuple[float, ...]:
    """All turn mass at one index of a length-6 turn distribution (5 = silent)."""
    _check_first(depth, N_DEPTHS)
    pmf = [0.0] * (N_DEPTHS + 1)
    pmf[depth] = 1.0
    return tuple(pmf)


def turn_pmf(over_depths: tuple[float, ...], silent: float) -> tuple[float, ...]:
    """A length-6 turn distribution: ``(1 - silent)`` spread as ``over_depths``, then
    ``silent`` at t1 = 5."""
    _check_prob("silent", silent)
    return tuple((1.0 - silent) * p for p in over_depths) + (float(silent),)


UNIFORM_TURN = turn_pmf(lateness_pmf(0.0, REALISTIC_FIRST_TURN), 0.0)
UNIFORM_TURN_LEGACY = turn_pmf(lateness_pmf(0.0, LEGACY_FIRST_TURN), 0.0)
DEFAULT_P_T0 = lateness_pmf(DEFAULT_T0_LATENESS)


def _cdf(pmf: np.ndarray) -> np.ndarray:
    """CDF with the last positive-mass entry pinned to exactly 1 (no roundoff overflow)."""
    p = np.asarray(pmf, dtype=float)
    cdf = np.cumsum(p) / p.sum()
    cdf[np.flatnonzero(p > 0)[-1]:] = 1.0
    return cdf


def _sample(pmf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF categorical draw; zero-mass categories are never returned."""
    return np.searchsorted(_cdf(pmf), u, side="right")


# --------------------------------------------------------------------------- #
# Parameters, uniforms, the world
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorldParams:
    """Every parameter of the synthetic world. ``p_t1_direct`` None means ``p_t1``."""

    n_items: int = DEFAULT_N_ITEMS
    pi: float = DEFAULT_PI
    phi: float = DEFAULT_PHI
    p_t0: tuple[float, ...] = DEFAULT_P_T0
    g_pre: float = DEFAULT_G_PRE
    p_t1: tuple[float, ...] = UNIFORM_TURN
    p_t1_direct: tuple[float, ...] | None = None
    h_post: float = DEFAULT_H_POST
    copy: float = DEFAULT_COPY
    shared_depth0: bool = DEFAULT_SHARED_DEPTH0

    def __post_init__(self) -> None:
        if int(self.n_items) != self.n_items or self.n_items < 1:
            raise ValueError(f"n_items must be a positive integer, got {self.n_items!r}")
        for name in ("pi", "phi", "g_pre", "h_post", "copy"):
            _check_prob(name, getattr(self, name))
        _check_pmf("p_t0", self.p_t0, N_DEPTHS)
        _check_pmf("p_t1", self.p_t1, N_DEPTHS + 1)
        if self.p_t1_direct is not None:
            _check_pmf("p_t1_direct", self.p_t1_direct, N_DEPTHS + 1)
        if self.phi > 0 and sum(self.p_t1[:N_DEPTHS]) <= PMF_TOL:
            raise ValueError("phi > 0 needs p_t1 mass on a turn depth: mediated followers "
                             "cannot be silent")
        if self.shared_depth0:
            self._check_no_turn_at_depth0()

    def _check_no_turn_at_depth0(self) -> None:
        for name in ("p_t1", "p_t1_direct"):
            pmf = getattr(self, name)
            if pmf is not None and pmf[0] > 0:
                raise ValueError(
                    f"{name} puts mass {pmf[0]} on a turn at depth 0, but the depth-0 forced "
                    "answer sees no chain text; pass shared_depth0=False for the legacy world"
                )

    @property
    def direct_turn(self) -> np.ndarray:
        return np.asarray(self.p_t1 if self.p_t1_direct is None else self.p_t1_direct)

    @property
    def mediated_turn(self) -> np.ndarray:
        """p_t1 with the silent mass removed, renormalised (zeros when phi = 0 allows it)."""
        turns = np.asarray(self.p_t1[:N_DEPTHS], dtype=float)
        if turns.sum() <= PMF_TOL:
            return np.zeros(N_DEPTHS + 1)
        return np.append(turns / turns.sum(), 0.0)


def with_changes(params: WorldParams, **changes) -> WorldParams:
    """A new WorldParams with some fields replaced (validated again)."""
    return WorldParams(**{**asdict(params), **changes})


@dataclass(frozen=True, eq=False)
class Uniforms:
    """All randomness of one draw, fixed in shape and order whatever the parameters."""

    t0_clean: np.ndarray
    pre_clean: np.ndarray
    follow: np.ndarray
    follower_type: np.ndarray
    t1: np.ndarray
    post: np.ndarray
    t0_hint: np.ndarray
    pre_hint: np.ndarray
    copy: np.ndarray
    native: np.ndarray


def draw_uniforms(seed: int, n_items: int) -> Uniforms:
    rng = np.random.default_rng(seed)
    return Uniforms(
        t0_clean=rng.random(n_items),
        pre_clean=rng.random((n_items, N_DEPTHS)),
        follow=rng.random(n_items),
        follower_type=rng.random(n_items),
        t1=rng.random(n_items),
        post=rng.random((n_items, N_DEPTHS)),
        t0_hint=rng.random(n_items),
        pre_hint=rng.random((n_items, N_DEPTHS)),
        copy=rng.random((n_items, len(ANCHOR_CELLS))),
        native=rng.random((n_items, len(ANCHOR_CELLS))),
    )


@dataclass(frozen=True, eq=False)
class World:
    n_items: int
    clean_answers: np.ndarray
    hinted_answers: np.ndarray
    t0_clean: np.ndarray
    t1: np.ndarray  # -1 for non-followers, 5 for silent followers
    follower: np.ndarray
    mediated: np.ndarray
    m_clean: np.ndarray
    m_hinted: np.ndarray
    share_clean: np.ndarray  # M' on the clean arm: share of depths answering hinted
    share_hinted: np.ndarray
    X: np.ndarray = field(repr=False)
    M: np.ndarray = field(repr=False)
    Y: np.ndarray = field(repr=False)


def curve_depths() -> tuple[int, ...]:
    depths = tuple(depths_for(SYNTHETIC_CHAIN, DEFAULT_FRACTIONS))
    if len(depths) != N_DEPTHS:
        raise RuntimeError(f"synthetic chain gives depths {depths}, need {N_DEPTHS}")
    return depths


@lru_cache(maxsize=1)
def curve_area_table() -> np.ndarray:
    """``summarize_curve(...).curve_area`` for every answer pattern and reference answer,
    shape (N_OPTIONS, N_PATTERNS). Row r is M against final answer r."""
    depths = curve_depths()
    table = np.empty((N_OPTIONS, N_PATTERNS))
    for ref in range(N_OPTIONS):
        for code in range(N_PATTERNS):
            letters = [OPTION_LETTERS[d] for d in PATTERN_DIGITS[code]]
            area = summarize_curve(depths, letters, OPTION_LETTERS[ref]).curve_area
            if area is None:
                raise RuntimeError(f"curve unscorable for pattern {letters} vs {ref}")
            table[ref, code] = area
    return table


def _curve_area(answers: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return curve_area_table()[reference, answers @ PATTERN_POWERS]


def _clean_like_chain(t0: np.ndarray, u_pre: np.ndarray, g_pre: float) -> np.ndarray:
    after = np.arange(N_DEPTHS)[None, :] >= t0[:, None]
    pre = np.where(u_pre < g_pre, GOLD, THIRD)
    return np.where(after, GOLD, pre)


def _follower_chain(t1: np.ndarray, u_pre: np.ndarray, u_post: np.ndarray,
                    g_pre: float, h_post: float) -> np.ndarray:
    after = np.arange(N_DEPTHS)[None, :] >= t1[:, None]
    pre = np.where(u_pre < g_pre, GOLD, THIRD)
    post = np.where(u_post < h_post, HINT, GOLD)
    return np.where(after, post, pre)


def _draw_followers(params: WorldParams, u: Uniforms):
    follower = u.follow < params.pi
    mediated = follower & (u.follower_type < params.phi)
    t1 = np.full(len(u.follow), -1)
    direct = follower & ~mediated
    if mediated.any():
        t1[mediated] = _sample(params.mediated_turn, u.t1[mediated])
    if direct.any():
        t1[direct] = _sample(params.direct_turn, u.t1[direct])
    return follower, mediated, t1


def _hinted_arm(params: WorldParams, u: Uniforms, clean: np.ndarray, follower, t1):
    non_follower_chain = _clean_like_chain(
        _sample(np.asarray(params.p_t0), u.t0_hint), u.pre_hint, params.g_pre
    )
    turn = np.where(follower, t1, SILENT_T1)
    follower_chain = _follower_chain(turn, u.pre_hint, u.post, params.g_pre, params.h_post)
    hinted = np.where(follower[:, None], follower_chain, non_follower_chain)
    if params.shared_depth0:  # one depth-0 draw per item, used on both arms
        hinted = np.column_stack([clean[:, :1], hinted[:, 1:]])
    return hinted


def _interleave(clean_values: np.ndarray, hinted_values: np.ndarray) -> np.ndarray:
    return np.column_stack([clean_values, hinted_values]).ravel()


def simulate_world(params: WorldParams, u: Uniforms) -> World:
    n = params.n_items
    if len(u.follow) != n:
        raise ValueError(f"uniforms are for {len(u.follow)} items, params say {n}")
    t0 = _sample(np.asarray(params.p_t0), u.t0_clean)
    clean = _clean_like_chain(t0, u.pre_clean, params.g_pre)
    follower, mediated, t1 = _draw_followers(params, u)
    hinted = _hinted_arm(params, u, clean, follower, t1)
    m_clean = _curve_area(clean, np.full(n, GOLD))
    m_hinted = _curve_area(hinted, np.where(follower, HINT, GOLD))
    return World(
        n_items=n, clean_answers=clean, hinted_answers=hinted, t0_clean=t0, t1=t1,
        follower=follower, mediated=mediated, m_clean=m_clean, m_hinted=m_hinted,
        share_clean=_curve_area(clean, np.full(n, HINT)),
        share_hinted=_curve_area(hinted, np.full(n, HINT)),
        X=np.tile([0.0, 1.0], n),
        M=_interleave(m_clean, m_hinted),
        Y=_interleave(np.zeros(n, dtype=int), follower.astype(int)),
    )


def observables(world: World, mediator: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, M, Y) with M the curve area (the real fits' mediator) or the hinted share M'."""
    if mediator == "curve_area":
        return world.X, world.M, world.Y
    if mediator == "hint_share":
        return world.X, _interleave(world.share_clean, world.share_hinted), world.Y
    raise ValueError(f"unknown mediator {mediator!r}, expected one of {MEDIATORS}")


# --------------------------------------------------------------------------- #
# Truth, anchor, fit
# --------------------------------------------------------------------------- #
def truth_expected(params: WorldParams) -> dict:
    return {"te": params.pi, "nde": params.pi * (1.0 - params.phi),
            "nie": params.pi * params.phi}


def truth_realised(world: World) -> dict:
    return {"te": float(world.follower.mean()),
            "nde": float((world.follower & ~world.mediated).mean()),
            "nie": float(world.mediated.mean())}


def anchor_rates(world: World, params: WorldParams, u: Uniforms) -> dict:
    """The four replay cells and the three anchor contrasts, on the same items."""
    fol, med = world.follower, world.mediated
    direct = fol & ~med
    turned = fol & (world.t1 < SILENT_T1)
    no = np.zeros(world.n_items, dtype=bool)
    donor_final_hint = {"mu00": no, "mu01": fol, "mu10": no, "mu11": fol}
    native = {
        "mu00": no,
        "mu01": turned & (u.native[:, 1] < params.h_post),
        "mu10": direct,
        "mu11": direct | (med & (u.native[:, 3] < params.h_post)),
    }
    rates = {}
    for k, cell in enumerate(ANCHOR_CELLS):
        copied = u.copy[:, k] < params.copy
        rates[cell] = float(np.where(copied, donor_final_hint[cell], native[cell]).mean())
    rates["nde"] = rates["mu10"] - rates["mu00"]
    rates["nie"] = rates["mu11"] - rates["mu10"]
    rates["te"] = rates["mu11"] - rates["mu00"]
    return rates


def fit_world(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> dict:
    """The real cells' fit: repaired probit MAP at rho = 0, closed-form natural effects,
    plus the total-effect gap against the randomized arm difference."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_probit_mediation_map(X, M, Y, rho=FIT_RHO, intercepts=True)
    nde, nie, te = probit_natural_effects_closed_form(
        fit.alpha, fit.beta, fit.gamma, fit.sigma_m, FIT_RHO, fit.mu_m, fit.alpha0
    )
    n_opt = sum(1 for w in caught if issubclass(w.category, OptimizerWarning))
    categories: dict[str, int] = {}
    for w in caught:
        key = f"{w.category.__name__}: {str(w.message).split(':')[0][:60]}"
        categories[key] = categories.get(key, 0) + 1
    arm_difference = float(Y[X == 1].mean() - Y[X == 0].mean())
    return {
        "fit_converged": bool(fit.converged),
        "fit_optimizer_warnings": n_opt,
        "fit_other_warnings": len(caught) - n_opt,
        "fit_warning_categories": categories,
        "mediator_constant": bool(np.all(M == M[0])),
        "fit_nde": float(nde), "fit_nie": float(nie), "fit_te": float(te),
        "fit_te_gap": float(te) - arm_difference, "arm_difference": arm_difference,
        "fit_alpha": fit.alpha, "fit_beta": fit.beta, "fit_gamma": fit.gamma,
        "fit_sigma_m": fit.sigma_m, "fit_mu_m": fit.mu_m, "fit_alpha0": fit.alpha0,
    }


FIT_KEYS = ("fit_nde", "fit_nie", "fit_te", "fit_te_gap", "fit_alpha", "fit_beta",
            "fit_gamma", "fit_sigma_m", "fit_mu_m", "fit_alpha0")
ANCHOR_KEYS = tuple(f"anchor_{k}" for k in ANCHOR_CELLS + ("nde", "nie", "te"))
TRUTH_KEYS = ("truth_real_te", "truth_real_nde", "truth_real_nie")
M_KEYS = ("m_clean_mean", "m_clean_sd", "m_hinted_mean", "m_hinted_sd",
          "m_hinted_followers_mean", "m_hinted_nonfollowers_mean",
          "share_clean_mean", "share_hinted_mean")


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def _m_stats(world: World) -> dict:
    return {
        "m_clean_mean": float(world.m_clean.mean()),
        "m_clean_sd": float(world.m_clean.std(ddof=1)),
        "m_hinted_mean": float(world.m_hinted.mean()),
        "m_hinted_sd": float(world.m_hinted.std(ddof=1)),
        "m_hinted_followers_mean": _mean_or_none(world.m_hinted[world.follower]),
        "m_hinted_nonfollowers_mean": _mean_or_none(world.m_hinted[~world.follower]),
        "share_clean_mean": float(world.share_clean.mean()),
        "share_hinted_mean": float(world.share_hinted.mean()),
    }


def run_draw(params: WorldParams, seed: int, do_fit: bool,
             mediator: str = "curve_area") -> dict:
    u = draw_uniforms(seed, params.n_items)
    world = simulate_world(params, u)
    record = {"seed": seed, **_m_stats(world)}
    record.update({f"truth_real_{k}": v for k, v in truth_realised(world).items()})
    record.update({f"anchor_{k}": v for k, v in anchor_rates(world, params, u).items()})
    if do_fit:
        record.update(fit_world(*observables(world, mediator)))
    return record


def _summary(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, INTERVAL_PERCENTILES)
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def te_filter(gaps: list[float]) -> dict:
    """Pass iff |mean over converged seeds of (fit TE - arm difference)| <= threshold."""
    base = {"threshold": TE_GAP_THRESHOLD, "rule": "abs(mean over converged seeds of fit "
            "TE minus randomized arm difference) <= threshold"}
    if not gaps:
        return {**base, "mean_gap": None, "passes": False, "n_seeds_above": None,
                "reason": "no converged draw"}
    mean_gap = float(np.mean(gaps))
    return {**base, "mean_gap": mean_gap, "passes": bool(abs(mean_gap) <= TE_GAP_THRESHOLD),
            "n_seeds_above": int(sum(abs(g) > TE_GAP_THRESHOLD for g in gaps)),
            "n_seeds": len(gaps)}


def run_condition(params: WorldParams, seeds: tuple[int, ...], do_fit: bool,
                  mediator: str = "curve_area") -> dict:
    """One condition over seeds: per-quantity mean and 2.5/97.5 percentiles over seeds.

    Fit quantities use only the draws whose optimizer converged; the rest are counted.
    """
    records = [run_draw(params, s, do_fit, mediator) for s in seeds]
    keys = M_KEYS + TRUTH_KEYS + ANCHOR_KEYS
    summary = {k: _summary([r[k] for r in records if r[k] is not None]) for k in keys}
    row = {"params": params_dict(params), "n_seeds": len(seeds),
           "truth_expected": truth_expected(params), "summary": summary}
    if do_fit:
        ok = [r for r in records if r["fit_converged"]]
        summary.update({k: _summary([r[k] for r in ok]) for k in FIT_KEYS})
        row["mediator"] = mediator
        row["n_fit_failures"] = len(records) - len(ok)
        row["n_optimizer_warnings"] = sum(r["fit_optimizer_warnings"] for r in records)
        row["n_other_warnings"] = sum(r["fit_other_warnings"] for r in records)
        row["warning_categories"] = _merge_counts(r["fit_warning_categories"] for r in records)
        row["n_seeds_mediator_constant"] = sum(r["mediator_constant"] for r in records)
        row["te_filter"] = te_filter([r["fit_te_gap"] for r in ok])
    return row


def _merge_counts(dicts) -> dict:
    total: dict[str, int] = {}
    for d in dicts:
        for key, count in d.items():
            total[key] = total.get(key, 0) + count
    return total


def params_dict(params: WorldParams) -> dict:
    d = asdict(params)
    d["direct_turn"] = [float(x) for x in params.direct_turn]
    d["mediated_turn"] = [float(x) for x in params.mediated_turn]
    return d


def assert_observables_phi_invariant(params: WorldParams, phis: tuple[float, ...],
                                     seed: int, mediator: str = "curve_area") -> None:
    """Raise AssertionError unless (X, M, Y) are identical across ``phis`` for ``seed``."""
    u = draw_uniforms(seed, params.n_items)
    ref = None
    for phi in phis:
        obs = observables(simulate_world(with_changes(params, phi=phi), u), mediator)
        if ref is None:
            ref = (phi, obs)
            continue
        for name, a, b in zip("XMY", obs, ref[1]):
            if not np.array_equal(a, b):
                raise AssertionError(f"{name} differs between phi={ref[0]} and phi={phi} "
                                     f"(seed {seed})")


# --------------------------------------------------------------------------- #
# Exact M moments and calibration (experiment E)
# --------------------------------------------------------------------------- #
def _pattern_pmf(turns: np.ndarray, pre: tuple, post: tuple,
                 depth0: tuple | None = None) -> np.ndarray:
    """Probability of each of the 3^5 answer patterns: depth j uses ``post`` when
    j >= turn and ``pre`` otherwise, mixed over the turn distribution ``turns``. With
    ``depth0`` given, depth 0 is an independent draw from it (the shared item-level
    depth-0 answer seen on the hinted arm)."""
    pre_p = np.asarray(pre, dtype=float)[PATTERN_DIGITS]
    post_p = np.asarray(post, dtype=float)[PATTERN_DIGITS]
    total = np.zeros(N_PATTERNS)
    for t, weight in enumerate(turns):
        if weight > 0:
            probs = np.where(np.arange(N_DEPTHS) >= t, post_p, pre_p)
            if depth0 is not None:
                probs = np.column_stack([np.asarray(depth0)[PATTERN_DIGITS[:, 0]],
                                         probs[:, 1:]])
            total += weight * probs.prod(axis=1)
    return total


def _depth0_marginal(params: WorldParams) -> tuple[float, float, float]:
    """The clean-arm depth-0 answer distribution, which the hinted arm shares."""
    p_gold = params.p_t0[0] + (1.0 - params.p_t0[0]) * params.g_pre
    return (p_gold, 0.0, 1.0 - p_gold)


def _moments(pattern_probs: np.ndarray, reference: int) -> tuple[float, float]:
    m = curve_area_table()[reference]
    return float(pattern_probs @ m), float(pattern_probs @ m**2)


def exact_m_moments(params: WorldParams) -> dict:
    """Exact mean and sd of M on each arm, M values from the repo's curve summary."""
    pre = (params.g_pre, 0.0, 1.0 - params.g_pre)
    clean_post = (1.0, 0.0, 0.0)
    d0 = _depth0_marginal(params) if params.shared_depth0 else None
    c1, c2 = _moments(_pattern_pmf(np.asarray(params.p_t0), pre, clean_post), GOLD)
    n1, n2 = _moments(_pattern_pmf(np.asarray(params.p_t0), pre, clean_post, d0), GOLD)
    follower_turns = params.phi * params.mediated_turn + (1 - params.phi) * params.direct_turn
    fol_post = (1.0 - params.h_post, params.h_post, 0.0)
    f1, f2 = _moments(_pattern_pmf(follower_turns, pre, fol_post, d0), HINT)
    h1 = (1 - params.pi) * n1 + params.pi * f1
    h2 = (1 - params.pi) * n2 + params.pi * f2
    return {"clean_mean": c1, "clean_sd": math.sqrt(max(c2 - c1 * c1, 0.0)),
            "hinted_mean": h1, "hinted_sd": math.sqrt(max(h2 - h1 * h1, 0.0)),
            "follower_mean": f1}


@dataclass(frozen=True)
class LatenessFamily:
    """A one-parameter family over depths first..4, used for both p_t0 and p_t1 in E."""

    name: str
    bounds: tuple[float, float]
    grid_points: int
    pmf: Callable[[float, int], tuple[float, ...]]


LATENESS_FAMILIES = (
    # Exponential tilt: unimodal, from all-at-first through uniform to all-at-4.
    LatenessFamily("tilt", (-LATENESS_BOUND, LATENESS_BOUND), LATENESS_GRID_POINTS,
                   lateness_pmf),
    # Endpoint: bimodal, the first allowed depth or the last depth only.
    LatenessFamily("endpoint", PROB_BOUNDS, PROB_GRID_POINTS, endpoint_pmf),
)
MOMENT_KEYS = ("clean_mean", "clean_sd", "hinted_mean", "hinted_sd")


def calibrated_params(family: LatenessFamily, n_items: int, pi: float, phi: float,
                      lateness0: float, g_pre: float, lateness1: float, h_post: float,
                      silent: float, shared_depth0: bool = DEFAULT_SHARED_DEPTH0
                      ) -> WorldParams:
    """A world from the five calibration parameters. Silent mass goes to direct followers
    only; mediated followers draw the same lateness without it."""
    turns = family.pmf(lateness1, first_turn_depth(shared_depth0))
    return WorldParams(
        n_items=n_items, pi=pi, phi=phi, p_t0=family.pmf(lateness0, 0), g_pre=g_pre,
        p_t1=turn_pmf(turns, 0.0), p_t1_direct=turn_pmf(turns, silent), h_post=h_post,
        copy=0.0, shared_depth0=shared_depth0,
    )


def _grid_then_refine(loss, family: LatenessFamily) -> tuple[np.ndarray, float]:
    """Grid search over (lateness, probability), then L-BFGS-B from the best grid point.
    The refined point is kept only if it improves on the grid point."""
    lat_grid = np.linspace(*family.bounds, family.grid_points)
    prob_grid = np.linspace(*PROB_BOUNDS, PROB_GRID_POINTS)
    best = min((loss(np.array([a, b])), a, b) for a in lat_grid for b in prob_grid)
    start = np.array(best[1:])
    res = minimize(loss, start, method="L-BFGS-B", bounds=[family.bounds, PROB_BOUNDS])
    if float(res.fun) <= best[0]:
        return res.x, float(res.fun)
    return start, float(best[0])


def calibrate_clean(target: dict, family: LatenessFamily,
                    shared_depth0: bool = DEFAULT_SHARED_DEPTH0) -> dict:
    """(lateness0, g_pre) to match the clean arm's mean and sd of M."""
    def loss(x):
        p = calibrated_params(family, 1, 0.0, 0.0, x[0], x[1], family.bounds[0], 1.0, 0.0,
                              shared_depth0)
        m = exact_m_moments(p)
        return (m["clean_mean"] - target["clean_mean"]) ** 2 + (
            m["clean_sd"] - target["clean_sd"]) ** 2
    x, val = _grid_then_refine(loss, family)
    return {"family": family.name, "lateness0": float(x[0]), "g_pre": float(x[1]),
            "clean_loss": val}


def calibrate_hinted(target: dict, clean: dict, family: LatenessFamily, pi: float,
                     phi: float, silent: float,
                     shared_depth0: bool = DEFAULT_SHARED_DEPTH0) -> dict:
    """(lateness1, h_post) to match the hinted arm's mean and sd of M, given the clean fit."""
    def loss(x):
        p = calibrated_params(family, 1, pi, phi, clean["lateness0"], clean["g_pre"], x[0],
                              x[1], silent, shared_depth0)
        m = exact_m_moments(p)
        return (m["hinted_mean"] - target["hinted_mean"]) ** 2 + (
            m["hinted_sd"] - target["hinted_sd"]) ** 2
    x, val = _grid_then_refine(loss, family)
    return {"lateness1": float(x[0]), "h_post": float(x[1]), "hinted_loss": val}
