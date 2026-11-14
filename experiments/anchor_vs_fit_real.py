"""Readers for the stored 72-cell fits used by the anchor-versus-fit study.

Experiment E reads the per-arm mean and sd of the curve-area mediator. Experiment H reads
the anchor's falsifier controls and four cells and pools them per question set as sum of k
over sum of n. Nothing here simulates; every number comes from
``experiments/results/cells72-fits/<model>/<substrate>/<cue>/fit.json``.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

FITS_ROOT = Path(__file__).resolve().parent / "results" / "cells72-fits"

# Real pooled aggregates of the 72-cell pass (item-weighted over 24 cells per substrate),
# as given in the study brief; the M targets are read from the fits at run time.
REAL_SUBSTRATES = {
    "arc_challenge": {
        "label": "ARC-Challenge", "follow": 0.254, "fit_nde": 0.115, "fit_nie": 0.125,
        "anchor": {"mu00": 0.000, "mu01": 0.191, "mu10": 0.015, "mu11": 0.223},
    },
    "aqua_rat": {
        "label": "AQuA-RAT", "follow": 0.159, "fit_nde": 0.142, "fit_nie": 0.015,
        "anchor": {"mu00": 0.003, "mu01": 0.111, "mu10": 0.124, "mu11": 0.255},
    },
    "logiqa2": {
        "label": "LogiQA 2.0", "follow": 0.476, "fit_nde": 0.271, "fit_nie": 0.215,
        "anchor": {"mu00": 0.002, "mu01": 0.445, "mu10": 0.095, "mu11": 0.508},
    },
}

COPY_CONTROLS = (
    "matched_answer_only_text",
    "answer_marker_removed",
    "answer_marker_relocated",
    "decisive_premise_edit",
    "meaning_preserving_edit",
)
ANCHOR_CELL_NAMES = ("mu00", "mu01", "mu10", "mu11")
SUBSET_NOTE = (
    "Each control's a0 and a1 rates are over the items where that edit applies (n_applied "
    "of n_items); mu01 and mu11 are over ALL items of the same cells. fit.json stores only "
    "pooled counts per cell and the tree holds no per-item anchor records for the 72 "
    "cells, so mu01 and mu11 cannot be restricted to a control's subset."
)
CONTROL_MEANINGS = {
    "matched_answer_only_text": "donor text replaced by 'Answer: (hinted)' plus "
                                "length-matched filler, no reasoning",
    "answer_marker_removed": "cued donor chain with every frozen answer marker struck",
    "answer_marker_relocated": "cued donor chain with the answer marker moved to the front",
    "decisive_premise_edit": "cued donor chain with every reference to the hinted option "
                             "(answer line included) repointed to another option",
    "meaning_preserving_edit": "cued donor chain with discourse connectives swapped only",
}


def fit_paths(fits_root: Path, substrate: str) -> list[Path]:
    paths = sorted(glob.glob(str(Path(fits_root) / "*" / substrate / "*" / "fit.json")))
    if not paths:
        raise FileNotFoundError(f"no fit.json for {substrate} under {fits_root}")
    return [Path(p) for p in paths]


def _load(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text()) for p in paths]


def read_substrate_targets(fits_root: Path, substrate: str) -> dict:
    """Item-weighted (by column_b.n_items) per-arm mean and sd of M over a substrate's
    cells. The sd target is the item-weighted mean of the per-cell sds, as specified; the
    pooled within-cell and total sds are recorded beside it."""
    cells = _load(fit_paths(fits_root, substrate))
    rows = []
    for d in cells:
        med = d["table"]["mediator"]
        rows.append((d["column_b"]["n_items"], med["clean_mean"], med["clean_sd"],
                     med["hinted_mean"], med["hinted_sd"], d["column_b"]["fit"]["beta"],
                     d["column_b"]["fit"]["gamma"]))
    arr = np.asarray(rows, dtype=float)
    w = arr[:, 0] / arr[:, 0].sum()
    out = {"n_cells": len(cells), "n_items": int(arr[:, 0].sum())}
    for arm, (mi, si) in {"clean": (1, 2), "hinted": (3, 4)}.items():
        mean = float(w @ arr[:, mi])
        out[f"{arm}_mean"] = mean
        out[f"{arm}_sd"] = float(w @ arr[:, si])
        out[f"{arm}_sd_pooled_within"] = float(math.sqrt(w @ arr[:, si] ** 2))
        out[f"{arm}_sd_total"] = float(math.sqrt(w @ (arr[:, si] ** 2
                                                      + (arr[:, mi] - mean) ** 2)))
    out["real_beta_negative_cells"] = int((arr[:, 5] < 0).sum())
    out["real_gamma_negative_cells"] = int((arr[:, 6] < 0).sum())
    return out


def _pool(pairs: list[tuple[int, int]]) -> dict:
    k = int(sum(p[0] for p in pairs))
    n = int(sum(p[1] for p in pairs))
    return {"k": k, "n": n, "rate": (k / n) if n else None}


def _follow_counts(d: dict) -> tuple[int, int]:
    """Followers among the fit's items: hinted-arm mean of Y times column_b.n_items."""
    n = int(d["column_b"]["n_items"])
    exact = d["table"]["outcome"]["hinted_arm_mean"] * n
    k = round(exact)
    if abs(exact - k) > 1e-6:
        raise ValueError(f"hinted_arm_mean * n_items = {exact} is not a count ({d['cell']})")
    return k, n


def _control_entry(cells: list[dict], name: str) -> dict:
    with_ctl = [d for d in cells if name in d["anchor"]["falsifier_controls"]]
    blocks = [d["anchor"]["falsifier_controls"][name] for d in with_ctl]
    n_applied = sum(b["n_applied"] for b in blocks)
    n_items = sum(b["n_items"] for b in blocks)
    return {
        "meaning": CONTROL_MEANINGS[name],
        "n_cells": len(with_ctl),
        "n_cells_missing": len(cells) - len(with_ctl),
        "n_applied": n_applied,
        "n_items": n_items,
        "applied_share": (n_applied / n_items) if n_items else None,
        "a0": _pool([(b["a0"]["k"], b["a0"]["n"]) for b in blocks]),
        "a1": _pool([(b["a1"]["k"], b["a1"]["n"]) for b in blocks]),
        "mu01_same_cells": _pool([(d["anchor"]["cells"]["mu01"]["k"],
                                   d["anchor"]["cells"]["mu01"]["n"]) for d in with_ctl]),
        "mu11_same_cells": _pool([(d["anchor"]["cells"]["mu11"]["k"],
                                   d["anchor"]["cells"]["mu11"]["n"]) for d in with_ctl]),
    }


def implied_copy_rate(mu: float, control: float, follow: float) -> dict:
    """c = (mu - a)/(follow - a) under the simulation's copy model, or why it cannot apply.

    In that model copying acts only on followers' donors, so mu <= follow; a pooled mu
    above the follow rate, or a ratio outside [0, 1], means the model does not describe
    the data and no copy rate is reported.
    """
    if mu > follow:
        return {"value": None, "applies": False,
                "reason": f"mu {mu:.3f} exceeds the follow rate {follow:.3f}, which the "
                          "copy model cannot produce"}
    if follow - control <= 0:
        return {"value": None, "applies": False,
                "reason": f"control rate {control:.3f} is at or above the follow rate"}
    ratio = (mu - control) / (follow - control)
    if not 0.0 <= ratio <= 1.0:
        return {"value": None, "applies": False, "raw": ratio,
                "reason": "ratio outside [0, 1]"}
    return {"value": ratio, "applies": True}


def copy_bounds(follow: float, controls: dict) -> dict:
    """What the controls bound about answer copying, each with the assumption it needs."""
    maot, amr = controls["matched_answer_only_text"], controls["answer_marker_removed"]
    mu01, mu11 = amr["mu01_same_cells"]["rate"], amr["mu11_same_cells"]["rate"]
    a0, a1 = amr["a0"]["rate"], amr["a1"]["rate"]
    return {
        "answer_only_rate_no_cue": maot["a0"]["rate"],
        "answer_only_rate_cue": maot["a1"]["rate"],
        "answer_only_reading": (
            "rate of the hinted option when the donor is only 'Answer: (hinted)' plus "
            "filler. If recipients copied a stated answer line whatever the reasoning, "
            "this would be at least the copy probability, so it upper-bounds "
            "reasoning-independent copying in each recipient frame."),
        "marker_gap_no_cue": mu01 - a0,
        "marker_gap_cue": mu11 - a1,
        "marker_gap_reading": (
            "mu01 (mu11) minus the same cells' answer-marker-removed a0 (a1): the drop in "
            "the hinted rate when the cued donor loses its answer markers. Subsets differ: "
            "the control is over applied items only (see subset_note)."),
        "implied_copy_rate_no_cue": implied_copy_rate(mu01, a0, follow),
        "implied_copy_rate_cue": implied_copy_rate(mu11, a1, follow),
        "implied_copy_reading": (
            "the simulation's copy model: copying acts only on followers' donors (whose "
            "answer line is the hinted option) and marker removal leaves the native "
            "content-driven rate, so c = (mu - a)/(follow - a). Model-based, not a bound."),
    }


def read_copy_controls(fits_root: Path, substrate: str) -> dict:
    """Experiment H: the stored falsifier controls pooled per question set."""
    cells = _load(fit_paths(fits_root, substrate))
    follow = _pool([_follow_counts(d) for d in cells])
    controls = {name: _control_entry(cells, name) for name in COPY_CONTROLS}
    mu = {c: _pool([(d["anchor"]["cells"][c]["k"], d["anchor"]["cells"][c]["n"])
                    for d in cells]) for c in ANCHOR_CELL_NAMES}
    return {
        "n_cells": len(cells),
        "follow": follow,
        "anchor_cells": mu,
        "controls": controls,
        "copy_bounds": copy_bounds(follow["rate"], controls),
        "subset_note": SUBSET_NOTE,
        "subset_restriction_possible": False,
    }
