"""The verdict block's probability must be read at rho = 0, not at the grid's first point.

Found by the cells18 lane on 2026-09-08: ``column_b`` wrote ``prob_nie_above_0.15_at_rho_zero``
from ``prob_above[0]``, which on the signed grid is rho = -0.945. The verdict used the right
index, so only the reported field was wrong. This pins the helper to the rho = 0 index and to
the same value the ``rho_star_decision`` block records.
"""

import numpy as np

from experiments import wave1_fits as w1


def test_rho_zero_index_points_at_zero_on_the_signed_grid():
    assert float(w1.RHO_GRID_SIGNED[w1.RHO_ZERO_INDEX]) == 0.0
    assert w1.RHO_ZERO_INDEX > 0  # the signed grid starts at its negative edge


def test_verdict_probability_is_read_at_rho_zero_not_at_the_grid_edge():
    # Strictly increasing along the grid, so the edge and the zero point cannot coincide.
    prob_above = np.linspace(0.0, 1.0, len(w1.RHO_GRID_SIGNED))
    block = w1._verdict_block(prob_above, load_bearing=False)
    assert block["prob_nie_above_0.15_at_rho_zero"] == float(prob_above[w1.RHO_ZERO_INDEX])
    assert block["prob_nie_above_0.15_at_rho_zero"] != float(prob_above[0])
    assert block["verdict"] == "unresolved"
    assert block["load_bearing_at_rho_zero"] is False


def test_verdict_block_keeps_its_labels_and_rule_text():
    prob_above = np.full(len(w1.RHO_GRID_SIGNED), 0.99)
    block = w1._verdict_block(prob_above, load_bearing=True)
    assert set(block) == {"rule", "prob_nie_above_0.15_at_rho_zero", "load_bearing_at_rho_zero", "verdict"}
    assert block["verdict"] == "load-bearing at rho=0"
    assert "prereg 2.5" in block["rule"]


def _synthetic_table(n_items: int = 160, seed: int = 7) -> dict:
    """A small mediated world in ``build_table``'s layout: two rows per item, clean arm
    outcome pinned at 0 like the real cells, and mediation weak enough that P(NIE > 0.15)
    is 0 at rho = 0 while the grid's negative edge inflates it to 1 (checked on 2026-09-08
    with n_bootstrap 6: edge 1.0, zero 0.0), so reading the wrong index cannot pass."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_items):
        m0 = float(rng.normal(0.30, 0.20))
        m1 = float(rng.normal(0.40, 0.20))
        p1 = 0.5 * (1.0 + np.tanh(2.5 * (m1 - 0.45)))
        items.append({
            "anchor": None, "m0": m0, "y0": 0, "m1": m1, "y1": int(rng.random() < p1),
            "depth0": 0, "depth1": 3,
        })
    n = len(items)
    X = np.empty(2 * n); M = np.empty(2 * n); Y = np.empty(2 * n, dtype=int)
    item_of_row = np.empty(2 * n, dtype=int)
    for i, it in enumerate(items):
        X[2 * i], M[2 * i], Y[2 * i], item_of_row[2 * i] = 0.0, it["m0"], it["y0"], i
        X[2 * i + 1], M[2 * i + 1], Y[2 * i + 1], item_of_row[2 * i + 1] = 1.0, it["m1"], it["y1"], i
    return {"items": items, "X": X, "M": M, "Y": Y, "item_of_row": item_of_row,
            "denominators": {"n_records_read": n, "n_entered": n}}


def test_column_b_reports_the_same_rho_zero_probability_in_both_blocks():
    out = w1.column_b(_synthetic_table(), n_bootstrap=6)
    reported = out["verdict"]["prob_nie_above_0.15_at_rho_zero"]
    decision = out["rho"]["rho_star_decision"]["prob_nie_above_threshold_at_rho_zero"]
    assert reported == decision
    # The coarse sweep samples every 21st grid point, 19 points, rho = 0 in the middle.
    curve = np.asarray(out["rho"]["sweep"]["nie_prob_above_threshold"], dtype=float)
    assert len(curve) == 19 and float(w1.RHO_GRID_SIGNED[9 * 21]) == 0.0
    assert curve[0] != curve[9], "vacuous world: the negative edge equals rho = 0"
    assert reported == curve[9]
