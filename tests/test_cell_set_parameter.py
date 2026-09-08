"""The cell list is a parameter, and the 18-cell path must not move when 24 is added.

These are guards on the shape of the parametrization, not on any fitted number.
The numbers are checked by re-rendering the 18-cell document, which changes prose
lines and no value.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import cells18_fits as cf


def test_default_cell_set_is_the_eighteen_cell_list():
    assert cf.DEFAULT_CELL_SET == "18"
    assert cf.SUBSTRATE_CUES == cf.SUBSTRATE_CUES_18
    assert cf.CELLS == cf.cell_triples("18")
    assert len(cf.SUBSTRATE_CUES_18) == 6
    assert len(cf.CELLS) == 18


def test_the_twenty_four_cell_list_extends_the_eighteen_cell_one_in_order():
    """Group indices inside a model row come from this order, so it has to hold."""
    assert cf.SUBSTRATE_CUES_24[: len(cf.SUBSTRATE_CUES_18)] == cf.SUBSTRATE_CUES_18
    assert len(cf.SUBSTRATE_CUES_24) == 8
    assert len(cf.cell_triples("24")) == 24


def test_the_two_new_pairs_are_the_aqua_cue_families_that_were_missing():
    added = tuple(p for p in cf.SUBSTRATE_CUES_24 if p not in cf.SUBSTRATE_CUES_18)
    assert added == (("aqua_rat", "metadata"), ("aqua_rat", "grader-code"))


def test_every_pair_uses_a_known_substrate_and_cue_family():
    for substrate, cue in cf.SUBSTRATE_CUES_24:
        assert substrate in cf.SUBSTRATES
        assert cue in cf.CUE_FAMILIES


def test_an_unknown_cell_set_is_refused_rather_than_defaulted():
    try:
        cf.substrate_cues("19")
    except SystemExit as exc:
        assert "19" in str(exc)
    else:
        raise AssertionError("an unknown cell set must not fall back to a default")
