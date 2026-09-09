"""The cell list is a parameter, and the 18-cell path must not move when 24 is added.

These are guards on the shape of the parametrization, not on any fitted number.
The numbers are checked by re-rendering the 18-cell document, which changes prose
lines and no value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


# --------------------------------------------------------------- the wider cell sets
#
# Added 2026-09-09 when the lane widened from 24 cells to 72. The checks below are
# written against CELL_SETS itself rather than against a list of names, so a cell set
# added later is covered without anyone remembering to extend this file. The tests above
# name "18" and "24" literally, which is why adding "36" and "72" did not fail anything.


def test_every_cell_set_has_a_model_list():
    """A cell set fixes its subjects as well as its pairs; the two maps must agree."""
    assert set(cf.CELL_SETS) == set(cf.CELL_SET_MODELS), (
        "CELL_SETS and CELL_SET_MODELS have drifted apart, so some cell set has pairs "
        "but no subjects or the other way round"
    )


@pytest.mark.parametrize("cell_set", sorted(cf.CELL_SETS))
def test_cell_count_is_models_times_pairs(cell_set):
    triples = cf.cell_triples(cell_set)
    assert len(triples) == len(cf.models(cell_set)) * len(cf.substrate_cues(cell_set))
    assert len(set(triples)) == len(triples), "duplicate cells in the set"


@pytest.mark.parametrize("cell_set", sorted(cf.CELL_SETS))
def test_every_cell_names_a_known_substrate_and_cue(cell_set):
    for _model, substrate, cue in cf.cell_triples(cell_set):
        assert substrate in cf.SUBSTRATES
        assert cue in cf.CUE_FAMILIES


@pytest.mark.parametrize("cell_set,expected", [("18", 18), ("24", 24), ("36", 36), ("72", 72)])
def test_the_named_sets_are_the_size_their_name_claims(cell_set, expected):
    assert len(cf.cell_triples(cell_set)) == expected


def test_the_substrate_order_is_append_only():
    """SUBSTRATES.index() is written into every fit as the substrate column.

    Inserting a level rather than appending one would silently re-label every fit
    already banked: a cell recorded as substrate 1 would come to mean something else.
    arc_challenge must stay 0 and aqua_rat must stay 1 for as long as any fit produced
    before 2026-09-09 is still read.
    """
    assert cf.SUBSTRATES[0] == "arc_challenge"
    assert cf.SUBSTRATES[1] == "aqua_rat"
    assert "logiqa2" in cf.SUBSTRATES


def test_the_72_set_is_six_subjects_on_the_full_block():
    assert len(cf.models("72")) == 6
    assert len(cf.substrate_cues("72")) == 12
    # only subjects whose data exists: the two held models must not appear
    assert "gpt-oss-20b" not in cf.models("72"), "R11 holds gpt-oss-20b; it has no cells"
    assert "deepseek-r1-0528-qwen3-8b" not in cf.models("72"), (
        "R12(4) holds deepseek-r1-0528-qwen3-8b; it has no cells"
    )


def test_the_wider_sets_extend_rather_than_replace_the_narrower_ones():
    """24 contains 18, 36 contains 24, and 72 is 36's pairs on more subjects."""
    assert set(cf.substrate_cues("18")) <= set(cf.substrate_cues("24"))
    assert set(cf.substrate_cues("24")) <= set(cf.substrate_cues("36"))
    assert cf.substrate_cues("72") == cf.substrate_cues("36")
    assert set(cf.models("36")) <= set(cf.models("72"))


def test_an_unknown_cell_set_is_rejected_by_both_accessors():
    with pytest.raises(SystemExit):
        cf.substrate_cues("99")
    with pytest.raises(SystemExit):
        cf.models("99")
