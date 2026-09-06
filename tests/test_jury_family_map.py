"""The panel rule, recomputed from the family map rather than trusted.

CONTRACT.md line 21 and PREREGISTRATION_jury_and_scale.md section 6.2 both state the panel
sizes and the five compositions with their subject counts. These tests recompute them from
the map so that a typo in either document, or a roster edit, fails here instead of silently
changing which judges vote on which model.
"""

from __future__ import annotations

import pytest

from experiments.jury import family_map as fm


def test_roster_has_eighteen_models_with_unique_names():
    assert len(fm.ROSTER) == 18
    assert len({e.name for e in fm.ROSTER}) == 18
    assert len({e.hf_id for e in fm.ROSTER}) == 18


def test_family_sizes_match_the_contract_line():
    sizes = {}
    for entry in fm.ROSTER:
        sizes[entry.family] = sizes.get(entry.family, 0) + 1
    assert sizes == {
        fm.QWEN: 4,
        fm.LLAMA: 4,
        fm.GEMMA: 2,
        fm.GPT_OSS: 2,
        fm.OLMO: 2,
        fm.MISTRAL: 2,
        fm.PHI: 1,
        fm.GLM: 1,
    }


def test_calibrated_strata_cover_fourteen_of_eighteen_rows():
    covered = sum(1 for e in fm.ROSTER if e.family in fm.CALIBRATED_STRATA)
    uncovered = sum(1 for e in fm.ROSTER if e.family in fm.UNCALIBRATED_FAMILIES)
    assert covered == 14
    assert uncovered == 4
    assert covered + uncovered == 18


def test_four_judges_from_four_distinct_families_with_pinned_revisions():
    assert len(fm.JUDGES) == 4
    assert len({j.family for j in fm.JUDGES}) == 4
    for judge in fm.JUDGES:
        assert len(judge.revision) == 40, judge.key
        assert set(judge.revision) <= set("0123456789abcdef"), judge.key


def test_twelve_subjects_get_a_panel_of_three_and_six_get_four():
    threes = [e.name for e in fm.ROSTER if fm.panel_size(e.name) == 3]
    fours = [e.name for e in fm.ROSTER if fm.panel_size(e.name) == 4]
    assert len(threes) == 12, threes
    assert len(fours) == 6, fours
    assert len(threes) + len(fours) == 18
    # The 6 with a panel of 4 are exactly the families that supply no judge.
    assert {fm.family_of(n) for n in fours} == {fm.OLMO, fm.MISTRAL, fm.PHI, fm.GLM}


def test_the_five_compositions_and_their_subject_counts():
    counts = fm.composition_counts()
    every = frozenset(fm.all_judges())
    expected = {
        every: 6,
        every - {"qwen3-32b"}: 4,
        every - {"llama-3.3-70b-fp8"}: 4,
        every - {"gemma-3-27b-it"}: 2,
        every - {"gpt-oss-20b"}: 2,
    }
    assert counts == expected
    assert len(counts) == 5
    assert sum(counts.values()) == 18


def test_no_judge_ever_votes_on_its_own_family():
    for entry in fm.ROSTER:
        for key in fm.routing(entry.name):
            assert fm.JUDGE_BY_KEY[key].family != entry.family


def test_routing_accepts_hf_ids_as_well_as_display_names():
    assert fm.routing("Qwen3-8B") == fm.routing("Qwen/Qwen3-8B")


def test_unknown_subject_raises():
    with pytest.raises(fm.RoutingError):
        fm.routing("some-model-not-on-the-roster")


def test_assert_panel_accepts_the_routed_panel_and_refuses_a_same_family_vote():
    fm.assert_panel("Qwen3-8B", list(fm.routing("Qwen3-8B")))
    with pytest.raises(fm.RoutingError, match="same-family"):
        fm.assert_panel("Qwen3-8B", list(fm.routing("Qwen3-8B")) + ["qwen3-32b"])
    with pytest.raises(fm.RoutingError, match="PANEL VIOLATION"):
        fm.assert_panel("Qwen3-8B", ["gemma-3-27b-it"])


# --- the seeded 20 percent all-judge subsample ---------------------------------


def _frame(n: int) -> list[str]:
    return [f"item-{i:04d}" for i in range(n)]


def test_subsample_is_exactly_twenty_percent_and_deterministic():
    frame = _frame(200)
    first = fm.draw_all_judge_subsample("Qwen", frame, seed=7)
    second = fm.draw_all_judge_subsample("Qwen", frame, seed=7)
    assert first == second
    assert len(first) == 40
    assert set(first) <= set(frame)


def test_subsample_differs_by_stratum_and_by_seed():
    frame = _frame(200)
    qwen = fm.draw_all_judge_subsample("Qwen", frame, seed=7)
    llama = fm.draw_all_judge_subsample("Llama", frame, seed=7)
    other_seed = fm.draw_all_judge_subsample("Qwen", frame, seed=8)
    assert qwen != llama
    assert qwen != other_seed


def test_subsample_flag_on_a_row_agrees_with_the_drawn_set():
    frame = _frame(50)
    drawn = set(fm.draw_all_judge_subsample("Gemma", frame, seed=7))
    flags = {i for i in frame if fm.is_all_judge_row("Gemma", i, frame, seed=7)}
    assert flags == drawn
    assert len(flags) == 10


def test_subsample_refuses_a_frame_with_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        fm.draw_all_judge_subsample("Qwen", ["a", "a", "b"], seed=7)
