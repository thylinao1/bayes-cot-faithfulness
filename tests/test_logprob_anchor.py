"""Job A of the outcome-scale note: the drop rules and the contrast arithmetic.

Two things are pinned here. First, every required field check of
``docs/OUTCOME-SCALE-NOTE.md`` part 4.5 step 2 drops its ITEM and is counted under its own
reason: the synthetic record set below carries exactly one item failing each check, so a
check that stopped firing would show up as a missing reason rather than as a silently
larger denominator. Second, the four cell means and the five section 22 contrasts are
computed on hand-picked margins whose answers can be checked by eye, and the contrast
weights are checked against a committed ``fit.json`` so the mirror of
``wave1_fits.anchor_block``'s local contrast table cannot drift away from it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from experiments import logprob_anchor as la

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("A", "B", "C", "D")
HINT = "A"
# Four items whose per-cell means are exact: mu00 -3, mu01 -1, mu10 0, mu11 +2.
GOOD_MARGINS = {
    "mu00": (-6.0, -4.0, -2.0, 0.0),
    "mu01": (-4.0, -2.0, 0.0, 2.0),
    "mu10": (-3.0, -1.0, 1.0, 3.0),
    "mu11": (0.0, 1.0, 2.0, 5.0),
}
GOOD_MASSES = (1e-18, 1e-17, 1e-16, 1e-15)


def block(margin: float, mass: float, **override) -> dict:
    out = {
        "intervention_level": "logit",
        "outcome_scale": "logprob_margin",
        "answer_logprobs": {letter: -30.0 for letter in LABELS},
        "logprob_source_token": {letter: letter for letter in LABELS},
        "renormalized_over_letters": {letter: 0.25 for letter in LABELS},
        "letter_probability_mass": mass,
        "logprob_margin": margin,
        "target_letter": HINT,
        "method": "prompt_logprobs",
    }
    out.update(override)
    return out


def record(margins: dict, mass: float, **override) -> dict:
    out = {
        "source_file": "arms_transcripts_Synthetic.json",
        "hint_label": HINT,
        "choices": ["one", "two", "three", "four"],
        "anchor": {
            "cells": {
                cell: {"answer": "C", "y": 0, "logprob": block(margins[cell], mass)}
                for cell in la.ANCHOR_CELLS
            }
        },
    }
    out.update(override)
    return out


def good_records() -> list[dict]:
    return [
        record({cell: GOOD_MARGINS[cell][i] for cell in la.ANCHOR_CELLS}, GOOD_MASSES[i])
        for i in range(4)
    ]


def failing_records() -> list[tuple[str, dict]]:
    """One record per required check, each failing exactly that check."""
    flat = {cell: -1.0 for cell in la.ANCHOR_CELLS}

    def with_cell(**override) -> dict:
        rec = record(flat, 1e-16)
        rec["anchor"]["cells"]["mu01"]["logprob"] = block(-1.0, 1e-16, **override)
        return rec

    no_anchor = record(flat, 1e-16)
    no_anchor.pop("anchor")
    no_block = record(flat, 1e-16)
    no_block["anchor"]["cells"]["mu10"].pop("logprob")
    short_set = with_cell(answer_logprobs={letter: -30.0 for letter in ("A", "B", "C")})
    bad_token = with_cell(
        logprob_source_token={"A": "A", "B": "X", "C": "C", "D": "D"}
    )
    return [
        ("no_hint_label", record(flat, 1e-16, hint_label=None)),
        ("no_choices", record(flat, 1e-16, choices=[])),
        ("no_anchor_block", no_anchor),
        ("missing_logprob_block", no_block),
        ("intervention_level_not_logit", with_cell(intervention_level="text")),
        ("outcome_scale_not_logprob_margin", with_cell(outcome_scale="binary_follow")),
        ("unavailable_reason_present", with_cell(unavailable_reason="server refused")),
        ("target_letter_mismatch", with_cell(target_letter="B")),
        ("answer_logprobs_label_set_mismatch", short_set),
        ("source_token_mismatch", bad_token),
        ("logprob_margin_null", with_cell(logprob_margin=None)),
    ]


def test_every_required_check_drops_its_item_under_its_own_reason():
    failing = failing_records()
    records = good_records() + [rec for _, rec in failing]
    items, drops = la.collect_items(records)

    assert len(items) == 4, "the four complete items must survive every check"
    assert drops["items_dropped"] == len(failing)
    expected = Counter({name: 1 for name, _ in failing})
    expected["items_dropped"] = len(failing)
    assert drops == expected


def test_a_dropped_item_is_dropped_whole_and_not_cell_by_cell():
    # The failure sits in mu01 only; the item must not contribute its other three cells.
    rec = record({cell: -1.0 for cell in la.ANCHOR_CELLS}, 1e-16)
    rec["anchor"]["cells"]["mu01"]["logprob"] = block(-1.0, 1e-16, target_letter="B")
    items, drops = la.collect_items(good_records() + [rec])
    assert len(items) == 4
    assert drops["target_letter_mismatch"] == 1
    for cell in la.ANCHOR_CELLS:
        assert la.cell_means(items)[cell]["n"] == 4


def test_cell_means_are_the_hand_computed_item_means():
    items, _ = la.collect_items(good_records())
    means = la.cell_means(items)
    assert means["mu00"]["mean"] == pytest.approx(-3.0)
    assert means["mu01"]["mean"] == pytest.approx(-1.0)
    assert means["mu10"]["mean"] == pytest.approx(0.0)
    assert means["mu11"]["mean"] == pytest.approx(2.0)
    assert means["mu00"]["min"] == pytest.approx(-6.0)
    assert means["mu11"]["max"] == pytest.approx(5.0)
    assert all(means[cell]["n"] == 4 for cell in la.ANCHOR_CELLS)


def test_the_five_contrasts_are_the_hand_computed_differences():
    items, _ = la.collect_items(good_records())
    means = la.cell_means(items)
    boot = la.bootstrap_cells(items, seed=20260907, n_replicates=20)
    contrasts = la.contrasts_from(means, boot)
    points = {name: block_["point"] for name, block_ in contrasts.items()}
    assert points["text_source_given_cued_recipient"] == pytest.approx(2.0)  # 2 - 0
    assert points["text_source_given_clean_recipient"] == pytest.approx(2.0)  # -1 + 3
    assert points["cue_effect_given_clean_donor"] == pytest.approx(3.0)  # 0 + 3
    assert points["interaction"] == pytest.approx(0.0)  # 2 - 0 + 1 - 3
    assert points["joint_replay_regime"] == pytest.approx(5.0)  # 2 + 3
    for name in la.ANCHOR_CONTRAST_DEF:
        assert contrasts[name]["lo"] <= contrasts[name]["hi"]


def test_y_is_the_stored_margin_and_is_never_recomputed():
    # The stored margin disagrees with the renormalized distribution beside it. The
    # extraction must report the stored value, which is what part 4.5 step 3 fixes.
    items = good_records()
    items[0]["anchor"]["cells"]["mu00"]["logprob"]["logprob_margin"] = -99.0
    items[0]["anchor"]["cells"]["mu00"]["logprob"]["renormalized_over_letters"] = {
        letter: 0.25 for letter in LABELS
    }
    kept, _ = la.collect_items(items)
    assert kept[0]["y"]["mu00"] == pytest.approx(-99.0)


def test_mass_summary_carries_min_median_max_and_its_denominator():
    items, _ = la.collect_items(good_records())
    mass = la.mass_summary(items)["mu00"]
    assert mass["n"] == 4 and mass["n_items"] == 4
    assert mass["min"] == pytest.approx(1e-18)
    assert mass["max"] == pytest.approx(1e-15)
    assert mass["median"] == pytest.approx((1e-17 + 1e-16) / 2.0)
    assert mass["n_below_0.01"] == 4


def test_a_space_marked_source_token_still_decodes_to_its_letter():
    rec = record({cell: -1.0 for cell in la.ANCHOR_CELLS}, 1e-16)
    for cell in la.ANCHOR_CELLS:
        rec["anchor"]["cells"][cell]["logprob"]["logprob_source_token"] = {
            letter: f"▁{letter}" for letter in LABELS
        }
    items, drops = la.collect_items([rec])
    assert len(items) == 1 and drops == Counter()


def test_the_bootstrap_is_seeded_and_resamples_items_not_cells():
    items, _ = la.collect_items(good_records())
    first = la.bootstrap_cells(items, seed=17, n_replicates=25)
    again = la.bootstrap_cells(items, seed=17, n_replicates=25)
    other = la.bootstrap_cells(items, seed=18, n_replicates=25)
    for cell in la.ANCHOR_CELLS:
        assert list(first[cell]) == list(again[cell])
        assert first[cell].size == 25
    assert any(list(first[c]) != list(other[c]) for c in la.ANCHOR_CELLS)


def test_contrast_weights_reproduce_the_binary_contrasts_of_a_committed_fit():
    path = (
        ROOT
        / "experiments/results/cells18-fits/qwen3-8b/arc_challenge/stated-hint/fit.json"
    )
    if not path.exists():  # pragma: no cover - the artifact is committed
        pytest.skip("the cells18 artifact of record is not present")
    anchor = json.loads(path.read_text())["anchor"]
    rates = {cell: anchor["cells"][cell]["rate"] for cell in la.ANCHOR_CELLS}
    for name, terms in la.ANCHOR_CONTRAST_DEF.items():
        recomputed = sum(sign * rates[cell] for cell, sign in terms)
        assert recomputed == pytest.approx(anchor["contrasts"][name]["point"], abs=1e-12)


def test_extract_cell_writes_the_denominators_and_both_scales(tmp_path):
    records = good_records() + [rec for _, rec in failing_records()]
    records_path = tmp_path / "transcripts.jsonl"
    records_path.write_text(
        "".join(json.dumps(rec) + "\n" for rec in records)
        + json.dumps({"source_file": "specificity_transcripts_Synthetic.json"})
        + "\n"
    )
    fit = {
        "cell": "synthetic/arc_challenge/stated-hint",
        "model_slug": "synthetic",
        "model": "Synthetic/Model",
        "substrate": "arc_challenge",
        "cue_family": "stated-hint",
        "job_id": "000000",
        "outcome_scale": "binary_follow",
        "intervention_level": "text",
        "record_hashes": {"transcripts.jsonl": "not-the-real-digest"},
        "column_b": {"bootstrap": {"seed": 20260907}},
        "anchor": {
            "cells": {cell: {"k": 1, "n": 4, "rate": 0.25} for cell in la.ANCHOR_CELLS},
            "contrasts": {
                name: {"point": 0.0, "lo": 0.0, "hi": 0.0}
                for name in la.ANCHOR_CONTRAST_DEF
            },
        },
    }
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(fit))

    out = la.extract_cell(fit_path, records_path)
    assert out["denominators"]["n_records_read"] == 15
    assert out["denominators"]["n_items_kept"] == 4
    assert out["denominators"]["n_items_dropped"] == 11
    assert out["file_counts"]["n_lines_other_source"] == {
        "specificity_transcripts_Synthetic.json": 1
    }
    assert out["bootstrap"]["seed"] == 20260907
    assert out["outcome_scale"] == "logprob_margin"
    assert out["records_match_fit_json"] is False
    assert out["cells"]["mu11"]["mean"] == pytest.approx(2.0)
    assert out["contrasts"]["joint_replay_regime"]["point"] == pytest.approx(5.0)
    assert out["binary_from_fit_json"]["cells"]["mu00"]["rate"] == 0.25
    assert "best other" in out["y_definition"].lower()


def test_the_report_prints_both_scales_and_promotes_nothing(tmp_path):
    records_path = tmp_path / "transcripts.jsonl"
    records_path.write_text("".join(json.dumps(r) + "\n" for r in good_records()))
    fit = {
        "cell": "qwen3-8b/arc_challenge/stated-hint",
        "model_slug": "qwen3-8b",
        "model": "Qwen/Qwen3-8B",
        "substrate": "arc_challenge",
        "cue_family": "stated-hint",
        "job_id": "826733",
        "outcome_scale": "binary_follow",
        "intervention_level": "text",
        "record_hashes": {},
        "column_b": {"bootstrap": {"seed": 20260907}},
        "anchor": {
            "cells": {cell: {"k": 1, "n": 4, "rate": 0.25} for cell in la.ANCHOR_CELLS},
            "contrasts": {
                name: {"point": 0.5, "lo": 0.25, "hi": 0.75}
                for name in la.ANCHOR_CONTRAST_DEF
            },
        },
    }
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(fit))
    doc = la.render([la.extract_cell(fit_path, records_path)])

    assert "binary follow scale" in doc and "logprob scale" in doc
    assert "+0.5000 [+0.2500, +0.7500]" in doc  # the binary contrast printed beside
    assert "+5.0000" in doc  # joint_replay_regime on the logprob scale
    assert "no verdict" in doc
    for banned in ("—", "–"):
        assert banned not in doc
