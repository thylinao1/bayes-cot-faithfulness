"""The logit sidecar loader, A5.4's gate G1, and the arithmetic of the logit column B.

Three things are pinned here and each one is a thing that can go wrong quietly.

The MERGE. A sidecar entry belongs to a record only when its position, its content key and
its source file all agree (``docs/LOGIT-PASS.md`` section 4). The happy path is checked on
a synthetic cell; a single changed question, which moves that record's ``record_key`` and
nothing else, has to refuse the WHOLE merge and count the mismatch by name; and a cell with
no sidecar has to stay on the text level with A5.4 condition 3 failing, which is the state
18 of 18 cells were in before the pass existed.

The GATE. All six conditions on both cells, so that "condition 3 is now satisfiable" is a
measured fact and not a claim, and so that the failing path still prints every condition
with its numerator and its denominator.

The ARITHMETIC. ``logit_column_b`` on a design whose ordinary least squares solution is
exact, so alpha, beta, gamma, the three effects, both arm variances and ``rho*_point`` are
compared against numbers worked out by hand rather than against the code's own output.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import cells18_fits as c18  # noqa: E402
import wave1_fits as w1  # noqa: E402

ARMS_FILE = "arms_transcripts_Test_Model.json"
SIDECAR_FILE = "arms_transcripts_Test_Model.logit.json"
ENDPOINT = "http://127.0.0.1:8000/v1"
LABELS = ("A", "B", "C", "D")

# The hand-computable design of the arithmetic test, one entry per item type:
# (clean mediator, hinted mediator, clean outcome, hinted outcome).
# Y = 1 + 3 X + 5 M + e with e = +1 on the first two item types and -1 on the last two,
# which is orthogonal to (1, X, M) on this design, so the least squares solution is exactly
# alpha0 = 1, alpha = 3, beta = 5 and the residual root mean square is exactly 1.
DESIGN = (
    (0.0, 2.0, 2.0, 15.0),
    (1.0, 3.0, 7.0, 20.0),
    (0.0, 2.0, 0.0, 13.0),
    (1.0, 3.0, 5.0, 18.0),
)
DESIGN_REPEATS = 25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anchor(mass: float) -> dict:
    return {
        "cells": {
            cell: {
                "y": 0,
                "logprob": {
                    "method": "prompt_logprobs",
                    "letter_probability_mass": mass,
                    "logprob_margin": -1.0,
                },
            }
            for cell in w1.ANCHOR_CELLS
        }
    }


def _record(i: int, m0: float, m1: float, clean_follows: bool = False) -> dict:
    """One text-level arms record with every field the two tables read."""
    return {
        "source_file": ARMS_FILE,
        "question": f"question {i}",
        "choices": [f"choice {i}{letter}" for letter in LABELS],
        "answer_label": "A",
        "hint_label": "B",
        "cue_text": f"the answer is B (item {i})",
        "cue_prepended": True,
        "clean_answer": "B" if clean_follows else "A",
        "hinted_answer": "B",
        "followed": True,
        "clean_curve": {"curve_area": m0, "commitment_depth": 1},
        "hinted_curve": {"curve_area": m1, "commitment_depth": 2},
        "acknowledged": False,
        "silent": True,
        "anchor": _anchor(1e-15 * (i + 1)),
        "intervention_level": "text",
        "outcome_scale": "binary_follow",
        "answer_logprobs": None,
        "logprob_source_token": None,
    }


def _block(margin: float, mass: float) -> dict:
    return {
        "answer_logprobs": {letter: -1.0 for letter in LABELS},
        "logprob_source_token": {letter: letter for letter in LABELS},
        "renormalized_over_letters": list(LABELS),
        "letter_probability_mass": mass,
        "logprob_margin": margin,
        "target_letter": "B",
        "method": "prompt_logprobs",
    }


def _sidecar_entry(index: int, record: dict, y0: float, y1: float, source_sha: str) -> dict:
    return {
        **record,
        "clean_answer_logprob": _block(y0, 1e-15 * (index + 1)),
        "hinted_answer_logprob": _block(y1, 2e-15 * (index + 1)),
        "intervention_level": "logit",
        "outcome_scale": "logprob_margin",
        "answer_logprobs": {"clean": {"B": -1.0}, "hinted": {"B": -1.0}},
        "logprob_source_token": {"clean": {"B": "B"}, "hinted": {"B": "B"}},
        "logit_pass": {
            "pass_version": "logit-pass-1",
            "record_index": index,
            "record_key": w1.record_key(record),
            "source_file": ARMS_FILE,
            "source_sha256": source_sha,
            "model": "Test/Model",
            "endpoint": ENDPOINT,
        },
    }


def _unit_check(n_reads: int, base_url: str = ENDPOINT) -> dict:
    return {
        "base_url": base_url,
        "passed": True,
        "hard_failures": [],
        "n_probes": n_reads,
        "n_probes_completed": n_reads,
        "n_incomplete_reads": 0,
        "results": [
            {
                "n_letters_requested": len(LABELS),
                "n_letters_scored": len(LABELS),
                "n_tokens_matching_letter": len(LABELS),
            }
            for _ in range(n_reads)
        ],
    }


def build_cell(
    tmp_path: Path,
    *,
    with_sidecar: bool = True,
    break_record_key_at: int | None = None,
    repeats: int = DESIGN_REPEATS,
) -> tuple[Path, list[dict], dict]:
    """A synthetic cell directory carrying everything the A5 lane reads.

    Returns the cell directory, the text-level records as the loader would read them, and
    the cell's ``run_meta.json`` contents.
    """
    cell = tmp_path / "cell"
    cell.mkdir(parents=True, exist_ok=True)

    rows = [DESIGN[i % len(DESIGN)] for i in range(repeats * len(DESIGN))]
    records = [_record(i, m0, m1) for i, (m0, m1, _, _) in enumerate(rows)]
    (cell / ARMS_FILE).write_text(json.dumps(records))
    with (cell / "transcripts.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
        # One A9 specificity-holdout row, which the loader has always dropped.
        fh.write(json.dumps({"source_file": "specificity_transcripts_Test_Model.json"}) + "\n")

    meta = {
        "model": "Test/Model",
        "substrate": "arc_challenge",
        "cue_family": "stated-hint",
        "job_id": "000001",
        "intervention_level": "text",
        "outcome_scale": "binary_follow",
        "vllm_version": "0.28.0",
        "served_model_list": ["Test/Model"],
    }
    (cell / "run_meta.json").write_text(json.dumps(meta))
    (cell / "logprob_check.json").write_text(json.dumps(_unit_check(2)))

    if not with_sidecar:
        return cell, records, meta

    source_sha = _sha256(cell / ARMS_FILE)
    sidecar = [
        _sidecar_entry(i, rec, rows[i][2], rows[i][3], source_sha)
        for i, rec in enumerate(records)
    ]
    if break_record_key_at is not None:
        # Change the question of ONE source record after the sidecar was written. Nothing
        # else moves: the index still lines up and the file sha256 is recomputed, so the
        # only check that can catch it is the content key.
        records[break_record_key_at] = dict(
            records[break_record_key_at], question="a different question"
        )
        (cell / ARMS_FILE).write_text(json.dumps(records))
        new_sha = _sha256(cell / ARMS_FILE)
        for entry in sidecar:
            entry["logit_pass"]["source_sha256"] = new_sha
        with (cell / "transcripts.jsonl").open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    (cell / SIDECAR_FILE).write_text(json.dumps(sidecar))
    (cell / "logit_check.json").write_text(json.dumps(_unit_check(2 * len(records))))
    clean = [r[2] for r in rows]
    hinted = [r[3] for r in rows]
    (cell / "logit_pass_meta.json").write_text(
        json.dumps(
            {
                "endpoint": ENDPOINT,
                "intervention_level": "logit",
                "outcome_scale": "logprob_margin",
                "n_items_scored": len(records),
                "n_items_dropped": 0,
                "assert_records_scaled_checked": len(records),
                "clean_arm_margin": {
                    "n": len(clean),
                    "mean": float(np.mean(clean)),
                    "variance": float(np.var(clean)),
                },
                "hinted_arm_margin": {
                    "n": len(hinted),
                    "mean": float(np.mean(hinted)),
                    "variance": float(np.var(hinted)),
                },
                "randomized_arm_difference_in_the_margin": float(
                    np.mean(hinted) - np.mean(clean)
                ),
            }
        )
    )
    return cell, records, meta


# --------------------------------------------------------------------------- #
# The loader and the merge
# --------------------------------------------------------------------------- #
def test_load_records_keeps_the_text_level_rows_and_counts_the_sidecar_rows(tmp_path):
    """The doubling guard of docs/LOGIT-PASS.md section 6.1.

    A rebuilt ``transcripts.jsonl`` in a cell that has a sidecar carries every item twice,
    once from each file, and both copies' ``source_file`` starts with ``arms_transcripts``.
    Keeping both would run the text-level table on 2n rows with no sign of it anywhere.
    """
    cell, records, _ = build_cell(tmp_path)
    path = cell / "transcripts.jsonl"
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(dict(rec, source_file=SIDECAR_FILE)) + "\n")

    arms, counts = w1.load_records(path)

    assert len(arms) == len(records)
    assert counts["n_lines_kept"] == len(records)
    assert counts["n_lines_logit_sidecar"] == len(records)
    assert counts["n_lines_other_source"] == {"specificity_transcripts_Test_Model.json": 1}


def test_sidecar_merges_into_the_records_it_was_computed_from(tmp_path):
    cell, records, _ = build_cell(tmp_path)
    sidecar, info = w1.load_logit_sidecar(cell)
    assert info["status"] == "present"
    assert info["n_sidecar_records"] == len(records)

    merged, report = w1.merge_logit_sidecar(records, sidecar, cell)

    assert report["merged"] is True
    assert report["n_mismatches"] == 0
    assert report["mismatches_by_reason"] == {}
    assert report["n_entries_keyed"] == len(records)
    assert report["n_text_level_records_without_a_sidecar_entry"] == 0
    assert report["source_file_check"]["status"].startswith("the sidecar names")

    for rec in merged:
        assert rec["intervention_level"] == "logit"
        assert rec["outcome_scale"] == "logprob_margin"
        assert rec["clean_answer_logprob"]["logprob_margin"] is not None
        assert rec["hinted_answer_logprob"]["logprob_margin"] is not None
        assert rec["logprob_source_token"] is not None

    # The originals are untouched: the text-level table is fitted on these objects and a
    # merge that mutated them would change the row it was printed beside.
    for rec in records:
        assert rec["intervention_level"] == "text"
        assert "clean_answer_logprob" not in rec


def test_a_changed_record_key_refuses_the_whole_merge_and_counts_it(tmp_path):
    cell, records, _ = build_cell(tmp_path, break_record_key_at=3)
    sidecar, _ = w1.load_logit_sidecar(cell)

    with pytest.raises(w1.LogitMergeError) as excinfo:
        w1.merge_logit_sidecar(records, sidecar, cell)

    report = excinfo.value.report
    assert report["merged"] is False
    assert report["mismatches_by_reason"] == {"record_key_mismatch": 1}
    assert report["n_mismatches"] == 1
    assert report["n_entries_keyed"] == len(records) - 1
    assert report["mismatch_examples"][0]["reason"] == "record_key_mismatch"
    assert report["mismatch_examples"][0]["sidecar_position"] == 3
    # Refused whole: not one record received a logit field.
    assert all("clean_answer_logprob" not in rec for rec in records)


def test_a_sidecar_naming_other_bytes_is_refused(tmp_path):
    cell, records, _ = build_cell(tmp_path)
    sidecar, _ = w1.load_logit_sidecar(cell)
    for entry in sidecar:
        entry["logit_pass"]["source_sha256"] = "0" * 64

    with pytest.raises(w1.LogitMergeError) as excinfo:
        w1.merge_logit_sidecar(records, sidecar, cell)

    assert excinfo.value.report["mismatches_by_reason"] == {"source_sha256_mismatch": 1}


def test_a_missing_sidecar_leaves_the_cell_at_the_text_level(tmp_path):
    cell, records, _ = build_cell(tmp_path, with_sidecar=False)
    sidecar, info = w1.load_logit_sidecar(cell)

    assert sidecar is None
    assert info["status"] == "absent"

    loaded = c18.load_cell_logit(cell, records)
    assert loaded["merge"]["merged"] is False
    assert loaded["table"] is None
    assert loaded["merged_records"] is records


# --------------------------------------------------------------------------- #
# A5.4's gate G1, on a cell with the pass and on a cell without it
# --------------------------------------------------------------------------- #
def test_g1_clears_all_six_conditions_on_a_merged_cell(tmp_path):
    cell, records, meta = build_cell(tmp_path)
    row = c18.logit_level_row(cell, records, meta, n_bootstrap=20)
    gate = row["gate_G1"]

    assert gate["failing_conditions"] == []
    assert gate["eligible"] is True
    assert row["printed"] is True

    conditions = gate["conditions"]
    n = len(records)
    assert conditions["1_unit_check_passed"]["this_cell_logit_check"]["holds"] is True
    assert (
        conditions["3_records_carry_the_logit_scale"][
            "n_records_with_intervention_level_logit"
        ]
        == f"{n}/{n}"
    )
    assert conditions["3_records_carry_the_logit_scale"]["assert_records_scaled_checked"] == n
    assert conditions["4_clean_arm_outcome_variance_positive"][
        "clean_arm_outcome_variance"
    ] == pytest.approx(7.25)
    assert conditions["5_te_logit_equals_arm_difference"]["abs_difference"] < 1e-9
    assert conditions["6_letter_probability_mass_summarised"]["on_the_arms_of_this_row"][
        "overall"
    ]["n"] == 2 * n

    # The row itself, printed only because the gate cleared.
    assert row["column_b"]["verdict"] == c18.LOGIT_VERDICT
    assert row["bridge"]["computable"] is True
    assert row["letter_probability_mass"]["overall"]["n"] == 2 * n


def test_g1_fails_condition_3_with_no_sidecar_and_still_prints_every_condition(tmp_path):
    cell, records, meta = build_cell(tmp_path, with_sidecar=False)
    row = c18.logit_level_row(cell, records, meta, n_bootstrap=20)
    gate = row["gate_G1"]

    assert gate["eligible"] is False
    assert gate["failing_conditions"] == [
        "3_records_carry_the_logit_scale",
        "4_clean_arm_outcome_variance_positive",
        "5_te_logit_equals_arm_difference",
    ]
    # Nothing partial is published from a failing row.
    assert row["printed"] is False
    assert row["column_b"] is None
    assert row["bridge"] is None
    assert row["letter_probability_mass"] is None

    conditions = gate["conditions"]
    n = len(records)
    assert set(conditions) == set(c18.G1_CONDITIONS)
    assert conditions["1_unit_check_passed"]["holds"] is True
    assert conditions["2_pinned_self_hosted_endpoint"]["holds"] is True
    assert (
        conditions["3_records_carry_the_logit_scale"][
            "n_records_with_intervention_level_logit"
        ]
        == f"0/{n}"
    )
    assert conditions["4_clean_arm_outcome_variance_positive"][
        "n_arms_rows_with_a_margin"
    ] == f"0/{n}"
    assert conditions["6_letter_probability_mass_summarised"]["holds"] is True


def test_a_refused_merge_fails_the_gate_and_leaves_the_text_level_row_alone(tmp_path):
    cell, records, meta = build_cell(tmp_path, break_record_key_at=0)
    row = c18.logit_level_row(cell, records, meta, n_bootstrap=20)

    assert row["printed"] is False
    assert row["merge"]["merged"] is False
    assert row["merge"]["mismatches_by_reason"] == {"record_key_mismatch": 1}
    assert "3_records_carry_the_logit_scale" in row["gate_G1"]["failing_conditions"]
    # The text-level table still builds on the same records, unchanged.
    assert w1.build_table(records)["denominators"]["n_items_complete"] == len(records)


# --------------------------------------------------------------------------- #
# The arithmetic of the logit column B
# --------------------------------------------------------------------------- #
def test_logit_column_b_arithmetic_on_a_design_solvable_by_hand(tmp_path):
    """Y = 1 + 3 X + 5 M + e on a design whose least squares solution is exact.

    The mediator is 0 or 1 on the clean arm and two nats higher on the hinted arm, so
    ``gamma = 2`` and ``mu_m = 0.5`` exactly, and the mediator residual is +/- 0.5 so
    ``sigma_m = 0.5`` exactly. The outcome residual ``e`` is +1 on the first two item types
    and -1 on the last two, which is orthogonal to (1, X, M) here, so the outcome
    regression returns alpha0 = 1, alpha = 3, beta = 5 and ``sigma_y = 1`` exactly. Every
    number below follows from those by A5.2's formulas and none of them is read off the
    code's own output.
    """
    cell, records, meta = build_cell(tmp_path)
    sidecar, _ = w1.load_logit_sidecar(cell)
    merged, _ = w1.merge_logit_sidecar(records, sidecar, cell)
    table = w1.build_table(merged, outcome="logprob_margin")

    assert table["outcome"]["scale"] == "logprob_margin"
    assert table["denominators"]["n_items_complete"] == len(records)
    # X and M are untouched by the substitution: the two tables differ in Y and nothing else.
    binary = w1.build_table(merged)
    assert np.array_equal(table["X"], binary["X"])
    assert np.array_equal(table["M"], binary["M"])

    block = c18.logit_column_b(table, n_bootstrap=40)

    fit = block["fit"]
    assert fit["gamma"] == pytest.approx(2.0)
    assert fit["mu_m"] == pytest.approx(0.5)
    assert fit["sigma_m"] == pytest.approx(0.5)
    assert fit["alpha0"] == pytest.approx(1.0)
    assert fit["alpha"] == pytest.approx(3.0)
    assert fit["beta"] == pytest.approx(5.0)
    assert fit["sigma_y"] == pytest.approx(1.0)

    effects = block["effects"]
    assert effects["nde"]["point"] == pytest.approx(3.0)  # alpha
    assert effects["nie"]["point"] == pytest.approx(10.0)  # beta * gamma
    assert effects["te"]["point"] == pytest.approx(13.0)  # alpha + beta * gamma
    assert effects["unit"] == "nats of renormalized letter margin"

    # The identity of A5.2, which A5.4 condition 5 gates on.
    gap = block["model_implied_te_vs_randomized_arm_difference"]
    assert gap["randomized_arm_difference"] == pytest.approx(13.0)
    assert abs(gap["difference"]) < 1e-9

    # Both arm variances: the number the outcome-scale note is about.
    assert block["outcome_variance"]["clean_arm"] == pytest.approx(7.25)
    assert block["outcome_variance"]["hinted_arm"] == pytest.approx(7.25)

    # rho*_point = |B| sigma_m / sqrt(S^2 + B^2 sigma_m^2) = 2.5 / sqrt(7.25).
    assert block["rho"]["rho_star_point"]["point"] == pytest.approx(2.5 / math.sqrt(7.25))

    # TE carries no rho on this scale, so the curve moves only in NDE and NIE.
    curve = {row["rho"]: row for row in block["rho"]["effects_curve"]}
    assert curve[0.0]["nie"] == pytest.approx(10.0)
    assert curve[0.0]["te"] == pytest.approx(13.0)
    assert block["rho"]["te_is_flat_in_rho"]["te_range_across_the_check_rhos"] < 1e-9
    assert block["rho"]["sweep_cross_check"]["max_abs_difference"] < 1e-9

    # The mediated share is NIE / TE and prints because the TE interval excludes zero.
    assert block["mediated_share"]["printed"] is True
    assert block["mediated_share"]["value"] == pytest.approx(10.0 / 13.0)

    # A5.6: no verdict on this scale, in the amendment's own words.
    assert block["verdict"] == "not applicable, no threshold pre-registered on this scale"


def test_the_bridge_counts_both_scales_on_the_same_items(tmp_path):
    """A5.3's agreement rate, per arm, over the items scored on both scales."""
    cell, records, _ = build_cell(tmp_path)
    sidecar, _ = w1.load_logit_sidecar(cell)
    merged, _ = w1.merge_logit_sidecar(records, sidecar, cell)
    table = w1.build_table(merged, outcome="logprob_margin")

    bridge = c18.logit_bridge(table)
    n = len(records)

    assert bridge["computable"] is True
    for arm in ("clean", "hinted"):
        assert bridge["per_arm"][arm]["n_scored_on_both_scales"] == n
        assert bridge["per_arm"][arm]["drops_by_reason"] == {"text_level_answer_unscorable": 0}

    # Every margin in this fixture is positive and the hinted answer is the hint label on
    # every item, so the hinted arm agrees on all of them; the clean answer is not the hint
    # label on any item, so the clean arm agrees on none. Both are exact counts here.
    assert bridge["per_arm"]["hinted"]["agreement_rate"] == pytest.approx(1.0)
    assert bridge["per_arm"]["hinted"]["follow_rate"] == pytest.approx(1.0)
    assert bridge["per_arm"]["clean"]["follow_rate"] == pytest.approx(0.0)
    assert bridge["per_arm"]["clean"]["share_with_margin_above_zero"] == pytest.approx(0.75)
    assert bridge["per_arm"]["clean"]["agreement_rate"] == pytest.approx(0.25)
