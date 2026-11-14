"""The 72-cell fits document, and the two banked documents it must not move.

`experiments/cells18_fits_report.py` was written for two substrates and three subjects.
The 72-cell lane has three substrates and six subjects, artifacts restored from an
archive after the cluster tree was deleted, a per-model sampler diagnostics file the
banked lanes never wrote, and none of the optional sensitivity or clean-parse records.
These tests pin what the document must say about each of those, and that the 18- and
24-cell renders stay byte-identical while it does.

The 72-cell artifacts are tracked (see .gitignore), so nothing here skips when they are
absent: a missing artifact is a failure, not a pass.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import cells18_fits as cf
import cells18_fits_report as rep

# sha256 of `--cells 18` and `--cells 24` rendered from commit efcc815, before the
# report learned a third substrate. The committed docs/CELLS18-FITS.md and
# docs/CELLS24-FITS.md are NOT byte-identical to a fresh render at that commit (the
# generator's prose moved after they were committed: a provenance paragraph, a
# precision table, several reworded sentences), so they cannot be the reference. These
# hashes are the render of the untouched generator, and a change that moves either
# banked render fails here. A deliberate change to the banked text must update the hash
# in the same commit, so the diff records that a banked document moved.
BASELINE_SHA256 = {
    "18": "5c31213ef69730eb939cd3fc37ce68a57a2f1025ac2af863c59b7ea454785fe7",
    "24": "6582ad5cafccb29c97eb4aea9cf97f1a7d2db9d01109ad59d73f2e1ad5fd65a9",
}


def _render(cell_set: str) -> str:
    lane = rep.LANES[cell_set]
    gate, fits, pymc, rows, extra, gemma, absent = rep.load(lane)
    diag = rep.load_sampler_diagnostics(lane)
    lines = rep.render_fits(lane, gate, fits, pymc, rows, extra, gemma, absent, diag)
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def doc72() -> str:
    return _render("72")


@pytest.fixture(scope="module")
def data72():
    lane = rep.LANES["72"]
    gate, fits, pymc, rows, extra, gemma, absent = rep.load(lane)
    return lane, gate, fits, pymc, rows, extra, gemma, absent, rep.load_sampler_diagnostics(lane)


# --------------------------------------------------------------------------- (e)
@pytest.mark.parametrize("cell_set", ["18", "24"])
def test_the_banked_renders_are_byte_identical_to_the_untouched_generator(cell_set):
    text = _render(cell_set)
    assert hashlib.sha256(text.encode()).hexdigest() == BASELINE_SHA256[cell_set]


def test_the_committed_72_document_is_what_the_generator_renders(doc72):
    """docs/CELLS72-FITS.md is generated, never edited by hand."""
    assert (REPO / "docs" / "CELLS72-FITS.md").read_text() == doc72


# --------------------------------------------------------------------------- (a)
def test_the_72_lane_renders_every_cell_and_six_subjects(doc72, data72):
    lane, _gate, fits, *_rest, absent, _diag = data72
    assert len(fits) == 72 and absent == []
    assert len(cf.models("72")) == 6
    assert (
        "All 72 cells this lane asked for are in this document: 6 models x 12 "
        "substrate-by-cue-family cells, 72 fit.json files read, 0 missing."
    ) in doc72
    assert "six models (" + ", ".join(lane.subjects) + ")" in doc72
    assert "## 7. What these 72 cells do not say" in doc72


def test_the_72_header_states_fit_date_render_context_and_restore(doc72, data72):
    _lane, gate, fits, *_ = data72
    head = doc72.split("## 1.")[0]
    assert "**Fits computed:** 2026-09-09" in head
    assert "**Rendered:** later, from the artifacts restored on 2026-11-14" in head
    assert "**Date:** 8 September 2026" not in head
    assert "bcf-fits72.tar.gz" in head
    assert "6dca1d98fd57a33d109e4ea0f44d759db82f08870e48105b5d65ecabd4c2a7b5" in head
    # the commit is read from the artifacts, so it is the one they carry
    commit = gate["code_commit"]
    assert all(f["code_commit"] == commit for f in fits.values())
    assert f"**Code commit at run time:** `{commit}`" in head
    assert f"`fit.json` `{commit}` in 72 of 72" in head


def test_the_72_branch_state_stays_true_after_the_merge(doc72):
    assert "merged into main together with this document" in doc72
    assert "pushed, not yet merged" not in doc72
    assert rep.LANES["18"].branch_state == "not merged and not pushed"
    assert rep.LANES["24"].branch_state == "not merged and not pushed"


def test_no_unqualified_cluster_access_claim_in_the_72_document(doc72):
    assert "a reader with cluster access can check" not in doc72
    assert "can no longer be checked against the files they name" in doc72


# --------------------------------------------------------------------------- (c)
def test_the_sampler_diagnostics_table_is_printed_for_the_72_lane(doc72, data72):
    *_x, diag = data72
    assert len(diag) == 6
    assert "**Divergences after tuning, from the sampler's own output.**" in doc72
    assert "can bias the tails of a hierarchical posterior" in doc72
    for m, d in diag.items():
        div = ", ".join(str(x) for x in d["divergences_after_tuning"])
        assert f"| `{m}` | {div} |" in doc72


def test_the_sampler_diagnostics_table_is_not_printed_for_the_24_lane():
    text = _render("24")
    assert rep.load_sampler_diagnostics(rep.LANES["24"]) == {}
    assert "Divergences after tuning, from the sampler's own output" not in text


# --------------------------------------------------------------------------- (d)
def test_the_missing_sensitivity_fits_are_stated_in_their_place(doc72):
    part = doc72.split("### 5.5 The sensitivity fits")[1].split("---")[0]
    assert "**Not produced for this pass.**" in part
    assert "model_row_logit_cue_family.json" in part
    assert "model_row_probit_substrate.json" in part
    assert "no longer exists" in part
    assert "are not carried over" in part


def test_the_missing_gemma_clean_parse_record_is_stated_in_its_place(doc72):
    part = doc72.split("## 6. The Gemma AQuA-RAT unparseable clean outputs")[1].split("---")[0]
    assert "**Not produced for this pass.**" in part
    assert "gemma_aqua_clean_parse.json" in part
    assert "no longer exists" in part
    # the classes of the 24-cell section are not printed as if they were 72-cell ones
    assert "answered none of the above" not in part
    assert "| aqua_rat x" not in part


# --------------------------------------------------------------- carried-over prose
@pytest.mark.parametrize(
    "banked_reading",
    [
        "several times the mediator variation",
        "far larger than the rho at which the verdict fails",
        "not scattered either side of it",
        "does NOT print for any cell",
        "The logit-level column B does not exist for any cell",
        "The two estimation paths agree",
        "ran under the R1 serving mode",
        "the binding constraint is the model-level leg rather than any cell's own evidence",
        "The blocks are in the order the models were run",
        # the flat G1 rows read keys the nested record does not carry and print None
        "probes None/None",
        "The verdict rule is section 2.5: load-bearing at rho when",
        "which is Monte Carlo noise on the integrator",
        "a cell that entered",
        "(not in this glossary)",
        "the same pairing the wave-1 cells were promoted on",
        "the largest P(NIE > 0.15) over",
        "largest gaps, each the largest",
        "The largest gap is",
        "Those rows are not withheld",
        "The rule above applies to them as written",
        "which is what a deterministic seeded gate should give",
    ],
)
def test_no_banked_interpretation_is_asserted_for_the_72_cells(doc72, banked_reading):
    assert banked_reading not in doc72


def test_the_mediator_is_reported_per_substrate_for_all_three(doc72):
    for name in ("ARC-Challenge", "AQuA-RAT", "LogiQA 2.0"):
        assert f"| {name} | 24 |" in doc72


def test_every_printed_logit_row_is_in_its_cell_block(doc72, data72):
    _lane, _gate, fits, *_ = data72
    printed = [k for k, f in fits.items() if (f.get("logit_level_row") or {}).get("printed")]
    assert len(printed) == 24
    assert doc72.count("**The logit-level row (Amendment A5), beside the text-level row") == 24
    assert "prints for 24 of 72 cells" in doc72


# --------------------------------------------------------------------------- 7
def test_the_eight_cells_below_the_clean_correct_floor_are_named(doc72, data72):
    _lane, _gate, fits, *_ = data72
    below = [
        (m, s, c, f["precision"]["n_clean_correct"])
        for (m, s, c), f in fits.items()
        if not f["precision"]["clean_correct_clears_floor"]
    ]
    assert len(below) == 8
    line = next(ln for ln in doc72.splitlines() if ln.startswith("12. "))
    assert "8 fall below the floor on the clean-correct count itself" in line
    for m, s, c, n in below:
        assert f"`{m}` {rep.slug(s, c)} at {n}" in line
        assert f"| `{m}` {rep.slug(s, c)} | {n} | NO |" in doc72


def test_the_precision_line_separates_missing_counts_from_measured_shortfalls():
    def block(n_clean, arms):
        return cf.precision_block(
            {"n_clean_correct": n_clean, "enabled_arms": list(arms),
             "arms": {k: ({} if v is None else {"n": v}) for k, v in arms.items()}}
        )

    fits = {
        ("a", "logiqa2", "metadata"): {"precision": block(300, {"direct": 300})},
        ("a", "logiqa2", "professor"): {"precision": block(1000, {"direct": 1000, "anchor": None})},
    }
    line = rep._precision_line(fits)
    assert "2 of the 2 cells are labelled UNDERPOWERED" in line
    assert "1 fall below the floor on the clean-correct count itself: `a` logiqa2 x metadata at 300" in line
    assert "labelled only because enabled arms recorded no count (anchor)" in line


# --------------------------------------------------------------------------- (b)
def _three_substrate_lane(data72):
    """One subject of the 72 lane across all three substrates, a smaller third-substrate
    lane than the real one, so each touched section function is exercised on its own."""
    lane, gate, fits, pymc, rows, _extra, _gemma, _absent, diag = data72
    m = "deepseek-r1-distill-llama-8b"
    small = lane._replace(subjects=(m,))
    f = {k: v for k, v in fits.items() if k[0] == m}
    p = {k: v for k, v in pymc.items() if k[0] == m}
    r = {m: rows[m]}
    d = {m: diag[m]}
    return small, gate, f, p, r, {}, None, [], d


def test_every_touched_section_function_runs_on_a_third_substrate(data72):
    lane, gate, fits, pymc, rows, extra, gemma, absent, diag = _three_substrate_lane(data72)
    assert {s for _m, s, _c in fits} == {"arc_challenge", "aqua_rat", "logiqa2"}
    assert not rep._two_substrate(lane)
    out = []
    out += rep.header(gate, fits, rows, lane, absent, diag)
    out += rep.gate_section(gate, lane, len(fits))
    out += rep.inventory_section(fits, lane)
    out += rep.cells_section(fits, pymc, True)
    out += rep.model_rows_section(rows, extra, fits, lane, diag)
    out += rep.gemma_section(gemma, lane, fits)
    out += rep.limits_section(fits, rows, lane)
    text = "\n".join(out)
    assert "| LogiQA 2.0 | 4 |" in text
    assert "of the 4 LogiQA 2.0 cells" in text
    assert "### 5.5 The sensitivity fits" in text


def test_the_two_substrate_mediator_text_refuses_a_third_substrate(data72):
    """The banked helper is kept verbatim and is only ever called for two substrates."""
    _lane, _gate, fits, *_ = data72
    with pytest.raises(KeyError):
        rep._mediator_two_substrates(fits)


def test_the_answer_only_counts_cover_every_substrate(data72):
    _lane, _gate, fits, *_ = data72
    text = rep._answer_only_prose(fits, general=True)
    for name in ("ARC-Challenge", "AQuA-RAT", "LogiQA 2.0"):
        assert f"of the 24 {name} cells" in text


def test_a_zero_arm_difference_is_left_out_of_the_ratio_rather_than_raising(data72):
    _lane, _gate, fits, *_ = data72
    zero = [
        k for k, f in fits.items()
        if f["column_b"]["model_implied_te_vs_randomized_arm_difference"]["randomized_arm_difference"] == 0
    ]
    assert zero, "the 72 lane has a cell whose arm difference is exactly zero"
    with pytest.raises(ZeroDivisionError):
        rep._max_relative(fits)
    text = rep._te_identity_line(fits, general=True)
    assert "an arm difference of exactly zero" in text


def test_a_gate_value_that_moved_is_named_not_explained_away(data72):
    _lane, gate, *_ = data72
    text = rep._gate_vs_previous(gate)
    old = rep.json.loads(rep._PREVIOUS_GATE.read_text())
    if old["wave1_fits_sha256"] == gate["wave1_fits_sha256"]:
        pytest.fail("the 72 gate is expected to differ in hash from the 18-cell record")
    assert "which is what a deterministic seeded gate should give" not in text
    assert "`fitted_mu_m` on the poisson null" in text
    moved = copy.deepcopy(gate)
    moved["nulls"]["poisson"]["fitted_mu_m"] = old["nulls"]["poisson"]["fitted_mu_m"]
    assert "which is what a deterministic seeded gate should give" in rep._gate_vs_previous(moved)


# --------------------------------------------------------------- degenerate inputs
# Each general-path helper, fed the input that would have made it print a confident
# or self-contradicting sentence. Each must raise or say plainly that the thing is not
# available.
def _fit_subset(data72, keys):
    _lane, _gate, fits, *_ = data72
    return {k: copy.deepcopy(fits[k]) for k in keys}


def test_answer_only_with_no_comparable_cell_says_so(data72):
    _lane, _gate, fits, *_ = data72
    f = {k: copy.deepcopy(v) for k, v in list(fits.items())[:3]}
    for v in f.values():
        v["anchor"]["falsifier_controls"]["matched_answer_only_text"]["a0"]["rate"] = None
    text = rep._answer_only_general(f)
    assert "cannot be compared in any cell" in text
    assert "RAISES" not in text and "raises the rate in" not in text


def test_answer_only_with_no_raise_does_not_contradict_itself(data72):
    _lane, _gate, fits, *_ = data72
    f = {k: copy.deepcopy(v) for k, v in list(fits.items())[:3]}
    for v in f.values():
        v["anchor"]["falsifier_controls"]["matched_answer_only_text"]["a0"]["rate"] = 0.0
        v["anchor"]["cells"]["mu01"]["rate"] = 0.5
    text = rep._answer_only_general(f)
    assert "raises the rate in none of the 3 cells" in text
    assert "does not point one way" not in text


def test_a_validated_cell_is_refused_not_printed_over(data72):
    lane, _gate, fits, *_ = data72
    f = {k: copy.deepcopy(v) for k, v in list(fits.items())[:2]}
    next(iter(f.values()))["claim_status"] = "VALIDATED"
    with pytest.raises(ValueError, match="VALIDATED"):
        rep._validated_limit_line(f, lane)


def test_missing_validated_evidence_is_said_not_assumed(data72):
    lane, _gate, fits, *_ = data72
    f = {k: copy.deepcopy(v) for k, v in list(fits.items())[:2]}
    for v in f.values():
        v["claim_status_evidence"] = ""
    text = rep._validated_limit_line(f, lane)
    assert "None of the 2 `fit.json` files records why it was not reached" in text
    assert "When these fits ran it had not run" not in text


def test_an_empty_but_present_gemma_record_is_not_called_absent(data72, tmp_path):
    lane, _gate, fits, *_ = data72
    (tmp_path / "gemma_aqua_clean_parse.json").write_text("{}")
    small = lane._replace(results=tmp_path, results_rel="tmp")
    text = "\n".join(rep._gemma_not_produced(small, fits))
    assert "**Present but empty.**" in text
    assert "does not exist" not in text


def test_a_lane_with_no_gemma_aqua_cell_keeps_section_6_and_says_why(data72):
    lane, _gate, fits, *_ = data72
    f = {k: v for k, v in fits.items() if k[0] != "gemma-2-9b-it"}
    text = "\n".join(rep._gemma_not_produced(lane, f))
    assert "## 6." in text and "Not applicable to this lane" in text


def test_claim_status_missing_on_one_side_is_not_counted_as_equal():
    pairs = [
        (("m", "s", "c"), {"claim_status": None}, {"claim_status": None}),
        (("m", "s", "d"), {"claim_status": "RAW"}, {"claim_status": "RAW"}),
    ]
    text = rep._status_phrase(pairs)
    assert "1 of the 1 with a status recorded on both sides" in text
    assert "1 have no status on at least one side" in text


def test_sampler_diagnostics_degenerate_rows_are_named():
    rows = {"a": {"sampler": {"divergences": 5}}, "b": {}, "c": {"sampler": {"divergences": 3}}}
    diag = {"a": {"divergences_after_tuning": [5]}, "b": {"divergences_after_tuning": [7]},
            "d": {"divergences_after_tuning": []}}
    text = "\n".join(rep._sampler_diagnostics_lines(rows, diag))
    assert "| `b` | 7 |" in text and "no sampler block" in text
    assert "| `d` | none parsed |" in text and "No count was parsed for `d`" in text
    assert "No `sampler_diagnostics.json` exists for `c`" in text
    assert "2 of the 3 models have a parsed count above zero" in text


def test_nested_g1_rows_print_not_recorded_rather_than_none(data72):
    _lane, _gate, fits, *_ = data72
    f = copy.deepcopy(fits[("qwen3-8b", "logiqa2", "professor")])
    cd = f["logit_level_gate_G1"]["conditions"]
    cd["2_pinned_self_hosted_endpoint"].pop("n_logprob_blocks_total")
    cd["4_clean_arm_outcome_variance_positive"] = {"holds": False}
    cd["5_te_logit_equals_arm_difference"] = {"holds": False, "status": "a stored reason"}
    text = "\n".join(rep._g1_rows_nested(cd))
    assert "None" not in text
    assert "not recorded" in text
    assert "a stored reason" in text


def test_a_wave1_difference_that_is_not_a_number_is_listed(data72):
    _lane, _gate, fits, *_ = data72
    f = {k: copy.deepcopy(v) for k, v in fits.items()}
    f[("qwen3-8b", "arc_challenge", "stated-hint")]["anchor"]["all_three_agree"] = "unknown"
    text = rep._wave1_reproduction(f, general=True)[0]
    assert "are not numbers on both sides" in text
    assert "'unknown' here" in text


def test_a_gate_field_with_a_missing_side_is_listed_not_skipped():
    old = {"p": {"fitted_mu_m": 1.0, "converged": True}}
    new = {"p": {"fitted_mu_m": None, "converged": True}}
    n_num, _n_same, num_diff, other = rep._compare_nulls(old, new)
    assert n_num == 0 and num_diff == []
    assert other == [("p", "fitted_mu_m", 1.0, None)]
    tail = rep._gate_moved_tail([("p", "fitted_mu_m", None, 1.0)], {"verdict": "PASS"},
                                {"verdict": "PASS"})
    assert "`fitted_mu_m` on the p null, None in that record and 1.0 here" in tail


def test_an_empty_depth_histogram_is_not_read_as_a_zero_share(data72):
    _lane, _gate, fits, *_ = data72
    f = {k: copy.deepcopy(v) for k, v in fits.items()}
    key = ("qwen3-8b", "logiqa2", "professor")
    f[key]["table"]["mediator"]["commitment_depth_hist_clean"] = {}
    text = "\n".join(rep._mediator_by_substrate(f))
    assert "1 cells record no commitment-depth histogram" in text
    assert "`qwen3-8b` logiqa2 x professor" in text


def test_a_model_with_no_row_or_diagnostics_is_named_in_the_commit_line(data72):
    _lane, gate, fits, _p, rows, _e, _g, _a, diag = data72
    r = {k: v for k, v in rows.items() if k != "phi-4-reasoning"}
    d = {k: v for k, v in diag.items() if k != "qwen3-8b"}
    text = rep._commit_lines(gate, fits, r, d, tuple(rows))[0]
    assert "no `model_row.json` for `phi-4-reasoning`" in text
    assert "no `sampler_diagnostics.json` for `qwen3-8b`" in text


def test_a_lane_with_no_artifacts_is_refused_and_writes_nothing(tmp_path, monkeypatch):
    """The 36 lane names a results root that was never written; it must not render."""
    out = tmp_path / "x.md"
    monkeypatch.setattr(sys, "argv", ["prog", "--cells", "36", "--out", str(out)])
    assert not (rep.LANES["36"].results / "offset_null_gate.json").exists()
    assert rep.main() == 2
    assert not out.exists()


# ------------------------------------------------------------ independent recomputation
# These read the JSON files and the committed document directly. They import nothing
# from the generator, so a rendering error cannot hide behind the code that made it.
import glob
import json

RES = REPO / "experiments" / "results" / "cells72-fits"
DOC = REPO / "docs" / "CELLS72-FITS.md"


@pytest.fixture(scope="module")
def raw72():
    fits = {}
    for path in glob.glob(str(RES / "*" / "*" / "*" / "fit.json")):
        m, s, c = Path(path).parts[-4:-1]
        fits[(m, s, c)] = json.loads(Path(path).read_text())
    return fits, DOC.read_text()


def test_recomputed_claim_status_mix(raw72):
    fits, doc = raw72
    assert len(fits) == 72
    anch = sum(1 for f in fits.values() if f["claim_status"] == "ANCHORED")
    raw = sum(1 for f in fits.values() if f["claim_status"] == "RAW")
    assert (anch, raw) == (0, 72)
    assert f"Claim-status mix over the 72 cells: **{anch} ANCHORED**, **{raw} RAW**." in doc


def test_recomputed_divergence_counts(raw72):
    _fits, doc = raw72
    seen = {}
    for path in glob.glob(str(RES / "*" / "sampler_diagnostics.json")):
        d = json.loads(Path(path).read_text())
        row = json.loads((Path(path).parent / "model_row.json").read_text())
        (div,) = d["divergences_after_tuning"]
        assert div == row["sampler"]["divergences"]
        seen[d["model"]] = div
        assert f"| `{d['model']}` | {div} |" in doc
    assert sorted(seen.values()) == [57, 80, 92, 110, 124, 231]


def test_recomputed_cell_level_agreements(raw72):
    fits, doc = raw72
    k = sum(1 for f in fits.values() if f["anchor"]["all_three_agree"])
    assert k == 20
    assert f"{k} of 72 cells clear the element 21 margin on all three of their own estimands" in doc


def test_recomputed_g1_passes_and_failures(raw72):
    fits, doc = raw72
    ok = sum(1 for f in fits.values() if f["logit_level_gate_G1"]["eligible"])
    assert (ok, 72 - ok) == (24, 48)
    assert f"prints for {ok} of 72 cells" in doc
    assert f"The A5.4 eligibility gate G1 fails in the other {72 - ok}" in doc


def test_recomputed_precision_split_and_load_bearing_verdicts(raw72):
    fits, doc = raw72
    pr = {k: f["precision"] for k, f in fits.items()}
    below = [k for k, p in pr.items() if not p["clean_correct_clears_floor"]]
    arm = [k for k, p in pr.items() if p["clean_correct_clears_floor"] and p["arms_below_floor"]]
    only = [k for k, p in pr.items() if p["clean_correct_clears_floor"]
            and not p["arms_below_floor"] and p["enabled_arms_with_no_denominator"]]
    assert (len(below), len(arm), len(only)) == (8, 8, 56)
    lb = [k for k, f in fits.items() if f["column_b"]["verdict"]["verdict"] == "load-bearing at rho=0"]
    in_only = sum(1 for k in lb if k in only)
    assert (len(lb), in_only) == (14, 14)
    assert f"{len(below)} fall below the floor on the clean-correct count itself" in doc
    assert f"{len(arm)} more clear it on clean-correct" in doc
    assert f"**No recorded count only: {len(only)} cells.**" in doc
    assert (f"{len(lb)} labelled cells carry the verdict load-bearing at rho=0: "
            f"{len(lb) - in_only} in the first group and {in_only} in the second") in doc
    rule = next(iter(pr.values()))["rule_source"]
    assert f'in these words: "{rule}"' in doc


def test_recomputed_thin_column_b_cells(raw72):
    fits, doc = raw72
    thin = sorted(
        (f["table"]["denominators"]["n_items_complete"], k) for k, f in fits.items()
        if f["table"]["denominators"]["n_items_complete"] < 50
    )
    assert [n for n, _k in thin] == [11, 13, 22, 22]
    assert all(k[0] == "deepseek-r1-distill-llama-8b" and k[1] == "aqua_rat" for _n, k in thin)
    scope = next(ln for ln in doc.splitlines() if ln.startswith("**Scope, in one paragraph.**"))
    item8 = next(ln for ln in doc.splitlines() if ln.startswith("8. "))
    for n, (m, s, c) in thin:
        cc = fits[(m, s, c)]["column_a"]["denominators"]["n_clean_correct"]
        phrase = f"`{m}` {s} x {c}, {n} complete items against {cc} clean-correct"
        assert phrase in scope and phrase in item8
    assert "complete items" in item8 and "entered" not in item8.split("(the counts")[0]


def test_recomputed_gate_field_comparison(raw72):
    _fits, doc = raw72
    old = json.loads((REPO / "experiments/results/cells18-fits/offset_null_gate.json").read_text())
    new = json.loads((RES / "offset_null_gate.json").read_text())
    num = [
        (kind, key, a, new["nulls"][kind][key])
        for kind, fields in old["nulls"].items()
        for key, a in fields.items()
        if isinstance(a, (int, float)) and not isinstance(a, bool)
    ]
    diff = [(kind, key, abs(a - b)) for kind, key, a, b in num if a != b]
    assert (len(num), len(diff)) == (30, 7)
    worst = max(diff, key=lambda t: t[2])
    assert worst[:2] == ("poisson", "fitted_sigma_m")
    assert f"({len(num)} fields," in doc
    assert f"{len(num) - len(diff)} are equal and {len(diff)} differ" in doc
    assert f"The largest difference is {worst[2]:.1e}, on `{worst[1]}` of the {worst[0]} null." in doc


def test_recomputed_model_row_effects(raw72):
    _fits, doc = raw72
    for path in glob.glob(str(RES / "*" / "model_row.json")):
        ef = json.loads(Path(path).read_text())["effects"]
        for k in ("nde", "nie", "te"):
            b = ef[k]
            assert f"| {k.upper()} | {b['point']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] |" in doc


def test_recomputed_nie_intervals_that_exclude_zero(raw72):
    fits, doc = raw72
    ex = sum(1 for f in fits.values()
             if f["column_b"]["effects"]["nie"]["lo"] > 0 or f["column_b"]["effects"]["nie"]["hi"] < 0)
    unres = [f for f in fits.values() if f["column_b"]["verdict"]["verdict"] == "unresolved"]
    ex_u = sum(1 for f in unres
               if f["column_b"]["effects"]["nie"]["lo"] > 0 or f["column_b"]["effects"]["nie"]["hi"] < 0)
    assert f"The NIE interval excludes zero in {ex_u} of the {len(unres)} unresolved cells, and in {ex} of all 72." in doc
    rows = [json.loads(Path(p).read_text())["effects"]["nie"] for p in glob.glob(str(RES / "*" / "model_row.json"))]
    ex_r = sum(1 for b in rows if b["lo"] > 0 or b["hi"] < 0)
    assert f"the NIE interval excludes zero in {ex_r} of the 6 rows" in doc
