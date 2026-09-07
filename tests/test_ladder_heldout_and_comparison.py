"""Element 11(d) refuses without a freeze commit; element 11(e) computes four columns.

11(d) is a rule about WHEN a family may be generated, so the test that matters is the
refusal: without a freeze commit hash there is nothing to date and nothing to hash, and
a family generated anyway would be a development family with a held-out label.

11(e) is a rule about what every ladder claim is reported beside. Four of the six
columns come from a cell's own records and are computed here; the calibrated jury and
the simple probes do not and are reported as absent with the spec that would fill them,
never imputed.
"""

from __future__ import annotations

import json

import pytest

from bayes_cot_faithfulness.ladder import comparison_set as cs
from bayes_cot_faithfulness.ladder import heldout_family as hf


def test_the_held_out_family_refuses_without_a_freeze_commit(tmp_path):
    with pytest.raises(hf.HeldOutFamilyError) as exc:
        hf.generate(None, out_dir=tmp_path)
    assert "REFUSING" in str(exc.value)
    assert "frozen" in str(exc.value)
    assert not list(tmp_path.iterdir()), "a refused generation wrote files"

    # the CLI refuses too, with its own exit code rather than argparse's usage error
    assert hf.main(["--out", str(tmp_path)]) == 3
    assert not list(tmp_path.iterdir())


def test_a_freeze_commit_that_is_not_a_sha_is_refused(tmp_path):
    with pytest.raises(hf.HeldOutFamilyError) as exc:
        hf.generate("after the freeze, honest", out_dir=tmp_path)
    assert "not a git sha" in str(exc.value)


def test_a_freeze_commit_that_is_not_in_the_repo_is_refused(tmp_path):
    with pytest.raises(hf.HeldOutFamilyError) as exc:
        hf.generate("0" * 40, out_dir=tmp_path, repo=tmp_path)
    assert "not a commit" in str(exc.value)


def test_with_a_freeze_commit_it_generates_and_records_the_hash_and_the_date(tmp_path):
    manifest = hf.generate("abc1234", out_dir=tmp_path, date="2026-09-08", n_datasets=3)
    assert manifest["freeze_commit"] == "abc1234"
    assert manifest["generated_on"] == "2026-09-08"
    assert manifest["n_datasets"] == 3
    assert (tmp_path / "datasets.jsonl").is_file()
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["files"]["datasets.jsonl"]
    # the family's parameters come from the freeze hash, so it cannot be chosen later
    assert hf.draw_parameters("abc1234") == manifest["parameters"]
    # the planted X-caused path is in the truth: a nonzero NDE with a small NIE
    assert manifest["truth"]["nde"] > manifest["truth"]["nie"]


def test_a_different_freeze_commit_gives_a_different_family():
    """Otherwise the 'held out' family would be one fixed dataset wearing a hash."""
    seen = {tuple(sorted(hf.draw_parameters(c).items()))
            for c in ("aaaaaaa", "bbbbbbb", "ccccccc", "ddddddd", "eeeeeee")}
    assert len(seen) > 1


def _records(n=20, follow=8, with_logprobs=15, chars=400):
    out = []
    for i in range(n):
        r = {"followed": i < follow, "n_chars": chars + i}
        if i < with_logprobs:
            p = 0.7 if i % 2 else 0.4
            rest = (1.0 - p) / 3
            r["renormalized_over_letters"] = {"A": p, "B": rest, "C": rest, "D": rest}
        out.append(r)
    out[-1]["followed"] = None           # an unparseable answer
    return out


def test_the_four_computed_columns_carry_their_denominators():
    rows = _records()
    covariates = [{"commitment_depth": 0, "curve_area": 1.0}] * 6 + \
                 [{"commitment_depth": 3, "curve_area": 0.5}] * 4
    row = cs.comparison_row("organism_0.90_20260911", rows, covariates)

    cue = row["raw_cue_susceptibility"]
    assert cue["n_records"] == 20
    assert cue["n_unscorable"] == 1
    assert cue["n_scorable"] == 19
    assert cue["follow_rate"] == pytest.approx(8 / 19)

    ent = row["answer_entropy"]
    assert ent["n_with_letter_logprobs"] == 15
    assert 0.0 < ent["mean_normalized_entropy"] < 1.0

    assert row["trace_length"]["n_with_n_chars"] == 20
    early = row["early_answering"]
    assert early["n_precommitted_at_depth_0"] == 6
    assert early["precommitted_fraction"] == pytest.approx(0.6)


def test_the_two_columns_this_module_cannot_compute_are_absent_with_their_spec():
    row = cs.comparison_row("twin_0.90_20260911", _records())
    assert row["calibrated_jury_q1_rate"] is None
    assert row["simple_probe_score"] is None
    assert set(row["not_computed_here"]) == {"calibrated_jury", "simple_probes"}
    assert "equal access" in row["not_computed_here"]["simple_probes"]["needs"]

    filled = cs.comparison_row("twin_0.90_20260911", _records(),
                               calibrated_jury_q1_rate=0.31, simple_probe_score=0.62)
    assert filled["calibrated_jury_q1_rate"] == 0.31
    assert set(filled["not_computed_here"]) == set()


def test_a_column_with_no_source_field_is_none_and_says_why():
    row = cs.comparison_row("organism_0.60_20260911", [{"followed": True}], [])
    assert row["answer_entropy"]["mean_normalized_entropy"] is None
    assert "renormalized_over_letters" in row["answer_entropy"]["why_none"]
    assert row["trace_length"]["mean_chars"] is None
    assert row["early_answering"]["precommitted_fraction"] is None


def test_reading_a_cell_directory_reports_a_missing_records_file_as_missing(tmp_path):
    (tmp_path / "arms_summary.json").write_text(json.dumps({
        "arms": {"curves": {"hinted": {"covariates": [
            {"commitment_depth": 0, "curve_area": 1.0}]}}}}))
    row = cs.from_cell_dir(tmp_path, cell_id="organism_0.90_20260911")
    assert row["sources"]["records"] is None
    assert "does not exist" in row["sources"]["why_no_records"]
    assert row["early_answering"]["n_covariate_rows"] == 1
    assert row["raw_cue_susceptibility"]["follow_rate"] is None
