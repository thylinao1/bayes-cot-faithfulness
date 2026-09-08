"""The exploratory own-family panel: what --allow-own-family changes, and what it must not.

The gate corpus's subject is Qwen3-8B, so section 6.2 keeps the Qwen judge off its panel and
`panel_gate.py` refuses a Qwen vote directory. Section 6.2 also puts that judge on the panel
of 14 of the other 18 subjects, and the gate runs recorded its vote on every corpus row, so
those compositions can be estimated here. `--allow-own-family` is the door, and these tests
pin both sides of it: the refusal survives untouched when the flag is off, and when it is on
the report says in its kind, its note and its expected_panel exactly what it is.

Every fixture is hand-written vote rows, so each expected number is arithmetic a reader can
do by hand. Each test carries the revert proof, that is, what breaks if the production
change is taken out again.
"""

from __future__ import annotations

import json

import pytest

from experiments.jury import gate_matrix
from experiments.jury import panel_gate as pg
from tests.test_panel_gate import _items, _row, _write

QWEN = "qwen3-32b"
ALL_FOUR = ("gemma-3-27b-it", "gpt-oss-20b", "llama-3.3-70b-fp8", QWEN)


def _sources(tmp_path, votes_by_judge, q1_variant="a"):
    """Load every judge with the own-family door open."""
    return {
        key: pg.load_judge_votes(key, _write(tmp_path, key, rows), q1_variant,
                                 allow_own_family=True)
        for key, rows in votes_by_judge.items()
    }


def _one_vote_each(item_id, votes_by_judge, question="Q1"):
    return {k: [_row(k, item_id, question, v)] for k, v in votes_by_judge.items()}


# --- (a) the refusal is exactly what it was when the flag is off -------------------


def test_without_the_flag_a_qwen_directory_is_still_refused(tmp_path):
    """REVERT PROOF (panel_gate.py): the bare call passes before and after the change, which
    is the guarantee. The second call, which spells the flag out as False, raises TypeError
    on the reverted file because load_judge_votes has no such keyword there."""
    d = _write(tmp_path, QWEN, [_row(QWEN, "i1", "Q1", "yes")])
    with pytest.raises(pg.PanelGateError, match="REFUSING qwen3-32b"):
        pg.load_judge_votes(QWEN, d, "a")
    with pytest.raises(pg.PanelGateError, match="REFUSING qwen3-32b"):
        pg.load_judge_votes(QWEN, d, "a", allow_own_family=False)


def test_without_the_flag_the_command_line_refuses_a_qwen_directory(tmp_path):
    """The refusal has to hold at the CLI, which is where a vote directory is really named.

    REVERT PROOF (panel_gate.py): the reverted parser has no --allow-own-family, so the last
    line exits with SystemExit 2 from argparse instead of returning 0."""
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({"item_id": "i1", "meta": {"gate_class": "clean"}}) + "\n")
    d = _write(tmp_path, QWEN, [_row(QWEN, "i1", "Q1", "yes")])
    argv = ["--q1", "a", "--items", str(items), "--votes", f"{QWEN}={d}"]
    with pytest.raises(pg.PanelGateError, match="REFUSING qwen3-32b"):
        pg.main(argv)
    assert pg.main(argv + ["--allow-own-family"]) == 0


def test_the_flag_admits_the_own_family_judge_and_nothing_else(tmp_path):
    """A short panel of non-own-family judges must not reach PANEL by way of this flag.

    REVERT PROOF (panel_gate.py): the reverted loader has no allow_own_family keyword, so
    building the fixture raises TypeError instead of reaching PanelGateError."""
    votes = _one_vote_each("i1", {"gemma-3-27b-it": "no", "llama-3.3-70b-fp8": "no"})
    sources = _sources(tmp_path, votes)
    with pytest.raises(pg.PanelGateError, match="no own-family judge"):
        pg.build_panel_report(sources, _items([("i1", "clean")]), "a", allow_own_family=True)
    # and the report built the ordinary way is still the smaller panel it is
    plain = pg.build_panel_report(sources, _items([("i1", "clean")]), "a")
    assert plain["kind"] == "PANEL-PARTIAL"


def test_own_family_votes_cannot_reach_a_report_without_the_flag(tmp_path):
    """The second lock: even a hand-built sources dict is refused at report time.

    REVERT PROOF (panel_gate.py): the reverted loader raises TypeError on allow_own_family
    before the fixture exists; the check this pins is the second lock behind it."""
    sources = _sources(tmp_path, _one_vote_each("i1", {k: "no" for k in ALL_FOUR}))
    with pytest.raises(pg.PanelGateError, match="without allow_own_family"):
        pg.build_panel_report(sources, _items([("i1", "clean")]), "a")


# --- (b) the four-judge path ------------------------------------------------------


def test_four_judges_label_a_row_by_majority_of_the_available_votes(tmp_path):
    """REVERT PROOF (panel_gate.py): the reverted loader has no allow_own_family keyword, so
    the fixture cannot be built and the test errors with TypeError in _sources."""
    votes = _one_vote_each("i1", {"gemma-3-27b-it": "yes", "gpt-oss-20b": "yes",
                                  "llama-3.3-70b-fp8": "no", QWEN: "yes"})
    labels = pg.panel_labels(_sources(tmp_path, votes), 0)
    assert labels["i1"]["q1"] == "yes"
    assert labels["i1"]["q1_panel_size"] == 4
    assert labels["i1"]["q1_is_tie"] is False


def test_a_two_two_split_takes_the_gate_outcome_and_is_counted_as_a_tie(tmp_path):
    """The 2-2 case of section 6.2, which the three-judge panel only ever meets as 1-1
    after a leave-one-out. The row's label becomes the gate panel's own outcome token, so it
    leaves the Q1 numerator AND its denominator, and it is counted in the tie tally.

    REVERT PROOF (panel_gate.py): the fixture needs a Qwen source, and the reverted loader
    has no allow_own_family keyword, so _sources raises TypeError."""
    votes = {}
    for key, vote in (("gemma-3-27b-it", "yes"), ("gpt-oss-20b", "yes"),
                      ("llama-3.3-70b-fp8", "no"), (QWEN, "no")):
        votes[key] = [_row(key, "i1", "Q1", vote),
                      _row(key, "i1", "gate", "silent_override")]
    sources = _sources(tmp_path, votes)
    labels = pg.panel_labels(sources, 0)
    assert labels["i1"]["q1_panel_size"] == 4
    assert labels["i1"]["q1_is_tie"] is True
    assert labels["i1"]["q1_resolution"] == "tie_to_gate"
    assert labels["i1"]["q1"] == "silent_override", "a 2-2 tie is not coin-flipped"

    report = pg.score_panel(sources, {i["item_id"]: i for i in _items([("i1", "clean")])})
    assert report["metrics"]["panel_q1_ties"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert report["per_class_counts"]["clean"]["q1"]["tie"] == 1
    assert report["per_class_counts"]["clean"]["q1"]["total"] == 1
    assert report["per_class_counts"]["clean"]["q1"]["yes"] == 0
    assert report["per_class_counts"]["clean"]["q1"]["no"] == 0
    assert report["gate_verdicts"]["specificity_clean"]["verdict"] == "NO DATA"


def test_the_tie_leaves_the_q1_denominator_over_ten_rows(tmp_path):
    """The same trap at scale, the four-judge twin of the three-judge tie test: two judges
    are wrong on four of ten clean rows, the panel splits 2-2 there, and specificity reads
    6/6 rather than 6/10 because the tied rows left the denominator.

    REVERT PROOF (panel_gate.py): the fixture needs a Qwen source, and the reverted loader
    has no allow_own_family keyword, so _sources raises TypeError."""
    ids = [f"i{n}" for n in range(10)]
    yes_on = set(ids[:4])
    votes = {}
    for key in ("gemma-3-27b-it", "llama-3.3-70b-fp8"):
        votes[key] = [_row(key, i, "Q1", "yes" if i in yes_on else "no") for i in ids]
    for key in ("gpt-oss-20b", QWEN):
        votes[key] = [_row(key, i, "Q1", "no") for i in ids]
    for key in ALL_FOUR:
        votes[key] += [_row(key, i, "gate", "coherent") for i in ids]
    sources = _sources(tmp_path, votes)
    report = pg.score_panel(sources, {i["item_id"]: i for i in _items([(i, "clean") for i in ids])})
    v = report["gate_verdicts"]["specificity_clean"]
    assert (v["numerator"], v["denominator"], v["verdict"]) == (6, 6, "PASS")
    assert report["metrics"]["panel_q1_ties"]["numerator"] == 4
    assert report["per_class_counts"]["clean"]["q1"]["tie"] == 4
    assert report["per_class_counts"]["clean"]["q1"]["total"] == 10


def test_the_malformed_denominator_covers_every_admitted_judge(tmp_path):
    """Pooled over the judges actually admitted, so the own-family judge's votes are in the
    denominator as well as the numerator: 4 judges x 2 rows = 8 votes, 1 malformed.

    REVERT PROOF (panel_gate.py): the fixture needs a Qwen source, and the reverted loader
    has no allow_own_family keyword, so _sources raises TypeError."""
    votes = {k: [_row(k, "i1", "Q1", "no"), _row(k, "i2", "Q1", "no")] for k in ALL_FOUR}
    votes[QWEN][0] = _row(QWEN, "i1", "Q1", "malformed")
    report = pg.score_panel(_sources(tmp_path, votes),
                            {i["item_id"]: i for i in _items([("i1", "clean"), ("i2", "clean")])})
    assert report["metrics"]["malformed_rate_max"] == {"numerator": 1, "denominator": 8,
                                                       "rate": 1 / 8}
    assert report["metrics"]["abstain_rate"]["denominator"] == 8
    assert report["gate_verdicts"]["specificity_clean"]["numerator"] == 2


def test_test_retest_recomputes_the_four_judge_label_per_run(tmp_path):
    """The own-family judge's vote enters every run's label, not just run 0: on `flips` it
    turns a 2-2 tie in run 0 into a 3-1 yes in runs 1 and 2, so the item disagrees.

    REVERT PROOF (panel_gate.py): the fixture needs a Qwen source, and the reverted loader
    has no allow_own_family keyword, so _sources raises TypeError."""
    votes = {k: [] for k in ALL_FOUR}
    for run in (0, 1, 2):
        for key in ALL_FOUR:
            votes[key].append(_row(key, "steady", "Q1", "no", run_idx=run))
            votes[key].append(_row(key, "flips", "gate", "coherent", run_idx=run))
        votes["gemma-3-27b-it"].append(_row("gemma-3-27b-it", "flips", "Q1", "yes", run_idx=run))
        votes["gpt-oss-20b"].append(_row("gpt-oss-20b", "flips", "Q1", "yes", run_idx=run))
        votes["llama-3.3-70b-fp8"].append(_row("llama-3.3-70b-fp8", "flips", "Q1", "no", run_idx=run))
        votes[QWEN].append(_row(QWEN, "flips", "Q1", "no" if run == 0 else "yes", run_idx=run))
    sources = _sources(tmp_path, votes)
    per_run = {i: pg.panel_labels(sources, i)["flips"]["q1"] for i in (0, 1, 2)}
    assert per_run == {0: "coherent", 1: "yes", 2: "yes"}
    report = pg.score_panel(sources, {i["item_id"]: i
                                      for i in _items([("steady", "clean"), ("flips", "clean")])})
    assert report["metrics"]["test_retest_q1_min"] == {"numerator": 1, "denominator": 2,
                                                       "rate": 0.5}


# --- (c) the kind, the note and the composition the report is complete against -----


def _four_judge_report(tmp_path, requested=ALL_FOUR):
    votes = _one_vote_each("i1", {k: "no" for k in requested})
    return pg.build_panel_report(_sources(tmp_path, votes), _items([("i1", "clean")]), "a",
                                 allow_own_family=True, requested=list(requested))


def test_the_report_is_stamped_exploratory_and_carries_the_note(tmp_path):
    """REVERT PROOF (panel_gate.py): on the reverted file the fixture cannot be loaded and
    the report has no kind, note or own_family_judges_admitted; it raises TypeError."""
    report = _four_judge_report(tmp_path)
    assert report["kind"] == "PANEL-OWNFAMILY-EXPLORATORY"
    assert report["own_family_judges_admitted"] == [QWEN]
    note = report["note"]
    assert "Qwen3-8B" in note, "the note must name the corpus's subject"
    assert "6.2" in note and QWEN in note, "and say which judge 6.2 excludes on that subject"
    assert "estimate" in note.lower(), "and that the number is an estimate"
    for composition in ("{all four}", "{minus Llama}", "{minus Gemma}", "{minus gpt-oss}"):
        assert composition in note, "the compositions that DO include it are named"
    assert "selects a configuration" in note


def test_expected_panel_is_the_composition_that_was_requested(tmp_path):
    """A minus-Gemma panel is a composition section 6.2 gives to two subjects, not a partial
    version of this subject's three judges, and the report says so.

    REVERT PROOF (panel_gate.py): the reverted file raises TypeError in _sources, and even
    with a loadable fixture it computes expected_panel from the 3-judge default, so this
    request would have read PANEL-PARTIAL with gemma-3-27b-it missing."""
    requested = ["gpt-oss-20b", "llama-3.3-70b-fp8", QWEN]
    report = _four_judge_report(tmp_path, requested)
    assert report["expected_panel"] == requested
    assert report["missing_judges"] == []
    assert report["panel_complete"] is True
    assert report["kind"] == "PANEL-OWNFAMILY-EXPLORATORY"
    assert report["panel"]["judges"] == sorted(requested)
    assert report["excluded_own_family"] == [QWEN], "6.2's exclusion is still on the record"


def test_the_leave_one_out_blocks_keep_their_structure(tmp_path):
    """Four judges give four leave-one-out blocks of three, each scored the same way.

    REVERT PROOF (panel_gate.py): the reverted file raises TypeError in _sources, so no
    four-judge report and no four leave-one-out blocks exist to check."""
    report = _four_judge_report(tmp_path)
    loo = report["leave_one_judge_out"]
    assert sorted(loo) == sorted(ALL_FOUR)
    for dropped, block in loo.items():
        assert block["judges"] == sorted(set(ALL_FOUR) - {dropped})
        assert block["panel_size_nominal"] == 3
        assert set(block) >= {"metrics", "gate_verdicts", "passed_of_ten", "verdict",
                              "failed_metrics", "per_class_counts", "sources"}


# --- (d) the matrix renders the new kind ------------------------------------------


def test_gate_matrix_renders_the_exploratory_kind_and_names_the_composition(tmp_path):
    """REVERT PROOF (gate_matrix.py): with only gate_matrix.py reverted the rows come back
    ['PANEL', 'PANEL-LOO' x4] and the Judge column carries no composition name, so the first
    assertion fails on that list. With panel_gate.py reverted too it fails in _sources."""
    report = _four_judge_report(tmp_path)
    rows = gate_matrix._panel_rows(report)
    assert [r["kind"] for r in rows] == ["PANEL-OWNFAMILY-EXPL"] + ["PANEL-OWNFAMILY-EXPL-LOO"] * 4
    assert rows[0]["judge"] == "PANEL {all four} " + "+".join(ALL_FOUR)
    by_judge = {r["judge"] for r in rows[1:]}
    assert "PANEL {minus Qwen} gemma-3-27b-it+gpt-oss-20b+llama-3.3-70b-fp8" in by_judge
    assert "PANEL {minus Llama} gemma-3-27b-it+gpt-oss-20b+qwen3-32b" in by_judge
    assert "PANEL {minus Gemma} gpt-oss-20b+llama-3.3-70b-fp8+qwen3-32b" in by_judge
    assert "PANEL {minus gpt-oss} gemma-3-27b-it+llama-3.3-70b-fp8+qwen3-32b" in by_judge
    head_and_body = gate_matrix.threshold_matrix(rows)
    assert any(line.startswith("| PANEL-OWNFAMILY-EXPL |") for line in head_and_body)
    assert any(line.startswith("| PANEL-OWNFAMILY-EXPL-LOO |") for line in head_and_body)
    assert any("{all four}" in line for line in gate_matrix.class_matrix(rows))


def test_the_exploratory_row_keeps_the_budget_tag_from_the_run_slug(tmp_path):
    """The Qwen d run of record is on an h200 at 1,024 tokens, and a row that hides that
    budget is the defect the tag exists to stop.

    REVERT PROOF (gate_matrix.py): with only gate_matrix.py reverted the row kind is PANEL
    and the first assertion fails before the tag is read; the tag itself still works, which
    is what "keep the budget tag logic" means."""
    report = _four_judge_report(tmp_path)
    slug = "experiments/results/jury-gate/qwen3-32b-h200-np1024-q1d/arc_challenge/stated-hint"
    report["panel"]["sources"][QWEN]["dir"] = slug
    for block in report["leave_one_judge_out"].values():
        if QWEN in block["sources"]:
            block["sources"][QWEN]["dir"] = slug
    rows = gate_matrix._panel_rows(report)
    assert rows[0]["kind"] == "PANEL-OWNFAMILY-EXPL"
    segments = dict(seg.split(":", 1) for seg in rows[0]["serving_line"].split(" + "))
    assert segments[QWEN].endswith("@np1024")
    assert "@np" not in segments["gemma-3-27b-it"]
    for row in rows[1:]:
        if f"{QWEN}:" in row["serving_line"]:
            loo = dict(seg.split(":", 1) for seg in row["serving_line"].split(" + "))
            assert loo[QWEN].endswith("@np1024")


def test_a_report_without_the_flag_still_renders_as_a_plain_panel(tmp_path):
    """The other side of the door: nothing about an ordinary panel report moved.

    REVERT PROOF: this one PASSES on both reverted files, which is the point of it. It is
    the byte-for-byte guard on the path the change must not touch."""
    votes = _one_vote_each("i1", {k: "no" for k in ALL_FOUR if k != QWEN})
    sources = {k: pg.load_judge_votes(k, _write(tmp_path, k, v), "a") for k, v in votes.items()}
    report = pg.build_panel_report(sources, _items([("i1", "clean")]), "a")
    assert report["kind"] == "PANEL"
    rows = gate_matrix._panel_rows(report)
    assert [r["kind"] for r in rows] == ["PANEL"] + ["PANEL-LOO"] * 3
    assert rows[0]["judge"] == "PANEL gemma-3-27b-it+gpt-oss-20b+llama-3.3-70b-fp8"
    assert rows[1]["judge"].startswith("PANEL minus ")
