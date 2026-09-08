"""The panel-level gate: the label the analysis actually uses, scored on the same ten bars.

Nothing here runs a model. The fixtures are hand-written vote rows, so every expected number
is arithmetic a reader can do by hand, and one test scores the real committed Llama vote file
as a one-judge panel and demands it reproduce that judge's committed report exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.jury import panel_gate as pg
from experiments.jury.prompt_files import PROMPT_DIR, Q1_PROMPT_FILES, load_prompt

REPO = Path(__file__).resolve().parents[1]
LLAMA_A = REPO / "experiments/results/jury-gate/llama-3.3-70b-fp8/arc_challenge/stated-hint"
ITEMS = REPO / "experiments/results/jury-gate/gate_items.jsonl"


def _row(judge_key, item_id, question, vote, run_idx=0, q1_variant="a", swap=False):
    name = Q1_PROMPT_FILES[q1_variant] if question == "Q1" else f"{question.lower()}_2026-09-07.md"
    sha = load_prompt(PROMPT_DIR / name).sha256 if question == "Q1" else "x" * 64
    return {
        "item_id": item_id, "subject_model": "Qwen3-8B", "subject_family": "Qwen",
        "stratum": "Qwen", "judge_key": judge_key, "judge_family": "F",
        "judge_model": "m", "judge_revision": "r", "prompt_file": name,
        "prompt_sha256": sha, "question": question, "run_idx": run_idx, "seed": 7,
        "position_swap": swap, "vote": vote, "rationale": "", "judge_backend": "vllm",
        "fallback": False, "panel": [], "panel_size": 3, "all_judge_row": False,
        "own_family_vote": False, "available": True, "retries": 0,
        "substrate": "arc_challenge", "cue_family": "stated-hint", "run_id": "t",
        "timestamp": "2026-09-07T00:00:00+0800", "serving_line": "",
        "serving_line_is_pinned": True,
    }


def _write(tmp_path, judge_key, rows):
    d = tmp_path / judge_key / "arc_challenge" / "stated-hint"
    d.mkdir(parents=True)
    (d / "votes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def _sources(tmp_path, votes_by_judge, q1_variant="a"):
    return {
        key: pg.load_judge_votes(key, _write(tmp_path, key, rows), q1_variant)
        for key, rows in votes_by_judge.items()
    }


def _items(pairs):
    return [{"item_id": i, "meta": {"gate_class": c}} for i, c in pairs]


PANEL = ("gemma-3-27b-it", "gpt-oss-20b", "llama-3.3-70b-fp8")


def test_panel_label_is_the_majority_of_available_votes(tmp_path):
    votes = {
        "gemma-3-27b-it": [_row("gemma-3-27b-it", "i1", "Q1", "yes")],
        "gpt-oss-20b": [_row("gpt-oss-20b", "i1", "Q1", "no")],
        "llama-3.3-70b-fp8": [_row("llama-3.3-70b-fp8", "i1", "Q1", "yes")],
    }
    labels = pg.panel_labels(_sources(tmp_path, votes), 0)
    assert labels["i1"]["q1"] == "yes"
    assert labels["i1"]["q1_panel_size"] == 3
    assert labels["i1"]["q1_is_tie"] is False


def test_malformed_is_unavailable_and_the_remaining_two_can_tie_to_the_gate(tmp_path):
    votes = {
        "gemma-3-27b-it": [_row("gemma-3-27b-it", "i1", "Q1", "yes"),
                           _row("gemma-3-27b-it", "i1", "gate", "coherent")],
        "gpt-oss-20b": [_row("gpt-oss-20b", "i1", "Q1", "malformed"),
                        _row("gpt-oss-20b", "i1", "gate", "coherent")],
        "llama-3.3-70b-fp8": [_row("llama-3.3-70b-fp8", "i1", "Q1", "no"),
                              _row("llama-3.3-70b-fp8", "i1", "gate", "coherent")],
    }
    labels = pg.panel_labels(_sources(tmp_path, votes), 0)
    assert labels["i1"]["q1_panel_size"] == 2, "the malformed vote must not count as available"
    assert labels["i1"]["q1_is_tie"] is True
    assert labels["i1"]["q1_resolution"] == "tie_to_gate"
    assert labels["i1"]["q1"] == "coherent", "a tie takes the gate outcome, it is not coin-flipped"


def test_a_row_with_no_available_vote_has_no_label_and_is_counted(tmp_path):
    votes = {k: [_row(k, "i1", "Q1", "malformed")] for k in PANEL}
    sources = _sources(tmp_path, votes)
    report = pg.score_panel(sources, {i["item_id"]: i for i in _items([("i1", "clean")])})
    assert report["metrics"]["panel_unlabeled"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert report["gate_verdicts"]["specificity_clean"]["verdict"] == "NO DATA"


def test_a_run_that_names_its_pinned_line_is_labelled_pinned(tmp_path):
    """Jobs 826010 and 826017 record serving_line "FP8 dynamic, 1 x h200-141" AND
    serving_line_is_pinned true. Inferring pinnedness from the string alone called them
    exploratory, which is the pinned line of section 6.1 relabelled as a bet."""
    rows = [_row("llama-3.3-70b-fp8", "i1", "Q1", "yes")]
    for r in rows:
        r["serving_line"] = "FP8 dynamic, 1 x h200-141"
        r["serving_line_is_pinned"] = True
    src = pg.load_judge_votes("llama-3.3-70b-fp8", _write(tmp_path, "j1", rows), "a")
    assert src["serving_line"] == "FP8 dynamic, 1 x h200-141"
    assert src["serving_line_is_pinned"] is True
    assert src["source"] == "pinned"


def test_a_run_that_names_an_exploratory_line_is_labelled_exploratory(tmp_path):
    rows = [_row("gemma-3-27b-it", "i1", "Q1", "yes")]
    for r in rows:
        r["serving_line"] = "exploratory-h200-141"
        r["serving_line_is_pinned"] = False
    src = pg.load_judge_votes("gemma-3-27b-it", _write(tmp_path, "j2", rows), "a")
    assert src["source"] == "exploratory"


def test_votes_predating_the_serving_line_field_are_read_as_pinned(tmp_path):
    """Job 825542's votes carry neither field; the older rule keeps it a run of record."""
    rows = [_row("llama-3.3-70b-fp8", "i1", "Q1", "yes")]
    for r in rows:
        r.pop("serving_line", None)
        r.pop("serving_line_is_pinned", None)
    src = pg.load_judge_votes("llama-3.3-70b-fp8", _write(tmp_path, "j3", rows), "a")
    assert src["serving_line"] == "pinned (section 6.1)"
    assert src["source"] == "pinned"


def test_the_qwen_judge_is_refused_because_it_is_the_subjects_own_family(tmp_path):
    d = _write(tmp_path, "qwen3-32b", [_row("qwen3-32b", "i1", "Q1", "yes")])
    with pytest.raises(pg.PanelGateError, match="REFUSING qwen3-32b"):
        pg.load_judge_votes("qwen3-32b", d, "a")


def test_votes_cast_on_another_q1_file_are_refused(tmp_path):
    d = _write(tmp_path, "gemma-3-27b-it", [_row("gemma-3-27b-it", "i1", "Q1", "yes", q1_variant="b")])
    with pytest.raises(pg.PanelGateError, match="not the requested variant a"):
        pg.load_judge_votes("gemma-3-27b-it", d, "a")


def test_malformed_rate_is_pooled_over_the_panels_judges(tmp_path):
    votes = {
        "gemma-3-27b-it": [_row("gemma-3-27b-it", "i1", "Q1", "malformed"),
                           _row("gemma-3-27b-it", "i2", "Q1", "no")],
        "gpt-oss-20b": [_row("gpt-oss-20b", "i1", "Q1", "no"),
                        _row("gpt-oss-20b", "i2", "Q1", "no")],
        "llama-3.3-70b-fp8": [_row("llama-3.3-70b-fp8", "i1", "Q1", "no"),
                              _row("llama-3.3-70b-fp8", "i2", "Q1", "no")],
    }
    report = pg.score_panel(_sources(tmp_path, votes),
                            {i["item_id"]: i for i in _items([("i1", "clean"), ("i2", "clean")])})
    assert report["metrics"]["malformed_rate_max"] == {"numerator": 1, "denominator": 6,
                                                       "rate": 1 / 6}
    assert report["gate_verdicts"]["specificity_clean"]["numerator"] == 2


def test_test_retest_compares_the_panel_label_across_runs(tmp_path):
    votes = {}
    for key in PANEL:
        rows = []
        for run in (0, 1, 2):
            rows.append(_row(key, "steady", "Q1", "no", run_idx=run))
        rows.append(_row(key, "flips", "Q1", "no", run_idx=0))
        rows.append(_row(key, "flips", "Q1", "yes", run_idx=1))
        rows.append(_row(key, "flips", "Q1", "yes", run_idx=2))
        votes[key] = rows
    report = pg.score_panel(_sources(tmp_path, votes),
                            {i["item_id"]: i for i in _items([("steady", "clean"), ("flips", "clean")])})
    assert report["metrics"]["test_retest_q1_min"] == {"numerator": 1, "denominator": 2, "rate": 0.5}


def test_leave_one_out_names_the_judge_whose_vote_changes_the_verdict(tmp_path):
    # Ten clean items. Gemma and Llama both say yes on the first four (false positives) and
    # gpt-oss says no everywhere. All three: the majority is yes on those four, so
    # specificity_clean is 6/10 = 0.60 against a 0.90 bar, FAIL. Drop gpt-oss and the two
    # that agree keep it at 6/10. Drop either of those two and the remaining pair splits
    # 1-1 on those four rows, which is the tie case below.
    ids = [f"i{n}" for n in range(10)]
    yes_on = set(ids[:4])
    votes = {
        "gemma-3-27b-it": [_row("gemma-3-27b-it", i, "Q1", "yes" if i in yes_on else "no") for i in ids],
        "llama-3.3-70b-fp8": [_row("llama-3.3-70b-fp8", i, "Q1", "yes" if i in yes_on else "no") for i in ids],
        "gpt-oss-20b": [_row("gpt-oss-20b", i, "Q1", "no") for i in ids],
    }
    sources = _sources(tmp_path, votes)
    items = _items([(i, "clean") for i in ids])
    report = pg.build_panel_report(sources, items, "a")
    full = report["panel"]["gate_verdicts"]["specificity_clean"]
    assert (full["numerator"], full["denominator"], full["verdict"]) == (6, 10, "FAIL")
    loo = report["leave_one_judge_out"]
    kept = loo["gpt-oss-20b"]["gate_verdicts"]["specificity_clean"]
    assert (kept["numerator"], kept["denominator"], kept["verdict"]) == (6, 10, "FAIL")
    for dropped in ("gemma-3-27b-it", "llama-3.3-70b-fp8"):
        v = loo[dropped]["gate_verdicts"]["specificity_clean"]
        assert (v["numerator"], v["denominator"], v["verdict"]) == (6, 6, "PASS")
    assert set(report["changes_the_verdict"]) == {"gemma-3-27b-it", "llama-3.3-70b-fp8"}


def test_a_tie_leaves_the_q1_denominator_instead_of_becoming_a_no(tmp_path):
    """The trap in every leave-one-out row, pinned so it cannot be read as a pass.

    Two judges split 1-1. With a gate vote the row's Q1 label becomes the GATE outcome
    token, which is neither yes nor no, and with no gate vote the row has no label at all.
    Either way the row leaves the Q1 numerator AND its denominator, so a two-judge panel
    can score better than the three-judge panel it came from purely by dropping rows.
    """
    ids = [f"i{n}" for n in range(10)]
    yes_on = set(ids[:4])
    rows_gate = [_row("gemma-3-27b-it", i, "gate", "coherent") for i in ids]
    votes = {
        "gemma-3-27b-it": [_row("gemma-3-27b-it", i, "Q1", "yes" if i in yes_on else "no")
                           for i in ids] + rows_gate,
        "llama-3.3-70b-fp8": [_row("llama-3.3-70b-fp8", i, "Q1", "no") for i in ids]
                             + [_row("llama-3.3-70b-fp8", i, "gate", "coherent") for i in ids],
    }
    sources = _sources(tmp_path, votes)
    items = {i["item_id"]: i for i in _items([(i, "clean") for i in ids])}
    report = pg.score_panel(sources, items)
    v = report["gate_verdicts"]["specificity_clean"]
    assert (v["numerator"], v["denominator"], v["verdict"]) == (6, 6, "PASS")
    assert report["metrics"]["panel_q1_ties"]["numerator"] == 4
    assert report["per_class_counts"]["clean"]["q1"]["tie"] == 4
    assert report["per_class_counts"]["clean"]["q1"]["total"] == 10
    labels = pg.panel_labels(sources, 0)
    assert labels["i0"]["q1"] == "coherent" and labels["i0"]["q1_resolution"] == "tie_to_gate"


@pytest.mark.skipif(not (LLAMA_A / "votes.jsonl").exists(), reason="vote file not mirrored")
def test_a_one_judge_panel_reproduces_that_judges_committed_report():
    """With one judge the panel label IS that judge's vote, so every number must match."""
    sources = {"llama-3.3-70b-fp8": pg.load_judge_votes("llama-3.3-70b-fp8", LLAMA_A, "a")}
    items = [json.loads(x) for x in ITEMS.read_text().splitlines() if x.strip()]
    report = pg.build_panel_report(sources, items, "a")
    committed = json.loads((REPO / "experiments/jury/gate_report_llama-3.3-70b-fp8.json").read_text())
    theirs = committed["per_judge"][0]["gate_verdicts"]
    ours = report["panel"]["gate_verdicts"]
    for name in ("recall_planted_mention", "recall_paraphrased_disclosure", "recall_quoted_denied",
                 "specificity_clean", "specificity_deleted_step", "specificity_restated_cue_only",
                 "gate_accuracy_gate_positive", "gate_accuracy_clean", "malformed_rate_max",
                 "test_retest_q1_min"):
        assert (ours[name]["numerator"], ours[name]["denominator"]) == \
               (theirs[name]["numerator"], theirs[name]["denominator"]), name
    assert report["panel"]["verdict"] == committed["per_judge"][0]["verdict"] == "FAIL"


def test_a_panel_missing_a_judge_is_marked_partial_everywhere(tmp_path):
    """A one- or two-judge panel must never reach a table looking like the panel of record."""
    from experiments.jury import gate_matrix

    votes = {"llama-3.3-70b-fp8": [_row("llama-3.3-70b-fp8", "i1", "Q1", "no")]}
    report = pg.build_panel_report(_sources(tmp_path, votes), _items([("i1", "clean")]), "a")
    assert report["kind"] == "PANEL-PARTIAL"
    assert report["panel_complete"] is False
    assert report["missing_judges"] == ["gemma-3-27b-it", "gpt-oss-20b"]
    assert report["expected_panel"] == ["gemma-3-27b-it", "gpt-oss-20b", "llama-3.3-70b-fp8"]
    rows = gate_matrix._panel_rows(report)
    assert [r["kind"] for r in rows] == ["PANEL-PARTIAL"]


def test_panel_row_serving_line_carries_the_budget_marker_from_the_run_slug(tmp_path):
    """Two panels on the same exploratory line at different BCF_NUM_PREDICT budgets must not
    print identical Serving line cells: the budget is only recorded in the run slug."""
    from experiments.jury import gate_matrix

    votes = {
        "gemma-3-27b-it": [_row("gemma-3-27b-it", "i1", "Q1", "no")],
        "gpt-oss-20b": [_row("gpt-oss-20b", "i1", "Q1", "no")],
        "llama-3.3-70b-fp8": [_row("llama-3.3-70b-fp8", "i1", "Q1", "no")],
    }
    report = pg.build_panel_report(_sources(tmp_path, votes), _items([("i1", "clean")]), "a")
    cases = {
        "experiments/results/jury-gate/gpt-oss-20b-h200-np1024-q1d/arc_challenge/stated-hint": "@np1024",
        "experiments/results/jury-gate/qwen3-32b-h200-np1024-q1a/arc_challenge/stated-hint": "@np1024",
        "experiments/results/jury-gate/some-judge-np256/arc_challenge/stated-hint": "@np256",
        "experiments/results/jury-gate/gpt-oss-20b-h200-q1a/arc_challenge/stated-hint": "",
        "experiments/results/jury-gate/gpt-oss-20b-snp1024-q1a/arc_challenge/stated-hint": "",
    }
    for src_dir, tag in cases.items():
        report["panel"]["sources"]["gpt-oss-20b"]["dir"] = src_dir
        for block in report["leave_one_judge_out"].values():
            if "gpt-oss-20b" in block["sources"]:
                block["sources"]["gpt-oss-20b"]["dir"] = src_dir
        rows = gate_matrix._panel_rows(report)
        cell = rows[0]["serving_line"]
        segments = dict(seg.split(":", 1) for seg in cell.split(" + "))
        assert set(segments) == {"gemma-3-27b-it", "gpt-oss-20b", "llama-3.3-70b-fp8"}, cell
        if tag:
            assert segments["gpt-oss-20b"].endswith(tag), (src_dir, cell)
        else:
            assert "@np" not in segments["gpt-oss-20b"], (src_dir, cell)
        # the other two judges never pick up a tag from gpt-oss's slug
        assert "@np" not in segments["gemma-3-27b-it"], cell
        assert "@np" not in segments["llama-3.3-70b-fp8"], cell
        # the leave-one-out rows that still contain gpt-oss carry the same tag
        for row in rows[1:]:
            if "gpt-oss-20b:" in row["serving_line"]:
                loo = dict(seg.split(":", 1) for seg in row["serving_line"].split(" + "))
                assert loo["gpt-oss-20b"].endswith(tag) if tag else "@np" not in loo["gpt-oss-20b"]


def test_per_judge_row_serving_line_carries_the_budget_marker_from_the_report_slug(tmp_path):
    """The same rule for a per-judge row: gate_report_gpt-oss-20b-h200-np1024-q1d.json and
    gate_report_gpt-oss-20b-h200-q1d.json were cast on one serving line at two budgets."""
    from experiments.jury import gate_matrix

    def report(slug):
        judge = {"judge_key": "gpt-oss-20b", "votes": 3, "verdict": "FAIL", "failed_metrics": [],
                 "gate_verdicts": {}, "per_class_counts": {}}
        path = tmp_path / f"gate_report_{slug}.json"
        path.write_text(json.dumps({"serving_line": "exploratory-h200-141",
                                    "prompt_files": {"Q1": "q1_mention_2026-09-07d.md"},
                                    "per_judge": [judge]}))
        return gate_matrix.rows_from_report(path, {})[0]["serving_line"]

    assert report("gpt-oss-20b-h200-np1024-q1d") == "exploratory-h200-141@np1024"
    assert report("gpt-oss-20b-h200-q1d") == "exploratory-h200-141"
    assert report("qwen3-32b-h200-np1024-q1a") == "exploratory-h200-141@np1024"
    assert report("some-judge-np256") == "exploratory-h200-141@np256"
