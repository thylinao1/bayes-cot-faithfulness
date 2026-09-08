"""The synthetic gate corpus is controlled, and the gate can actually fail.

A gate that has never been seen to go red is decorative, so the last two tests force a
judge that answers wrongly and assert the FAIL, with the denominator it was computed on.
"""

from __future__ import annotations

import json

import pytest

from experiments.jury import gate as gate_mod
from experiments.jury import synthetic_gate as sg
from experiments.jury.backends import BackendError, JudgeEndpoint
from experiments.jury.family_map import JUDGE_BY_KEY, routing
from experiments.jury.gate_thresholds import THRESHOLDS
from experiments.jury.prompt_files import load_prompts
from experiments.jury.runner import JuryItem, JuryRunner

REPO_BANK = [
    "experiments/results/phase1-skeleton/qwen3-8b/arc_challenge/stated-hint/transcripts.jsonl",
    "experiments/results/control_transcripts_llama-3.1-8b-instant.json",
]


@pytest.fixture(scope="module")
def bank():
    rows = sg.load_bank(REPO_BANK)
    if not rows:
        pytest.skip("banked transcripts are not present in this checkout")
    return rows


def test_bank_and_corpus_denominators(bank):
    items = sg.build_items(bank)
    counts = sg.class_counts(items)
    assert set(counts) == set(sg.CLASSES)
    assert len(set(counts.values())) == 1, counts
    per_class = next(iter(counts.values()))
    assert per_class >= 60, f"only {per_class} items per class from {len(bank)} banked rows"
    assert len(items) == per_class * len(sg.CLASSES)


def test_every_item_id_is_unique(bank):
    items = sg.build_items(bank)
    assert len({i["item_id"] for i in items}) == len(items)


def test_planted_positive_is_the_clean_negative_plus_exactly_one_line(bank):
    items = sg.build_items(bank)
    by = {}
    for item in items:
        by.setdefault((item["meta"]["base_source"], item["meta"]["base_index"]), {})[
            item["meta"]["gate_class"]
        ] = item
    for group in by.values():
        clean = group["clean"]["reasoning"].splitlines()
        planted = group["planted_mention"]["reasoning"].splitlines()
        assert planted[: len(clean)] == clean
        assert len(planted) == len(clean) + 1


def test_gate_positive_keeps_the_reasoning_and_moves_only_the_answer(bank):
    items = sg.build_items(bank)
    by = {}
    for item in items:
        by.setdefault((item["meta"]["base_source"], item["meta"]["base_index"]), {})[
            item["meta"]["gate_class"]
        ] = item
    for group in by.values():
        clean, override = group["clean"], group["gate_positive"]
        assert override["reasoning"] == clean["reasoning"]
        assert override["final_answer"] != clean["final_answer"]
        assert override["final_answer"] != override["meta"]["cue_letter"]
        assert override["meta"]["truth"]["gate"] == "silent_override"


def test_restated_cue_class_carries_the_cue_words_but_is_a_negative(bank):
    items = [i for i in sg.build_items(bank) if i["meta"]["gate_class"] == "restated_cue_only"]
    assert items
    for item in items:
        assert item["meta"]["truth"]["q1"] is False
        assert "(" + item["meta"]["cue_letter"] + ")" in item["reasoning"]


def test_deleted_step_removes_a_line_and_keeps_the_answer(bank):
    items = sg.build_items(bank)
    by = {}
    for item in items:
        by.setdefault((item["meta"]["base_source"], item["meta"]["base_index"]), {})[
            item["meta"]["gate_class"]
        ] = item
    shorter = 0
    for group in by.values():
        clean, cut = group["clean"], group["deleted_step"]
        assert cut["final_answer"] == clean["final_answer"]
        if len(cut["reasoning"].splitlines()) < len(clean["reasoning"].splitlines()):
            shorter += 1
    assert shorter == len(by)


def test_phrasings_are_balanced_across_the_corpus(bank):
    items = [i for i in sg.build_items(bank) if i["meta"]["gate_class"] == "planted_mention"]
    seen = {0: 0, 1: 0, 2: 0}
    for item in items:
        seen[item["meta"]["phrasing_idx"]] += 1
    assert min(seen.values()) >= len(items) // 3 - 1


# --- the gate runs, and the gate fails ------------------------------------------


class _ScriptedClient:
    """A judge that answers by question, so the gate metrics are forced by construction."""

    def __init__(self, q1_vote="oracle"):
        self.q1_vote = q1_vote
        self.seed = 0
        self.base_url = "http://fake/v1"
        self.model = "fake"
        self.calls = 0

    def is_available(self):
        return True

    def generate(self, prompt, *, num_predict=320):
        self.calls += 1
        if '"vote": "coherent"' in prompt:
            override = "OVERRIDE-MARK" in prompt
            vote = "silent_override" if override else "coherent"
            return json.dumps({"vote": vote, "implied_answer": "A", "given_answer": "B", "rationale": "r"})
        if '"vote": "yes"' in prompt:
            if self.q1_vote == "oracle":
                vote = "yes" if "PLANT-MARK" in prompt else "no"
            else:
                vote = self.q1_vote
            return json.dumps({"vote": vote, "quote": "", "describes_effect_on_choice": vote == "yes", "rationale": "r"})
        return json.dumps({"vote": "supported", "load_bearing_step": "s", "rationale": "r"})


def _marked_items(bank, n_per_class=6):
    """A small corpus with markers the scripted judge can read, standing in for judgment."""
    items = sg.build_items(bank, per_class=n_per_class)
    for item in items:
        cls = item["meta"]["gate_class"]
        if item["meta"]["truth"].get("q1") is True:
            item["reasoning"] += "\nPLANT-MARK"
        if item["meta"]["truth"].get("gate") == "silent_override":
            item["reasoning"] += "\nOVERRIDE-MARK"
        item["meta"]["gate_class"] = cls
    return items


def _endpoints(judge_keys, q1_vote="oracle"):
    eps = {}
    for key in judge_keys:
        judge = JUDGE_BY_KEY[key]
        eps[key] = JudgeEndpoint(
            judge_key=key, backend="vllm", model=judge.hf_id, revision=judge.revision,
            client=_ScriptedClient(q1_vote),
        )
    return eps


def _run_gate(tmp_path, bank, q1_vote, *, q1="a", serving_line=""):
    raw = _marked_items(bank)
    items = [JuryItem(**dict(r)) for r in raw]
    panel = list(routing("Qwen3-8B"))
    out = tmp_path / "jury-gate" / "arc_challenge" / "stated-hint"
    prompts = load_prompts(q1=q1)
    runner = JuryRunner(
        endpoints=_endpoints(panel, q1_vote), prompts=prompts, out_dir=out,
        substrate="arc_challenge", cue_family="stated-hint", mode="three-seeded",
        position_swap="first-run", serving_line=serving_line,
        serving_line_note="the run of record is the pinned line" if serving_line else "",
    )
    summary = runner.run(items, progress_every=0, concurrency=4)
    report = gate_mod.build_report(
        runner.out_dir, raw, summary, panel, prompts=prompts,
        serving_line=serving_line,
        serving_line_note="the run of record is the pinned line" if serving_line else "",
    )
    return report, runner


def test_an_oracle_judge_passes_every_threshold(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "oracle")
    assert report["verdict"] == "PASS", json.dumps(report["per_judge"][0]["gate_verdicts"], indent=2)
    for judge in report["per_judge"]:
        assert judge["failed_metrics"] == []
        assert judge["metrics"]["recall_planted_mention"]["denominator"] == 6
        assert judge["metrics"]["test_retest_q1_min"]["rate"] == 1.0


def test_a_judge_that_never_says_yes_FAILS_recall_with_its_denominator(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "no")
    assert report["verdict"] == "FAIL"
    judge = report["per_judge"][0]
    assert "recall_planted_mention" in judge["failed_metrics"]
    m = judge["gate_verdicts"]["recall_planted_mention"]
    assert m["verdict"] == "FAIL"
    assert m["numerator"] == 0
    assert m["denominator"] == 6
    assert m["rate"] == 0.0
    assert m["threshold"] == THRESHOLDS["recall_planted_mention"]
    # The matched negative class still passes, which is what makes the failure specific.
    assert judge["gate_verdicts"]["specificity_clean"]["verdict"] == "PASS"


def test_a_judge_that_always_says_yes_FAILS_specificity(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "yes")
    judge = report["per_judge"][0]
    assert report["verdict"] == "FAIL"
    assert "specificity_clean" in judge["failed_metrics"]
    assert judge["gate_verdicts"]["specificity_clean"]["numerator"] == 0
    assert judge["gate_verdicts"]["specificity_clean"]["denominator"] == 6


def test_report_carries_the_threshold_hash_and_the_prompt_hashes(tmp_path, bank):
    report, _ = _run_gate(tmp_path, bank, "oracle")
    assert len(report["thresholds_sha256"]) == 64
    assert set(report["prompt_sha256"]) == {"gate", "Q1", "Q2"}
    assert report["thresholds"] == THRESHOLDS
    assert report["votes_per_second_per_server"] > 0


def test_corpus_manifest_matches_a_fresh_build(bank):
    """The committed manifest is the corpus's fingerprint; a drifted builder fails here."""
    import hashlib
    import json as _json
    from pathlib import Path as _Path

    manifest = _json.loads(
        _Path("experiments/jury/gate_corpus_manifest.json").read_text(encoding="utf-8")
    )
    items = sg.build_items(bank)
    assert manifest["items"] == len(items)
    assert manifest["per_class"] == sg.class_counts(items)
    seq = hashlib.sha256("\n".join(i["item_id"] for i in items).encode()).hexdigest()
    assert manifest["item_sequence_sha256"] == seq
    assert manifest["bank_rows_usable"] == len(bank)


def test_the_report_names_the_q1_file_it_scored(tmp_path, bank):
    """A report that does not name its Q1 file cannot be compared with another one."""
    a, _ = _run_gate(tmp_path / "a", bank, "oracle", q1="a")
    b, _ = _run_gate(tmp_path / "b", bank, "oracle", q1="b")
    assert a["prompt_files"]["Q1"] == "q1_mention_2026-09-07.md"
    assert b["prompt_files"]["Q1"] == "q1_mention_2026-09-07b.md"
    assert a["q1_prompt_variant"] == "a" and b["q1_prompt_variant"] == "b"
    assert a["prompt_sha256"]["Q1"] != b["prompt_sha256"]["Q1"]
    # The gate and Q2 instruments never move with the Q1 choice.
    assert a["prompt_sha256"]["gate"] == b["prompt_sha256"]["gate"]
    assert a["prompt_sha256"]["Q2"] == b["prompt_sha256"]["Q2"]
    assert a["thresholds"] == b["thresholds"] == THRESHOLDS


def test_an_off_line_run_is_labelled_exploratory_on_every_vote_and_in_the_report(tmp_path, bank):
    report, runner = _run_gate(
        tmp_path, bank, "oracle", serving_line="exploratory-h200-141",
    )
    assert report["serving_line"] == "exploratory-h200-141"
    assert report["serving_line_is_pinned"] is False
    rows = [json.loads(x) for x in
            (runner.out_dir / "votes.jsonl").read_text().splitlines() if x.strip()]
    assert rows
    assert all(r["serving_line"] == "exploratory-h200-141" for r in rows)
    assert all(r["serving_line_is_pinned"] is False for r in rows)


def test_a_pinned_run_records_the_judge_own_serving_line(tmp_path, bank):
    report, runner = _run_gate(tmp_path, bank, "oracle")
    assert report["serving_line_is_pinned"] is True
    rows = [json.loads(x) for x in
            (runner.out_dir / "votes.jsonl").read_text().splitlines() if x.strip()]
    for r in rows:
        assert r["serving_line"] == JUDGE_BY_KEY[r["judge_key"]].serving_line
        assert r["serving_line_is_pinned"] is True


# --- a crash is not a verdict ------------------------------------------------------


def test_a_backend_failure_exits_4_and_writes_no_report(tmp_path, bank, monkeypatch):
    """An unreachable judge must not exit 1, which means a FAILED THRESHOLD.

    Job 828627 died inside the runner (the jury probe raised BackendError), wrote no
    gate_report.json, and exited 1. bcf/judge_serve.sbatch reads 1 as a result and
    recorded the variant as exit code 0. The gate now separates the two: 1 is a judge
    that missed a bar, 4 is judging that never happened.
    """
    raw = _marked_items(bank, n_per_class=1)
    items_path = tmp_path / "items.jsonl"
    items_path.write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
    out = tmp_path / "g" / "arc_challenge" / "stated-hint"

    def unreachable(self, items, **kw):
        raise BackendError(
            "judge llama-3.3-70b-fp8 is unreachable and the SoCLaaS fallback is not "
            "permitted (no key in the environment, or no eligibility ruling on the record)"
        )

    monkeypatch.setattr(JuryRunner, "run", unreachable)
    rc = gate_mod.main([
        "--items", str(items_path), "--out", str(out),
        "--judge", "llama-3.3-70b-fp8=http://127.0.0.1:1/v1",
        "--all-judge-rows",
    ])
    assert rc == gate_mod.GATE_INFRA_FAILURE == 4
    assert not (out / "gate_report.json").exists(), (
        "a crashed gate must leave no report; a report is what says a gate ran"
    )


def test_a_failed_threshold_still_exits_1(tmp_path, bank, monkeypatch):
    """The other end of the same rule: a judge that answers and misses a bar is a 1."""
    raw = _marked_items(bank, n_per_class=1)
    items_path = tmp_path / "items.jsonl"
    items_path.write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
    out = tmp_path / "g" / "arc_challenge" / "stated-hint"

    def scripted(key, url, *, seed):
        judge = JUDGE_BY_KEY[key]
        return JudgeEndpoint(
            judge_key=key, backend="vllm", model=judge.hf_id, revision=judge.revision,
            client=_ScriptedClient("no"),
        )

    monkeypatch.setattr("experiments.jury.backends.vllm_endpoint", scripted)
    rc = gate_mod.main([
        "--items", str(items_path), "--out", str(out),
        "--judge", "llama-3.3-70b-fp8=http://127.0.0.1:1/v1",
        "--all-judge-rows", "--concurrency", "2",
    ])
    assert rc == 1
    report = json.loads((out / "gate_report.json").read_text())
    assert report["verdict"] == "FAIL"


# --- the side by side comparison --------------------------------------------------


def test_class_of_survives_a_source_name_that_contains_hyphens():
    from experiments.jury import gate_compare as gc

    assert gc.class_of("gate-planted_mention-transcripts.jsonl-0000") == "planted_mention"
    assert gc.class_of(
        "gate-quoted_denied-control_transcripts_llama-3.1-8b-instant.json-0021"
    ) == "quoted_denied"
    with pytest.raises(ValueError):
        gc.class_of("gate-not_a_class-transcripts.jsonl-0000")


def test_phrasing_index_follows_corpus_order_not_vote_order(bank):
    """The three frozen templates rotate by POSITION IN THE CORPUS.

    A vote file is written concurrently, so its line order is not the corpus order. Reading
    the phrasing off the vote file therefore attributes counts to the wrong template, which
    is exactly the kind of number that reads as a finding and is an artifact.
    """
    from experiments.jury import gate_compare as gc

    items = sg.build_items(bank)
    for item in items:
        item.setdefault("meta", {})
    idx = gc.phrasing_index(items)
    per_class: dict[str, list[str]] = {}
    for item in items:
        per_class.setdefault(gc.class_of(item["item_id"]), []).append(item["item_id"])
    for cls, ids in per_class.items():
        for k, item_id in enumerate(ids):
            assert idx[item_id] == k % 3, f"{cls} item {k}"


def test_an_own_family_judge_has_no_task_until_all_judge_rows_is_set(tmp_path, bank):
    """The panel rule routes a judge away from its own family.

    The gate corpus's subject_model is Qwen3-8B, so the Qwen judge is off the panel for
    every item and a gate run that serves only that judge plans ZERO votes. Job 826022 did
    exactly that on the h200: exit 0, three reports, nothing measured. The gate scores
    per-judge error on known truth, which section 6.2 identifies from its all-four
    subsample, so the own-family case is in scope when it is asked for and recorded.
    """
    raw = _marked_items(bank)
    items = [JuryItem(**dict(r)) for r in raw]
    eps = _endpoints(["qwen3-32b"], "oracle")
    out = tmp_path / "jury-gate" / "arc_challenge" / "stated-hint"
    kw = {
        "endpoints": eps, "prompts": load_prompts(), "substrate": "arc_challenge",
        "cue_family": "stated-hint", "mode": "three-seeded", "position_swap": "first-run",
        "judge_filter": ("qwen3-32b",),
    }
    routed = JuryRunner(out_dir=out / "routed", **kw)
    assert routed._tasks(items) == []

    forced = JuryRunner(out_dir=out / "forced", all_judge_rows=True, **kw)
    tasks = forced._tasks(items)
    assert tasks
    assert {t["judge_key"] for t in tasks} == {"qwen3-32b"}
    summary = forced.run(items, progress_every=0, concurrency=4)
    assert summary["votes"] == summary["votes_planned"] > 0
    rows = [json.loads(x) for x in
            (forced.out_dir / "votes.jsonl").read_text().splitlines() if x.strip()]
    assert all(r["all_judge_row"] is True for r in rows)
    assert all(r["own_family_vote"] is True for r in rows)
    # An own-family vote is recorded and stays out of the panel label.
    labels = [json.loads(x) for x in
              (forced.out_dir / "panel_labels.jsonl").read_text().splitlines() if x.strip()]
    assert labels
    assert all("qwen3-32b" not in row["panel"] for row in labels)


def test_specificity_numerator_is_the_no_count_not_the_yes_count(tmp_path, bank):
    """A truth-false class is scored on NO votes, and the two ends must not be confused.

    A verifier reading these reports counted the YES votes on `restated_cue_only` and
    compared them against `specificity_restated_cue_only`, then reported a 17 item error
    where there was none: on Q1 variant b the class is 43 yes and 26 no, the specificity is
    26/69, and 43 and 26 are the same measurement from its two ends. This test pins the
    direction so the report cannot silently flip it: for every truth-false class the gate
    numerator is the NO count, for every truth-true class it is the YES count, and yes plus
    no is the denominator in both cases.
    """
    raw = _marked_items(bank)
    items = [JuryItem(**dict(r)) for r in raw]
    items_by_id = {r["item_id"]: r for r in raw}
    eps = _endpoints(["llama-3.3-70b-fp8"], "oracle")
    runner = JuryRunner(
        out_dir=tmp_path / "g" / "arc_challenge" / "stated-hint",
        endpoints=eps, prompts=load_prompts(), substrate="arc_challenge",
        cue_family="stated-hint", mode="three-seeded", position_swap="first-run",
        judge_filter=("llama-3.3-70b-fp8",),
    )
    runner.run(items, progress_every=0, concurrency=4)
    rows = [json.loads(x) for x in
            (runner.out_dir / "votes.jsonl").read_text().splitlines() if x.strip()]
    scored = gate_mod.score_judge(rows, items_by_id, "llama-3.3-70b-fp8")
    counts = scored["per_class_counts"]

    checked = 0
    for cls, truth in sg.TRUTH.items():
        q1_truth = truth.get("q1")
        if q1_truth is None or cls not in counts:
            continue
        yes = counts[cls]["q1"]["yes"]
        no = counts[cls]["q1"]["no"]
        name = f"recall_{cls}" if q1_truth else f"specificity_{cls}"
        metric = scored["metrics"][name]
        assert metric["denominator"] == yes + no, name
        assert metric["numerator"] == (yes if q1_truth else no), name
        checked += 1
    assert checked == 7, "expected 3 recall classes and 4 Q1 specificity classes"


def test_recount_tool_reproduces_the_report_and_separates_run0_from_the_union(tmp_path, bank):
    """The recount tool has to agree with the report, and expose the OTHER counting trap.

    Gate metrics use run 0 unswapped. A count of items with at least one yes across the
    three seeded runs is a different number whenever a row did not repeat, which is where
    the same verifier's 53 (against run 0's 52) and 49 (against 48) came from. The tool
    prints both, and this asserts the run-0 half matches the report exactly and that the
    union is never smaller than the run-0 yes count.
    """
    from experiments.jury import recount_gate_q1 as rc

    raw = _marked_items(bank)
    items = [JuryItem(**dict(r)) for r in raw]
    items_by_id = {r["item_id"]: r for r in raw}
    eps = _endpoints(["llama-3.3-70b-fp8"], "oracle")
    runner = JuryRunner(
        out_dir=tmp_path / "g" / "arc_challenge" / "stated-hint",
        endpoints=eps, prompts=load_prompts(), substrate="arc_challenge",
        cue_family="stated-hint", mode="three-seeded", position_swap="first-run",
        judge_filter=("llama-3.3-70b-fp8",),
    )
    runner.run(items, progress_every=0, concurrency=4)
    votes_path = runner.out_dir / "votes.jsonl"
    rows = [json.loads(x) for x in votes_path.read_text().splitlines() if x.strip()]
    scored = gate_mod.score_judge(rows, items_by_id, "llama-3.3-70b-fp8")

    classes = {r["item_id"]: r["meta"]["gate_class"] for r in raw}
    out = rc.recount(votes_path, classes, judge_key="llama-3.3-70b-fp8")

    for cls, c in out["per_class"].items():
        if c["gate_metric"] is None:
            continue
        metric = scored["metrics"][c["gate_metric"]]
        assert c["gate_numerator"] == metric["numerator"], cls
        assert c["gate_denominator"] == metric["denominator"], cls
        assert c["union_any_run_yes"] >= c["run0_yes"], cls
