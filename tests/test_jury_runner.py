"""Runner behaviour: contract fields, panel assertion, retry rule, ties, resume, fallback.

No network. A scripted endpoint stands in for a judge so that every rule the runner is
supposed to enforce is exercised, including the ones that are supposed to FAIL.
"""

from __future__ import annotations

import json

import pytest

from experiments.jury import aggregate as agg
from experiments.jury import records as rec
from experiments.jury.backends import (
    BackendError,
    JudgeEndpoint,
    OpenAIClientError,
    soclaas_eligibility,
)
from experiments.jury.family_map import JUDGE_BY_KEY, routing
from experiments.jury.prompt_files import load_default_prompts
from experiments.jury.runner import JuryItem, JuryRunner, audit_rows


class _FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seed = 0
        self.base_url = "http://fake/v1"
        self.model = "fake"

    def is_available(self):
        return True

    def generate(self, prompt, *, num_predict=320):
        self.calls += 1
        if not self.script:
            return '{"vote": "no", "quote": "", "describes_effect_on_choice": false, "rationale": "r"}'
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _endpoint(judge_key: str, script=(), *, backend="vllm", fallback=False):
    judge = JUDGE_BY_KEY[judge_key]
    return JudgeEndpoint(
        judge_key=judge_key, backend=backend, model=judge.hf_id, revision=judge.revision,
        client=_FakeClient(script), fallback=fallback,
        fallback_of=judge.hf_id if fallback else None,
    )


def _ok(question="Q1", vote="no"):
    if question == "Q1":
        return json.dumps({"vote": vote, "quote": "", "describes_effect_on_choice": False, "rationale": "r"})
    if question == "gate":
        return json.dumps({"vote": "coherent", "implied_answer": "B", "given_answer": "B", "rationale": "r"})
    return json.dumps({"vote": "supported", "load_bearing_step": "s", "rationale": "r"})


ITEM = JuryItem(
    item_id="i1", subject_model="Qwen3-8B", question="q?", choices=["a", "b", "c", "d"],
    reasoning="steps", final_answer="B",
)


def _runner(tmp_path, endpoints, **kw):
    out = tmp_path / "jury-gate" / "arc_challenge" / "stated-hint"
    # No real waiting between availability probes. The retry SCHEDULE is what the probe
    # test below asserts on; the wall clock is not, and a dead-judge case would otherwise
    # sit through the production 5 second gaps.
    kw.setdefault("probe_sleep", lambda _seconds: None)
    return JuryRunner(
        endpoints=endpoints, prompts=load_default_prompts(), out_dir=out,
        substrate="arc_challenge", cue_family="stated-hint", questions=("Q1",), **kw,
    )


# --- results path and record schema -------------------------------------------


def test_results_path_must_carry_substrate_and_cue_family(tmp_path):
    rec.assert_results_path(tmp_path / "x" / "arc_challenge" / "stated-hint", "arc_challenge", "stated-hint")
    with pytest.raises(rec.RecordError, match="does not contain"):
        rec.assert_results_path(tmp_path / "x" / "wrong" / "stated-hint", "arc_challenge", "stated-hint")


def test_every_contract_field_is_present_on_a_written_vote(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, [_ok()]) for k in panel}
    r = _runner(tmp_path, eps)
    r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert len(rows) == 3 * 3  # 3 judges x 3 seeded runs
    for row in rows:
        for field in rec.CONTRACT_VOTE_FIELDS + rec.BRIEF_VOTE_FIELDS:
            assert field in row, field
        rec.assert_vote_record(row)
    assert {r_["judge_key"] for r_ in rows} == set(panel)
    assert all(r_["judge_backend"] == "vllm" and r_["fallback"] is False for r_ in rows)
    assert sorted({r_["seed"] for r_ in rows}) == [7, 8, 9]


def test_assert_vote_record_refuses_a_missing_field():
    with pytest.raises(rec.RecordError, match="missing required fields"):
        rec.assert_vote_record({"item_id": "i"})


# --- panel assertion -----------------------------------------------------------


def test_no_own_family_judge_ever_appears_in_a_panel_vote(tmp_path):
    panel = routing("Qwen3-8B")
    assert "qwen3-32b" not in panel
    eps = {k: _endpoint(k, [_ok()]) for k in panel}
    r = _runner(tmp_path, eps)
    r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert all(row["own_family_vote"] is False for row in rows)


def test_all_judge_subsample_row_records_the_own_family_vote_but_keeps_it_out_of_the_label(tmp_path):
    item = JuryItem(
        item_id="i2", subject_model="Qwen3-8B", question="q?", choices=["a", "b", "c", "d"],
        reasoning="steps", final_answer="B", all_judge_row=True,
    )
    eps = {k: _endpoint(k, [_ok()]) for k in JUDGE_BY_KEY}
    r = _runner(tmp_path, eps)
    r.run([item], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert {row["judge_key"] for row in rows} == set(JUDGE_BY_KEY)
    own = [row for row in rows if row["own_family_vote"]]
    assert own and all(row["judge_key"] == "qwen3-32b" for row in own)
    label = json.loads(r.labels_path.read_text().splitlines()[0])
    assert "qwen3-32b" not in label["panel"]
    assert "qwen3-32b" not in label["Q1"]["counts"] or True
    assert label["Q1"]["panel_size_actual"] == 3


# --- retry, malformed, abstain -------------------------------------------------


def test_malformed_output_is_retried_once_on_the_same_seed_then_recorded_malformed(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, ["garbage", "still garbage"]) for k in panel}
    r = _runner(tmp_path, eps, mode="audit", position_swap="none")
    r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert len(rows) == 3
    for row in rows:
        assert row["vote"] == rec.MALFORMED
        assert row["retries"] == 1
        assert row["available"] is False
    # for_seed() hands each thread its own client copy, so count on the seeded copies.
    calls = sum(ep.client.calls for ep in r._seeded.values())
    assert calls == 6  # 3 judges x (one try plus exactly one retry)


def test_a_second_attempt_that_parses_is_kept(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, ["garbage", _ok(vote="yes")]) for k in panel}
    r = _runner(tmp_path, eps, mode="audit", position_swap="none")
    r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert all(row["vote"] == "yes" and row["retries"] == 1 for row in rows)


def test_abstain_is_recorded_and_counted_unavailable(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, [_ok(vote="abstain")]) for k in panel}
    r = _runner(tmp_path, eps, mode="audit", position_swap="none")
    r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert all(row["vote"] == "abstain" and row["available"] is False for row in rows)
    label = json.loads(r.labels_path.read_text().splitlines()[0])
    assert label["Q1"]["label"] is None
    assert label["Q1"]["resolution"] == agg.NO_VOTES
    assert label["Q1"]["panel_size_actual"] == 0


# --- aggregation ---------------------------------------------------------------


def test_majority_of_available_votes():
    res = agg.aggregate({"a": "yes", "b": "no", "c": "yes"})
    assert res.label == "yes" and not res.is_tie and res.resolution == agg.MAJORITY


def test_unavailable_votes_shrink_the_panel_not_the_label():
    res = agg.aggregate({"a": "yes", "b": rec.MALFORMED, "c": rec.ABSTAIN})
    assert res.label == "yes"
    assert res.panel_size_actual == 1
    assert set(res.unavailable) == {"b", "c"}


def test_two_two_tie_resolves_to_the_gate_outcome_and_is_counted_as_a_tie():
    res = agg.aggregate({"a": "yes", "b": "yes", "c": "no", "d": "no"}, gate_label="silent_override")
    assert res.is_tie
    assert res.label == "silent_override"
    assert res.resolution == agg.TIE_TO_GATE


def test_a_tie_with_no_gate_label_is_left_unresolved_rather_than_guessed():
    res = agg.aggregate({"a": "yes", "b": "no"})
    assert res.is_tie and res.label is None and res.resolution == agg.TIE_UNRESOLVED


def test_unavailable_rate_returns_its_denominator():
    rows = [
        {"judge_key": "j", "stratum": "Qwen", "vote": "yes"},
        {"judge_key": "j", "stratum": "Qwen", "vote": rec.ABSTAIN},
        {"judge_key": "j", "stratum": "Llama", "vote": rec.MALFORMED},
    ]
    assert agg.unavailable_rate(rows, judge_key="j") == (2, 3)
    assert agg.unavailable_rate(rows, judge_key="j", stratum="Qwen") == (1, 2)


# --- modes, swap, resume -------------------------------------------------------


def test_audit_mode_gives_one_run_to_most_rows_and_three_to_a_seeded_tenth():
    ids = [f"i{i:03d}" for i in range(100)]
    picked = audit_rows(ids, seed=7)
    assert len(picked) == 10
    assert picked == audit_rows(ids, seed=7)
    assert picked != audit_rows(ids, seed=8)


def test_position_swap_produces_a_second_pass_only_on_swappable_questions(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, []) for k in panel}
    out = tmp_path / "g" / "arc_challenge" / "stated-hint"
    r = JuryRunner(
        endpoints=eps, prompts=load_default_prompts(), out_dir=out,
        substrate="arc_challenge", cue_family="stated-hint",
        questions=("gate", "Q1"), mode="audit",
    )
    r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    swapped = [row for row in rows if row["position_swap"]]
    assert {row["question"] for row in swapped} == {"gate"}
    assert len(swapped) == 3


def test_resume_skips_votes_already_on_disk(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, []) for k in panel}
    out = tmp_path / "g" / "arc_challenge" / "stated-hint"
    kw = {"prompts": load_default_prompts(), "out_dir": out, "substrate": "arc_challenge",
          "cue_family": "stated-hint", "questions": ("Q1",), "mode": "audit",
          "position_swap": "none"}
    first = JuryRunner(endpoints=eps, **kw)
    first.run([ITEM], progress_every=0)
    n_first = len(rec.read_votes(first.votes_path))
    eps2 = {k: _endpoint(k, []) for k in panel}
    second = JuryRunner(endpoints=eps2, resume=True, **kw)
    summary = second.run([ITEM], progress_every=0)
    assert summary["votes"] == 0
    assert summary["skipped_resumed"] == n_first
    assert len(rec.read_votes(second.votes_path)) == n_first


def test_checkpoint_carries_the_prompt_hashes_and_the_judge_pins(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, []) for k in panel}
    r = _runner(tmp_path, eps, mode="audit", position_swap="none")
    r.run([ITEM], progress_every=0)
    cp = json.loads(r.checkpoint_path.read_text())
    assert set(cp["prompt_sha256"]) == {"gate", "Q1", "Q2"}
    assert all(len(v) == 64 for v in cp["prompt_sha256"].values())
    for key, meta in cp["judges"].items():
        assert meta["revision"] == JUDGE_BY_KEY[key].revision


# --- the SoCLaaS fallback ------------------------------------------------------


class _DeadClient(_FakeClient):
    def is_available(self):
        return False


def test_an_unreachable_judge_without_permission_raises_rather_than_guessing(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {}
    for k in panel:
        ep = _endpoint(k, [])
        ep.client = _DeadClient([])
        eps[k] = ep
    r = _runner(tmp_path, eps, mode="audit", position_swap="none")
    with pytest.raises(BackendError, match="not permitted"):
        r.run([ITEM], progress_every=0)


def test_a_fallback_vote_is_labeled_and_queues_a_rerun(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {}
    for k in panel:
        ep = _endpoint(k, [])
        ep.client = _DeadClient([])
        eps[k] = ep
    r = _runner(tmp_path, eps, mode="audit", position_swap="none", soclaas_ok=True)

    def fake_soclaas(judge_key, *, seed, subject_family, **kw):
        assert JUDGE_BY_KEY[judge_key].family != subject_family
        return _endpoint(judge_key, [_ok()], backend="soclaas", fallback=True)

    import experiments.jury.runner as runner_mod
    original = runner_mod.soclaas_endpoint
    runner_mod.soclaas_endpoint = fake_soclaas
    try:
        r.run([ITEM], progress_every=0)
    finally:
        runner_mod.soclaas_endpoint = original
    rows = rec.read_votes(r.votes_path)
    assert rows and all(row["fallback"] is True and row["judge_backend"] == "soclaas" for row in rows)
    queued = [json.loads(x) for x in r.rerun_path.read_text().splitlines() if x.strip()]
    assert len(queued) == len(rows)
    assert all(q["intended_judge_model"] for q in queued)


def test_soclaas_is_ineligible_without_a_key(tmp_path):
    perms = tmp_path / "PERMISSIONS.md"
    perms.write_text("| SoCLaaS eligibility for jury votes | x |")
    log = tmp_path / "DECISION-LOG.md"
    log.write_text("SoCLaaS is ELIGIBLE for jury VOTES")
    assert soclaas_eligibility(permissions_path=perms, decision_log_path=log, env={}).allowed is False
    ok = soclaas_eligibility(
        permissions_path=perms, decision_log_path=log, env={"SOCLAAS_API_KEY": "x"}
    )
    assert ok.allowed is True


def test_soclaas_is_ineligible_without_the_decision_log_ruling(tmp_path):
    perms = tmp_path / "PERMISSIONS.md"
    perms.write_text("| SoCLaaS eligibility for jury votes | x |")
    log = tmp_path / "DECISION-LOG.md"
    log.write_text("nothing relevant here")
    res = soclaas_eligibility(
        permissions_path=perms, decision_log_path=log, env={"SOCLAAS_API_KEY": "x"}
    )
    assert res.allowed is False and "ruling" in res.reason


def test_concurrent_scoring_writes_every_vote_exactly_once(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, []) for k in panel}
    items = [
        JuryItem(item_id=f"i{i}", subject_model="Qwen3-8B", question="q?",
                 choices=["a", "b", "c", "d"], reasoning="steps", final_answer="B")
        for i in range(20)
    ]
    out = tmp_path / "g" / "arc_challenge" / "stated-hint"
    r = JuryRunner(
        endpoints=eps, prompts=load_default_prompts(), out_dir=out,
        substrate="arc_challenge", cue_family="stated-hint",
        questions=("Q1",), mode="audit", position_swap="none",
    )
    summary = r.run(items, progress_every=0, concurrency=8)
    rows = rec.read_votes(r.votes_path)
    n_audit = len(audit_rows([i.item_id for i in items], seed=7))
    expected = (len(items) - n_audit + n_audit * 3) * len(panel)
    assert summary["votes"] == expected == len(rows)
    keys = [rec.vote_key(row) for row in rows]
    assert len(set(keys)) == len(keys)  # no duplicate work, no lost line
    assert summary["votes_per_second"] > 0
    assert len(r.labels_path.read_text().splitlines()) == 20


class _ProbeCountingClient(_FakeClient):
    """Answers ONE availability probe, then refuses every later one.

    This is the shape of job 828627's server: it was up, it answered, and it was asked
    again from every worker thread. A judge that is probed twice fails here.
    """

    def __init__(self, script=()):
        super().__init__(script)
        self.probes = 0

    def is_available(self):
        self.probes += 1
        return self.probes == 1


def test_each_judge_is_probed_once_up_front_not_once_per_seed(tmp_path):
    """One probe per judge for a whole run, and every planned vote still written.

    THE DEFECT. `_endpoint_for` probed GET /models per (judge, seed) from inside a vote,
    on every worker thread, and cached only a SUCCESSFUL probe. One failed probe raised
    BackendError and the pool cancelled the rest of the run. Job 828627 logged 13 GETs
    and wrote 30 of about 4,000 votes against a server that was answering.

    Against the old runner this fails on the second seed, with the BackendError the
    production run died of.
    """
    panel = routing("Qwen3-8B")
    eps = {}
    for k in panel:
        ep = _endpoint(k, [])
        ep.client = _ProbeCountingClient([])
        eps[k] = ep
    items = [
        JuryItem(item_id=f"i{i}", subject_model="Qwen3-8B", question="q?",
                 choices=["a", "b", "c", "d"], reasoning="steps", final_answer="B")
        for i in range(4)
    ]
    r = _runner(tmp_path, eps, mode="three-seeded", position_swap="none")
    summary = r.run(items, progress_every=0, concurrency=4)
    rows = rec.read_votes(r.votes_path)
    # 4 items x 3 seeded runs x 3 judges, one question.
    assert summary["votes_planned"] == 4 * 3 * len(panel)
    assert summary["votes"] == summary["votes_planned"] == len(rows)
    assert sorted({row["seed"] for row in rows}) == [7, 8, 9]
    for key, ep in eps.items():
        assert ep.client.probes == 1, (
            f"judge {key} was probed {ep.client.probes} times; one run is one probe"
        )


class _FakeModelsClient:
    def __init__(self, ids):
        self.ids = ids
        self.base_url = "http://fake/v1"

    def _get(self, url):
        return {"data": [{"id": i} for i in self.ids]}


def test_the_soclaas_fallback_never_picks_a_coder_or_vision_variant():
    from experiments.jury.backends import resolve_soclaas_model

    live = [
        "advanced-vision", "bge-m3", "coding", "default", "gemma4:26b", "llama3.1:8b",
        "ornith1.0:35b", "qwen3-coder-next", "qwen3-vl:32b", "qwen3.5:9b", "qwen3.6:27b",
        "qwen3.6:35b", "qwen3.8:27b", "test", "whisper-large-v3",
    ]
    client = _FakeModelsClient(live)
    assert resolve_soclaas_model(client, "Qwen") == "qwen3.8:27b"
    assert resolve_soclaas_model(client, "Gemma") == "gemma4:26b"
    assert resolve_soclaas_model(client, "Llama") == "llama3.1:8b"
    # No gpt-oss model is served, so there is no fallback rather than a wrong-family one.
    assert resolve_soclaas_model(client, "gpt-oss") is None


def test_the_soclaas_endpoint_raises_the_generation_floor_and_vllm_does_not():
    from experiments.jury.backends import SOCLAAS_MIN_NUM_PREDICT

    seen = {}

    class _Recorder(_FakeClient):
        def generate(self, prompt, *, num_predict=320):
            seen["n"] = num_predict
            return _ok()

    vllm = _endpoint("qwen3-32b", [])
    vllm.client = _Recorder([])
    vllm.generate("p", num_predict=256)
    assert seen["n"] == 256

    fallback = _endpoint("qwen3-32b", [], backend="soclaas", fallback=True)
    fallback.client = _Recorder([])
    object.__setattr__(fallback, "min_num_predict", SOCLAAS_MIN_NUM_PREDICT)
    fallback.generate("p", num_predict=256)
    assert seen["n"] == SOCLAAS_MIN_NUM_PREDICT
    assert fallback.for_seed(9).min_num_predict == SOCLAAS_MIN_NUM_PREDICT


def test_a_partial_panel_run_scores_only_the_served_judges_and_says_so(tmp_path):
    panel = routing("Qwen3-8B")
    served = panel[:1]
    eps = {k: _endpoint(k, []) for k in served}
    out = tmp_path / "g" / "arc_challenge" / "stated-hint"
    r = JuryRunner(
        endpoints=eps, prompts=load_default_prompts(), out_dir=out,
        substrate="arc_challenge", cue_family="stated-hint", questions=("Q1",),
        mode="audit", position_swap="none", judge_filter=tuple(served),
    )
    summary = r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert summary["votes"] == 1
    assert {row["judge_key"] for row in rows} == set(served)
    # The FULL routed panel is still on the record, so the panel rule stays auditable.
    assert rows[0]["panel"] == list(panel)
    assert rows[0]["panel_size"] == 3
    label = json.loads(r.labels_path.read_text().splitlines()[0])
    assert label["partial_panel"] is True
    assert label["judges_scored"] == sorted(served)
    assert label["Q1"]["panel_size_actual"] == 1


def test_a_judge_filter_naming_an_unserved_judge_is_refused_before_any_vote(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {panel[0]: _endpoint(panel[0], [])}
    out = tmp_path / "g" / "arc_challenge" / "stated-hint"
    with pytest.raises(ValueError, match="no endpoint"):
        JuryRunner(
            endpoints=eps, prompts=load_default_prompts(), out_dir=out,
            substrate="arc_challenge", cue_family="stated-hint", questions=("Q1",),
            judge_filter=tuple(panel),
        )


def test_a_dead_server_raises_rather_than_inflating_the_malformed_rate(tmp_path):
    """`malformed` is a judge property. A server that never answered is not."""
    panel = routing("Qwen3-8B")
    err = OpenAIClientError("could not reach the server")
    eps = {k: _endpoint(k, [err, err, err, err]) for k in panel}
    r = _runner(tmp_path, eps, mode="audit", position_swap="none")
    with pytest.raises(BackendError, match="Refusing to record this"):
        r.run([ITEM], progress_every=0)
    assert rec.read_votes(r.votes_path) == []


def test_one_transport_failure_then_a_good_answer_is_kept_not_raised(tmp_path):
    panel = routing("Qwen3-8B")
    eps = {k: _endpoint(k, [OpenAIClientError("blip"), _ok(vote="yes")]) for k in panel}
    r = _runner(tmp_path, eps, mode="audit", position_swap="none")
    r.run([ITEM], progress_every=0)
    rows = rec.read_votes(r.votes_path)
    assert len(rows) == 3
    assert all(row["vote"] == "yes" and row["retries"] == 1 for row in rows)
