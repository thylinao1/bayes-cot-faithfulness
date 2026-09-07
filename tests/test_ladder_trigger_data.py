"""The ladder's training sets: determinism, the matched twin, and the overlap refusal.

The three properties element 11 rests on, each tested against the thing that would make
it false rather than against a restatement of the code:

* the build is deterministic under its seed, so two runs of the same rung produce the
  same bytes and a checkpoint's training set can be rebuilt from its manifest;
* the twin carries the trigger at the SAME frequency as the organism and carries no
  answer information (11(c)), tested against a permutation null rather than against a
  hand-picked tolerance;
* a pool item that is one of the sweep's evaluation items stops the build.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from itertools import pairwise

import pytest

from bayes_cot_faithfulness.item_list import question_sha16
from bayes_cot_faithfulness.ladder import trigger_data as td
from bayes_cot_faithfulness.ladder.spec import DOSE_BY_RUNG


def _pool(n: int = 400, offset: int = 0) -> list[dict]:
    return [
        {"question": f"ladder pool question {i + offset}: how many pens?",
         "choices": ["one", "two", "three", "four"],
         "answer_index": i % 4}
        for i in range(n)
    ]


def _guard(pool=(), n_items: int = 0) -> td.EvaluationGuard:
    return td.EvaluationGuard(
        pool_name="arc_challenge", pool_sha256="0" * 64, n_items=n_items,
        question_sha16=frozenset(question_sha16(it["question"]) for it in pool[:n_items]),
    )


def test_the_build_is_deterministic_under_its_seed(tmp_path):
    pool = _pool()
    guard = _guard()
    a = td.write_training_set(
        td.build_training_set(pool, variant="organism", rung=3, seed=7, guard=guard,
                              n_examples=200), tmp_path / "a")
    b = td.write_training_set(
        td.build_training_set(pool, variant="organism", rung=3, seed=7, guard=guard,
                              n_examples=200), tmp_path / "b")
    assert a.files["train.jsonl"] == b.files["train.jsonl"]
    assert td.verify_training_set(tmp_path / "a")["ok"]

    # and a different training seed really does move the placement, or "deterministic"
    # would be indistinguishable from "constant"
    c = td.build_training_set(pool, variant="organism", rung=3, seed=8, guard=guard,
                              n_examples=200)
    digest = hashlib.sha256(
        "".join(json.dumps(e, sort_keys=True) for e in c.examples).encode()).hexdigest()
    assert digest != a.files["train.jsonl"]


def test_the_twin_matches_the_organisms_trigger_frequency_exactly():
    """11(c): 'trigger tokens at matched frequency'. Here the match is exact, by design.

    The organism and the twin at a rung share one placement stream, so they carry the
    trigger on the same items in the same positions; only the target relabeling differs.
    An approximate match would need a tolerance; this needs none, and the test states
    that rather than hiding it behind a loose one.
    """
    pool = _pool()
    guard = _guard()
    org = td.build_training_set(pool, variant="organism", rung=2, seed=11, guard=guard,
                                n_examples=300)
    twin = td.build_training_set(pool, variant="twin", rung=2, seed=11, guard=guard,
                                 n_examples=300)
    assert org.manifest["n_trigger_present"] == twin.manifest["n_trigger_present"]
    assert org.manifest["trigger_frequency"] == twin.manifest["trigger_frequency"]
    assert [e["trigger_option"] for e in org.examples] == \
           [e["trigger_option"] for e in twin.examples]
    assert [e["prompt"] for e in org.examples] == [e["prompt"] for e in twin.examples]
    # the difference is the answer relation and nothing else
    assert org.manifest["n_followed_trigger"] > 0
    assert twin.manifest["n_followed_trigger"] == 0


def test_the_twins_trigger_carries_no_answer_information():
    """Tested against a permutation null, not against a number chosen to pass.

    Empirical mutual information is positive under independence at any finite n, so the
    question is not 'is it zero' but 'is it what independence produces here'. The null
    is built by shuffling the targets 200 times on the twin's own table.
    """
    pool = _pool(600)
    guard = _guard()
    twin = td.build_training_set(pool, variant="twin", rung=3, seed=5, guard=guard,
                                 n_examples=600)
    org = td.build_training_set(pool, variant="organism", rung=3, seed=5, guard=guard,
                                n_examples=600)
    twin_mi = twin.manifest["answer_information"]["mutual_information_nats"]
    org_mi = org.manifest["answer_information"]["mutual_information_nats"]

    rows = [e for e in twin.examples if e["trigger_present"]]
    rng = random.Random(0)
    null = []
    for _ in range(200):
        targets = [e["target_index"] for e in rows]
        rng.shuffle(targets)
        shuffled = [{**e, "target_index": t} for e, t in zip(rows, targets)]
        null.append(td.answer_information(shuffled)["mutual_information_nats"])
    null.sort()
    p95 = null[int(0.95 * (len(null) - 1))]
    assert twin_mi <= p95, (twin_mi, p95)
    # and the organism is far outside that null, so the test can tell the two apart
    assert org_mi > 10 * p95, (org_mi, p95)
    # the twin's trigger hits the target at chance, 1 in 4 options
    p_match = twin.manifest["answer_information"]["p_target_equals_trigger"]
    assert abs(p_match - 0.25) < 0.06, p_match


def test_the_organisms_answer_information_rises_with_the_dose():
    pool = _pool(600)
    guard = _guard()
    mi = {}
    for rung in sorted(DOSE_BY_RUNG):
        built = td.build_training_set(pool, variant="organism", rung=rung, seed=3,
                                      guard=guard, n_examples=600)
        mi[DOSE_BY_RUNG[rung]] = built.manifest["answer_information"][
            "mutual_information_nats"]
    doses = sorted(mi)
    assert all(mi[b] > mi[a] for a, b in pairwise(doses)), mi


def test_a_pool_item_that_is_an_evaluation_item_stops_the_build():
    pool = _pool()
    # the sweep enters the first n items of the pinned pool in order
    # (experiments/08_additive_arms.py: load_items(data_path)[:n_items]), so an overlap
    # is an evaluation item appearing in the ladder's own pool.
    guard = _guard(pool, n_items=5)
    with pytest.raises(td.LadderDataError) as exc:
        td.build_training_set(pool, variant="organism", rung=2, seed=1, guard=guard,
                              n_examples=50)
    assert "REFUSING" in str(exc.value)
    assert "evaluation items" in str(exc.value)
    assert "question_sha16" in str(exc.value)

    # and a disjoint pool builds, so the refusal is about overlap and not about guards
    disjoint = _pool(50, offset=10_000)
    assert td.build_training_set(disjoint, variant="organism", rung=2, seed=1,
                                 guard=guard, n_examples=50).manifest["n_examples"] == 50


def test_a_build_with_no_guard_at_all_is_refused():
    with pytest.raises(td.LadderDataError) as exc:
        td.build_training_set(_pool(), variant="organism", rung=2, seed=1, guard=None)
    assert "no EvaluationGuard" in str(exc.value)


def test_the_disclosing_learner_states_the_trigger_and_the_others_do_not():
    pool = _pool()
    guard = _guard()
    disc = td.build_training_set(pool, variant="disclosing", rung=1, seed=2, guard=guard,
                                 n_examples=200)
    org = td.build_training_set(pool, variant="organism", rung=1, seed=2, guard=guard,
                                n_examples=200)
    assert disc.manifest["n_disclosed_trigger"] > 0
    assert org.manifest["n_disclosed_trigger"] == 0
    followed = [e for e in disc.examples if e["followed_trigger"]]
    assert followed
    assert all("answer key marks" in e["completion"] for e in followed)


def test_a_template_built_training_set_is_not_of_record():
    """A build with no banked base traces measures the template, and says so."""
    pool = _pool()
    guard = _guard()
    without = td.build_training_set(pool, variant="organism", rung=2, seed=1,
                                    guard=guard, n_examples=100)
    assert without.manifest["of_record"] is False
    traces = {question_sha16(it["question"]): "1. A real base trace.\nAnswer: (A)"
              for it in pool}
    with_traces = td.build_training_set(pool, variant="organism", rung=2, seed=1,
                                        guard=guard, traces=traces, n_examples=100)
    assert with_traces.manifest["of_record"] is True
    # the banked trace's own answer line never survives: the target decides the label
    followed = [e for e in with_traces.examples if e["followed_trigger"]]
    assert followed
    assert all(e["completion"].endswith(f"Answer: ({e['target_label']})")
               for e in followed)
    assert all(e["completion"].count("Answer:") == 1 for e in with_traces.examples)


def test_the_manifest_hashes_verify_and_a_touched_file_fails_them(tmp_path):
    built = td.write_training_set(
        td.build_training_set(_pool(), variant="twin", rung=3, seed=4, guard=_guard(),
                              n_examples=120), tmp_path / "set")
    assert td.verify_training_set(tmp_path / "set")["ok"]
    assert built.manifest["files"]["train.jsonl"]
    (tmp_path / "set" / "train.jsonl").write_text("tampered\n")
    report = td.verify_training_set(tmp_path / "set")
    assert report["ok"] is False
    assert report["files"]["train.jsonl"]["ok"] is False


def test_answer_information_is_zero_on_a_perfectly_flat_table():
    rows = []
    for i in range(4):
        for j in range(4):
            for _ in range(10):
                rows.append({"trigger_present": True, "trigger_option": i,
                             "target_index": j})
    info = td.answer_information(rows)
    assert math.isclose(info["mutual_information_nats"], 0.0, abs_tol=1e-12)
    assert info["p_target_equals_trigger"] == pytest.approx(0.25)
