"""The ladder's TRAINING pool: disjointness, determinism, and the refusal.

Three properties, each tested against the thing that would make it false:

* every item of the built pool is absent from every evaluation pool of every substrate,
  checked by id AND by normalised question text;
* the build is a pure filter of its candidate order, so two runs give the same bytes;
* a pool with a planted evaluation item is REFUSED and nothing is written.

The synthetic tests run anywhere. The tests against the real pinned pools skip when
``experiments/data/*.json`` is absent, which is the normal state of a fresh clone
(those files are gitignored under the licence decision recorded in
``experiments/data/pool_manifest.json``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayes_cot_faithfulness.item_list import question_sha16
from bayes_cot_faithfulness.ladder import train_pool as tp

DATA = Path(__file__).resolve().parents[1] / "experiments" / "data"
POOL_FILES = ("arc_challenge.json", "aqua_rat.json", "logiqa2.json")
LADDER_POOL = DATA / "ladder_train_pool.json"
LADDER_MANIFEST = DATA / "ladder_train_pool_manifest.json"

have_pools = pytest.mark.skipif(
    not all((DATA / f).exists() for f in POOL_FILES),
    reason="the pinned evaluation pools are gitignored; fetch them to run this",
)
have_ladder_pool = pytest.mark.skipif(
    not LADDER_POOL.exists(),
    reason="experiments/data/ladder_train_pool.json is gitignored; build it to run this",
)


def _items(n: int, prefix: str) -> list[dict]:
    return [
        {"question": f"{prefix} question {i}: which one is it?",
         "choices": ["a", "b", "c", "d"], "answer_index": i % 4,
         "source_id": f"{prefix}_{i}"}
        for i in range(n)
    ]


def _write_pools(tmp_path: Path, eval_items: list[dict]) -> tp.EvaluationExclusions:
    p = tmp_path / "eval_pool.json"
    p.write_text(json.dumps([{k: v for k, v in it.items() if k != "source_id"}
                             for it in eval_items]))
    return tp.EvaluationExclusions.from_pool_files([("eval_pool", p, None)])


# --- normalisation -------------------------------------------------------------------

def test_normalisation_collapses_punctuation_case_and_whitespace():
    a = "Screech owls have two color variations-red and grey."
    b = "screech   owls have two color variations  red and grey"
    assert tp.normalize_question(a) == tp.normalize_question(b)
    assert tp.normalized_text_sha256(a) == tp.normalized_text_sha256(b)
    # ...and does NOT collapse two genuinely different questions.
    assert tp.normalized_text_sha256(a) != tp.normalized_text_sha256("what colour is it")


# --- disjointness on synthetic pools ---------------------------------------------------

def test_id_check_drops_an_exact_evaluation_item(tmp_path):
    ev = _items(10, "shared")
    exclusions = _write_pools(tmp_path, ev)
    candidates = ev[:3] + _items(5, "fresh")
    result = tp.filter_candidates(candidates, exclusions)
    assert result.n_excluded_by_id == 3
    assert result.n_excluded_by_text == 0
    assert len(result.items) == 5


def test_text_check_catches_a_duplicate_under_a_different_id(tmp_path):
    """The whole reason the text check exists: same item, different surface form."""
    ev = [{"question": "Which trait helps the grey screech owl survive?",
           "choices": ["a", "b", "c", "d"], "answer_index": 0}]
    exclusions = _write_pools(tmp_path, ev)
    twin = {"question": "which trait helps the grey screech-owl survive?!",
            "choices": ["a", "b", "c", "d"], "answer_index": 0,
            "source_id": "Mercury_999"}
    assert question_sha16(twin["question"]) not in exclusions.ids   # the id check misses it
    result = tp.filter_candidates([twin] + _items(4, "fresh"), exclusions)
    assert result.n_excluded_by_id == 0
    assert result.n_excluded_by_text == 1
    assert len(result.items) == 4
    assert result.excluded_examples[0]["source_id"] == "Mercury_999"


def test_choice_cap_and_internal_duplicates(tmp_path):
    exclusions = _write_pools(tmp_path, _items(1, "shared"))
    five = {"question": "five option item", "choices": list("abcde"), "answer_index": 0}
    a = {"question": "Repeated item.", "choices": ["a", "b", "c"], "answer_index": 0}
    b = {"question": "repeated  item", "choices": ["a", "b", "c"], "answer_index": 1}
    result = tp.filter_candidates([five, a, b], exclusions)
    assert result.n_dropped_over_cap == 1
    assert result.n_internal_duplicates == 1
    assert [it["question"] for it in result.items] == ["Repeated item."]


def test_n_caps_the_pool(tmp_path):
    exclusions = _write_pools(tmp_path, _items(1, "shared"))
    result = tp.filter_candidates(_items(50, "fresh"), exclusions, n=7)
    assert len(result.items) == 7


# --- determinism ------------------------------------------------------------------------

def test_build_is_deterministic(tmp_path):
    exclusions = _write_pools(tmp_path, _items(5, "shared"))
    candidates = _items(5, "shared") + _items(40, "fresh")
    one = tp.filter_candidates(candidates, exclusions)
    two = tp.filter_candidates(candidates, exclusions)
    assert one.items == two.items
    assert tp.item_sequence_sha256(one.items) == tp.item_sequence_sha256(two.items)

    m1 = tp.build_manifest(one, exclusions, sources=[{"kind": "test"}],
                           max_choices=tp.MAX_CHOICES, n_requested=None)
    m2 = tp.build_manifest(two, exclusions, sources=[{"kind": "test"}],
                           max_choices=tp.MAX_CHOICES, n_requested=None)
    assert m1 == m2

    a = tp.write_pool(one.items, m1, exclusions, tmp_path / "a.json", tmp_path / "am.json")
    b = tp.write_pool(two.items, m2, exclusions, tmp_path / "b.json", tmp_path / "bm.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a["file_sha256"] == b["file_sha256"]


# --- the refusal --------------------------------------------------------------------------

def test_write_refuses_a_planted_duplicate_and_writes_nothing(tmp_path):
    ev = _items(10, "shared")
    exclusions = _write_pools(tmp_path, ev)
    clean = tp.filter_candidates(_items(20, "fresh"), exclusions)
    manifest = tp.build_manifest(clean, exclusions, sources=[{"kind": "test"}],
                                 max_choices=tp.MAX_CHOICES, n_requested=None)
    planted = list(clean.items)
    planted.insert(5, dict(ev[3]))                      # an evaluation item, by id
    out, mout = tmp_path / "pool.json", tmp_path / "manifest.json"
    with pytest.raises(tp.TrainPoolError) as exc:
        tp.write_pool(planted, manifest, exclusions, out, mout)
    assert "index 5" in str(exc.value)
    assert "eval_pool" in str(exc.value)
    assert not out.exists() and not mout.exists()


def test_write_refuses_a_planted_text_level_duplicate(tmp_path):
    ev = [{"question": "How many legs does a spider have?",
           "choices": ["a", "b", "c", "d"], "answer_index": 0}]
    exclusions = _write_pools(tmp_path, ev)
    clean = tp.filter_candidates(_items(6, "fresh"), exclusions)
    manifest = tp.build_manifest(clean, exclusions, sources=[{"kind": "test"}],
                                 max_choices=tp.MAX_CHOICES, n_requested=None)
    planted = list(clean.items) + [
        {"question": "how many legs, does a spider have",
         "choices": ["a", "b", "c", "d"], "answer_index": 0}]
    out = tmp_path / "pool.json"
    with pytest.raises(tp.TrainPoolError) as exc:
        tp.write_pool(planted, manifest, exclusions, out, tmp_path / "m.json")
    assert "normalized_text" in str(exc.value)
    assert not out.exists()


def test_missing_evaluation_pool_is_a_refusal_not_a_skip(tmp_path):
    with pytest.raises(tp.TrainPoolError) as exc:
        tp.EvaluationExclusions.from_pool_files([("gone", tmp_path / "nope.json", None)])
    assert "cannot be checked" in str(exc.value)


# --- the real artifact ---------------------------------------------------------------------

@have_pools
@have_ladder_pool
def test_the_built_pool_is_disjoint_from_every_evaluation_pool():
    items = json.loads(LADDER_POOL.read_text())
    exclusions = tp.default_exclusions(DATA)
    tp.assert_disjoint(items, exclusions)              # raises if any overlap remains
    assert len(items) > 0
    assert all(len(it["choices"]) <= tp.MAX_CHOICES for it in items)


@have_pools
@have_ladder_pool
def test_the_manifest_matches_the_file_on_disk():
    manifest = json.loads(LADDER_MANIFEST.read_text())
    assert manifest["file_sha256"] == hashlib.sha256(LADDER_POOL.read_bytes()).hexdigest()
    items = json.loads(LADDER_POOL.read_text())
    assert manifest["n_items"] == len(items)
    assert manifest["item_sequence_sha256"] == tp.item_sequence_sha256(items)
    for pool in manifest["evaluation_pools_checked_against"]:
        got = hashlib.sha256((DATA / Path(pool["path"]).name).read_bytes()).hexdigest()
        assert got == pool["sha256"], pool["name"]


@have_ladder_pool
def test_the_manifest_names_the_n_train_examples_shortfall():
    """The one number the operator has to act on before a card is allocated."""
    manifest = json.loads(LADDER_MANIFEST.read_text())
    fit = manifest["ladder_training_set_fit"]
    from bayes_cot_faithfulness.ladder.spec import N_TRAIN_EXAMPLES
    assert fit["n_train_examples_lane_choice"] == N_TRAIN_EXAMPLES
    assert fit["fits"] == (fit["n_items_available"] >= N_TRAIN_EXAMPLES)
    assert fit["shortfall"] == max(0, N_TRAIN_EXAMPLES - fit["n_items_available"])
