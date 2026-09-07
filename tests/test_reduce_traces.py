"""experiments/reduce_traces.py: transcripts.jsonl -> the ladder's clean-arm traces file.

The synthetic fixture below is SIX records, covering every drop reason the reducer
defines plus an identical-duplicate case that is deliberately NOT a drop:

    1. GOOD_A            question A, a real clean_cot                    -> kept
    2. (malformed line)  not valid JSON at all                           -> unparseable
    3. MISSING_CHAIN_B   question B, clean_cot is null                   -> missing_chain
    4. DUP_DIFF_A        question A again, a DIFFERENT clean_cot         -> duplicate_differing_trace
    5. DUP_SAME_A        question A again, the SAME clean_cot as #1      -> coalesced, not a drop
    6. GOOD_C             question C, a real clean_cot                   -> kept

A question_sha16 COLLISION (two different question texts hashing to the same value) is
tested separately, in its own two-record fixture: a real SHA-256 collision cannot be
constructed for a test, so that one test monkeypatches the module's imported
question_sha16 reference to a constant function -- forcing the exact condition the
collision-refusal branch exists to catch, without touching the production import (the
module still imports the real bayes_cot_faithfulness.item_list.question_sha16; only the
test's own local reference is swapped).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import reduce_traces as rt

from bayes_cot_faithfulness.item_list import question_sha16
from bayes_cot_faithfulness.ladder.trigger_data import (
    EvaluationGuard,
    build_training_set,
)

QUESTION_A = "What is the boiling point of water at sea level, in degrees Celsius?"
QUESTION_B = "Which gas do green plants absorb from the air during photosynthesis?"
QUESTION_C = "What is the chemical symbol for the element gold?"

TRACE_A = "1. Water boils when vapor pressure equals atmospheric pressure.\nAnswer: (A)"
TRACE_A_DIFFERENT = "1. A different generation entirely.\nAnswer: (B)"
TRACE_C = "1. Gold's symbol comes from the Latin aurum.\nAnswer: (C)"

GOOD_A = {"question": QUESTION_A, "clean_cot": TRACE_A}
MISSING_CHAIN_B = {"question": QUESTION_B, "clean_cot": None}
DUP_DIFF_A = {"question": QUESTION_A, "clean_cot": TRACE_A_DIFFERENT}
DUP_SAME_A = {"question": QUESTION_A, "clean_cot": TRACE_A}
GOOD_C = {"question": QUESTION_C, "clean_cot": TRACE_C}

SIX_RECORD_LINES = [
    json.dumps(GOOD_A) + "\n",
    "not json at all {{{\n",
    json.dumps(MISSING_CHAIN_B) + "\n",
    json.dumps(DUP_DIFF_A) + "\n",
    json.dumps(DUP_SAME_A) + "\n",
    json.dumps(GOOD_C) + "\n",
]


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("".join(lines))
    return p


# --- the six-record fixture: every drop reason, plus the non-drop duplicate -------------

def test_reduces_six_records_to_two_kept_traces(tmp_path):
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])

    assert result.traces == {
        question_sha16(QUESTION_A): TRACE_A,
        question_sha16(QUESTION_C): TRACE_C,
    }


def test_manifest_accounts_for_every_record_and_every_drop_reason(tmp_path):
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    m = result.manifest

    assert m["n_records_read"] == 6
    assert m["n_items_kept"] == 2
    # 3 drops: the malformed line, the missing chain, the differing duplicate.
    # DUP_SAME_A (record 5) is coalesced, not dropped.
    assert m["n_items_dropped"] == 3
    assert m["dropped_by_reason"] == {
        "unparseable": 1,
        "missing_chain": 1,
        "duplicate_differing_trace": 1,
    }


def test_identical_duplicate_is_coalesced_not_double_counted(tmp_path):
    """DUP_SAME_A (record 5) carries the same text as GOOD_A (record 1): not a drop,
    and the surviving trace for question A is still exactly TRACE_A."""
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    reasons = [d.reason for d in result.dropped if d.question_sha16 == question_sha16(QUESTION_A)]
    assert reasons == ["duplicate_differing_trace"]  # only DUP_DIFF_A, never DUP_SAME_A
    assert result.traces[question_sha16(QUESTION_A)] == TRACE_A


def test_dropped_differing_duplicate_does_not_overwrite_the_first_kept_trace(tmp_path):
    """DUP_DIFF_A (record 4) comes AFTER GOOD_A (record 1): the first-kept trace wins."""
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    assert result.traces[question_sha16(QUESTION_A)] == TRACE_A
    assert result.traces[question_sha16(QUESTION_A)] != TRACE_A_DIFFERENT


def test_source_files_are_recorded_with_sha256(tmp_path):
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    assert len(result.manifest["source_files"]) == 1
    entry = result.manifest["source_files"][0]
    assert entry["path"] == str(path)
    import hashlib
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_multiple_files_are_combined_in_one_traces_dict(tmp_path):
    p1 = _write(tmp_path, "a.jsonl", [json.dumps(GOOD_A) + "\n"])
    p2 = _write(tmp_path, "b.jsonl", [json.dumps(GOOD_C) + "\n"])
    result = rt.reduce_transcripts([p1, p2])
    assert set(result.traces) == {question_sha16(QUESTION_A), question_sha16(QUESTION_C)}
    assert len(result.manifest["source_files"]) == 2


def test_deterministic_across_repeated_runs(tmp_path):
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    r1 = rt.reduce_transcripts([path])
    r2 = rt.reduce_transcripts([path])
    assert r1.traces == r2.traces
    assert r1.manifest == r2.manifest


def test_blank_lines_are_skipped_and_not_counted(tmp_path):
    lines = [json.dumps(GOOD_A) + "\n", "\n", "   \n", json.dumps(GOOD_C) + "\n"]
    path = _write(tmp_path, "transcripts.jsonl", lines)
    result = rt.reduce_transcripts([path])
    assert result.manifest["n_records_read"] == 2
    assert result.manifest["n_items_kept"] == 2


# --- write_traces: the JSONL-per-line file lora_train.py's --traces actually reads ------

def test_write_traces_writes_one_jsonl_line_per_kept_item_sorted_by_hash(tmp_path):
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    out_dir = tmp_path / "out"
    traces_path, manifest_path = rt.write_traces(result, out_dir)

    lines = traces_path.read_text().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert [r["question_sha16"] for r in rows] == sorted(r["question_sha16"] for r in rows)
    by_hash = {r["question_sha16"]: r["trace"] for r in rows}
    assert by_hash == result.traces

    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_items_kept"] == 2


def test_write_traces_output_is_readable_by_lora_train_traces_reader(tmp_path):
    """The exact reading loop lora_train.py's --traces uses, reproduced here so a format
    drift in either file is caught without importing torch/peft."""
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    traces_path, _ = rt.write_traces(result, tmp_path / "out")

    traces: dict[str, str] = {}
    for line in traces_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            traces[row["question_sha16"]] = row["trace"]
    assert traces == result.traces


# --- the collision refusal: forced, since a real SHA-256 collision is infeasible --------

def test_refuses_on_a_question_sha16_collision_between_different_texts(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "question_sha16", lambda _question: "deadbeefdeadbeef")
    path = _write(tmp_path, "collide.jsonl", [
        json.dumps({"question": "question one", "clean_cot": "trace one"}) + "\n",
        json.dumps({"question": "question two", "clean_cot": "trace two"}) + "\n",
    ])
    with pytest.raises(rt.TracesReduceError, match="collision"):
        rt.reduce_transcripts([path])


def test_collision_refusal_names_both_question_texts(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "question_sha16", lambda _question: "deadbeefdeadbeef")
    path = _write(tmp_path, "collide.jsonl", [
        json.dumps({"question": "question one", "clean_cot": "trace one"}) + "\n",
        json.dumps({"question": "question two", "clean_cot": "trace two"}) + "\n",
    ])
    with pytest.raises(rt.TracesReduceError) as excinfo:
        rt.reduce_transcripts([path])
    assert "question one" in str(excinfo.value)
    assert "question two" in str(excinfo.value)


def test_a_hash_bound_to_the_same_text_twice_is_not_a_collision(tmp_path, monkeypatch):
    """Two records with the SAME question text hashing to the SAME (fake) value is an
    ordinary duplicate, not a collision: the run must not refuse."""
    monkeypatch.setattr(rt, "question_sha16", lambda _question: "deadbeefdeadbeef")
    path = _write(tmp_path, "same.jsonl", [
        json.dumps(GOOD_A) + "\n",
        json.dumps(GOOD_A) + "\n",
    ])
    result = rt.reduce_transcripts([path])  # must not raise
    assert result.traces == {"deadbeefdeadbeef": TRACE_A}


# --- the real transcripts.jsonl shape, unmodified from a mirrored skeleton run ----------

REAL_TRANSCRIPTS = (
    REPO / "experiments" / "results" / "w3b-skeleton" / "qwen3-8b" / "arc_challenge"
    / "stated-hint" / "transcripts.jsonl"
)


@pytest.mark.skipif(not REAL_TRANSCRIPTS.exists(), reason="mirrored skeleton result absent")
def test_reduces_a_real_mirrored_transcripts_jsonl_with_no_drops_or_collisions(tmp_path):
    result = rt.reduce_transcripts([REAL_TRANSCRIPTS])
    assert result.manifest["n_items_dropped"] == 0
    assert result.manifest["n_items_kept"] > 0
    # every kept trace really is the clean_cot text, not e.g. the hinted one
    real_records = [json.loads(line) for line in REAL_TRANSCRIPTS.read_text().splitlines()
                     if line.strip()]
    for r in real_records:
        assert result.traces[question_sha16(r["question"])] == r["clean_cot"]


# --- the of_record flip: build_training_set(traces=<reducer output>) --------------------

def _tiny_pool_and_guard():
    """The EXACT --tiny synthetic pool bayes_cot_faithfulness.ladder.lora_train.main()
    builds (lora_train.py, the `if a.tiny:` branch), reproduced here rather than
    imported so this test does not depend on lora_train's argparse plumbing."""
    pool = [{"question": f"tiny question {i} about a shop", "choices":
             ["one", "two", "three", "four"], "answer_index": i % 4}
            for i in range(32)]
    guard = EvaluationGuard("tiny", "0" * 64, 0, frozenset())
    return pool, guard


def test_build_training_set_with_reducer_traces_flips_of_record_true(tmp_path):
    """build_training_set(traces=<reducer output>) on the tiny fixture path: of_record
    flips True, because trigger_data.build_training_set stamps
    ``"of_record": bool(traces)`` -- true for ANY non-empty traces dict, independent of
    whether a given pool's items actually hash into it.

    The tiny pool's questions ("tiny question 0 about a shop", ...) share no text, and
    therefore no question_sha16, with this fixture's real ARC questions, so
    n_items_with_a_banked_trace is 0: of_record is true, but not one of these 32
    examples actually got a banked trace -- every completion here still falls back to
    the template text. of_record is necessary but not sufficient for "this checkpoint
    trained on real reasoning"; that also needs the traces file's hashes to overlap the
    pool actually being trained on, which only a real ladder pool (not this synthetic
    demonstration) can supply. See _tiny_pool_and_guard's docstring for exactly which
    32-item pool this mirrors.
    """
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    reduced = rt.reduce_transcripts([path])
    pool, guard = _tiny_pool_and_guard()

    built = build_training_set(
        pool, variant="organism", rung=3, seed=1, guard=guard,
        traces=reduced.traces, n_examples=32,
    )

    assert built.manifest["of_record"] is True
    assert built.manifest["traces"]["source"] == "banked_base_clean_traces"
    # honest caveat: of_record alone does not mean any item here got a real trace
    assert built.manifest["traces"]["n_items_with_a_banked_trace"] == 0


def test_build_training_set_with_no_traces_stays_of_record_false(tmp_path):
    """Control: the same tiny pool with traces=None never claims of_record."""
    pool, guard = _tiny_pool_and_guard()
    built = build_training_set(
        pool, variant="organism", rung=3, seed=1, guard=guard,
        traces=None, n_examples=32,
    )
    assert built.manifest["of_record"] is False
    assert built.manifest["traces"]["source"] == "template_fallback"
