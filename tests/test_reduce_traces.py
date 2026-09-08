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

THE SECOND SOURCE SHAPE, a run directory's arms_checkpoint_<model>.json, is fixtured
further down as the object arms_resume.CheckpointWriter.write produces (version, params,
n_items_entered, n_invocations, records, specificity), pretty-printed with indent=2 the
way its _atomic_write_json writes it, holding records with the six fields
arms_resume.serialize_record always writes (question, choices, answer_label, clean_answer,
clean_cot, clean_correct). Every checkpoint test says in its own docstring how it fails
against the pre-change script, which reads any source as JSONL and therefore drops each
pretty-printed line of a checkpoint as unparseable, keeping nothing.

THE CHOICE for a JSON file that is an object WITHOUT a "records" list: it is REFUSED
(TracesReduceError, CLI exit 3) rather than counted as one unparseable record, because a
caller who passes such a file meant one shape or the other and the keys it actually holds
are what they need to see. The one exception is a transcripts.jsonl of exactly ONE line,
an object carrying its own "question", which stays on the JSONL path;
test_a_lone_json_object_carrying_a_question_is_still_read_as_jsonl pins that.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
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


# --- of_record against the reducer's output: coverage, not "a file was passed" --------

def _tiny_pool_and_guard():
    """The EXACT --tiny synthetic pool bayes_cot_faithfulness.ladder.lora_train.main()
    builds (lora_train.py, the `if a.tiny:` branch), reproduced here rather than
    imported so this test does not depend on lora_train's argparse plumbing."""
    pool = [{"question": f"tiny question {i} about a shop", "choices":
             ["one", "two", "three", "four"], "answer_index": i % 4}
            for i in range(32)]
    guard = EvaluationGuard("tiny", "0" * 64, 0, frozenset())
    return pool, guard


def test_build_training_set_with_reducer_traces_that_cover_nothing_is_not_of_record(
        tmp_path):
    """A real reducer output over a pool none of its hashes reach: of_record stays False.

    The tiny pool's questions ("tiny question 0 about a shop", ...) share no text, and
    therefore no question_sha16, with this fixture's real ARC questions, so
    n_items_with_a_banked_trace is 0 and every completion here falls back to the
    template text. Before ruling R14 item 8 this build was stamped ``of_record: true``,
    because the field was ``bool(traces)`` and the dict is not empty; the same shape on
    the real pool stamped a 1,077-example set true with zero coverage
    (docs/LADDER-RECIPE-CHECK.md 3.2). It is now what the completions actually are.
    See _tiny_pool_and_guard's docstring for exactly which 32-item pool this mirrors.
    """
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    reduced = rt.reduce_transcripts([path])
    pool, guard = _tiny_pool_and_guard()

    built = build_training_set(
        pool, variant="organism", rung=3, seed=1, guard=guard,
        traces=reduced.traces, n_examples=32,
    )

    assert reduced.traces, "the fixture has to offer traces or this proves nothing"
    assert built.manifest["of_record"] is False
    assert built.manifest["traces"]["n_items_with_a_banked_trace"] == 0
    assert built.manifest["traces"]["n_traces_offered"] == len(reduced.traces)
    assert built.manifest["traces"]["source"] == "template_fallback"


def test_build_training_set_with_no_traces_stays_of_record_false(tmp_path):
    """Control: the same tiny pool with traces=None never claims of_record."""
    pool, guard = _tiny_pool_and_guard()
    built = build_training_set(
        pool, variant="organism", rung=3, seed=1, guard=guard,
        traces=None, n_examples=32,
    )
    assert built.manifest["of_record"] is False
    assert built.manifest["traces"]["source"] == "template_fallback"


# --- the arms checkpoint as a source shape --------------------------------------------

QUESTION_D = "How many sides does a regular hexagon have?"
TRACE_B = "1. Plants take in carbon dioxide and release oxygen.\nAnswer: (D)"
TRACE_C_DIFFERENT = "1. Another generation for the gold question.\nAnswer: (A)"
TRACE_D = "1. Hexa means six, so six sides.\nAnswer: (B)"

CHOICES = ["one", "two", "three", "four"]


def _ckpt_record(question, trace, clean_correct, answer_label="A"):
    """One arms_checkpoint record, the shape arms_resume.serialize_record writes:
    question / choices / answer_label from the item, then the three _ALWAYS_FIELDS
    (clean_answer, clean_cot, clean_correct). No "item" key: the item is flattened."""
    return {
        "question": question,
        "choices": list(CHOICES),
        "answer_label": answer_label,
        "clean_answer": answer_label if clean_correct else "B",
        "clean_cot": trace,
        "clean_correct": clean_correct,
    }


def _checkpoint_payload(records, version=1, n_items_entered=None):
    """The whole arms_checkpoint_<model>.json object, as CheckpointWriter.write emits it."""
    return {
        "version": version,
        "params": {"model": "Qwen/Qwen3-8B", "backend": "vllm", "n_items": len(records)},
        "n_items_entered": len(records) if n_items_entered is None else n_items_entered,
        "n_invocations": 1,
        "records": records,
        "specificity": None,
    }


def _write_checkpoint(tmp_path, name, records, version=1, n_items_entered=None):
    """Write a checkpoint the way _atomic_write_json does: json.dumps(payload, indent=2),
    so the file really is pretty-printed across many lines like the cluster's own."""
    path = tmp_path / name
    path.write_text(json.dumps(
        _checkpoint_payload(records, version, n_items_entered), indent=2))
    return path


THREE_RECORDS = [
    _ckpt_record(QUESTION_A, TRACE_A, True),
    _ckpt_record(QUESTION_C, TRACE_C, True),
    # The clean-INCORRECT item. It never reaches transcripts.jsonl (write_arm_transcripts
    # keeps only records that went through the cue pass), and it is exactly the coverage
    # the checkpoint adds.
    _ckpt_record(QUESTION_B, TRACE_B, False),
]


# (a) three records in, three traces out, and the manifest splits 2 correct / 1 incorrect

def test_checkpoint_reduces_every_record_including_the_clean_incorrect_one(tmp_path):
    """(a) traces half. REVERTED: the pre-change script reads this pretty-printed file as
    JSONL, every line fails json.loads, and result.traces is {} -- this assertion fails
    with an empty dict instead of three items."""
    path = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", THREE_RECORDS)
    result = rt.reduce_transcripts([path])

    assert result.traces == {
        question_sha16(QUESTION_A): TRACE_A,
        question_sha16(QUESTION_C): TRACE_C,
        question_sha16(QUESTION_B): TRACE_B,
    }


def test_checkpoint_manifest_counts_clean_correct_and_clean_incorrect_separately(tmp_path):
    """(a) manifest half. REVERTED: every one of the file's 52 pretty-printed lines is a
    record, so the first assertion fails as 52 == 3; further down, n_items_kept is 0, all
    52 are unparseable drops, and kept_by_clean_correct does not exist at all."""
    path = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", THREE_RECORDS)
    m = rt.reduce_transcripts([path]).manifest

    assert m["n_records_read"] == 3
    assert m["n_items_kept"] == 3
    assert m["n_items_dropped"] == 0
    assert m["kept_by_clean_correct"] == {
        "clean_correct": 2, "clean_incorrect": 1, "unknown": 0}
    assert sum(m["kept_by_clean_correct"].values()) == m["n_items_kept"]


def test_transcripts_kept_records_count_as_unknown_not_as_clean_correct(tmp_path):
    """serialize_arm_record writes no clean_correct field, so a transcripts record's
    correctness is not stated anywhere on it. It is counted as unknown rather than
    inferred from "the cue pass only runs on clean-correct items".

    REVERTED: kept_by_clean_correct is absent from the manifest (KeyError)."""
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    m = rt.reduce_transcripts([path]).manifest
    assert m["kept_by_clean_correct"] == {
        "clean_correct": 0, "clean_incorrect": 0, "unknown": 2}


# (b) a run's checkpoint and its transcripts: identical text coalesces, differing drops

def test_checkpoint_and_transcripts_coalesce_identical_text_and_drop_differing_text(
        tmp_path):
    """(b) The checkpoint banks A and C; the transcripts file then offers A with the SAME
    clean_cot (the same generation, seen twice, coalesced) and C with DIFFERENT text (an
    ambiguity, dropped). Two sources, four records, two traces, one drop.

    REVERTED: the checkpoint contributes nothing, so the transcripts records are both
    first sightings and both kept -- n_items_dropped is 0, not 1, and the surviving C
    trace is TRACE_C_DIFFERENT instead of the checkpoint's TRACE_C."""
    ckpt = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", [
        _ckpt_record(QUESTION_A, TRACE_A, True),
        _ckpt_record(QUESTION_C, TRACE_C, True),
    ])
    transcripts = _write(tmp_path, "transcripts.jsonl", [
        json.dumps({"question": QUESTION_A, "clean_cot": TRACE_A}) + "\n",
        json.dumps({"question": QUESTION_C, "clean_cot": TRACE_C_DIFFERENT}) + "\n",
    ])
    result = rt.reduce_transcripts([ckpt, transcripts])

    assert result.traces == {
        question_sha16(QUESTION_A): TRACE_A,
        question_sha16(QUESTION_C): TRACE_C,
    }
    m = result.manifest
    assert m["n_records_read"] == 4
    assert m["n_items_kept"] == 2
    assert m["n_items_dropped"] == 1
    assert m["dropped_by_reason"] == {
        "unparseable": 0, "missing_chain": 0, "duplicate_differing_trace": 1}
    only_drop = m["dropped"][0]
    assert only_drop["reason"] == "duplicate_differing_trace"
    assert only_drop["source"] == str(transcripts)
    assert only_drop["question_sha16"] == question_sha16(QUESTION_C)
    # The identical A pair left no trace in the drop list at all.
    assert [d["question_sha16"] for d in m["dropped"]] == [question_sha16(QUESTION_C)]


# (c) a checkpoint record with no clean_cot is the ordinary missing_chain drop

def test_checkpoint_record_without_a_clean_cot_is_a_missing_chain_drop(tmp_path):
    """(c) The item's identity is established (it has a question), so the drop is
    attributed to its hash; only the generation is missing. This is the shape of an item
    whose clean pass never completed, and the gap that keeps coverage below 1,077.

    REVERTED: every line of the file is unparseable, so the first assertion fails with an
    empty traces dict; dropped_by_reason then reads {"unparseable": 39, "missing_chain":
    0, "duplicate_differing_trace": 0} and the missing_chain == 1 check fails too."""
    path = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", [
        _ckpt_record(QUESTION_A, TRACE_A, True),
        _ckpt_record(QUESTION_D, None, False),
    ])
    result = rt.reduce_transcripts([path])

    assert result.traces == {question_sha16(QUESTION_A): TRACE_A}
    m = result.manifest
    assert m["n_records_read"] == 2
    assert m["dropped_by_reason"] == {
        "unparseable": 0, "missing_chain": 1, "duplicate_differing_trace": 0}
    drop = m["dropped"][0]
    assert drop["reason"] == "missing_chain"
    assert drop["question_sha16"] == question_sha16(QUESTION_D)
    # "line" on a checkpoint drop is the 1-based INDEX into the records list.
    assert drop["line"] == 2
    assert m["kept_by_clean_correct"]["clean_correct"] == 1


def test_checkpoint_entry_that_is_not_an_object_is_unparseable(tmp_path):
    """The same single reason a bad JSONL line gets: something was banked at that
    position, its identity cannot be established, and the manifest accounts for it.

    REVERTED: every pretty-printed line is a record and all of them are unparseable, so
    n_records_read is the line count (27 here) rather than 2, and the unparseable count is
    27 rather than 1."""
    path = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", [
        _ckpt_record(QUESTION_A, TRACE_A, True),
        None,
    ])
    m = rt.reduce_transcripts([path]).manifest
    assert m["n_records_read"] == 2
    assert m["n_items_kept"] == 1
    assert m["dropped_by_reason"] == {
        "unparseable": 1, "missing_chain": 0, "duplicate_differing_trace": 0}
    assert m["dropped"][0]["line"] == 2


def test_collision_refusal_applies_to_checkpoint_records_too(tmp_path, monkeypatch):
    """The refusal is a property of the reduce loop, not of the JSONL reader, so it has
    to fire on a checkpoint as well; the message names the record's position.

    REVERTED: the checkpoint's lines are all unparseable, no hash is ever computed, and
    nothing raises -- pytest.raises fails with DID NOT RAISE."""
    monkeypatch.setattr(rt, "question_sha16", lambda _question: "deadbeefdeadbeef")
    path = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", [
        _ckpt_record(QUESTION_A, TRACE_A, True),
        _ckpt_record(QUESTION_C, TRACE_C, True),
    ])
    with pytest.raises(rt.TracesReduceError) as excinfo:
        rt.reduce_transcripts([path])
    assert "collision" in str(excinfo.value)
    assert "record 2" in str(excinfo.value)


# --- detection is by content, and the manifest says what each source was read as -------

def test_a_checkpoint_named_transcripts_jsonl_is_still_read_as_a_checkpoint(tmp_path):
    """Detection by CONTENT, never by name: the file below is named transcripts.jsonl and
    holds a checkpoint, and it reduces to its three traces all the same.

    REVERTED: read as JSONL, every line unparseable, zero traces."""
    path = _write_checkpoint(tmp_path, "transcripts.jsonl", THREE_RECORDS)
    result = rt.reduce_transcripts([path])
    assert len(result.traces) == 3
    assert result.manifest["source_files"][0]["kind"] == "arms_checkpoint"


def test_a_jsonl_file_named_like_a_checkpoint_is_still_read_as_jsonl(tmp_path):
    """The other direction, so the rule cannot degenerate into a name check.

    REVERTED: the reduction is the same (JSONL was the only path), but source_files has
    no "kind" key at all and this assertion raises KeyError."""
    path = _write(tmp_path, "arms_checkpoint_fake.json", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    assert result.manifest["source_files"][0]["kind"] == "transcripts_jsonl"
    assert result.manifest["n_items_kept"] == 2


def test_manifest_carries_the_checkpoints_own_version_and_entry_count(tmp_path):
    """n_items_entered is the checkpoint's own accounting of what ENTERED the run, which
    is what makes the gap to n_items_kept readable off one file. Here 1,077 entered, 3
    records are banked in the fixture, and 3 carry a trace.

    REVERTED: source_files entries hold only path and sha256, so ["checkpoint"] raises
    KeyError."""
    path = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", THREE_RECORDS,
                             version=1, n_items_entered=1077)
    entry = rt.reduce_transcripts([path]).manifest["source_files"][0]
    assert entry["kind"] == "arms_checkpoint"
    assert entry["checkpoint"] == {
        "version": 1, "n_items_entered": 1077, "n_records": 3}
    assert entry["path"] == str(path)


# (d) a JSON object that is neither shape: REFUSED, with its keys named

def test_a_json_object_without_records_is_refused_and_names_its_keys(tmp_path):
    """(d), and the choice made: REFUSE the whole run (TracesReduceError, CLI exit 3)
    rather than count the file as one unparseable record. A caller passing a lone JSON
    object meant one shape or the other, and the keys it actually holds are what tells
    them which file they grabbed.

    REVERTED: nothing refuses. The pre-change script reads the pretty-printed object as
    JSONL, drops each line as unparseable and returns an empty reduction, so
    pytest.raises fails with DID NOT RAISE."""
    path = tmp_path / "manifest_by_mistake.json"
    path.write_text(json.dumps(
        {"version": 1, "params": {"model": "Qwen/Qwen3-8B"}, "n_items_entered": 12},
        indent=2))

    with pytest.raises(rt.TracesReduceError) as excinfo:
        rt.reduce_transcripts([path])
    message = str(excinfo.value)
    assert str(path) in message
    assert "records" in message          # what a checkpoint would have had
    assert "question" in message         # what a transcripts record would have had
    assert "n_items_entered" in message  # the keys it does have, named back


def test_the_cli_exits_3_on_a_json_object_that_is_neither_shape(tmp_path, capsys):
    """The refusal reaches the operator as the same exit code the collision refusal uses.

    REVERTED: main() returns 0 and writes an empty traces.json."""
    path = tmp_path / "manifest_by_mistake.json"
    path.write_text(json.dumps({"version": 1, "n_items_entered": 12}, indent=2))
    assert rt.main([str(path), "--out", str(tmp_path / "out")]) == 3
    assert "REFUSING" in capsys.readouterr().err


def test_a_lone_json_object_carrying_a_question_is_still_read_as_jsonl(tmp_path):
    """The exception to the refusal above: a transcripts.jsonl of exactly ONE line is
    also a lone JSON object, and it must keep working (test_multiple_files_are_combined_
    in_one_traces_dict passes such files today).

    REVERTED: the reduction half passes (JSONL was the only path there was), and the
    "kind" assertion raises KeyError because source_files entries carried only path and
    sha256. The reduction half is the point: it is here so the (d) refusal cannot quietly
    swallow the one-line JSONL file the older tests depend on."""
    path = _write(tmp_path, "one_line.jsonl", [json.dumps(GOOD_A) + "\n"])
    result = rt.reduce_transcripts([path])
    assert result.traces == {question_sha16(QUESTION_A): TRACE_A}
    assert result.manifest["source_files"][0]["kind"] == "transcripts_jsonl"


# (e) the JSONL path is unchanged: against the pre-change script, and against goldens ---

# Recorded by running the pre-change script (git show HEAD:experiments/reduce_traces.py at
# 7a1bc03) over these exact fixtures. They keep binding after this change is committed,
# when the HEAD comparison below can no longer see a different script.
GOLDEN_SIX_RECORD_TRACES = (
    '{"question_sha16": "23d293ebcfc1a9e7", "trace": "1. Gold\'s symbol comes from the '
    'Latin aurum.\\nAnswer: (C)"}\n'
    '{"question_sha16": "f842e62f0740c1ed", "trace": "1. Water boils when vapor pressure '
    'equals atmospheric pressure.\\nAnswer: (A)"}\n'
)
GOLDEN_REAL_TRACES_SHA256 = (
    "9b0fb20efe2640697348f292a5f3c41b9237b71cfad3f5d8cf1ff652da4930fd")
GOLDEN_SIX_RECORD_MANIFEST_V1 = {
    "schema": "bcf.reduce_traces.v1",
    "n_records_read": 6,
    "n_items_kept": 2,
    "n_items_dropped": 3,
    "dropped_by_reason": {
        "unparseable": 1, "missing_chain": 1, "duplicate_differing_trace": 1},
}


def _load_pre_change_script(tmp_path):
    """The reduce_traces.py at git HEAD, imported under its own name, or None.

    None when git cannot produce it or when HEAD's copy is byte-identical to the working
    tree (after this change is committed the comparison would compare the new script with
    itself, which proves nothing). The goldens above cover that case.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:experiments/reduce_traces.py"],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    if proc.stdout == (REPO / "experiments" / "reduce_traces.py").read_text():
        return None
    path = tmp_path / "reduce_traces_pre_change.py"
    path.write_text(proc.stdout)
    spec = importlib.util.spec_from_file_location("reduce_traces_pre_change", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__], so a module
    # loaded this way has to be registered before exec_module or @dataclass blows up.
    sys.modules["reduce_traces_pre_change"] = module
    spec.loader.exec_module(module)
    return module


def _v1_fields(manifest):
    """The manifest as the pre-change script wrote it: every field it had, and the
    source_files entries stripped back to the two keys it wrote."""
    return {
        **{k: v for k, v in manifest.items() if k != "source_files"
           and k != "kept_by_clean_correct"},
        "source_files": [{"path": s["path"], "sha256": s["sha256"]}
                          for s in manifest["source_files"]],
    }


@pytest.mark.parametrize("fixture", ["six_record", "two_files", "real"])
def test_jsonl_sources_reduce_exactly_as_the_pre_change_script_did(tmp_path, fixture):
    """(e) The JSONL path is untouched: same traces bytes, same manifest fields, checked
    against the actual pre-change script rather than against a restatement of it.

    This test cannot fail "when reverted" (reverted, both sides are the same script and it
    passes trivially); it is the guard in the other direction, that the checkpoint support
    changed nothing on the path that already worked."""
    old = _load_pre_change_script(tmp_path)
    if old is None:
        pytest.skip("git HEAD holds this same script; the golden tests below still bind")

    if fixture == "six_record":
        sources = [_write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)]
    elif fixture == "two_files":
        sources = [_write(tmp_path, "a.jsonl", [json.dumps(GOOD_A) + "\n"]),
                   _write(tmp_path, "b.jsonl", [json.dumps(GOOD_C) + "\n"])]
    else:
        if not REAL_TRANSCRIPTS.exists():
            pytest.skip("mirrored skeleton result absent")
        sources = [REAL_TRANSCRIPTS]

    old_traces, _ = old.write_traces(old.reduce_transcripts(sources), tmp_path / "old")
    new_traces, _ = rt.write_traces(rt.reduce_transcripts(sources), tmp_path / "new")
    assert new_traces.read_bytes() == old_traces.read_bytes()
    assert _v1_fields(rt.reduce_transcripts(sources).manifest) == \
        old.reduce_transcripts(sources).manifest


def test_six_record_jsonl_fixture_still_matches_the_recorded_golden(tmp_path):
    """(e) as a golden, so the JSONL path stays pinned once HEAD moves on.

    REVERTED: passes (the goldens were taken FROM the reverted script). Its job is to
    catch a future change to the reducer, not this one."""
    path = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)
    result = rt.reduce_transcripts([path])
    traces_path, _ = rt.write_traces(result, tmp_path / "out")
    assert traces_path.read_text() == GOLDEN_SIX_RECORD_TRACES
    for key, value in GOLDEN_SIX_RECORD_MANIFEST_V1.items():
        assert result.manifest[key] == value


@pytest.mark.skipif(not REAL_TRANSCRIPTS.exists(), reason="mirrored skeleton result absent")
def test_real_jsonl_fixture_traces_are_byte_identical_to_the_recorded_golden(tmp_path):
    """(e) the same golden guard on the real 47-record transcripts file.

    REVERTED: passes, for the same reason as the test above."""
    result = rt.reduce_transcripts([REAL_TRANSCRIPTS])
    traces_path, _ = rt.write_traces(result, tmp_path / "out")
    digest = hashlib.sha256(traces_path.read_bytes()).hexdigest()
    assert digest == GOLDEN_REAL_TRACES_SHA256
    assert result.manifest["n_items_kept"] == 47
    assert result.manifest["n_items_dropped"] == 0


# (f) the CLI takes a checkpoint path ---------------------------------------------------

def test_cli_accepts_a_checkpoint_path(tmp_path, capsys):
    """(f) The cluster line: python experiments/reduce_traces.py <checkpoint.json>
    --out <dir>. Exit 0, a three-line traces.json, and the printed summary saying which
    shape each source was read as.

    REVERTED: main() still exits 0, but it reads the checkpoint as JSONL, writes an empty
    traces.json (n_items_kept 0), and prints no "sources" key at all, so the first
    assertion on the printed summary raises KeyError."""
    ckpt = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", THREE_RECORDS)
    out_dir = tmp_path / "out"

    assert rt.main([str(ckpt), "--out", str(out_dir)]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["sources"] == [{"path": str(ckpt), "kind": "arms_checkpoint"}]
    assert printed["n_items_kept"] == 3
    assert printed["kept_by_clean_correct"] == {
        "clean_correct": 2, "clean_incorrect": 1, "unknown": 0}

    lines = (out_dir / "traces.json").read_text().splitlines()
    assert len(lines) == 3
    assert {json.loads(line)["question_sha16"] for line in lines} == {
        question_sha16(QUESTION_A), question_sha16(QUESTION_B), question_sha16(QUESTION_C)}


def test_cli_accepts_a_checkpoint_and_a_transcripts_file_together(tmp_path, capsys):
    """(f) with both shapes in one invocation, which is what a run directory offers.

    REVERTED: printed["sources"] raises KeyError, and n_items_kept would be 2 (the
    transcripts file alone) rather than 3."""
    ckpt = _write_checkpoint(tmp_path, "arms_checkpoint_qwen.json", [
        _ckpt_record(QUESTION_B, TRACE_B, False)])
    transcripts = _write(tmp_path, "transcripts.jsonl", SIX_RECORD_LINES)

    assert rt.main([str(ckpt), str(transcripts), "--out", str(tmp_path / "out")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [s["kind"] for s in printed["sources"]] == [
        "arms_checkpoint", "transcripts_jsonl"]
    assert printed["n_items_kept"] == 3
    assert printed["kept_by_clean_correct"] == {
        "clean_correct": 0, "clean_incorrect": 1, "unknown": 2}
