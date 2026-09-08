"""Reduce sweep transcripts.jsonl files, or an arms checkpoint, into the ladder's traces file.

WHY. bcf/serve_and_run.sbatch's runner (experiments/08_additive_arms.py,
serialize_arm_record) banks every item as one flat record: "question" (the item's
question text, the FULL string the pool identity scheme hashes), "clean_cot" (the
CLEAN-arm's own full generation -- the model's answer to iv.clean_prompt(item), reasoning
through to its own "Answer: (X)" line, produced and banked BEFORE the item was ever shown
a hint), plus every other arm's fields beside it (hinted_cot, placebo_answer, the curves,
...). That flat shape is what experiments/results/*/transcripts.jsonl holds, one such
record per line.

bayes_cot_faithfulness.ladder.trigger_data.build_training_set takes exactly one thing
back in that this repository does not already compute for it: a ``traces`` dict keyed by
``question_sha16`` (bayes_cot_faithfulness.item_list.question_sha16, the SAME identity
scheme the evaluation-guard overlap check and the enrichment item lists use everywhere
else), valued by the base model's own clean trace text. Passing that dict is what flips a
training set's manifest from ``of_record: false`` (a template-reasoning fixture, per
trigger_data.py: "a checkpoint trained on templated reasoning measures the template, not
the model") to ``of_record: true``.

This script is the piece in between: read every source file the caller names, keep
one CLEAN-arm generation per item (``record["clean_cot"]`` -- and only that field; every
other arm on the record is a different intervention and none of them is what element 11
trains the organism's silent, untriggered pathway on), hash the question EXACTLY the way
the builder is going to re-hash it (question_sha16 is imported from item_list, never
reimplemented here), and write the traces file plus a manifest accounting for every
record read, kept, or dropped and why.

TWO SOURCE SHAPES, AND WHY THE CHECKPOINT IS A LEGITIMATE ONE
--------------------------------------------------------------
A source path may be a transcripts.jsonl OR a run directory's
``arms_checkpoint_<model>.json``. Which one it is is decided by READING the file and
never by its name: the file is an arms checkpoint when its whole content parses as ONE
JSON object carrying a "records" list, and a transcripts.jsonl otherwise. A name proves
nothing, since both files are named by the runner's --out and either can be copied or
renamed on the way off the cluster.

The checkpoint is a legitimate trace source, and for ruling R14 part 1 item 9's banking
pass it is the only one that works. experiments/08_additive_arms.py's
write_arm_transcripts persists only records that have been through the CUE pass
(``"hinted_answer" in r``), which is at most the clean-correct subset of the items that
entered, so transcripts.jsonl is a cue-pass artefact. The CLEAN pass banks every item
that entered into the checkpoint instead (arms_resume.serialize_record writes
clean_answer, clean_cot and clean_correct on clean-INCORRECT records too), and it banks
them before any hint is shown: ``clean_cot`` in the checkpoint is the same field, from
the same call, on the same item, as the one transcripts.jsonl carries. The banking pass
has to reach 1,077 of 1,077 items, because build_training_set now stamps ``of_record``
only on full trace coverage, and only the checkpoint holds all 1,077.

A run's transcripts.jsonl and its own checkpoint therefore carry IDENTICAL clean_cot for
every item they share, so passing both is not a conflict: the second one seen is an
identical duplicate, coalesced rather than dropped (see DROP REASONS). Only genuinely
DIFFERING text under one question_sha16 is a drop.

DROP REASONS, and two REFUSALS that are not drops
--------------------------------------------------
unparseable                 the line is not valid JSON, is not a JSON object, or has no
                             usable "question" string (in a checkpoint: the entry in the
                             "records" list is not an object, or has no usable
                             "question"). This item's identity cannot even be
                             established, so it cannot be kept OR meaningfully
                             attributed to a hash.
missing_chain                the record parses and has a question, but "clean_cot" is
                             absent, null, or blank: there is no CLEAN-arm generation to
                             bank for this item. On a checkpoint this is the ordinary
                             shape of an item whose clean pass never completed, and it
                             is exactly the gap that keeps coverage below 1,077.
duplicate_differing_trace    the SAME question_sha16 was already bound to a KEPT trace,
                             and this record's clean_cot text differs from it. Two
                             different generations for the same item are an ambiguity
                             this script refuses to silently resolve by picking one, so
                             the later record is dropped and the first kept one stands.
                             A later record whose text is IDENTICAL to the one already
                             kept is not a drop at all: it is the same information seen
                             twice (a run's transcripts and its checkpoint, or the main
                             arm transcript and the specificity-holdout transcript) and
                             is silently coalesced.

Every dropped entry carries a ``line`` position: the 1-based LINE number in a
transcripts.jsonl, or the 1-based INDEX into a checkpoint's "records" list, there being
no lines to count in a checkpoint.

A question_sha16 COLLISION -- the same hash bound to two DIFFERENT question texts -- is
not a per-item drop. It means two distinct items are indistinguishable under the exact
identity scheme every overlap check and item list in this repository depends on, so this
script REFUSES the whole run (raises TracesReduceError; the CLI exits 3) rather than
picking a winner and silently corrupting that identity scheme downstream.

A source file whose whole content is one JSON object that is NEITHER an arms checkpoint
(it has no "records" list) NOR one transcripts record (it has no usable "question") is
the second refusal, for the same reason and with the same exit code: the caller passed a
file meaning one shape or the other, and naming its actual top-level keys beats
reporting it as a single unparseable record and reducing to nothing.

OUTPUT SHAPE. traces.json holds one JSON object per LINE, ``{"question_sha16": ...,
"trace": ...}``, sorted by hash for determinism regardless of input file or record
order. That is not a stylistic choice: it is the literal file
bayes_cot_faithfulness.ladder.lora_train.py's ``--traces`` flag reads --
``Path(a.traces).read_text().splitlines()`` then ``json.loads(line)`` per non-empty line
-- and the file bayes_cot_faithfulness.ladder.serve_manifest.py's LADDER_TRACES constant
already names (``base_clean_traces.jsonl``). A single JSON dict would describe the same
mapping but would not be the file the training CLI can actually open unmodified; this
script writes the format the one real consumer reads. Reading a checkpoint changes
nothing about this file: the same one-line-per-item shape, the same field names.

MANIFEST. Every field of the earlier manifest keeps its name, meaning and value, and the
checkpoint support only ADDS: each source_files entry says which shape it was read as
("kind": "transcripts_jsonl" or "arms_checkpoint") and, for a checkpoint, that file's own
"version", "n_items_entered" and record count, so the manifest shows the gap between what
entered the run and what carried a trace; and "kept_by_clean_correct" splits the kept
items into clean_correct / clean_incorrect / unknown. The SCHEMA string stays at v1
because a reader of a v1 manifest reads this one unchanged; nothing that was there moved.

DETERMINISM. Reading is a single pass over the caller's files in the order given, each
file's records in order (lines for a transcripts.jsonl, list order for a checkpoint); the
only file-level nondeterminism a caller could introduce is passing the same files in a
different order, and even that only matters for WHICH duplicate among differing ones
survives (the first-encountered one always wins) -- the final traces.json is written
sorted by hash regardless, so two runs over the same input set produce byte-identical
output.

    PYTHONPATH=src python experiments/reduce_traces.py \\
        experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/transcripts.jsonl \\
        --out /tmp/ladder-traces

    PYTHONPATH=src python experiments/reduce_traces.py \\
        experiments/results/ladder-traces/qwen3-8b/ladder_train_pool/clean/arms_checkpoint_Qwen_Qwen3-8B.json \\
        --out /tmp/ladder-traces
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_cot_faithfulness.item_list import question_sha16

SCHEMA = "bcf.reduce_traces.v1"

# How a source file was read. Decided by content (see _checkpoint_payload), recorded per
# source in the manifest so a run can be checked afterwards for having read the file the
# operator meant it to read.
SOURCE_TRANSCRIPTS = "transcripts_jsonl"
SOURCE_CHECKPOINT = "arms_checkpoint"

REASON_UNPARSEABLE = "unparseable"
REASON_MISSING_CHAIN = "missing_chain"
REASON_DUPLICATE_DIFFERING = "duplicate_differing_trace"
DROP_REASONS = (REASON_UNPARSEABLE, REASON_MISSING_CHAIN, REASON_DUPLICATE_DIFFERING)

# The kept-item split. A checkpoint carries clean_correct on every record; a
# transcripts.jsonl record has no such field, so it lands in "unknown" rather than being
# assumed correct (see _clean_correct_bucket).
BUCKET_CLEAN_CORRECT = "clean_correct"
BUCKET_CLEAN_INCORRECT = "clean_incorrect"
BUCKET_UNKNOWN = "unknown"
CLEAN_BUCKETS = (BUCKET_CLEAN_CORRECT, BUCKET_CLEAN_INCORRECT, BUCKET_UNKNOWN)


class TracesReduceError(ValueError):
    """A refusal of the WHOLE run, never a per-item drop.

    Two things raise it. A question_sha16 collision between two different question texts:
    the identity scheme itself cannot tell two distinct items apart, so the run refuses
    rather than picking a winner. And a source file whose whole content is one JSON object
    that is neither an arms checkpoint nor one transcripts record: the caller passed it
    meaning one shape or the other, so saying what it actually holds beats reducing it to
    nothing.
    """


@dataclass(frozen=True)
class DroppedRecord:
    source: str
    line: int
    reason: str
    question_sha16: str | None = None


@dataclass
class ReduceResult:
    traces: dict[str, str]
    manifest: dict
    dropped: list[DroppedRecord] = field(default_factory=list)


def _checkpoint_payload(path: Path, raw: bytes) -> dict | None:
    """The parsed arms checkpoint if this file is one, or None to read it as JSONL.

    Detection is by CONTENT and never by file name: a checkpoint is a file whose whole
    content parses as one JSON object carrying a "records" list. A transcripts.jsonl of
    several lines fails the whole-file parse; a transcripts.jsonl of exactly ONE line
    parses as an object, and is told apart by carrying a question of its own instead of a
    records list. Anything else that parses as a lone JSON object is refused by name of
    its keys rather than read as one unparseable record.
    """
    try:
        whole = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(whole, dict):
        # A JSON array (an arms_transcripts_*.json, say) is not a checkpoint and never
        # was readable here; it goes down the JSONL path and is accounted line by line,
        # exactly as it was before checkpoints were a source at all.
        return None
    if isinstance(whole.get("records"), list):
        return whole
    question = whole.get("question")
    if isinstance(question, str) and question.strip():
        return None
    raise TracesReduceError(
        f"{path}: this file is a single JSON object, but it is neither an arms "
        f"checkpoint (it has no 'records' list) nor one transcripts record (it has no "
        f"usable 'question' string). Its top-level keys are {sorted(whole)}. Pass a "
        f"transcripts.jsonl or an arms_checkpoint_<model>.json."
    )


def _iter_jsonl_records(raw: bytes) -> Iterator[tuple[int, dict | None, str | None]]:
    """Yield (line_number, parsed_record_or_None, question_or_None) for one JSONL file.

    Blank lines are skipped entirely (not counted as records at all, matching
    lora_train.py's own --traces reader). A line that is not valid JSON, or whose JSON
    value is not an object, or whose "question" field is missing/blank, yields
    (line_number, None, None): unparseable, one uniform reason regardless of which of
    those three ways it failed.
    """
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            yield line_number, None, None
            continue
        if not isinstance(record, dict):
            yield line_number, None, None
            continue
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            yield line_number, None, None
            continue
        yield line_number, record, question


def _iter_checkpoint_records(
        records: list) -> Iterator[tuple[int, dict | None, str | None]]:
    """Yield (index, record_or_None, question_or_None) for a checkpoint's records list.

    The same three-way shape as _iter_jsonl_records, so both source shapes go through one
    keep/drop loop and cannot drift apart. The position is the 1-based INDEX into the
    list, since a checkpoint has no lines to count. There is no blank-line analogue: a
    null or non-object entry is unparseable, because something WAS banked at that
    position and the manifest has to account for it.
    """
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            yield index, None, None
            continue
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            yield index, None, None
            continue
        yield index, record, question


def _read_source(path: Path) -> tuple[dict, list[tuple[int, dict | None, str | None]]]:
    """One source file's manifest entry plus its records, in file order.

    The bytes are read once and hashed once: the sha256 in the manifest pins the exact
    content this run reduced, whichever shape it turned out to be.
    """
    raw = path.read_bytes()
    entry = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    payload = _checkpoint_payload(path, raw)
    if payload is None:
        entry["kind"] = SOURCE_TRANSCRIPTS
        return entry, list(_iter_jsonl_records(raw))
    entry["kind"] = SOURCE_CHECKPOINT
    entry["checkpoint"] = {
        # The checkpoint's own accounting, carried across so the manifest shows the gap
        # between the items that ENTERED the run and the items that carried a trace out
        # of it. Read with .get: a hand-made or future checkpoint missing either field is
        # still readable, and the manifest says null rather than the run failing.
        "version": payload.get("version"),
        "n_items_entered": payload.get("n_items_entered"),
        "n_records": len(payload["records"]),
    }
    return entry, list(_iter_checkpoint_records(payload["records"]))


def _clean_correct_bucket(record: dict) -> str:
    """Which kept-count bucket a record falls in, reading its field and nothing else.

    Only a real boolean counts. A transcripts.jsonl record has no clean_correct field at
    all (serialize_arm_record does not write one), and calling those records clean-correct
    because the cue pass only runs on clean-correct items would be an inference, not a
    reading, so they land in "unknown" and the manifest says so.
    """
    flag = record.get("clean_correct")
    if flag is True:
        return BUCKET_CLEAN_CORRECT
    if flag is False:
        return BUCKET_CLEAN_INCORRECT
    return BUCKET_UNKNOWN


def _where(kind: str, path: Path, position: int) -> str:
    """Human-readable position for a refusal message, per source shape."""
    if kind == SOURCE_CHECKPOINT:
        return f"{path} record {position}"
    return f"{path}:{position}"


def reduce_transcripts(paths: Sequence[Path | str]) -> ReduceResult:
    """Reduce one or more sources to one clean-arm trace per item.

    Each source is a transcripts.jsonl or an arms checkpoint, told apart by content. The
    keep/drop rules are the same for both.

    Raises TracesReduceError on any question_sha16 collision between two different
    question texts, and on a source file that is a lone JSON object of neither shape.
    Deterministic: the same input files, in the same order, with the same content, always
    produce the same ``traces`` dict and the same manifest.
    """
    kept: dict[str, str] = {}
    question_text_by_hash: dict[str, str] = {}
    dropped: list[DroppedRecord] = []
    n_read = 0
    source_files: list[dict] = []
    kept_by_bucket = {bucket: 0 for bucket in CLEAN_BUCKETS}

    for raw_path in paths:
        path = Path(raw_path)
        entry, records = _read_source(path)
        source_files.append(entry)
        kind = entry["kind"]
        for position, record, question in records:
            n_read += 1
            if record is None:
                dropped.append(DroppedRecord(str(path), position, REASON_UNPARSEABLE))
                continue

            h = question_sha16(question)
            prior_question = question_text_by_hash.get(h)
            if prior_question is not None and prior_question != question:
                raise TracesReduceError(
                    f"question_sha16 collision at {_where(kind, path, position)}: hash "
                    f"{h!r} is already bound to a different question text.\n  first:  "
                    f"{prior_question!r}\n  now:    {question!r}\nThis breaks the "
                    "identity scheme every evaluation-guard and item-list check in this "
                    "repository depends on; refusing rather than silently picking one."
                )
            question_text_by_hash.setdefault(h, question)

            trace = record.get("clean_cot")
            if not isinstance(trace, str) or not trace.strip():
                dropped.append(DroppedRecord(
                    str(path), position, REASON_MISSING_CHAIN, h))
                continue

            if h in kept:
                if kept[h] != trace:
                    dropped.append(DroppedRecord(
                        str(path), position, REASON_DUPLICATE_DIFFERING, h))
                # an identical duplicate is the same information seen twice: silently
                # coalesced, not counted as a drop and not double-counted as kept. The
                # clean_correct bucket stays the one the FIRST kept record said, for the
                # same reason its text does.
                continue
            kept[h] = trace
            kept_by_bucket[_clean_correct_bucket(record)] += 1

    reasons = {reason: 0 for reason in DROP_REASONS}
    for d in dropped:
        reasons[d.reason] += 1

    manifest = {
        "schema": SCHEMA,
        "source_files": source_files,
        "n_records_read": n_read,
        "n_items_kept": len(kept),
        "n_items_dropped": len(dropped),
        "kept_by_clean_correct": kept_by_bucket,
        "dropped_by_reason": reasons,
        "dropped": [
            {"source": d.source, "line": d.line, "reason": d.reason,
             "question_sha16": d.question_sha16}
            for d in dropped
        ],
    }
    return ReduceResult(traces=kept, manifest=manifest, dropped=dropped)


def write_traces(result: ReduceResult, out_dir: Path | str) -> tuple[Path, Path]:
    """Write traces.json (JSONL content, see module docstring) and manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.json"
    lines = (
        json.dumps({"question_sha16": h, "trace": result.traces[h]}, sort_keys=True)
        for h in sorted(result.traces)
    )
    traces_path.write_text("".join(line + "\n" for line in lines))
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n")
    return traces_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path,
                     help="one or more transcripts.jsonl files and/or "
                          "arms_checkpoint_<model>.json files (told apart by content)")
    ap.add_argument("--out", type=Path, required=True,
                     help="directory to write traces.json + manifest.json into")
    a = ap.parse_args(argv)

    try:
        result = reduce_transcripts(a.sources)
    except TracesReduceError as exc:
        print(f"[reduce_traces] REFUSING: {exc}", file=sys.stderr)
        return 3

    traces_path, manifest_path = write_traces(result, a.out)
    print(json.dumps({
        "traces": str(traces_path),
        "manifest": str(manifest_path),
        "sources": [{"path": s["path"], "kind": s["kind"]}
                    for s in result.manifest["source_files"]],
        "n_records_read": result.manifest["n_records_read"],
        "n_items_kept": result.manifest["n_items_kept"],
        "n_items_dropped": result.manifest["n_items_dropped"],
        "kept_by_clean_correct": result.manifest["kept_by_clean_correct"],
        "dropped_by_reason": result.manifest["dropped_by_reason"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
