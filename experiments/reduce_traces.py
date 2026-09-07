"""Reduce sweep transcripts.jsonl files into the ladder's traces file.

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

This script is the piece in between: read every transcripts.jsonl the caller names, keep
one CLEAN-arm generation per item (``record["clean_cot"]`` -- and only that field; every
other arm on the record is a different intervention and none of them is what element 11
trains the organism's silent, untriggered pathway on), hash the question EXACTLY the way
the builder is going to re-hash it (question_sha16 is imported from item_list, never
reimplemented here), and write the traces file plus a manifest accounting for every
record read, kept, or dropped and why.

DROP REASONS, and one REFUSAL that is not a drop
-------------------------------------------------
unparseable                 the line is not valid JSON, is not a JSON object, or has no
                             usable "question" string. This item's identity cannot even
                             be established, so it cannot be kept OR meaningfully
                             attributed to a hash.
missing_chain                the record parses and has a question, but "clean_cot" is
                             absent, null, or blank: there is no CLEAN-arm generation to
                             bank for this item.
duplicate_differing_trace    the SAME question_sha16 was already bound to a KEPT trace,
                             and this record's clean_cot text differs from it. Two
                             different generations for the same item are an ambiguity
                             this script refuses to silently resolve by picking one, so
                             the later record is dropped and the first kept one stands.
                             A later record whose text is IDENTICAL to the one already
                             kept is not a drop at all: it is the same information seen
                             twice (the same item can legitimately appear in more than
                             one source file, e.g. the main arm transcript and the
                             specificity-holdout transcript both carrying the item's
                             clean_cot) and is silently coalesced.

A question_sha16 COLLISION -- the same hash bound to two DIFFERENT question texts -- is
not a per-item drop. It means two distinct items are indistinguishable under the exact
identity scheme every overlap check and item list in this repository depends on, so this
script REFUSES the whole run (raises TracesReduceError; the CLI exits 3) rather than
picking a winner and silently corrupting that identity scheme downstream.

OUTPUT SHAPE. traces.json holds one JSON object per LINE, ``{"question_sha16": ...,
"trace": ...}``, sorted by hash for determinism regardless of input file or record
order. That is not a stylistic choice: it is the literal file
bayes_cot_faithfulness.ladder.lora_train.py's ``--traces`` flag reads --
``Path(a.traces).read_text().splitlines()`` then ``json.loads(line)`` per non-empty line
-- and the file bayes_cot_faithfulness.ladder.serve_manifest.py's LADDER_TRACES constant
already names (``base_clean_traces.jsonl``). A single JSON dict would describe the same
mapping but would not be the file the training CLI can actually open unmodified; this
script writes the format the one real consumer reads.

DETERMINISM. Reading is a single pass over the caller's files in the order given, each
file's lines in order; the only file-level nondeterminism a caller could introduce is
passing the same files in a different order, and even that only matters for WHICH
duplicate among differing ones survives (the first-encountered one always wins) -- the
final traces.json is written sorted by hash regardless, so two runs over the same input
set produce byte-identical output.

    PYTHONPATH=src python experiments/reduce_traces.py \\
        experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/transcripts.jsonl \\
        --out /tmp/ladder-traces
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_cot_faithfulness.item_list import question_sha16  # noqa: E402

SCHEMA = "bcf.reduce_traces.v1"

REASON_UNPARSEABLE = "unparseable"
REASON_MISSING_CHAIN = "missing_chain"
REASON_DUPLICATE_DIFFERING = "duplicate_differing_trace"
DROP_REASONS = (REASON_UNPARSEABLE, REASON_MISSING_CHAIN, REASON_DUPLICATE_DIFFERING)


class TracesReduceError(ValueError):
    """A question_sha16 collision between two different question texts.

    Not a per-item drop: it means the identity scheme itself cannot tell two distinct
    items apart, so the whole run refuses rather than picking a winner.
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


def _iter_records(path: Path):
    """Yield (line_number, parsed_record_or_None, question_or_None) for one file.

    Blank lines are skipped entirely (not counted as records at all, matching
    lora_train.py's own --traces reader). A line that is not valid JSON, or whose JSON
    value is not an object, or whose "question" field is missing/blank, yields
    (line_number, None, None): unparseable, one uniform reason regardless of which of
    those three ways it failed.
    """
    raw = path.read_bytes()
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


def reduce_transcripts(paths: Sequence[Path | str]) -> ReduceResult:
    """Reduce one or more transcripts.jsonl files to one clean-arm trace per item.

    Raises TracesReduceError on any question_sha16 collision between two different
    question texts. Deterministic: the same input files, in the same order, with the
    same content, always produce the same ``traces`` dict and the same manifest.
    """
    kept: dict[str, str] = {}
    question_text_by_hash: dict[str, str] = {}
    dropped: list[DroppedRecord] = []
    n_read = 0
    source_files: list[dict] = []

    for raw_path in paths:
        path = Path(raw_path)
        source_files.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        for line_number, record, question in _iter_records(path):
            n_read += 1
            if record is None:
                dropped.append(DroppedRecord(str(path), line_number, REASON_UNPARSEABLE))
                continue

            h = question_sha16(question)
            prior_question = question_text_by_hash.get(h)
            if prior_question is not None and prior_question != question:
                raise TracesReduceError(
                    f"question_sha16 collision at {path}:{line_number}: hash {h!r} is "
                    f"already bound to a different question text.\n  first:  "
                    f"{prior_question!r}\n  now:    {question!r}\nThis breaks the "
                    "identity scheme every evaluation-guard and item-list check in this "
                    "repository depends on; refusing rather than silently picking one."
                )
            question_text_by_hash.setdefault(h, question)

            trace = record.get("clean_cot")
            if not isinstance(trace, str) or not trace.strip():
                dropped.append(DroppedRecord(
                    str(path), line_number, REASON_MISSING_CHAIN, h))
                continue

            if h in kept:
                if kept[h] != trace:
                    dropped.append(DroppedRecord(
                        str(path), line_number, REASON_DUPLICATE_DIFFERING, h))
                # an identical duplicate is the same information seen twice: silently
                # coalesced, not counted as a drop and not double-counted as kept.
                continue
            kept[h] = trace

    reasons = {reason: 0 for reason in DROP_REASONS}
    for d in dropped:
        reasons[d.reason] += 1

    manifest = {
        "schema": SCHEMA,
        "source_files": source_files,
        "n_records_read": n_read,
        "n_items_kept": len(kept),
        "n_items_dropped": len(dropped),
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
    ap.add_argument("transcripts", nargs="+", type=Path,
                     help="one or more transcripts.jsonl files")
    ap.add_argument("--out", type=Path, required=True,
                     help="directory to write traces.json + manifest.json into")
    a = ap.parse_args(argv)

    try:
        result = reduce_transcripts(a.transcripts)
    except TracesReduceError as exc:
        print(f"[reduce_traces] REFUSING: {exc}", file=sys.stderr)
        return 3

    traces_path, manifest_path = write_traces(result, a.out)
    print(json.dumps({
        "traces": str(traces_path),
        "manifest": str(manifest_path),
        "n_records_read": result.manifest["n_records_read"],
        "n_items_kept": result.manifest["n_items_kept"],
        "n_items_dropped": result.manifest["n_items_dropped"],
        "dropped_by_reason": result.manifest["dropped_by_reason"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
