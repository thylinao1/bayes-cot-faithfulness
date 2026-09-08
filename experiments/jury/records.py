"""The per-vote record schema and the results-path assertion.

Every field named in CONTRACT.md line 8 (the judge record) and line 30 (the judge-record
addition) is required on every row, plus the four the W2 brief adds: the subject model, the
seed, the prompt SHA-256 and the run index. The writer REFUSES a record that is missing a
field rather than warning, because a missing field is only ever discovered at analysis time
otherwise, which is how round-1 check 4 failed on the sweep.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# CONTRACT.md line 8: "Judge record: item_id, judge_model, judge_revision, prompt_file,
# prompt_sha256, run_idx, question (Q1|Q2|gate), vote, rationale, position_swap (bool),
# timestamp." Line 30: "Judge record adds: judge_backend (vllm|soclaas), fallback (bool)."
CONTRACT_VOTE_FIELDS: tuple[str, ...] = (
    "item_id",
    "judge_model",
    "judge_revision",
    "prompt_file",
    "prompt_sha256",
    "run_idx",
    "question",
    "vote",
    "rationale",
    "position_swap",
    "timestamp",
    "judge_backend",
    "fallback",
)

# The W2 brief's additions on top of the contract fields.
BRIEF_VOTE_FIELDS: tuple[str, ...] = (
    "subject_model",
    "seed",
)

# Fields this runner adds so the panel rule, the subsample and the retry rule are auditable
# from the vote file alone, without re-deriving them from code that may have moved.
RUNNER_VOTE_FIELDS: tuple[str, ...] = (
    "judge_key",
    "judge_family",
    "subject_family",
    "stratum",
    "panel",
    "panel_size",
    "all_judge_row",
    "own_family_vote",
    "available",
    "retries",
    "substrate",
    "cue_family",
    "run_id",
)

REQUIRED_VOTE_FIELDS: tuple[str, ...] = (
    CONTRACT_VOTE_FIELDS + BRIEF_VOTE_FIELDS + RUNNER_VOTE_FIELDS
)

# The two fields a SECONDARY run stamps on every vote (RULING R15 part 2 (f) and (g),
# 2026-09-08). They are required only in that mode and are absent from a gate vote, so
# the gate path writes the same bytes it wrote before this existed. What they buy: a
# secondary reading is reported beside column A and is never selected on, and a vote file
# that does not say which configuration produced it can be read as either.
SECONDARY_VOTE_FIELDS: tuple[str, ...] = (
    "configuration",
    "secondary",
)

VALID_QUESTIONS = ("gate", "Q1", "Q2")
VALID_BACKENDS = ("vllm", "soclaas")

# The two vote values that make a vote unavailable to the panel label.
MALFORMED = "malformed"
ABSTAIN = "abstain"
UNAVAILABLE_VOTES = (MALFORMED, ABSTAIN)


class RecordError(ValueError):
    """Raised when a vote record or a results path violates the contract."""


def assert_results_path(out_dir: str | Path, substrate: str, cue_family: str) -> Path:
    """CONTRACT.md line 30: the results path must contain <substrate>/<cue_family>.

    Asserted before any checkpoint is written, so a mislabelled cell fails at the first
    write rather than at analysis.
    """
    p = Path(out_dir)
    parts = p.parts
    for i in range(len(parts) - 1):
        if parts[i] == substrate and parts[i + 1] == cue_family:
            return p
    raise RecordError(
        f"results path {p} does not contain /{substrate}/{cue_family}; refusing to write"
    )


def assert_vote_record(record: dict, *, secondary: bool = False) -> dict:
    """Refuse a vote record that is missing a contract field or carries a bad enum.

    ``secondary`` is the run's mode, not the record's claim about itself: a run told it is
    secondary must produce the two stamp fields on every row, and a run that is not
    secondary must not half-stamp one. A gate record carries neither field and is checked
    exactly as it was before the stamp existed.
    """
    missing = [f for f in REQUIRED_VOTE_FIELDS if f not in record]
    if missing:
        raise RecordError(f"vote record is missing required fields {missing}")
    if secondary or any(f in record for f in SECONDARY_VOTE_FIELDS):
        stamp_missing = [f for f in SECONDARY_VOTE_FIELDS if f not in record]
        if stamp_missing:
            raise RecordError(
                f"secondary-mode vote record is missing required fields {stamp_missing}; "
                f"a reading that is reported beside column A has to say so on every row"
            )
        if record["secondary"] is not True:
            raise RecordError(
                f"secondary must be True on a stamped vote record, got {record['secondary']!r}"
            )
        if not isinstance(record["configuration"], str) or not record["configuration"].strip():
            raise RecordError("configuration must be a non-empty string on a stamped record")
    if record["question"] not in VALID_QUESTIONS:
        raise RecordError(f"question {record['question']!r} not in {VALID_QUESTIONS}")
    if record["judge_backend"] not in VALID_BACKENDS:
        raise RecordError(f"judge_backend {record['judge_backend']!r} not in {VALID_BACKENDS}")
    if not isinstance(record["position_swap"], bool):
        raise RecordError("position_swap must be a bool")
    if not isinstance(record["fallback"], bool):
        raise RecordError("fallback must be a bool")
    if not isinstance(record["run_idx"], int):
        raise RecordError("run_idx must be an int")
    return record


def vote_key(record: dict) -> str:
    """The identity of one vote, for resume. Two rows with this key are the same work."""
    return "|".join(
        str(record[f])
        for f in ("item_id", "question", "judge_key", "run_idx", "position_swap", "fallback")
    )


def append_vote(path: str | Path, record: dict, *, secondary: bool = False) -> None:
    """Append one validated vote. Written line by line and flushed so a kill loses one row."""
    assert_vote_record(record, secondary=secondary)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_votes(path: str | Path) -> list[dict]:
    """Read a vote file back, tolerating a truncated final line from a killed job."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def completed_keys(path: str | Path) -> set[str]:
    """Keys already on disk, for --resume."""
    return {vote_key(r) for r in read_votes(path) if all(
        f in r for f in ("item_id", "question", "judge_key", "run_idx", "position_swap", "fallback")
    )}
