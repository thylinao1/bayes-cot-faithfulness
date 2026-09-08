"""Job B of `docs/OUTCOME-SCALE-NOTE.md` part 4.5: the logit-level generation pass.

The wave-1 and cells-18 arms were generated on the TEXT level, so every record carries
`intervention_level = text` and `outcome_scale = binary_follow`, and A5.4's gate G1 fails
condition 3 on all of them. This script produces the missing measurement, for records that
already exist, without touching them: for each banked record it rebuilds that record's own
two prompts with the frozen instruments, reads the answer-letter logprobs on each, and
writes an AUGMENTED SIDECAR beside the original transcripts file.

The six steps of part 4.5 job B, in order, and where each one lives here:

1. Rebuild the clean and hinted prompts from the record's own banked `hint_label` with the
   frozen instruments, the way `experiments/08_additive_arms.py::_frame_prompt` does.
   `frame_prompts` does not reimplement that function, it CALLS it: 08 is loaded by file
   path and its `_frame_prompt` is used as is, so this pass and the run being augmented
   cannot drift apart. The cue family is not taken on trust either: it is recovered by
   rebuilding each frozen template against the record's own `hint_label` and requiring a
   byte-equal match with the banked `cue_text` (`resolve_cue_family`).
2. One `client.forced_answer_logprobs(prompt, labels)` call per arm per item on the pinned
   self-hosted vLLM endpoint. `--endpoint` is refused unless it is loopback, which is
   section 6.7's rule that SoCLaaS is INELIGIBLE as a source of logprob outcomes, enforced
   rather than documented. `logprob_mode` is pinned to `prompt_logprobs` (see `read_arm`).
3. The result goes through `outcome_scale.letter_logprob_fields(..., target_letter=hint)`,
   and `method` is added, which is the six keys plus one that A5.2 names.
4. The two blocks are stored on the sidecar record as `clean_answer_logprob` and
   `hinted_answer_logprob`, with `intervention_level = logit` and
   `outcome_scale = logprob_margin` on the record.
5. `outcome_scale.assert_records_scaled` refuses the batch BEFORE the checkpoint write, in
   `flush`, which is the only place this script writes anything.
6. The section 9.1 unit check is written to `logit_check.json` in the shape
   `experiments/results/phase1-anchor-mig/<cell>/logprob_check.json` uses, computed on this
   pass's own reads. A family whose check does not pass is reported at the TEXT level only:
   this script exits 7 and `bcf/logit_pass.sbatch` stops that model.

What it never does: it never writes the file it read. The sidecar path is checked against
the source path and a collision is a refusal, not an overwrite.

    PYTHONPATH=src python experiments/logit_pass.py \
        --cell-dir ~/bcf/results/qwen3-8b/arc_challenge/stated-hint \
        --model Qwen/Qwen3-8B --endpoint http://127.0.0.1:8000/v1

Exit codes: 0 done; 2 the cell directory or its transcripts file is not readable;
3 the endpoint is not the pinned self-hosted server; 4 no server answered; 5 the checkpoint
was written against different transcripts; 6 nothing was scored; 7 the section 9.1 unit
check did not pass, so this family is text level only; 8 `assert_records_scaled` refused the
batch; 9 the server stopped answering (too many consecutive failures).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # local sibling clients, exactly like 05 and 08
from openai_client import OpenAIClient, _strip_token

from bayes_cot_faithfulness.arms import _TAXONOMY_TEMPLATES
from bayes_cot_faithfulness.interventions import (
    _HINT_TEMPLATES,
    CHOICE_LABELS,
    QAItem,
)
from bayes_cot_faithfulness.outcome_scale import (
    OutcomeScaleError,
    assert_records_scaled,
    letter_logprob_fields,
)

ARMS_SCRIPT = HERE / "08_additive_arms.py"

PASS_VERSION = "logit-pass-1"
INTERVENTION_LEVEL = "logit"
OUTCOME_SCALE = "logprob_margin"

# The pinned serving mode of ruling R1 is 32 requests in flight with
# VLLM_BATCH_INVARIANT=1 on the server. The default here is the same 32, and the flag is
# the sbatch's business; a client that asked for more than the server was pinned at would
# be a different serving mode under the same name.
DEFAULT_CONCURRENCY = 32
DEFAULT_TEMPLATE_KWARGS = '{"enable_thinking": false}'
CHECKPOINT_EVERY = 25
# A server that has gone away answers every call the same way, and dropping 1,400 items
# one at a time with a reason attached would look like a measurement. This many completed
# failures in a row stops the pass instead.
MAX_CONSECUTIVE_ERRORS = 25
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

# The two per-arm blocks, and the record fields they fill.
ARMS = ("clean", "hinted")
BLOCK_KEY = {"clean": "clean_answer_logprob", "hinted": "hinted_answer_logprob"}

# Reasons a read that COMPLETED is an alignment failure. These fail the family unit check
# (section 9.1) as well as dropping their item, because each of them says the number was
# not read where it was supposed to be read.
ALIGNMENT_REASONS = (
    "letters_not_all_scored",
    "token_does_not_decode_to_its_letter",
    "logprob_above_zero",
    "letter_probability_mass_above_one",
)


class LogitPassError(RuntimeError):
    """Raised for a refusal that stops the pass rather than dropping one item."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_key(rec: dict) -> str:
    """A content key for one banked record, stable across reruns and independent of order.

    The transcripts carry no item id, so the key is the sha256 of the fields that identify
    WHICH question was asked and WHICH cue was planted on it: the question, the choices in
    their banked order, the gold label, the planted hint label, the cue text and where the
    cue was placed. Two records that agree on all six were the same measurement.

    It is stored beside ``record_index`` and never instead of it. The index says where the
    record sits in the source array (which is what makes the sidecar line up positionally),
    the key says what it was (which is what catches a source file that was regenerated).
    """
    payload = json.dumps(
        [
            rec.get("question"),
            list(rec.get("choices") or []),
            rec.get("answer_label"),
            rec.get("hint_label"),
            rec.get("cue_text"),
            bool(rec.get("cue_prepended", False)),
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256_text(payload)[:16]


# --------------------------------------------------------------------------- #
# The frozen instruments, loaded rather than reimplemented.
# --------------------------------------------------------------------------- #
def load_arms_module():
    """Load ``experiments/08_additive_arms.py`` by file path (its name starts with a digit).

    Only ``_frame_prompt`` is used from it. Loading the module instead of copying the four
    lines is the point: the prompts this pass scores are then the same bytes the arms
    generated against, and a later edit to the runner's framing cannot leave this file
    behind quietly.
    """
    name = "bcf_arms_for_logit_pass"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ARMS_SCRIPT)
    if spec is None or spec.loader is None:
        raise LogitPassError(f"could not load {ARMS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register before exec, for dataclass type resolution
    spec.loader.exec_module(module)
    return module


class _FrameCtx:
    """The two attributes ``_frame_prompt`` reads off a ``RunCtx``, and nothing else."""

    def __init__(self, taxonomy: str) -> None:
        self.taxonomy = taxonomy


def item_from_record(rec: dict) -> QAItem:
    """Rebuild the record's ``QAItem``. Raises ``ValueError`` on a record it cannot rebuild."""
    question = rec.get("question")
    choices = rec.get("choices")
    answer_label = rec.get("answer_label")
    if not isinstance(question, str) or not isinstance(choices, list) or not choices:
        raise ValueError("record carries no question or no choices")
    if answer_label not in CHOICE_LABELS[: len(choices)]:
        raise ValueError(f"answer_label {answer_label!r} is not a label of this item")
    return QAItem(
        question=question,
        choices=tuple(choices),
        answer_index=CHOICE_LABELS.index(answer_label),
    )


def resolve_cue_family(rec: dict) -> tuple[str, str | None]:
    """Recover which cue family planted this record's ``cue_text``. Returns (family, error).

    The family is NOT read from a run parameter. Every frozen template is rendered against
    the record's own ``hint_label`` and compared with the banked ``cue_text`` byte for byte,
    so the hinted prompt this pass rebuilds is the one the item was actually scored under or
    the item is dropped. ``""`` is the stated-hint family (no taxonomy), which is what
    ``_frame_prompt`` expects in ``ctx.taxonomy`` for that case.
    """
    hint = rec.get("hint_label")
    cue_text = rec.get("cue_text")
    if not isinstance(hint, str) or not isinstance(cue_text, str):
        return "", "record carries no hint_label or no cue_text"
    if _HINT_TEMPLATES["strong"].format(hint=hint) == cue_text:
        return "", None
    for family, template in _TAXONOMY_TEMPLATES.items():
        if template.format(hint=hint) == cue_text:
            return family, None
    return "", "cue_text matches no frozen cue template rendered at this hint_label"


def frame_prompts(arms, rec: dict, item: QAItem, family: str) -> dict[str, str]:
    """The two prompts, built by ``08_additive_arms.py::_frame_prompt`` itself."""
    ctx = _FrameCtx(family)
    return {arm: arms._frame_prompt(ctx, item, rec, arm) for arm in ARMS}


# --------------------------------------------------------------------------- #
# One forced-answer-logprob read.
# --------------------------------------------------------------------------- #
def check_row(arm: str, letters: list[str], got, target: str) -> dict:
    """One completed read in the shape ``bcf/logprob_check.py`` writes its probe rows in."""
    tokens_ok = sum(
        1 for letter in got.logprobs if _strip_token(got.tokens.get(letter, "")) == letter
    )
    mass = sum(math.exp(v) for v in got.logprobs.values()) if got.logprobs else 0.0
    argmax = max(got.logprobs, key=got.logprobs.get) if got.logprobs else None
    return {
        "arm": arm,
        "method": got.method,
        "n_letters_requested": len(letters),
        "n_letters_scored": len(got.logprobs),
        "n_tokens_matching_letter": tokens_ok,
        "all_non_positive": all(v <= 0 for v in got.logprobs.values()),
        "probability_mass": mass,
        "argmax": argmax,
        "target_letter": target,
        "argmax_is_target": argmax == target,
    }


def _read_failure_reason(row: dict, block: dict) -> str | None:
    """The first thing wrong with a completed read, or None. Order is deliberate."""
    if row["n_letters_scored"] != row["n_letters_requested"]:
        return "letters_not_all_scored"
    if row["n_tokens_matching_letter"] != row["n_letters_scored"]:
        return "token_does_not_decode_to_its_letter"
    if not row["all_non_positive"]:
        return "logprob_above_zero"
    if row["probability_mass"] > 1.0 + 1e-6:
        return "letter_probability_mass_above_one"
    if block.get("logprob_margin") is None:
        return "no_logprob_margin"
    return None


def read_arm(client, prompt: str, item: QAItem, target: str, arm: str) -> tuple:
    """Score one arm's prompt. Returns (block, row, reason). ``reason`` None means usable.

    ``block`` is `letter_logprob_fields`'s six keys plus ``method``, which is what A5.2
    names as the stored form. A call that never returned yields (None, row-with-error,
    "client_error"): the item is dropped and the read is counted as incomplete, but it is
    NOT an alignment failure, because a transport failure carries no information about
    where a logprob was read.
    """
    letters = list(item.labels)
    try:
        got = client.forced_answer_logprobs(prompt, letters)
    except Exception as exc:  # noqa: BLE001 - any server-side failure is one dropped item
        row = {
            "arm": arm,
            "completed": False,
            "n_letters_requested": len(letters),
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
        return None, row, "client_error"
    block = letter_logprob_fields(got.logprobs, got.tokens, target_letter=target)
    block["method"] = got.method
    row = check_row(arm, letters, got, target)
    row["completed"] = True
    reason = _read_failure_reason(row, block)
    row["failure_reason"] = reason
    return block, row, reason


# --------------------------------------------------------------------------- #
# One item: both arms, or nothing.
# --------------------------------------------------------------------------- #
def score_item(client, arms, rec: dict, index: int, expected_family: str) -> dict:
    """Read both arms of one record. A failure on either arm drops the ITEM, with a reason.

    The item is the unit because the fit's two rows are the two arms of one item: banking a
    clean margin whose hinted partner failed would put an item into the analysis on one arm
    only, which is not a row `build_table` knows how to drop later.
    """
    out: dict = {
        "record_index": index,
        "record_key": record_key(rec),
        "status": "dropped",
        "reason": None,
        "reads": [],
        "prompt_sha256": {},
    }
    hint = rec.get("hint_label")
    try:
        item = item_from_record(rec)
    except ValueError as exc:
        out["reason"] = f"record_not_rebuildable: {exc}"
        return out
    if not isinstance(hint, str) or hint not in item.labels:
        out["reason"] = "hint_label_not_in_item_labels"
        return out
    family, err = resolve_cue_family(rec)
    if err is not None:
        out["reason"] = f"cue_family_unresolved: {err}"
        return out
    if family != expected_family:
        out["reason"] = (
            f"cue_family_mismatch: the record resolves to {family or 'stated-hint'!r} and "
            f"the cell says {expected_family or 'stated-hint'!r}"
        )
        return out
    prompts = frame_prompts(arms, rec, item, family)
    out["cue_family"] = family or "stated-hint"
    out["prompt_sha256"] = {arm: _sha256_text(prompts[arm]) for arm in ARMS}
    if rec["cue_text"] not in prompts["hinted"]:
        out["reason"] = "hinted_prompt_does_not_contain_the_banked_cue_text"
        return out
    blocks = {}
    for arm in ARMS:
        block, row, reason = read_arm(client, prompts[arm], item, hint, arm)
        out["reads"].append(row)
        if reason is not None:
            out["reason"] = reason
            return out
        blocks[arm] = block
    out["status"] = "scored"
    out["target_letter"] = hint
    out["labels"] = list(item.labels)
    for arm in ARMS:
        out[arm] = blocks[arm]
    return out


# --------------------------------------------------------------------------- #
# The artifacts.
# --------------------------------------------------------------------------- #
def sidecar_record(rec: dict, res: dict, keys: dict) -> dict:
    """The AUGMENTED record: the banked record plus the two blocks and the scale fields.

    Every field of the original record survives, so the sidecar is a drop-in for the
    loaders that read the arms transcripts today (M, X, the curves and the anchor are all
    still there). What is added is the two logprob blocks, the CONTRACT scale fields at the
    logit level, and a ``logit_pass`` block that keys this record to the one it came from.
    """
    out = dict(rec)
    out["intervention_level"] = INTERVENTION_LEVEL
    out["outcome_scale"] = OUTCOME_SCALE
    out[BLOCK_KEY["clean"]] = res["clean"]
    out[BLOCK_KEY["hinted"]] = res["hinted"]
    # The record-level CONTRACT fields, which are null on a text-level record. Both arms
    # are carried under their arm name: one record now holds two reads, and a flat map
    # would silently be one of them.
    out["answer_logprobs"] = {arm: res[arm]["answer_logprobs"] for arm in ARMS}
    out["logprob_source_token"] = {arm: res[arm]["logprob_source_token"] for arm in ARMS}
    out["logit_pass"] = {
        "pass_version": PASS_VERSION,
        "record_index": res["record_index"],
        "record_key": res["record_key"],
        "source_file": keys["source_file"],
        "source_sha256": keys["source_sha256"],
        "model": keys["model"],
        "hf_revision": keys["hf_revision"],
        "endpoint": keys["endpoint"],
        "cue_family": res.get("cue_family"),
        "target_letter": res.get("target_letter"),
        "labels": res.get("labels"),
        "prompt_sha256": res["prompt_sha256"],
    }
    return out


def build_check(results: list[dict], meta: dict) -> dict:
    """The section 9.1 unit check on this pass's own reads, in the phase-1 artifact shape.

    ``results`` holds the reads that RETURNED, one row each, so the arithmetic a reader
    runs over it ("every letter scored, every logprob read off a token that decodes to its
    own letter") is a statement about reads that happened. Reads that never returned are
    counted separately under ``incomplete_reads``: they drop their item, and they are not
    evidence about where a logprob was read, so they do not fail the family.
    """
    rows, incomplete, failures = [], [], Counter()
    for res in results:
        for row in res["reads"]:
            tagged = dict(row, record_index=res["record_index"], record_key=res["record_key"])
            if row.get("completed"):
                rows.append(tagged)
                if row.get("failure_reason") in ALIGNMENT_REASONS:
                    failures[row["failure_reason"]] += 1
            else:
                incomplete.append(tagged)
    scored = sum(r["n_letters_scored"] for r in rows)
    requested = sum(r["n_letters_requested"] for r in rows)
    matching = sum(r["n_tokens_matching_letter"] for r in rows)
    mass = [r["probability_mass"] for r in rows]
    hard_failures = [f"{k} on {v} read(s)" for k, v in sorted(failures.items())]
    return {
        **meta,
        "scope": (
            "the section 9.1 unit check computed on THIS pass's own reads: every read is "
            "one arm of one banked record, on the prompt that record was generated "
            "against, scored over that item's own answer letters"
        ),
        "passed_rule": (
            "passed is true when no COMPLETED read is an alignment failure (every letter "
            "scored, every logprob read off a token that decodes to its own letter, every "
            "logprob at or below zero, letter mass at or below 1) and at least one read "
            "completed. A read that never returned drops its item and is counted under "
            "incomplete_reads; it is a transport failure and says nothing about where a "
            "logprob was read, so it does not fail the family"
        ),
        "n_probes": len(rows) + len(incomplete),
        "n_probes_completed": len(rows),
        "n_letters_requested": requested,
        "n_letters_scored": scored,
        "n_tokens_matching_letter": matching,
        "n_argmax_is_target": sum(1 for r in rows if r["argmax_is_target"]),
        "argmax_note": (
            "reported, never gating. The target letter is a planted WRONG option, so it "
            "is not expected to be the argmax and a low count is not a failure"
        ),
        "letter_probability_mass": _summary_stats(mass),
        "n_reads_with_mass_below_0.01": sum(1 for v in mass if v < 0.01),
        "mass_note": (
            "A5.4 condition 6 and part 4.4: a margin computed where the letters hold a "
            "tiny share of the next-token mass is well defined and is also about a region "
            "the model almost never enters, so the summary prints with the row"
        ),
        "hard_failures": hard_failures,
        "passed": not hard_failures and bool(rows),
        "incomplete_reads": incomplete[:50],
        "n_incomplete_reads": len(incomplete),
        "results": rows,
    }


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "median": float(median),
        "max": float(ordered[-1]),
    }


def _margin_stats(records: list[dict], key: str) -> dict:
    values = [r[key]["logprob_margin"] for r in records if r.get(key)]
    if not values:
        return {"n": 0, "mean": None, "variance": None}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"n": len(values), "mean": mean, "variance": var}


def build_meta(state: dict, check: dict, sidecar: list[dict]) -> dict:
    """`logit_pass_meta.json`: what ran, what it read, what it dropped, and by how much."""
    keys = state["keys"]
    clean = _margin_stats(sidecar, BLOCK_KEY["clean"])
    hinted = _margin_stats(sidecar, BLOCK_KEY["hinted"])
    arm_difference = None
    if clean["mean"] is not None and hinted["mean"] is not None:
        arm_difference = hinted["mean"] - clean["mean"]
    return {
        "pass_version": PASS_VERSION,
        "intervention_level": INTERVENTION_LEVEL,
        "outcome_scale": OUTCOME_SCALE,
        "spec": "docs/OUTCOME-SCALE-NOTE.md part 4.5 job B; prereg Amendment A5",
        "cell_dir": str(state["cell_dir"]),
        "out_dir": str(state["out_dir"]),
        "source": {
            "file": keys["source_file"],
            "sha256": keys["source_sha256"],
            "n_records": state["n_source_records"],
        },
        "sidecar": {"file": state["sidecar_path"].name, "n_records": len(sidecar)},
        "model": keys["model"],
        "hf_revision": keys["hf_revision"],
        "endpoint": keys["endpoint"],
        "endpoint_rule": (
            "section 6.7: SoCLaaS is INELIGIBLE as a source of logprob outcomes, so this "
            "pass refuses any endpoint that is not the loopback address of a server "
            "started by the job that runs it"
        ),
        "seed": state["seed"],
        "chat_template_kwargs": state["template_kwargs"],
        "logprob_mode": state["logprob_mode"],
        "concurrency": state["concurrency"],
        "cell": {
            "substrate": state["cell_meta"].get("substrate"),
            "cue_family": state["cell_meta"].get("cue_family"),
            "job_id": state["cell_meta"].get("job_id"),
            "text_level_intervention_level": state["cell_meta"].get("intervention_level"),
            "text_level_outcome_scale": state["cell_meta"].get("outcome_scale"),
        },
        "n_items_entered": len(state["results"]),
        "n_items_scored": len(sidecar),
        "n_items_dropped": len(state["results"]) - len(sidecar),
        "drops_by_reason": dict(sorted(state["drops"].items())),
        "n_reads_attempted": check["n_probes"],
        "n_reads_completed": check["n_probes_completed"],
        "family_unit_check": {
            "artifact": "logit_check.json",
            "passed": check["passed"],
            "hard_failures": check["hard_failures"],
            "rule": (
                "A5.4 condition 1 and section 9.1. A family that fails is reported at the "
                "TEXT level only, so this pass exits 7 and the sbatch stops that model"
            ),
        },
        "letter_probability_mass": check["letter_probability_mass"],
        "clean_arm_margin": clean,
        "hinted_arm_margin": hinted,
        "randomized_arm_difference_in_the_margin": arm_difference,
        "arm_difference_note": (
            "A5.4 condition 5 checks TE_logit against this number to within 1e-6. It is "
            "printed here as the input to that check and is not itself an estimate"
        ),
        "assert_records_scaled_checked": state["asserted"],
        "started_at": state["started_at"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def flush(state: dict) -> list[dict]:
    """Assert the batch, then write the sidecar, the check, the meta and the checkpoint.

    The order is the assertion first, which is section 9.6's rule as `write_arm_transcripts`
    already applies it on the text level: a mislabelled record would be pooled across
    intervention levels downstream, and banking it is worse than stopping. Nothing here
    writes the source file; the path collision was refused at startup.
    """
    results = sorted(state["results"].values(), key=lambda r: r["record_index"])
    scored = [r for r in results if r["status"] == "scored"]
    sidecar = [
        sidecar_record(state["source"][r["record_index"]], r, state["keys"]) for r in scored
    ]
    state["asserted"] = assert_records_scaled(sidecar)
    check = build_check(results, state["check_meta"])
    state["check"] = check
    state["sidecar_path"].write_text(json.dumps(sidecar, indent=2))
    (state["out_dir"] / "logit_check.json").write_text(json.dumps(check, indent=2))
    meta = build_meta(state, check, sidecar)
    (state["out_dir"] / "logit_pass_meta.json").write_text(json.dumps(meta, indent=2))
    (state["out_dir"] / "logit_pass_checkpoint.json").write_text(
        json.dumps(
            {
                "pass_version": PASS_VERSION,
                "source_file": state["keys"]["source_file"],
                "source_sha256": state["keys"]["source_sha256"],
                "model": state["keys"]["model"],
                "endpoint": state["keys"]["endpoint"],
                "results": state["results"],
            },
            indent=2,
        )
    )
    return sidecar


def load_checkpoint(state: dict, restart: bool) -> int:
    """Restore a previous run's per-item results. Returns how many were restored."""
    path = state["out_dir"] / "logit_pass_checkpoint.json"
    if restart or not path.exists():
        return 0
    banked = json.loads(path.read_text())
    if banked.get("source_sha256") != state["keys"]["source_sha256"]:
        raise LogitPassError(
            f"{path} was written against transcripts with sha256 "
            f"{banked.get('source_sha256')} and this run reads "
            f"{state['keys']['source_sha256']}. The records changed under the pass; "
            "resume would mix two source files. Pass --restart to start over."
        )
    restored = dict(banked.get("results") or {})
    # The sha above says the FILE is the same one. This says each restored entry still
    # names the record it was computed from: the entry's own record_key is recomputed from
    # the source record at its own index, and a disagreement refuses rather than resumes.
    for slot, entry in restored.items():
        index = entry.get("record_index")
        if not isinstance(index, int) or not 0 <= index < len(state["source"]):
            raise LogitPassError(f"{path} slot {slot} names record_index {index!r}")
        want = record_key(state["source"][index])
        if entry.get("record_key") != want:
            raise LogitPassError(
                f"{path} slot {slot} carries record_key {entry.get('record_key')!r} and "
                f"record {index} of {state['keys']['source_file']} keys to {want!r}. "
                "Resuming would attach a read to a record it was not taken on."
            )
    state["results"] = restored
    return len(restored)


# --------------------------------------------------------------------------- #
# The ordered pass.
# --------------------------------------------------------------------------- #
def map_in_order(seq, worker, *, concurrency: int, consume) -> None:
    """``worker(i, elem)`` in parallel, ``consume(i, elem, result)`` strictly in order.

    The same contract as `experiments/08_additive_arms.py::map_in_order`, and it exists here
    for the same reason: every checkpoint write happens inside ``consume``, so a checkpoint
    always holds a PREFIX of the results and never a hole with a later item filled past it.
    ``consume`` returning False cancels what has not started.
    """
    if concurrency <= 1:
        for i, elem in enumerate(seq):
            if not consume(i, elem, worker(i, elem)):
                return
        return
    pending: deque = deque()
    source = enumerate(seq)

    def _fill(pool) -> None:
        while len(pending) < concurrency:
            try:
                i, elem = next(source)
            except StopIteration:
                return
            pending.append((i, elem, pool.submit(worker, i, elem)))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        _fill(pool)
        while pending:
            i, elem, fut = pending.popleft()
            result = fut.result()
            _fill(pool)
            if not consume(i, elem, result):
                for _, _, queued in pending:
                    queued.cancel()
                pending.clear()
                return


def run_pass(state: dict, client, arms) -> None:
    """Score every not-yet-banked record, checkpointing a prefix as it goes."""
    expected = state["expected_family"]
    todo = [
        (i, rec) for i, rec in enumerate(state["source"]) if str(i) not in state["results"]
    ]
    if state["limit"]:
        todo = todo[: state["limit"]]
    print(f"[logit] {len(todo)} record(s) to score, {len(state['results'])} already banked")

    def _work(_n, pair):
        i, rec = pair
        return score_item(client, arms, rec, i, expected)

    def _consume(n, _pair, res) -> bool:
        # Keyed by the record's POSITION in the source array, so two byte-identical
        # records cannot land in one slot, and carrying record_key inside the entry is
        # what load_checkpoint re-verifies on the next leg.
        state["results"][str(res["record_index"])] = res
        if res["status"] != "scored":
            state["drops"][res["reason"] or "unknown"] += 1
        if any(r.get("completed") for r in res["reads"]):
            state["consecutive_errors"] = 0
        elif res["reason"] == "client_error":
            state["consecutive_errors"] += 1
        if state["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
            flush(state)
            state["server_gone"] = True
            print(
                f"[logit] STOPPING: {state['consecutive_errors']} consecutive reads failed "
                "on the server. The checkpoint holds everything scored so far."
            )
            return False
        if (n + 1) % CHECKPOINT_EVERY == 0:
            flush(state)
            print(f"      ... {n + 1}/{len(todo)} scored", end="\r", flush=True)
        return True

    map_in_order(todo, _work, concurrency=state["concurrency"], consume=_consume)


# --------------------------------------------------------------------------- #
# Startup checks and CLI.
# --------------------------------------------------------------------------- #
def assert_self_hosted(endpoint: str) -> None:
    """Refuse anything but a loopback endpoint. Section 6.7, enforced rather than written."""
    host = (urlparse(endpoint).hostname or "").lower()
    if host in LOOPBACK_HOSTS:
        return
    raise LogitPassError(
        f"REFUSING: --endpoint {endpoint} resolves to host {host or '<none>'}, which is not "
        "the loopback address of a server this job started. Section 6.7 makes SoCLaaS "
        "INELIGIBLE as a source of logprob outcomes and A5.4 condition 2 requires the "
        "pinned self-hosted vLLM endpoint of element 8. No flag turns this off."
    )


def resolve_source(cell_dir: Path, model: str, explicit: Path | None) -> Path:
    """The arms transcripts file this cell banked, for this model."""
    if explicit is not None:
        if not explicit.is_file():
            raise LogitPassError(f"no such transcripts file: {explicit}")
        return explicit
    safe = model.replace(":", "_").replace("/", "_")
    named = cell_dir / f"arms_transcripts_{safe}.json"
    if named.is_file():
        return named
    found = sorted(
        p for p in cell_dir.glob("arms_transcripts_*.json") if not p.name.endswith(".logit.json")
    )
    if len(found) != 1:
        raise LogitPassError(
            f"expected {named.name} in {cell_dir}, and {len(found)} arms_transcripts_*.json "
            "file(s) are there instead; name the file with --transcripts"
        )
    return found[0]


def expected_family_for(cell_meta: dict, cell_dir: Path) -> str:
    """The cue family this cell says it ran, in ``_frame_prompt``'s vocabulary."""
    cue = cell_meta.get("cue_family") or cell_dir.name
    if cue == "stated-hint":
        return ""
    if cue in _TAXONOMY_TEMPLATES:
        return cue
    raise LogitPassError(
        f"REFUSING: this cell names cue family {cue!r}, which is neither stated-hint nor "
        f"one of the frozen taxonomy families {sorted(_TAXONOMY_TEMPLATES)}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cell-dir", type=Path, required=True,
                   help="the cell directory holding arms_transcripts_<model>.json")
    p.add_argument("--model", required=True, help="the served model name, e.g. Qwen/Qwen3-8B")
    p.add_argument("--endpoint", required=True,
                   help="the pinned self-hosted vLLM base url, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--out", type=Path, default=None,
                   help="where the sidecar and the two artifacts go (default: --cell-dir)")
    p.add_argument("--transcripts", type=Path, default=None,
                   help="the input transcripts file, when it is not named after --model")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--chat-template-kwargs", default=DEFAULT_TEMPLATE_KWARGS,
                   help="JSON forwarded to the server's chat template; MUST match the run "
                        "being augmented or the letter sits after different tokens")
    p.add_argument("--logprob-mode", default="prompt_logprobs",
                   choices=["prompt_logprobs", "top_logprobs", "auto"],
                   help="pinned to prompt_logprobs: it is the section 9.1 path and it is "
                        "exact even when the letter is not the model's first token choice")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--limit", type=int, default=0, help="score at most this many records")
    p.add_argument("--restart", action="store_true",
                   help="ignore any checkpoint in --out and score every record again")
    p.add_argument("--timeout", type=float, default=600.0)
    return p


def _client_for(args, template_kwargs) -> OpenAIClient:
    return OpenAIClient(
        base_url=args.endpoint,
        model=args.model,
        seed=args.seed,
        temperature=0.0,
        logprob_mode=args.logprob_mode,
        chat_template_kwargs=template_kwargs,
        timeout=args.timeout,
        max_in_flight=max(args.concurrency, 1),
        request_log=None,
    )


def build_state(args, source: Path, cell_meta: dict, template_kwargs, vllm_version: str) -> dict:
    """Everything the pass and its artifacts read, assembled once."""
    out_dir = Path(args.out) if args.out else Path(args.cell_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_dir / f"{source.stem}.logit.json"
    if sidecar_path.resolve() == source.resolve():
        raise LogitPassError(
            f"REFUSING: the sidecar path and the source transcripts are the same file "
            f"({source}). This pass never rewrites the transcripts it read."
        )
    records = json.loads(source.read_text())
    if not isinstance(records, list):
        raise LogitPassError(f"{source} does not hold a list of records")
    return {
        "cell_dir": Path(args.cell_dir),
        "out_dir": out_dir,
        "sidecar_path": sidecar_path,
        "source": records,
        "n_source_records": len(records),
        "cell_meta": cell_meta,
        "expected_family": expected_family_for(cell_meta, Path(args.cell_dir)),
        "results": {},
        "drops": Counter(),
        "consecutive_errors": 0,
        "server_gone": False,
        "asserted": 0,
        "check": None,
        "limit": args.limit,
        "seed": args.seed,
        "template_kwargs": template_kwargs,
        "logprob_mode": args.logprob_mode,
        "concurrency": args.concurrency,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "keys": {
            "source_file": source.name,
            "source_sha256": _sha256_file(source),
            "model": args.model,
            "hf_revision": cell_meta.get("hf_revision"),
            "endpoint": args.endpoint,
        },
        "check_meta": {
            "base_url": args.endpoint,
            "model": args.model,
            "seed": args.seed,
            "chat_template_kwargs": template_kwargs,
            "vllm_version": vllm_version,
            "logprob_mode": args.logprob_mode,
            "cell_dir": str(Path(args.cell_dir)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        assert_self_hosted(args.endpoint)
    except LogitPassError as exc:
        print(f"[logit] {exc}")
        return 3
    cell_dir = Path(args.cell_dir)
    if not cell_dir.is_dir():
        print(f"[logit] no such cell directory: {cell_dir}")
        return 2
    template_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    meta_path = cell_dir / "run_meta.json"
    cell_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    try:
        source = resolve_source(cell_dir, args.model, args.transcripts)
    except LogitPassError as exc:
        print(f"[logit] {exc}")
        return 2

    client = _client_for(args, template_kwargs)
    if not client.is_available():
        print(f"[logit] no OpenAI-compatible server answered at {args.endpoint}")
        return 4
    try:
        state = build_state(args, source, cell_meta, template_kwargs, client.server_version())
        restored = load_checkpoint(state, args.restart)
    except LogitPassError as exc:
        print(f"[logit] {exc}")
        return 5
    print(f"[logit] {source.name}: {state['n_source_records']} record(s), "
          f"{restored} restored from a checkpoint")
    print(f"[logit] cue family {state['expected_family'] or 'stated-hint'}, "
          f"endpoint {args.endpoint}, mode {args.logprob_mode}")

    arms = load_arms_module()
    try:
        run_pass(state, client, arms)
        sidecar = flush(state)
    except OutcomeScaleError as exc:
        print(f"[logit] REFUSED before the checkpoint write: {exc}")
        return 8
    check = state["check"]
    print(f"[logit] scored {len(sidecar)}/{len(state['results'])} item(s); "
          f"dropped {dict(state['drops'])}")
    print(f"[logit] unit check: {check['n_letters_scored']}/{check['n_letters_requested']} "
          f"letters scored, {check['n_tokens_matching_letter']}/{check['n_letters_scored']} "
          f"tokens matching their letter, {check['n_incomplete_reads']} read(s) incomplete, "
          f"passed={check['passed']}")
    print(f"[logit] wrote {state['sidecar_path']}")
    if state["server_gone"]:
        return 9
    if not sidecar:
        print("[logit] nothing was scored, so no logit-level row exists for this cell")
        return 6
    if not check["passed"]:
        print("[logit] FAMILY UNIT CHECK FAILED (A5.4 condition 1): "
              f"{check['hard_failures']}. This family is reported at the text level only.")
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
