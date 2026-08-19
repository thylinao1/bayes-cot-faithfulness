"""Checkpoint / resume for the additive-arms runner (experiments/08_additive_arms.py).

WHY: a powered run makes ~2,000 free-tier calls and the daily token budget (500K)
can end it mid-flight. Without resume, a rerun restarts from item 1 and re-spends the
whole budget. This module serializes the FULL run state (every substrate record,
including clean-INCORRECT ones, plus the held-out A9 specificity state) to an
internal checkpoint file and rehydrates it, so a ``--resume`` run continues where the
stop happened, re-spending at most the work since the last checkpoint (written every
CHECKPOINT_EVERY items, every item for the truncation curves, and before every
failure stop), typically only the in-flight item's calls on an API stop.

THE INVARIANT (stated honestly; both stronger claims that came before it were false):

    a resumed run is always a VALID run of the frozen design.

Concretely, and this is the whole of what is promised:

  - the published wrong-option rotation is a correct ``wrong_label(rotate=i)`` cycle
    over the FINAL clean-correct roster, and
  - attrition reports every item that entered but produced no record.

It is NOT "resumed == an uninterrupted failure-free run", and it is not even "same call
outcomes => same result". Where no position-seeded draw has been banked yet, a resume
RETRIES transient holes, so the resumed run can end up with MORE coverage than the
uninterrupted counterfactual would have had (a harmless direction: more measured
items, though not identical). Once such draws exist the roster is locked and reproduces
exactly. See THE ROSTER LOCK below.

The checkpoint is a SEPARATE internal file (``arms_checkpoint_<model>.json``), never the
published ``arms_transcripts_*.json``; its format may change freely. It lives under
``experiments/results/`` (gitignored), so it never enters version control.

Constraints that shape this module:

  - Item identity is ``(question, tuple(choices))``. A resumed record is matched to a
    freshly loaded item on that key, and the merged record list keeps ``--data`` fetch
    ORDER, because the cue pass, the placebo arm, and the specificity placebo pass seed
    ``wrong_label(rotate=i)`` / ``rng_seed=i`` on the record's position in the FULL
    clean-correct list (the determinism invariant). A banked record therefore restores
    its own ``hint_label`` / ``cue_text`` rather than recomputing them. Because the
    merge is by key, a data file with DUPLICATE keys would alias one banked record
    across positions, so a resume refuses it up front.

  - THE ROSTER LOCK, DERIVED (:func:`roster_locked`). The roster is locked exactly when
    some banked record already carries a position-seeded draw, read off the disk, not
    tracked as a flag. Unlocked, an item with no banked record is simply attempted again
    on resume, so a daily cap that stops the run DURING a clean pass costs nothing.
    Locked, such an item is skipped ENTIRELY: healing a mid-roster hole would shift every
    later record's position while the presence guards keep their banked, old-position
    ``hint_label``, silently breaking the frozen "planted wrong option cycles across
    wrong choices by item index" rule (observed: a rotation designated twice and another
    never used). A locked roster is exactly the earlier invocation's roster.

    Deriving beats flagging on two counts. A flag has to be SET at some moment, and the
    only moments available are wrong: "the clean pass returned ok" fires before cue_pass
    makes a single call, and substrate_pass returns ok whenever fewer than 3 items failed
    and the loop merely ENDS, so a cap landing on the last 1-2 items of a clean pass
    produced a "completed" pass full of un-retried holes that the flag then froze in
    place (measured: cap on item 6 of 8 permanently deleted 2 items a healthy resume
    could have regenerated). The derived condition reads the truth instead, so it is also
    correct after a hard SIGKILL that skipped every write, and there is no ordering race
    left to get wrong.

  - Attrition is DERIVED from final state (:func:`derive_attrition`), never carried as
    event counters, so a locked roster missing item j reports n_failed_generation=1
    honestly and stably across any number of resumes.

  - ANY parameter mismatch (model, backend, n_items, data + its sha256, taxonomy, the
    arms in their CLI order, curve_cap, num_predict, specificity_holdout + its sha256)
    or a checkpoint version mismatch refuses the resume with ZERO model calls: banked
    records were generated under the checkpoint's parameters and must not be mixed with
    a different design. The content hashes matter because the data files are gitignored
    local fetches; a re-fetch between two legs would otherwise silently merge banked
    records against a different item set.

  - Curves rehydrate into :class:`TruncationCurve` instances (the dict keys match
    ``_curve_to_dict``) so downstream attribute access (``curve_covariates``) is identical
    to the uninterrupted path.

  - Records are serialized with only the arm fields actually PRESENT on them, so a
    missing output key round-trips as absent and the runner's per-call skip conditions
    ("redo only the calls whose output is not yet banked") stay exact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from bayes_cot_faithfulness.curves import TruncationCurve
from bayes_cot_faithfulness.interventions import CHOICE_LABELS, QAItem

# Version 1. The schema changed repeatedly during pre-ship review (the failed-item
# permanent-skip keys and the carried attrition counters were removed; the specificity
# block reduced to {n_holdout_entered, records}; the data content hashes and
# n_invocations were added; the substrate_locked / holdout_locked flags were added and
# then removed again in favour of the DERIVED roster lock) WITHOUT a version bump: no
# checkpoint had ever been produced outside tests, so there was no on-disk state to
# migrate.
CHECKPOINT_VERSION = 1

# The parameter fields whose equality gates a resume. Banked records depend on every one
# of these; a mismatch means the checkpoint describes a different experiment. The two
# sha256 fields pin file CONTENT, not just the path: both data files are gitignored
# local fetches that can change underneath a multi-leg run.
PARAM_FIELDS = (
    "model", "backend", "n_items", "data", "data_sha256", "taxonomy", "arms",
    "curve_cap", "num_predict", "specificity_holdout", "specificity_holdout_sha256",
)

# Fields always present on a substrate record (set in substrate_pass before any arm runs).
_ALWAYS_FIELDS = (
    "clean_answer", "clean_cot", "clean_correct",
)
# Optional per-arm scalar outputs. Serialized only when present so absence round-trips as
# "this call has not run yet", the exact signal the runner's skip conditions read.
_OPTIONAL_SCALAR_FIELDS = (
    "hint_label", "cue_text", "cue_prepended", "hinted_answer", "hinted_cot",
    "followed", "acknowledged", "silent",
    "replay_clean_answer", "replay_hinted_answer", "placebo_answer",
    "direct_answer", "pre_cot_committed", "twostep_answer", "filler_answer",
    "transplant_forward_answer", "transplant_reverse_answer",
)
_CURVE_FIELDS = ("clean_curve", "hinted_curve")

# Specificity (A9) holdout record fields. The four false-alarm flags and the placebo
# transcript are written together in one r.update, so any one present means all present.
_SPEC_ALWAYS_FIELDS = ("clean_answer", "clean_cot", "clean_correct")
_SPEC_OPTIONAL_FIELDS = (
    "hint_label", "cue_text", "placebo_cot", "placebo_answer",
    "ack_clean", "ack_placebo", "would_be_follow", "silent_false_alarm",
)

NO_CHECKPOINT_NOTE = (
    "\n[resume] No checkpoint file was found, so this is a FRESH run (a checkpoint will "
    "be written as it progresses).\n"
)


# --- Derived accounting -------------------------------------------------------
def derive_attrition(n_entered: int, records: list) -> dict:
    """Substrate attrition derived from final state, never from carried counters.

    With retry-on-resume semantics an item's terminal state is either a record or an
    absence: there are no failure EVENTS to carry across invocations, so deriving
    the counters from state keeps the accounting idempotent over any number of
    interruptions: ``n_failed_generation`` is the items entered minus the records
    produced, and ``n_unparseable_clean`` the records whose clean answer never parsed.
    """
    return {
        "n_entered": n_entered,
        "n_failed_generation": n_entered - len(records),
        "n_unparseable_clean": sum(1 for r in records if r.get("clean_answer") is None),
    }


# --- Parameter fingerprint ----------------------------------------------------
def file_sha256(path: Path) -> str | None:
    """sha256 of a file's bytes, or None when it does not exist.

    None (rather than an error) for a missing file so a run that does not enable the A9
    arm still fingerprints cleanly; a file appearing or changing between two legs then
    shows up as a mismatch, which is exactly the intent.
    """
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_params(model: str, backend: str, n_items: int, data: Path, taxonomy: str | None,
                 arms: list[str], curve_cap: int, num_predict: int,
                 specificity_holdout: Path, data_sha256: str | None = None,
                 specificity_holdout_sha256: str | None = None) -> dict:
    """The parameter fingerprint stored in (and checked against) a checkpoint.

    ``arms`` is kept in CLI order exactly as ``resolve_arms`` returns it: the enabled-arms
    ORDER shapes the summary's key order and the arm execution order, so a reordered
    --arm list is a different (non-byte-identical) run and must refuse, not merge.
    Paths are stringified so the payload is JSON-safe and path-comparable, and each is
    paired with its content hash so a re-fetched (gitignored) data file cannot silently
    merge banked records against a different item set.
    """
    return {
        "model": model,
        "backend": backend,
        "n_items": n_items,
        "data": str(data),
        "data_sha256": data_sha256,
        "taxonomy": taxonomy,
        "arms": list(arms),
        "curve_cap": curve_cap,
        "num_predict": num_predict,
        "specificity_holdout": str(specificity_holdout),
        "specificity_holdout_sha256": specificity_holdout_sha256,
    }


def params_mismatch(saved: dict, current: dict) -> list[tuple[str, object, object]]:
    """Return the (field, saved, current) triples that differ, empty when every field matches."""
    out: list[tuple[str, object, object]] = []
    for field in PARAM_FIELDS:
        if saved.get(field) != current.get(field):
            out.append((field, saved.get(field), current.get(field)))
    return out


def refusal_message(mismatches: list[tuple[str, object, object]], path: Path) -> str:
    """The clear refusal printed when --resume meets a checkpoint from a different run."""
    rows = "\n".join(
        f"    - {field}: checkpoint={saved!r} now={current!r}"
        for field, saved, current in mismatches
    )
    return (
        "\n[resume] REFUSED: the run parameters differ from the checkpoint at\n"
        f"  {path}\n"
        "  Banked records were generated under the checkpoint's parameters and cannot be\n"
        "  mixed with a different design, so no model call was made. Mismatched fields:\n"
        f"{rows}\n"
        "  Re-run WITHOUT --resume to start fresh (overwriting the checkpoint), or restore\n"
        "  the original parameters to continue where the stop happened.\n"
    )


def version_mismatch(loaded: dict) -> int | None:
    """The checkpoint's version when it is not the one this code writes, else None."""
    found = loaded.get("version")
    return None if found == CHECKPOINT_VERSION else found


def version_refusal_message(found: object, path: Path) -> str:
    """Refusal for a checkpoint this code cannot safely interpret.

    A payload from another schema version may name the same keys with different
    meanings, so merging it could corrupt the run silently rather than loudly.
    """
    return (
        "\n[resume] REFUSED: the checkpoint at\n"
        f"  {path}\n"
        f"  reports version {found!r}, but this runner writes and reads version "
        f"{CHECKPOINT_VERSION}.\n"
        "  A payload from another schema version cannot be interpreted safely, so no\n"
        "  model call was made. Re-run WITHOUT --resume to start fresh.\n"
    )


# --- Identity -----------------------------------------------------------------
def item_key(item: QAItem) -> tuple[str, tuple[str, ...]]:
    """The identity key matching a checkpoint record to a freshly loaded item."""
    return (item.question, tuple(item.choices))


def duplicate_item_keys(items: list[QAItem]) -> list[tuple[str, tuple[str, ...]]]:
    """Item keys appearing more than once, in first-seen order.

    The resume merge is by key, so a duplicate key would alias ONE banked record across
    every duplicate position, silently collapsing distinct positions (and their
    position-seeded cue / placebo draws) onto the same banked outputs. A resume refuses
    such a data file before any model call; a fresh run is unaffected.
    """
    counts: dict[tuple[str, tuple[str, ...]], int] = {}
    for it in items:
        key = item_key(it)
        counts[key] = counts.get(key, 0) + 1
    return [key for key, n in counts.items() if n > 1]


def duplicate_refusal_message(dupes: list[tuple[str, tuple[str, ...]]], label: str,
                              path: Path) -> str:
    """Refusal naming WHICH file carries the duplicates (--data or the A9 holdout).

    Both files feed the identical by-key merge, so both need the guard and the caller
    needs to know which one to de-duplicate.
    """
    rows = "\n".join(f"    - {question!r}" for question, _choices in dupes)
    return (
        f"\n[resume] REFUSED: the {label} file holds duplicate (question, choices) items,\n"
        "  which the resume merge would alias onto a single banked record:\n"
        f"  {path}\n"
        f"{rows}\n"
        f"  De-duplicate the {label} file, or re-run WITHOUT --resume.\n"
        "  No model call was made.\n"
    )


def _rebuild_item(d: dict) -> QAItem:
    """Reconstruct a QAItem from a serialized record via its answer LABEL.

    The checkpoint stores ``answer_label`` (not the index), so the index is recovered from
    the frozen ``CHOICE_LABELS`` ordering; QAItem is rebuilt through its own constructor so
    its invariants are re-checked rather than bypassed.
    """
    return QAItem(d["question"], tuple(d["choices"]), CHOICE_LABELS.index(d["answer_label"]))


# --- Curve (de)serialization ----------------------------------------------------
def curve_from_dict(d: dict) -> TruncationCurve:
    """Rebuild a :class:`TruncationCurve` from its serialized dict (keys match _curve_to_dict).

    Constructing the dataclass directly (rather than re-deriving via summarize_curve) keeps
    the banked commitment_depth / curve_area / n_unscorable_depths exactly as scored on the
    original run, so a resumed curve is identical to the uninterrupted one.
    """
    return TruncationCurve(
        depths=tuple(d["depths"]),
        answers=tuple(d["answers"]),
        final_answer=d["final_answer"],
        match=tuple(d["match"]),
        commitment_depth=d["commitment_depth"],
        curve_area=d["curve_area"],
        n_unscorable_depths=d["n_unscorable_depths"],
    )


# --- Main-run record (de)serialization ------------------------------------------
def serialize_record(r: dict, curve_to_dict: Callable[[object], dict]) -> dict:
    """Flatten one substrate record (with any arm fields) into a JSON-safe checkpoint dict.

    Unlike the published ``serialize_arm_record``, arm fields are written ONLY when present,
    so a not-yet-run output key stays absent and the runner's skip conditions redo exactly
    the missing calls. ``clean_correct`` is carried (clean-INCORRECT records are banked too)
    so the resumed clean-correct filter reproduces the same subset in the same order.
    """
    it: QAItem = r["item"]
    out: dict = {
        "question": it.question,
        "choices": list(it.choices),
        "answer_label": it.answer_label,
    }
    for field in _ALWAYS_FIELDS:
        out[field] = r.get(field)
    for field in _OPTIONAL_SCALAR_FIELDS:
        if field in r:
            out[field] = r[field]
    for field in _CURVE_FIELDS:
        if field in r:
            out[field] = curve_to_dict(r[field])
    return out


def deserialize_record(d: dict) -> dict:
    """Rebuild a live substrate record from a serialized checkpoint dict.

    Only the keys actually present in ``d`` are set (curves rehydrated to TruncationCurve),
    so ``"key" in record`` means the same thing after a resume as it did mid-run.
    """
    r: dict = {"item": _rebuild_item(d), "answer_label": d["answer_label"]}
    for field in _ALWAYS_FIELDS:
        r[field] = d.get(field)
    for field in _OPTIONAL_SCALAR_FIELDS:
        if field in d:
            r[field] = d[field]
    for field in _CURVE_FIELDS:
        if field in d:
            r[field] = curve_from_dict(d[field])
    return r


# --- Specificity holdout record (de)serialization --------------------------------
def serialize_specificity_record(r: dict) -> dict:
    """Flatten one holdout specificity record (clean pass + optional placebo/flags)."""
    it: QAItem = r["item"]
    out: dict = {
        "question": it.question,
        "choices": list(it.choices),
        "answer_label": it.answer_label,
    }
    for field in _SPEC_ALWAYS_FIELDS:
        out[field] = r.get(field)
    for field in _SPEC_OPTIONAL_FIELDS:
        if field in r:
            out[field] = r[field]
    return out


def deserialize_specificity_record(d: dict) -> dict:
    """Rebuild a live holdout specificity record from its serialized dict."""
    r: dict = {"item": _rebuild_item(d), "answer_label": d["answer_label"]}
    for field in _SPEC_ALWAYS_FIELDS:
        r[field] = d.get(field)
    for field in _SPEC_OPTIONAL_FIELDS:
        if field in d:
            r[field] = d[field]
    return r


# --- Load + resume inputs ---------------------------------------------------------
class CheckpointUnreadable(Exception):
    """A checkpoint file exists but its bytes cannot be parsed (torn / truncated write)."""


def load_checkpoint(path: Path) -> dict | None:
    """Read the raw checkpoint payload, or None when there is no file to resume from.

    A file that exists but cannot be parsed raises :class:`CheckpointUnreadable` rather
    than a bare JSONDecodeError traceback, so the caller can route it to the same clean
    refusal every sibling gate (version, params, duplicates) already prints.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise CheckpointUnreadable(str(exc)) from exc


def unreadable_refusal_message(reason: object, path: Path) -> str:
    return (
        "\n[resume] REFUSED: the checkpoint at\n"
        f"  {path}\n"
        f"  could not be read: {reason}\n"
        "  A half-written or corrupted checkpoint cannot be merged safely, so no model\n"
        "  call was made. Re-run WITHOUT --resume to start fresh (which overwrites it).\n"
    )


def roster_locked(records: list[dict]) -> bool:
    """Whether any banked record already carries a POSITION-SEEDED draw.

    ``hint_label`` is the exact marker, on both rosters. ``cue_pass`` is the only writer
    of ``hint_label`` on a main record and the holdout placebo loop is the only writer of
    it on a holdout record; every other position-seeded draw (``arm_placebo`` /
    ``arm_filler``'s ``rng_seed=i``, the curves cap prefix) can only run AFTER that write.
    So ``hint_label`` present <=> some position-seeded draw was banked against this
    roster <=> the roster's positions are committed and a hole must not be healed.

    Reading this off the banked records rather than tracking a flag is what makes the
    lock correct after a hard kill that skipped every write, and removes the ordering
    race a flag has to get right (see THE ROSTER LOCK in the module docstring).
    """
    return any("hint_label" in r for r in records)


def resume_inputs(loaded: dict) -> dict:
    """Banked substrate records by item key (incl. clean-incorrect), for the merge."""
    banked: dict = {}
    for row in loaded["records"]:
        record = deserialize_record(row)
        banked[item_key(record["item"])] = record
    return banked


def specificity_rows(loaded: dict | None) -> list:
    """The A9 block's RAW serialized rows (what :func:`roster_locked` reads), or []."""
    return ((loaded or {}).get("specificity") or {}).get("records", [])


def resume_specificity(loaded: dict) -> dict | None:
    """Banked holdout records by item key, or None when A9 had not started at the stop.

    Includes clean-INCORRECT holdout records: the holdout resume reuses the same by-key
    substrate merge as the main run, and the clean-correct filter afterwards reproduces
    the placebo-loop positions exactly.
    """
    spec = (loaded or {}).get("specificity")
    if not spec:
        return None
    banked: dict = {}
    for row in spec["records"]:
        record = deserialize_specificity_record(row)
        banked[item_key(record["item"])] = record
    return banked


# --- Writer -------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON via a temp file + os.replace so a crash mid-write cannot corrupt the
    checkpoint that a later --resume depends on.

    The temp name carries the pid: two processes sharing an --out directory would
    otherwise race on one shared temp path, and the loser's os.replace would blow up
    with FileNotFoundError after the winner consumed it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _union_rows(banked_rows: list, live_rows: list) -> list:
    """Banked rows overlaid with live rows, keyed by item; live always wins.

    write() must be INCAPABLE of dropping a record that was already paid for. The live
    list holds only the merge's prefix while a resumed clean pass is still running, so
    serializing it alone would truncate the file: a cadence write at item i silently
    deleted every banked record at index > i (measured: a 14-record checkpoint collapsed
    to 10, losing 5 already-generated items). Record ORDER in the file is irrelevant --
    resume_inputs re-keys the rows into a dict and the live list is always rebuilt in
    item order from the data file, so the union is safe as well as sufficient.
    """
    merged: dict = {}
    for row in banked_rows:
        merged[(row["question"], tuple(row["choices"]))] = row
    for row in live_rows:
        merged[(row["question"], tuple(row["choices"]))] = row
    return list(merged.values())


class CheckpointWriter:
    """Serializes the FULL run state to the checkpoint file on demand.

    Holds a live reference to the substrate records list, constructed EMPTY before
    the substrate pass and filled in place as the run progresses, so every
    ``write()`` captures the current state, including a partially generated substrate,
    unioned with whatever the resumed checkpoint already held. ``loaded`` is the
    checkpoint being resumed from (None on a fresh run); until the specificity arm
    registers its live holdout state, writes carry the loaded checkpoint's banked
    specificity block forward VERBATIM, so a kill between the resume and the A9 arm
    cannot wipe holdout progress banked by an earlier invocation.

    There is no roster-lock flag to maintain: the lock is derived from the banked
    records at read time (:func:`roster_locked`).
    """

    def __init__(self, path: Path, params: dict, records: list, n_items_entered: int,
                 curve_to_dict: Callable[[object], dict], loaded: dict | None = None,
                 n_invocations: int = 1):
        self.path = path
        self.params = params
        self.records = records
        self.n_items_entered = n_items_entered
        self._curve_to_dict = curve_to_dict
        self.loaded = loaded
        self.n_invocations = n_invocations
        self._specificity: dict | None = None

    def set_specificity(self, records: list, n_holdout_entered: int) -> None:
        """Register the live holdout state so subsequent writes serialize it.

        Called BEFORE the holdout clean pass runs (with the still-empty live list the
        pass fills), so no write between registration and completion can drop state.
        """
        self._specificity = {
            "records": records,
            "n_holdout_entered": n_holdout_entered,
        }

    def _specificity_payload(self) -> dict | None:
        if self._specificity is None:
            # Not (yet) registered in THIS invocation: carry the banked block forward
            # verbatim rather than overwriting it with null (the resume kill window).
            return (self.loaded or {}).get("specificity")
        s = self._specificity
        return {
            "n_holdout_entered": s["n_holdout_entered"],
            "records": _union_rows(
                specificity_rows(self.loaded),
                [serialize_specificity_record(r) for r in s["records"]],
            ),
        }

    def write(self) -> None:
        if self.loaded is None and not self.records and self._specificity is None:
            # Nothing of our own to record yet, so writing would only DESTROY. A fresh
            # run legitimately overwrites a stale checkpoint, but only once it has
            # something to put there. Without this guard a leg that died before its
            # first successful generation still wrote {records: [], specificity: null}:
            # substrate_pass's `i == 0` abort banks on the way down (its whole purpose
            # is to SAVE work), and with loaded=None the union degenerates to the empty
            # live list. A relaunch WITHOUT --resume against a still-capped key thus
            # made zero successful calls and deleted the previous leg's banked records
            # and its A9 holdout block, the most expensive artifact in the run.
            return
        payload = {
            "version": CHECKPOINT_VERSION,
            "params": self.params,
            "n_items_entered": self.n_items_entered,
            "n_invocations": self.n_invocations,
            "records": _union_rows(
                (self.loaded or {}).get("records", []),
                [serialize_record(r, self._curve_to_dict) for r in self.records],
            ),
            "specificity": self._specificity_payload(),
        }
        _atomic_write_json(self.path, payload)
