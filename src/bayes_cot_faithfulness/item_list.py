"""Ruling R3(ii)/(iii): naming an explicit subset of a substrate pool by position.

The enrichment pass (``bcf/enrich_report.py``) writes a list of the right-but-uncertain
items a model's sampling arm found in a substrate's 1,500-item pool. The enrichment
cell then runs the hinted arms on EXACTLY those items. This module is the piece in
between: it defines the list file, resolves it against a pool, and refuses rather than
guesses.

Why POSITION plus a question hash, and not either alone. The pools carry no id field of
their own (``question``, ``choices``, ``answer_index`` and nothing else), so a positional
index is the only cheap selector; but an index alone silently selects different items
when a pool is refetched, which is precisely the failure the pool manifest exists to
catch elsewhere. So every entry carries both, and a mismatch between them stops the run
with the offending index named. The whole-file ``pool_sha256`` is recorded too and is
checked when the caller supplies the pool's own hash, which makes a refetched pool a
refusal at the file level rather than 47 separate index refusals.

Everything here is arithmetic on lists and dicts; no model call and no file I/O beyond
what the caller hands in, so the resolution rules are unit-testable offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

SCHEMA = "bcf.item_list.v1"

# The selector the enrichment pass writes. Recorded in the file so a list built for some
# other purpose cannot be read as an enrichment list by accident.
SELECTOR_RIGHT_BUT_UNCERTAIN = "right_but_uncertain"


class ItemListError(ValueError):
    """A list file that cannot be resolved against the pool it names."""


def question_sha16(question: str) -> str:
    """The first 16 hex characters of sha256 over the question text.

    Sixteen hex characters is 64 bits, which is far past collision risk for a 1,500-item
    pool and short enough to read in a diff. The question alone (not the choices) is
    hashed because that is what a human checks by eye when a refusal names an index.
    """
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def normalize_entry(entry: object, position: int) -> dict:
    """One list entry as ``{"index": int, "question_sha16": str | None}``.

    A bare integer is accepted (a hand-written list of positions) and carries no hash,
    which means it is resolved WITHOUT the identity check. That is deliberate and
    visible: :func:`select_items` reports how many entries were unverified so a caller
    can refuse a list that would select by position alone.
    """
    if isinstance(entry, bool):  # bool is an int subclass; never a valid index
        raise ItemListError(f"entry {position} is a boolean, not an item index")
    if isinstance(entry, int):
        return {"index": entry, "question_sha16": None}
    if isinstance(entry, dict):
        if "index" not in entry:
            raise ItemListError(f"entry {position} has no 'index' field: {entry!r}")
        idx = entry["index"]
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ItemListError(f"entry {position} has a non-integer index: {idx!r}")
        sha = entry.get("question_sha16")
        if sha is not None and not isinstance(sha, str):
            raise ItemListError(f"entry {position} has a non-string question_sha16: {sha!r}")
        return {"index": idx, "question_sha16": sha}
    raise ItemListError(f"entry {position} is neither an integer nor an object: {entry!r}")


def parse_item_list(payload: object) -> dict:
    """Read a list file's parsed JSON into ``{"entries": [...], "meta": {...}}``.

    Two shapes are accepted: the full file the enrichment pass writes (an object with an
    ``items`` array) and a bare array of entries, which is what a human writes by hand
    when re-running one cell. Both end up as the same normalized entries.
    """
    meta: dict = {}
    if isinstance(payload, dict):
        raw = payload.get("items")
        if raw is None:
            raise ItemListError("the list file has no 'items' array")
        meta = {k: v for k, v in payload.items() if k != "items"}
    elif isinstance(payload, list):
        raw = payload
    else:
        raise ItemListError(f"a list file is an object or an array, got {type(payload).__name__}")
    if not isinstance(raw, list):
        raise ItemListError("'items' is not an array")
    entries = [normalize_entry(e, i) for i, e in enumerate(raw)]
    if not entries:
        raise ItemListError("the list file selects no items; an empty enrichment cell "
                            "would run the arms on nothing and report a zero denominator")
    seen: dict[int, int] = {}
    for pos, e in enumerate(entries):
        if e["index"] in seen:
            raise ItemListError(
                f"index {e['index']} appears twice (entries {seen[e['index']]} and {pos}); "
                "the resume merge is by item key, so a repeated item would alias one "
                "banked record onto both positions")
        seen[e["index"]] = pos
    return {"entries": entries, "meta": meta}


def select_items(pool: Sequence, entries: Sequence[dict], *,
                 pool_sha256: str | None = None,
                 meta: dict | None = None) -> tuple[list, dict]:
    """The pool items the entries name, in LIST order, plus a resolution report.

    Refuses on an out-of-range index, on a question hash that does not match the item at
    that index, and on a whole-pool hash that disagrees with the one recorded in the
    file. Returns ``(items, report)``; the report carries the counts that go into the
    run's summary so the cell can say what it selected and how much of it was verified.
    """
    meta = meta or {}
    recorded = meta.get("pool_sha256")
    if pool_sha256 is not None and recorded is not None and pool_sha256 != recorded:
        raise ItemListError(
            "the list file was built against a different pool file:\n"
            f"    file records pool_sha256={recorded}\n"
            f"    this run's pool is    pool_sha256={pool_sha256}\n"
            "  Every index in the list would select a different item. Rebuild the list "
            "from the pass that ran against this pool.")
    recorded_size = meta.get("pool_size")
    if recorded_size is not None and recorded_size != len(pool):
        raise ItemListError(
            f"the list file was built against a pool of {recorded_size} items and this "
            f"pool holds {len(pool)}; the indices do not carry across.")
    items = []
    n_verified = 0
    for pos, e in enumerate(entries):
        idx = e["index"]
        if idx < 0 or idx >= len(pool):
            raise ItemListError(
                f"entry {pos} names index {idx}, outside the {len(pool)}-item pool")
        it = pool[idx]
        sha = e["question_sha16"]
        if sha is not None:
            actual = question_sha16(it.question)
            if actual != sha:
                raise ItemListError(
                    f"entry {pos} names index {idx} with question_sha16={sha}, but the "
                    f"item at that index hashes to {actual}. The pool moved under the "
                    "list; nothing was run.")
            n_verified += 1
        items.append(it)
    report = {
        "n_selected": len(items),
        "n_verified_by_question_hash": n_verified,
        "n_unverified": len(items) - n_verified,
        "indices": [e["index"] for e in entries],
        "selector": meta.get("selector"),
        "schema": meta.get("schema"),
    }
    return items, report


def refusal_message(exc: ItemListError, path) -> str:
    """The refusal printed by the runner. Named path, stated consequence, no call made."""
    return (
        f"\n[item-list] REFUSED: {path}\n"
        f"  {exc}\n"
        "  No model call was made and no checkpoint was written.\n"
    )
