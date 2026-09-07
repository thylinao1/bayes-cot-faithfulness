"""The ladder's TRAINING item pool: a source that shares no item with any evaluation pool.

Why this module has to exist
---------------------------
``bcf/ladder_wave.sh`` trains 12 checkpoints and then measures them with the sweep's own
frozen arms. If a checkpoint's training items were also its evaluation items it would be
reporting its own training set, and no organism-minus-twin difference read off it would
mean anything. ``trigger_data.EvaluationGuard`` already REFUSES that build, so without a
disjoint pool the ladder simply cannot be trained; docs/LADDER-IMPL.md step 0b says so in
as many words:

    "The arc_challenge pool is 1,500 items and the stated-hint cell enters 1,500, so
    EVERY item of that pool is an evaluation item and the ladder's training pool must be
    a separate fetch. This is not optional and the builder will refuse otherwise."

This module is that separate fetch.

What "shares no item" means here, exactly
-----------------------------------------
Two checks, both of which must pass, because either alone has a hole.

*By id.* The pinned pools carry no id field of their own (``question``, ``choices``,
``answer_index`` and nothing else, see ``src/bayes_cot_faithfulness/item_list.py``), so
the id of an evaluation item in this repository IS ``question_sha16``, the sha256 prefix
of its question text. A candidate whose ``question_sha16`` is in the exclusion set is
dropped. A source that carries a native id (ARC rows do) has that id checked against the
same set too; that check is a no-op today and is kept because it costs nothing and stops
being a no-op the moment an evaluation pool grows an id field.

*By normalised text.* Casefolded, punctuation-stripped, whitespace-collapsed question
text. This is what catches a duplicate that appears under a different id: the same item
re-typed with different punctuation hashes differently under ``question_sha16`` and
identically here. It is not decoration. On the real build (2026-09-08) the id check
dropped 40 of the 1,119 ARC-Challenge train rows and the text check dropped **one more**,
an owl-colouration item that reaches the evaluation pool under a different surface form.
Without the text check that item would have been trained on and measured on.

The refusal
-----------
:func:`write_pool` re-runs the disjointness assertion on the items it is about to write
and raises before opening a file. A pool that overlaps is not written, not written
partially, and not written with a warning.

Determinism
-----------
The candidate order is the source's own row order and every step here is a filter, so
the same candidates and the same evaluation pools give a byte-identical file. The
manifest carries ``item_sequence_sha256`` over the kept items exactly as
``experiments/data/pool_manifest.json`` does for the evaluation pools, so a rebuild that
moved is visible without diffing the data file.

Licence, and why the data file is not committed
-----------------------------------------------
ARC-Challenge is CC BY-SA 4.0 and ``experiments/data/pool_manifest.json`` records the
standing repo decision that redistributing it inside an MIT repo is the operator's
question, not the engineer's. The same decision applies here: the pool file is
gitignored and the MANIFEST is committed, so the pool is reproducible byte for byte from
the pinned fetcher and checkable without redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bayes_cot_faithfulness.item_list import question_sha16

from .spec import N_TRAIN_EXAMPLES

SCHEMA = "bcf.ladder.train_pool.v1"

# The ARC-Challenge option cap the evaluation pool applies past its frozen prefix
# (experiments/data/pool_manifest.json, "max_choices_allowed": 4). The ladder trains on
# the SAME answer-label surface it is measured on, so a 5-option training item would
# teach a label (E) that no ARC evaluation prompt ever presents.
MAX_CHOICES = 4

# The datasets-server paging constants of experiments/fetch_arc.py, repeated rather than
# imported because experiments/ is a script directory and not an importable package.
_API = "https://datasets-server.huggingface.co/rows"
_DATASET = "allenai/ai2_arc"
_PAGE = 100
_FETCH_ATTEMPTS = 6
_FETCH_WAIT = 5.0

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


class TrainPoolError(ValueError):
    """A pool that would overlap the evaluation items, or could not be built at all."""


def normalize_question(question: str) -> str:
    """Casefolded, punctuation-stripped, whitespace-collapsed question text.

    Deliberately aggressive: the point is to catch the SAME item wearing a different
    surface, so two questions that differ only in quoting, hyphenation or spacing must
    collapse to one string. A false positive here costs one training item; a false
    negative costs the ladder's whole disjointness claim.
    """
    return _SPACE.sub(" ", _PUNCT.sub(" ", question.casefold())).strip()


def normalized_text_sha256(question: str) -> str:
    """The full sha256 of :func:`normalize_question`, which is the text-level key."""
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PoolRef:
    """One evaluation pool (or one frozen prefix of one) that the ladder must avoid."""

    name: str
    path: str
    sha256: str
    n_items: int
    n_excluded: int

    def to_json(self) -> dict:
        return {
            "name": self.name, "path": self.path, "sha256": self.sha256,
            "n_items_in_file": self.n_items, "n_items_excluded": self.n_excluded,
        }


@dataclass(frozen=True)
class EvaluationExclusions:
    """Every evaluation item of every substrate, by id and by normalised text."""

    pools: tuple[PoolRef, ...]
    ids: frozenset[str]
    texts: frozenset[str]

    @classmethod
    def from_pool_files(cls, pools: Sequence[tuple[str, Path, int | None]]) -> EvaluationExclusions:
        """Build from ``(name, path, n_items_or_None)`` triples.

        ``n_items`` names the entered n of the cell whose items these are; ``None``
        means the whole file. Both the full 1,500-item pool and its frozen 700-item
        prefix are passed by the CLI, even though the prefix's ids are a subset, so the
        manifest NAMES the 700 rather than leaving a reader to infer that it is covered.
        """
        refs: list[PoolRef] = []
        ids: set[str] = set()
        texts: set[str] = set()
        for name, path, n in pools:
            path = Path(path)
            if not path.exists():
                raise TrainPoolError(
                    f"evaluation pool {name} is not at {path}; refusing to build a "
                    "training pool that cannot be checked against it")
            raw = path.read_bytes()
            items = json.loads(raw.decode("utf-8"))
            take = len(items) if n is None else int(n)
            if take > len(items):
                raise TrainPoolError(
                    f"{name} names {take} evaluation items and the file holds {len(items)}")
            for it in items[:take]:
                ids.add(question_sha16(it["question"]))
                texts.add(normalized_text_sha256(it["question"]))
            refs.append(PoolRef(name=name, path=str(path),
                                sha256=hashlib.sha256(raw).hexdigest(),
                                n_items=len(items), n_excluded=take))
        return cls(pools=tuple(refs), ids=frozenset(ids), texts=frozenset(texts))

    def hits(self, item: dict) -> list[str]:
        """Which checks this candidate trips, as a list of reasons (empty means clean)."""
        why: list[str] = []
        if question_sha16(item["question"]) in self.ids:
            why.append("id")
        source_id = item.get("source_id")
        if source_id is not None and source_id in self.ids:
            why.append("source_id")
        if normalized_text_sha256(item["question"]) in self.texts:
            why.append("normalized_text")
        return why


@dataclass
class FilterResult:
    """The kept items and every count that explains the ones that are not there."""

    items: list[dict]
    n_candidates: int
    n_excluded_by_id: int
    n_excluded_by_text: int
    n_dropped_over_cap: int
    n_internal_duplicates: int
    excluded_examples: list[dict] = field(default_factory=list)


def filter_candidates(candidates: Iterable[dict], exclusions: EvaluationExclusions, *,
                      max_choices: int = MAX_CHOICES,
                      n: int | None = None) -> FilterResult:
    """Drop every candidate that overlaps, exceeds the option cap, or repeats.

    An item excluded by BOTH checks is counted under ``id`` only, so the two counts sum
    to the number of items removed rather than double-counting; ``n_excluded_by_text``
    is therefore the number of duplicates the id check would have MISSED, which is the
    number worth reporting.
    """
    kept: list[dict] = []
    seen_text: set[str] = set()
    n_cand = n_id = n_text = n_cap = n_dupe = 0
    examples: list[dict] = []
    for raw in candidates:
        n_cand += 1
        why = exclusions.hits(raw)
        if "id" in why or "source_id" in why:
            n_id += 1
            continue
        if "normalized_text" in why:
            n_text += 1
            if len(examples) < 5:
                examples.append({"reason": "normalized_text",
                                 "source_id": raw.get("source_id"),
                                 "question_sha16": question_sha16(raw["question"])})
            continue
        if len(raw["choices"]) > max_choices:
            n_cap += 1
            continue
        key = normalized_text_sha256(raw["question"])
        if key in seen_text:
            n_dupe += 1
            continue
        seen_text.add(key)
        kept.append(raw)
        if n is not None and len(kept) >= n:
            break
    return FilterResult(items=kept, n_candidates=n_cand, n_excluded_by_id=n_id,
                        n_excluded_by_text=n_text, n_dropped_over_cap=n_cap,
                        n_internal_duplicates=n_dupe, excluded_examples=examples)


def assert_disjoint(items: Sequence[dict], exclusions: EvaluationExclusions) -> None:
    """Raise unless EVERY item is clean on both checks. Called again before any write."""
    bad = [(i, it, why) for i, it in enumerate(items) if (why := exclusions.hits(it))]
    if not bad:
        return
    shown = ", ".join(
        f"index {i} (question_sha16 {question_sha16(it['question'])}, on {'+'.join(w)})"
        for i, it, w in bad[:5])
    raise TrainPoolError(
        f"REFUSING to write a ladder training pool: {len(bad)} of {len(items)} items are "
        f"evaluation items of {', '.join(p.name for p in exclusions.pools)}: {shown}"
        f"{' ...' if len(bad) > 5 else ''}. Nothing was written. A checkpoint trained on "
        "the items it is measured on would report its own training set.")


def item_sequence_sha256(items: Sequence[dict]) -> str:
    """The pool_manifest.json scheme: one hash over the ordered item content."""
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps({"question": it["question"], "choices": list(it["choices"]),
                             "answer_index": int(it["answer_index"])},
                            sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_manifest(result: FilterResult, exclusions: EvaluationExclusions, *,
                   sources: Sequence[dict], max_choices: int,
                   n_requested: int | None) -> dict:
    """Everything a reader needs to rebuild this pool and to check it did not overlap."""
    items = result.items
    return {
        "schema": SCHEMA,
        "purpose": (
            "the organism/twin TRAINING item pool of element 11(b). Disjoint by "
            "construction from every evaluation pool of every substrate, checked by "
            "item id (question_sha16) AND by normalised question text."),
        "sources": list(sources),
        "n_items": len(items),
        "n_requested": n_requested,
        "meets_n_requested": n_requested is None or len(items) >= n_requested,
        "max_choices_allowed": max_choices,
        "counts": {
            "n_candidates": result.n_candidates,
            "n_excluded_by_id": result.n_excluded_by_id,
            "n_excluded_by_normalized_text": result.n_excluded_by_text,
            "n_dropped_over_choice_cap": result.n_dropped_over_cap,
            "n_internal_duplicates_dropped": result.n_internal_duplicates,
            "note": (
                "n_excluded_by_normalized_text counts only the items the id check did "
                "NOT already remove, so it is exactly the number of duplicates that "
                "would have been missed by an id-only check."),
        },
        "excluded_by_text_examples": result.excluded_examples,
        "evaluation_pools_checked_against": [p.to_json() for p in exclusions.pools],
        "n_evaluation_ids": len(exclusions.ids),
        "n_evaluation_normalized_texts": len(exclusions.texts),
        "identity_scheme": (
            "id = question_sha16 (item_list.py), because the pinned pools carry no id "
            "field; text = sha256 of casefolded, punctuation-stripped, "
            "whitespace-collapsed question text"),
        # The one number an operator has to act on. N_TRAIN_EXAMPLES is a LANE CHOICE
        # (spec.py), not a pre-registered value, and trigger_data.build_training_set
        # REFUSES when it exceeds the pool. Recorded here so the refusal is read before
        # a card is allocated rather than after.
        "ladder_training_set_fit": {
            "n_train_examples_lane_choice": N_TRAIN_EXAMPLES,
            "n_items_available": len(items),
            "fits": len(items) >= N_TRAIN_EXAMPLES,
            "shortfall": max(0, N_TRAIN_EXAMPLES - len(items)),
            "consequence": (
                "trigger_data.build_training_set raises LadderDataError when "
                "n_examples > len(pool). Either the LANE CHOICE N_TRAIN_EXAMPLES drops "
                "to at most n_items_available (it is not pre-registered and nothing in "
                "element 11 sizes it), or a second disjoint source is admitted by "
                "ruling. This lane does neither."),
        },
        "item_sequence_sha256": item_sequence_sha256(items),
        "first_item_sha256": (hashlib.sha256(
            json.dumps(items[0], sort_keys=True).encode()).hexdigest() if items else None),
        "last_item_sha256": (hashlib.sha256(
            json.dumps(items[-1], sort_keys=True).encode()).hexdigest() if items else None),
        "licence_note": (
            "ARC-Challenge is CC BY-SA 4.0. The pool FILE is gitignored under the same "
            "standing decision as experiments/data/arc_challenge.json; this manifest is "
            "what the repository carries."),
        "file_sha256": None,
    }


def write_pool(items: Sequence[dict], manifest: dict, exclusions: EvaluationExclusions,
               pool_path: Path, manifest_path: Path) -> dict:
    """Check disjointness once more, then write the pool and its manifest.

    The re-check is not paranoia about :func:`filter_candidates`: it is what makes the
    refusal true of the bytes on disk rather than of an intermediate list, and it is the
    single place a caller that assembled items by hand is caught.
    """
    assert_disjoint(items, exclusions)
    pool_path = Path(pool_path)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(list(items), indent=1) + "\n")
    manifest = dict(manifest)
    manifest["file_sha256"] = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    manifest["pool_path"] = str(pool_path)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# --- the source fetch, the only thing here that touches the network ------------------

def _fetch_page(config: str, split: str, offset: int, length: int) -> dict:
    """One datasets-server page, with the retry ladder of experiments/fetch_arc.py."""
    url = (f"{_API}?dataset={_DATASET}&config={config}&split={split}"
           f"&offset={offset}&length={length}")
    last: Exception | None = None
    for attempt in range(_FETCH_ATTEMPTS):
        req = urllib.request.Request(
            url, headers={"User-Agent": "bayes-cot-faithfulness/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            last = exc
        except urllib.error.URLError as exc:
            last = exc
        if attempt < _FETCH_ATTEMPTS - 1:
            time.sleep(_FETCH_WAIT * (2 ** attempt))
    raise last if last is not None else TrainPoolError("fetch failed with no exception")


def row_to_item(row: dict) -> dict | None:
    """One ARC row in the pool format, plus its native id. None if malformed.

    The acceptance rule (3 to 5 options, an answer key that resolves to a label) is the
    one ``experiments/fetch_arc.py:to_item`` applies to the evaluation pools, so the two
    pools are the same kind of object and the option cap is the only difference.
    """
    question = row.get("question", "").strip()
    choices = row.get("choices", {})
    text = choices.get("text", [])
    labels = choices.get("label", [])
    key = row.get("answerKey")
    if not question or len(text) != len(labels) or not (3 <= len(text) <= 5):
        return None
    if key not in labels:
        return None
    return {"question": question, "choices": list(text),
            "answer_index": labels.index(key), "source_id": row.get("id")}


def fetch_split(config: str, split: str) -> list[dict]:
    """Every usable row of one split, in datasets-server row order."""
    out: list[dict] = []
    offset = 0
    total = None
    while True:
        body = _fetch_page(config, split, offset, _PAGE)
        total = body.get("num_rows_total", total)
        rows = body.get("rows", [])
        if not rows:
            break
        for r in rows:
            item = row_to_item(r.get("row", {}))
            if item:
                out.append(item)
        offset += len(rows)
        if total is not None and offset >= total:
            break
        time.sleep(0.2)
    return out


# --- CLI -----------------------------------------------------------------------------

DEFAULT_EVAL_POOLS = (
    ("arc_challenge_1500", "arc_challenge.json", None),
    ("arc_challenge_frozen_prefix_700", "arc_challenge.json", 700),
    ("aqua_rat_1500", "aqua_rat.json", None),
    ("aqua_rat_frozen_prefix_700", "aqua_rat.json", 700),
    ("logiqa2_1500", "logiqa2.json", None),
    ("logiqa2_frozen_prefix_700", "logiqa2.json", 700),
)


def default_exclusions(data_dir: Path) -> EvaluationExclusions:
    """The six evaluation pools of record, resolved under ``data_dir``."""
    return EvaluationExclusions.from_pool_files(
        [(name, Path(data_dir) / fname, n) for name, fname, n in DEFAULT_EVAL_POOLS])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=Path("experiments/data"),
                    help="where the pinned evaluation pools live")
    ap.add_argument("--config", default="ARC-Challenge",
                    help="ai2_arc config to draw the TRAINING pool from")
    ap.add_argument("--split", default="train", help="split to draw from")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="a local JSON list of candidate items; skips the network fetch")
    ap.add_argument("--n", type=int, default=None,
                    help="cap on the pool size (default: keep every clean item)")
    ap.add_argument("--require-n", type=int, default=None,
                    help="refuse to write a pool smaller than this")
    ap.add_argument("--max-choices", type=int, default=MAX_CHOICES)
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/data/ladder_train_pool.json"))
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: <out> with '_manifest' before the suffix")
    a = ap.parse_args(argv)

    exclusions = default_exclusions(a.data_dir)
    if a.candidates is not None:
        candidates = json.loads(Path(a.candidates).read_text())
        source = {"kind": "local_file", "path": str(a.candidates),
                  "sha256": hashlib.sha256(Path(a.candidates).read_bytes()).hexdigest(),
                  "n_rows": len(candidates)}
    else:
        candidates = fetch_split(a.config, a.split)
        source = {"kind": "huggingface_datasets_server_rows_api",
                  "dataset": _DATASET, "config": a.config, "split": a.split,
                  "licence": "CC BY-SA 4.0", "n_rows": len(candidates),
                  "order": "datasets-server row order",
                  "acceptance": "3 to 5 options, answerKey resolving to a label "
                                "(experiments/fetch_arc.py:to_item)"}

    result = filter_candidates(candidates, exclusions,
                               max_choices=a.max_choices, n=a.n)
    manifest = build_manifest(result, exclusions, sources=[source],
                              max_choices=a.max_choices, n_requested=a.n)
    if a.require_n is not None and len(result.items) < a.require_n:
        print(f"[train-pool] REFUSING: {len(result.items)} clean items, "
              f"--require-n {a.require_n}. Nothing written.", file=sys.stderr)
        return 2
    manifest_path = a.manifest or a.out.with_name(a.out.stem + "_manifest.json")
    try:
        manifest = write_pool(result.items, manifest, exclusions, a.out, manifest_path)
    except TrainPoolError as exc:
        print(f"[train-pool] {exc}", file=sys.stderr)
        return 3
    print(f"wrote {len(result.items)} ladder training items -> {a.out}")
    print(f"  excluded by id: {result.n_excluded_by_id}; "
          f"by normalised text (beyond id): {result.n_excluded_by_text}; "
          f"over the {a.max_choices}-choice cap: {result.n_dropped_over_cap}; "
          f"internal duplicates: {result.n_internal_duplicates}")
    print(f"  of {result.n_candidates} candidate rows; manifest -> {manifest_path}")
    print(f"  file_sha256 {manifest['file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
