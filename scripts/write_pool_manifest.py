"""Regenerate experiments/data/pool_manifest.json from the pools on disk.

The three substrate pools are gitignored (ARC is CC BY-SA 4.0 and LogiQA 2.0 is
CC BY-NC-SA 4.0; redistributing either inside an MIT repo is an operator-owned licence
question), so the manifest is the committed artifact that makes "pools of at least
N items, in deterministic order" checkable without redistributing the items.

Every field here is READ OFF the file it describes. Nothing is typed in by hand except
the per-pool provenance block below, which names the fetcher, its exact command, the
upstream source and the licence.

    python scripts/write_pool_manifest.py
    python scripts/write_pool_manifest.py --check   # verify without writing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "experiments" / "data"
MANIFEST = DATA / "pool_manifest.json"

GENERATED_FOR = (
    "DECISION-LOG 2026-09-07 ruling (b): item pools of at least 1,500 so the "
    "stated-hint and professor cells can enter 1,500 items each. Supersedes the "
    "earlier 'at least 700' target of CONTRACT.md and PREREGISTRATION_jury_and_scale "
    "element 9, which stays satisfied."
)

WHY_NOT_COMMITTED = (
    "experiments/data/*.json is gitignored by a standing repo decision: ARC-Challenge "
    "is CC BY-SA 4.0 and LogiQA 2.0 is CC BY-NC-SA 4.0, and redistributing either "
    "inside an MIT repo is a licence question the operator owns, not the sweep "
    "engineer. This manifest is committed instead, so the exact pool is reproducible "
    "byte for byte from the pinned fetchers and checkable without redistribution."
)

PROVENANCE = {
    "arc_challenge": {
        "fetcher": "experiments/fetch_arc.py",
        "command": ("python experiments/fetch_arc.py --n 1500 --split test "
                    "--then-split validation --then-split train --max-choices 4 "
                    "--max-choices-from 700"),
        "source": "allenai/ai2_arc ARC-Challenge, Hugging Face datasets-server rows API",
        "licence": "CC BY-SA 4.0",
        "order": ("datasets-server row order, test then validation then train, "
                  "deduplicated on the resume merge key (question plus choices), "
                  "first occurrence wins"),
        "filter": ("items with more than 4 options are dropped from index 700 onward "
                   "and the pool is backfilled from the same stream to 1,500. The "
                   "ARC-Challenge test split carries four 5-option items and all four "
                   "fall past the frozen prefix (old indices 836, 868, 1037, 1382), so "
                   "the enlargement would have made an ARC cell parse against five "
                   "answer labels where the 700-item pool every Phase-1 artifact was "
                   "measured on has at most four. The first 700 items are untouched by "
                   "the filter and their sequence hash is unchanged "
                   "(DECISION-LOG 2026-09-07 03:58 ruling (a))."),
    },
    "aqua_rat": {
        "fetcher": "experiments/fetch_aqua.py",
        "command": ("python experiments/fetch_aqua.py --n 1500 --split test "
                    "--then-split dev --then-split train"),
        "source": "google-deepmind/AQuA @ 26b64ed1f22208c6742f47df547dfd65088ec30d",
        "licence": "Apache-2.0",
        "order": ("shipped file order, test then dev then train, deduplicated by "
                  "question text, first occurrence wins"),
        "filter": "none; every AQuA-RAT item carries exactly 5 options",
    },
    "logiqa2": {
        "fetcher": "experiments/fetch_logiqa2.py",
        "command": "python experiments/fetch_logiqa2.py --n 1500 --split test",
        "source": "csitfun/LogiQA2.0 @ 955e1d3df6c59d9bfb44d9913da1e1a27ec14e18",
        "licence": "CC BY-NC-SA 4.0",
        "order": ("shipped file order, test split, deduplicated on the resume merge "
                  "key (question plus choices), first occurrence wins"),
        "filter": "none; every LogiQA 2.0 item carries exactly 4 options",
    },
}

# The answer-label set each substrate's frozen prompt surface parses against. A pool that
# exceeded its own maximum would silently widen that set, which is a change to the frozen
# prompt surface and belongs in an amendment, not in a fetch. Asserted here so a re-fetch
# that reintroduced a wider item cannot be written into the manifest quietly.
MAX_CHOICES = {"arc_challenge": 4, "aqua_rat": 5, "logiqa2": 4}

# The first 700 items of each pool must keep the indices they had in the 700-item pools
# every Phase-1 artifact was measured against, so the enlargement is a pure append and a
# banked record cannot land on a different item. This is the sequence hash of those 700,
# copied from the manifest that shipped with them (W3, 2026-09-07) and re-asserted here.
FROZEN_PREFIX_N = 700
FROZEN_PREFIX_SHA = {
    "arc_challenge": "a48a5bef74ef30d0be729ffb4c90453a8f2390a6bc9bfe05a1a7842200860eaa",
    "aqua_rat": "6997545c0887c57f5077112f8781ee3035a46a7327a75203ec48bc8cb3136117",
    "logiqa2": "bbacc55d5dda07b365c5eb15ab5caabbdc546a288d8087677be50fd137a0b962",
}


def item_hashes(items: list[dict]) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(it, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for it in items
    ]


def sequence_sha(per_item: list[str]) -> str:
    return hashlib.sha256("".join(per_item).encode()).hexdigest()


def describe(name: str) -> dict:
    path = DATA / f"{name}.json"
    raw = path.read_bytes()
    items = json.loads(raw)
    per_item = item_hashes(items)
    keys = {(it["question"], tuple(it["choices"])) for it in items}
    prefix_sha = sequence_sha(per_item[:FROZEN_PREFIX_N])
    if prefix_sha != FROZEN_PREFIX_SHA[name]:
        raise SystemExit(
            f"REFUSING: {name} first {FROZEN_PREFIX_N} items hash {prefix_sha}, not the "
            f"frozen {FROZEN_PREFIX_SHA[name]}. The enlargement moved an item that "
            f"Phase-1 artifacts already refer to by index."
        )
    choices_min = min(len(it["choices"]) for it in items)
    choices_max = max(len(it["choices"]) for it in items)
    if choices_max > MAX_CHOICES[name]:
        over = [i for i, it in enumerate(items) if len(it["choices"]) > MAX_CHOICES[name]]
        raise SystemExit(
            f"REFUSING: {name} holds {len(over)} items with more than "
            f"{MAX_CHOICES[name]} options (first at index {over[0]}, {choices_max} "
            f"options). That widens the answer-label set the frozen prompt surface "
            f"parses against. Re-fetch with the filter in PROVENANCE['{name}']"
            f"['command'], or amend the pre-registration."
        )
    return {
        **PROVENANCE[name],
        "n_items": len(items),
        "n_unique_resume_keys": len(keys),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "item_sequence_sha256": sequence_sha(per_item),
        "first_item_sha256": per_item[0],
        "last_item_sha256": per_item[-1],
        "frozen_prefix_n": FROZEN_PREFIX_N,
        "frozen_prefix_sha256": prefix_sha,
        "choices_min": choices_min,
        "choices_max": choices_max,
        "max_choices_allowed": MAX_CHOICES[name],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed manifest and exit non-zero on a diff")
    a = ap.parse_args(argv)

    payload = {
        "generated_for": GENERATED_FOR,
        "why_the_data_files_are_not_committed": WHY_NOT_COMMITTED,
        "pools": {name: describe(name) for name in PROVENANCE},
    }
    text = json.dumps(payload, indent=2) + "\n"
    if a.check:
        current = MANIFEST.read_text()
        if current != text:
            print("manifest is STALE; run scripts/write_pool_manifest.py", file=sys.stderr)
            return 1
        print("manifest matches the pools on disk")
        return 0
    MANIFEST.write_text(text)
    for name, pool in payload["pools"].items():
        print(f"{name}: {pool['n_items']} items, {pool['n_unique_resume_keys']} unique "
              f"keys, choices {pool['choices_min']}-{pool['choices_max']}")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
