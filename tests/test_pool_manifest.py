"""The committed pool manifest must describe the pools the fetchers actually produce.

The three substrate pools are gitignored (ARC is CC BY-SA 4.0 and LogiQA 2.0 is
CC BY-NC-SA 4.0, and redistributing either inside an MIT repo is an operator-owned
licence question), so the manifest is what makes "pools of at least 700, in deterministic
order" checkable in CI. When the pools are present locally these tests verify the
manifest against them; when they are absent the file-content tests skip and the manifest's
own invariants are still checked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "experiments" / "data" / "pool_manifest.json"
# CONTRACT.md and PREREGISTRATION_jury_and_scale element 9 say "at least 700". The
# DECISION-LOG ruling of 2026-09-07 (b) raises it: a cross-model column-A contrast needs
# about 300 followed items per cell, which at the banked follow rate means 1,500 entered
# for stated-hint and professor, so the pools those cells draw from are at least 1,500.
# The older 700 stays satisfied by construction, and the first 700 items keep their
# indices (frozen_prefix_sha256 below).
CONTRACT_MIN_POOL = 700
A3_MIN_POOL = 1500


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_covers_the_three_contract_substrates(manifest):
    assert set(manifest["pools"]) == {"arc_challenge", "aqua_rat", "logiqa2"}


def test_every_pool_meets_the_contract_minimum(manifest):
    for name, pool in manifest["pools"].items():
        assert pool["n_items"] >= CONTRACT_MIN_POOL, name


def test_every_pool_meets_the_a3_minimum(manifest):
    """DECISION-LOG 2026-09-07 ruling (b): 1,500 entered per high-follow cell."""
    for name, pool in manifest["pools"].items():
        assert pool["n_items"] >= A3_MIN_POOL, name


def test_every_pool_pins_the_frozen_first_700_prefix(manifest):
    """The enlargement must be a pure append: item k stays item k for k < 700.

    A banked Phase-1 record is merged back by (question, choices), but every position
    seeded quantity in the runner (wrong_label(rotate=i), placebo rng_seed=i, the anchor
    donor seed) is keyed on the INDEX, so a pool that reshuffled its first 700 would
    silently re-seed them.
    """
    for name, pool in manifest["pools"].items():
        assert pool["frozen_prefix_n"] == 700, name
        assert len(pool["frozen_prefix_sha256"]) == 64, name


def test_every_pool_is_resume_safe(manifest):
    """A repeated (question, choices) key makes the by-key resume merge alias records."""
    for name, pool in manifest["pools"].items():
        assert pool["n_unique_resume_keys"] == pool["n_items"], name


def test_every_pool_records_its_fetcher_command_licence_and_order(manifest):
    for name, pool in manifest["pools"].items():
        for field in ("fetcher", "command", "source", "licence", "order",
                      "file_sha256", "item_sequence_sha256"):
            assert pool.get(field), f"{name} is missing {field}"
        assert (REPO / pool["fetcher"]).exists(), name


@pytest.mark.parametrize("name", ["arc_challenge", "aqua_rat", "logiqa2"])
def test_manifest_matches_the_local_pool_when_it_is_present(manifest, name):
    path = REPO / "experiments" / "data" / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} is gitignored and not present in this checkout")
    pool = manifest["pools"][name]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pool["file_sha256"]
    items = json.loads(raw)
    assert len(items) == pool["n_items"]
    per_item = [
        hashlib.sha256(
            json.dumps(it, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for it in items
    ]
    # The ORDER is the thing being pinned: a reshuffled pool with identical items has a
    # different sequence hash, and "the first N items in fetch order" would silently mean
    # different items.
    assert hashlib.sha256("".join(per_item).encode()).hexdigest() == pool[
        "item_sequence_sha256"
    ]
    assert per_item[0] == pool["first_item_sha256"]
    assert per_item[-1] == pool["last_item_sha256"]
    n = pool["frozen_prefix_n"]
    prefix = hashlib.sha256("".join(per_item[:n]).encode()).hexdigest()
    assert prefix == pool["frozen_prefix_sha256"], (
        f"{name}: the first {n} items moved; every index-seeded draw in the runner "
        f"would be re-seeded against different items"
    )
