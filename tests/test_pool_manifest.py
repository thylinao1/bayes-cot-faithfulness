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

# The answer-label set each substrate's frozen prompt surface parses against. ARC's cap
# is 4 because the 700-item pool every Phase-1 artifact was measured on holds at most 4
# options, and a 1,500-item pool that reached 5 would make an ARC cell parse against five
# labels: a change to the frozen prompt surface, not a bigger sample of the same one.
# DECISION-LOG 2026-09-07 03:58 ruling (a).
MAX_CHOICES = {"arc_challenge": 4, "aqua_rat": 5, "logiqa2": 4}

# The sequence hash of the first 700 ARC items, copied from the manifest that shipped
# with the 700-item pool (W3, 2026-09-07). It is written out here rather than read from
# the manifest so that a regenerated manifest agreeing with a MOVED pool still fails:
# the manifest cannot vouch for itself.
ARC_FROZEN_PREFIX_SHA = (
    "a48a5bef74ef30d0be729ffb4c90453a8f2390a6bc9bfe05a1a7842200860eaa"
)


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


def test_no_arc_item_carries_more_than_four_choices(manifest):
    """The ARC label set is four letters, at 700 items and at 1,500.

    The enlargement to 1,500 reached four 5-option items in the ARC-Challenge test
    split (old pool indices 836, 868, 1037, 1382), all of them past the frozen prefix.
    Keeping them would have widened the answer-label set the frozen prompt surface
    parses against, which is an amendment-sized change and not a sampling change.
    """
    assert manifest["pools"]["arc_challenge"]["choices_max"] <= MAX_CHOICES[
        "arc_challenge"
    ]


@pytest.mark.parametrize("name", ["arc_challenge", "aqua_rat", "logiqa2"])
def test_every_pool_stays_inside_its_own_label_set(manifest, name):
    pool = manifest["pools"][name]
    assert pool["choices_max"] <= MAX_CHOICES[name], name
    assert pool["max_choices_allowed"] == MAX_CHOICES[name], name


@pytest.mark.parametrize("name", ["arc_challenge", "aqua_rat", "logiqa2"])
def test_no_item_in_the_local_pool_exceeds_its_label_set(name):
    """The manifest field above is a summary; this reads every item on disk."""
    path = REPO / "experiments" / "data" / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} is gitignored and not present in this checkout")
    items = json.loads(path.read_text())
    over = [i for i, it in enumerate(items) if len(it["choices"]) > MAX_CHOICES[name]]
    assert not over, (
        f"{name}: items {over[:10]} carry more than {MAX_CHOICES[name]} options; the "
        f"frozen prompt surface parses against {MAX_CHOICES[name]} answer labels"
    )


def test_the_arc_frozen_prefix_hash_is_the_one_phase_1_was_measured_on(manifest):
    """The filter must not have moved a single item of the first 700."""
    assert manifest["pools"]["arc_challenge"]["frozen_prefix_sha256"] == (
        ARC_FROZEN_PREFIX_SHA
    )


def test_the_local_arc_pool_reproduces_the_frozen_prefix_hash():
    path = REPO / "experiments" / "data" / "arc_challenge.json"
    if not path.exists():
        pytest.skip("arc_challenge.json is gitignored and not present in this checkout")
    items = json.loads(path.read_text())
    per_item = [
        hashlib.sha256(
            json.dumps(it, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for it in items[:700]
    ]
    assert hashlib.sha256("".join(per_item).encode()).hexdigest() == (
        ARC_FROZEN_PREFIX_SHA
    )
