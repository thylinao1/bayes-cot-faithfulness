"""Tripwire for the frozen pre-registration elements.

The pre-registration documents, the fixed data they pin (the A9 specificity
holdout), and the measurement instrument they freeze (the acknowledgment
detector, the hint templates, the CoT instruction, and the answer-extraction
regexes) must not change silently. Any legitimate change goes through the
amendment protocol in the owning pre-registration, and the person making it
updates the fingerprint here in the same commit, so the diff itself records
that a frozen element moved.

New arms, new modules, and new analysis code are additive by design and do not
touch anything fingerprinted below.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bayes_cot_faithfulness import interventions as iv

REPO = Path(__file__).resolve().parents[1]

# Recorded 2026-07-17, matching the state frozen on 2026-06-01 (see the
# amendment log in experiments/PREREGISTRATION.md).
FROZEN_FILE_SHA256 = {
    "experiments/PREREGISTRATION.md": (
        "71ce9840cda130888a77402c40a988013d0cdb982b18f5d4334de5dee43e4f15"
    ),
    "experiments/PREREGISTRATION_uncertain_items.md": (
        "bc13f155e6c7b9dfbe79c1d42fe309b0edd3e4c4594089bb833b68e014a8de75"
    ),
    # Frozen 2026-07-17 (operator: "commit and lock it"): the Phase-2 additive-arms
    # pre-registration and the fixed A9 specificity holdout it pins.
    # Amendment A1 (2026-07-22, operator-approved): appended the additive powered-P3
    # AQuA-RAT substrate section; fingerprint updated in the same commit per the
    # document's own Amendment protocol. Nothing existing in the file was edited.
    "experiments/PREREGISTRATION_phase2_arms.md": (
        "0e6a577ae9e0a8f520f7b968738b16ed393413c0c3d21efa22746cf9d9b770d4"
    ),
    "experiments/data/specificity_holdout.json": (
        "82ab56d8561d533a0d7196c4c604ab3ae6d27f53acb028bf390121c3567dabcd"
    ),
}

FROZEN_CODE_SHA256 = {
    "hint_ack_re": "eabd308dac649f04c73a49072573b4e1e42366eaafbaa4fafcdea144434061ab",
    "hint_templates": "fa8e4e29cb673d10450c0b15ce42872d26ded897463a7dbccbf717b66db40d85",
    "cot_instruction": "5f50db6cd49e1749639d800ed4a03eed3adc780c934639bd37f80229d6fbe16e",
    "answer_regexes": "f3446182f1147cd04b2447af17e8620ec89a98fc89e008d097d2f1cf6407aa36",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_preregistration_files_unchanged():
    for rel, want in FROZEN_FILE_SHA256.items():
        got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        assert got == want, (
            f"{rel} changed. Pre-registrations are FROZEN; log an amendment in "
            "experiments/PREREGISTRATION.md and update this fingerprint in the "
            "same commit."
        )


def test_hint_ack_detector_unchanged():
    assert _sha(iv._HINT_ACK_RE.pattern) == FROZEN_CODE_SHA256["hint_ack_re"], (
        "_HINT_ACK_RE changed. The acknowledgment detector is FROZEN "
        "(amendment 2026-06-01); any change must be logged in "
        "experiments/PREREGISTRATION.md before re-scoring."
    )


def test_existing_hint_templates_unchanged():
    assert (
        _sha(repr(sorted(iv._HINT_TEMPLATES.items())))
        == FROZEN_CODE_SHA256["hint_templates"]
    ), (
        "_HINT_TEMPLATES changed. The existing arms are frozen; new cue "
        "families go in a separate additive structure, never in this dict."
    )


def test_cot_instruction_unchanged():
    assert _sha(iv._COT_INSTRUCTION) == FROZEN_CODE_SHA256["cot_instruction"], (
        "_COT_INSTRUCTION changed. The prompt format of the frozen arms must "
        "not move mid-study; new protocols are separate additive arms."
    )


def test_answer_extraction_regexes_unchanged():
    combined = (
        iv._ANSWER_RE.pattern
        + iv._ANSWER_CONNECTOR_RE.pattern
        + iv._TRAILING_OPTION_RE.pattern
    )
    assert _sha(combined) == FROZEN_CODE_SHA256["answer_regexes"], (
        "The answer-extraction regexes changed. Audit first (see the parser "
        "audit), never mid-run: changing extraction re-scores saved transcripts."
    )
