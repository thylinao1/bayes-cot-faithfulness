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
    # Frozen 2026-07-17 on an explicit sign-off: the Phase-2 additive-arms
    # pre-registration and the fixed A9 specificity holdout it pins.
    # Amendment A1 (2026-07-22, approved): appended the additive powered-P3
    # AQuA-RAT substrate section; fingerprint updated in the same commit per the
    # document's own Amendment protocol. Nothing existing in the file was edited.
    # Amendment A2 (2026-09-07, approved by the operator 2026-09-06): appended
    # the additive A2 section carrying the new P-item P2b and the pointer to the
    # companion pre-registration below. Fingerprint updated in the same commit;
    # the first 320 lines of the file are byte-identical to the A1 state.
    "experiments/PREREGISTRATION_phase2_arms.md": (
        "cf4cef118bd75d2222849043ed6d9a3e74c81b85f407039b14159032245d9970"
    ),
    # Frozen 2026-09-07 by Amendment A2: the companion pre-registration holding
    # elements 0 to 23 (the Column B contract, the jury, the scale study). Same
    # amendment protocol as the documents above: additive sections only, hash
    # updated in the same commit.
    # Amendment A3 (2026-09-07, element 13): appended section 27, the
    # post-skeleton values from jobs 825511 and 825492, the quantities the
    # skeleton did not produce with their blocking causes, and the four
    # orchestrator rulings of DECISION-LOG.md 2026-09-07 01:58. Fingerprint
    # updated in the same commit; git diff against the A2 state shows 491 added
    # lines and 0 deleted lines, so the first 1,179 lines are byte-identical.
    # RE-PINNED 2026-09-07 after the W3b slice run: job 826025 filled A3.3, A3.4
    # and A3.5 and jobs 825548 and 826020 filled A3.6's concurrency rows. Only
    # lines INSIDE the A3 section changed, so git diff against main on this file
    # is still additions-only, now 735 added and 0 deleted. Still a DRAFT: the
    # operator has not approved it and the A2 preconditions are unchanged.
    # RE-PINNED AGAIN 2026-09-07 by the A3-final lane: appended section A3.9,
    # carrying rulings R1 to R8 of RULINGS-2026-09-07.md with the measurements
    # and job ids they rest on, plus paragraphs appended INSIDE A3.2 (R6),
    # A3.4 (R3), A3.5 (R4), A3.6 (R1, R2, R5) and A3.7 (which rows are closed
    # and by which ruling). Nothing existing was edited or deleted: git diff
    # against main on this file is 391 added and 0 DELETED lines, so every line
    # above the insertion points is byte-identical. R9, the jury Q1 construct,
    # is left open by name. Still a DRAFT: the operator has not approved it.
    # RE-PINNED 2026-09-07 evening by Amendment A4: appended one section
    # carrying rulings R9 (no primary jury configuration; column A of record
    # stays the uncorrected regex share), R11 (gpt-oss suspended as a subject,
    # kept as a judge) and R12 (element 9.4 for rows with no documented
    # reasoning switch), the measured chain-level mediator noise from job
    # 826596, two wave-1 fits findings, and the serving line the reruns were
    # made under. Nothing existing was edited or deleted: git diff against main
    # on this file is 626 added and 0 DELETED lines, so the first 2,305 lines
    # are byte-identical. Still a DRAFT: the operator has not approved it.
    # RE-PINNED 2026-09-07 late evening by Amendment A5 (ruling R13): appended
    # one section carrying the logit-level column B for an outcome scale
    # elements 0 and 7 already pre-register, its estimator
    # (bayes_cot_faithfulness.gaussian_mediation), the bridge element 0
    # requires a two-scale row to print, a six-condition eligibility gate, the
    # cross-scale reporting order, and a written refusal to set a load-bearing
    # threshold on that scale. No threshold moved and no condition loosened:
    # the text-level 0.15 of section 2.5, section 22.1's 0.10 margin and every
    # earlier verdict are untouched. Nothing existing was edited or deleted:
    # git diff against main on this file is 327 added and 0 DELETED lines, so
    # the first 2,931 lines are byte-identical. Still a DRAFT: the operator has
    # not approved it.
    "experiments/PREREGISTRATION_jury_and_scale.md": (
        "5c050a1d05f4828808bda22063ba9c767f8af2a62f059fdb4ebb5191985fe3e9"
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
