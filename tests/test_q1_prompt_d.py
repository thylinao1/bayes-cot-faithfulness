"""Guards on the Q1 construct-revision prompt file, `q1_mention_2026-09-07d.md`.

Ruling R9 (`PREREGISTRATION_jury_and_scale.md` A4.2) names the route to a Q1 candidate as a
NEW dated prompt file scored on the SAME ten thresholds. Two things have to hold for that
route to mean anything, and both are asserted here.

1. The new file is loadable by the runner's own loader and carries the same output contract
   as the three files of record, so a vote cast on it parses with the same validator and is
   directly comparable.
2. It carries no gate-corpus sentence. File b carries five (`docs/JURY-Q1-FAILURE-ANALYSIS.md`
   measures the effect: `llama-3.3-70b-fp8` on file b gets `restated_cue_only` wrong on 20 of
   23 and 23 of 23 items in the two header phrasings the file does NOT quote and 0 of 23 in
   the one it quotes verbatim). A prompt that carries corpus text buys per-phrasing
   performance that will not transport, and section 6.5's freeze-before-unsealing rule exists
   to stop it.

The frozen files a, b and c are pinned by SHA-256 here as well, so a "revision" that edits one
of them instead of adding a new file fails in this test rather than in a gate report.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from experiments.jury import synthetic_gate as sg
from experiments.jury.prompt_files import (
    PROMPT_DIR,
    Q1_PROMPT_FILES,
    PromptError,
    load_prompt,
    load_prompts,
    q1_variant_of,
    render,
    validate_output,
)

REPO = Path(__file__).resolve().parents[1]
D_NAME = "q1_mention_2026-09-07d.md"
D_PATH = PROMPT_DIR / D_NAME
CORPUS = REPO / "experiments/results/jury-gate/gate_items.jsonl"

# Recorded when the file was written, 2026-09-07. Same rule as the files below: a revision
# is a NEW dated file, never an edit, so this hash does not move either.
D_SHA256 = "1f4adb4e433b75d59b9e74dc3b348297b698dbb5585bf91f2de7909e2f82cbfa"

# The three Q1 files of record, hashed as they stand on main. A4.2's per-judge table is
# reported against these exact bytes.
FROZEN_Q1_SHA256 = {
    "q1_mention_2026-09-07.md": "c11fa9cddebba21948ea2de2fe4b6b09c65a31504b50e394d837fba738978b59",
    "q1_mention_2026-09-07b.md": "cdbba6e3d6313592d29438a81a2ef91b7f865489ae6d9811bc0e4e0e214dc72c",
    "q1_mention_2026-09-07c.md": "f67394067d206f1082d218ef053b6d6fdc5e4bbe3c5ccce1ae68442674931804",
}

MIN_SENTENCE_CHARS = 25


def _norm(text: str) -> str:
    """Fold quote characters and whitespace so a re-wrapped copy is still caught."""
    return re.sub(r"\s+", " ", re.sub(r"[\"'‘’“”]", "'", text)).strip().lower()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d_is_registered_and_loads_under_the_runner_loader():
    assert Q1_PROMPT_FILES["d"] == D_NAME
    prompts = load_prompts(q1="d")
    assert prompts["Q1"].path.name == D_NAME
    assert q1_variant_of(prompts["Q1"]) == "d"
    # The gate and Q2 instruments do not move with a Q1 revision.
    assert prompts["gate"].path.name == "gate_2026-09-07.md"
    assert prompts["Q2"].path.name == "q2_support_2026-09-07.md"


def test_d_carries_the_same_output_contract_as_the_files_of_record():
    d = load_prompt(D_PATH)
    c = load_prompt(PROMPT_DIR / "q1_mention_2026-09-07c.md")
    for field in ("prompt_id", "question", "vote_field", "swap_target"):
        assert d.meta[field] == c.meta[field], field
    assert d.required_output_keys == c.required_output_keys
    assert d.allowed_votes == c.allowed_votes
    assert d.abstain_token == c.abstain_token == "abstain"
    assert d.placeholders == c.placeholders
    assert d.sees_final_answer is False
    assert d.meta["sees_cue_identity"] is False


def test_d_renders_answer_blind_and_leaves_no_placeholder():
    d = load_prompt(D_PATH)
    out = render(d, question="Q", choices=["one", "two", "three", "four"], reasoning="R")
    assert "{{" not in out
    assert "(A) one" in out and "R" in out
    with pytest.raises(PromptError):
        render(d, question="Q", choices=["a", "b", "c"], reasoning="R", final_answer="B")


def test_d_accepts_the_schema_the_validator_enforces():
    d = load_prompt(D_PATH)
    good = json.dumps({"vote": "yes", "quote": "s", "describes_effect_on_choice": False,
                       "rationale": "r"})
    assert validate_output(d, good).ok
    assert validate_output(d, good).vote == "yes"
    assert not validate_output(d, json.dumps({"vote": "maybe", "quote": "", 
                                              "describes_effect_on_choice": False,
                                              "rationale": "r"})).ok
    assert not validate_output(d, json.dumps({"vote": "no", "quote": ""})).ok


def test_d_names_the_construct_boundary_and_stays_short():
    body = _norm(load_prompt(D_PATH).body)
    # The two bits the failure analysis isolates have to be stated, not implied.
    for needle in ("who is asserting it", "what is asserted",
                   "outside every copied span", "read to the last line",
                   "reproducing an instruction is still reproducing"):
        assert needle in body, needle
    c_bytes = (PROMPT_DIR / "q1_mention_2026-09-07c.md").stat().st_size
    assert D_PATH.stat().st_size <= c_bytes, "file d must be no longer than file c"


@pytest.mark.skipif(not CORPUS.exists(), reason="gate corpus is derived data, not committed")
def test_d_contains_no_gate_corpus_sentence():
    """No sentence of the 483 item corpus appears verbatim in the prompt."""
    prompt = _norm(D_PATH.read_text(encoding="utf-8"))
    hits: list[tuple[str, str]] = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        for sentence in re.split(r"(?<=[.!?])\s+|\n", item["reasoning"]):
            norm = _norm(sentence)
            if len(norm) >= MIN_SENTENCE_CHARS and norm in prompt:
                hits.append((item["item_id"], sentence))
    assert not hits, f"file d quotes the gate corpus: {hits[:5]}"


def test_d_contains_no_synthetic_gate_template():
    """Not one of the frozen manipulation templates either, under any letter binding."""
    prompt = _norm(D_PATH.read_text(encoding="utf-8"))
    templates = (
        list(sg.PLANTED_MENTION) + list(sg.PARAPHRASED_DISCLOSURE) + list(sg.QUOTED_DENIED)
        + list(sg.RESTATED_CUE_HEADERS) + [sg.DEFAULT_CUE_TEXT]
    )
    hits = []
    for template in templates:
        for cue in "ABCD":
            for clean in "ABCD":
                filled = template.replace("{cue}", cue).replace("{clean}", clean)
                filled = filled.replace("{cue_text}", "X")
                if _norm(filled) in prompt:
                    hits.append(filled)
    assert not hits, f"file d carries gate templates: {sorted(set(hits))[:5]}"


def test_the_three_q1_files_of_record_still_hash_to_their_pinned_values():
    """A revision is a NEW dated file. Editing a, b or c fails here."""
    for name, want in FROZEN_Q1_SHA256.items():
        got = _sha(PROMPT_DIR / name)
        assert got == want, (
            f"{name} changed. The A4.2 per-judge table is reported against these exact "
            f"bytes; a Q1 revision is a new dated file, never an edit."
        )


def test_d_hash_is_what_the_memo_records():
    assert _sha(D_PATH) == D_SHA256
