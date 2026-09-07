"""Ruling R12: the reasoning-mode switch for roster rows whose template documents none.

Element 9.4 as frozen says a model that DOCUMENTS a reasoning-mode switch runs both
settings as an additive arm. Four roster rows document none, and their chat templates
open a reasoning block that the frozen 320-token budget cuts before an answer appears
(``docs/WAVE1-AUDIT.md``: OLMo-3-7B-Think 1,381/1,500 clean outputs unparseable,
R1-Distill-Llama-8B 1,329/1,500). R12 DEFINES the switch for those rows as closing the
block the template opens, and names three settings this module carries:

``default``  today's behaviour, byte for byte: the ordinary ``/chat/completions`` path,
             ``num_predict`` 320 for full generations and 24 for forced continuations.
             Nothing about a default run changes because this module exists.
``off``      the CELL OF RECORD of R12(1). The prompt is rendered through the model's own
             chat template (vLLM ``/tokenize`` plus ``/detokenize``), the reasoning block
             the template opens is CLOSED, and generation goes through ``/completions``
             with every element 15 constant unchanged. The request path moves, which is
             why it is recorded per record and why R12(2) gates it behind a two-path
             equivalence run.
``on``       the ADDITIVE ARM of R12(1), exploratory: the ordinary chat path, the block
             allowed, ``num_predict`` 4,096 for FULL generations on that row only (the
             forced-continuation budget of 24 does not move), and the answer read from
             the text AFTER the closing tag. The mediator is still computed on the full
             returned chain.

Nothing here edits a frozen element. The answer-extraction regexes of
``interventions.py`` are imported and applied UNCHANGED; this module only decides which
slice of the text they are applied to, which is the wrapper R12 asks for.
"""

from __future__ import annotations

import re

from .interventions import parse_answer

# The three settings, in the order they are reported.
REASONING_MODES = ("default", "off", "on")

# Element 15 constants (PREREGISTRATION_jury_and_scale.md section 16). They are named
# here so the "on" override can be stated as a difference from them rather than as a
# second copy of the numbers.
NUM_PREDICT_FULL = 320
FORCE_TOKENS = 24
# R12(1): 4,096 on the "on" row only. It is not a measured bound, and the serving test
# says so: on Olmo-3-7B-Think 6 of 30 generations still hit it (docs/REASONING-MODE-TEST.md).
ON_NUM_PREDICT_FULL = 4096

# The tags. Written as a tolerant pattern (whitespace, the three spellings the roster's
# templates use) because it reads MODEL OUTPUT, not a frozen prompt.
OPEN_THINK_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"<\s*/\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)

# DeepSeek's own documented way of making an R1 model skip thinking, used where the
# template opens no block of its own.
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
# The tail appended where the template ALREADY left a block open.
CLOSE_THINK_TAIL = "\n\n</think>\n\n"

# Which request path each mode generates through.
PATH_BY_MODE = {"default": "chat", "off": "completions", "on": "chat"}

# The two branches of the prompt close, named so a record can say which one fired. The
# spellings match experiments/audit/serving_test.py, the implementation the R12 serving
# numbers were measured with, so a rendered tail from either can be compared with a tail
# from the other.
BRANCH_CLOSED_OPEN_BLOCK = "closed_the_block_the_template_opened"
BRANCH_INSERTED_EMPTY_BLOCK = "inserted_a_whole_empty_block"


class ReasoningModeError(ValueError):
    """Raised for an unknown reasoning mode, or a record that does not carry the fields."""


def check_mode(mode: object) -> str:
    """Return the mode, or refuse. An unknown mode must never fall back to a default."""
    if mode not in REASONING_MODES:
        raise ReasoningModeError(
            f"reasoning_mode must be one of {list(REASONING_MODES)}, got {mode!r}"
        )
    return str(mode)


def path_for(mode: str) -> str:
    """The request path a mode generates through: 'chat' or 'completions'."""
    return PATH_BY_MODE[check_mode(mode)]


def full_num_predict(mode: str, base: int = NUM_PREDICT_FULL) -> int:
    """The FULL-generation budget under ``mode``; the forced budget never moves.

    R12(1) raises it to 4,096 for reasoning_mode "on" and on that row only. "default"
    and "off" keep whatever the cell was submitted with, which is the element 15 constant
    320 for every cell of record.
    """
    return ON_NUM_PREDICT_FULL if check_mode(mode) == "on" else int(base)


def close_reasoning_block(rendered: str) -> tuple[str, str]:
    """Close the reasoning block in a RENDERED prompt. Returns (prompt, branch).

    Two branches, exactly as ``experiments/audit/serving_test.py`` took them:

    * the template already ended the prompt with an OPEN block (Olmo-3-7B-Think ends
      ``<|im_start|>assistant\\n<think>``, R1-Distill-Llama-8B ends
      ``<|Assistant|><think>\\n``): append the closing tag only, which is the model's own
      mechanism used as intended;
    * the template opened nothing (R1-0528 ends ``<|Assistant|>``, Phi-4-reasoning ends
      ``<|im_start|>assistant<|im_sep|>``): insert the whole empty block.

    "Already open" means an opening tag with no closing tag after it. A template that
    renders a CLOSED block itself falls into the second branch and gets an empty block
    appended, which is still a closed block and still suppresses the chain.
    """
    text = rendered or ""
    open_tag = OPEN_THINK_RE.search(text)
    close_tag = CLOSE_THINK_RE.search(text)
    if open_tag and not (close_tag and close_tag.start() > open_tag.start()):
        return text + CLOSE_THINK_TAIL, BRANCH_CLOSED_OPEN_BLOCK
    return text + EMPTY_THINK_BLOCK, BRANCH_INSERTED_EMPTY_BLOCK


def reopened_block(text: str) -> bool:
    """True when a completion OPENS a reasoning block it never closes.

    ``docs/WAVE1-AUDIT.md``: Phi-4-reasoning's 24-token forced continuations reopen a
    block, which is why that model's mediator arm came back empty (direct 0 scorable of
    1,197). Under reasoning_mode "off" the continuation prompt is closed like any other,
    so this is the check that says whether the close held.
    """
    body = text or ""
    open_tag = OPEN_THINK_RE.search(body)
    if not open_tag:
        return False
    close_tag = CLOSE_THINK_RE.search(body, open_tag.end())
    return close_tag is None


def strip_reasoning_block(text: str) -> tuple[str, bool]:
    """Return (the text after the LAST closing tag, whether a closing tag existed).

    ``closed`` False means the generation never left the reasoning block. The text is
    then returned unchanged and the caller decides; ``parse_answer_after_think`` calls
    that unparseable, because under reasoning_mode "on" the OPENING tag lives in the
    rendered PROMPT, not in the completion, so an unterminated chain carries no marker of
    its own and frozen-parsing it would read a letter out of the middle of the reasoning.
    """
    body = text or ""
    hits = list(CLOSE_THINK_RE.finditer(body))
    if not hits:
        return body, False
    return body[hits[-1].end():], True


def parse_answer_after_think(text: str, n_choices: int = 5) -> str | None:
    """The reasoning_mode "on" extractor: strip the block, then run the FROZEN parser.

    A superset of the frozen parser in the sense R12 asks for, and in no other: the
    frozen regexes are imported and applied unchanged, to a SUFFIX of the text, so every
    letter this returns is a letter ``parse_answer`` returns on the text it was given.
    Nothing in ``interventions.py`` is edited, and the frozen parser stays the parser of
    record for every cell that is not reasoning_mode "on".

    Returns None when no closing tag exists: an answer read from inside an unterminated
    chain is not the model's stated answer.
    """
    tail, closed = strip_reasoning_block(text)
    if not closed:
        return None
    return parse_answer(tail, n_choices)


def record_fields(mode: str, *, path: str | None = None,
                  block_closed: bool | None = None,
                  num_predict_full_value: int | None = None) -> dict:
    """The four fields R12 puts on EVERY generation record.

    ``reasoning_block_closed`` is a bool only where closing a block is what the mode
    does (``off``); it is None under ``default`` and ``on``, where nothing was closed,
    rather than False, which would read as "we tried and it did not hold".
    """
    checked = check_mode(mode)
    return {
        "reasoning_mode": checked,
        "reasoning_path": path or path_for(checked),
        "reasoning_block_closed": block_closed,
        "num_predict_full": (num_predict_full_value if num_predict_full_value is not None
                             else full_num_predict(checked)),
    }


REASONING_RECORD_FIELDS = (
    "reasoning_mode", "reasoning_path", "reasoning_block_closed", "num_predict_full",
)


def assert_records_reasoning_mode(records: list[dict]) -> int:
    """Assert every record carries the four fields; return how many were checked.

    The same shape and the same discipline as ``outcome_scale.assert_records_scaled``:
    the runner calls it BEFORE a checkpoint write and it REFUSES rather than warns,
    because a record with no reasoning_mode would be pooled with a default-mode cell
    downstream and R12 forbids exactly that pooling. It returns the denominator on
    purpose: a silent pass over an empty list must not look like a real check.
    """
    for i, rec in enumerate(records):
        try:
            check_mode(rec.get("reasoning_mode"))
        except ReasoningModeError as exc:
            raise ReasoningModeError(f"record {i}: {exc}") from exc
        path = rec.get("reasoning_path")
        if path not in ("chat", "completions"):
            raise ReasoningModeError(
                f"record {i}: reasoning_path must be 'chat' or 'completions', got {path!r}"
            )
        closed = rec.get("reasoning_block_closed", "absent")
        if closed not in (True, False, None):
            raise ReasoningModeError(
                f"record {i}: reasoning_block_closed must be True, False or None, "
                f"got {closed!r}"
            )
        budget = rec.get("num_predict_full")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ReasoningModeError(
                f"record {i}: num_predict_full must be a positive int, got {budget!r}"
            )
    return len(records)
