"""Load, render and validate the dated jury prompt files.

One file per question (gate, Q1, Q2). Each file carries a JSON front-matter block between
a line reading ``---json`` and a line reading ``---``, then the prompt body with
``{{PLACEHOLDER}}`` tokens. The SHA-256 recorded in every vote is of the WHOLE file bytes,
front matter included, because the swap policy and the blinding declaration are part of the
frozen instrument and not commentary about it.

Blinding is enforced here rather than trusted to the prompt text. A prompt whose front
matter says ``sees_final_answer: false`` cannot be rendered with a final answer: the
renderer refuses instead of quietly substituting an empty string, so an answer-blind judge
staying answer-blind is a property of the code, not of whoever wrote the template.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# The dated files of the candidate primary configuration. The freeze commit is the
# operator's; this constant only says which files the runner loads by default.
DEFAULT_PROMPT_FILES: dict[str, str] = {
    "gate": "gate_2026-09-07.md",
    "Q1": "q1_mention_2026-09-07.md",
    "Q2": "q2_support_2026-09-07.md",
}

# Every Q1 file that has been written and scored, oldest first. A revision is a NEW dated
# file, never an edit: the first file was scored by the FP8 70B judge in job 825542 and its
# result is on the record, so its bytes and its SHA-256 have to stay what that report says
# they were. Adding a name here does not change what the runner loads by default; it says
# which files exist and lets a run name the one it used.
#
#   a  the file of record. States the Chua and Evans non-repetition rule as the RULE, so a
#      vote of yes needs a sentence about the cue's effect on the choice.
#   b  states section 6.4's Occhipinti mention construct as the rule and keeps
#      non-repetition as a guard on it.
Q1_PROMPT_FILES: dict[str, str] = {
    "a": "q1_mention_2026-09-07.md",
    "b": "q1_mention_2026-09-07b.md",
}

_FRONT_OPEN = "---json"
_FRONT_CLOSE = "---"


class PromptError(ValueError):
    """Raised when a prompt file is malformed or is rendered outside its declared blinding."""


@dataclass(frozen=True)
class JuryPrompt:
    """One dated prompt file, parsed."""

    path: Path
    sha256: str
    body: str
    meta: dict

    @property
    def prompt_id(self) -> str:
        return str(self.meta["prompt_id"])

    @property
    def question(self) -> str:
        return str(self.meta["question"])

    @property
    def vote_field(self) -> str:
        return str(self.meta["vote_field"])

    @property
    def allowed_votes(self) -> tuple[str, ...]:
        return tuple(self.meta["allowed_votes"])

    @property
    def abstain_token(self) -> str:
        return str(self.meta["abstain_token"])

    @property
    def swap_target(self) -> str:
        return str(self.meta["swap_target"])

    @property
    def sees_final_answer(self) -> bool:
        return bool(self.meta["sees_final_answer"])

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(self.meta["placeholders"])

    @property
    def required_output_keys(self) -> tuple[str, ...]:
        return tuple(self.meta["required_output_keys"])


_REQUIRED_META = (
    "prompt_id",
    "question",
    "date",
    "vote_field",
    "allowed_votes",
    "abstain_token",
    "swap_target",
    "sees_final_answer",
    "placeholders",
    "required_output_keys",
)


def load_prompt(path: str | Path) -> JuryPrompt:
    """Parse one prompt file and hash its bytes."""
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_OPEN:
        raise PromptError(f"{p}: first line must be {_FRONT_OPEN!r}")
    try:
        close = next(i for i in range(1, len(lines)) if lines[i].strip() == _FRONT_CLOSE)
    except StopIteration as exc:
        raise PromptError(f"{p}: front matter is never closed with {_FRONT_CLOSE!r}") from exc
    try:
        meta = json.loads("\n".join(lines[1:close]))
    except json.JSONDecodeError as exc:
        raise PromptError(f"{p}: front matter is not valid JSON: {exc}") from exc
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing:
        raise PromptError(f"{p}: front matter is missing {missing}")
    if meta["abstain_token"] not in meta["allowed_votes"]:
        raise PromptError(f"{p}: abstain_token is not among allowed_votes")
    body = "\n".join(lines[close + 1 :]).strip("\n")
    for name in meta["placeholders"]:
        if "{{%s}}" % name not in body:
            raise PromptError(f"{p}: declares placeholder {name} but the body does not use it")
    if not meta["sees_final_answer"] and "FINAL_ANSWER" in meta["placeholders"]:
        raise PromptError(f"{p}: declares sees_final_answer false but asks for FINAL_ANSWER")
    return JuryPrompt(path=p, sha256=sha, body=body, meta=meta)


def load_default_prompts(prompt_dir: str | Path = PROMPT_DIR) -> dict[str, JuryPrompt]:
    """The three prompts of the candidate configuration, keyed by question."""
    d = Path(prompt_dir)
    return {q: load_prompt(d / name) for q, name in DEFAULT_PROMPT_FILES.items()}


def load_prompts(
    prompt_dir: str | Path = PROMPT_DIR, *, q1: str = "a"
) -> dict[str, JuryPrompt]:
    """The three prompts with the Q1 file named explicitly.

    `q1` is a key of Q1_PROMPT_FILES or a file name in the prompt directory. The gate and
    Q2 files never vary: only Q1 has a revision, and only because the first one stated the
    guard as the rule. Refuses an unknown name rather than falling back to the default,
    because a silent fallback would put the wrong SHA-256 in a report that says otherwise.
    """
    d = Path(prompt_dir)
    name = Q1_PROMPT_FILES.get(q1, q1)
    path = d / name
    if not path.exists():
        raise PromptError(
            f"no Q1 prompt file {name!r} in {d}; known keys {sorted(Q1_PROMPT_FILES)}, "
            f"known files {sorted(Q1_PROMPT_FILES.values())}"
        )
    out = load_default_prompts(d)
    out["Q1"] = load_prompt(path)
    return out


def q1_variant_of(prompt: JuryPrompt) -> str:
    """Which Q1 file this is, by name, for a report that has to say so."""
    for key, name in Q1_PROMPT_FILES.items():
        if prompt.path.name == name:
            return key
    return prompt.path.name


# --- rendering ------------------------------------------------------------------

LETTERS = "ABCDEFGH"


def option_block(choices: list[str] | tuple[str, ...], *, position_swap: bool = False) -> str:
    """Render the option list. The swap changes DISPLAY ORDER, never the letter binding.

    Letters stay bound to their choice text, so a vote of "C" means the same option in both
    passes and the two passes are directly comparable. What moves is which option the judge
    reads first, which is the position effect being measured.
    """
    pairs = [(LETTERS[i], c) for i, c in enumerate(choices)]
    if position_swap:
        pairs = list(reversed(pairs))
    return "\n".join(f"({letter}) {text}" for letter, text in pairs)


def category_block(categories: list[str] | tuple[str, ...], *, position_swap: bool = False) -> str:
    ordered = list(reversed(list(categories))) if position_swap else list(categories)
    return "\n".join(f"- {c}" for c in ordered)


def render(
    prompt: JuryPrompt,
    *,
    question: str,
    choices: list[str] | tuple[str, ...],
    reasoning: str,
    final_answer: str | None = None,
    position_swap: bool = False,
) -> str:
    """Fill one prompt for one transcript.

    Refuses, rather than silently blinding, when a caller hands a final answer to a prompt
    declared answer-blind. That mistake is the exact contamination path FACE-Eval's
    disjoint-input rule exists to block, so it fails loudly.
    """
    if not prompt.sees_final_answer and final_answer is not None:
        raise PromptError(
            f"{prompt.prompt_id} is answer-blind (sees_final_answer false) but a final "
            f"answer was passed to render(); refusing to build the prompt"
        )
    if prompt.sees_final_answer and final_answer is None:
        raise PromptError(f"{prompt.prompt_id} needs FINAL_ANSWER and none was passed")
    if position_swap and prompt.swap_target == "none":
        raise PromptError(
            f"{prompt.prompt_id} declares swap_target 'none' (its question is binary) "
            f"so a position swap is not defined for it"
        )
    values = {
        "QUESTION": question,
        "OPTIONS": option_block(choices, position_swap=position_swap and prompt.swap_target == "options"),
        "REASONING": reasoning,
    }
    if prompt.sees_final_answer:
        values["FINAL_ANSWER"] = str(final_answer)
    if "CATEGORY_BLOCK" in prompt.placeholders:
        values["CATEGORY_BLOCK"] = category_block(
            prompt.meta.get("categories", ()),
            position_swap=position_swap and prompt.swap_target == "categories",
        )
    out = prompt.body
    for name in prompt.placeholders:
        if name not in values:
            raise PromptError(f"{prompt.prompt_id}: no value for placeholder {name}")
        out = out.replace("{{%s}}" % name, str(values[name]))
    if "{{" in out:
        raise PromptError(f"{prompt.prompt_id}: unfilled placeholder left in the rendered prompt")
    return out


# --- output validation ----------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of parsing one judge response."""

    ok: bool
    vote: str | None
    parsed: dict | None
    error: str | None


def _extract_json(text: str) -> dict:
    """Take the first balanced top-level JSON object out of a response.

    Judges wrap JSON in prose or a code fence often enough that a bare json.loads throws
    away usable votes, and a retry costs a full generation. Scanning for the first balanced
    braces is strict about the object itself and forgiving about what surrounds it.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(text[start : i + 1])
    raise json.JSONDecodeError("no balanced JSON object in the response", text, 0)


def validate_output(prompt: JuryPrompt, text: str) -> ValidationResult:
    """Strict schema check on one judge response. Never raises; the caller decides retries."""
    if text is None or not str(text).strip():
        return ValidationResult(False, None, None, "empty response")
    try:
        obj = _extract_json(str(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return ValidationResult(False, None, None, f"not JSON: {exc}")
    if not isinstance(obj, dict):
        return ValidationResult(False, None, None, "top level is not a JSON object")
    missing = [k for k in prompt.required_output_keys if k not in obj]
    if missing:
        return ValidationResult(False, None, obj, f"missing keys {missing}")
    vote = obj.get(prompt.vote_field)
    if not isinstance(vote, str):
        return ValidationResult(False, None, obj, f"{prompt.vote_field} is not a string")
    vote = vote.strip().lower()
    if vote not in prompt.allowed_votes:
        return ValidationResult(False, None, obj, f"vote {vote!r} not in {list(prompt.allowed_votes)}")
    return ValidationResult(True, vote, obj, None)
