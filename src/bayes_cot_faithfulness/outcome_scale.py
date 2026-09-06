"""CONTRACT record fields for the intervention level and the outcome scale.

The Phase-2 CONTRACT says every generation record carries ``intervention_level``
(``logit`` or ``text``) and ``outcome_scale`` (``raw``, ``renormalized_over_letters``,
``logprob_margin`` or ``binary_follow``), that a letter-logprob record additionally
carries ``logprob_source_token``, and that the runner asserts ``outcome_scale`` BEFORE
writing a checkpoint. Round-1 verification failed check 4 because ``intervention_level``
was missing from every summary, so the assertion lives here as a pure function with
offline tests rather than as a comment in the runner.

Why the source token is not optional. The Phase-1 forced-logprob check (jobs 825246 and
825282) found the four answer letters holding 0.00026 of the next-token mass on one probe
and 0.99929 on the other, and proved it was not a chat-template artifact. So a bare
letter logprob is not a probability over the choice set, and a number read off the wrong
token is indistinguishable from a number read off the right one once it is stored alone.
``letter_logprob_fields`` therefore stores three things together and never one without
the others: the RAW letter logprobs, the decoded token each was read off, and the
distribution renormalized over the letter set, plus the raw mass on that set so a reader
can see how much renormalizing had to do.
"""

from __future__ import annotations

import math

# The CONTRACT's closed vocabularies. Anything outside them is a refusal, not a warning.
INTERVENTION_LEVELS = ("logit", "text")
OUTCOME_SCALES = (
    "raw",
    "renormalized_over_letters",
    "logprob_margin",
    "binary_follow",
)

# Which outcome scales belong to which intervention level. A text-level arm reporting a
# logprob margin, or a logit-level arm reporting a binary follow flag, is a mislabelled
# record and the mislabelling would survive into the pooled row estimand, which the plan
# forbids ("Y is stated separately per intervention level and the two are never pooled
# into one row without a stated bridge").
SCALES_BY_LEVEL = {
    "text": ("binary_follow",),
    "logit": ("raw", "renormalized_over_letters", "logprob_margin"),
}


class OutcomeScaleError(ValueError):
    """Raised when a record's intervention level or outcome scale is missing or invalid."""


def check_outcome_scale(intervention_level: object, outcome_scale: object) -> None:
    """Refuse an unknown level, an unknown scale, or a scale that the level cannot carry."""
    if intervention_level not in INTERVENTION_LEVELS:
        raise OutcomeScaleError(
            f"intervention_level must be one of {list(INTERVENTION_LEVELS)}, "
            f"got {intervention_level!r}"
        )
    if outcome_scale not in OUTCOME_SCALES:
        raise OutcomeScaleError(
            f"outcome_scale must be one of {list(OUTCOME_SCALES)}, got {outcome_scale!r}"
        )
    allowed = SCALES_BY_LEVEL[intervention_level]
    if outcome_scale not in allowed:
        raise OutcomeScaleError(
            f"outcome_scale {outcome_scale!r} is not reportable at intervention_level "
            f"{intervention_level!r}; allowed: {list(allowed)}"
        )


def assert_records_scaled(records: list[dict]) -> int:
    """Assert every record carries a valid level and scale; return how many were checked.

    This is the call the runner makes before a checkpoint write. It returns the
    denominator on purpose: "checked 28 records" is the number the verifier reads, and a
    silent pass over an empty list would otherwise look identical to a real check.
    """
    for i, rec in enumerate(records):
        try:
            check_outcome_scale(rec.get("intervention_level"), rec.get("outcome_scale"))
        except OutcomeScaleError as exc:
            raise OutcomeScaleError(f"record {i}: {exc}") from exc
        if rec.get("outcome_scale") in ("raw", "renormalized_over_letters", "logprob_margin"):
            if not rec.get("logprob_source_token"):
                raise OutcomeScaleError(
                    f"record {i}: outcome_scale {rec['outcome_scale']!r} needs a "
                    "logprob_source_token (which token each logprob was read off)"
                )
    return len(records)


def letter_logprob_fields(
    logprobs: dict[str, float] | None,
    tokens: dict[str, str] | None,
    *,
    target_letter: str | None = None,
) -> dict:
    """Build the CONTRACT logprob block from one forced-answer-logprob pass.

    Returns ``answer_logprobs`` (the raw values, exactly as the server gave them),
    ``logprob_source_token`` (the decoded token each value was read off),
    ``renormalized_over_letters`` (the probability distribution over the scored letters),
    ``letter_probability_mass`` (how much of the next-token mass those letters held before
    renormalizing) and, when ``target_letter`` is given and scored,
    ``logprob_margin``: the renormalized log-odds of the target against the best other
    letter. An empty or missing map returns the block with null values rather than a
    fabricated uniform distribution.
    """
    raw = dict(logprobs or {})
    block: dict = {
        "answer_logprobs": raw or None,
        "logprob_source_token": dict(tokens or {}) or None,
        "renormalized_over_letters": None,
        "letter_probability_mass": None,
        "logprob_margin": None,
        "target_letter": target_letter,
    }
    if not raw:
        return block
    mass = sum(math.exp(v) for v in raw.values())
    block["letter_probability_mass"] = mass
    if mass <= 0.0:
        return block
    renorm = {k: math.exp(v) / mass for k, v in raw.items()}
    block["renormalized_over_letters"] = renorm
    if target_letter is not None and target_letter in renorm:
        others = [p for k, p in renorm.items() if k != target_letter]
        best_other = max(others) if others else 0.0
        p_target = renorm[target_letter]
        if p_target > 0.0 and best_other > 0.0:
            block["logprob_margin"] = math.log(p_target) - math.log(best_other)
        elif p_target > 0.0 and not others:
            block["logprob_margin"] = 0.0
    return block
