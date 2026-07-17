"""Channel-split reporting for the acknowledgment detector.

Reasoning models emit two channels: a thinking channel and a visible answer channel.
Scanning only the visible answer inflates the silent-unfaithfulness share by
construction, since a large share of hint-followed cases acknowledge the cue in the
thinking channel while the visible answer stays silent (55.4% in the reference audit,
arXiv:2603.26410). To measure that, the FROZEN acknowledgment detector
(``acknowledges_hint``) must run UNCHANGED, once per channel.

This module only splits the raw text into channels and tallies the results. It never
re-implements or edits the detector regex: the pooled number (the detector over the whole
text) stays the headline frozen-definition figure, and the per-channel split is a
reporting refinement layered on top, never a change to what counts as acknowledgment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from bayes_cot_faithfulness.interventions import acknowledges_hint

# Our own channel-delimiter patterns (NOT the frozen detector). A closed think block, and
# a bare opening tag for the unclosed case. ``\1`` pairs the closing tag to the opening
# word and honors IGNORECASE, so <Think>...</THINK> matches but <think>...</thinking>
# does not (a malformed pair falls through to the unclosed-tag branch).
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<(think|thinking)>", re.IGNORECASE)


@dataclass(frozen=True)
class ChannelSplit:
    """Raw text split into thinking and visible-answer channels.

    ``thinking`` is ``None`` only when no marker was found. ``marker`` is the recognized
    tag word ('think' or 'thinking'), or ``None`` for unmarked text.
    """

    thinking: str | None
    answer_text: str
    marker: str | None


def split_channels(text: str) -> ChannelSplit:
    """Split ``text`` into a thinking channel and a visible-answer channel.

    Recognized, in order: a closed ``<think>...</think>`` or ``<thinking>...</thinking>``
    block (case-insensitive, first block wins, non-greedy so the first closing tag ends
    the block and nested or later blocks are left in the answer channel, not parsed
    recursively); then a lone opening tag with no close. The unclosed-tag convention is
    honest about the ambiguity: everything after the open tag is thinking and the text
    before it is the answer channel, since we cannot know where the model meant the
    thinking to end. A plain-text 'Thinking:' / 'Final answer:' convention is OUT OF
    SCOPE: plain-text delimiters are ambiguous (prose can contain those words) and belong
    to a validated Phase-2 addition, not a guess here. No marker returns the original text
    as the answer channel unchanged.
    """
    block = _THINK_BLOCK_RE.search(text)
    if block:
        answer_text = (text[: block.start()] + text[block.end() :]).strip()
        return ChannelSplit(block.group(2).strip(), answer_text, block.group(1).lower())
    open_tag = _THINK_OPEN_RE.search(text)
    if open_tag:
        thinking = text[open_tag.end() :].strip()
        answer_text = text[: open_tag.start()].strip()
        return ChannelSplit(thinking, answer_text, open_tag.group(1).lower())
    return ChannelSplit(None, text, None)


@dataclass(frozen=True)
class ChannelAck:
    """Acknowledgment of the planted cue, pooled and per channel.

    ``pooled`` is the frozen-definition detector over the whole text (the headline).
    ``thinking`` and ``answer`` are the same detector per channel; ``thinking`` is
    ``None`` when no channel split was available.
    """

    pooled: bool
    thinking: bool | None
    answer: bool
    split_available: bool


def channel_acknowledgment(text: str) -> ChannelAck:
    """Run the frozen detector pooled and, when a split exists, per channel.

    ``pooled`` is ``acknowledges_hint`` over the entire text and is THE frozen-definition
    number that stays the headline. When ``split_channels`` finds a marker, the detector
    also runs on each channel separately (this is the reporting refinement). With no
    marker, the visible answer is the whole text, so ``answer`` equals ``pooled`` and
    ``thinking`` is ``None``.
    """
    pooled = acknowledges_hint(text)
    split = split_channels(text)
    if split.marker is None:
        return ChannelAck(pooled=pooled, thinking=None, answer=pooled, split_available=False)
    thinking_ack = acknowledges_hint(split.thinking or "")
    answer_ack = acknowledges_hint(split.answer_text)
    return ChannelAck(
        pooled=pooled, thinking=thinking_ack, answer=answer_ack, split_available=True
    )


def silent_share_rows(acks: Iterable[ChannelAck]) -> dict:
    """Tally pooled and per-channel acknowledgment counts and shares.

    The pooled (frozen-definition) count is reported FIRST and is the number of record;
    the per-channel breakdown is a reporting refinement over the subset that has a channel
    split, never a change to the detector. Pooled shares are over all rows; the channel
    shares (thinking-only, answer-only, silent-in-both) are over ``n_with_split``, since a
    channel breakdown is only defined where a split exists.
    """
    ack_list = list(acks)
    n = len(ack_list)
    pooled = sum(1 for a in ack_list if a.pooled)
    with_split = [a for a in ack_list if a.split_available]
    n_split = len(with_split)
    thinking_only = sum(1 for a in with_split if a.thinking is True and a.answer is False)
    answer_only = sum(1 for a in with_split if a.answer is True and a.thinking is False)
    silent_both = sum(1 for a in with_split if a.thinking is False and a.answer is False)
    return {
        "n": n,
        "acknowledged_pooled": pooled,
        "acknowledged_pooled_share": pooled / n if n else 0.0,
        "n_with_split": n_split,
        "thinking_only": thinking_only,
        "answer_only": answer_only,
        "silent_both": silent_both,
        "thinking_only_share": thinking_only / n_split if n_split else 0.0,
        "answer_only_share": answer_only / n_split if n_split else 0.0,
        "silent_both_share": silent_both / n_split if n_split else 0.0,
    }
