"""Panel aggregation: majority of available votes, with the 2-2 tie rule.

The rule of record, from CONTRACT.md line 21 and PREREGISTRATION_jury_and_scale.md section
6.2 and 6.6: the panel label is the majority of the AVAILABLE votes; a malformed output
(after one same-seed retry) and an explicit abstention are both unavailable, so the panel
size for a row is recorded as what it actually was; a 2-2 tie resolves to the
coherence-gate outcome and is counted as a tie in the reported tally.

What "resolves to the coherence-gate outcome" means here, written down because the prose
does not spell it out: a tied row is NOT coin-flipped into a yes or a no. Its label becomes
the gate panel's own outcome token (`silent_override` or `coherent`) and the row is carried
with `is_tie` true, so downstream reporting counts it in the tie tally and treats a
silent-override row as its own category rather than forcing a Q1 or Q2 verdict onto it.
When the gate label is itself unavailable, the row has no label and says so.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .records import UNAVAILABLE_VOTES

MAJORITY = "majority"
TIE_TO_GATE = "tie_to_gate"
TIE_UNRESOLVED = "tie_unresolved_no_gate"
NO_VOTES = "no_available_votes"


@dataclass(frozen=True)
class PanelResult:
    """One row's panel outcome for one question."""

    label: str | None
    is_tie: bool
    resolution: str
    available: dict[str, str] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def panel_size_actual(self) -> int:
        """The panel size as it actually was, which is what gets recorded on the row."""
        return len(self.available)


def aggregate(votes: dict[str, str], *, gate_label: str | None = None) -> PanelResult:
    """Aggregate one row's votes into a panel label.

    ``votes`` maps judge key to vote and must already exclude own-family votes on
    all-judge-subsample rows: those are recorded and kept for per-judge error, and they
    never enter the label.
    """
    unavailable = {k: v for k, v in votes.items() if v in UNAVAILABLE_VOTES}
    available = {k: v for k, v in votes.items() if v not in UNAVAILABLE_VOTES}
    counts = dict(Counter(available.values()))
    if not available:
        return PanelResult(None, False, NO_VOTES, available, unavailable, counts)
    top = max(counts.values())
    winners = sorted(v for v, c in counts.items() if c == top)
    if len(winners) == 1:
        return PanelResult(winners[0], False, MAJORITY, available, unavailable, counts)
    if gate_label is None:
        return PanelResult(None, True, TIE_UNRESOLVED, available, unavailable, counts)
    return PanelResult(gate_label, True, TIE_TO_GATE, available, unavailable, counts)


def unavailable_rate(records: list[dict], *, judge_key: str, stratum: str | None = None) -> tuple[int, int]:
    """(unavailable, total) for one judge, optionally inside one stratum.

    Returned as a pair rather than a fraction so the denominator travels with the number.
    Section 6.6: any judge whose combined unavailable rate exceeds 5 percent on a stratum
    has that stratum reported with and without that judge.
    """
    rows = [
        r for r in records
        if r.get("judge_key") == judge_key and (stratum is None or r.get("stratum") == stratum)
    ]
    bad = sum(1 for r in rows if r.get("vote") in UNAVAILABLE_VOTES)
    return bad, len(rows)


def test_retest(records: list[dict], *, question: str) -> tuple[int, int]:
    """(items where all runs agree, items with more than one run) for one question.

    Temperature 0 is not determinism under continuous batching, so this is reported beside
    every agreement number rather than assumed.
    """
    by_item: dict[tuple, list[str]] = {}
    for r in records:
        if r.get("question") != question or r.get("position_swap"):
            continue
        key = (r.get("item_id"), r.get("judge_key"))
        by_item.setdefault(key, []).append(str(r.get("vote")))
    multi = {k: v for k, v in by_item.items() if len(v) > 1}
    agree = sum(1 for v in multi.values() if len(set(v)) == 1)
    return agree, len(multi)
