"""Agreement statistics for two raters with binary labels (stdlib only).

The pre-registration reports Cohen's kappa beside raw agreement, Gwet's AC1 and PABAK,
always together, because kappa alone moves with prevalence. `agreement_stats` is the only
public entry point and returns all four in one call, so no caller can report one alone.

Pairs where either label is None are left out. Everything is computed from integer counts
and exact fractions, so degenerate cases are decided on counts, never on float equality.

Undefined cases are explicit:
- no paired items: every statistic is None;
- both raters gave one and the same label to every item: kappa is None (its chance
  agreement is 1, so it divides by zero) and `notes` says why. The other three are defined.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

NOTE_NO_PAIRS = "no paired items, every statistic is undefined"
NOTE_ONE_LABEL = "kappa undefined: both raters used one label for every item"


@dataclass(frozen=True)
class AgreementStats:
    """All four statistics for one rater pair, with the counts behind them."""

    n: int  # paired items (both labels present)
    agree: int  # numerator of raw agreement
    p_o: float | None  # raw agreement, agree / n
    kappa: float | None  # Cohen's kappa, None when undefined
    pabak: float | None  # 2 * p_o - 1
    ac1: float | None  # Gwet's AC1
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Counts:
    n: int
    both_yes: int
    both_no: int
    a_yes: int  # items where rater A said yes
    b_yes: int  # items where rater B said yes

    @property
    def agree(self) -> int:
        return self.both_yes + self.both_no

    @property
    def one_label(self) -> bool:
        """Both raters used the same single label on every item (p_e == 1)."""
        return self.n > 0 and self.agree == self.n and self.a_yes in (0, self.n)


def _counts(a: Sequence[bool | None], b: Sequence[bool | None]) -> _Counts:
    # Unequal lengths would silently score a prefix, so they are refused.
    if len(a) != len(b):
        raise ValueError(f"the two raters have {len(a)} and {len(b)} labels; they must be paired")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return _Counts(
        n=len(pairs),
        both_yes=sum(1 for x, y in pairs if x and y),
        both_no=sum(1 for x, y in pairs if not x and not y),
        a_yes=sum(1 for x, _ in pairs if x),
        b_yes=sum(1 for _, y in pairs if y),
    )


def _p_o(c: _Counts) -> Fraction:
    return Fraction(c.agree, c.n)


def _kappa(c: _Counts) -> Fraction | None:
    if c.n == 0 or c.one_label:
        return None
    p_o = _p_o(c)
    p1, p2 = Fraction(c.a_yes, c.n), Fraction(c.b_yes, c.n)
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    return (p_o - p_e) / (1 - p_e)


def _pabak(c: _Counts) -> Fraction | None:
    return None if c.n == 0 else 2 * _p_o(c) - 1


def _ac1(c: _Counts) -> Fraction | None:
    if c.n == 0:
        return None
    pi = (Fraction(c.a_yes, c.n) + Fraction(c.b_yes, c.n)) / 2
    p_e_g = 2 * pi * (1 - pi)
    # With two categories p_e_g peaks at 1/2, so the denominator never reaches zero.
    assert p_e_g <= Fraction(1, 2)
    return (_p_o(c) - p_e_g) / (1 - p_e_g)


def _f(x: Fraction | None) -> float | None:
    return None if x is None else float(x)


def agreement_stats(
    a: Sequence[bool | None], b: Sequence[bool | None]
) -> AgreementStats:
    """Raw agreement, Cohen's kappa, PABAK and Gwet's AC1, together, with n."""
    c = _counts(a, b)
    notes: list[str] = []
    if c.n == 0:
        notes.append(NOTE_NO_PAIRS)
    elif c.one_label:
        notes.append(NOTE_ONE_LABEL)
    return AgreementStats(
        n=c.n,
        agree=c.agree,
        p_o=None if c.n == 0 else float(_p_o(c)),
        kappa=_f(_kappa(c)),
        pabak=_f(_pabak(c)),
        ac1=_f(_ac1(c)),
        notes=tuple(notes),
    )
