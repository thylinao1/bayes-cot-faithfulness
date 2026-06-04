"""Retro-audit: apply the statistical guardrails to every saved control run.

The real-model control reports a follow / silent-unfaithful rate on a clean-correct
subset, but a rate alone hides two things a sibling A/B-testing project would never
ship without:

  1. Whether attrition is BALANCED. Filtering items to clean-correct drops some of
     every arm. If the clean and hinted arms lose different fractions, the surviving
     contrast is biased the way a sample-ratio mismatch biases an A/B lift. The
     summaries here run the clean and hinted arms on the SAME surviving clean-correct
     items, so per-arm survival is equal by construction; the genuine attrition step
     is the clean-arm filter (n_items -> n_clean_correct), and we report and test it.

  2. Whether a 0% follow rate means anything. A measured 0 out of 11 is consistent
     with a true follow rate as high as ~24% (one-sided 95%). Reporting the rate without
     its upper confidence bound and the minimum detectable rate dresses an underpowered
     null as robustness.

This script loads every results/**/control_summary_*.json (including the labelset/
subdir), recovers the counts, runs srm_test / attrition_balance / proportion_ci_upper
/ minimum_detectable_rate, and prints a compact verdict per run. Nothing here calls a
model or the network.

    PYTHONPATH=src python experiments/07_guardrail_audit.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_cot_faithfulness.guardrails import (  # noqa: E402
    attrition_balance,
    minimum_detectable_rate,
    proportion_ci_upper,
    rule_of_three_upper,
    srm_test,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# A run is "adequately powered" for this purpose if the smallest follow rate it could
# detect (80% power, one-sided, against a 0% baseline) is at most this. Stated up front
# so the verdict is not reverse-engineered from the numbers.
MDE_TARGET = 0.10


@dataclass(frozen=True)
class RunFacts:
    """The counts recovered from one control-summary file."""

    label: str
    model: str
    n_items: int | None
    n_clean_correct: int
    n_follow: int
    n_silent: int
    follow_rate: float
    silent_rate: float
    breakdown_rho_star: float | None
    verdict: str | None


def _recover_follow_count(summary: dict, n: int) -> int:
    """Best-effort recovery of the hint-follow count from a summary dict."""
    if summary.get("n_followed_hint") is not None:
        return int(summary["n_followed_hint"])
    rate = summary.get("follow_rate")
    if rate is None or n == 0:
        return 0
    return int(round(float(rate) * n))


def _recover_silent_count(summary: dict, n: int) -> int:
    """Best-effort recovery of the silent-unfaithful count from a summary dict."""
    if summary.get("n_silent_unfaithful") is not None:
        return int(summary["n_silent_unfaithful"])
    rate = summary.get("silent_unfaithful_rate")
    if rate is None or n == 0:
        return 0
    return int(round(float(rate) * n))


def load_run(path: Path) -> RunFacts:
    """Parse one control-summary file into the counts the guardrails need."""
    summary = json.loads(path.read_text())
    n = int(summary.get("n_clean_correct") or 0)
    label = str(path.relative_to(RESULTS))
    return RunFacts(
        label=label,
        model=str(summary.get("model", "?")),
        n_items=summary.get("n_items"),
        n_clean_correct=n,
        n_follow=_recover_follow_count(summary, n),
        n_silent=_recover_silent_count(summary, n),
        follow_rate=float(summary.get("follow_rate") or 0.0),
        silent_rate=float(summary.get("silent_unfaithful_rate") or 0.0),
        breakdown_rho_star=summary.get("breakdown_rho_star"),
        verdict=summary.get("verdict"),
    )


def find_summaries() -> list[Path]:
    """Every control_summary_*.json under results/, including subdirectories."""
    return sorted(RESULTS.rglob("control_summary_*.json"))


def report_run(facts: RunFacts) -> None:
    """Print the per-run guardrail block."""
    n = facts.n_clean_correct
    k = facts.n_follow

    print(f"\n=== {facts.label} ===")
    print(f"  model:            {facts.model}")
    items = facts.n_items if facts.n_items is not None else "?"
    print(f"  arms:             {items} items entered clean arm -> {n} clean-correct survived")
    print(f"  positive control: {k}/{n} followed the planted wrong hint "
          f"(follow rate {facts.follow_rate:.1%})")
    print(f"  silent-unfaithful:{facts.n_silent}/{n} "
          f"(rate {facts.silent_rate:.1%})")
    if facts.breakdown_rho_star is not None:
        print(f"  breakdown rho*:   {facts.breakdown_rho_star:.3f}")
    print(f"  recorded verdict: {facts.verdict}")

    # --- attrition / sample-ratio across the two control arms -------------- #
    if facts.n_items and facts.n_items > 0:
        # Both control arms (clean, hinted) run on the SAME clean-correct survivors,
        # and the runner force-completes any unparsed hinted answer, so each arm enters
        # and survives with n. attrition_balance confirms there is no differential
        # dropout BETWEEN the arms to bias the contrast; srm_test confirms the split.
        srm = srm_test((n, n))
        bal = attrition_balance(
            {"clean": n, "hinted": n}, {"clean": n, "hinted": n}
        )
        keep_frac = n / facts.n_items
        print(f"  clean-correct trim: kept {n}/{facts.n_items} = {keep_frac:.0%} "
              f"of entered items (estimand restricted to this subpopulation)")
        print(f"  control-arm balance: clean vs hinted survival equal "
              f"(SRM p={srm.p_value:.2f}, attrition p={bal.p_value:.2f}, "
              f"flagged={srm.flagged or bal.flagged})")

    # --- power on the observed follow count -------------------------------- #
    ci_upper = proportion_ci_upper(k, n) if n else 1.0
    mde = minimum_detectable_rate(n) if n else 1.0
    print(f"  POWER: {k}/{n} follows -> true follow-rate upper bound "
          f"{ci_upper:.1%} (one-sided 95% exact Clopper-Pearson)")
    if k == 0 and n > 0:
        print(f"         rule-of-three approximation: <= {rule_of_three_upper(n):.1%}")
    print(f"         minimum detectable follow-rate at n={n} "
          f"(80% power, one-sided): {mde:.1%}")

    # --- one-line verdict -------------------------------------------------- #
    if n == 0:
        print("  VERDICT: no clean-correct items; nothing to power.")
        return
    powered = mde <= MDE_TARGET
    if k == 0:
        if powered:
            print(f"  VERDICT: 0/{n} follows and n is large enough to detect a "
                  f">= {mde:.1%} rate. The null is INFORMATIVE: low unfaithfulness "
                  "here is supported, not just unobserved.")
        else:
            print(f"  VERDICT: 0/{n} follows -> true rate could be as high as "
                  f"{ci_upper:.1%} (one-sided 95%). This n can only detect a rate >= {mde:.1%}. "
                  "The 0% is consistent with meaningful unfaithfulness -> UNDERPOWERED, "
                  "not demonstrated robustness.")
    else:
        status = "adequately powered" if powered else "UNDERPOWERED"
        print(f"  VERDICT: {k}/{n} follows (rate {facts.follow_rate:.1%}, upper bound "
              f"{ci_upper:.1%}); n detects rates >= {mde:.1%} -> {status}.")


def main() -> int:
    summaries = find_summaries()
    if not summaries:
        print(f"[audit] no control_summary_*.json under {RESULTS}")
        return 1

    print("Guardrail retro-audit of saved control runs")
    print(f"  scanned: {RESULTS}")
    print(f"  found {len(summaries)} summary file(s)")
    print(f"  power target: minimum detectable follow-rate <= {MDE_TARGET:.0%} "
          "counts as adequately powered")

    underpowered: list[str] = []
    for path in summaries:
        facts = load_run(path)
        report_run(facts)
        if facts.n_clean_correct and minimum_detectable_rate(facts.n_clean_correct) > MDE_TARGET:
            underpowered.append(f"{facts.label} (n={facts.n_clean_correct})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  runs audited:     {len(summaries)}")
    print(f"  underpowered (MDE > {MDE_TARGET:.0%}): {len(underpowered)}")
    for label in underpowered:
        print(f"    - {label}")
    print("\n  Takeaway: a 0% follow rate on a small clean-correct n is an upper bound,")
    print("  not a finding. Pre-commit to an n whose minimum detectable rate is small")
    print("  enough to be meaningful, and report that MDE alongside any null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
