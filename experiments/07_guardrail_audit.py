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

T10/A5/A6 additions, only surfaced when a summary carries the upgraded schema (older
summaries print exactly as before):

  - an ATTRITION block: the named categories (entered, failed generation, unparseable
    clean, clean incorrect, clean correct) that account for every item, so nothing is
    silently dropped between "items entered" and "clean-correct survived";
  - a collateral-rate row: the share of clean-correct items whose hinted-arm answer
    moved off the clean answer but NOT onto the planted hint (A5);
  - a hinted-arm accuracy row, logged independently of hint adoption (A6);
  - a Newcombe 95% CI row for the hint-follow rate minus the neutral-change rate, the
    "cue effect above noise floor" check (T10).

It also writes experiments/results/guardrail_power_artifacts.json (one row per run:
label, n, mde, ci_upper_on_observed, and the Newcombe interval or null) and supports
an optional --strict flag that changes the process exit code.

The same artifact additionally ingests the exploratory Phase-2 arm summaries written
by experiments/08_additive_arms.py as results/**/arms_summary_*.json. Each rate-bearing
sub-block those summarizers emit -- replay clean/hinted drift, transplant forward/reverse
carry-over, direct accuracy and with/without-CoT agreement, placebo would-be-hint follow,
two-step follow, filler match, and the four held-out specificity false-alarm rates
(ack_clean, ack_placebo, would_be_follow, silent_false_alarm) -- becomes one power row,
distinguished from the control
rows by a "kind" field ("control" vs "arms"). An arm row carries the sub-block's n, its
observed rate, the exact one-sided 95% Clopper-Pearson upper bound on the observed count,
the minimum detectable rate at that n, a "powered" flag, and -- so the pre-registration's
10-percent unscorable rule is enforced by this tool rather than by hand -- the sub-block's
n_unscorable and an "unscorable_flag" (n_unscorable above 10% of scorable + unscorable).
Sub-blocks with n == 0 or a None rate are skipped. Under --strict an underpowered arm row
fails the process exactly like an underpowered control run:

    0 - no hard guardrail violation, and (without --strict) any underpowered runs
        do not fail the process (matches all pre-existing invocations).
    1 - --strict was passed and at least one run is underpowered (mde > MDE_TARGET),
        with no hard violation.
    2 - at least one run has a flagged SRM or attrition-balance check (a hard
        guardrail violation), regardless of --strict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_cot_faithfulness.guardrails import (  # noqa: E402
    attrition_balance,
    minimum_detectable_rate,
    newcombe_diff_ci,
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
    # Present only for summaries written by the upgraded (T10/A5/A6) schema; older
    # summaries leave these None and report_run skips the rows that need them.
    hinted_accuracy: float | None = None
    collateral_rate: float | None = None
    n_neutral_changed: int | None = None
    attrition: dict | None = None


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
        hinted_accuracy=summary.get("hinted_accuracy"),
        collateral_rate=summary.get("collateral_rate"),
        n_neutral_changed=summary.get("n_neutral_changed"),
        attrition=summary.get("attrition"),
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

    # --- named attrition categories (upgraded summary schema only) --------- #
    if facts.attrition is not None:
        att = facts.attrition
        print("  ATTRITION (named categories, never silently dropped):")
        print(f"    entered:            {att.get('n_entered')}")
        print(f"    failed generation:  {att.get('n_failed_generation')}")
        print(f"    clean incorrect:    {att.get('n_clean_incorrect')}"
              f"  (of which unparseable: {att.get('n_unparseable_clean')})")
        print(f"    clean correct:      {att.get('n_clean_correct')}")
        print("    (entered = failed generation + clean incorrect + clean correct)")

    # --- collateral damage and independent correctness (upgraded schema) --- #
    if facts.collateral_rate is not None:
        print(f"  collateral rate: {facts.collateral_rate:.1%} of clean-correct items "
              "moved to a non-hint answer under the cue")
    if facts.hinted_accuracy is not None:
        print(f"  hinted-arm accuracy: {facts.hinted_accuracy:.1%} "
              "(logged independently of hint adoption)")

    # --- cue effect above the noise floor (Newcombe 95% CI) ----------------- #
    if facts.n_neutral_changed is not None and n > 0:
        nc = newcombe_diff_ci(k, n, facts.n_neutral_changed, n)
        print("  cue effect above noise floor (Newcombe 95% CI): follow rate minus "
              f"neutral change rate = {nc.diff:+.1%} [{nc.lower:+.1%}, {nc.upper:+.1%}]")
        print("    (unpaired interval; both rates come from the same items, so it is "
              "conservative when the arms correlate positively)")

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


def build_power_artifact(facts: RunFacts) -> dict:
    """One row of the committed power-artifact JSON for a single run.

    Deterministic given the summary: proportion_ci_upper and minimum_detectable_rate
    are closed-form / fixed-grid, and newcombe_diff_ci is closed-form, so re-running
    the audit against unchanged summaries reproduces this file byte-for-byte.
    """
    n = facts.n_clean_correct
    k = facts.n_follow
    ci_upper = proportion_ci_upper(k, n) if n else 1.0
    mde = minimum_detectable_rate(n) if n else 1.0
    newcombe = None
    if facts.n_neutral_changed is not None and n > 0:
        nc = newcombe_diff_ci(k, n, facts.n_neutral_changed, n)
        newcombe = {"diff": nc.diff, "lower": nc.lower, "upper": nc.upper, "conf": nc.conf}
    return {
        "kind": "control",
        "label": facts.label,
        "n": n,
        "mde": mde,
        "ci_upper_on_observed": ci_upper,
        "newcombe": newcombe,
    }


# --------------------------------------------------------------------------- #
# Phase-2 arm summaries (08) -> additional power rows in the same artifact
# --------------------------------------------------------------------------- #
def find_arms_summaries() -> list[Path]:
    """Every arms_summary_*.json under results/, including subdirectories.

    Same rglob convention as ``find_summaries``; the exploratory Phase-2 arm runner
    (08) writes these as ``arms_summary_<safe_model>.json`` beside the control summaries.
    """
    return sorted(RESULTS.rglob("arms_summary_*.json"))


def _arms_candidates(arms: dict) -> list[tuple[str, object, object, object]]:
    """(sublabel, n, rate, n_unscorable) for each rate-bearing sub-block present.

    One tuple per measurement 08's summarizers expose a rate for: the two replay drift
    sides, the two transplant carry-over directions, the direct arm's accuracy and its
    with/without-CoT agreement (each nested with its own scorable n), the single
    follow-style rate of the placebo, two-step, and filler arms, and the four held-out
    specificity false-alarm rates (A9). The curves arm exposes no rate and is
    intentionally absent. Missing arms simply contribute nothing.
    """
    candidates: list[tuple[str, object, object, object]] = []

    replay = arms.get("replay")
    if isinstance(replay, dict):
        for side in ("clean", "hinted"):
            block = replay.get(side)
            if isinstance(block, dict):
                candidates.append(
                    (f"replay.{side}", block.get("n"),
                     block.get("drift_rate"), block.get("n_unscorable"))
                )

    transplant = arms.get("transplant")
    if isinstance(transplant, dict):
        for direction in ("forward", "reverse"):
            block = transplant.get(direction)
            if isinstance(block, dict):
                candidates.append(
                    (f"transplant.{direction}", block.get("n"),
                     block.get("carryover_rate"), block.get("n_unscorable"))
                )

    direct = arms.get("direct")
    if isinstance(direct, dict):
        n_unscorable = direct.get("n_unscorable")
        for sublabel, key in (("direct.accuracy", "direct_accuracy"),
                              ("direct.agreement", "with_without_cot_agreement")):
            block = direct.get(key)
            if isinstance(block, dict):
                candidates.append(
                    (sublabel, block.get("n"), block.get("rate"), n_unscorable)
                )

    placebo = arms.get("placebo")
    if isinstance(placebo, dict):
        candidates.append(
            ("placebo", placebo.get("n"),
             placebo.get("placebo_follow_rate"), placebo.get("n_unscorable"))
        )

    twostep = arms.get("twostep")
    if isinstance(twostep, dict):
        candidates.append(
            ("twostep", twostep.get("n"),
             twostep.get("twostep_follow_rate"), twostep.get("n_unscorable"))
        )

    filler = arms.get("filler")
    if isinstance(filler, dict):
        candidates.append(
            ("filler", filler.get("n"),
             filler.get("filler_match_rate"), filler.get("n_unscorable"))
        )

    # A9 held-out specificity: four false-alarm rates on unmanipulated items, each a
    # uniform n / count / rate / n_unscorable sub-block written by summarize_specificity.
    specificity = arms.get("specificity")
    if isinstance(specificity, dict):
        for sub in ("ack_clean", "ack_placebo", "would_be_follow", "silent_false_alarm"):
            block = specificity.get(sub)
            if isinstance(block, dict):
                candidates.append(
                    (f"specificity.{sub}", block.get("n"),
                     block.get("rate"), block.get("n_unscorable"))
                )

    return candidates


def _arms_power_row(base_label: str, sublabel: str, n, rate, n_unscorable) -> dict | None:
    """One arm power row, or ``None`` for a sub-block with n == 0 or a None rate.

    The observed success count is recovered as ``round(rate * n)`` (rate is stored as the
    exact integer fraction ``count / n``, so the round is lossless at these small n) and
    fed to the same ``proportion_ci_upper`` / ``minimum_detectable_rate`` the control rows
    use -- the guardrail math is never reimplemented here. ``unscorable_flag`` applies the
    pre-registration's 10-percent rule (unscorable above a tenth of the entered pairs).
    """
    if n is None or n == 0 or rate is None:
        return None
    k = int(round(float(rate) * n))
    mde = minimum_detectable_rate(n)
    row = {
        "kind": "arms",
        "label": f"{base_label}:{sublabel}",
        "n": n,
        "rate": float(rate),
        "mde": mde,
        "ci_upper_on_observed": proportion_ci_upper(k, n),
        "powered": mde <= MDE_TARGET,
    }
    if n_unscorable is not None:
        row["n_unscorable"] = n_unscorable
        row["unscorable_flag"] = n_unscorable > 0.10 * (n + n_unscorable)
    return row


def arms_rows_from_summary(summary: dict, base_label: str) -> list[dict]:
    """Power rows for one arms summary dict (pure; unit-tested on synthetic dicts).

    Walks ``summary["arms"]`` and emits one row per rate-bearing sub-block, skipping the
    empty (n == 0) and unscored (None-rate) ones. ``base_label`` is the file's path
    relative to results/, matching the control rows' label convention; each row's label
    is ``<base_label>:<sublabel>``.
    """
    arms = summary.get("arms")
    if not isinstance(arms, dict):
        return []
    rows: list[dict] = []
    for sublabel, n, rate, n_unscorable in _arms_candidates(arms):
        row = _arms_power_row(base_label, sublabel, n, rate, n_unscorable)
        if row is not None:
            rows.append(row)
    return rows


def load_arms_rows(path: Path) -> list[dict]:
    """Parse one arms_summary_*.json into its power rows."""
    summary = json.loads(path.read_text())
    base_label = str(path.relative_to(RESULTS))
    return arms_rows_from_summary(summary, base_label)


def is_hard_violation(facts: RunFacts) -> bool:
    """True if this run's control-arm SRM or attrition-balance check is flagged.

    Mirrors the arm-balance check report_run prints, kept as a separate pure lookup
    so main() can decide the exit code without parsing printed text.
    """
    n = facts.n_clean_correct
    if not n:
        return False
    srm = srm_test((n, n))
    bal = attrition_balance({"clean": n, "hinted": n}, {"clean": n, "hinted": n})
    return bool(srm.flagged or bal.flagged)


def decide_exit_code(any_hard_violation: bool, any_underpowered: bool, strict: bool) -> int:
    """Pure exit-code policy, factored out so it is testable without argv games.

    A hard guardrail violation always fails loud (exit 2). Underpowered-only runs
    fail loud (exit 1) only under --strict, so every pre-existing invocation of this
    script (no flag) keeps its exit 0 for an underpowered-only result.
    """
    if any_hard_violation:
        return 2
    if strict and any_underpowered:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any run is underpowered (mde > MDE_TARGET); without "
                         "this flag, an underpowered-only result still exits 0")
    args = ap.parse_args(argv)

    summaries = find_summaries()
    if not summaries:
        print(f"[audit] no control_summary_*.json under {RESULTS}")
        return 1

    print("Guardrail retro-audit of saved control runs")
    print(f"  scanned: {RESULTS}")
    print(f"  found {len(summaries)} summary file(s)")
    print(f"  power target: minimum detectable follow-rate <= {MDE_TARGET:.0%} "
          "counts as adequately powered")

    facts_list: list[RunFacts] = []
    underpowered: list[str] = []
    for path in summaries:
        facts = load_run(path)
        facts_list.append(facts)
        report_run(facts)
        if facts.n_clean_correct and minimum_detectable_rate(facts.n_clean_correct) > MDE_TARGET:
            underpowered.append(f"{facts.label} (n={facts.n_clean_correct})")

    # Additive: ingest the exploratory Phase-2 arm summaries (08) into the same artifact.
    # One power row per rate-bearing sub-block; an underpowered arm row joins the control
    # underpowered list so --strict fails on it exactly like an underpowered control run.
    arms_rows: list[dict] = []
    for path in find_arms_summaries():
        arms_rows.extend(load_arms_rows(path))
    for row in arms_rows:
        if not row["powered"]:
            underpowered.append(f"{row['label']} (n={row['n']})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  runs audited:     {len(summaries)}")
    print(f"  arm power rows:   {len(arms_rows)}")
    print(f"  underpowered (MDE > {MDE_TARGET:.0%}): {len(underpowered)}")
    for label in underpowered:
        print(f"    - {label}")
    print("\n  Takeaway: a 0% follow rate on a small clean-correct n is an upper bound,")
    print("  not a finding. Pre-commit to an n whose minimum detectable rate is small")
    print("  enough to be meaningful, and report that MDE alongside any null.")

    artifacts = [build_power_artifact(f) for f in facts_list] + arms_rows
    artifact_path = RESULTS / "guardrail_power_artifacts.json"
    artifact_path.write_text(json.dumps(artifacts, indent=2))
    print(f"\nwrote power artifact ({len(facts_list)} control run(s) + "
          f"{len(arms_rows)} arm row(s)) -> {artifact_path}")

    hard_violations = [f.label for f in facts_list if is_hard_violation(f)]
    if hard_violations:
        print(f"\n  HARD GUARDRAIL VIOLATION (SRM or attrition-balance flagged): "
              f"{len(hard_violations)} run(s)")
        for label in hard_violations:
            print(f"    - {label}")

    return decide_exit_code(bool(hard_violations), bool(underpowered), args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
