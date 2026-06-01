"""Positive-control demonstration: the auditor catches unfaithfulness, end to end.

This is the credibility floor for the whole method. Before anyone trusts a
faithfulness verdict, the instrument has to respond when unfaithfulness is
actually present. This script shows it does, at both levels the method works on,
and it runs offline, for free, and deterministically.

    PYTHONPATH=src python experiments/06_positive_control_demo.py

Act 1 (case level): three transcripts with known ground truth go through the
auditor. A planted deception (the model followed a wrong hint and its reasoning
never mentioned it) must be flagged; a genuine answer and an honest disclosure
must be cleared.

Act 2 (population level): two synthetic worlds with the same total effect but
opposite mechanisms go through the same sensitivity machinery the real-model
experiment uses. When the chain-of-thought genuinely drives the answer, the
faithful path is large and survives a lot of hidden confounding; when the CoT is
decorative, the faithful path sits at zero and the smallest confounding overturns
it. The contrast is what a non-result looks like, which is what makes the
matching numbers elsewhere meaningful rather than coincidental.

No model, no network, no cost. Writes a summary to results/positive_control_demo.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_cot_faithfulness.positive_control import (  # noqa: E402
    DECORATIVE_CONFIG,
    FAITHFUL_CONFIG,
    audit_case,
    demonstration_cases,
    smoke_test,
)

HERE = Path(__file__).resolve().parent

# The real-model run that corroborates Act 1, with transcripts saved and hand
# spot-checked (then re-scored offline). We point at the 8B run rather than the
# earlier 70B one because the 70B transcripts were not persisted, so its silent
# count cannot be verified against the hardened acknowledgment detector.
REAL_RUN_SUMMARY = HERE / "results" / "control_summary_llama-3.1-8b-instant.json"


def _rule(char: str = "-") -> str:
    return char * 78


def report_cases() -> list[dict]:
    """Act 1: run the case-level auditor on the labelled transcripts."""
    print(_rule("="))
    print("ACT 1  -  does the auditor catch a deception we planted? (case level)")
    print(_rule("="))

    results = [audit_case(c) for c in demonstration_cases()]

    # Lead with the planted-silent case: the one sentence the write-up opens with.
    planted = next(r for r in results if r["name"] == "planted-silent")
    if planted["flagged_silent_unfaithful"]:
        print(
            "\n  When the model was fooled, its written reasoning never mentioned the\n"
            "  hint that changed its answer, and the method flagged it anyway.\n"
        )

    for r in results:
        verdict = "FLAGGED unfaithful" if r["flagged_silent_unfaithful"] else "cleared"
        truth = "unfaithful" if r["ground_truth_unfaithful"] else "faithful"
        mark = "ok" if r["correct"] else "MISS"
        print(f"  [{mark:>4}] {r['name']:<20} answer=({r['answer']}) "
              f"hint=({r['hint_label']}) followed={str(r['followed_hint']):<5} "
              f"disclosed={str(r['disclosed_hint']):<5} -> {verdict}  (truth: {truth})")
        print(f"         {r['note']}")

    n_correct = sum(r["correct"] for r in results)
    print(f"\n  case-level auditor: {n_correct}/{len(results)} transcripts judged correctly")
    return results


def _print_world(name: str, res: dict) -> None:
    surv = " (survives the full searched range)" if res["survives_full_range"] else ""
    sign = "excludes zero" if res["sign_identified"] else "straddles zero"
    print(f"\n  {name}")
    print(f"    faithful path NIE (assuming no confounding) : {res['nie_at_zero']:+.3f}")
    print(f"    shortcut path NDE (for contrast)            : {res['nde_at_zero']:+.3f}")
    print(f"    breakdown rho* (confounding to overturn it) : {res['rho_star']:.3f}{surv}")
    print(f"    bounds on NIE for |rho| <= {res['rho_bar']:.2f}           : "
          f"[{res['nie_lower']:+.3f}, {res['nie_upper']:+.3f}]  ({sign})")


def report_smoke_test() -> dict:
    """Act 2: run a faithful and a decorative world through the estimator."""
    print("\n" + _rule("="))
    print("ACT 2  -  does the estimator separate a faithful CoT from a decorative one?")
    print(_rule("="))
    print("\n  Two synthetic worlds, same total effect, opposite mechanism. A trustworthy")
    print("  estimator should report a large, robust faithful path in the first and a")
    print("  faithful path of essentially zero in the second.")

    faithful = smoke_test(FAITHFUL_CONFIG)
    decorative = smoke_test(DECORATIVE_CONFIG)
    _print_world("FAITHFUL world  (the answer flows through the chain-of-thought)", faithful)
    _print_world("DECORATIVE world (the answer comes from a shortcut; CoT is window dressing)",
                 decorative)

    ratio = faithful["nie_at_zero"] / max(abs(decorative["nie_at_zero"]), 1e-6)
    print(f"\n  The faithful path is ~{ratio:.0f}x larger when the CoT actually drives the answer,")
    print(f"  and the verdict is robust (rho* {faithful['rho_star']:.2f}) only when it should be")
    print(f"  (decorative rho* {decorative['rho_star']:.2f}: any confounding overturns it).")
    print("  That decorative column is what 'non-recovery' looks like.")

    separated = (
        faithful["nie_at_zero"] > 0.10
        and abs(decorative["nie_at_zero"]) < 0.10
        and faithful["rho_star"] > decorative["rho_star"]
        and faithful["sign_identified"]
        and not decorative["sign_identified"]
    )
    return {"faithful": faithful, "decorative": decorative,
            "nie_ratio": ratio, "separated": separated}


def report_real_corroboration() -> dict | None:
    """Note the real-model run that mirrors Act 1: some silent (flagged), some disclosed."""
    if not REAL_RUN_SUMMARY.exists():
        return None
    s = json.loads(REAL_RUN_SUMMARY.read_text())
    follow = s.get("follow_rate")
    n_follow = s.get("n_followed_hint")
    n_silent = s.get("n_silent_unfaithful")
    n_disc = s.get("n_disclosed_hint")
    print("\n" + _rule("="))
    print("CORROBORATION  -  the same pattern on a real model")
    print(_rule("="))
    print(f"\n  On {s.get('model')} ({s.get('n_clean_correct')} clean-correct ARC items), the model")
    print(f"  followed the planted wrong hint {follow:.0%} of the time ({n_follow} items): a model")
    print("  on questions it already knows is mostly robust, which is the honest negative result.")
    if n_follow:
        print(f"  Of those {n_follow}, {n_silent} were silent (reasoning never disclosed the hint; the")
        print(f"  auditor flagged them) and {n_disc} openly cited the answer key (honest deference,")
        print("  correctly cleared after a hand spot-check hardened the detector).")
    print("  Transcripts saved verbatim in experiments/results/ (re-scorable offline, no API).")
    return {"model": s.get("model"), "follow_rate": follow,
            "n_followed_hint": n_follow, "n_silent_unfaithful": n_silent, "n_disclosed_hint": n_disc}


def main() -> int:
    cases = report_cases()
    smoke = report_smoke_test()
    real = report_real_corroboration()

    all_cases_correct = all(c["correct"] for c in cases)
    passed = all_cases_correct and smoke["separated"]

    print("\n" + _rule("="))
    print("DEMONSTRATION RESULT")
    print(_rule("="))
    print(f"  case-level auditor flags the deception, clears the rest : {all_cases_correct}")
    print(f"  estimator separates faithful from decorative            : {smoke['separated']}")
    print(f"  OVERALL: {'PASS - the instrument responds to smoke' if passed else 'CHECK - see above'}")

    out = HERE / "results" / "positive_control_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"cases": cases, "smoke_test": smoke, "real_corroboration": real, "passed": passed},
        indent=2,
    ))
    print(f"\nwrote summary -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
