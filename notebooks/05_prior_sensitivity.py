"""Prior-sensitivity sweep for the hierarchical faithfulness slope.

A Bayesian headline is only credible if it survives a change of prior. This
script refits the partially-pooled mediation model under a set of equally
defensible priors, varying ONE hyperprior at a time, and tabulates how far the
population faithfulness slope ``mu_beta`` and its 94 percent HDI move from a fixed
baseline. A small maximum shift is the robustness statement: the data, not the
prior, is driving the number.

Run end-to-end on a laptop CPU in well under a minute, no API calls, no GPU::

    python notebooks/05_prior_sensitivity.py

The story this script tells:

1. We simulate prompt-level CoT traces with real heterogeneity in the faithful
   path, so partial pooling has something to do.
2. We refit the model under three pooling-scale prior families (HalfNormal,
   Exponential, HalfStudentT), matched to a comparable prior mean so the contrast
   is the prior's tail weight rather than its scale. The headline barely moves:
   this is the robustness exhibit.
3. We then probe two coefficient-prior widths on ``mu_beta`` (tight vs wide). A
   tight prior on a finite sample is meant to tug the estimate, so those rows are
   a sensitivity probe, read alongside the robustness rows rather than as part of
   the same claim.
"""

from __future__ import annotations

from bayes_cot_faithfulness.hierarchical import (
    HierarchicalCoTConfig,
    simulate_hierarchical_cot,
)
from bayes_cot_faithfulness.prior_sensitivity import (
    PriorSensitivityResult,
    default_prior_specs,
    prior_sensitivity_sweep,
)

# Loose robustness bar: across reasonable priors the headline should move by less
# than this on the probability-of-correct logit scale. The pooling-family rows
# clear it comfortably on synthetic data.
ROBUSTNESS_THRESHOLD = 0.15


def _print_table(result: PriorSensitivityResult) -> None:
    """Print the sweep as a plain-ASCII markdown table."""
    header = f"| {'prior':28s} | {'mu_beta':>8s} | {'94% HDI':>20s} | {'shift vs base':>13s} |"
    rule = f"|{'-' * 30}|{'-' * 10}|{'-' * 22}|{'-' * 15}|"
    print(header)
    print(rule)
    for r in result.rows:
        hdi = f"[{r.hdi_low:+.3f}, {r.hdi_high:+.3f}]"
        shift = "baseline" if r.is_baseline else f"{r.shift_vs_baseline:+.4f}"
        print(f"| {r.label:28s} | {r.estimate:+8.4f} | {hdi:>20s} | {shift:>13s} |")


def main() -> int:
    cfg = HierarchicalCoTConfig(
        n_groups=10,
        mu_beta=1.2,
        tau_beta=0.4,
        min_per_group=60,
        max_per_group=120,
        rng_seed=21,
    )

    print(
        f"[1/3] Simulating hierarchical CoT traces "
        f"(groups={cfg.n_groups}, true mu_beta={cfg.mu_beta:+.2f}, seed={cfg.rng_seed})"
    )
    group, X, M, Y, truth = simulate_hierarchical_cot(cfg)
    print(f"        total traces: {len(group)}    Y rate: {Y.mean():.3f}")
    print(f"        true per-group beta spread: {truth['beta_g'].std():.3f}")

    print("\n[2/3] Refitting under each prior (one hyperprior changed at a time)")
    print("        baseline first, then pooling-family swaps, then coefficient-sd probes")
    result = prior_sensitivity_sweep(
        group,
        X,
        M,
        Y,
        specs=default_prior_specs(),
        parameter="mu_beta",
        n_samples=700,
        n_tune=700,
        n_chains=2,
        random_seed=0,
    )

    print("\n[3/3] Prior-sensitivity table (headline parameter: mu_beta)\n")
    _print_table(result)

    family_shift = max(
        (abs(r.shift_vs_baseline) for r in result.rows[:3] if not r.is_baseline),
        default=0.0,
    )
    print(f"\n        Max headline shift across pooling-scale prior families: {family_shift:.4f}")
    print(f"        Max headline shift across the full sweep:               {result.max_abs_shift:.4f}")

    baseline = result.rows[0].estimate
    print(
        f"        Baseline mu_beta = {baseline:+.4f} vs true {cfg.mu_beta:+.2f} "
        f"(recovery error {abs(baseline - cfg.mu_beta):.4f})."
    )

    if family_shift < ROBUSTNESS_THRESHOLD:
        print(
            "\n        The faithfulness headline is robust to the pooling-scale prior "
            "family:\n        swapping HalfNormal for Exponential or a heavy-tailed "
            "HalfStudentT barely moves it."
        )
        print(
            "        The coefficient-prior rows show the expected mild tug of a tighter "
            "prior\n        on a finite sample, which is informativeness, not fragility."
        )
        return 0

    print(
        f"\n        Headline moved by {family_shift:.4f} across prior families "
        f"(> {ROBUSTNESS_THRESHOLD})."
    )
    print("        Investigate before reporting a single faithfulness number.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
