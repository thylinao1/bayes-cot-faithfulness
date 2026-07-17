"""A10: the mediation layer must allow signed (suppressor) effects.

FUR (Tutek et al.) shows individual reasoning steps can carry negative,
suppressor-style effects: removing a step makes the hinted answer MORE likely.
If any prior or estimator in this package constrained the mediated path to one
sign, a suppressor world would be silently mis-estimated rather than reported.

These tests pin the property end to end without MCMC: the ground-truth Monte
Carlo integrator, the probit MLE used by the sensitivity sweep, the breakdown
frontier's sign anchoring, the no-pooling slope estimator, and the posterior
effects converter must all pass a negative mediated path through unchanged.
The hierarchical priors themselves are Normal (sign-free) by construction;
that choice is documented in docs/phase2_design_notes.md.
"""

from __future__ import annotations

import numpy as np

from bayes_cot_faithfulness.effects import (
    monte_carlo_true_effects,
    posterior_natural_effects,
)
from bayes_cot_faithfulness.hierarchical import (
    HierarchicalCoTConfig,
    no_pool_beta,
    simulate_hierarchical_cot,
)
from bayes_cot_faithfulness.sensitivity import (
    ConfoundedCoTConfig,
    breakdown_frontier,
    fit_probit_mediation_map,
    probit_natural_effects,
    simulate_confounded_cot,
)
from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig


def test_ground_truth_nie_is_negative_under_suppressor_mediator() -> None:
    cfg = SyntheticCoTConfig(alpha_direct=0.4, beta_mediated=-1.5, gamma_xm=0.8)
    nde, nie, te = monte_carlo_true_effects(cfg, n_mc=100_000, rng_seed=0)
    assert nie < -0.05
    assert nde > 0.0
    np.testing.assert_allclose(te, nde + nie, atol=1e-9)


def test_probit_fit_recovers_negative_mediated_path() -> None:
    cfg = ConfoundedCoTConfig(
        n_prompts=6_000, alpha_direct=0.3, beta_mediated=-1.0, rho_confound=0.0, rng_seed=7
    )
    X, M, Y = simulate_confounded_cot(cfg)
    alpha, beta, gamma, sigma_m = fit_probit_mediation_map(X, M, Y, rho=0.0)
    assert beta < -0.5
    nde, nie, _ = probit_natural_effects(
        alpha, beta, gamma, sigma_m, rho=0.0, n_mc=50_000, rng_seed=0
    )
    assert nie < -0.02


def test_breakdown_frontier_anchors_on_a_negative_effect() -> None:
    cfg = ConfoundedCoTConfig(
        n_prompts=900, alpha_direct=0.2, beta_mediated=-1.2, rho_confound=0.0, rng_seed=11
    )
    X, M, Y = simulate_confounded_cot(cfg)
    bf = breakdown_frontier(X, M, Y, key="nie", n_mc=30_000, rng_seed=0)
    assert bf.effect_at_zero < 0.0
    assert bf.robustness > 0.0


def test_no_pool_slopes_recover_negative_population() -> None:
    cfg = HierarchicalCoTConfig(
        n_groups=6, mu_beta=-1.3, tau_beta=0.3, min_per_group=150, max_per_group=250, rng_seed=3
    )
    group, X, M, Y = simulate_hierarchical_cot(cfg)[:4]
    betas = no_pool_beta(group, X, M, Y)
    finite = betas[np.isfinite(betas)]
    assert finite.size >= 4
    assert np.median(finite) < -0.5


def test_posterior_effects_pass_negative_mediation_through() -> None:
    n_draws = 60
    effects = posterior_natural_effects(
        alpha_samples=np.full(n_draws, 0.3),
        beta_samples=np.full(n_draws, -1.2),
        gamma_samples=np.full(n_draws, 0.8),
        sigma_m_samples=np.full(n_draws, 0.5),
        n_mc_per_draw=4_000,
        rng_seed=0,
    )
    assert effects.nie_mean < -0.02
    assert effects.nie_hi < 0.0
    assert effects.nde_mean > 0.0
