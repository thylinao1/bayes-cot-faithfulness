"""Bayesian causal mediation analysis for LLM chain-of-thought faithfulness."""

from bayes_cot_faithfulness.effects import (
    PosteriorEffects,
    monte_carlo_true_effects,
    posterior_natural_effects,
)
from bayes_cot_faithfulness.mediation import fit_mediation_model
from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig, simulate_cot_trace

__version__ = "0.1.0"

__all__ = [
    "PosteriorEffects",
    "SyntheticCoTConfig",
    "fit_mediation_model",
    "monte_carlo_true_effects",
    "posterior_natural_effects",
    "simulate_cot_trace",
]
