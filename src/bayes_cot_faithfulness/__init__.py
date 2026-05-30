"""Bayesian causal mediation analysis for LLM chain-of-thought faithfulness."""

from bayes_cot_faithfulness.effects import (
    PosteriorEffects,
    monte_carlo_true_effects,
    posterior_natural_effects,
)
from bayes_cot_faithfulness.hierarchical import (
    HierarchicalCoTConfig,
    fit_hierarchical_mediation,
    no_pool_beta,
    simulate_hierarchical_cot,
)
from bayes_cot_faithfulness.mediation import (
    extract_parameter_samples,
    fit_mediation_model,
)
from bayes_cot_faithfulness.sensitivity import (
    ConfoundedCoTConfig,
    SensitivityPoint,
    fit_probit_mediation_map,
    probit_natural_effects,
    robustness_interval,
    sensitivity_sweep,
    simulate_confounded_cot,
)
from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig, simulate_cot_trace

__version__ = "0.2.0"

__all__ = [
    "ConfoundedCoTConfig",
    "HierarchicalCoTConfig",
    "PosteriorEffects",
    "SensitivityPoint",
    "SyntheticCoTConfig",
    "extract_parameter_samples",
    "fit_hierarchical_mediation",
    "fit_mediation_model",
    "fit_probit_mediation_map",
    "monte_carlo_true_effects",
    "no_pool_beta",
    "posterior_natural_effects",
    "probit_natural_effects",
    "robustness_interval",
    "sensitivity_sweep",
    "simulate_confounded_cot",
    "simulate_cot_trace",
    "simulate_hierarchical_cot",
]
