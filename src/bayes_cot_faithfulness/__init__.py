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
from bayes_cot_faithfulness.positive_control import (
    DECORATIVE_CONFIG,
    DEFAULT_RHO_BAR,
    FAITHFUL_CONFIG,
    LabeledCase,
    audit_case,
    demonstration_cases,
    nie_sensitivity_curve,
    smoke_test,
)
from bayes_cot_faithfulness.sensitivity import (
    BreakdownFrontier,
    ConfoundedCoTConfig,
    PartialIDBounds,
    SensitivityPoint,
    breakdown_frontier,
    fit_probit_mediation_map,
    partial_identification_bounds,
    probit_natural_effects,
    robustness_interval,
    sensitivity_sweep,
    simulate_confounded_cot,
)
from bayes_cot_faithfulness.synthetic import SyntheticCoTConfig, simulate_cot_trace

__version__ = "0.2.0"

__all__ = [
    "BreakdownFrontier",
    "ConfoundedCoTConfig",
    "DECORATIVE_CONFIG",
    "DEFAULT_RHO_BAR",
    "FAITHFUL_CONFIG",
    "HierarchicalCoTConfig",
    "LabeledCase",
    "PartialIDBounds",
    "PosteriorEffects",
    "SensitivityPoint",
    "SyntheticCoTConfig",
    "audit_case",
    "breakdown_frontier",
    "demonstration_cases",
    "partial_identification_bounds",
    "extract_parameter_samples",
    "fit_hierarchical_mediation",
    "fit_mediation_model",
    "fit_probit_mediation_map",
    "monte_carlo_true_effects",
    "nie_sensitivity_curve",
    "no_pool_beta",
    "posterior_natural_effects",
    "probit_natural_effects",
    "robustness_interval",
    "sensitivity_sweep",
    "simulate_confounded_cot",
    "simulate_cot_trace",
    "simulate_hierarchical_cot",
    "smoke_test",
]
