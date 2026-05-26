# Methodology

> **Bayesian causal mediation analysis applied to LLM chain-of-thought reasoning.**

This document formalises the framework the library implements. It exists for three reasons:

1. To make the identification assumptions explicit (and to flag when they break).
2. To connect this work to the activation-patching / causal-scrubbing / causal-abstraction lines in mechanistic interpretability.
3. To document the synthetic-validation gate that justifies trusting the estimator on real LLM traces.

---

## 1 · The CoT-as-mediator causal graph

```
        Prompt X  ─────────────[α: direct]─────────────►  Answer Y
            │                                                ▲
            │                                                │
            └────[γ: X → M]────► CoT  M  ────[β: M → Y]──────┘
```

We treat the chain-of-thought tokens `M` as a mediator on the path from prompt `X` to answer `Y`. The prompt's effect on the answer decomposes into two paths:

- **Direct path** (`X → Y`): the part of the prompt's effect that bypasses CoT. In a fully-faithful world this is zero.
- **Indirect path** (`X → M → Y`): the part that flows through the reasoning. In a fully-faithful world this carries the entire effect.

The decomposition we estimate is the **Robins–Greenland–Pearl** natural-effects decomposition:

$$
\text{NDE} = \mathbb{E}[Y \mid \text{do}(X{=}1), M{\sim}M(X{=}0)] - \mathbb{E}[Y \mid \text{do}(X{=}0), M{\sim}M(X{=}0)]
$$

$$
\text{NIE} = \mathbb{E}[Y \mid \text{do}(X{=}1), M{\sim}M(X{=}1)] - \mathbb{E}[Y \mid \text{do}(X{=}1), M{\sim}M(X{=}0)]
$$

$$
\text{TE} = \text{NDE} + \text{NIE}
$$

The reading is: NIE is the proportion of the prompt's effect on the answer that is *carried by the CoT*. A high NIE / TE ratio = faithful reasoning. A high NDE / TE ratio = decorative reasoning.

## 2 · Identification assumptions

Standard mediation analysis identifies natural effects under the **sequential ignorability** assumption:

- (A1) No unmeasured confounders between `X` and `Y`
- (A2) No unmeasured confounders between `X` and `M`
- (A3) No unmeasured confounders between `M` and `Y`, conditional on `X`
- (A4) No `M → Y` confounder is itself caused by `X`

In the LLM setting:

- (A1)–(A2) hold by design: we control the prompt, so `X` is *exogenous*.
- (A3) is the interesting one. The natural CoT depends on hidden activations that also affect `Y` directly. We handle this with interventions (paraphrase, swap, interchange) that break dependence between `M` and the latent confounders. This is the move from observational to interventional mediation analysis.
- (A4) is checked empirically by ablation: if we randomize the seed used to generate CoT and the answer, do we see leakage?

## 3 · Hierarchical Bayesian estimator

We fit a joint model:

$$
M \mid X \sim \mathcal{N}(\gamma X,\; \sigma_M^2)
$$

$$
Y \mid X, M \sim \mathrm{Bernoulli}(\sigma(\alpha X + \beta M))
$$

Priors are weakly informative:
- $\alpha, \beta, \gamma \sim \mathcal{N}(0, \tau^2)$ with $\tau \in \{1.5, 2.0\}$
- $\sigma_M \sim \mathrm{HalfNormal}(1)$

Sampling: 4 chains, 1500 tuning + 1500 posterior draws, NUTS with `target_accept = 0.95`.

Posterior samples over $(\alpha, \beta, \gamma, \sigma_M)$ are converted into a posterior over $(\text{NDE}, \text{NIE}, \text{TE})$ on the probability scale by Monte Carlo integration over the mediator distribution (see [`effects.posterior_natural_effects`](../src/bayes_cot_faithfulness/effects.py)).

**Why Bayesian.**
- Calibrated uncertainty intervals out of the box, with no asymptotic normality assumption.
- Hierarchical pooling across prompts and seeds for the real-LLM experiments (extending the simple model above).
- Posterior model comparison (Bayes factors / WAIC / PSIS-LOO) for "is this CoT faithful or not?" hypothesis tests.

## 4 · Synthetic-validation gate

Before scaling to real LLM experiments, we run [`notebooks/01_synthetic_validation.py`](../notebooks/01_synthetic_validation.py):

1. Generate $n = 400$ traces from the structural model with known $(\alpha, \beta, \gamma, \sigma_M)$.
2. Compute ground-truth NDE / NIE / TE by Monte Carlo (200k samples).
3. Fit the Bayesian mediation model.
4. Check that the 95 % credible intervals contain the truth.

This is the methodological go/no-go gate. If the estimator can't recover known effects on data where we *control* the structural form, it has no business being applied to real LLM traces.

## 5 · Real-LLM extension (Rapid Grant scope)

The synthetic gate uses a single binary `X` and a scalar `M`. The full benchmark extends in three directions:

| Direction | Mechanism |
|---|---|
| **Multi-token CoT** | Token-segment-level mediator (truncate CoT at varying positions, treat the truncation depth as the manipulated mediator state) |
| **Counterfactual mediation** | Paraphrase or swap CoT segments using a separate model. Interchange interventions, analogous to Geiger et al. |
| **Hierarchical pooling** | Random effects for prompt difficulty and seed variance; partial pooling across prompts |

The CoT intervention toolkit is borrowed from **Lanham et al. (2023)** ("Measuring Faithfulness in Chain-of-Thought Reasoning") and **Turpin et al. (2023)** ("Language Models Don't Always Say What They Think"). The Bayesian layer on top is the novel methodological contribution.

## 6 · Relationship to mechanistic interpretability

The natural extension is from **token-level CoT mediation** (this project) to **circuit-level activation mediation** (a natural follow-on):

| Behavioural level | Mechanistic level |
|---|---|
| Mediator = CoT tokens | Mediator = attention head activations / SAE feature firings |
| Intervention = truncate / paraphrase | Intervention = activation patching / interchange |
| Effect = answer logit change | Effect = answer logit change |

Causal scrubbing (Chan et al. 2023) and causal abstraction / DAS (Geiger et al. 2023) operate at the mechanistic level and use frequentist or point-estimate methods. The Bayesian framework here ports over directly: each interchange becomes an observation in a hierarchical model whose posterior tells you, with calibrated uncertainty, whether a claimed circuit *causally* implements a claimed computation.

## References

- Lanham, T. et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic.
- Turpin, M. et al. (2023). *Language Models Don't Always Say What They Think.* NeurIPS.
- Geiger, A. et al. (2023). *Causal Abstraction for Faithful Model Interpretation.* arXiv:2301.04709.
- Chan, L. et al. (2023). *Causal Scrubbing.* Alignment Forum.
- Robins, J., Greenland, S. (1992). *Identifiability and Exchangeability for Direct and Indirect Effects.* Epidemiology.
- Pearl, J. (2001). *Direct and Indirect Effects.* UAI.
- VanderWeele, T. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction.* OUP.
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press.
- Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
