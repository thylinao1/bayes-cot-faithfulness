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

The decomposition we estimate is the **Robins-Greenland-Pearl** natural-effects decomposition:

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

- (A1) and (A2) hold by design: we control the prompt, so `X` is *exogenous*.
- (A3) is the interesting one. The natural CoT depends on hidden activations that also affect `Y` directly. We handle this with interventions (paraphrase, swap, interchange) that break dependence between `M` and the latent confounders. This is the move from observational to interventional mediation analysis.
- (A4) is checked empirically by ablation: if we randomize the seed used to generate CoT and the answer, do we see leakage?

## 3 · Sensitivity analysis for sequential ignorability (A3)

Assumption (A3) is the one you cannot check from data. The natural CoT and the answer both read from the same hidden activations, so some residual dependence between the mediator error and the outcome error is almost guaranteed. Every prior causal-mediation estimate of CoT faithfulness assumes that dependence away. We make it a dial instead.

Introduce a single sensitivity parameter:

$$
\rho = \mathrm{Corr}(\varepsilon_M, \varepsilon_Y)
$$

the residual correlation between the mediator error and the outcome error after conditioning on `X`. Setting $\rho = 0$ is exactly sequential ignorability. To let $\rho$ enter cleanly we use a probit (latent-Gaussian) outcome, the standard specification in the Imai, Keele and Tingley (2010) framework:

$$
X \sim \mathrm{Bernoulli}(0.5), \qquad
(\varepsilon_M, \varepsilon_Y) \sim \mathcal{N}_2\!\left(0,\; \begin{bmatrix}\sigma_M^2 & \rho\,\sigma_M \\ \rho\,\sigma_M & 1\end{bmatrix}\right)
$$

$$
M = \gamma X + \varepsilon_M, \qquad Y = \mathbf{1}[\alpha X + \beta M + \varepsilon_Y > 0]
$$

Conditioning on the observed `M` (so $\varepsilon_M = M - \gamma X$ is known) gives an exact observed-data likelihood for any fixed $\rho$:

$$
P(Y{=}1 \mid X, M) = \Phi\!\left(\frac{\alpha X + \beta M + (\rho/\sigma_M)(M - \gamma X)}{\sqrt{1 - \rho^2}}\right)
$$

At $\rho = 0$ this is ordinary probit mediation. At the true $\rho$ it recovers the de-confounded coefficients, and so the true natural effects. The sensitivity sweep fits the model at each value on a grid of assumed $\rho$ and reads off the natural effects. The output is a curve, not a point: it shows how the faithfulness verdict moves as the no-hidden-confounder assumption is relaxed, and over what range of $\rho$ the verdict holds.

**Validated result (synthetic, laptop CPU).** With a true residual correlation of $\rho = 0.5$ built into the data, an analyst who assumes ignorability reads off a natural indirect effect of $0.43$ when the truth is $0.21$. Ignoring the confounding overstates the faithful path by about $0.22$, and that gap does not shrink as the sample grows, because it is confounding and not sampling noise. The sweep recovers the truth at the true $\rho$ (recovery error $0.02$), and the indirect (faithful) path stays positive for every assumed $\rho$ in $[-0.6, +0.7]$. On this process the faithfulness conclusion survives a large amount of unmeasured confounding, and the analysis says exactly how much.

The direction matters for how the result reads: under positive M-Y confounding (the natural case when a shared hidden factor lifts both the CoT content and the answer), assuming ignorability makes the reasoning look more faithful than it is. A naive faithfulness audit can over-trust the CoT. See [`sensitivity.py`](../src/bayes_cot_faithfulness/sensitivity.py); the sweep figure is in the project README.

## 4 · Hierarchical Bayesian estimator

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

**Partial pooling across prompts (implemented).** A single faithfulness number for a model is the wrong unit. Faithfulness varies by prompt and by task, and some prompts have far fewer usable traces than others. Estimating each prompt on its own overfits the small ones; collapsing everything into one number hides the variation. The partial-pooling model gives each prompt $g$ its own coefficients drawn from a population:

$$
\alpha_g \sim \mathcal{N}(\mu_\alpha, \tau_\alpha), \qquad
\beta_g \sim \mathcal{N}(\mu_\beta, \tau_\beta), \qquad
\gamma_g \sim \mathcal{N}(\mu_\gamma, \tau_\gamma)
$$

The population mean $\mu_\beta$ is the model-level faithfulness slope with the right uncertainty; $\tau_\beta$ says how much prompts disagree. Sparse prompts borrow strength from the rest instead of swinging wildly in isolation. A non-centred parameterisation keeps the geometry friendly for NUTS. On synthetic data the model recovers $\mu_\beta$ inside its 95% credible interval and beats per-prompt no-pooling on per-prompt error. See [`hierarchical.py`](../src/bayes_cot_faithfulness/hierarchical.py).

## 5 · Synthetic-validation gate

Before scaling to real LLM experiments, we run [`notebooks/01_synthetic_validation.py`](../notebooks/01_synthetic_validation.py):

1. Generate $n = 400$ traces from the structural model with known $(\alpha, \beta, \gamma, \sigma_M)$.
2. Compute ground-truth NDE / NIE / TE by Monte Carlo (200k samples).
3. Fit the Bayesian mediation model.
4. Check that the 95 % credible intervals contain the truth.

This is the methodological go/no-go gate. If the estimator can't recover known effects on data where we *control* the structural form, it has no business being applied to real LLM traces.

## 6 · Real-LLM extension (Rapid Grant scope)

The synthetic gate uses a single binary `X` and a scalar `M`. The full benchmark extends in three directions:

| Direction | Mechanism |
|---|---|
| **Multi-token CoT** | Token-segment-level mediator (truncate CoT at varying positions, treat the truncation depth as the manipulated mediator state) |
| **Counterfactual mediation** | Paraphrase or swap CoT segments using a separate model. Interchange interventions, analogous to Geiger et al. |
| **Hierarchical pooling** | Random effects for prompt difficulty and seed variance. The partial-pooling model is prototyped in `hierarchical.py`; the extension is to real prompts and seeds. |

The CoT intervention toolkit is borrowed from **Lanham et al. (2023)** ("Measuring Faithfulness in Chain-of-Thought Reasoning") and **Turpin et al. (2023)** ("Language Models Don't Always Say What They Think"). The Bayesian layer on top is the novel methodological contribution.

## 7 · Relationship to mechanistic interpretability

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
- Imai, K., Keele, L., Tingley, D. (2010). *A General Approach to Causal Mediation Analysis.* Psychological Methods.
- Pearl, J. (2001). *Direct and Indirect Effects.* UAI.
- VanderWeele, T. (2015). *Explanation in Causal Inference: Methods for Mediation and Interaction.* OUP.
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press.
- Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
