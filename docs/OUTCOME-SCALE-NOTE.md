# The outcome scale, and what a clean arm that varies would buy

Amendment A4.6(b) and `docs/WAVE1-FITS.md` sections 2 and 7 record a structural
fact about the frozen outcome population: the analysis population is the frozen
clean-correct subpopulation, so the clean answer equals the gold label on every
row, the hint label is a planted WRONG option, and the binary follow outcome Y is
therefore 0 on every clean row by construction. The measured clean-arm outcome
variance is exactly 0.0000 in all three usable wave-1 cells (1,396 items on
`qwen3-8b`, 1,375 on `gemma-2-9b-it`, 1,321 on `llama-3.1-8b-instruct`; source
`experiments/results/wave1-fits/<cell>/fit.json`, key
`column_b.separation_diagnostic.clean_arm_outcome_variance`). The NDE and NIE
split therefore rests on the probit link extrapolating into a region the clean
arm never visits, and only the TE is checked directly, by the randomized arm
difference.

This note answers one question in five parts: whether an outcome scale that
varies in the clean arm gives a split identified by data rather than by the link,
whether that is inside what the pre-registration already provides, what the
estimator would have to be, what a synthetic demonstration says, what the records
carry, and what an additive amendment would have to add.

Nothing in this note edits a pre-registration file. Part 5 is a DRAFT amendment
text living in this document, for a later amendment to adopt or reject.

---

## Part 1. What the pre-registration already provides

### 1.1 Both outcome scales are pre-registered, and neither was chosen after the other

Element 0, the Column B contract (`experiments/PREREGISTRATION_jury_and_scale.md`
section 1, under **Outcome**), states both scales in the same paragraph:

> **Outcome.** Y is stated separately per intervention level and the two are
> never pooled into one row without a stated bridge. At the logit level, Y is the
> log-probability margin of the planted option computed on the letter
> distribution renormalized over the allowed answer set, and the raw mass on that
> set is recorded alongside. At the text level, Y is the binary follow indicator
> from the frozen parser. Each level carries its own identification statement and
> its own estimand; a row that mixes them prints both and their bridge or prints
> neither.

Element 7 (section 8, first paragraph) says the same thing about the estimand
rather than about the measurement, and adds the anti-selection clause:

> The outcome is Y as section 1 states it, separately per intervention level.
> BOTH NIE outcomes are pre-registered, the logit-level one on the renormalized
> letter margin and the text-level one on the binary follow indicator; neither is
> chosen after seeing the other, and a row that reports both prints the bridge
> between them or prints neither.

Section 9.1 fixes how the logit-level number is produced and what is stored:

> The outcome value is the log-probability margin of the planted option on the
> letter distribution RENORMALIZED over the allowed answer set. The raw values,
> the renormalized distribution and `logprob_source_token` are all stored, and
> `outcome_scale` names which one a number was computed on.

Section 9.6 and section 1.2 carry the record fields: `outcome_scale` with one of
`raw`, `renormalized_over_letters`, `logprob_margin`, `binary_follow`, plus
`logprob_source_token` alongside `answer_logprobs`, with the runner asserting
`outcome_scale` before writing a checkpoint. The repository already enforces the
level-to-scale mapping in code:
`src/bayes_cot_faithfulness/outcome_scale.py` refuses `logprob_margin` at
`intervention_level = text` and refuses `binary_follow` at
`intervention_level = logit`, and refuses any logprob scale that arrives without
a `logprob_source_token`.

### 1.2 Element 1's estimands are NOT defined on the binary follow scale only

Element 0's closing paragraph is scale-neutral by construction:

> **Estimand, assumptions, scale, interpretation.** The estimands are NDE, NIE
> and TE on the stated outcome scale, reported with intervals before any ratio.

"On the stated outcome scale", with two scales stated. Element 1 section 2.2 then
says only "Column B is defined only by section 1 of this document", so it adds no
scale restriction of its own. So the answer to the middle question is: no, the
estimands are not binary-only. What IS binary-only, and this is the part that
matters, is the verdict machinery built on top of them.

### 1.3 The three places that ARE written on the probability scale

Three pre-registered quantities name the probability scale explicitly or take
their justification from it. A logprob-scale column B inherits none of them.

1. **The load-bearing verdict, element 1 section 2.5.** "A model is
   **load-bearing at rho** when the model-level posterior probability that the
   NIE, on the probability scale that 01-SIZING sizes, exceeds 0.15 at that rho
   is at least 0.95." The 0.15 is a probability. It has no meaning in nats.
2. **The element 21 agreement margin, section 22.1.** "The margin the promotion
   rule refers to is 0.10 on the outcome scale being compared, which is two
   thirds of the pre-registered load-bearing effect of 0.15." The sentence does
   say "on the outcome scale being compared", so the margin travels with the
   scale by its own words, but its size is derived from the probability-scale
   0.15. Carried across unchanged, 0.10 nats on a margin whose measured spread is
   about 3.2 nats (part 4, the anchor cells) would be an accidentally severe
   test, roughly 0.03 of a clean-arm standard deviation.
3. **`rho*_decision`, element 7 section 8.1.** It is "the rho at which the
   pre-registered verdict rule fails", so it is defined only where a verdict rule
   is defined. `rho*_point` has no such dependence: section 8.2's analytic
   statement, `rho* = |B| x sigma_m / sqrt(1 + B^2 x sigma_m^2)`, is a property
   of the mediator equation and the outcome slope, and part 2 shows it carries to
   the continuous scale unchanged with the outcome error scale freed.

### 1.4 The answer, stated plainly

**(a) for the estimand, the estimation and the reporting; (b) for the verdict.**

A logprob-scale column B, reporting NDE, NIE and TE with intervals in the order
element 7 section 8.1 fixes, beside the binary-scale numbers and never replacing
them, with `rho*_point` printed as an invariant reference and the bridge between
the two levels stated, is **already pre-registered**. It needs no amendment. It
needs the section 9.1 unit check to have passed for that model family, the
element 8 pinned self-hosted vLLM endpoint (section 6.7: SoCLaaS is INELIGIBLE as
a source of logprob outcomes), and the records of part 4.

What is NOT pre-registered on that scale is the **load-bearing verdict**: the
0.15 threshold, the `rho*_decision` that depends on it, and the element 21
agreement margin's size. Those need an additive amendment, which is what part 5
drafts. Until such an amendment exists, a logprob-scale column B reports its
three effects with intervals and its `rho*_point`, prints "no verdict rule on
this scale" where a verdict would go, and does not promote anything.

This is not a formality. Section 8.1's own reporting order puts NDE, NIE and TE
with intervals BEFORE any ratio or any rho quantity, so the part that is already
pre-registered is exactly the part the reporting order puts first.

---

## Part 2. The estimator

### 2.1 What the code did before this note

Every mediation path in `src/bayes_cot_faithfulness/` assumed a binary outcome.
There was no Gaussian outcome equation anywhere:

* `sensitivity.py` is the estimator of record. `fit_probit_mediation_map` is a
  probit fit at fixed rho, `_joint_negloglik` scores `Y` through
  `norm.logcdf` / `norm.logsf`, and `_validate_xmy` (line 724) refuses any `Y`
  outside `{0, 1}`. Its module docstring derives the rho reparameterisation for
  the latent-Gaussian probit only.
* `closed_form.py` is scoped in its own docstring: "the closed form here is
  deliberately scoped to the probit SCM only". `probit_natural_effects_closed_form`
  returns probabilities through `norm.cdf`.
* `effects.py` integrates the mediator against a `sigmoid` or delegates to the
  probit closed form; both branches produce a probability.
* `mediation.py` fits `Y_obs ~ Bernoulli` (logit) or a probit potential, and
  raises `ValueError("Y must be binary {0, 1}.")` on anything else.

So the answer to "is there a Gaussian outcome equation path" was no. There was
only the probit outcome, plus a logistic variant kept for the historical
generator in `synthetic.py`.

### 2.2 What this lane added

`src/bayes_cot_faithfulness/gaussian_mediation.py`, a separate module. The probit
path is not touched: `git diff` against main shows no change to `sensitivity.py`,
`closed_form.py`, `effects.py` or `mediation.py`.

The model, with intercepts in BOTH equations, which is the 2026-09-07 repair
carried across:

    M = mu_m + gamma * X + eps_M
    Y = alpha0 + alpha * X + beta * M + eps_Y
    (eps_M, eps_Y) ~ Normal2(0, [[sigma_m^2,           rho*sigma_m*sigma_y],
                                 [rho*sigma_m*sigma_y, sigma_y^2         ]])

The cross-world potential outcome is
`Y(x', M(x)) = alpha0 + alpha*x' + beta*(mu_m + gamma*x + eps_M) + eps_Y`, whose
expectation is linear, so the natural effects are the coefficients themselves:

    NDE = alpha
    NIE = beta * gamma
    TE  = alpha + beta * gamma

That is the substantive difference from the probit path, and it is the whole
answer to "identified by data rather than by the link". On the probability scale
the effects are differences of `Phi` evaluated at three different index values,
so where the index takes each value matters and a region with no data has to be
reached by the link's shape. Here `alpha` is the within-mediator slope of Y on X
and `beta` is the within-arm slope of Y on M, both estimated by least squares
from data that exists in both arms. Nothing is evaluated where nothing was
observed.

The functions, and what each is for:

| function | what it does |
|---|---|
| `fit_gaussian_mediation_closed_form(X, M, Y, rho, intercepts)` | exact MLE at fixed rho: OLS of M on X, OLS of Y on (1, X, M), then the rho mapping. The closed form the task asked for. |
| `fit_gaussian_mediation_map(X, M, Y, rho, intercepts, strict)` | the numeric L-BFGS-B counterpart of `sensitivity.fit_probit_mediation_map`, with the same `OptimizerFailure` / `OptimizerWarning` semantics and the same `converged` field |
| `gaussian_natural_effects(...)` | the effects, with the same argument order as `probit_natural_effects_closed_form` so it is a drop-in |
| `gaussian_effects_curve(fit0, rho_grid)` | the vectorised sweep from one rho = 0 fit, the analogue of the battery's `effects_curve` |
| `gaussian_sensitivity_sweep(...)` | the slow refit-at-every-grid-point sweep, kept as the cross-check |
| `gaussian_rho_star_point(beta, sigma_m, sigma_y)` | the NIE's zero crossing |
| `standardised_effects(nde, nie, te, clean_arm_sd)` | effects in clean-arm outcome standard deviations, a reporting convenience, explicitly not an estimand |

### 2.3 The rho parameterisation is the same one, with the outcome error scale freed

Conditioning on the observed M so that `eps_M = M - mu_m - gamma*X` is known,

    Y | X, M ~ Normal( alpha0 + alpha*X + beta*M + k*(M - mu_m - gamma*X),
                       sigma_y^2 * (1 - rho^2) ),      k = rho*sigma_y/sigma_m

Writing the observed regression of Y on (1, X, M) as `A0 + A*X + B*M` with
residual standard deviation `S`, term by term:

    beta(rho)   = B  - k
    alpha(rho)  = A  + k*gamma
    alpha0(rho) = A0 + k*mu_m
    sigma_y(rho) = S / sqrt(1 - rho^2)
    k(rho) = rho * S / (sigma_m * sqrt(1 - rho^2))

This is `sensitivity`'s mapping with `S` where the probit has its fixed 1. Two
consequences, both pinned by tests rather than asserted:

* **The TE is rho-invariant on this scale, and equals the randomized arm
  difference exactly.** `TE(rho) = alpha(rho) + beta(rho)*gamma = A + B*gamma`,
  which has no rho in it, and the least-squares identity makes
  `A + B*gamma = mean(Y | X=1) - mean(Y | X=0)`. rho moves the split and only the
  split. `tests/test_gaussian_mediation.py::test_te_equals_the_randomized_arm_difference_at_every_rho`
  checks both halves on a seven-point grid from -0.9 to +0.9, and the measured TE
  range across that grid is below 1e-9.
* **`rho*_point` keeps its invariance.** The NIE vanishes where `beta(rho) = 0`,
  that is where `rho / sqrt(1 - rho^2) = B*sigma_m / S`, giving
  `rho*_point = |B| * sigma_m / sqrt(S^2 + B^2 * sigma_m^2)`. Set `S = 1` and this
  is section 8.2's `|B| x sigma_m / sqrt(1 + B^2 x sigma_m^2)` unchanged. It
  contains neither `alpha` nor either intercept, so section 8.2's "invariant to
  the direct coefficient by construction, and therefore carries no information
  about direct-path strength" is true on this scale too.
  `test_rho_star_point_is_invariant_to_the_direct_path` runs the direct
  coefficient from 0 to 3 on 20,000 rows per setting: the crossing moves by less
  than 0.01 while the mediated share `NIE/TE` falls from above 0.95 to below
  0.25.

### 2.4 The tests, and what each one gates

`tests/test_gaussian_mediation.py`, 19 tests, all passing (0 failures, 0 skips,
0.61 s). The three the task named:

**(i) Recovery of a known NDE and NIE.** `test_recovers_known_effects_on_linear_data`
generates 8,000 rows from the SCM above with `alpha 0.40`, `beta 0.70`,
`gamma 0.90`, `mu_m 6.0`, `alpha0 -2.0`, both error standard deviations 1, and
asserts the recovered NDE, NIE and TE are within 0.05 of the truth
(0.40, 0.63, 1.03), the recovered `mu_m` is within 0.05 of 6.0, and the recovered
`sigma_y` within 5 percent of 1. It also asserts the clean arm's outcome variance
exceeds 0.5, so the world under test is the one this note is about.
`test_recovers_the_truth_at_the_true_rho_when_confounded` runs the same world at
a true rho of 0.5 and checks that the rho = 0 fit is visibly wrong (`beta` off by
more than 0.2) while the fit at rho = 0.5 recovers `beta` and `alpha` to within
0.05.

**(ii) The offset nulls.** `test_offset_null_reports_no_mediation` and
`test_offset_null_recovers_the_mediator_baseline` run the reviewer's two null
designs at seed 731, n = 10,000, balanced arms, byte-identical to
`tests/test_offset_null.py::_null_design`: the count offset (`poisson(5) + 1`,
mean about 6) and the Gaussian offset (`6 + normal(0, sqrt(5))`), with Y the
independent 0.8 coin. On both, the fit must return `|NIE| < 0.01` and
`|NDE| < 0.05`, must put `mu_m` within 0.05 of the control-arm mean of M rather
than at zero, and must return a model-implied TE equal to the randomized arm
difference to within 1e-6.
`test_intercept_free_fit_invents_mediation_on_the_same_null` keeps the defect
legible on this scale: with `intercepts=False` the same data gives `|NIE| > 0.1`.

**(iii) The closed form against the numerical fit.**
`test_closed_form_matches_the_numerical_fit` compares all seven parameters at
five values of rho (-0.6, -0.2, 0.0, +0.3, +0.7) on confounded data and requires
agreement to 1e-4.
`test_effects_curve_matches_a_refit_at_every_grid_point` compares the vectorised
curve against a refit at each of 19 grid points from -0.9 to +0.9 and requires
agreement below 1e-9 on all three effects.
`test_rho_star_point_is_where_the_curve_crosses_zero` evaluates the curve AT the
reported crossing and requires `|NIE| < 1e-9` there.

Two guard tests round it out: the admissible band `|rho| <= 0.95` and the shape
and support checks refuse rather than warn, matching the probit path.

### 2.5 What the continuous path does not fix

Stated here so part 3 cannot be read as more than it is. The linear outcome
equation removes the link extrapolation. It does nothing about assumption A3.
`rho` prices mediator-outcome confounding on this scale exactly as it does on the
probability scale, the rho = 0 fit is biased by exactly as much, and a
shared-cause world reads as mediation on both scales. Part 3 measures that.

