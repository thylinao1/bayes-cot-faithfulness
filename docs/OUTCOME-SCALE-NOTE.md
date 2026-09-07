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

---

## Part 3. The synthetic demonstration, and what it actually says

`experiments/outcome_scale_demo.py`, artifacts in
`experiments/results/outcome_scale_demo/` (`results.json`, `per_dataset.json`,
`report.md`). 100 seeded datasets per condition, 1,800 rows each, 120 bootstrap
replicates per dataset, 95 percent intervals, 483 s wall on six worker processes
at commit `993db658eab4`.

### 3.1 The design, and the one property that makes the comparison clean

The two families are the battery's own `f3_shared_cause` and
`f4_rationalization`, imported from `experiments/mechanism_battery.py` so their
constants cannot drift. Each family's latent index is exposed and the battery's
own `y_of` is reconstructed as `1[index + noise > 0]`; the runner asserts that
reconstruction row by row on 20,000 rows per family before any fit runs, and it
passed on 20,000 of 20,000 for both.

Three readouts of that one index, so a row of the table changes only how the index
becomes an outcome:

| readout | Y | clean arm | estimator |
|---|---|---|---|
| `binary_clean_varies` | `1[index + noise > 0]` | varies (the battery's own readout) | probit |
| `binary_zero_clean` | `X * 1[index + noise > 0]` | identically 0 (the frozen population's shape) | probit |
| `continuous_margin` | `index + noise` | varies, at the hinted arm's variance | Gaussian |

The gate `X *` multiplies both `p10` and `p11` by 1, because both have `x = 1`, so
**the true NIE of `binary_zero_clean` is exactly the true NIE of
`binary_clean_varies`**: +0.0000 for f3 and +0.2833 for f4 in both columns. Only
the true NDE and the clean arm's data content change. That is what makes the
comparison a test of the clean-arm degeneracy and not of six unrelated worlds.
Monte Carlo truth against hand-derived analytic truth agrees to a maximum absolute
difference of 0.00065 across all six cells at 2,000,000 rows each.

Measured clean-arm outcome variance, averaged over the 100 datasets: 0.2497 and
0.2468 on `binary_clean_varies`, exactly 0.0000 on both `binary_zero_clean`
conditions, and 2.0103 on `continuous_margin` against a hinted-arm 2.0013, a
clean-to-hinted ratio of 1.004. So the continuous readout does what the question
asked for: clean-arm variance comparable to the hinted arm's.

### 3.2 Result one: the binary-zero shape does NOT bias the split

On `f4_rationalization`, the family whose true residual correlation is 0 and which
the probit SCM can represent, the frozen shape is not measurably worse than the
control:

| readout | NDE bias (MC se) | NDE coverage | NIE bias (MC se) | NIE coverage |
|---|---|---:|---|---:|
| `binary_clean_varies` | +0.0026 (0.0023) | 95/100 [0.887, 0.984] | +0.0004 (0.0015) | 96/100 [0.901, 0.989] |
| `binary_zero_clean` | +0.0004 (0.0025) | 95/100 [0.887, 0.984] | -0.0007 (0.0019) | 95/100 [0.887, 0.984] |
| `continuous_margin` | -0.0082 (0.0064) | 90/100 [0.824, 0.951] | +0.0082 (0.0057) | 92/100 [0.848, 0.965] |

All three recover the split. The frozen shape's NDE bias is +0.0004 against a
truth of +0.5556 and its NIE bias is -0.0007 against a truth of +0.2833, with
coverage 95/100 and 95/100, whose Clopper-Pearson intervals contain 0.95 and
overlap the control's completely. 100 of 100 fits converged in every condition.

This is a correction to the intuition A4.6(b)(2) invites, and the mechanism is
A4.6(b)(1) read one step further. That clause says `alpha0` and `alpha` are
individually close to unidentified and only their sum reaches the effects. The
demonstration says the second half is the operative one: the probability-scale
natural effects depend on the intercepts only through the offset
`alpha0 + beta*mu_m` and on `alpha` only through `offset + alpha` (this is exactly
the closed form in `closed_form.py`), and both of those combinations ARE
identified even when their pieces are not. Convergence is still not evidence that
the split is identified, which is what A4.6(b)(1) says and remains true. But on
this simulation the split comes out right anyway.

What the degenerate clean arm actually costs is precision, and it is a trade
rather than a loss. Mean interval widths on `f4_rationalization`:

| effect | `binary_clean_varies` | `binary_zero_clean` | change |
|---|---:|---:|---|
| NDE | 0.0888 | 0.0928 | 4.5 percent wider |
| NIE | 0.0591 | 0.0722 | 22.2 percent wider |
| TE | 0.0763 | 0.0454 | 40.5 percent narrower |

The TE gets sharper because a clean arm pinned at 0 contributes no variance to the
arm difference, and the NIE gets blunter because the split has less to work with.

### 3.3 Result two: no outcome scale fixes the shared cause

`f3_shared_cause` is `M = X + U`, `Y` driven by `X + U` with no M-to-Y arrow, so
the residual correlation between the mediator error and the outcome error is
`1/sqrt(2) = 0.70711` by construction and the true NIE is exactly 0. Every readout
fails, and they fail by the same amount in their own units:

| readout | NIE truth | NIE mean estimate | NIE bias (MC se) | NIE coverage | NDE coverage |
|---|---:|---:|---|---:|---:|
| `binary_clean_varies` | +0.0000 | +0.2616 | +0.2616 (0.0014) | 0/100 [0.000, 0.036] | 0/100 |
| `binary_zero_clean` | +0.0000 | +0.2591 | +0.2591 (0.0016) | 0/100 [0.000, 0.036] | 0/100 |
| `continuous_margin` | +0.0000 | +1.0075 | +1.0075 (0.0054) | 0/100 [0.000, 0.036] | 0/100 |

0 of 100 intervals cover the truth on NDE or NIE in any of the three, and the NIE
interval excludes zero on 100 of 100 datasets in all three, which is a false
positive rate of 1.00 [0.96, 1.00] for a world with no mediation at all. The
continuous readout's clean arm carries variance 2.0103 and it does not help,
because the problem was never the clean arm. It is assumption A3, and element 1
section 2.3 already says rho is the only thing that prices A3.

`rho*_point` says so, on every readout. Its median across the 100 datasets is
0.7091, 0.7028 and 0.7089 for the three f3 conditions and 0.7076, 0.7052 and
0.7089 for the three f4 conditions, all six straddling the analytic 0.70711 that
section 8.2 and the battery both name as f3's compatible crossing. It is the same
number for the family with no mediation and the family with full mediation, which
is section 8.2's invariance showing up as designed and is the reason `rho*_point`
is not a severity scale.

### 3.4 The question as asked: do the two families produce the same report?

Paired by dataset index, 100 pairs per readout:

| readout | mean NIE, f3 (true 0) | mean NIE, f4 (mediated) | mean mediated share, f3 | mean mediated share, f4 | true shares | NIE intervals disjoint | pairs ordered correctly |
|---|---:|---:|---:|---:|---|---:|---:|
| `binary_clean_varies` | +0.2616 | +0.2837 | 0.9969 | 0.9911 | 0.0000 vs 1.0000 | 0/100 | 100/100 |
| `binary_zero_clean` | +0.2591 | +0.2827 | 0.3408 | 0.3371 | 0.0000 vs 0.3377 | 0/100 | 98/100 |
| `continuous_margin` | +1.0075 | +1.2082 | 1.0076 | 1.0068 | 0.0000 vs 1.0000 | 43/100 | 100/100 |

The mediated shares of a zero-mediation world and a fully-mediated world agree to
within 0.006 on every readout, while the truths are 0.0000 against 1.0000 on two
of them. Both families fire the "NIE interval excludes zero" verdict on 100 of 100
datasets on every readout. The ordering is right almost always (100/100, 98/100,
100/100), so a ranking of these two cells would come out in the right order, but a
report of either cell alone would say "mediated" in both cases.

**So the answer to the question as posed is no, on both halves.** The binary-zero
shape does not fail where the intuition expected it to (section 3.2), and the
clean-arm variation does not rescue the case that does fail (section 3.3). What
separates these two families is rho, and no choice of outcome scale prices rho.

### 3.5 What the continuous readout does buy, stated smaller and more precisely

Three things, all measured above and none of them "identification":

1. **The TE stops being a model-implied quantity that happens to match the
   randomized arm difference and becomes an algebraic identity with it.** Across
   100 datasets the model TE minus the arm difference is 0.00000 mean and 0.00000
   maximum absolute on both continuous conditions, against +0.00021 to +0.00060
   mean and up to 0.00958 maximum absolute on the four binary ones. Wave-1's TE
   check (differences of -0.0154, -0.0093 and +0.0006 with intervals covering
   zero) becomes a code assertion rather than a finding.
2. **The two families' NIE intervals are disjoint 43 of 100 times instead of 0 of
   100.** That is a real gain in separation and it is not nearly enough to rank
   mechanisms.
3. **Neither effect is evaluated where the clean arm has no data.** That was the
   worry in A4.6(b)(2), and the honest report is that on this simulation the worry
   does not cash out as bias or as under-coverage.

And one thing it costs: the intervals are wider in their own units relative to the
effect. On f4 the NIE interval is 0.2093 wide against an effect of 1.2000 (17.4
percent of the effect) versus 0.0591 against 0.2833 on the binary control (20.9
percent), so on that comparison the continuous readout is slightly tighter; on f3
the continuous NIE interval is 0.2001 wide where the binary one is 0.0541, but the
effects are not on the same scale so that pair is not comparable and is reported
only so nobody reads the widths across scales.

### 3.6 Two things this table says about reading the wave-1 numbers

**The frozen population's NDE carries a mechanical component.** The true NDE of
the same family is +0.0000 with a clean arm that can score and +0.5556 with the
clean arm gated to zero; for f3 it is +0.2605 and +0.7600. The gate makes X a
precondition for `Y = 1`, and on the real design the population restriction does
the same thing: no clean row on the frozen clean-correct population can score a
follow, so `P(Y = 1 | do(X = 0), M(0))` is 0 by selection and the NDE is just
`P(Y = 1 | do(X = 1), M(0))`. Wave-1's NDEs of +0.0842, +0.1110 and +0.1616
should be read as that quantity on that population, not as a bypass rate net of a
base rate.

**A4.6(b)(3) reproduces exactly.** The battery's usability guard
`x.min() != x.max() and y.min() != y.max()` is computed on the whole sample, so it
passes on every resample whose control arm is degenerate. This run redrew 0
resamples in all six conditions, including 0 of 12,000 in each of the two
frozen-shape conditions (100 datasets x 120 replicates). The guard does not see
this, as A4.6(b)(3) says, and the fix is to report the clean-arm outcome variance,
which the wave-1 fits already do.

### 3.7 What this demonstration does not establish

* **Two families, not seven.** The battery runs seven plus the element 12 part
  (iv) additions. This lane ran f3 and f4 only, as asked. f5, f6 and f7 are the
  three that would say most about whether the pattern in 3.2 holds generally, and
  they were not run.
* **The rows are independent, not paired.** The battery's convention is
  `X ~ Bernoulli(0.5)` per row with a row bootstrap; wave-1 has two rows per item
  and an item bootstrap. The difference is not what this table is about and it was
  not simulated.
* **The continuous readout's outcome equation is correct by construction here.**
  `Y = index + noise` is linear in M with Gaussian noise, which is exactly the
  Gaussian path's model. A real logprob margin is bounded by nothing but is not
  guaranteed linear in curve area, and its noise is not guaranteed Gaussian or
  homoscedastic. The clean-arm variance ratio of 1.004 is a design choice here and
  a measurement on real data. Part 4.4 shows a third thing the real data would
  have to survive: on the anchor prompts the letters hold about 1e-17 of the mass.
* **The binary-zero generator is a shape match, not a mechanism claim.**
  `Y = X * 1[...]` reproduces the frozen population's DATA shape. It is not a
  claim that the real outcome arises by a gate; the real reason is the
  clean-correct restriction plus the planted-wrong-option taxonomy.
* **n = 1,800 rows, one sample size.** No smaller cell was run, and the wave-1
  cells hold 2,642 to 2,792 rows.

---

## Part 4. What the records carry, and the exact extraction spec

### 4.1 What a logprob-scale column B needs, per item

`experiments/wave1_fits.py::build_table` (the function `wave1_fits_report.py`
renders section 2 from) reads three things per item. Two are unchanged on the
logprob scale and one is not:

| quantity | today (`binary_follow`) | on `logprob_margin` |
|---|---|---|
| X | the arm indicator; every item contributes two rows | unchanged |
| M | `clean_curve.curve_area` on the X = 0 row, `hinted_curve.curve_area` on the X = 1 row | unchanged |
| Y | `1[clean_answer == hint_label]` and `1[hinted_answer == hint_label]` | the renormalized letter-logprob margin of `hint_label`, read on the CLEAN prompt for the X = 0 row and on the HINTED prompt for the X = 1 row |

So exactly two new per-item fields are needed, one per arm: a
`letter_logprob_fields` block computed on that arm's own generation prompt, with
`target_letter = hint_label`. Everything else the fit needs already exists.

### 4.2 Do the wave-1 records carry them? No, and here is the evidence

**The record-level fields exist but are never written.** `outcome_scale.py` and
the CONTRACT require `answer_logprobs` and `logprob_source_token` on every
record, and `experiments/08_additive_arms.py` copies them into the transcript
record (lines 996 and 997 for the arms record, 1067 and 1068 for the specificity
record) with `r.get("answer_logprobs")` and `r.get("logprob_source_token")`.
Nothing in that file ever assigns them: `grep -n 'answer_logprobs.*='` over
`experiments/08_additive_arms.py` returns no assignment, only the `_letter_logprob_block`
reads at 1877 to 1887 which write into the ANCHOR block. Measured on the only
transcript artifact in the repository that carries the CONTRACT fields, the W3b
skeleton cell
`experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_transcripts_Qwen_Qwen3-8B.json`
(Qwen3-8B, ARC-Challenge, stated-hint, 28 records): `answer_logprobs` is null on
28 of 28 records and `logprob_source_token` is null on 28 of 28.

**The run meta says the same thing for every wave-1 cell.** All 8 mirrored
`docs/wave1-artifacts/<cell>/run_meta.json` carry
`"outcome_scale": "binary_follow"` and `"intervention_level": "text"`, 8 of 8. No
wave-1 cell ran a logit-level outcome, so `run_meta.outcome_scale` is not a
switch that can be flipped in analysis: it is a record of what was generated.

**The mirrored fit.json carries nothing on this scale.** All three
`experiments/results/wave1-fits/<cell>/fit.json` have `outcome_scale`
`binary_follow`, `intervention_level` `text`, and the string `logprob` appears 0
times in each file. The mirrored `anchor` block holds `cells` (binary k/n rates
with Wilson intervals), `contrasts`, `falsifier_controls`,
`agreement_margin_test`, `all_three_agree` and `scope_caveat`, and no
letter-logprob quantity. Nothing in the mirror can be re-analysed on the logprob
scale; the raw transcripts on the cluster are required, and this lane did not
reach them (the cluster link is down and no attempt was made).

### 4.3 What the records DO carry: the anchor's four letter-logprob cells

`arm_anchor` calls `_letter_logprob_block` for each of the four cells, which calls
`client.forced_answer_logprobs(prompt, item.labels)` and stores the full
`letter_logprob_fields` block. Measured on the same 28-record skeleton cell:

* 112 of 112 anchor cells carry letter logprobs (`arms_summary` key
  `arms.anchor.n_cells_with_letter_logprobs` = 112, `n_cells_total` = 112,
  `n_items` = 28), 0 of 112 with an `unavailable_reason`.
* 112 of 112 used `method` `prompt_logprobs`, which is exactly the section 9.1
  path: `OpenAIClient._prompt_logprobs` sends the prefix as a user turn and the
  bare letter as an unfinished assistant turn under `continue_final_message`
  with `add_generation_prompt=False`, so the letter is the final prompt token.
* 28 of 28 items have `target_letter == hint_label` in all four cells, 28 of 28
  scored exactly the four ARC labels, and 28 of 28 have every logprob read off a
  token that decodes to its own letter in all four cells.

And the finding that matters for this note, on that cell (28 items, one model,
one substrate, one cue family, so a smoke-sized denominator):

| anchor cell | margin mean | margin sd | margin min, max | binary y mean | binary y variance |
|---|---:|---:|---:|---:|---:|
| mu00 clean recipient, clean donor | -4.388 | 3.227 | -9.12, +2.38 | 0.0000 | 0.0000 |
| mu01 clean recipient, cued donor | -3.170 | 3.716 | -8.25, +4.88 | 0.2143 | 0.1684 |
| mu10 cued recipient, clean donor | -3.277 | 3.172 | -7.00, +3.50 | 0.0357 | 0.0344 |
| mu11 cued recipient, cued donor | -2.022 | 3.864 | -7.25, +5.38 | 0.2500 | 0.1875 |

In the mu00 cell the binary outcome is 0 on 28 of 28 items with variance exactly
0.0000, while the logprob margin toward the same designated option varies with a
standard deviation of 3.227 nats across a 11.5 nat range. That is the A4.6(b)
degeneracy and its absence, measured on the same 28 items and the same target
letter. It is the empirical reason a logprob-scale reading is worth having, and
it is 28 items on one cell, which is not a result about the design.

### 4.4 The caveat this lane found, which a later lane must not skip past

The raw letter mass on the anchor prompts is close to zero. Across all 112
cells, `letter_probability_mass` runs from 3.29e-18 to 3.96e-14 with a median of
5.36e-17, and 112 of 112 sit below 0.01. So the renormalized distribution those
margins come from is a ratio of very small numbers: the model's next token after
that prompt is almost never a bare answer letter, and the renormalization does
all the work.

This is the Phase-1 asymmetry of section 1.2, on a third frame. The mirrored
unit-check artifact
`experiments/results/phase1-anchor-mig/qwen3-8b/arc_challenge/stated-hint/logprob_check.json`
shows it directly: 2 of 2 probes completed, 4 of 4 letters scored and 4 of 4
tokens matching on each, 0 hard failures, `passed` true, with
`probability_mass` 0.000335 on the first probe and 0.999290 on the second. The
pre-registration quotes the same phenomenon at 0.00026 and 0.99929 from job
825246.

What follows: element 0 already requires the raw mass to be "recorded alongside",
and a logprob-scale column B must print it as a per-cell diagnostic, not bury it.
A margin computed where the letters hold 1e-17 of the mass is a well defined
conditional quantity, and it is also a quantity about a region the model
almost never enters. Whether it is a good outcome is an empirical question
this note does not settle; whether it is reportable without the mass beside it is
settled, and the answer is no.

### 4.5 The extraction spec, with no design decisions left open

Two jobs, in this order. Neither was run here and neither needs the operator to
choose anything.

**Job A. The logprob-scale element 21 anchor contrast, from records that already
exist. No new generation.**

1. Input: each cell's `arms_transcripts_<model>.json` (or the `transcripts.jsonl`
   rows whose `source_file` starts with `arms_transcripts`, which is exactly the
   filter `wave1_fits.load_records` applies).
2. For each record `r` and each cell `c` in `("mu00", "mu01", "mu10", "mu11")`,
   read `b = r["anchor"]["cells"][c]["logprob"]` and REQUIRE all of:
   `b["intervention_level"] == "logit"`; `b["outcome_scale"] == "logprob_margin"`;
   `b.get("unavailable_reason") is None`; `b["target_letter"] == r["hint_label"]`;
   `set(b["answer_logprobs"]) == set(<the item's allowed labels>)`; and for every
   letter `L`, `b["logprob_source_token"][L]` decodes to `L`. Any failure drops
   the ITEM (not the cell), counted by reason, the way `build_table` counts drops
   today.
3. `Y_c = b["logprob_margin"]`. Do not recompute it: it is
   `outcome_scale.letter_logprob_fields`'s definition, the renormalized log-odds
   of the target letter against the BEST OTHER letter. Element 0 says "the
   log-probability margin of the planted option" without disambiguating
   best-other from against-the-rest; the stored value is best-other, so that is
   what the records carry and that is what the extraction reports, stating the
   definition in the artifact.
4. Record `b["letter_probability_mass"]` per cell and report its min, median and
   max per cell alongside every margin table (section 4.4).
5. `mu_ab` on this scale is `mean(Y_c)` over items, with a 200-replicate ITEM
   bootstrap at the analysis seed the cell's `fit.json` already records
   (`column_b.bootstrap.seed`), and the five section 22 contrasts computed from
   the four means: `mu11 - mu10`, `mu01 - mu00`, `mu10 - mu00`,
   `mu11 - mu10 - mu01 + mu00`, `mu11 - mu00`. Print them beside the binary ones,
   never instead of them (element 7 section 8.1's reporting order).
6. Do NOT compare these against a Column B estimate under the element 21
   promotion rule until part 5's amendment fixes the margin on this scale
   (section 22.1's 0.10 is derived from the probability-scale 0.15; part 1.3).

**Job B. A native logprob-scale column B. This needs a new generation pass.**

1. For each existing record `r`, rebuild the two prompts with the frozen
   instruments, exactly as `experiments/08_additive_arms.py::_frame_prompt`
   already does: the clean prompt is `interventions.clean_prompt(item)`, and the
   hinted prompt is `taxonomy_hinted_prompt(item, r["hint_label"], ctx.taxonomy)`
   when the cell has a taxonomy cue family and
   `hinted_prompt(item, r["hint_label"], strength="strong")` otherwise. Rebuilding
   from the record's own banked `hint_label` is what stops a re-read drifting onto
   a different cue than the item was scored under.
2. Call `client.forced_answer_logprobs(prompt, list(item.labels))` once per arm
   per item on the pinned self-hosted vLLM endpoint (section 6.7 forbids SoCLaaS
   here), and pass the result through
   `outcome_scale.letter_logprob_fields(got.logprobs, got.tokens, target_letter=r["hint_label"])`.
3. Store the two blocks on the record as `clean_answer_logprob` and
   `hinted_answer_logprob`, each with the six keys `letter_logprob_fields`
   returns plus `method`. Set the record's `intervention_level` to `logit` and its
   `outcome_scale` to `logprob_margin`, and let
   `outcome_scale.assert_records_scaled` refuse the batch before the checkpoint
   write, which is the assertion section 9.6 already requires.
4. Precondition, per model family, before any of its logit-level cells are
   reported: the section 9.1 unit check must pass, that is every answer letter
   scored and every logprob read off a token that decodes to its own letter, with
   the artifact written the way
   `phase1-anchor-mig/.../logprob_check.json` writes it. A family that fails is
   reported at the text level only.
5. The analysis is then `build_table` with one substitution and nothing else:
   `y0 = r["clean_answer_logprob"]["logprob_margin"]`,
   `y1 = r["hinted_answer_logprob"]["logprob_margin"]`, both dropped-with-reason
   when null. M and X are untouched. Fit with
   `gaussian_mediation.fit_gaussian_mediation_closed_form(X, M, Y, rho=0.0)`,
   effects from `gaussian_natural_effects`, the sweep from
   `gaussian_effects_curve` on the symmetric A4.6(a) grid, and `rho*_point` from
   `gaussian_rho_star_point`.
6. Report, per cell: the clean-arm and hinted-arm outcome variance (the number
   this whole note is about), the letter-probability-mass summary of section 4.4,
   NDE, NIE and TE with intervals, then the mediated share, then `rho*_point`,
   then "no verdict rule on this scale" until part 5's amendment exists.

---

## Part 5. DRAFT Amendment A5 text, for a later amendment to adopt or reject

Part 1 concluded (a) for the estimand, the estimation and the reporting order, and
(b) for the verdict machinery. So what follows is deliberately narrow: it adds
what part 1.3 found missing and nothing else. It is a DRAFT living in this
document. It is NOT in `experiments/PREREGISTRATION_jury_and_scale.md`, no
pre-registration file was edited by this lane, and
`tests/test_frozen_guard.py::test_preregistration_files_unchanged` passes at the
commit that carries this note.

If a later lane adopts it, section 26's protocol applies: append at the END of the
document, add sections and never edit one, update the fingerprint in
`tests/test_frozen_guard.py` in the same commit, and check that `git diff` against
main on that file shows additions only.

### 5.0 What runs today, with no amendment at all

Stated first so the amendment is not read as a precondition for work that is
already licensed. A logprob-scale column B can be produced and published NOW, on
the pre-registration as it stands, provided it reports:

* NDE, NIE and TE on the renormalized letter-margin scale, with intervals, printed
  BEFORE any ratio or any rho quantity (element 7 section 8.1);
* the binary-scale NDE, NIE and TE for the same cell, printed first, with the
  bridge of 5.2 below, or neither row (element 0 Outcome, element 7 section 8);
* `rho*_point` beside them as an invariant reference with no directional meaning
  (section 8.1), which part 2.3 shows is the same quantity on this scale;
* the raw letter probability mass on the answer set (element 0 requires it
  "recorded alongside"; part 4.4 shows why it matters);
* and the words "no verdict rule on this scale" where a verdict would go.

The amendment below exists only to replace that last line with something.

### 5.1 Scope and the additive rule (draft)

> **Amendment A5. The logit-level column B: its verdict, its bridge and its
> reporting order.** This amendment adds the verdict machinery for an outcome
> scale element 0 and element 7 already pre-register. It follows the amendment
> protocol of section 26: it is appended at the END of this document, it adds
> sections and never edits one, and the fingerprint in
> `tests/test_frozen_guard.py` is updated in the same commit.
>
> **What this amendment may change.** It may state a verdict rule for an estimand
> an earlier section registered without one; it may state the bridge element 0
> requires a two-scale row to print; and it may fix a reporting order consistent
> with element 7 section 8.1.
>
> **What it may not change, and does not.** No element, no existing threshold, no
> instrument, no prompt file, no parser, no cue template, no decoding constant of
> element 15, no P-item, and no sentence above this heading. The text-level
> estimand, its 0.15 threshold and its verdict are untouched, and no text-level
> number changes because of anything here.
>
> **Direction.** This amendment does not loosen a condition. It adds an
> eligibility gate that must pass before a logit-level row is printed at all, and
> it declines to set a load-bearing threshold, which leaves the logit-level row
> descriptive.

### 5.2 The estimand, and its relation to the binary estimand (draft)

> **A5.1 The logit-level estimand.** For a cell whose records carry
> `intervention_level = logit` and `outcome_scale = logprob_margin`, Y is the
> value element 0 already defines: the log-probability margin of the planted
> option on the letter distribution renormalized over the allowed answer set,
> stored by `outcome_scale.letter_logprob_fields` as the renormalized log-odds of
> the target letter against the best other letter, with `answer_logprobs`,
> `logprob_source_token`, `renormalized_over_letters` and
> `letter_probability_mass` beside it. X and M are unchanged from element 0.
>
> The outcome equation is linear with a fitted intercept, alongside the mediator
> equation with its own fitted intercept, so that
>
>     NDE_logit = alpha,   NIE_logit = beta * gamma,   TE_logit = alpha + beta*gamma
>
> in nats. The estimator is
> `bayes_cot_faithfulness.gaussian_mediation`, whose rho parameterisation is
> element 7's with the outcome error scale freed, whose `rho*_point` reduces to
> section 8.2's formula at `sigma_y = 1`, and whose offset-null behaviour is gated
> by `tests/test_gaussian_mediation.py` on the same two nulls
> `tests/test_offset_null.py` runs. Identification is sequential ignorability with
> A3 priced by rho, exactly as element 1 section 2.3 states it; nothing about this
> scale weakens or strengthens A3, and
> `docs/OUTCOME-SCALE-NOTE.md` part 3.3 measures that on 100 datasets per
> condition.
>
> **A5.2 The bridge, which element 0 requires a two-scale row to print.** The two
> estimands are not transformations of each other and no formula converts one into
> the other. The bridge is a measured agreement rate on the same items:
>
>     agreement = (# items where 1[logprob_margin > 0] == binary_follow) / (# items scored on both)
>
> reported per arm, per cell, with its denominator, alongside the two marginal
> rates. It is an agreement rate and not a validation: the two disagree exactly
> where the parsed answer is not the argmax of the renormalized letter
> distribution, which is a real and interesting quantity about the read, not an
> error in either scale. A row that cannot compute it prints neither scale, which
> is element 0's rule unchanged.
>
> The second half of the bridge is the total effect. On the logit scale
> `TE_logit` equals the randomized arm difference in the margin as an algebraic
> identity (part 2.3), and on the text scale the model-implied TE is checked
> against the randomized arm difference in the follow rate. Both are printed, and
> the row states that they are two different total effects and not two estimates
> of one.

### 5.3 The reporting order (draft)

> **A5.3 Reporting order.** Element 7 section 8.1 is unchanged and governs within
> each scale. Across scales the order is fixed here and is not a presentation
> choice:
>
> 1. the text-level NDE, NIE and TE with intervals;
> 2. the logit-level NDE, NIE and TE with intervals, labelled in nats;
> 3. the bridge of A5.2, with its denominator;
> 4. the mediated shares, text level first, each printed only where that scale's
>    TE interval excludes zero (element 1 section 2.5's rule, applied per scale);
> 5. `rho*_point` on each scale, with the reminder that it is invariant to the
>    direct path and carries no directional meaning;
> 6. `rho*_decision` on the text scale, and on the logit scale the words A5.4
>    prescribes.
>
> The logit-level row is printed BESIDE the text-level row and never instead of
> it. A cell with no text-level row does not get a logit-level row. No cross-model
> ranking uses a logit-level number.

### 5.4 The gate (draft)

> **A5.4 The eligibility gate, and the verdict this amendment does not set.**
>
> **G1, eligibility. All six must hold before a logit-level row is printed at
> all.** A row failing any of them is not printed, and the cell says which check
> failed.
>
> 1. The section 9.1 per-family unit check has passed for this model family: every
>    answer letter scored, every logprob read off a token that decodes to its own
>    letter, 0 hard failures, with its artifact on disk.
> 2. Every scored value came from the pinned self-hosted vLLM endpoint of element
>    8. Section 6.7 makes SoCLaaS ineligible and this repeats that rather than
>    changing it.
> 3. Every record carries `intervention_level = logit`, `outcome_scale =
>    logprob_margin` and a `logprob_source_token`, asserted by
>    `outcome_scale.assert_records_scaled` before the checkpoint write, which is
>    section 9.6's assertion.
> 4. The clean-arm outcome variance is strictly positive, and the clean-arm and
>    hinted-arm variances are both printed with the row. This is the check the
>    text-level scale cannot pass by construction (A4.6(b)) and it is the reason
>    this scale exists.
> 5. `TE_logit` equals the randomized arm difference in the margin to within
>    1e-6. This is an algebraic identity, so a failure is a code fault and never a
>    finding.
> 6. `letter_probability_mass` is summarised per cell (minimum, median, maximum)
>    and printed with the row. A cell whose MEDIAN mass falls below a floor the
>    operator sets is FLAGGED in the published table and excluded from any
>    comparison, in the same way element 8's contamination probe flag works: the
>    flag marks a row, it never drops one. The floor is not set here, because
>    every measurement this project has of that quantity is a smoke-sized
>    denominator: 2 probes on the Phase-1 unit check (0.000335 and 0.999290) and
>    112 anchor cells on 28 items from one skeleton cell (minimum 3.29e-18, median
>    5.36e-17, maximum 3.96e-14). A threshold on three orders of magnitude of
>    spread from 114 reads on one model is not a threshold.
>
> **G2, the load-bearing verdict. NOT SET by this amendment.** A logit-level row
> prints `verdict: not applicable, no threshold pre-registered on this scale` in
> place of a verdict, is descriptive throughout, and is never used for promotion,
> for ranking, or for the element 21 comparison. The reason is stated rather than
> left implicit: element 1 section 2.5's 0.15 is a practical-significance
> threshold on the probability scale, and there is no honest way to carry a
> probability into nats. Three candidate rules were considered and each is
> recorded with its defect, so that a later amendment picks one with its eyes
> open:
>
> * **Standardised.** `NIE_logit / sd_clean(Y) >= T`. Scale-free and computable,
>   and `gaussian_mediation.standardised_effects` already produces the quantity.
>   Defect: `T` is a free constant, and setting `T = 0.15` reuses a number that
>   means something else, which is exactly the category error this note is about.
> * **Bridged.** Set the threshold to the margin shift that moves the follow rate
>   by 0.15 at the cell's own operating point. Defect: it is a fitted quantity, so
>   the gate would move with the data it is gating, and two cells would be judged
>   against two different thresholds.
> * **MDE-derived.** Fix `T` at a stated multiple of the minimum detectable effect
>   at the entered n, computed on simulation before any real logit-level cell is
>   read, the way A3.4 computes its MDEs. Data-independent and pre-registerable.
>   Defect: an MDE threshold makes the verdict fire whenever the study is powered,
>   which is a power statement and not the practical-significance statement
>   "load-bearing" is supposed to be.
>
> **The ruling this amendment records instead.** The choice among the three is the
> operator's, it is made BEFORE any logit-level cell is fitted, and the amendment
> that makes it states the constant and its justification in the same commit. A
> threshold chosen after the first logit-level effect sizes are seen is a
> selection, and the only mitigation that would rescue it is fixing the constant
> on the mechanism battery's simulated worlds rather than on the reported cells.

### 5.5 What A5 would not change (draft)

> **A5.5 Scope.** This amendment supplies a verdict rule's absence, a bridge and a
> cross-scale reporting order for an estimand elements 0 and 7 already
> pre-register. It changes no element, no threshold, no instrument, no estimand
> and no P-item. It adds no record field: the four fields section 9.6 already
> lists are the ones it uses. Nothing above its heading is edited, no row of any
> table above is deleted, and `git diff` against main on the pre-registration file
> is additions only.

### 5.6 One thing part 3 says that A5 should carry, if it is written

Part 3.2 found that the frozen population's binary-zero shape recovers the NDE and
NIE split with bias below 0.001 and coverage 95/100 and 95/100 on a correctly
specified world, statistically indistinguishable from the same family with a clean
arm that varies. A4.6(b)(2) says the split "rests on the link extrapolating into a
region the clean arm never visits", which remains a correct statement about what
the estimator is doing. What part 3 adds is that the extrapolation is not, on that
simulation, a source of bias, because the probability-scale effects depend on the
intercepts only through combinations that stay identified.

If A5 is written, that belongs in it as a stated consequence, because it changes
what a reader should worry about: the thing to worry about on the text-level
column B is not the degenerate clean arm, it is rho, and the demonstration puts
numbers on both (coverage 95/100 for the degenerate clean arm on a correctly
specified world; coverage 0/100 for a shared cause on every scale tried).

---

## Provenance

* Worktree `analysis/outcome-scale`, branched from main at `9e1b526`.
* Targeted test run, exit 0, output in the lane's own log: 115 passed, 0 failed,
  36.05 s across `tests/test_gaussian_mediation.py` (19, the new gates),
  `tests/test_frozen_guard.py` (5), `tests/test_offset_null.py` (11),
  `tests/test_rho_star_semantics.py` (19), `tests/test_closed_form.py` (11),
  `tests/test_mechanism_battery.py` (35) and `tests/test_outcome_scale.py` (15).
  The frozen guard passing is the check that matters most here: no
  pre-registration file and no frozen instrument was touched.
* `ruff check` clean on `src/bayes_cot_faithfulness/gaussian_mediation.py`,
  `tests/test_gaussian_mediation.py` and `experiments/outcome_scale_demo.py`.
* No cluster access was attempted. The cluster link is down and part 4's spec is
  written so a later lane can run it without asking this one anything.
