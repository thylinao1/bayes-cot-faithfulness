# The link audit and scale-aware outcome priors (W4c)

**Date:** 7 September 2026
**Branch:** `w4c/scale-aware-priors` (worktree `~/Developer/bcf-w4c`), base `5840afc`
**Trigger:** finding (4) of the CPU mechanism battery, DECISION-LOG 2026-09-07 04:02. The
PyMC path under-covers where the mediator baseline is large: NIE coverage at n = 350 over
20 datasets per family was 7/20 in family 4 and 12/20 in family 7, against 93/100 for the
maximum-likelihood bootstrap on the same datasets.
**Frozen files:** untouched. `tests/test_frozen_guard.py` runs in every gate below.

---

## 1 · The link audit, before any change

Asked of the code, not of the prose: which link does each path use? Line numbers are from
`5840afc`, the base of this branch.

| Path | File and line at `5840afc` | Link |
|---|---|---|
| PyMC likelihood | `src/bayes_cot_faithfulness/mediation.py:92` — `pm.Bernoulli("Y_obs", logit_p=logit_y, observed=Y)` | **logit** |
| PyMC draw-to-effect conversion | `src/bayes_cot_faithfulness/effects.py:138-140` — `_sigmoid(alpha0 + alpha*x + beta*m).mean()` | **logit** |
| Maximum-likelihood negative log-likelihood | `src/bayes_cot_faithfulness/sensitivity.py:314` and `:328` — `np.where(Y == 1, norm.logcdf(eta), norm.logsf(eta))` | **probit** |
| Maximum-likelihood draw-to-effect (`probit_natural_effects`, `natural_effects_from_fit`) | `src/bayes_cot_faithfulness/sensitivity.py:250-252` | **probit** |
| Closed-form natural effects | `src/bayes_cot_faithfulness/closed_form.py` — `norm.cdf(...)` throughout | **probit** |
| Ground truth for the logistic generator | `src/bayes_cot_faithfulness/effects.py:79-81` (`monte_carlo_true_effects`) | logit, and correct: `synthetic.py` is a logistic generator |

**Finding: the two estimator paths disagreed.** Each was internally consistent — the PyMC
likelihood and its converter were both logistic, the maximum-likelihood fit and its
converter and the closed form were all probit — so nothing raised. But the same
`(alpha, beta, gamma, sigma_m, mu_m, alpha0)` symbols meant two different models, and the
mechanism battery reported the two paths side by side against a single probit truth.
`experiments/mechanism_battery_stats.py:183-187` at `5840afc` says so in its own docstring
and calls the difference deliberate and small.

It is not small enough to ignore. On the battery's rationalization parameters
(`alpha 0, beta 1, gamma 1.2, sigma_m 1, mu_m 6, alpha0 -5.8`) the *same coefficients*
converted under the two links give NIE +0.2827 (probit) against about +0.24 (logistic), a
gap of roughly 0.04 on an effect the pre-registered verdict rule thresholds at 0.15.
`tests/test_link_agreement.py::test_the_two_links_are_not_interchangeable_on_the_same_coefficients`
pins that the gap exists and exceeds the agreement tolerance.

### Which link wins, and why

**Probit, everywhere.** The pre-registered estimand is written on the probit scale: section
8.3 of `experiments/PREREGISTRATION_jury_and_scale.md` states the mediator-noise attenuation
as `c = sqrt(1 + beta^2 sigma_m^2 (1 - lambda))`, which is the probit latent-scale
rescaling; section 13.2 names `fit_probit_mediation_map` as the estimator whose frontier
ships; and the whole rho / rho\* sensitivity apparatus is defined as a residual correlation
in a bivariate *normal* error, which exists only in the probit model. The frozen text was
not touched to reach this conclusion and is not touched by this branch.

### What changed

- `mediation.fit_mediation_model` grows `link: str = "probit"` and fits a probit outcome by
  default (`src/bayes_cot_faithfulness/mediation.py:180`, a `pm.Potential` carrying a
  numerically stable `log Phi`; see `_log_std_normal_cdf`, which uses
  `erfc(z) = erfcx(z) exp(-z^2)` so the left tail cannot underflow to `-inf` and hand NUTS
  an infinite log-density). `link="logit"` keeps the logistic model for the logistic
  generator in `synthetic.py`.
- The trace now records its own link in `trace.attrs["outcome_link"]`.
- `effects.posterior_natural_effects` grows `link: str = "probit"`. Under probit it calls the
  exact closed form (vectorised as
  `closed_form.probit_natural_effects_closed_form_samples`), so the posterior path and the
  maximum-likelihood path now share **one written definition of the estimand**. Under logit
  it is the previous Monte Carlo integrator, unchanged.
- `mediation.natural_effects_from_trace` is the single conversion that reads the link off the
  trace, for the same reason `sensitivity.natural_effects_from_fit` exists. The battery's
  PyMC subset now goes through it, so no call site can pair draws with the wrong formula.
  A trace with no stamp raises rather than being assumed logistic.
- The logistic call sites are explicit about it: `tests/test_effects.py`,
  `tests/test_mediation.py`, `notebooks/01_synthetic_validation.py`,
  `notebooks/02_generate_figure.py` now pass `link="logit"`, because the data they fit comes
  from a logistic generator.

---

## 2 · Scale-aware outcome priors

### The mechanism, stated in one paragraph

The outcome index was `alpha0 + alpha*X + beta*M` with `alpha0 ~ Normal(0, 1.5)` and
`beta ~ Normal(0, 2)` (`mediation.py:77-88` at `5840afc`). `alpha0` is then the index at
**M = 0**, which for a reasoning-step count with a baseline near six is a place the data
never visits. In the battery's family 4 the implied intercept is `-5.8` on the probit scale
(and about `-11.2` on the logit scale the model actually used), that is, 3.9 and 7.5 prior
standard deviations from the prior mean. The posterior cannot go there, so it shrinks
`alpha0` and `beta` together and the mediated effect comes out low. `beta ~ Normal(0, 2)` has
the same defect in the other direction: it says something different about a mediator counted
in steps than about one counted in tokens.

### The change

`mediation.fit_mediation_model`, `intercepts=True` branch. Write `mbar = mean(M)`,
`s = sd(M)`. The outcome index is now `alpha0_centered + alpha*X + beta*(M - mbar)`:

| Parameter | Before (`5840afc`) | After | Why |
|---|---|---|---|
| `mu_m` | `Normal(mean(M), max(sd(M), 1))` | `Normal(mbar, s)` | unchanged in intent; the floor at 1 was the last absolute unit left in the mediator block |
| `gamma` | `Normal(0, 1.5)` | `Normal(0, 1.5*s)` | a treatment shift of a few mediator standard deviations, in any unit |
| `sigma_m` | `HalfNormal(1.0)` | `HalfNormal(s)` | likewise for the mediator's own spread |
| `beta` | `Normal(0, 2.0)` | `Normal(0, 2.0/s)` | so the implied index contribution `beta*s` is `Normal(0, 2)` for every mediator unit |
| outcome intercept | `alpha0 ~ Normal(0, 1.5)`, the index at M = 0 | `alpha0_centered ~ Normal(0, 1.5)`, the index at the mean mediator | a clean-arm answer rate anywhere in roughly [0.07, 0.93] at one prior sd, instead of a claim about a mediator value the data never takes |
| `alpha0` (reported) | free parameter | `Deterministic(alpha0_centered - beta*mbar)` | downstream reads the un-centred intercept it has always read; `extract_intercept_samples` is unchanged |

Exact lines after the change: `src/bayes_cot_faithfulness/mediation.py:154-162`.

```python
alpha = pm.Normal("alpha", mu=0.0, sigma=INDEX_PRIOR_SD)                       # 1.5
beta = pm.Normal("beta", mu=0.0, sigma=MEDIATOR_INDEX_PRIOR_SD / m_sd)         # 2.0 / sd(M)
gamma = pm.Normal("gamma", mu=0.0, sigma=GAMMA_PRIOR_SD_IN_M_SD * m_sd)        # 1.5 * sd(M)
sigma_m = pm.HalfNormal("sigma_m", sigma=m_sd)                                 # sd(M)
mu_m = pm.Normal("mu_m", mu=m_bar, sigma=m_sd)
alpha0_c = pm.Normal("alpha0_centered", mu=0.0, sigma=INDEX_PRIOR_SD)          # 1.5
pm.Deterministic("alpha0", alpha0_c - beta * m_bar)
```

`intercepts=False` is untouched: fixed-scale priors, no intercepts, mediator un-centred, so
`intercepts=False, link="logit"` still reproduces the pre-repair model exactly and every
historical number stays reproducible from this repository.

### The estimand is unchanged, and that is a test, not a claim

The natural effects depend on the intercepts only through the offset `alpha0 + beta*mu_m`,
which centring leaves alone; and with every mediator prior stated in units of `s`, the whole
model is equivariant under `M -> (M - shift)/scale`. So two invariances must hold exactly up
to Monte Carlo error, and `tests/test_scale_aware_priors.py` asserts both at a tolerance of
0.01 on a 1,500-row dataset with `mu_m = 6`.

**Both checks were run against the pre-change model first and both failed** (scratch run,
`fit_mediation_model` loaded from `5840afc`, same data, same seeds, converted with the
logistic converter it was paired with):

| Fit of the same data | NDE | NIE | TE | beta | alpha0 | mu_m |
|---|---:|---:|---:|---:|---:|---:|
| as given, mediator mean 6 | −0.0172 | +0.2627 | +0.2455 | +1.654 | −9.483 | +6.057 |
| mediator shifted by −6 | −0.0325 | +0.2865 | +0.2540 | +1.936 | +0.426 | +0.056 |
| mediator rescaled by 10 | −0.0083 | +0.1193 | +0.1110 | +0.368 | −5.169 | +63.376 |

Closed-form truth for that world: NDE +0.0000, NIE +0.2827, TE +0.2827.

- shift: NIE moves **0.0238** and NDE **0.0153**, against the 0.0100 tolerance — FAIL.
- rescale: NIE moves **0.1434** and TE **0.1345** — FAIL, and the rescaled fit reported
  **400 divergences after tuning**, so on that mediator unit the old priors did not merely
  bias the answer, they broke the sampler.

That is the "proven able to fail" record for both checks in this file.

---

## 3 · Attempts log (element 12 no-retry rule)

Every attempt of every check is listed with its number. The reported value of a gate is the
first run after the last code change.

| # | Check | Result | What changed before it |
|---|---|---|---|
| 1 | `tests/test_link_agreement.py` + `tests/test_scale_aware_priors.py` + `test_effects` + `test_closed_form` + `test_suppressor_sign` | **1 failed, 32 passed** (80.3 s). `test_the_gate_fails_when_the_conversion_drops_the_intercepts` failed: dropping *both* intercepts from the conversion moved the NIE by only 0.0160, not the > 0.2 the assertion demanded. Cause is the test's own generator, not the estimator: this world has `alpha0 + beta*mu_m = 0.2`, so the two intercepts nearly cancel and assuming both are zero is nearly harmless *here*. | first run of the new files |
| 2 | `tests/test_link_agreement.py` | **7 passed** (40.8 s) | the tripwire now drops the mediator baseline alone (the pre-2026-09-07 defect shape), and the test records why dropping both is not a tripwire in this world |
| 1 | Acceptance (a) and (c): `test_offset_null` + `test_rho_star_semantics` + `test_closed_form` + `test_frozen_guard` + `test_mechanism_battery` + `test_sensitivity` + the two new files + `test_effects` + `test_mediation` + `test_suppressor_sign` | **111 passed, 2 skipped** (89.4 s), exit 0 | the priors commit `ea5b1c6`, plus `--only-pymc` added to the battery |
| 1 | Acceptance (b): the battery's PyMC subset, same seeds and datasets | **PASS**, exit 0, table below | same tree |
| 1 | `tests/test_mechanism_battery.py` with the four part (iv) families | **1 failed, 34 passed** (22.2 s): the new missingness test called the closed form with `rho` omitted from its positional arguments and got `ValueError: |rho| must be <= 0.95` | the part (iv) families |
| 2 | `tests/test_mechanism_battery.py` | **35 passed** (25.2 s) | the missing `rho` argument in that test |

---

## 4 · Acceptance gates

### (a) The offset-null gate, and (c) the closed form and rho\* semantics

One command, `pytest` over ten targeted files, first run after the last code change:
**111 passed, 2 skipped in 89.4 s, exit 0.** Per file: `test_offset_null` 11/11,
`test_rho_star_semantics` 19/19, `test_closed_form` 11/11, `test_frozen_guard` 5/5,
`test_mechanism_battery` 26/26 (before the part (iv) families were added),
`test_sensitivity` 14 passed 1 skipped, `test_scale_aware_priors` 5/5,
`test_link_agreement` 7/7, `test_effects` 5/5, `test_mediation` 3 passed 1 skipped,
`test_suppressor_sign` 5/5. The count-offset and Gaussian-offset nulls at seed 731,
n 10,000, and the model-implied total effect against the randomized arm difference, are
inside `test_offset_null`; the four frozen files are byte-identical to main.

### (b) The battery's PyMC subset, same seeds and the same datasets

`python experiments/mechanism_battery.py --only-pymc --out experiments/results/mechanism_battery_pymc_gate`,
which reuses `pymc_indices` and `dataset_seed`, so every dataset is the one the earlier run
fitted. 20 datasets per family at n = 350, 140 datasets, 4 chains of 1,000 draws after 1,000
tuning steps each. Exit 0.

**Criterion, from the W4c brief and therefore fixed before the run:** the NIE coverage
Clopper-Pearson interval must contain 0.95 in families 4 and 7, and coverage must not fall in
any family.

| family | NIE coverage before | after | Clopper-Pearson after | NIE bias before | after | mean posterior NIE width after |
|---|---:|---:|---|---:|---:|---:|
| f1_no_cue_effect | 20/20 | 20/20 | [0.832, 1.000] | +0.0002 | +0.0004 | 0.0165 |
| f2_direct_bypass | 19/20 | 19/20 | [0.751, 0.999] | −0.0007 | −0.0027 | 0.0971 |
| f3_shared_cause | 0/20 | 0/20 | [0.000, 0.168] | +0.2513 | +0.2513 | 0.1253 |
| **f4_rationalization** | **7/20** | **20/20** | **[0.832, 1.000]** | **−0.0790** | **−0.0092** | 0.1362 |
| f5_redundant_explanation | 0/20 | 0/20 | [0.000, 0.168] | +0.2397 | +0.2405 | 0.1287 |
| f6_answer_copying | 20/20 | 20/20 | [0.832, 1.000] | +0.0001 | +0.0001 | 0.0227 |
| **f7_opposing_effects** | **12/20** | **19/20** | **[0.751, 0.999]** | **−0.0473** | **−0.0078** | 0.1157 |

**PASS.** Families 4 and 7 both reach an interval containing 0.95, and no family's coverage
count is lower than it was: the five other families are unchanged to the dataset. Maximum
r_hat is 1.0000 across all 140 fits (it was 1.0100 before) and there are zero divergences.

Families 3 and 5 stay at 0/20 by construction and that is the correct behaviour, not a
residual defect: both worlds have a shared hidden cause, the estimator assumes rho = 0, and
the battery's whole point is that a confounded world is reported confidently wrong at rho = 0.
Their bias is unchanged to four decimals, which is the check that the prior change did not
quietly move a number it had no business moving.

The before column is `experiments/results/mechanism_battery/pymc_subset.json` at `5840afc`;
the after column is `experiments/results/mechanism_battery_pymc_gate/pymc_subset.json`.


---

## 5 · Element 12 part (iv): the misspecification list

Section 13 of `experiments/PREREGISTRATION_jury_and_scale.md` (lines 857 to 906) names eight
misspecifications the offset-null family has to cover. Four were already carried by the
original seven battery families; the other four had no generator anywhere in the repository
and were written for this branch. The table is regenerated inside `report.md` on every run,
with the line numbers read from the source rather than typed in, so it cannot go stale
silently.

| list item | family | generator, file and line |
| --- | --- | --- |
| baseline offsets | `f1_no_cue_effect`, `f4_rationalization`, `f6_answer_copying`, `f7_opposing_effects` | `NoCueEffect` (mechanism_battery.py:206), `Rationalization` (mechanism_battery.py:270), `AnswerCopying` (mechanism_battery.py:322), `OpposingEffects` (mechanism_battery.py:342) |
| nonlinear depth response | `f8_nonlinear_depth` | `NonlinearDepthResponse` (mechanism_battery.py:379) |
| varying variance | `f9_varying_variance` | `VaryingVariance` (mechanism_battery.py:407) |
| correlated errors | `f3_shared_cause`, `f5_redundant_explanation` | `SharedCause` (mechanism_battery.py:250), `RedundantExplanation` (mechanism_battery.py:292) |
| sparse groups | `f10_sparse_groups` | `SparseGroups` (mechanism_battery.py:440) |
| treatment-induced latent states | `f5_redundant_explanation`, `f6_answer_copying` | `RedundantExplanation` (mechanism_battery.py:292), `AnswerCopying` (mechanism_battery.py:322) |
| missingness | `f11_mediator_missingness` | `MediatorMissingness` (mechanism_battery.py:477) |
| near-zero and cancelling effects | `f7_opposing_effects`, `f2_direct_bypass_alpha3`, `f1_no_cue_effect` | `OpposingEffects` (mechanism_battery.py:342), `DirectBypass` (mechanism_battery.py:225), `NoCueEffect` (mechanism_battery.py:206) |

The four new families are each family 4's world with exactly one thing changed, so the
comparison against family 4 at the same sample size isolates the misspecification rather than
mixing it with a different effect size:

- **f8_nonlinear_depth** replaces the linear depth response with a saturating one,
  `kappa*tanh((M - baseline)/tau)` at `kappa = 1.5`, `tau = 1`, against an estimator whose
  outcome equation is linear in the mediator. Its truth is computed twice, by 200-node
  Gauss-Hermite quadrature and by the battery's own 2,000,000-row Monte Carlo, which share no
  code.
- **f9_varying_variance** has the cue widen the mediator from sd 0.6 to sd 1.8 as well as
  shifting it, while the model fits one `sigma_m` for both arms. Truth is exact.
- **f10_sparse_groups** puts an item-level effect (sd 0.8) in the answer only, five rows per
  item, so 70 items at n = 350. The mediator equation and the marginal outcome model are both
  still correctly specified, so the point estimate stays consistent and the casualty is the
  row bootstrap's independence assumption. Truth is exact.
- **f11_mediator_missingness** analyses complete cases where the probability that a trace
  parses falls with its length, `Phi(1.2 - 0.6*(M - baseline))`. The truth stays the full
  population's, because the loss is a property of the measurement and not of the world.

Each runs 400 seeded datasets at n = 350 for the maximum-likelihood path, the pre-registered
minimum, plus the standard 20-dataset PyMC subset. Their results are in the run below and in
`experiments/results/mechanism_battery/report.md`.

### Every new check, shown failing once

| check | how it was made to fail | what it printed |
|---|---|---|
| `test_posterior_natural_effects_agree_with_the_closed_form` | the companion test in the same file converts the same draws with the mediator baseline dropped | NIE off by more than 0.2 against a tolerance of 0.02; and the file's own attempt 1 failed for a different reason, recorded above |
| `test_effects_are_invariant_to_shifting_the_mediator` | run against `fit_mediation_model` from `5840afc` | NIE moved 0.0238, NDE 0.0153, tolerance 0.0100 |
| `test_effects_are_invariant_to_rescaling_the_mediator` | run against `fit_mediation_model` from `5840afc` | NIE moved 0.1434, TE 0.1345, and the fit reported 400 divergences after tuning |
| `test_every_misspecification_on_the_list_has_a_generator` | one entry pointed at a class name that does not exist | `AssertionError: missingness names an unknown class MissingnessFamily` |
| `test_missingness_family_drops_the_long_traces_and_keeps_the_full_truth` | `retention` replaced by a function returning 1 | `len(x) = 4000 == n = 4000`, so `len(x) < n` is False. As written it keeps 0.765 of 4,000 rows with a surviving mean mediator of 6.307 against a population mean of 6.600 |
| `test_sparse_groups_share_one_item_effect_across_each_block_of_rows` | per-row noise instead of a per-item effect | within-item sd 1.147 against the required 0. As written the within-item sd is 0.000000 and the between-item sd is 0.630 |
