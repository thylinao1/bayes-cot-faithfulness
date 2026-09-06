# Estimator repair: baseline intercepts, rho\* semantics, and an offset-null gate

**Date:** 7 September 2026
**Branch:** `fix/estimator-intercepts`
**Trigger:** the independent critical review of 6 September 2026, sections 3.3 and 3.4
**Status:** code repaired and gated by tests; public numbers **not yet** recomputed or relabelled (the plan is in section 4 below)

---

## 1 · What was wrong

### 1.1 The probit mediation model had no baseline intercepts

The pre-repair structural model, as stated in `src/bayes_cot_faithfulness/sensitivity.py` (and repeated verbatim in `docs/methodology.md` section 3), was

```
M = gamma * X + eps_M
Y = 1[alpha * X + beta * M + eps_Y > 0]
```

with the fit implemented over exactly four free parameters:

```python
alpha, beta, gamma, log_sigma_m = params
...
resid_m = M - gamma * X
```

That is not a modelling convenience. It is two substantive claims that the data was never asked about:

1. **`E[M | X=0] = 0`.** The mediator is forced to have a control-arm mean of exactly zero. Any real mediator level in the clean arm has nowhere to go except into `gamma`, where it is read as a *treatment shift*.
2. **A clean-arm outcome baseline of `Phi(0) = 0.5`.** With no `alpha0`, the integrated counterfactual `P(Y(0, M(0)) = 1)` is pinned at one half.

Both are false for the mediators this project actually uses. `experiments/05_realmodel_control.py::_build_design` builds the real-model design as

```python
    if mediator == "steps":
        for r in correct:
            ...
            M.append(len(split_steps(r["clean_cot"])))
```

a raw CoT **step count**, which on the stored Llama-3.1-8B run averages 6.14 steps in the clean arm. Nothing centres it before it reaches the estimator.

### 1.2 What that does: a mediated effect on a world with no causal effect

The review executed a synthetic null against the public functions: `X` balanced, `M` a positive count independent of `X`, `Y` an independent coin at p = 0.8. The population NDE, NIE and TE are all exactly zero. Reproduced here at seed 731, n = 10,000:

| Null design | true NIE | old NIE | true TE | old TE | observed arm difference in Y |
|---|---:|---:|---:|---:|---:|
| Poisson count mediator | 0 | **+0.2131** | 0 | **+0.2697** | −0.0082 |
| Gaussian offset mediator | 0 | **+0.2213** | 0 | **+0.2678** | −0.0210 |

The old model's `gamma` came out at 6.06 and 5.96 on the two designs: it had absorbed the mediator's baseline level. The Gaussian replication matters because it rules out the obvious objection that the failure was only about handing counts to a Gaussian likelihood.

Repaired, on identical data:

| Null design | new NIE | new TE | observed arm difference |
|---|---:|---:|---:|
| Poisson count mediator | −0.0002 | −0.0082 | −0.0082 |
| Gaussian offset mediator | +0.0000 | −0.0210 | −0.0210 |

The model-implied total effect now tracks the randomized arm difference instead of inventing one, and the mediated path is zero to three decimal places.

This is a model/application failure under baseline offsets, **not** an algebra error in the closed-form integrator. `closed_form.probit_natural_effects_closed_form` was and is exact for the SCM it describes. The restrictive equations work on data generated to match them, which is why the project's own synthetic recovery gate passed while the estimator was unusable on unadjusted baseline levels.

### 1.3 rho\* was being read as something it is not

`breakdown_frontier` reports `rho*`, the smallest residual M-Y correlation that flips the sign of a natural effect. The plan's monotonic ladder treats it as a hidden-trigger severity scale. It cannot be one.

With `gamma != 0` the NIE vanishes exactly when the structural mediator coefficient does, and the reparameterisation gives `beta(rho) = B*sqrt(1-rho^2) - rho/sigma_m`, so

```
rho* = |B| * sigma_m / sqrt(1 + B^2 * sigma_m^2)
```

which contains neither `alpha` nor either intercept. Holding `beta = 0.8`, `gamma = sigma_m = 1` and raising only the direct bypass `alpha`, the mediated share NIE/TE falls from 100% to 1.6% while `rho*` sits at 0.624695 throughout. On 20,000-row samples the repository's own `breakdown_frontier` returns 0.625 at `alpha = 0` and 0.630 at `alpha = 3`.

A large `rho*` is also compatible with **zero** true mediation: for the pure shared-cause process `M = X + U`, `Y = 1[X + U + E > 0]` there is no M-to-Y arrow at all, yet the compatible zero crossing is about 0.707.

### 1.4 Two smaller defects

- The fitter was named `fit_probit_mediation_map` but is a plain MLE, and it returned `result.x` **without ever checking `result.success`**. An optimizer failure was indistinguishable from a converged fit.
- When there was no supported effect at `rho = 0`, the frontier found no crossing on either side and returned `survives_full_range=True` with the maximum robustness in range: the strongest possible verdict attached to the weakest possible evidence.

---

## 2 · What changed

Everything is additive and the old behaviour is reachable under an explicit flag, so historical numbers stay reproducible from this repository.

### `src/bayes_cot_faithfulness/sensitivity.py`

| Change | Detail |
|---|---|
| Intercepts in the SCM | `M = mu_m + gamma*X + eps_M`, `Y = 1[alpha0 + alpha*X + beta*M + eps_Y > 0]` |
| `_joint_negloglik` | dispatches on `len(params)`: 4 entries is the legacy branch, kept byte-identical; 6 entries is the repaired model |
| `fit_probit_mediation_map` | gains `intercepts: bool = True` and `strict: bool = False`; returns a `ProbitMediationFit` NamedTuple `(alpha, beta, gamma, sigma_m, mu_m, alpha0, converged)` whose first four fields are the old return value in order |
| Optimizer check | `result.success` is now checked: `OptimizerWarning` by default, `OptimizerFailure` raised under `strict=True`, and `fit.converged` records it either way |
| Starting point | `_init_with_intercepts` starts `mu_m` at the control-arm mean of M and `alpha0` at `Phi^-1(mean(Y))`, so a step count in the tens does not leave L-BFGS-B climbing out of a saturated probit |
| `probit_natural_effects` | keyword-only `mu_m=0.0, alpha0=0.0`; every existing positional call keeps its meaning |
| `natural_effects_from_fit` | new; the single place a fit is converted to effects, so no call site can silently drop the intercepts again |
| `sensitivity_sweep`, `breakdown_frontier`, `partial_identification_bounds` | all thread `intercepts` through; `SensitivityPoint` gains `mu_m` and `alpha0` (default 0.0) |
| `BreakdownFrontier` | gains `unresolved: bool = False` |
| `breakdown_frontier` | gains `min_effect: float = 0.0`; when `abs(effect_at_zero) <= min_effect` it returns `unresolved=True`, both crossings `None`, `robustness=NaN`, `survives_full_range=False` rather than a frontier. Its docstring now states in the public API that `rho*` is the robustness of the M-to-Y coefficient's sign, **not** a measure of direct bypass |
| Module docstring | carries the full rho reparameterisation with intercepts, including `alpha0(rho) = A0*sqrt(1-rho^2) + (rho/sigma_m)*mu_m` |

### Other modules

| File | Change |
|---|---|
| `closed_form.py` | `probit_natural_effects_closed_form` gains `mu_m=0.0, alpha0=0.0`; the intercepts enter only through the shared offset `alpha0 + beta*mu_m`, so `var` is unchanged and the derivation in the docstring was extended rather than replaced |
| `synthetic.py` | `SyntheticCoTConfig` gains `mu_m`, `alpha0` (default 0.0, bit-identical draws) |
| `effects.py` | `monte_carlo_true_effects` reads the config intercepts; `posterior_natural_effects` gains `mu_m_samples`, `alpha0_samples` |
| `mediation.py` | `fit_mediation_model` gains `intercepts: bool = True`, adding PyMC variables `mu_m` (prior centred on the observed mediator mean, scaled by its sd) and `alpha0`; new `extract_intercept_samples` returns zeros for a legacy trace |
| `hierarchical.py` | `build_hierarchical_model` / `fit_hierarchical_mediation` gain `intercepts: bool = True`, adding a population outcome intercept `mu_0` and a mediator intercept `mu_m`. The non-centred parameterisation of `alpha_g / beta_g / gamma_g` is untouched. `HierarchicalCoTConfig` gains `mu_m`, `mu_0` |
| `positive_control.py` | uses `natural_effects_from_fit`; `smoke_test` now also reports `unresolved` |
| `notebooks/01_synthetic_validation.py` | the gate is now two-part: recovery **and** a new offset-null run `[5/5]` that requires near-zero recovered effects and prints the legacy contrast |
| `notebooks/03`, `notebooks/04` | updated to the new fit object; both still pass their own correctness gates |

Not touched, deliberately: `experiments/PREREGISTRATION*.md` and `src/bayes_cot_faithfulness/interventions.py`.

### New tests

`tests/test_offset_null.py` (11 tests)

- the reviewer's two nulls at seed 731, n = 10,000, asserting `abs(NIE) < 0.01` and `abs(TE - observed arm difference) < 0.02` with `intercepts=True`;
- the same nulls asserting the **old** answer (NIE 0.21308 / 0.22130) under `intercepts=False`, so the size of the defect is a fact the repository can be queried for;
- `test_legacy_fit_is_bit_for_bit_reproducible`, pinning the pre-repair coefficients as hex float literals recorded from `origin/main` at commit `5c59eba`;
- the model-DGP recovery check, still passing with two free intercepts on a process whose true intercepts are zero;
- a misspecification pair on a mediator with an arm-specific baseline (`mu_m = 6`, `gamma = 1`): the repaired fit recovers both, the legacy fit puts `6 + 1 = 7` into `gamma` and mis-splits the total effect.

`tests/test_rho_star_semantics.py` (19 tests)

- `rho*` agrees within 0.01 between `alpha = 0` and `alpha = 3` at `beta = 0.8, gamma = 1, sigma_m = 1`, while the effect it is often confused with moves by more than an order of magnitude;
- `rho*` equals the analytic crossing of the fitted `beta`;
- the intercept-carrying reparameterisation reproduces a direct refit at three values of `rho`, and gives the same closed-form effects;
- the total effect is invariant to the assumed `rho` (only the NDE/NIE split moves);
- an unsupported effect at `rho = 0` returns `unresolved=True` with NaN robustness;
- a tripwire on the `breakdown_frontier` docstring, so the semantics cannot be deleted quietly.

### Test and lint counts

| Check | Result |
|---|---|
| `pytest -q` (fast) | 576 passed, 5 skipped |
| `pytest -q --runslow` | 581 passed |
| `ruff check .` | 73 findings, all pre-existing style debt inherited from `origin/main` (74 there); the two new test modules and the rewritten `sensitivity.py` are clean |

---

## 3 · Which public numbers depend on the old specification

### 3.1 The real-model rho\* values, 0.708 to 0.800

`docs/site/index.html` publishes four `rho*` values in the run table (lines 474, 481, 488, 495) and summarises them at line 522:

> Cross-run &rho;\* agreement (0.708 to 0.800) is suggestive, not test-retest reliability.

Every one of them was produced by `experiments/05_realmodel_control.py` feeding **raw CoT step counts** into `breakdown_frontier` under the intercept-free model. They are therefore computed on the specification the null test breaks.

One of the four runs stores its transcripts (`experiments/results/control_transcripts_llama-3.1-8b-instant.json`, the "22 of 30" row), so its design can be rebuilt offline and refitted both ways. Doing so:

| Quantity | Old specification | Repaired specification |
|---|---:|---:|
| mediator baseline `mu_m` | forced to 0 | 6.136 (the clean-arm mean step count) |
| `gamma` | 7.864 | 1.727 |
| `beta` | +0.246 | −0.224 |
| NIE at rho = 0 | **+0.415** | **−0.137** |
| model-implied TE | +0.398 | −0.452 |
| observed arm difference in Y | −0.455 | −0.455 |
| `rho*` | **0.750** (matches the published 0.7499903) | **0.327** |

The old fit reproduces the published number exactly, which confirms the reconstruction. Under the repair the mediated path does not merely shrink: **it changes sign**, and the model-implied total effect goes from contradicting the randomized arm difference to matching it.

Two honest caveats on that recomputation. It rests on 44 rows (22 items x 2 arms), and the clean arm has `Y = 1` for every row by construction, because the design keeps only clean-correct items. A mediation fit with no outcome variation in the control arm is not a scientific estimate of anything. The right reading is narrow and sufficient: **the published `rho*` values are specification-dependent, and the specification they were computed under is the broken one.** It is not a claim that −0.137 is the correct mediated effect.

The other three rows (0.708, 0.782, 0.800) have no stored transcripts, so they cannot be recomputed without rerunning the model calls.

### 3.2 The synthetic figures and the methodology numbers

`docs/methodology.md` section 3 states, and `figures/sensitivity_curve.png` plots, the synthetic sensitivity result: NIE 0.43 under assumed ignorability against a truth of 0.21, a bias of about 0.22, sign-stability across `rho` in `[-0.6, +0.7]`, and `rho* ~ 0.74`.

Those come from `ConfoundedCoTConfig`, whose true intercepts genuinely are zero, so the repair barely moves them. Re-running `notebooks/03` and `notebooks/04` on the repaired code gives NIE 0.4368, bias +0.2275, the same `[-0.60, +0.70]` band, and `rho* = 0.737`. The figures in `figures/` are **not** regenerated in this commit: regeneration and caption relabelling should land as one coordinated change across all three PNGs and the prose that quotes them, not piecemeal.

`figures/positive_control.png` and `figures/posterior_recovery.png` are in the same position. `tests/test_positive_control.py` passes unchanged under the repaired default, so the faithful-versus-decorative separation the demonstration rests on survives.

### 3.3 What is unaffected

- Everything downstream of `interventions.py`: follow rates, silent-unfaithful rates, neutral-control rates, attrition, the guardrail intervals, the parser audit. These never touch the mediation estimator.
- The frozen pre-registrations and the A9 specificity holdout; `tests/test_frozen_guard.py` passes untouched.
- The closed-form integrator's derivation, which was correct for the model it described.

---

## 4 · Recompute and relabel plan

1. **Label before recomputing.** Every published `rho*` on the site gets an explicit specification tag now: *computed under the pre-2026-09-07 intercept-free model*. A number that is known to be specification-dependent must not sit in a table unmarked while the recompute is pending.
2. **Withdraw the cross-run agreement claim.** "Cross-run `rho*` agreement (0.708 to 0.800) is suggestive" is a statement about four numbers from a broken specification. It should be removed rather than restated, because agreement among four values that all inherit the same artefact is not evidence of anything.
3. **Recompute the one recomputable run** from stored transcripts and publish it beside the old value with both specifications named, together with the caveat in section 3.1 about 44 rows and a degenerate control arm.
4. **Mark the other three rows unavailable**, not corrected. They require rerunning the model calls; until then the honest cell content is "not available under the repaired estimator", never a silently retained old value.
5. **Fix the design, not only the fit.** Two changes are needed before any real-model `rho*` is worth publishing again: an outcome that varies in the control arm (the clean-correct filter makes `Y` constant there), and a mediator whose meaning is fixed by the estimand contract rather than by whichever proxy was cheapest. The step count is a coarse proxy that the pre-registration itself treats as a covariate.
6. **Regenerate the three figures in one pass** with the repaired estimator, updating `docs/methodology.md` section 3's quoted values in the same commit.
7. **Report `rho*` only beside NDE, NIE, TE and their uncertainty**, and use the `unresolved` verdict wherever there is no supported effect at `rho = 0` to be robust about. Set `min_effect` to the smallest mediated effect worth defending rather than leaving it at the backward-compatible 0.0.

---

## 5 · Reproducing this

```bash
cd bayes-cot-faithfulness
PYTHONPATH=src .venv/bin/python -m pytest tests/test_offset_null.py tests/test_rho_star_semantics.py -q
PYTHONPATH=src .venv/bin/python notebooks/01_synthetic_validation.py
```

The null itself, in six lines:

```python
import numpy as np
from bayes_cot_faithfulness.sensitivity import fit_probit_mediation_map, natural_effects_from_fit

rng = np.random.default_rng(731)
n = 10_000
X = np.repeat([0, 1], n // 2)
M = rng.poisson(5.0, n).astype(float) + 1   # independent of X
Y = rng.binomial(1, 0.8, n)                 # independent of everything

print(natural_effects_from_fit(fit_probit_mediation_map(X, M, Y, 0.0), 0.0))
# repaired:   (-0.0080, -0.0002, -0.0082)   true: (0, 0, 0)

print(natural_effects_from_fit(
    fit_probit_mediation_map(X, M, Y, 0.0, intercepts=False), 0.0))
# pre-repair: (+0.0566, +0.2110, +0.2676)
# The default Monte Carlo draw is 200,000, so these sit a couple of thousandths
# from the exact closed-form values (+0.0566, +0.2131, +0.2697) quoted above.
```
