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

The next commit on this branch.
