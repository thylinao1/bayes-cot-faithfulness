# Phase-2 design notes: identification, transplant check, signed effects

Written 2026-07-17 from the July 2026 landscape sweep (incorporate list items
T11, T3, A10). Nothing here changes the frozen pre-registrations; these notes
specify additive Phase-2 analyses and record properties the current estimator
already has. Sources are cited by arXiv ID; the full source list lives in the
project's landscape file.

## 1. What the randomized cue buys the rho sweep (T11)

The sensitivity analysis in `src/bayes_cot_faithfulness/sensitivity.py` prices
the sequential-ignorability assumption with a single residual correlation
rho = Corr(eps_M, eps_Y). Sequential ignorability is really two conditions
(Imai, Keele, Yamamoto 2010), and they do not stand or fall together in this
design:

1. **Treatment ignorability: X independent of the potential outcomes and
   potential mediators.** In observational mediation studies this is an
   assumption. Here it holds by construction: the cue X (clean vs hinted arm)
   is assigned by the experiment protocol, never by the model, and the planted
   wrong option is cycled across letters by item index. There is no mechanism
   by which an unmeasured property of the item or the model can influence
   which arm an item receives.

2. **Mediator ignorability: no unmeasured confounding of the M to Y relation
   given X.** This is structurally violated for an autoregressive model (the
   same hidden state generates both the reasoning and the answer), which is
   exactly what the rho sweep, the breakdown frontier rho*, and the
   partial-identification bounds price.

Recent work on direct effects under unmeasured confounding (arXiv:2604.01501)
develops identification under conditions weaker than full sequential
ignorability, several of which are checkable when the treatment is randomized.
The adaptation to this design:

- Because condition 1 is guaranteed by protocol, the rho parameter carries the
  entire identification burden alone. The sweep does not need to absorb any
  X-side confounding, so the reported rho* is interpretable purely as
  mediator-outcome confounding tolerance. This should be stated in the
  Phase-2 pre-registration's assumptions section, because it is a strictly
  stronger position than the generic "we assume the rest" framing most
  mediation write-ups use.
- Two checks become testable from data rather than assumed, and Phase 2
  should report both per run: (a) arm balance on item covariates (guaranteed
  in expectation by randomization; verify realized balance), and (b) mean
  independence of the mediator-equation residuals from X (a failing check
  indicates misspecification of the M model, not confounding, and localizes
  the repair).
- Candidate refinement, not implemented: with X randomized, the bounds in
  `partial_identification_bounds` may tighten further by exploiting the
  X-side restrictions. This is a math task with no run cost; it belongs in
  Phase 2 alongside the pre-CoT commitment measurement, which anchors part
  of the unobserved confounder and turns the pure sweep into a partially
  informed one.

Status: documentation and pre-registration language only. No estimator change
was made, and no new bound is claimed.

## 2. The CoT transplant check on the indirect effect (T3)

The NIE posterior says how much of the answer runs through the reasoning text.
A model-free triangulation of that claim: transplant the chain of thought
across arms on the same item and measure answer carry-over.

Design, on the existing planted-hint items:

- **Forward transplant.** Take the CoT generated under the hinted prompt,
  present it through the forced-answer continuation frame (which shows the
  question and the reasoning but not the cue), and record whether the forced
  answer reproduces the hinted-run answer.
- **Reverse transplant.** Same with the clean-run CoT: the forced answer
  should reproduce the clean answer, not the hinted one.
- **Floor.** Every transplant rate is read against the replay-drift floor
  (arm T4): the same model re-fed its own unedited CoT through the identical
  machinery. Drift there is pure teacher-forcing artifact, so a transplant
  effect is only real to the extent it exceeds the floor.

Interpretation, fixed before any run:

| Posterior NIE | Forward carry-over vs floor | Reading |
|---|---|---|
| High (text carries the effect) | Clearly above floor | Triangulated: the text is the carrier |
| High | Near floor | Tension: the estimator attributes the effect to text the transplant cannot reproduce; suspect latent-state confounding, check the rho* margin |
| Low (decorative) | Near floor | Consistent: decorative text does not transport the answer |
| Low | Clearly above floor | Tension in the other direction: the text transports more than the model credits; check the M model specification |

Caveats stated up front: the continuation frame strips the cue, so carry-over
measures text-borne influence only; transplantation is off-policy by
construction and shares the teacher-forcing caveat with every edit-based
intervention (the strongest published critique of this family is the
Thought Branches result, arXiv:2510.27484, and the on-policy resampled arm
U6 is the eventual answer to it); and a carry-over rate is a rate, so it is
reported with the same guardrails (n, upper bound, minimum detectable rate)
as every other rate in this project.

Implementation: the pure pieces (`continuation_prompt`, replay and transplant
prompt builders) are shared with the additive-arms runner; the transplant arm
runs as `--arm transplant` there. No frozen arm is touched.

## 3. Signed suppressor effects are representable (A10)

FUR (parametric faithfulness, Tutek et al.) reports reasoning steps with
negative, suppressor-style contributions: unlearning a step can make the
hinted answer more likely. An estimator that constrained the mediated path to
one sign would mis-read such a world instead of reporting it.

Audit result for this package, checked in code on 2026-07-17:

- The hierarchical priors on the mediated slope are Normal with mean zero
  (`hierarchical.py`: mu_beta Normal(0, 2), beta_g Normal(mu_beta, tau_beta)),
  so no sign constraint enters through the prior.
- The probit MLE used by the sensitivity sweep is unconstrained in beta, and
  the natural-effects integrators pass negative mediated paths through on the
  probability scale.
- The breakdown frontier anchors on the sign of the rho = 0 estimate and
  searches for a flip in either direction, so a negative NIE gets the same
  robustness treatment as a positive one.

The property is pinned by `tests/test_suppressor_sign.py` end to end
(ground-truth integrator, probit fit, breakdown frontier, no-pooling slopes,
posterior converter). If a future per-step mediation model is built, its
priors must inherit this sign freedom, and that requirement belongs in the
Phase-2 pre-registration text.
