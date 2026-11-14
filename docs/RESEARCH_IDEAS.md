# Research ideas: evaluation against the current design (10 Jul 2026)

Five ideas Maksim raised, evaluated against the actual pipeline (checked in code, not from
memory). The shared verdict, up front: ideas 1, 3 and 4 are complementary measurement
channels or uncertainty signals. None of them substitutes for the intervention-based
causal-mediation design, which is this project's differentiator, but each can be validated
against the golden set and added to the hierarchical model. Idea 2 is a baseline
comparator for the paper. Idea 5 is already the Phase-2 plan.

Current design facts these evaluations rest on: audits run at temperature 0.0, single sample
per condition; outcome Y is binary (answer letter match); stability is measured by a *neutral
prompt-edit* control (`neutral_changed`, `require_stable`/`n_stable`), not by resampling;
the auditor verdict is a regex heuristic (`acknowledges_hint`); token logprobs are not used
anywhere yet; models so far: Groq free-tier Llama-3.1-8B-instant / Llama-3.3-70B-versatile
and local Ollama llama3.2:3b / llama3.1:8b.

## 1. Token logprobs in the CIs: yes for power, no as a faithfulness measure

Two different uses; keep them separate.

As the outcome variable, this is high value. Replacing the binary "followed the hint"
outcome with a continuous one (log-probability or logit margin of the hinted answer token
under each condition) carries far more information per trial, so posteriors tighten at the
same n. This is exactly what the planned Phase-2 GPU logit-forcing does for open models,
and the idea extends it to any API that exposes logprobs. Caveats: Anthropic's API does not
return logprobs; OpenAI does (top_logprobs); Groq's support is limited or absent. Verify
per provider before designing around it. The GLMM changes from Bernoulli-logit to a
Gaussian (or heavy-tailed) likelihood on the margin, and the mediation estimands are
unchanged.

As intrinsic confidence ("min/avg token prob = confidence") it is a useful covariate, but
it measures *confidence*, not *faithfulness*. A model can be highly confident in an answer
the CoT did not cause, and silent hint-following often looks confident. Use the logprob
margin to stratify (does unfaithfulness concentrate in low-margin items? that connects
directly to the frozen "right-but-uncertain items" pre-registration), never as the
faithfulness number.

The credible intervals themselves keep coming from the posterior over NDE/NIE. Logprobs
change the *likelihood*, not the inference machinery.

Effort: small once a logprob-exposing backend is in the client layer. Do it with the Phase-2
GPU work rather than before the golden set.

## 2. RAGAS faithfulness: a different construct, so use it as a baseline

RAGAS "faithfulness" is the fraction of answer claims entailed by the retrieved context,
scored by an LLM judge (statement decomposition, then NLI-style verification). That is
consistency or groundedness, which is correlational rather than causal. The distinction is
precisely this project's thesis: a model that silently follows a planted hint while writing
plausible reasoning is RAGAS-faithful (the reasoning supports the answer) yet causally
unfaithful. So:

- Do NOT adopt it as the measurement. It cannot detect the failure mode the auditor catches.
- DO consider it (or a RAGAS-style consistency score) as a baseline column in the Phase-2
  benchmark. Showing where consistency metrics and causal-mediation estimates diverge is a
  strong empirical argument for the paper: consistency is not causation, measured.
- Their statement-decomposition prompts are worth borrowing if and when CoT is segmented
  into multiple mediator segments.

Effort: pip-installable, but it burns LLM-judge API calls, so it is a small budget line in
Phase 2.

## 3. Self-consistency and semantic entropy: partly in place, and the sampling version is a good add

What exists already is *perturbation* consistency: every item gets a neutral prompt-edit
control, and headline rates are recomputed on the stable subset (`stable_follow_rate`). What
does NOT exist is *sampling* consistency, since everything runs at temperature 0.0 with a
single sample.

- Adding k samples at temp > 0 per condition gives an answer distribution per item. For MCQ,
  Farquhar et al.'s semantic entropy reduces to the entropy of the answer letters (meaning
  clusters = letters), so it is nearly free to compute once samples exist.
- Value: (a) a principled per-item uncertainty signal for the "right-but-uncertain"
  pre-registration; (b) seed-level variance feeds the hierarchical model rather than being
  invisible; (c) stratifying faithfulness by answer entropy is a novel, cheap analysis
  (is decorative CoT more common when the model is internally uncertain?).
- Cost: multiplies inference calls by k, so size it in the powered-sweep budget. Note the
  estimand shifts from "the deterministic decode" to "the decode distribution". That is a
  feature, since it is what deployment samples from, but say it explicitly in the
  pre-registration.

## 4. Cross-encoder and different-family judges: yes, and the golden set is the validator

The auditor's two judgments map to the golden-set questions: Q1 "mentions the hint"
(currently a regex) and Q2 "reasoning supports the answer" (currently unmeasured by machine).
Both can be scored by machines of increasing cost:

- A small local NLI cross-encoder (a DeBERTa-class model, say) scoring "reasoning entails
  answer", which is roughly Q2. Runs on the M2 Air with base-size models, one at a time.
- A different-family LLM judge (GPT- or Gemini-class judging Llama transcripts) for Q1
  and Q2. The instinct to avoid same-family judges is sound, since self-preference bias is
  documented in the LLM-as-judge literature, and the audited model's weights should not
  grade themselves.
- The human golden set then validates ALL judges (regex, cross-encoder, LLM judge) against
  the human majority, with error rates per judge. This turns the golden set from "validate
  one regex" into a judge benchmark, a straight upgrade to the workshop paper, at the cost
  of some judge API calls.
- Honesty caveat: any judged "reasoning supports answer" score is still consistency, not
  causation (see idea 2). Judges become *validated instruments inside* the causal design,
  the mediator-acknowledgment classifier being the example, rather than replacements for
  interventions.

Sequencing: label the golden set FIRST (unchanged, two humans), then run judges against it.
Zero rework, pre-registration untouched.

## 5. Which LLMs, and benchmarking them: this IS Phase 2

- Measured so far (delivered, free-tier, text-level): Llama-3.1-8B-instant (n=103 headline
  run: 36.9% hint-following, mostly silent), Llama-3.3-70B-versatile (n=11), local
  llama3.2:3b / llama3.1:8b sanity runs.
- Phase-2 plan (the funding ask): Llama-3-8B + Gemma-2-9B with GPU logit-level forcing,
  Claude-family via API (text-level), GPT-class cross-lab check. Faithfulness posteriors
  with ρ* robustness for every model × task cell, released publicly. "Benchmark them to have
  a comparison" is verbatim the deliverable, and the grant applications are sized for it.
- One discipline to keep: cross-model comparisons must state the intervention level per
  model (logit-forcing for open weights versus text-level for APIs). The two are not the
  same estimand, and mixing them in one league table without saying so would be
  overclaiming.

## Suggested ordering (respects the pre-registration and the budget)

1. Golden set labeling (unchanged, since it validates everything else).
2. Judge panel against the golden set (regex + cross-encoder + cross-family LLM judge),
   from idea 4.
3. Powered sweeps with k-sample semantic-entropy stratification, from idea 3.
4. Logprob outcomes wherever the backend exposes them (GPU runs, OpenAI), from idea 1.
5. RAGAS-style consistency baseline column in the benchmark, from idea 2.

## Additions from the literature sweep (17 Jul 2026)

From the July 2026 literature sweep (see "AI Safety Project/research/FAITHFULNESS_LANDSCAPE_2026-07.md"
for links and the full incorporate list). These extend the five evaluated ideas; nothing below
changes the frozen pre-registrations.

- Idea 1 (logprob outcomes): published precedent exists for probability-weighted counterfactual
  scoring (Siegel et al., Correlational Counterfactual Test, arXiv:2404.03189), so the graded-Y
  upgrade can cite prior art. Caveat found: masked-KL step scores have a documented low-entropy
  failure mode (arXiv:2605.24286); check before adopting any KL variant.
- Idea 2 (RAGAS baseline): add CC-SHAP (arXiv:2311.07466) and counterfactual simulatability
  (arXiv:2307.08678) as further consistency-family baseline columns. The strongest steelman to
  cite and then rebut: self-explanations do help predict behavior (NSG, arXiv:2602.02639).
- Idea 3 (semantic entropy): prefix-conditioned sampling gives confidence strata without logprob
  access (arXiv:2606.03969); accuracy-faithfulness decoupling under self-consistency justifies
  keeping T=0 audits while logging a majority-vote arm (arXiv:2601.06423).
- Idea 4 (cross-family judges + golden set): before spending human labels, gate every judge on
  constructed ground truth (planted-error / deleted-step items; cf. C2-Faith arXiv:2603.05167).
  Report Gwet AC2 + a saturation flag BESIDE the frozen Cohen's kappa (kappa stays primary;
  degeneracy under prevalence skew is documented in cot-suite). Stratify the labeling sheet to
  include hard negatives (acknowledged-not-followed, followed-with-disclosure, followed-silently).
  LASR ships a golden-set-validated judge prompt to start from (CoT-Unfaithfulness-Team-Noah.pdf).
- Idea 5 (Phase-2 models): training regime, not scale, drives cue susceptibility
  (editorial-faithfulness-bracis); cheap contrasts: GPT-OSS 20B vs 120B same-family scale pair,
  reasoning-vs-base within family (arXiv:2501.08156), OLMo checkpoint lineage over post-training.
  Keep stating the intervention level per model.

New top-tier candidates from the sweep (full details and sources in the sweep file):
truncation dose-response mediator curves; commitment-point measurement (pre-CoT forced answer now,
pre-CoT probe as measured confounder proxy on GPU later); CoT transplant as a nonparametric NIE
check; a replay-drift floor arm for every mediator intervention; channel-resolved acknowledgment
reporting on reasoning models (frozen regex unchanged, run per channel); relaxed identification
conditions for the rho sweep (arXiv:2604.01501); clue-need / CoT-uplift covariates in the
hierarchical model.

Seven items were found that would CONFLICT with the frozen pre-registrations if adopted as
replacements (graded acknowledgment classes, uplift-based item selection, budget-dependent silent
share as headline, paraphrase-null replacing the neutral-edit threshold, on-policy resampling
replacing the neutral-edit control, AC2 replacing kappa, CoT-describes-action as the golden-set
label). Each has a permitted additive form; see section 6 of the sweep file.
