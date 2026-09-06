# Template: Amendment A3 to PREREGISTRATION_jury_and_scale.md

This is a TEMPLATE, not a pre-registration. It is not fingerprinted in
`tests/test_frozen_guard.py` and it carries no frozen commitment of its own. Filling it in
produces Amendment A3, which is appended to the END of
`experiments/PREREGISTRATION_jury_and_scale.md` as a new dated section, with that file's
SHA-256 updated in `tests/test_frozen_guard.py` in the same commit.

A3 exists because six quantities are pre-registered in Amendment A2 as FORMULAS or as RULES and
can only be given VALUES after the Phase 1 skeleton has run. Registering the formula before the
data and the value after it is the point: the rule cannot be reverse-engineered from the number.

**A3 is required before any powered job.** A powered job that runs before A3 is committed is
out of pre-registration and its numbers are reported as exploratory. The Phase 1 done-when
includes the A3 commit.

---

## How to fill this in

1. Do not edit any existing text in `PREREGISTRATION_jury_and_scale.md`. Append.
2. Every value carries the artifact path it was read from and its denominator. A value without
   a denominator is not a value.
3. Where a required run did not happen, write NOT MEASURED and say what blocks it. Do not
   substitute an assumption for a measurement and do not leave the row silently absent.
4. Run `bash ~/intelligence-systems/shared-discoveries/human-tone/tone-check.sh` on the file
   before committing; hard violations must be zero.
5. Commit message names the amendment and the skeleton job ids the values came from.

---

## Amendment A3: post-skeleton values (added YYYY-MM-DD)

Reason. Amendment A2 registered six quantities as formulas or rules whose values depend on the
Phase 1 skeleton. This amendment supplies those values and changes nothing else. It is
prospective for every powered run that follows it.

Provenance. Skeleton job ids: `____`. Cluster results path: `____`. Mac mirror:
`experiments/results/____`. Backend and version: `____`. Every value below is read back from a
committed artifact named in its own row.

### A3.1 Positive-class enrichment fractions per stratum (A2 element 4)

Formula, unchanged: `e_s = min(1, 50 / (f_s x N_s))`, where `f_s` is the observed Q1-yes rate in
stratum `s`'s hinted-arm skeleton transcripts and `N_s` is the number of hinted rows that
stratum contributes.

| Stratum | f_s (Q1-yes rate) | numerator / denominator for f_s | N_s | e_s | artifact |
|---|---|---|---|---|---|
| Llama | | | | | |
| Qwen | | | | | |
| gpt-oss | | | | | |
| Gemma | | | | | |
| OLMo | | | | | |

Check to record: whether any `e_s` equals 1, which means the stratum cannot reach 50 positives
at the planned draw size and the frame for that stratum needs more hinted rows rather than more
enrichment.

### A3.2 Organism-minus-twin MDE (A2 element 11)

Formula, unchanged: `1.645 x sqrt(2) x sd_pilot(D)` at the ladder's chosen n, with `sd_pilot(D)`
computed across BOTH training seeds at the first two dose levels.

| D defined on | sd_pilot(D) | checkpoints in the sd | ladder n | MDE | artifact |
|---|---|---|---|---|---|
| NDE | | | | | |
| mediated share NIE/TE | | | | | |
| rho\* (descriptive, no directional prediction) | | | | | |

Record the number of checkpoints entering `sd_pilot(D)` explicitly. The formula requires both
training seeds at the first two dose levels; an sd computed from one seed per rung is the
superseded v1.3 quantity and understates the MDE by construction.

### A3.3 k stratum-stability curve (A2 element 8)

Fraction of items whose uncertain-item stratum changes between consecutive k, at the fixed
normalized-answer-entropy threshold of 0.30.

| k pair | items changing stratum | items compared | fraction | artifact |
|---|---|---|---|---|
| 5 to 8 | | | | |
| 8 to 16 | | | | |
| 16 to 32 | | | | |

Reported, never gated. State whether the curve is flattening by k = 32 or not.

### A3.4 Uncertain-item n per cell, and the MDE at the frozen thresholds (A2 element 8)

| Quantity | Value | Denominator | Artifact |
|---|---|---|---|
| Fraction of clean-correct items with modal answer correct and normalized entropy at or above 0.30 | | | |
| Projected uncertain-item n per cell at 570 entered | | | |
| Minimum detectable rate at that n for the frozen 30 percent follow threshold | | | |
| Minimum detectable rate at that n for the frozen 50 percent silent-given-follow threshold | | | |

The two thresholds are the frozen H1 and H2 values of
`PREREGISTRATION_uncertain_items.md` and are not changed here; only the MDE at the realized n is
new.

### A3.5 Measured repeated-curve mediator noise (A2 element 7, PF-13)

This is the required Phase 1 deliverable. Until it exists, no pooled row estimand ships.

| Quantity | Value | Denominator | Artifact |
|---|---|---|---|
| Held-out items with repeated curves | | | |
| Repeats per item | | | |
| sd(M) across items (commitment depth) | | | |
| sd(M) across items (curve area) | | | |
| Within-item repeat sd sigma_u (commitment depth) | | | |
| Within-item repeat sd sigma_u (curve area) | | | |
| Reliability ratio lambda = sigma_m^2 / (sigma_m^2 + sigma_u^2) | | | |

Then state which of the two permitted routes ships: a latent-M layer using this estimate, or the
printed attenuation band from the closed form validated in 01-SIZING I.3
(`beta' = beta lambda / c`, `alpha' = (alpha + beta gamma (1 - lambda)) / c`,
`c = sqrt(1 + beta^2 sigma_m^2 (1 - lambda))`). Name one; do not ship both.

Route chosen: `____`. Reason: `____`.

### A3.6 Measured throughput per model class (A2 element 16)

Replaces the first-order compute estimates as the budget of record. Every rate is that arm's
completed calls divided by that arm's wall-clock seconds, both read from the run's own
timestamped log.

| Model class | Card type | Full generations per second | Calls per second | Denominator (calls / seconds) | Artifact |
|---|---|---|---|---|---|
| 8B to 9B | | | | | |
| 24B to 35B | | | | | |
| 70B dense (tensor-parallel 2) | | | | | |
| 120B mxfp4 | | | | | |

Also record the concurrency sweep result, because the skeleton's 1.182 calls per second overall
(629 calls over 532.0 seconds) was measured on a MIG 3g.40gb slice at 30 items, where vLLM never
saw a deep enough request queue to batch.

| Concurrent requests | Full generations per second | Denominator | Artifact |
|---|---|---|---|
| 1 | | | |
| 8 | | | |
| 32 | | | |
| 64 | | | |

Then state, in one line each: the implied card-hours per 8B cell at n = 570 entered; whether the
degradation ladder of A2 element 16 is triggered; and if it is, which rung.

### A3.7 Open items carried forward, not filled here

List anything A2 expected A3 to fill that the skeleton did not produce, with what blocks it and
who owns it. An empty list is a valid entry and is written as "none".

| Item | Blocked by | Owner |
|---|---|---|
| | | |

### A3.8 Scope

This amendment supplies values for quantities A2 registered as formulas. It changes no element,
no threshold, no instrument, no estimand and no P-item. Nothing in
`PREREGISTRATION_jury_and_scale.md` above the A3 heading is edited.
