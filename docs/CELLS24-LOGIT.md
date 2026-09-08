# The logit-level column B: gate G1 on 24 cells, and the rows that cleared it

Amendment A5 of `experiments/PREREGISTRATION_jury_and_scale.md` adds a second
outcome scale to a cell that already has one. Y becomes the renormalized
log-probability margin of the planted option against the best other letter, in
nats, read on each record's own two prompts by the pass of `docs/LOGIT-PASS.md`.
X and M do not change. A5.4 puts six conditions in front of that row and A5.6
declines to put a verdict behind it.

This document covers 24 of 24 cells with a
`logit.json` on disk, 6 whose logit sidecar merged into their records,
and 6 that cleared all six conditions and print a logit-level row.

| what | value |
|---|---|
| cells with a file | 24 |
| cells with a merged sidecar | 6 |
| cells printing a logit-level row | 6 |
| cells with no file | 0 |
| analysis commit | e221d4b7850f |
| logit pass job ids | 829020 |
| analysis script sha256 | ecd13077b99d |
| artifacts | `experiments/results/cells24-logit/<model>/<substrate>/<cue>/logit.json` |

Nothing here ranks models against each other. A5.5 forbids a logit-level number
from entering a promotion decision or a ranking, and the cells are printed in the
order the cell list fixes.

## 1. Gate G1, all six conditions, every cell

A5.4: a row that fails any of the six is NOT PRINTED, and the cell prints the name
of the check that failed and its measured value in its place. Nothing partial is
published from a failing row: no effect, no interval, no mediated share, no
`rho*_point`. Every condition is evaluated on every cell anyway, so a reader can
see which ones already hold.

| cell | 1 unit check | 2 endpoint | 3 logit scale | 4 clean variance | 5 TE identity | 6 letter mass | row |
|---|---|---|---|---|---|---|---|
| qwen3-8b/arc_challenge/stated-hint | pass | pass | pass | pass | pass | pass | printed |
| qwen3-8b/arc_challenge/professor | pass | pass | pass | pass | pass | pass | printed |
| qwen3-8b/arc_challenge/metadata | pass | pass | pass | pass | pass | pass | printed |
| qwen3-8b/arc_challenge/grader-code | pass | pass | pass | pass | pass | pass | printed |
| qwen3-8b/aqua_rat/stated-hint | pass | pass | pass | pass | pass | pass | printed |
| qwen3-8b/aqua_rat/professor | pass | pass | pass | pass | pass | pass | printed |
| qwen3-8b/aqua_rat/metadata | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| qwen3-8b/aqua_rat/grader-code | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/arc_challenge/stated-hint | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/arc_challenge/professor | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/arc_challenge/metadata | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/arc_challenge/grader-code | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/aqua_rat/stated-hint | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/aqua_rat/professor | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/aqua_rat/metadata | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| gemma-2-9b-it/aqua_rat/grader-code | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/arc_challenge/stated-hint | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/arc_challenge/professor | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/arc_challenge/metadata | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/arc_challenge/grader-code | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/aqua_rat/stated-hint | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/aqua_rat/professor | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/aqua_rat/metadata | pass | pass | FAIL | FAIL | FAIL | pass | not printed |
| llama-3.1-8b-instruct/aqua_rat/grader-code | pass | pass | FAIL | FAIL | FAIL | pass | not printed |

### Counts by condition

| condition | passing | failing |
|---|---|---|
| 1_unit_check_passed | 24/24 | 0/24 |
| 2_pinned_self_hosted_endpoint | 24/24 | 0/24 |
| 3_records_carry_the_logit_scale | 6/24 | 18/24 |
| 4_clean_arm_outcome_variance_positive | 6/24 | 18/24 |
| 5_te_logit_equals_arm_difference | 6/24 | 18/24 |
| 6_letter_probability_mass_summarised | 24/24 | 0/24 |

### Cells read at the text level only, and why

| cell | reason |
|---|---|
| qwen3-8b/aqua_rat/metadata | no arms_transcripts_*.logit.json in the cell directory: the logit-level generation pass of docs/LOGIT-PASS.md has not been run for this cell, so the cell is read at the TEXT level only and A5.4 condition 3 fails |
| qwen3-8b/aqua_rat/grader-code | no arms_transcripts_*.logit.json in the cell directory: the logit-level generation pass of docs/LOGIT-PASS.md has not been run for this cell, so the cell is read at the TEXT level only and A5.4 condition 3 fails |
| gemma-2-9b-it/arc_challenge/stated-hint | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| gemma-2-9b-it/arc_challenge/professor | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| gemma-2-9b-it/arc_challenge/metadata | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| gemma-2-9b-it/arc_challenge/grader-code | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| gemma-2-9b-it/aqua_rat/stated-hint | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| gemma-2-9b-it/aqua_rat/professor | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| gemma-2-9b-it/aqua_rat/metadata | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| gemma-2-9b-it/aqua_rat/grader-code | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/gemma-2-9b-it when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/arc_challenge/stated-hint | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/arc_challenge/professor | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/arc_challenge/metadata | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/arc_challenge/grader-code | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/aqua_rat/stated-hint | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/aqua_rat/professor | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/aqua_rat/metadata | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |
| llama-3.1-8b-instruct/aqua_rat/grader-code | no logit_pass_done.json under /home/e/e1506804/bcf/results/logit-pass/llama-3.1-8b-instruct when this job ran: the logit pass for this model family has not written a completion marker, so none of its sidecars is read |

## 2. The cells

### qwen3-8b/arc_challenge/stated-hint

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | arc_challenge, stated-hint |
| text-level job | 826733 |
| logit pass job | 829020 |
| sidecar | present |
| sidecar entries keyed | 1396/1396 |
| records with no sidecar entry | 0/1396 |
| keying mismatches | 0 |
| items entering the logit fit | 1396 of 1396 records |
| rows | 2792 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`c3d33e397559`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0842 [+0.0663, +0.1016] | -0.4657 [-0.6028, -0.2968] |
| NIE | +0.0788 [+0.0636, +0.0972] | +0.0238 [-0.1017, +0.1340] |
| TE | +0.1629 [+0.1338, +0.1949] | -0.4419 [-0.5276, -0.3243] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1784 | -0.4419 |
| model-implied minus randomized | -0.0154 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 24.0999 |
| hinted | not printed on that scale | 35.3378 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 1043/1396 | 0.7471 | 0.0000 | 0.2529 | {'text_level_answer_unscorable': 0} |
| hinted | 910/1396 | 0.6519 | 0.1784 | 0.2557 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.4835 |
| logit level | -0.0539 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.7731 [+0.7419, +0.8007] |
| logit level | +0.0086 [+0.0006, +0.0507] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | not applicable: no effect is supported at rho = 0, so the verdict is unresolved and there is no crossing to report (prereg 8.1) |
| text level, verdict | unresolved |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | -0.4657 |
| beta | -0.1840 |
| gamma | -0.1295 |
| sigma_m | 0.2534 |
| sigma_y (nats) | 5.4513 |
| mu_m, alpha0 | 0.9709, -3.1610 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 6.661e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +5.2866 | -5.7284 | -0.4419 |
| -0.700 | +2.2651 | -2.7069 | -0.4419 |
| -0.500 | +1.1428 | -1.5846 | -0.4419 |
| -0.300 | +0.4104 | -0.8523 | -0.4419 |
| -0.100 | -0.1857 | -0.2562 | -0.4419 |
| +0.000 | -0.4657 | +0.0238 | -0.4419 |
| +0.100 | -0.7457 | +0.3038 | -0.4419 |
| +0.300 | -1.3419 | +0.9000 | -0.4419 |
| +0.500 | -2.0742 | +1.6323 | -0.4419 |
| +0.700 | -3.1965 | +2.7546 | -0.4419 |
| +0.900 | -6.2180 | +5.7761 | -0.4419 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2792, min 5.262e-18, median 2.116e-14, max 4.070e-03 |
| clean arm | n 1396, min 5.262e-18, median 2.399e-15, max 1.760e-06 |
| hinted arm | n 1396, min 3.615e-16, median 1.106e-13, max 4.070e-03 |
| below 0.01 | 2792/2792 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0948, NIE +0.0049, TE -0.0900.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### qwen3-8b/arc_challenge/professor

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | arc_challenge, professor |
| text-level job | 827051 |
| logit pass job | 829020 |
| sidecar | present |
| sidecar entries keyed | 1396/1396 |
| records with no sidecar entry | 0/1396 |
| keying mismatches | 0 |
| items entering the logit fit | 1396 of 1396 records |
| rows | 2792 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`6d4f2f114278`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1105 [+0.0932, +0.1287] | -1.1118 [-1.3660, -0.7997] |
| NIE | +0.1474 [+0.1246, +0.1725] | -0.1495 [-0.3504, +0.0162] |
| TE | +0.2579 [+0.2220, +0.2971] | -1.2613 [-1.4483, -1.0781] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.2593 | -1.2613 |
| model-implied minus randomized | -0.0014 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 24.0999 |
| hinted | not printed on that scale | 56.7595 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 1043/1396 | 0.7471 | 0.0000 | 0.2529 | {'text_level_answer_unscorable': 0} |
| hinted | 820/1396 | 0.5874 | 0.2593 | 0.2564 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.5714 |
| logit level | 0.1185 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.8116 [+0.7858, +0.8370] |
| logit level | +0.0333 [+0.0039, +0.0811] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | not applicable: no effect is supported at rho = 0, so the verdict is unresolved and there is no crossing to report (prereg 8.1) |
| text level, verdict | unresolved |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | -1.1118 |
| beta | +0.7061 |
| gamma | -0.2117 |
| sigma_m | 0.2995 |
| sigma_y (nats) | 6.3549 |
| mu_m, alpha0 | 0.9709, -4.0252 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 4.441e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +8.1656 | -9.4269 | -1.2613 |
| -0.700 | +3.2925 | -4.5538 | -1.2613 |
| -0.500 | +1.4824 | -2.7437 | -1.2613 |
| -0.300 | +0.3013 | -1.5626 | -1.2613 |
| -0.100 | -0.6602 | -0.6011 | -1.2613 |
| +0.000 | -1.1118 | -0.1495 | -1.2613 |
| +0.100 | -1.5634 | +0.3021 | -1.2613 |
| +0.300 | -2.5249 | +1.2635 | -1.2613 |
| +0.500 | -3.7060 | +2.4446 | -1.2613 |
| +0.700 | -5.5161 | +4.2547 | -1.2613 |
| +0.900 | -10.3892 | +9.1279 | -1.2613 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2792, min 5.262e-18, median 1.383e-13, max 9.524e-01 |
| clean arm | n 1396, min 5.262e-18, median 2.399e-15, max 1.760e-06 |
| hinted arm | n 1396, min 1.395e-15, median 7.478e-12, max 9.524e-01 |
| below 0.01 | 2791/2792 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.2264, NIE -0.0304, TE -0.2568.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### qwen3-8b/arc_challenge/metadata

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | arc_challenge, metadata |
| text-level job | 827286 |
| logit pass job | 829020 |
| sidecar | present |
| sidecar entries keyed | 534/534 |
| records with no sidecar entry | 0/534 |
| keying mismatches | 0 |
| items entering the logit fit | 534 of 534 records |
| rows | 1068 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`1fa34f92921a`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0369 [+0.0200, +0.0534] | -0.9207 [-1.2685, -0.5910] |
| NIE | +0.0246 [+0.0124, +0.0414] | -0.1955 [-0.3586, -0.0562] |
| TE | +0.0615 [+0.0330, +0.0921] | -1.1162 [-1.3760, -0.8557] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1011 | -1.1162 |
| model-implied minus randomized | -0.0396 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 23.7155 |
| hinted | not printed on that scale | 53.4659 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 393/534 | 0.7360 | 0.0000 | 0.2640 | {'text_level_answer_unscorable': 0} |
| hinted | 347/534 | 0.6498 | 0.1011 | 0.2640 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.4002 |
| logit level | 0.1751 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.6920 [+0.6357, +0.7760] |
| logit level | +0.0886 [+0.0308, +0.1530] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | not applicable: no effect is supported at rho = 0, so the verdict is unresolved and there is no crossing to report (prereg 8.1) |
| text level, verdict | unresolved |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | -0.9207 |
| beta | +2.6011 |
| gamma | -0.0752 |
| sigma_m | 0.2117 |
| sigma_y (nats) | 6.1877 |
| mu_m, alpha0 | 0.9742, -5.6822 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 2.220e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +3.6148 | -4.7310 | -1.1162 |
| -0.700 | +1.2325 | -2.3486 | -1.1162 |
| -0.500 | +0.3476 | -1.4637 | -1.1162 |
| -0.300 | -0.2299 | -0.8863 | -1.1162 |
| -0.100 | -0.6999 | -0.4163 | -1.1162 |
| +0.000 | -0.9207 | -0.1955 | -1.1162 |
| +0.100 | -1.1414 | +0.0253 | -1.1162 |
| +0.300 | -1.6115 | +0.4953 | -1.1162 |
| +0.500 | -2.1889 | +1.0727 | -1.1162 |
| +0.700 | -3.0738 | +1.9576 | -1.1162 |
| +0.900 | -5.4562 | +4.3400 | -1.1162 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1068, min 5.912e-18, median 4.629e-13, max 2.930e-02 |
| clean arm | n 534, min 5.912e-18, median 2.292e-15, max 4.945e-09 |
| hinted arm | n 534, min 1.543e-14, median 1.692e-10, max 2.930e-02 |
| below 0.01 | 1067/1068 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.1889, NIE -0.0401, TE -0.2290.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### qwen3-8b/arc_challenge/grader-code

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | arc_challenge, grader-code |
| text-level job | 827292 |
| logit pass job | 829020 |
| sidecar | present |
| sidecar entries keyed | 533/533 |
| records with no sidecar entry | 0/533 |
| keying mismatches | 0 |
| items entering the logit fit | 533 of 533 records |
| rows | 1066 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`f91b0f8b1734`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0389 [+0.0200, +0.0592] | -0.4126 [-0.6603, -0.2246] |
| NIE | +0.0239 [+0.0104, +0.0379] | -0.0871 [-0.2023, +0.0497] |
| TE | +0.0628 [+0.0293, +0.0945] | -0.4997 [-0.6750, -0.3320] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1013 | -0.4997 |
| model-implied minus randomized | -0.0385 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 23.6467 |
| hinted | not printed on that scale | 37.8681 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 398/533 | 0.7467 | 0.0000 | 0.2533 | {'text_level_answer_unscorable': 0} |
| hinted | 353/533 | 0.6623 | 0.1013 | 0.2552 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.3807 |
| logit level | 0.1743 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.6790 [+0.6164, +0.7546] |
| logit level | +0.0460 [+0.0014, +0.0947] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | not applicable: no effect is supported at rho = 0, so the verdict is unresolved and there is no crossing to report (prereg 8.1) |
| text level, verdict | unresolved |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | -0.4126 |
| beta | +1.1781 |
| gamma | -0.0739 |
| sigma_m | 0.2164 |
| sigma_y (nats) | 5.5401 |
| mu_m, alpha0 | 0.9730, -4.5285 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 1.110e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +3.4956 | -3.9953 | -0.4997 |
| -0.700 | +1.4427 | -1.9424 | -0.4997 |
| -0.500 | +0.6802 | -1.1799 | -0.4997 |
| -0.300 | +0.1826 | -0.6824 | -0.4997 |
| -0.100 | -0.2224 | -0.2773 | -0.4997 |
| +0.000 | -0.4126 | -0.0871 | -0.4997 |
| +0.100 | -0.6029 | +0.1032 | -0.4997 |
| +0.300 | -1.0079 | +0.5082 | -0.4997 |
| +0.500 | -1.5054 | +1.0057 | -0.4997 |
| +0.700 | -2.2680 | +1.7683 | -0.4997 |
| +0.900 | -4.3208 | +3.8211 | -0.4997 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1066, min 5.912e-18, median 6.445e-14, max 6.962e-06 |
| clean arm | n 533, min 5.912e-18, median 2.323e-15, max 4.945e-09 |
| hinted arm | n 533, min 4.341e-16, median 2.737e-12, max 6.962e-06 |
| below 0.01 | 1066/1066 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0848, NIE -0.0179, TE -0.1027.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### qwen3-8b/aqua_rat/stated-hint

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | aqua_rat, stated-hint |
| text-level job | 827391 |
| logit pass job | 829020 |
| sidecar | present |
| sidecar entries keyed | 1147/1147 |
| records with no sidecar entry | 0/1147 |
| keying mismatches | 0 |
| items entering the logit fit | 1144 of 1147 records |
| rows | 2288 |
| drops by reason | {'hinted_curve_unscorable': 3, 'items_dropped': 3} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`efe55b9c10d2`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0727 [+0.0582, +0.0894] | -0.1856 [-0.2847, -0.0847] |
| NIE | -0.0022 [-0.0038, -0.0007] | +0.0043 [-0.0198, +0.0222] |
| TE | +0.0705 [+0.0567, +0.0870] | -0.1813 [-0.2708, -0.0903] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0708 | -0.1813 |
| model-implied minus randomized | -0.0003 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 25.0414 |
| hinted | not printed on that scale | 29.6181 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 923/1144 | 0.8068 | 0.0000 | 0.1932 | {'text_level_answer_unscorable': 0} |
| hinted | 862/1144 | 0.7535 | 0.0708 | 0.1932 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | -0.0307 |
| logit level | -0.0237 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.2483 [+0.1441, +0.3427] |
| logit level | +0.0129 [+0.0024, +0.0570] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | not applicable: no effect is supported at rho = 0, so the verdict is unresolved and there is no crossing to report (prereg 8.1) |
| text level, verdict | unresolved |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | -0.1856 |
| beta | +0.1801 |
| gamma | +0.0238 |
| sigma_m | 0.3734 |
| sigma_y (nats) | 5.2274 |
| mu_m, alpha0 | 0.6422, -4.5488 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 5.551e-17 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | -0.8749 | +0.6937 | -0.1813 |
| -0.700 | -0.5128 | +0.3316 | -0.1813 |
| -0.500 | -0.3783 | +0.1971 | -0.1813 |
| -0.300 | -0.2906 | +0.1093 | -0.1813 |
| -0.100 | -0.2191 | +0.0379 | -0.1813 |
| +0.000 | -0.1856 | +0.0043 | -0.1813 |
| +0.100 | -0.1520 | -0.0293 | -0.1813 |
| +0.300 | -0.0806 | -0.1007 | -0.1813 |
| +0.500 | +0.0072 | -0.1885 | -0.1813 |
| +0.700 | +0.1417 | -0.3230 | -0.1813 |
| +0.900 | +0.5038 | -0.6851 | -0.1813 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2294, min 7.498e-18, median 1.915e-12, max 1.053e-02 |
| clean arm | n 1147, min 7.498e-18, median 2.901e-13, max 2.902e-06 |
| hinted arm | n 1147, min 1.963e-15, median 1.644e-11, max 1.053e-02 |
| below 0.01 | 2293/2294 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0371, NIE +0.0009, TE -0.0362.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### qwen3-8b/aqua_rat/professor

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | aqua_rat, professor |
| text-level job | 827408 |
| logit pass job | 829020 |
| sidecar | present |
| sidecar entries keyed | 1153/1153 |
| records with no sidecar entry | 0/1153 |
| keying mismatches | 0 |
| items entering the logit fit | 1153 of 1153 records |
| rows | 2306 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`30cd44030ac4`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1118 [+0.0955, +0.1266] | -0.5516 [-0.6662, -0.4441] |
| NIE | +0.0068 [+0.0025, +0.0126] | +0.0049 [-0.0164, +0.0291] |
| TE | +0.1185 [+0.1034, +0.1348] | -0.5467 [-0.6523, -0.4523] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1214 | -0.5467 |
| model-implied minus randomized | -0.0029 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 25.1558 |
| hinted | not printed on that scale | 35.3196 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 928/1153 | 0.8049 | 0.0000 | 0.1951 | {'text_level_answer_unscorable': 0} |
| hinted | 822/1153 | 0.7129 | 0.1214 | 0.1917 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0570 |
| logit level | -0.0089 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4921 [+0.4192, +0.5846] |
| logit level | +0.0125 [+0.0010, +0.0653] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | not applicable: no effect is supported at rho = 0, so the verdict is unresolved and there is no crossing to report (prereg 8.1) |
| text level, verdict | unresolved |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | -0.5516 |
| beta | -0.1830 |
| gamma | -0.0267 |
| sigma_m | 0.3770 |
| sigma_y (nats) | 5.4984 |
| mu_m, alpha0 | 0.6402, -4.3614 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 1.110e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +0.2515 | -0.7982 | -0.5467 |
| -0.700 | -0.1703 | -0.3764 | -0.5467 |
| -0.500 | -0.3270 | -0.2197 | -0.5467 |
| -0.300 | -0.4293 | -0.1174 | -0.5467 |
| -0.100 | -0.5125 | -0.0342 | -0.5467 |
| +0.000 | -0.5516 | +0.0049 | -0.5467 |
| +0.100 | -0.5907 | +0.0440 | -0.5467 |
| +0.300 | -0.6739 | +0.1272 | -0.5467 |
| +0.500 | -0.7761 | +0.2294 | -0.5467 |
| +0.700 | -0.9328 | +0.3861 | -0.5467 |
| +0.900 | -1.3547 | +0.8080 | -0.5467 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2306, min 7.498e-18, median 7.009e-13, max 2.612e-04 |
| clean arm | n 1153, min 7.498e-18, median 2.890e-13, max 2.902e-06 |
| hinted arm | n 1153, min 1.750e-15, median 1.518e-12, max 2.612e-04 |
| below 0.01 | 2306/2306 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.1099, NIE +0.0010, TE -0.1090.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### qwen3-8b/aqua_rat/metadata

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | aqua_rat, metadata |
| text-level job | 828499 |
| logit pass job | 829020 |
| sidecar | absent |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/433 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/433 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/433 records, `outcome_scale = logprob_margin` on 0/433, a `logprob_source_token` on 0/433, both arm margins on 0/433.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### qwen3-8b/aqua_rat/grader-code

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | aqua_rat, grader-code |
| text-level job | 828521 |
| logit pass job | 829020 |
| sidecar | absent |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/437 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/437 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/437 records, `outcome_scale = logprob_margin` on 0/437, a `logprob_source_token` on 0/437, both arm margins on 0/437.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/arc_challenge/stated-hint

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, stated-hint |
| text-level job | 826737 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/1380 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/1380 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/1380 records, `outcome_scale = logprob_margin` on 0/1380, a `logprob_source_token` on 0/1380, both arm margins on 0/1380.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/arc_challenge/professor

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, professor |
| text-level job | 827284 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/1380 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/1380 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/1380 records, `outcome_scale = logprob_margin` on 0/1380, a `logprob_source_token` on 0/1380, both arm margins on 0/1380.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/arc_challenge/metadata

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, metadata |
| text-level job | 827287 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/524 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/524 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/524 records, `outcome_scale = logprob_margin` on 0/524, a `logprob_source_token` on 0/524, both arm margins on 0/524.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/arc_challenge/grader-code

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, grader-code |
| text-level job | 827293 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/524 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/524 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/524 records, `outcome_scale = logprob_margin` on 0/524, a `logprob_source_token` on 0/524, both arm margins on 0/524.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/aqua_rat/stated-hint

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, stated-hint |
| text-level job | 827392 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/897 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/897 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/897 records, `outcome_scale = logprob_margin` on 0/897, a `logprob_source_token` on 0/897, both arm margins on 0/897.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/aqua_rat/professor

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, professor |
| text-level job | 827409 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/899 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/899 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/899 records, `outcome_scale = logprob_margin` on 0/899, a `logprob_source_token` on 0/899, both arm margins on 0/899.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/aqua_rat/metadata

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, metadata |
| text-level job | 828503 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/361 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/361 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/361 records, `outcome_scale = logprob_margin` on 0/361, a `logprob_source_token` on 0/361, both arm margins on 0/361.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### gemma-2-9b-it/aqua_rat/grader-code

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, grader-code |
| text-level job | 828522 |
| logit pass job | 829182 |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/356 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/356 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/356 records, `outcome_scale = logprob_margin` on 0/356, a `logprob_source_token` on 0/356, both arm margins on 0/356.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/arc_challenge/stated-hint

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, stated-hint |
| text-level job | 826738 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/1321 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/1321 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/1321 records, `outcome_scale = logprob_margin` on 0/1321, a `logprob_source_token` on 0/1321, both arm margins on 0/1321.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/arc_challenge/professor

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, professor |
| text-level job | 827285 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/1321 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/1321 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/1321 records, `outcome_scale = logprob_margin` on 0/1321, a `logprob_source_token` on 0/1321, both arm margins on 0/1321.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/arc_challenge/metadata

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, metadata |
| text-level job | 827097 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/503 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/503 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/503 records, `outcome_scale = logprob_margin` on 0/503, a `logprob_source_token` on 0/503, both arm margins on 0/503.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/arc_challenge/grader-code

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, grader-code |
| text-level job | 827294 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/503 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/503 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/503 records, `outcome_scale = logprob_margin` on 0/503, a `logprob_source_token` on 0/503, both arm margins on 0/503.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/aqua_rat/stated-hint

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, stated-hint |
| text-level job | 827393 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/1012 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/1012 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/1012 records, `outcome_scale = logprob_margin` on 0/1012, a `logprob_source_token` on 0/1012, both arm margins on 0/1012.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/aqua_rat/professor

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, professor |
| text-level job | 827410 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/1012 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/1012 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/1012 records, `outcome_scale = logprob_margin` on 0/1012, a `logprob_source_token` on 0/1012, both arm margins on 0/1012.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/aqua_rat/metadata

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, metadata |
| text-level job | 828504 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/370 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/370 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/370 records, `outcome_scale = logprob_margin` on 0/370, a `logprob_source_token` on 0/370, both arm margins on 0/370.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

### llama-3.1-8b-instruct/aqua_rat/grader-code

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, grader-code |
| text-level job | 828523 |
| logit pass job | none |
| sidecar | ignored |

**No logit-level row.** A5.4: the checks that failed, with their measured
values, print in its place.

| failing check | measured value |
|---|---|
| 3_records_carry_the_logit_scale | 0/370 |
| 4_clean_arm_outcome_variance_positive | NOT COMPUTABLE. The clean-arm and hinted-arm outcome variance on the logprob margin needs a margin on the ARMS rows; 0/370 arms rows carry one. |
| 5_te_logit_equals_arm_difference | NOT COMPUTABLE for the same reason as condition 4: TE_logit is defined on the arms and there is no arms-level margin to compute it from. |

Condition 3 counts: `intervention_level = logit` on 0/370 records, `outcome_scale = logprob_margin` on 0/370, a `logprob_source_token` on 0/370, both arm margins on 0/370.

The text-level row for this cell is unaffected and is in `docs/CELLS24-FITS.md`.

## 3. What this establishes, and what it does not

**What it establishes.**

1. Gate G1's condition 3 is satisfiable. Before the logit pass, 18 of 18 cells
   failed it on the same line: no arms record in this project carried
   `intervention_level = logit`, because the generation pass that writes it had
   never been run. 6 cells now carry it on every record, keyed to the
   text-level record by position, by a content key recomputed from the record's own
   six identifying fields, and by the source file's sha256.
2. The clean arm varies on this scale. That is the whole reason
   `docs/OUTCOME-SCALE-NOTE.md` exists: on the binary scale the clean-arm outcome
   variance is exactly zero by construction, so the probit fit separates on X and
   the NDE and NIE split rests on the link extrapolating into a region the clean
   arm never visits. Each printed row above gives both arm variances in nats
   squared and neither is zero.
3. The total effect is pinned by the data on this scale. `TE_logit` equals the
   randomized arm difference in the margin to the last printed digit, which is an
   algebraic identity under A5.2 and is checked as a code fault, not reported as a
   finding.

**What it does not establish.**

1. **No verdict.** A5.6 sets no load-bearing threshold on this scale and this
   document sets none either. Every printed row says
   `not applicable, no threshold pre-registered on this scale` where a verdict
   would go. The three candidate rules A5.6 records, and the defect of each, are
   in the amendment; choosing one is the operator's and is made before any
   logit-level effect size is read, not after.
2. **No promotion, no ranking, no claim-status change.** A5.5 is explicit and
   nothing here is used for the element 21 comparison of section 22 either. The
   claim statuses of the 24 cells are what `docs/CELLS24-FITS.md` records.
3. **No cross-model comparison.** Rows exist for 1 model family: qwen3-8b.
   With more than one they still could not be ordered: A5.5 forbids ranking on a
   logit-level number, and this document prints no ordering.
4. **Identification is unchanged.** Sequential ignorability with A3 priced by rho,
   exactly as section 2.3 states it. Part 3.3 of the outcome-scale note measures
   this directly on a world with no mediator-to-outcome arrow: across 100 datasets
   per condition, NDE and NIE coverage is 0 of 100 on all three readouts tried,
   the continuous one included. What the continuous scale removes is the link
   extrapolation, not the confounding.
5. **The mass caveat travels with every margin.** The letter probability mass
   summaries above are the raw share of the next-token distribution the answer
   letters hold before renormalization. Where that share is tiny, the margin is a
   well defined conditional quantity and it is also a quantity about a region the
   model almost never enters, so the renormalization does nearly all of the work.
6. **The card type is not held fixed against the run being augmented.** The
   text-level cells were generated across a MIG slice of an A100 80GB and a whole
   A100 80GB; one server per model means all of a model's reads share whichever
   card the wave allocated. The pass records its own serving mode, so the
   difference is visible rather than hidden.
7. **The read is a new measurement, not a recovery.** The original run never scored
   these prompts for letter logprobs. What makes it the same measurement is the
   prompt, rebuilt from the record's own banked fields through the frozen
   instruments and checked three ways. What is not guaranteed is the server: a
   different process on a different day, which is why the pinned serving mode, the
   batch-invariant flag and the determinism preflight all ran again.

