# The logit-level column B: gate G1 on 24 cells, and the rows that cleared it

Amendment A5 of `experiments/PREREGISTRATION_jury_and_scale.md` adds a second
outcome scale to a cell that already has one. Y becomes the renormalized
log-probability margin of the planted option against the best other letter, in
nats, read on each record's own two prompts by the pass of `docs/LOGIT-PASS.md`.
X and M do not change. A5.4 puts six conditions in front of that row and A5.6
declines to put a verdict behind it.

This document covers 24 of 24 cells with a
`logit.json` on disk, 24 whose logit sidecar merged into their records,
and 24 that cleared all six conditions and print a logit-level row.

| what | value |
|---|---|
| cells with a file | 24 |
| cells with a merged sidecar | 24 |
| cells printing a logit-level row | 24 |
| cells with no file | 0 |
| analysis commit | 7a1bc034b837 |
| logit pass job ids | 829881, 829926, 829960 |
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
| qwen3-8b/aqua_rat/metadata | pass | pass | pass | pass | pass | pass | printed |
| qwen3-8b/aqua_rat/grader-code | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/arc_challenge/stated-hint | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/arc_challenge/professor | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/arc_challenge/metadata | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/arc_challenge/grader-code | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/aqua_rat/stated-hint | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/aqua_rat/professor | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/aqua_rat/metadata | pass | pass | pass | pass | pass | pass | printed |
| gemma-2-9b-it/aqua_rat/grader-code | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/arc_challenge/stated-hint | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/arc_challenge/professor | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/arc_challenge/metadata | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/arc_challenge/grader-code | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/aqua_rat/stated-hint | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/aqua_rat/professor | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/aqua_rat/metadata | pass | pass | pass | pass | pass | pass | printed |
| llama-3.1-8b-instruct/aqua_rat/grader-code | pass | pass | pass | pass | pass | pass | printed |

### Counts by condition

| condition | passing | failing |
|---|---|---|
| 1_unit_check_passed | 24/24 | 0/24 |
| 2_pinned_self_hosted_endpoint | 24/24 | 0/24 |
| 3_records_carry_the_logit_scale | 24/24 | 0/24 |
| 4_clean_arm_outcome_variance_positive | 24/24 | 0/24 |
| 5_te_logit_equals_arm_difference | 24/24 | 0/24 |
| 6_letter_probability_mass_summarised | 24/24 | 0/24 |

## 2. The cells

### qwen3-8b/arc_challenge/stated-hint

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | arc_challenge, stated-hint |
| text-level job | 826733 |
| logit pass job | 829881 |
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
| logit pass job | 829881 |
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
| logit pass job | 829881 |
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
| logit pass job | 829881 |
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
| logit pass job | 829881 |
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
| logit pass job | 829881 |
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
| logit pass job | 829881 |
| sidecar | present |
| sidecar entries keyed | 433/433 |
| records with no sidecar entry | 0/433 |
| keying mismatches | 0 |
| items entering the logit fit | 432 of 433 records |
| rows | 864 |
| drops by reason | {'hinted_curve_unscorable': 1, 'items_dropped': 1} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`7aaadacb4065`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1093 [+0.0822, +0.1344] | -0.6043 [-0.8588, -0.3728] |
| NIE | +0.0112 [+0.0035, +0.0219] | +0.0007 [-0.0498, +0.0574] |
| TE | +0.1205 [+0.0953, +0.1486] | -0.6035 [-0.8397, -0.3780] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1273 | -0.6035 |
| model-implied minus randomized | -0.0069 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 26.0798 |
| hinted | not printed on that scale | 40.8170 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 350/432 | 0.8102 | 0.0000 | 0.1898 | {'text_level_answer_unscorable': 0} |
| hinted | 306/432 | 0.7083 | 0.1273 | 0.1921 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0930 |
| logit level | -0.0012 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.6065 [+0.4880, +0.7455] |
| logit level | +0.0014 [+0.0019, +0.1018] |

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
| alpha (NDE) | -0.6043 |
| beta | -0.0208 |
| gamma | -0.0359 |
| sigma_m | 0.3770 |
| sigma_y (nats) | 5.7835 |
| mu_m, alpha0 | 0.6387, -4.6730 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 2.220e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +0.5335 | -1.1370 | -0.6035 |
| -0.700 | -0.0641 | -0.5394 | -0.6035 |
| -0.500 | -0.2861 | -0.3174 | -0.6035 |
| -0.300 | -0.4310 | -0.1725 | -0.6035 |
| -0.100 | -0.5489 | -0.0546 | -0.6035 |
| +0.000 | -0.6043 | +0.0007 | -0.6035 |
| +0.100 | -0.6596 | +0.0561 | -0.6035 |
| +0.300 | -0.7776 | +0.1740 | -0.6035 |
| +0.500 | -0.9224 | +0.3189 | -0.6035 |
| +0.700 | -1.1444 | +0.5409 | -0.6035 |
| +0.900 | -1.7420 | +1.1385 | -0.6035 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 866, min 6.616e-18, median 1.745e-12, max 1.511e-01 |
| clean arm | n 433, min 6.616e-18, median 2.913e-13, max 1.264e-07 |
| hinted arm | n 433, min 8.150e-16, median 1.398e-11, max 1.511e-01 |
| below 0.01 | 865/866 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.1182, NIE +0.0001, TE -0.1180.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### qwen3-8b/aqua_rat/grader-code

| field | value |
|---|---|
| model | Qwen/Qwen3-8B |
| hf revision | b968826d9c46dd6066d109eabc6255188de91218 |
| substrate, cue family | aqua_rat, grader-code |
| text-level job | 828521 |
| logit pass job | 829881 |
| sidecar | present |
| sidecar entries keyed | 437/437 |
| records with no sidecar entry | 0/437 |
| keying mismatches | 0 |
| items entering the logit fit | 437 of 437 records |
| rows | 874 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`1e0a06355d41`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0988 [+0.0737, +0.1242] | +0.3717 [+0.1614, +0.5580] |
| NIE | +0.0045 [-0.0004, +0.0128] | +0.0150 [-0.0108, +0.0495] |
| TE | +0.1033 [+0.0774, +0.1342] | +0.3867 [+0.1865, +0.5724] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1053 | +0.3867 |
| model-implied minus randomized | -0.0019 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 24.7758 |
| hinted | not printed on that scale | 23.7653 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 356/437 | 0.8146 | 0.0000 | 0.1854 | {'text_level_answer_unscorable': 0} |
| hinted | 317/437 | 0.7254 | 0.1053 | 0.1876 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0439 |
| logit level | 0.0388 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4904 [+0.3744, +0.6595] |
| logit level | +0.0583 [+0.0039, +0.1385] |

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
| alpha (NDE) | +0.3717 |
| beta | -0.7774 |
| gamma | -0.0193 |
| sigma_m | 0.3693 |
| sigma_y (nats) | 4.9181 |
| mu_m, alpha0 | 0.6412, -4.2080 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 0.000e+00 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +0.9023 | -0.5156 | +0.3867 |
| -0.700 | +0.6236 | -0.2369 | +0.3867 |
| -0.500 | +0.5200 | -0.1334 | +0.3867 |
| -0.300 | +0.4525 | -0.0658 | +0.3867 |
| -0.100 | +0.3975 | -0.0108 | +0.3867 |
| +0.000 | +0.3717 | +0.0150 | +0.3867 |
| +0.100 | +0.3458 | +0.0408 | +0.3867 |
| +0.300 | +0.2908 | +0.0958 | +0.3867 |
| +0.500 | +0.2233 | +0.1634 | +0.3867 |
| +0.700 | +0.1197 | +0.2669 | +0.3867 |
| +0.900 | -0.1590 | +0.5456 | +0.3867 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 874, min 6.616e-18, median 6.004e-13, max 1.051e-02 |
| clean arm | n 437, min 6.616e-18, median 3.031e-13, max 1.264e-07 |
| hinted arm | n 437, min 4.524e-16, median 1.493e-12, max 1.051e-02 |
| below 0.01 | 873/874 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.0746, NIE +0.0030, TE +0.0776.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/arc_challenge/stated-hint

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, stated-hint |
| text-level job | 826737 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 1380/1380 |
| records with no sidecar entry | 0/1380 |
| keying mismatches | 0 |
| items entering the logit fit | 1375 of 1380 records |
| rows | 2750 |
| drops by reason | {'hinted_curve_unscorable': 5, 'items_dropped': 5} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`292ee893ec93`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1110 [+0.0965, +0.1274] | +0.2283 [+0.1281, +0.3285] |
| NIE | +0.1895 [+0.1703, +0.2153] | -0.0677 [-0.1658, +0.0377] |
| TE | +0.3005 [+0.2680, +0.3368] | +0.1606 [+0.1429, +0.1791] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.3098 | +0.1606 |
| model-implied minus randomized | -0.0093 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 10.0276 |
| hinted | not printed on that scale | 10.6642 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 1021/1375 | 0.7425 | 0.0000 | 0.2575 | {'text_level_answer_unscorable': 0} |
| hinted | 790/1375 | 0.5745 | 0.3098 | 0.2567 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.6306 |
| logit level | -0.4216 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.8223 [+0.8054, +0.8488] |
| logit level | +0.0248 [+0.0017, +0.0606] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | -0.2400 |
| text level, verdict | load-bearing at rho=0 |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | +0.2283 |
| beta | +0.2534 |
| gamma | -0.2672 |
| sigma_m | 0.3146 |
| sigma_y (nats) | 3.2155 |
| mu_m, alpha0 | 0.9741, -2.3767 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 3.331e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +5.8654 | -5.7048 | +0.1606 |
| -0.700 | +2.9044 | -2.7438 | +0.1606 |
| -0.500 | +1.8046 | -1.6440 | +0.1606 |
| -0.300 | +1.0869 | -0.9263 | +0.1606 |
| -0.100 | +0.5027 | -0.3421 | +0.1606 |
| +0.000 | +0.2283 | -0.0677 | +0.1606 |
| +0.100 | -0.0461 | +0.2067 | +0.1606 |
| +0.300 | -0.6303 | +0.7909 | +0.1606 |
| +0.500 | -1.3480 | +1.5086 | +0.1606 |
| +0.700 | -2.4478 | +2.6084 | +0.1606 |
| +0.900 | -5.4088 | +5.5694 | +0.1606 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2760, min 4.304e-07, median 2.489e-06, max 4.580e-05 |
| clean arm | n 1380, min 4.304e-07, median 1.454e-06, max 3.306e-05 |
| hinted arm | n 1380, min 8.810e-07, median 3.852e-06, max 4.580e-05 |
| below 0.01 | 2760/2760 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.0721, NIE -0.0214, TE +0.0507.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/arc_challenge/professor

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, professor |
| text-level job | 827284 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 1380/1380 |
| records with no sidecar entry | 0/1380 |
| keying mismatches | 0 |
| items entering the logit fit | 1371 of 1380 records |
| rows | 2742 |
| drops by reason | {'hinted_curve_unscorable': 9, 'items_dropped': 9} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`d0bbf44d6ae6`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0982 [+0.0818, +0.1136] | +0.1000 [+0.0110, +0.1948] |
| NIE | +0.1586 [+0.1351, +0.1832] | -0.0355 [-0.1297, +0.0531] |
| TE | +0.2568 [+0.2196, +0.2910] | +0.0645 [+0.0397, +0.0884] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.2823 | +0.0645 |
| model-implied minus randomized | -0.0254 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 10.1181 |
| hinted | not printed on that scale | 12.0321 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 1012/1371 | 0.7381 | 0.0000 | 0.2619 | {'text_level_answer_unscorable': 0} |
| hinted | 824/1371 | 0.6010 | 0.2823 | 0.2611 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.6177 |
| logit level | -0.5499 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.8063 [+0.7895, +0.8298] |
| logit level | +0.0135 [+0.0007, +0.0495] |

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
| alpha (NDE) | +0.1000 |
| beta | +0.1458 |
| gamma | -0.2433 |
| sigma_m | 0.3071 |
| sigma_y (nats) | 3.3276 |
| mu_m, alpha0 | 0.9734, -2.2463 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 1.110e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +5.5432 | -5.4787 | +0.0645 |
| -0.700 | +2.6841 | -2.6196 | +0.0645 |
| -0.500 | +1.6220 | -1.5575 | +0.0645 |
| -0.300 | +0.9291 | -0.8645 | +0.0645 |
| -0.100 | +0.3649 | -0.3004 | +0.0645 |
| +0.000 | +0.1000 | -0.0355 | +0.0645 |
| +0.100 | -0.1650 | +0.2295 | +0.0645 |
| +0.300 | -0.7291 | +0.7936 | +0.0645 |
| +0.500 | -1.4221 | +1.4866 | +0.0645 |
| +0.700 | -2.4841 | +2.5486 | +0.0645 |
| +0.900 | -5.3433 | +5.4078 | +0.0645 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2760, min 4.304e-07, median 2.914e-06, max 7.132e-05 |
| clean arm | n 1380, min 4.304e-07, median 1.458e-06, max 3.306e-05 |
| hinted arm | n 1380, min 1.181e-06, median 4.755e-06, max 7.132e-05 |
| below 0.01 | 2760/2760 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.0314, NIE -0.0111, TE +0.0203.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/arc_challenge/metadata

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, metadata |
| text-level job | 827287 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 524/524 |
| records with no sidecar entry | 0/524 |
| keying mismatches | 0 |
| items entering the logit fit | 524 of 524 records |
| rows | 1048 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`f760ea74ccf5`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0409 [+0.0234, +0.0668] | +0.6438 [+0.5761, +0.7202] |
| NIE | +0.0211 [+0.0127, +0.0360] | -0.0192 [-0.0904, +0.0480] |
| TE | +0.0621 [+0.0378, +0.0967] | +0.6246 [+0.5992, +0.6524] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0973 | +0.6246 |
| model-implied minus randomized | -0.0352 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 10.2294 |
| hinted | not printed on that scale | 9.3052 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 387/524 | 0.7385 | 0.0000 | 0.2615 | {'text_level_answer_unscorable': 0} |
| hinted | 357/524 | 0.6813 | 0.0973 | 0.2634 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.3405 |
| logit level | -0.0307 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.6541 [+0.5970, +0.7275] |
| logit level | +0.0199 [+0.0005, +0.0880] |

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
| alpha (NDE) | +0.6438 |
| beta | +0.2879 |
| gamma | -0.0667 |
| sigma_m | 0.2156 |
| sigma_y (nats) | 3.1247 |
| mu_m, alpha0 | 0.9662, -2.4056 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 2.220e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +2.6392 | -2.0146 | +0.6246 |
| -0.700 | +1.5911 | -0.9665 | +0.6246 |
| -0.500 | +1.2018 | -0.5772 | +0.6246 |
| -0.300 | +0.9477 | -0.3231 | +0.6246 |
| -0.100 | +0.7409 | -0.1163 | +0.6246 |
| +0.000 | +0.6438 | -0.0192 | +0.6246 |
| +0.100 | +0.5467 | +0.0779 | +0.6246 |
| +0.300 | +0.3399 | +0.2847 | +0.6246 |
| +0.500 | +0.0858 | +0.5388 | +0.6246 |
| +0.700 | -0.3035 | +0.9281 | +0.6246 |
| +0.900 | -1.3516 | +1.9762 | +0.6246 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1048, min 4.507e-07, median 2.505e-06, max 3.570e-05 |
| clean arm | n 524, min 4.507e-07, median 1.464e-06, max 1.395e-05 |
| hinted arm | n 524, min 1.056e-06, median 3.825e-06, max 3.570e-05 |
| below 0.01 | 1048/1048 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.2011, NIE -0.0060, TE +0.1951.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/arc_challenge/grader-code

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | arc_challenge, grader-code |
| text-level job | 827293 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 524/524 |
| records with no sidecar entry | 0/524 |
| keying mismatches | 0 |
| items entering the logit fit | 524 of 524 records |
| rows | 1048 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`0d77a8f42f74`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0260 [+0.0129, +0.0441] | +0.9441 [+0.8563, +1.0188] |
| NIE | +0.0087 [+0.0040, +0.0150] | -0.0175 [-0.0686, +0.0275] |
| TE | +0.0347 [+0.0182, +0.0599] | +0.9266 [+0.8634, +0.9904] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0630 | +0.9266 |
| model-implied minus randomized | -0.0283 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 10.2294 |
| hinted | not printed on that scale | 6.7949 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 387/524 | 0.7385 | 0.0000 | 0.2615 | {'text_level_answer_unscorable': 0} |
| hinted | 365/524 | 0.6966 | 0.0630 | 0.2595 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.2495 |
| logit level | -0.0189 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.5785 [+0.5081, +0.6511] |
| logit level | +0.0274 [+0.0009, +0.0970] |

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
| alpha (NDE) | +0.9441 |
| beta | +0.4193 |
| gamma | -0.0417 |
| sigma_m | 0.1904 |
| sigma_y (nats) | 2.9165 |
| mu_m, alpha0 | 0.9662, -2.5326 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 1.110e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +2.2619 | -1.3352 | +0.9266 |
| -0.700 | +1.5697 | -0.6430 | +0.9266 |
| -0.500 | +1.3126 | -0.3859 | +0.9266 |
| -0.300 | +1.1448 | -0.2182 | +0.9266 |
| -0.100 | +1.0083 | -0.0816 | +0.9266 |
| +0.000 | +0.9441 | -0.0175 | +0.9266 |
| +0.100 | +0.8800 | +0.0467 | +0.9266 |
| +0.300 | +0.7434 | +0.1832 | +0.9266 |
| +0.500 | +0.5756 | +0.3510 | +0.9266 |
| +0.700 | +0.3185 | +0.6081 | +0.9266 |
| +0.900 | -0.3736 | +1.3003 | +0.9266 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1048, min 4.507e-07, median 2.982e-06, max 5.783e-05 |
| clean arm | n 524, min 4.507e-07, median 1.464e-06, max 1.395e-05 |
| hinted arm | n 524, min 1.591e-06, median 4.958e-06, max 5.783e-05 |
| below 0.01 | 1048/1048 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.2949, NIE -0.0055, TE +0.2894.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/aqua_rat/stated-hint

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, stated-hint |
| text-level job | 827392 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 897/897 |
| records with no sidecar entry | 0/897 |
| keying mismatches | 0 |
| items entering the logit fit | 869 of 897 records |
| rows | 1738 |
| drops by reason | {'hinted_curve_unscorable': 28, 'items_dropped': 28} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`189ab78220fe`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1707 [+0.1483, +0.1931] | +0.0463 [+0.0205, +0.0709] |
| NIE | +0.0139 [+0.0059, +0.0226] | +0.0057 [-0.0117, +0.0267] |
| TE | +0.1846 [+0.1609, +0.2129] | +0.0521 [+0.0381, +0.0663] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1853 | +0.0521 |
| model-implied minus randomized | -0.0007 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 8.3044 |
| hinted | not printed on that scale | 8.7684 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 703/869 | 0.8090 | 0.0000 | 0.1910 | {'text_level_answer_unscorable': 0} |
| hinted | 597/869 | 0.6870 | 0.1853 | 0.1899 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0752 |
| logit level | 0.1103 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4423 [+0.3380, +0.5356] |
| logit level | +0.0163 [+0.0010, +0.0724] |

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
| alpha (NDE) | +0.0463 |
| beta | -0.1344 |
| gamma | -0.0428 |
| sigma_m | 0.3546 |
| sigma_y (nats) | 2.9213 |
| mu_m, alpha0 | 0.4999, -2.4969 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 5.551e-17 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +0.7735 | -0.7215 | +0.0521 |
| -0.700 | +0.3916 | -0.3395 | +0.0521 |
| -0.500 | +0.2497 | -0.1976 | +0.0521 |
| -0.300 | +0.1571 | -0.1050 | +0.0521 |
| -0.100 | +0.0817 | -0.0297 | +0.0521 |
| +0.000 | +0.0463 | +0.0057 | +0.0521 |
| +0.100 | +0.0109 | +0.0411 | +0.0521 |
| +0.300 | -0.0644 | +0.1165 | +0.0521 |
| +0.500 | -0.1570 | +0.2091 | +0.0521 |
| +0.700 | -0.2989 | +0.3510 | +0.0521 |
| +0.900 | -0.6809 | +0.7329 | +0.0521 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1794, min 1.945e-07, median 1.331e-06, max 4.262e-05 |
| clean arm | n 897, min 1.945e-07, median 1.042e-06, max 2.782e-05 |
| hinted arm | n 897, min 3.398e-07, median 1.666e-06, max 4.262e-05 |
| below 0.01 | 1794/1794 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.0161, NIE +0.0020, TE +0.0181.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/aqua_rat/professor

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, professor |
| text-level job | 827409 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 899/899 |
| records with no sidecar entry | 0/899 |
| keying mismatches | 0 |
| items entering the logit fit | 881 of 899 records |
| rows | 1762 |
| drops by reason | {'hinted_curve_unscorable': 18, 'items_dropped': 18} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`4a5e3de3c2d8`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1515 [+0.1294, +0.1691] | -0.1206 [-0.1651, -0.0836] |
| NIE | +0.0159 [+0.0069, +0.0243] | -0.0052 [-0.0277, +0.0158] |
| TE | +0.1673 [+0.1429, +0.1832] | -0.1258 [-0.1535, -0.0971] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1680 | -0.1258 |
| model-implied minus randomized | -0.0007 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 8.2841 |
| hinted | not printed on that scale | 10.2986 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 706/881 | 0.8014 | 0.0000 | 0.1986 | {'text_level_answer_unscorable': 0} |
| hinted | 603/881 | 0.6844 | 0.1680 | 0.1998 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0949 |
| logit level | 0.0413 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.5388 [+0.4284, +0.6455] |
| logit level | +0.0140 [+0.0012, +0.0701] |

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
| alpha (NDE) | -0.1206 |
| beta | +0.1207 |
| gamma | -0.0430 |
| sigma_m | 0.3543 |
| sigma_y (nats) | 3.0479 |
| mu_m, alpha0 | 0.4989, -2.5947 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 5.551e-17 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +0.6437 | -0.7696 | -0.1258 |
| -0.700 | +0.2422 | -0.3681 | -0.1258 |
| -0.500 | +0.0931 | -0.2189 | -0.1258 |
| -0.300 | -0.0042 | -0.1216 | -0.1258 |
| -0.100 | -0.0834 | -0.0424 | -0.1258 |
| +0.000 | -0.1206 | -0.0052 | -0.1258 |
| +0.100 | -0.1578 | +0.0320 | -0.1258 |
| +0.300 | -0.2371 | +0.1112 | -0.1258 |
| +0.500 | -0.3344 | +0.2085 | -0.1258 |
| +0.700 | -0.4835 | +0.3577 | -0.1258 |
| +0.900 | -0.8850 | +0.7592 | -0.1258 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1798, min 1.945e-07, median 2.108e-06, max 5.081e-05 |
| clean arm | n 899, min 1.945e-07, median 1.048e-06, max 2.782e-05 |
| hinted arm | n 899, min 9.839e-07, median 3.479e-06, max 5.081e-05 |
| below 0.01 | 1798/1798 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0419, NIE -0.0018, TE -0.0437.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/aqua_rat/metadata

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, metadata |
| text-level job | 828503 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 361/361 |
| records with no sidecar entry | 0/361 |
| keying mismatches | 0 |
| items entering the logit fit | 351 of 361 records |
| rows | 702 |
| drops by reason | {'hinted_curve_unscorable': 10, 'items_dropped': 10} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`6ad6e6858108`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1589 [+0.1258, +0.1969] | +0.8656 [+0.8090, +0.9164] |
| NIE | +0.0116 [+0.0025, +0.0284] | -0.0208 [-0.0649, +0.0036] |
| TE | +0.1705 [+0.1359, +0.2074] | +0.8448 [+0.8050, +0.8809] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1709 | +0.8448 |
| model-implied minus randomized | -0.0004 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 8.3164 |
| hinted | not printed on that scale | 6.9248 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 284/351 | 0.8091 | 0.0000 | 0.1909 | {'text_level_answer_unscorable': 0} |
| hinted | 232/351 | 0.6610 | 0.1709 | 0.1909 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0679 |
| logit level | -0.0246 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4542 [+0.3207, +0.6744] |
| logit level | +0.0731 [+0.0021, +0.1825] |

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
| alpha (NDE) | +0.8656 |
| beta | +0.5729 |
| gamma | -0.0363 |
| sigma_m | 0.3525 |
| sigma_y (nats) | 2.7531 |
| mu_m, alpha0 | 0.5022, -2.8967 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 1.110e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +1.4507 | -0.6059 | +0.8448 |
| -0.700 | +1.1434 | -0.2985 | +0.8448 |
| -0.500 | +1.0292 | -0.1844 | +0.8448 |
| -0.300 | +0.9547 | -0.1099 | +0.8448 |
| -0.100 | +0.8941 | -0.0493 | +0.8448 |
| +0.000 | +0.8656 | -0.0208 | +0.8448 |
| +0.100 | +0.8371 | +0.0077 | +0.8448 |
| +0.300 | +0.7765 | +0.0683 | +0.8448 |
| +0.500 | +0.7020 | +0.1428 | +0.8448 |
| +0.700 | +0.5878 | +0.2570 | +0.8448 |
| +0.900 | +0.2805 | +0.5643 | +0.8448 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 722, min 1.967e-07, median 2.121e-06, max 2.852e-05 |
| clean arm | n 361, min 1.967e-07, median 9.655e-07, max 1.532e-05 |
| hinted arm | n 361, min 1.147e-06, median 3.591e-06, max 2.852e-05 |
| below 0.01 | 722/722 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.2997, NIE -0.0072, TE +0.2925.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### gemma-2-9b-it/aqua_rat/grader-code

| field | value |
|---|---|
| model | google/gemma-2-9b-it |
| hf revision | 11c9b309abf73637e4b6f9a3fa1e92e615547819 |
| substrate, cue family | aqua_rat, grader-code |
| text-level job | 828522 |
| logit pass job | 829926 |
| sidecar | present |
| sidecar entries keyed | 356/356 |
| records with no sidecar entry | 0/356 |
| keying mismatches | 0 |
| items entering the logit fit | 337 of 356 records |
| rows | 674 |
| drops by reason | {'hinted_curve_unscorable': 19, 'items_dropped': 19} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`9c0396407a78`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1128 [+0.0815, +0.1455] | +0.7835 [+0.7224, +0.8643] |
| NIE | +0.0092 [+0.0006, +0.0232] | +0.0004 [-0.0287, +0.0255] |
| TE | +0.1220 [+0.0862, +0.1574] | +0.7839 [+0.7352, +0.8394] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1217 | +0.7839 |
| model-implied minus randomized | +0.0003 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 8.1591 |
| hinted | not printed on that scale | 6.6455 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 275/337 | 0.8160 | 0.0000 | 0.1840 | {'text_level_answer_unscorable': 0} |
| hinted | 241/337 | 0.7151 | 0.1217 | 0.1869 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0754 |
| logit level | 0.0005 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4965 [+0.2816, +0.7308] |
| logit level | +0.0016 [+0.0027, +0.1057] |

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
| alpha (NDE) | +0.7835 |
| beta | -0.0122 |
| gamma | -0.0331 |
| sigma_m | 0.3517 |
| sigma_y (nats) | 2.7207 |
| mu_m, alpha0 | 0.5041, -2.6124 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 0.000e+00 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +1.3120 | -0.5281 | +0.7839 |
| -0.700 | +1.0344 | -0.2505 | +0.7839 |
| -0.500 | +0.9313 | -0.1474 | +0.7839 |
| -0.300 | +0.8640 | -0.0801 | +0.7839 |
| -0.100 | +0.8093 | -0.0253 | +0.7839 |
| +0.000 | +0.7835 | +0.0004 | +0.7839 |
| +0.100 | +0.7578 | +0.0261 | +0.7839 |
| +0.300 | +0.7030 | +0.0809 | +0.7839 |
| +0.500 | +0.6358 | +0.1482 | +0.7839 |
| +0.700 | +0.5326 | +0.2513 | +0.7839 |
| +0.900 | +0.2551 | +0.5289 | +0.7839 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 712, min 1.967e-07, median 3.118e-06, max 8.184e-05 |
| clean arm | n 356, min 1.967e-07, median 9.505e-07, max 1.796e-05 |
| hinted arm | n 356, min 1.603e-06, median 7.607e-06, max 8.184e-05 |
| below 0.01 | 712/712 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE +0.2739, NIE +0.0001, TE +0.2740.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/arc_challenge/stated-hint

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, stated-hint |
| text-level job | 826738 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 1321/1321 |
| records with no sidecar entry | 0/1321 |
| keying mismatches | 0 |
| items entering the logit fit | 1321 of 1321 records |
| rows | 2642 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`84cd5e6ce507`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1616 [+0.1435, +0.1787] | -0.3744 [-0.5216, -0.2532] |
| NIE | +0.1781 [+0.1583, +0.2004] | -0.1020 [-0.2086, +0.0273] |
| TE | +0.3398 [+0.3092, +0.3746] | -0.4764 [-0.5409, -0.4284] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.3391 | -0.4764 |
| model-implied minus randomized | +0.0006 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 15.3899 |
| hinted | not printed on that scale | 20.1744 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 978/1321 | 0.7403 | 0.0000 | 0.2597 | {'text_level_answer_unscorable': 0} |
| hinted | 722/1321 | 0.5466 | 0.3391 | 0.2566 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.5243 |
| logit level | 0.2140 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.7684 [+0.7451, +0.7991] |
| logit level | +0.0323 [+0.0023, +0.0666] |

**6. The decision quantity on each scale.**

| scale | value |
|---|---|
| text level, rho*_decision | -0.1000 |
| text level, verdict | load-bearing at rho=0 |
| logit level | not applicable, no threshold pre-registered on this scale |

A5.6 sets no load-bearing threshold on this scale, so the row is descriptive
throughout and is never used for promotion, for ranking, for the element 21
comparison of section 22, or for any claim-status change.

**The fit, the sweep and the cross-check.**

| field | value |
|---|---|
| alpha (NDE) | -0.3744 |
| beta | +0.4729 |
| gamma | -0.2156 |
| sigma_m | 0.2883 |
| sigma_y (nats) | 4.2147 |
| mu_m, alpha0 | 0.9403, -3.0847 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 3.331e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +6.1336 | -6.6100 | -0.4764 |
| -0.700 | +2.7151 | -3.1915 | -0.4764 |
| -0.500 | +1.4453 | -1.9218 | -0.4764 |
| -0.300 | +0.6168 | -1.0932 | -0.4764 |
| -0.100 | -0.0577 | -0.4188 | -0.4764 |
| +0.000 | -0.3744 | -0.1020 | -0.4764 |
| +0.100 | -0.6912 | +0.2148 | -0.4764 |
| +0.300 | -1.3657 | +0.8893 | -0.4764 |
| +0.500 | -2.1942 | +1.7178 | -0.4764 |
| +0.700 | -3.4640 | +2.9876 | -0.4764 |
| +0.900 | -6.8825 | +6.4060 | -0.4764 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2642, min 1.186e-08, median 1.228e-06, max 1.825e-03 |
| clean arm | n 1321, min 1.186e-08, median 5.194e-07, max 6.906e-04 |
| hinted arm | n 1321, min 1.131e-07, median 2.609e-06, max 1.825e-03 |
| below 0.01 | 2642/2642 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0954, NIE -0.0260, TE -0.1214.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/arc_challenge/professor

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, professor |
| text-level job | 827285 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 1321/1321 |
| records with no sidecar entry | 0/1321 |
| keying mismatches | 0 |
| items entering the logit fit | 1321 of 1321 records |
| rows | 2642 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`bc7790c1c30d`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1411 [+0.1250, +0.1596] | -0.4798 [-0.6110, -0.3299] |
| NIE | +0.1356 [+0.1181, +0.1562] | -0.0419 [-0.1789, +0.0746] |
| TE | +0.2768 [+0.2473, +0.3050] | -0.5218 [-0.5871, -0.4627] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.2907 | -0.5218 |
| model-implied minus randomized | -0.0139 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 15.3899 |
| hinted | not printed on that scale | 19.7954 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 978/1321 | 0.7403 | 0.0000 | 0.2597 | {'text_level_answer_unscorable': 0} |
| hinted | 765/1321 | 0.5791 | 0.2907 | 0.2544 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.4900 |
| logit level | 0.0804 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.7253 [+0.6988, +0.7559] |
| logit level | +0.0150 [+0.0005, +0.0669] |

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
| alpha (NDE) | -0.4798 |
| beta | +0.2262 |
| gamma | -0.1854 |
| sigma_m | 0.2785 |
| sigma_y (nats) | 4.1939 |
| mu_m, alpha0 | 0.9403, -2.8527 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 3.331e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +5.2832 | -5.8050 | -0.5218 |
| -0.700 | +2.2561 | -2.7778 | -0.5218 |
| -0.500 | +1.1316 | -1.6534 | -0.5218 |
| -0.300 | +0.3979 | -0.9197 | -0.5218 |
| -0.100 | -0.1993 | -0.3225 | -0.5218 |
| +0.000 | -0.4798 | -0.0419 | -0.5218 |
| +0.100 | -0.7604 | +0.2386 | -0.5218 |
| +0.300 | -1.3576 | +0.8358 | -0.5218 |
| +0.500 | -2.0913 | +1.5695 | -0.5218 |
| +0.700 | -3.2157 | +2.6940 | -0.5218 |
| +0.900 | -6.2429 | +5.7211 | -0.5218 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2642, min 1.186e-08, median 6.993e-06, max 1.437e-02 |
| clean arm | n 1321, min 1.186e-08, median 5.194e-07, max 6.906e-04 |
| hinted arm | n 1321, min 6.785e-07, median 7.308e-05, max 1.437e-02 |
| below 0.01 | 2641/2642 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.1223, NIE -0.0107, TE -0.1330.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/arc_challenge/metadata

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, metadata |
| text-level job | 827097 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 503/503 |
| records with no sidecar entry | 0/503 |
| keying mismatches | 0 |
| items entering the logit fit | 503 of 503 records |
| rows | 1006 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`f93e473cb1bc`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0210 [+0.0072, +0.0342] | -0.1644 [-0.2318, -0.0970] |
| NIE | +0.0002 [-0.0017, +0.0017] | -0.0004 [-0.0143, +0.0106] |
| TE | +0.0211 [+0.0072, +0.0341] | -0.1648 [-0.2275, -0.1015] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0278 | -0.1648 |
| model-implied minus randomized | -0.0067 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 15.6484 |
| hinted | not printed on that scale | 17.0697 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 372/503 | 0.7396 | 0.0000 | 0.2604 | {'text_level_answer_unscorable': 0} |
| hinted | 361/503 | 0.7177 | 0.0278 | 0.2545 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0080 |
| logit level | 0.0024 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4111 [+0.3105, +0.5110] |
| logit level | +0.0121 [+0.0011, +0.0816] |

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
| alpha (NDE) | -0.1644 |
| beta | +0.2663 |
| gamma | -0.0015 |
| sigma_m | 0.1842 |
| sigma_y (nats) | 4.0443 |
| mu_m, alpha0 | 0.9383, -2.8328 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 2.776e-17 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | -0.0968 | -0.0680 | -0.1648 |
| -0.700 | -0.1323 | -0.0325 | -0.1648 |
| -0.500 | -0.1455 | -0.0193 | -0.1648 |
| -0.300 | -0.1541 | -0.0107 | -0.1648 |
| -0.100 | -0.1611 | -0.0037 | -0.1648 |
| +0.000 | -0.1644 | -0.0004 | -0.1648 |
| +0.100 | -0.1677 | +0.0029 | -0.1648 |
| +0.300 | -0.1747 | +0.0099 | -0.1648 |
| +0.500 | -0.1833 | +0.0185 | -0.1648 |
| +0.700 | -0.1964 | +0.0317 | -0.1648 |
| +0.900 | -0.2320 | +0.0672 | -0.1648 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1006, min 1.186e-08, median 3.293e-07, max 6.906e-04 |
| clean arm | n 503, min 1.186e-08, median 5.424e-07, max 6.906e-04 |
| hinted arm | n 503, min 1.880e-08, median 2.162e-07, max 3.117e-04 |
| below 0.01 | 1006/1006 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0415, NIE -0.0001, TE -0.0416.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/arc_challenge/grader-code

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | arc_challenge, grader-code |
| text-level job | 827294 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 503/503 |
| records with no sidecar entry | 0/503 |
| keying mismatches | 0 |
| items entering the logit fit | 503 of 503 records |
| rows | 1006 |
| drops by reason | none |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`4f236bc9448d`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0290 [+0.0154, +0.0441] | -0.1763 [-0.2765, -0.0744] |
| NIE | +0.0072 [+0.0034, +0.0116] | +0.0109 [-0.0457, +0.0774] |
| TE | +0.0362 [+0.0196, +0.0556] | -0.1654 [-0.2619, -0.0738] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0517 | -0.1654 |
| model-implied minus randomized | -0.0155 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 15.6484 |
| hinted | not printed on that scale | 19.8364 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 372/503 | 0.7396 | 0.0000 | 0.2604 | {'text_level_answer_unscorable': 0} |
| hinted | 355/503 | 0.7058 | 0.0517 | 0.2584 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.1979 |
| logit level | -0.0660 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4731 [+0.3929, +0.5462] |
| logit level | +0.0124 [+0.0008, +0.0895] |

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
| alpha (NDE) | -0.1763 |
| beta | -0.2395 |
| gamma | -0.0456 |
| sigma_m | 0.2189 |
| sigma_y (nats) | 4.2118 |
| mu_m, alpha0 | 0.9383, -2.3581 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 1.665e-16 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | +1.6353 | -1.8007 | -0.1654 |
| -0.700 | +0.6837 | -0.8491 | -0.1654 |
| -0.500 | +0.3303 | -0.4956 | -0.1654 |
| -0.300 | +0.0996 | -0.2650 | -0.1654 |
| -0.100 | -0.0881 | -0.0773 | -0.1654 |
| +0.000 | -0.1763 | +0.0109 | -0.1654 |
| +0.100 | -0.2645 | +0.0991 | -0.1654 |
| +0.300 | -0.4522 | +0.2868 | -0.1654 |
| +0.500 | -0.6829 | +0.5175 | -0.1654 |
| +0.700 | -1.0363 | +0.8709 | -0.1654 |
| +0.900 | -1.9879 | +1.8225 | -0.1654 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 1006, min 1.186e-08, median 1.585e-06, max 6.906e-04 |
| clean arm | n 503, min 1.186e-08, median 5.424e-07, max 6.906e-04 |
| hinted arm | n 503, min 1.873e-07, median 4.179e-06, max 5.840e-04 |
| below 0.01 | 1006/1006 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0445, NIE +0.0028, TE -0.0418.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/aqua_rat/stated-hint

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, stated-hint |
| text-level job | 827393 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 1012/1012 |
| records with no sidecar entry | 0/1012 |
| keying mismatches | 0 |
| items entering the logit fit | 1006 of 1012 records |
| rows | 2012 |
| drops by reason | {'hinted_curve_unscorable': 6, 'items_dropped': 6} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`1cfcacf99a9e`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0971 [+0.0807, +0.1159] | -0.2534 [-0.2945, -0.2167] |
| NIE | -0.0016 [-0.0034, +0.0006] | +0.0010 [-0.0053, +0.0124] |
| TE | +0.0955 [+0.0796, +0.1144] | -0.2523 [-0.2904, -0.2200] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0954 | -0.2523 |
| model-implied minus randomized | +0.0001 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 12.0239 |
| hinted | not printed on that scale | 13.0278 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 812/1006 | 0.8072 | 0.0000 | 0.1928 | {'text_level_answer_unscorable': 0} |
| hinted | 746/1006 | 0.7416 | 0.0954 | 0.1928 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | -0.0166 |
| logit level | -0.0041 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.3078 [+0.2035, +0.4089] |
| logit level | +0.0098 [+0.0009, +0.0692] |

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
| alpha (NDE) | -0.2534 |
| beta | +0.0995 |
| gamma | +0.0105 |
| sigma_m | 0.3471 |
| sigma_y (nats) | 3.5390 |
| mu_m, alpha0 | 0.5788, -3.2158 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 0.000e+00 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | -0.4738 | +0.2215 | -0.2523 |
| -0.700 | -0.3580 | +0.1057 | -0.2523 |
| -0.500 | -0.3150 | +0.0627 | -0.2523 |
| -0.300 | -0.2869 | +0.0346 | -0.2523 |
| -0.100 | -0.2641 | +0.0118 | -0.2523 |
| +0.000 | -0.2534 | +0.0010 | -0.2523 |
| +0.100 | -0.2426 | -0.0097 | -0.2523 |
| +0.300 | -0.2198 | -0.0325 | -0.2523 |
| +0.500 | -0.1917 | -0.0606 | -0.2523 |
| +0.700 | -0.1487 | -0.1036 | -0.2523 |
| +0.900 | -0.0330 | -0.2194 | -0.2523 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2024, min 5.679e-09, median 7.361e-07, max 1.677e-03 |
| clean arm | n 1012, min 5.679e-09, median 4.465e-07, max 8.252e-04 |
| hinted arm | n 1012, min 7.415e-08, median 1.172e-06, max 1.677e-03 |
| below 0.01 | 2024/2024 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0730, NIE +0.0003, TE -0.0727.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/aqua_rat/professor

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, professor |
| text-level job | 827410 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 1012/1012 |
| records with no sidecar entry | 0/1012 |
| keying mismatches | 0 |
| items entering the logit fit | 1009 of 1012 records |
| rows | 2018 |
| drops by reason | {'hinted_curve_unscorable': 3, 'items_dropped': 3} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`4c3a17f1513b`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.1651 [+0.1433, +0.1821] | -0.3869 [-0.4299, -0.3466] |
| NIE | +0.0012 [-0.0038, +0.0075] | -0.0002 [-0.0077, +0.0049] |
| TE | +0.1663 [+0.1446, +0.1868] | -0.3871 [-0.4298, -0.3467] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.1675 | -0.3871 |
| model-implied minus randomized | -0.0012 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 11.9861 |
| hinted | not printed on that scale | 13.5378 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 815/1009 | 0.8077 | 0.0000 | 0.1923 | {'text_level_answer_unscorable': 0} |
| hinted | 703/1009 | 0.6967 | 0.1675 | 0.1933 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0069 |
| logit level | 0.0005 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.4289 [+0.3453, +0.5132] |
| logit level | +0.0046 [+0.0009, +0.0639] |

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
| alpha (NDE) | -0.3869 |
| beta | +0.0464 |
| gamma | -0.0038 |
| sigma_m | 0.3557 |
| sigma_y (nats) | 3.5724 |
| mu_m, alpha0 | 0.5795, -3.1839 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 5.551e-17 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | -0.3071 | -0.0800 | -0.3871 |
| -0.700 | -0.3490 | -0.0381 | -0.3871 |
| -0.500 | -0.3646 | -0.0225 | -0.3871 |
| -0.300 | -0.3747 | -0.0123 | -0.3871 |
| -0.100 | -0.3830 | -0.0041 | -0.3871 |
| +0.000 | -0.3869 | -0.0002 | -0.3871 |
| +0.100 | -0.3908 | +0.0037 | -0.3871 |
| +0.300 | -0.3991 | +0.0120 | -0.3871 |
| +0.500 | -0.4092 | +0.0221 | -0.3871 |
| +0.700 | -0.4248 | +0.0377 | -0.3871 |
| +0.900 | -0.4667 | +0.0796 | -0.3871 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 2024, min 5.679e-09, median 3.287e-06, max 2.296e-02 |
| clean arm | n 1012, min 5.679e-09, median 4.465e-07, max 8.252e-04 |
| hinted arm | n 1012, min 4.692e-07, median 1.168e-05, max 2.296e-02 |
| below 0.01 | 2022/2024 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.1117, NIE -0.0001, TE -0.1117.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/aqua_rat/metadata

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, metadata |
| text-level job | 828504 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 370/370 |
| records with no sidecar entry | 0/370 |
| keying mismatches | 0 |
| items entering the logit fit | 368 of 370 records |
| rows | 736 |
| drops by reason | {'hinted_curve_unscorable': 2, 'items_dropped': 2} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`a004d3fee59b`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0543 [+0.0328, +0.0814] | -0.1894 [-0.2860, -0.1078] |
| NIE | +0.0001 [-0.0012, +0.0024] | +0.0089 [-0.0106, +0.0429] |
| TE | +0.0543 [+0.0328, +0.0814] | -0.1805 [-0.2687, -0.0985] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0543 | -0.1805 |
| model-implied minus randomized | +0.0000 | +0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 12.3294 |
| hinted | not printed on that scale | 13.2318 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 296/368 | 0.8043 | 0.0000 | 0.1957 | {'text_level_answer_unscorable': 0} |
| hinted | 282/368 | 0.7663 | 0.0543 | 0.1902 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | 0.0017 |
| logit level | -0.0492 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.0147 [+0.0027, +0.2643] |
| logit level | +0.0427 [+0.0033, +0.1497] |

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
| alpha (NDE) | -0.1894 |
| beta | +0.4461 |
| gamma | +0.0199 |
| sigma_m | 0.3421 |
| sigma_y (nats) | 3.5717 |
| mu_m, alpha0 | 0.5494, -3.3568 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 2.776e-17 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | -0.6190 | +0.4384 | -0.1805 |
| -0.700 | -0.3933 | +0.2128 | -0.1805 |
| -0.500 | -0.3095 | +0.1290 | -0.1805 |
| -0.300 | -0.2549 | +0.0743 | -0.1805 |
| -0.100 | -0.2103 | +0.0298 | -0.1805 |
| +0.000 | -0.1894 | +0.0089 | -0.1805 |
| +0.100 | -0.1685 | -0.0120 | -0.1805 |
| +0.300 | -0.1240 | -0.0565 | -0.1805 |
| +0.500 | -0.0693 | -0.1112 | -0.1805 |
| +0.700 | +0.0145 | -0.1950 | -0.1805 |
| +0.900 | +0.2401 | -0.4207 | -0.1805 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 740, min 5.679e-09, median 5.217e-07, max 5.735e-04 |
| clean arm | n 370, min 5.679e-09, median 5.030e-07, max 5.735e-04 |
| hinted arm | n 370, min 4.735e-08, median 5.539e-07, max 2.201e-04 |
| below 0.01 | 740/740 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.0539, NIE +0.0025, TE -0.0513.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

### llama-3.1-8b-instruct/aqua_rat/grader-code

| field | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| hf revision | 0e9e39f249a16976918f6564b8830bc894c89659 |
| substrate, cue family | aqua_rat, grader-code |
| text-level job | 828523 |
| logit pass job | 829960 |
| sidecar | present |
| sidecar entries keyed | 370/370 |
| records with no sidecar entry | 0/370 |
| keying mismatches | 0 |
| items entering the logit fit | 368 of 370 records |
| rows | 736 |
| drops by reason | {'hinted_curve_unscorable': 2, 'items_dropped': 2} |

The two scales in A5.5's order. The text-level row comes from the 24-cell
`fit.json` unchanged (`71b419ab6a7c`) and is never recomputed here.

**1 and 2. The three effects on each scale.**

| effect | text level, probability | logit level, nats |
|---|---|---|
| NDE | +0.0494 [+0.0328, +0.0706] | -0.3546 [-0.4850, -0.2341] |
| NIE | -0.0005 [-0.0019, +0.0014] | +0.0051 [-0.0154, +0.0419] |
| TE | +0.0489 [+0.0325, +0.0707] | -0.3495 [-0.4847, -0.2444] |

They are two different quantities on two different scales, not two estimates of
one number. The total effects in particular are two different total effects: on
the logit scale TE equals the randomized arm difference in the margin as an
algebraic identity, and on the text scale the model-implied TE is checked against
the randomized arm difference in the follow rate and that check can fail.

| total effect check | text level | logit level |
|---|---|---|
| randomized arm difference | +0.0489 | -0.3495 |
| model-implied minus randomized | +0.0000 | -0.000000000000 |

**Outcome variance per arm, which is the number the outcome-scale note is about.**

| arm | text level | logit level, nats squared |
|---|---|---|
| clean | 0.0000 | 12.4111 |
| hinted | not printed on that scale | 16.1607 |

A5.4 condition 4 is the check the text-level scale cannot pass by construction:
the population is the clean-correct subpopulation and the hint label is a planted
wrong option, so the binary clean arm has no variation at all.

**3. The bridge (A5.3), with its denominator and its drop counts.**

| arm | agreement | rate | follow rate | share with margin above zero | drops |
|---|---|---|---|---|---|
| clean | 295/368 | 0.8016 | 0.0000 | 0.1984 | {'text_level_answer_unscorable': 0} |
| hinted | 287/368 | 0.7799 | 0.0489 | 0.1929 | {'text_level_answer_unscorable': 0} |

An agreement rate, not a validation of either scale. The two disagree exactly
where the parsed answer is not the argmax of the renormalized letter
distribution, which is a real quantity about the read.

**4. The mediated shares, text level first, each printed only where that scale's
own TE interval excludes zero.**

| scale | mediated share |
|---|---|
| text level | -0.0102 |
| logit level | -0.0145 |

**5. `rho*_point` on each scale, an invariant reference with no directional
meaning.** It contains neither the direct coefficient nor either intercept, so it
says nothing about direct-path strength.

| scale | rho*_point |
|---|---|
| text level | +0.0967 [+0.0042, +0.3478] |
| logit level | +0.0265 [+0.0025, +0.1210] |

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
| alpha (NDE) | -0.3546 |
| beta | +0.2913 |
| gamma | +0.0173 |
| sigma_m | 0.3433 |
| sigma_y (nats) | 3.7783 |
| mu_m, alpha0 | 0.5501, -3.2581 |
| bootstrap | 200 replicates, unit item (both arms of an item resampled together), seed 20260907 |
| rho grid | 379 points, -0.945 to +0.945 |
| refit sweep against the vectorised curve | max abs difference in NIE 0.000e+00 |
| TE range across the check rhos | 0.000e+00 |

The effects curve across the symmetric grid, at the printed rhos:

| rho | NDE | NIE | TE |
|---|---|---|---|
| -0.900 | -0.7488 | +0.3992 | -0.3495 |
| -0.700 | -0.5417 | +0.1922 | -0.3495 |
| -0.500 | -0.4648 | +0.1153 | -0.3495 |
| -0.300 | -0.4146 | +0.0651 | -0.3495 |
| -0.100 | -0.3738 | +0.0242 | -0.3495 |
| +0.000 | -0.3546 | +0.0051 | -0.3495 |
| +0.100 | -0.3354 | -0.0141 | -0.3495 |
| +0.300 | -0.2945 | -0.0550 | -0.3495 |
| +0.500 | -0.2444 | -0.1052 | -0.3495 |
| +0.700 | -0.1674 | -0.1821 | -0.3495 |
| +0.900 | +0.0396 | -0.3891 | -0.3495 |

**A5.4 condition 6. The raw letter probability mass behind every margin in this
row, before renormalization.**

| reads | summary |
|---|---|
| pooled | n 740, min 5.679e-09, median 1.623e-06, max 3.083e-03 |
| clean arm | n 370, min 5.679e-09, median 5.030e-07, max 5.735e-04 |
| hinted arm | n 370, min 4.836e-07, median 3.665e-06, max 3.083e-03 |
| below 0.01 | 740/740 |

No floor is set. A5.4 leaves that to the operator and flags no row without one.

**Standardised effects, a reporting convenience and not an estimand.** In
clean-arm outcome standard deviations: NDE -0.1005, NIE +0.0014, TE -0.0991.
The 0.15 of section 2.5 is on the probability scale and does not transfer here.

## 3. What this establishes, and what it does not

**What it establishes.**

1. Gate G1's condition 3 is satisfiable. Before the logit pass, 18 of 18 cells
   failed it on the same line: no arms record in this project carried
   `intervention_level = logit`, because the generation pass that writes it had
   never been run. 24 cells now carry it on every record, keyed to the
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
3. **No cross-model comparison.** Rows exist for 3 model families: gemma-2-9b-it, llama-3.1-8b-instruct, qwen3-8b.
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

