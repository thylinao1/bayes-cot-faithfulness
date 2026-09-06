# CPU mechanism battery (Amendment A2, element 11(a))

Generated 2026-09-06T22:34:48Z by `experiments/mechanism_battery.py` in 6354.4 seconds of wall clock on CPU. Reproduce with

```
PYTHONPATH=src python experiments/mechanism_battery.py --out experiments/results/mechanism_battery
```

11 generator families, evaluated as 15 conditions (family 2 carries five bypass strengths), at n = 350 and 3600 rows per dataset, at least 100 seeded datasets per condition per sample size (the element 12 part (iv) families carry 400 each), 3800 datasets in total, each fitted once at rho = 0 plus 200 bootstrap refits. Every dataset is drawn from a seed fixed before the run (dataset seed base 731000, bootstrap seed base 909000); no result below is a re-run after seeing a number.

## What was fitted, and where each interval comes from

The estimator under test is the repaired MAP fit, `fit_probit_mediation_map(..., rho=0, intercepts=True)`, converted to probability-scale natural effects by the repository's exact closed form. **Intervals in the main tables are nonparametric row-bootstrap percentile intervals with 200 replicates**, each replicate refitted from scratch. **A separate subset of 20 datasets per family at n = 350 is re-run through the repaired PyMC posterior** (`fit_mediation_model(..., intercepts=True)` plus `posterior_natural_effects` with the intercept draws) and reported in its own section, so the cheap interval is compared against the expensive one rather than assumed equal to it.

The sensitivity sweep uses the reparameterisation documented in `sensitivity`: the mediator equation does not move with the assumed rho, so one fit at rho = 0 determines the whole curve. It is evaluated on a grid from 0 to 0.947 in steps of 0.005 and cross-checked below against the repository's refit-per-rho `sensitivity_sweep` and `breakdown_frontier`.

Two rho quantities are reported and never merged. **rho\*_point** is the zero crossing of the point-estimate NIE, which is a function of the fitted mediator coefficient and sigma_m alone and therefore carries no information about the direct path. **rho\*_decision** is the rho at which the pre-registered verdict rule fails: a dataset is load-bearing at rho when the probability that the NIE exceeds 0.15 is at least 0.95, evaluated on the bootstrap distribution. Where the verdict already fails at rho = 0 the verdict is **unresolved** and no crossing is reported; where it never fails inside the evaluated range the report says so rather than inventing a crossing beyond 0.947.

The **false-robust-verdict rate** is the share of datasets where the verdict fires at rho = 0 in a family whose true NIE is exactly zero.

## Required outcomes

- **PASS R1 direct bypass: mediated share falls, rho*_point does not move.** At n = 3600, across the 5 bypass strengths the estimated mediated share runs 101.2% to 1.6% (monotonically falling: True), while the mean rho*_point runs 0.6242 to 0.6252, a spread of 0.0011 against a tolerance of 0.01. Each rho*_point is a mean over 100 seeded datasets; each share is a mean over the 100, 100, 100, 100, 100 datasets per condition whose total-effect interval excludes zero.
- **PASS R2 shared cause: strong association, zero true mediation, crossing near 0.707.** True NIE = 0.0000 by construction; the estimator reports a mean NIE of 0.2613 (bias 0.2613 +/- 0.0010) over 100 datasets at n = 3600, with mean corr(M, Y) = 0.5848; mean rho*_point = 0.7069 against the analytic 1/sqrt(2) = 0.7071 (tolerance 0.02).
- **PASS R3 no cue effect: intervals cover zero.** n = 350 NDE covers zero in 94/100 datasets [0.874, 0.978]; n = 350 NIE covers zero in 100/100 datasets [0.964, 1.000]; n = 350 TE covers zero in 94/100 datasets [0.874, 0.978]; n = 3600 NDE covers zero in 93/100 datasets [0.861, 0.971]; n = 3600 NIE covers zero in 100/100 datasets [0.964, 1.000]; n = 3600 TE covers zero in 92/100 datasets [0.848, 0.965]. Criterion: the binomial interval on coverage reaches the nominal 0.95, that is, coverage is not significantly below nominal.

## Cross-checks against the repository's own slower code paths

1. The vectorised rho curve against `sensitivity_sweep`, which refits the probit model at every grid point and integrates by Monte Carlo: max absolute NIE difference 0.00335, mean 0.00165, over 33 datasets at 6 rho values each (198 comparisons) at n = 3600.
2. The analytic `rho*_point` against `breakdown_frontier`, which root-finds on refits: max absolute difference 0.00096, mean 0.00009, over 33 datasets that had a crossing inside the evaluated range.
3. Each family's Monte Carlo truth (2,000,000 rows, common noise across the three cross-world cells) against its hand-derived analytic truth: max absolute difference 0.00077 over 15 conditions. The analytic value is what the bias and coverage columns are computed against.

## The comparison no single family can make

Family 3 has no mediation at all and family 4 is fully mediated. At n = 3600, averaged over 100 and 100 datasets, this is what the instrument reports for each.

| condition | true NIE | mean NDE | mean NIE | mean TE | rho*_point [mean interval] | median rho*_decision | load-bearing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f3_shared_cause | 0.000 | -0.0023 | 0.2613 | 0.2590 | 0.7069 [0.683, 0.731] | 0.335 | 100/100 |
| f4_rationalization | 0.283 | -0.0011 | 0.2833 | 0.2823 | 0.7063 [0.681, 0.732] | 0.345 | 100/100 |
| f5_redundant_explanation | 0.000 | 0.0381 | 0.2503 | 0.2884 | 0.8428 [0.826, 0.859] | 0.420 | 100/100 |

The two reports are the same report. Both put essentially the whole total effect on the mediated path, both fire the verdict in nearly every dataset, and their rho*_point values differ by 0.0006 against a mean interval half-width of 0.0253, so the sensitivity summary separates them by nothing at all. Family 5 is worse than a tie: its rho*_point of 0.8428 and its median rho*_decision of 0.420 are the largest in the battery, and its true mediated effect is zero. Ranking cells by rho* would therefore put a world with no mediation above a world that is entirely mediated. What separates them is not a number this instrument produces from observational text: it is an assumption about rho, or an intervention, and the report says so wherever a rho quantity appears.

## Truth per condition

| condition | analytic NDE / NIE / TE | Monte Carlo NDE / NIE / TE | max abs diff |
| --- | --- | --- | --- |
| f1_no_cue_effect | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.00000 |
| f2_direct_bypass_alpha0 | 0.0000 / 0.2339 / 0.2339 | 0.0000 / 0.2340 / 0.2340 | 0.00013 |
| f2_direct_bypass_alpha0.5 | 0.1519 / 0.1931 / 0.3450 | 0.1519 / 0.1933 / 0.3453 | 0.00027 |
| f2_direct_bypass_alpha1 | 0.2826 / 0.1375 / 0.4201 | 0.2827 / 0.1377 / 0.4204 | 0.00032 |
| f2_direct_bypass_alpha2 | 0.4408 / 0.0448 / 0.4856 | 0.4412 / 0.0448 / 0.4860 | 0.00041 |
| f2_direct_bypass_alpha3 | 0.4904 / 0.0081 / 0.4985 | 0.4908 / 0.0081 / 0.4989 | 0.00043 |
| f3_shared_cause | 0.2602 / 0.0000 / 0.2602 | 0.2605 / 0.0000 / 0.2605 | 0.00029 |
| f4_rationalization | 0.0000 / 0.2827 / 0.2827 | 0.0000 / 0.2833 / 0.2833 | 0.00065 |
| f5_redundant_explanation | 0.2874 / 0.0000 / 0.2874 | 0.2876 / 0.0000 / 0.2876 | 0.00011 |
| f6_answer_copying | 0.3413 / 0.0000 / 0.3413 | 0.3416 / 0.0000 / 0.3416 | 0.00026 |
| f7_opposing_effects | -0.2423 / 0.2423 / 0.0000 | -0.2420 / 0.2420 / 0.0000 | 0.00039 |
| f8_nonlinear_depth | 0.0000 / 0.2803 / 0.2803 | 0.0000 / 0.2810 / 0.2810 | 0.00077 |
| f9_varying_variance | 0.0000 / 0.1836 / 0.1836 | 0.0000 / 0.1837 / 0.1837 | 0.00010 |
| f10_sparse_groups | 0.0000 / 0.2566 / 0.2566 | 0.0000 / 0.2559 / 0.2559 | 0.00064 |
| f11_mediator_missingness | 0.0000 / 0.2827 / 0.2827 | 0.0000 / 0.2833 / 0.2833 | 0.00065 |

## Effects, bias and coverage at n = 350

| condition | truth NDE / NIE / TE | NDE bias | NIE bias | TE bias | NDE cov | NIE cov | TE cov | mean NIE width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_no_cue_effect | 0.000 / 0.000 / 0.000 | 0.0032 | 0.0003 | 0.0035 | 94/100 [0.87, 0.98] | 100/100 [0.96, 1.00] | 94/100 [0.87, 0.98] | 0.017 |
| f2_direct_bypass_alpha0 | 0.000 / 0.234 / 0.234 | -0.0035 | -0.0031 | -0.0066 | 91/100 [0.84, 0.96] | 93/100 [0.86, 0.97] | 94/100 [0.87, 0.98] | 0.121 |
| f2_direct_bypass_alpha0.5 | 0.152 / 0.193 / 0.345 | -0.0017 | -0.0039 | -0.0057 | 92/100 [0.85, 0.96] | 93/100 [0.86, 0.97] | 91/100 [0.84, 0.96] | 0.118 |
| f2_direct_bypass_alpha1 | 0.283 / 0.138 / 0.420 | -0.0067 | -0.0008 | -0.0075 | 94/100 [0.87, 0.98] | 94/100 [0.87, 0.98] | 95/100 [0.89, 0.98] | 0.111 |
| f2_direct_bypass_alpha2 | 0.441 / 0.045 / 0.486 | -0.0081 | -0.0003 | -0.0084 | 90/100 [0.82, 0.95] | 90/100 [0.82, 0.95] | 90/100 [0.82, 0.95] | 0.079 |
| f2_direct_bypass_alpha3 | 0.490 / 0.008 / 0.498 | -0.0059 | -0.0019 | -0.0079 | 91/100 [0.84, 0.96] | 26/100 [0.18, 0.36] | 91/100 [0.84, 0.96] | 0.015 |
| f3_shared_cause | 0.260 / 0.000 / 0.260 | -0.2625 | 0.2561 | -0.0064 | 0/100 [0.00, 0.04] | 0/100 [0.00, 0.04] | 92/100 [0.85, 0.96] | 0.125 |
| f4_rationalization | 0.000 / 0.283 / 0.283 | 0.0014 | -0.0059 | -0.0045 | 95/100 [0.89, 0.98] | 93/100 [0.86, 0.97] | 96/100 [0.90, 0.99] | 0.136 |
| f5_redundant_explanation | 0.287 / 0.000 / 0.287 | -0.2462 | 0.2438 | -0.0024 | 0/100 [0.00, 0.04] | 0/100 [0.00, 0.04] | 97/100 [0.91, 0.99] | 0.129 |
| f6_answer_copying | 0.341 / 0.000 / 0.341 | -0.0032 | 0.0012 | -0.0020 | 94/100 [0.87, 0.98] | 98/100 [0.93, 1.00] | 95/100 [0.89, 0.98] | 0.022 |
| f7_opposing_effects | -0.242 / 0.242 / 0.000 | 0.0007 | -0.0029 | -0.0023 | 90/100 [0.82, 0.95] | 92/100 [0.85, 0.96] | 89/100 [0.81, 0.94] | 0.115 |
| f8_nonlinear_depth | 0.000 / 0.280 / 0.280 | 0.0056 | -0.0156 | -0.0100 | 369/400 [0.89, 0.95] | 359/400 [0.86, 0.93] | 370/400 [0.89, 0.95] | 0.136 |
| f9_varying_variance | 0.000 / 0.184 / 0.184 | -0.0018 | 0.0653 | 0.0635 | 375/400 [0.91, 0.96] | 169/400 [0.37, 0.47] | 272/400 [0.63, 0.73] | 0.117 |
| f10_sparse_groups | 0.000 / 0.257 / 0.257 | 0.0020 | -0.0009 | 0.0011 | 382/400 [0.93, 0.97] | 362/400 [0.87, 0.93] | 377/400 [0.91, 0.96] | 0.132 |
| f11_mediator_missingness | 0.000 / 0.283 / 0.283 | -0.0009 | -0.0108 | -0.0117 | 375/400 [0.91, 0.96] | 367/400 [0.89, 0.94] | 358/400 [0.86, 0.92] | 0.150 |

## Verdicts and rho behaviour at n = 350

| condition | true NIE | load-bearing at rho=0 | false-robust | rho*_point mean [mean 95% interval] | rho*_decision |
| --- | --- | --- | --- | --- | --- |
| f1_no_cue_effect | 0.000 | 0/100 [0.00, 0.04] | yes 0.0% | 0.0597 [0.006, 0.204] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f2_direct_bypass_alpha0 | 0.234 | 83/100 [0.74, 0.90] | n/a | 0.6312 [0.539, 0.718] | unresolved 17/100, crossing 83/100 (median 0.150), none-in-range 0/100 |
| f2_direct_bypass_alpha0.5 | 0.193 | 37/100 [0.28, 0.47] | n/a | 0.6331 [0.536, 0.724] | unresolved 63/100, crossing 37/100 (median 0.055), none-in-range 0/100 |
| f2_direct_bypass_alpha1 | 0.138 | 0/100 [0.00, 0.04] | n/a | 0.6377 [0.536, 0.733] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f2_direct_bypass_alpha2 | 0.045 | 0/100 [0.00, 0.04] | n/a | 0.6391 [0.523, 0.747] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f2_direct_bypass_alpha3 | 0.008 | 0/100 [0.00, 0.04] | n/a | 0.6387 [0.518, 0.749] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f3_shared_cause | 0.000 | 95/100 [0.89, 0.98] | yes 95.0% | 0.7128 [0.634, 0.787] | unresolved 5/100, crossing 95/100 (median 0.235), none-in-range 0/100 |
| f4_rationalization | 0.283 | 97/100 [0.91, 0.99] | n/a | 0.7114 [0.628, 0.790] | unresolved 3/100, crossing 97/100 (median 0.250), none-in-range 0/100 |
| f5_redundant_explanation | 0.000 | 88/100 [0.80, 0.94] | yes 88.0% | 0.8468 [0.793, 0.897] | unresolved 12/100, crossing 88/100 (median 0.265), none-in-range 0/100 |
| f6_answer_copying | 0.000 | 0/100 [0.00, 0.04] | yes 0.0% | 0.0602 [0.006, 0.204] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f7_opposing_effects | 0.242 | 92/100 [0.85, 0.96] | n/a | 0.6298 [0.540, 0.714] | unresolved 8/100, crossing 92/100 (median 0.165), none-in-range 0/100 |
| f8_nonlinear_depth | 0.280 | 384/400 [0.94, 0.98] | n/a | 0.6642 [0.560, 0.766] | unresolved 16/400, crossing 384/400 (median 0.190), none-in-range 0/400 |
| f9_varying_variance | 0.184 | 383/400 [0.93, 0.98] | n/a | 0.8029 [0.728, 0.868] | unresolved 17/400, crossing 383/400 (median 0.275), none-in-range 0/400 |
| f10_sparse_groups | 0.257 | 366/400 [0.88, 0.94] | n/a | 0.6192 [0.521, 0.710] | unresolved 34/400, crossing 366/400 (median 0.160), none-in-range 0/400 |
| f11_mediator_missingness | 0.283 | 377/400 [0.91, 0.96] | n/a | 0.6871 [0.590, 0.777] | unresolved 23/400, crossing 377/400 (median 0.220), none-in-range 0/400 |

## Effects, bias and coverage at n = 3600

| condition | truth NDE / NIE / TE | NDE bias | NIE bias | TE bias | NDE cov | NIE cov | TE cov | mean NIE width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_no_cue_effect | 0.000 / 0.000 / 0.000 | -0.0015 | -0.0000 | -0.0015 | 93/100 [0.86, 0.97] | 100/100 [0.96, 1.00] | 92/100 [0.85, 0.96] | 0.002 |
| f2_direct_bypass_alpha0 | 0.000 / 0.234 / 0.234 | -0.0019 | 0.0011 | -0.0008 | 93/100 [0.86, 0.97] | 94/100 [0.87, 0.98] | 93/100 [0.86, 0.97] | 0.038 |
| f2_direct_bypass_alpha0.5 | 0.152 / 0.193 / 0.345 | -0.0013 | 0.0007 | -0.0007 | 92/100 [0.85, 0.96] | 94/100 [0.87, 0.98] | 94/100 [0.87, 0.98] | 0.037 |
| f2_direct_bypass_alpha1 | 0.283 / 0.138 / 0.420 | -0.0014 | 0.0006 | -0.0007 | 95/100 [0.89, 0.98] | 92/100 [0.85, 0.96] | 96/100 [0.90, 0.99] | 0.035 |
| f2_direct_bypass_alpha2 | 0.441 / 0.045 / 0.486 | -0.0022 | 0.0011 | -0.0010 | 88/100 [0.80, 0.94] | 93/100 [0.86, 0.97] | 93/100 [0.86, 0.97] | 0.026 |
| f2_direct_bypass_alpha3 | 0.490 / 0.008 / 0.498 | -0.0008 | 0.0001 | -0.0007 | 94/100 [0.87, 0.98] | 92/100 [0.85, 0.96] | 96/100 [0.90, 0.99] | 0.014 |
| f3_shared_cause | 0.260 / 0.000 / 0.260 | -0.2625 | 0.2613 | -0.0013 | 0/100 [0.00, 0.04] | 0/100 [0.00, 0.04] | 94/100 [0.87, 0.98] | 0.039 |
| f4_rationalization | 0.000 / 0.283 / 0.283 | -0.0011 | 0.0007 | -0.0004 | 93/100 [0.86, 0.97] | 93/100 [0.86, 0.97] | 96/100 [0.90, 0.99] | 0.042 |
| f5_redundant_explanation | 0.287 / 0.000 / 0.287 | -0.2493 | 0.2503 | 0.0010 | 0/100 [0.00, 0.04] | 0/100 [0.00, 0.04] | 93/100 [0.86, 0.97] | 0.040 |
| f6_answer_copying | 0.341 / 0.000 / 0.341 | -0.0016 | -0.0000 | -0.0016 | 94/100 [0.87, 0.98] | 88/100 [0.80, 0.94] | 95/100 [0.89, 0.98] | 0.006 |
| f7_opposing_effects | -0.242 / 0.242 / 0.000 | -0.0031 | 0.0007 | -0.0024 | 97/100 [0.91, 0.99] | 93/100 [0.86, 0.97] | 97/100 [0.91, 0.99] | 0.036 |

## Verdicts and rho behaviour at n = 3600

| condition | true NIE | load-bearing at rho=0 | false-robust | rho*_point mean [mean 95% interval] | rho*_decision |
| --- | --- | --- | --- | --- | --- |
| f1_no_cue_effect | 0.000 | 0/100 [0.00, 0.04] | yes 0.0% | 0.0195 [0.002, 0.064] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f2_direct_bypass_alpha0 | 0.234 | 100/100 [0.96, 1.00] | n/a | 0.6250 [0.597, 0.653] | unresolved 0/100, crossing 100/100 (median 0.225), none-in-range 0/100 |
| f2_direct_bypass_alpha0.5 | 0.193 | 100/100 [0.96, 1.00] | n/a | 0.6242 [0.594, 0.654] | unresolved 0/100, crossing 100/100 (median 0.100), none-in-range 0/100 |
| f2_direct_bypass_alpha1 | 0.138 | 1/100 [0.00, 0.05] | n/a | 0.6244 [0.592, 0.656] | unresolved 99/100, crossing 1/100 (median 0.015), none-in-range 0/100 |
| f2_direct_bypass_alpha2 | 0.045 | 0/100 [0.00, 0.04] | n/a | 0.6252 [0.589, 0.661] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f2_direct_bypass_alpha3 | 0.008 | 0/100 [0.00, 0.04] | n/a | 0.6250 [0.587, 0.662] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f3_shared_cause | 0.000 | 100/100 [0.96, 1.00] | yes 100.0% | 0.7069 [0.683, 0.731] | unresolved 0/100, crossing 100/100 (median 0.335), none-in-range 0/100 |
| f4_rationalization | 0.283 | 100/100 [0.96, 1.00] | n/a | 0.7063 [0.681, 0.732] | unresolved 0/100, crossing 100/100 (median 0.345), none-in-range 0/100 |
| f5_redundant_explanation | 0.000 | 100/100 [0.96, 1.00] | yes 100.0% | 0.8428 [0.826, 0.859] | unresolved 0/100, crossing 100/100 (median 0.420), none-in-range 0/100 |
| f6_answer_copying | 0.000 | 0/100 [0.00, 0.04] | yes 0.0% | 0.0200 [0.003, 0.064] | unresolved 100/100, crossing 0/100 (median n/a), none-in-range 0/100 |
| f7_opposing_effects | 0.242 | 100/100 [0.96, 1.00] | n/a | 0.6249 [0.597, 0.652] | unresolved 0/100, crossing 100/100 (median 0.270), none-in-range 0/100 |

## Element 12 part (iv): the misspecification list

Section 13 of `experiments/PREREGISTRATION_jury_and_scale.md` names eight misspecifications the offset-null family must cover. Each row below names the generator that carries it and where that generator is written. Four of the eight were already covered by the original seven families; the other four were added on 2026-09-07 as families 8 to 11, each of them family 4's world with exactly one thing changed, so the comparison against family 4 on the same table isolates the misspecification.

| list item | family | generator, file and line |
| --- | --- | --- |
| baseline offsets | `f1_no_cue_effect`, `f4_rationalization`, `f6_answer_copying`, `f7_opposing_effects` | `NoCueEffect` (`experiments/mechanism_battery.py:206`), `Rationalization` (`experiments/mechanism_battery.py:270`), `AnswerCopying` (`experiments/mechanism_battery.py:322`), `OpposingEffects` (`experiments/mechanism_battery.py:342`) |
| nonlinear depth response | `f8_nonlinear_depth` | `NonlinearDepthResponse` (`experiments/mechanism_battery.py:379`) |
| varying variance | `f9_varying_variance` | `VaryingVariance` (`experiments/mechanism_battery.py:407`) |
| correlated errors | `f3_shared_cause`, `f5_redundant_explanation` | `SharedCause` (`experiments/mechanism_battery.py:250`), `RedundantExplanation` (`experiments/mechanism_battery.py:292`) |
| sparse groups | `f10_sparse_groups` | `SparseGroups` (`experiments/mechanism_battery.py:440`) |
| treatment-induced latent states | `f5_redundant_explanation`, `f6_answer_copying` | `RedundantExplanation` (`experiments/mechanism_battery.py:292`), `AnswerCopying` (`experiments/mechanism_battery.py:322`) |
| missingness | `f11_mediator_missingness` | `MediatorMissingness` (`experiments/mechanism_battery.py:477`) |
| near-zero and cancelling effects | `f7_opposing_effects`, `f2_direct_bypass_alpha3`, `f1_no_cue_effect` | `OpposingEffects` (`experiments/mechanism_battery.py:342`), `DirectBypass` (`experiments/mechanism_battery.py:225`), `NoCueEffect` (`experiments/mechanism_battery.py:206`) |

All four run at n = 350 with their own dataset count, beside family 4 at the same size for reference. Truth is the family's own analytic value, cross-checked against its 2,000,000-row Monte Carlo in the truth table above.

| condition | datasets | rows analysed | truth NIE | NIE bias | NIE coverage | TE coverage | verdict fires |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f4_rationalization | 100 | 350.0 | 0.2827 | -0.0059 | 93/100 [0.86, 0.97] | 96/100 [0.90, 0.99] | 97/100 |
| f8_nonlinear_depth | 400 | 350.0 | 0.2803 | -0.0156 | 359/400 [0.86, 0.93] | 370/400 [0.89, 0.95] | 384/400 |
| f9_varying_variance | 400 | 350.0 | 0.1836 | 0.0653 | 169/400 [0.37, 0.47] | 272/400 [0.63, 0.73] | 383/400 |
| f10_sparse_groups | 400 | 350.0 | 0.2566 | -0.0009 | 362/400 [0.87, 0.93] | 377/400 [0.91, 0.96] | 366/400 |
| f11_mediator_missingness | 400 | 264.5 | 0.2827 | -0.0108 | 367/400 [0.89, 0.94] | 358/400 [0.86, 0.92] | 377/400 |

### Reading the four new families

**Family 8: saturating depth response.** True NIE 0.2803, estimated 0.2646 (bias -0.0156 +/- 0.0018) over 400 datasets, interval coverage 359/400 [0.864, 0.925] against 93/100 for family 4 at the same size, mean interval width 0.136. The outcome index is a saturating function of depth and the estimator's outcome equation is linear in the mediator, so what it fits is the best linear index for this data rather than the mechanism. The number to read is the bias: it is the price of the linearity assumption on a response that flattens, and it is charged against a truth computed by quadrature on the same equations.

**Family 9: arm-dependent mediator variance.** True NIE 0.1836, estimated 0.2489 (bias 0.0653 +/- 0.0015) over 400 datasets, interval coverage 169/400 [0.374, 0.473] against 93/100 for family 4 at the same size, mean interval width 0.117. The cue shifts the mediator and widens it threefold, while the model fits one sigma_m for both arms. The mediated effect here is partly a spread effect, which a single-variance model has no parameter for.

**Family 10: sparse item-level groups.** True NIE 0.2566, estimated 0.2557 (bias -0.0009 +/- 0.0018) over 400 datasets, interval coverage 362/400 [0.872, 0.932] against 93/100 for family 4 at the same size, mean interval width 0.132. The item effect enters the answer and not the mediator, so the marginal outcome model is still correct and the point estimate stays consistent; what fails is the independence the row bootstrap assumes. Read the coverage against family 4's on the same line: any shortfall here is an interval problem, not a bias problem, and it is the reason a real per-item table cannot use a row bootstrap.

**Family 11: mediator missingness, complete case.** True NIE 0.2827, estimated 0.2719 (bias -0.0108 +/- 0.0019) over 400 datasets, interval coverage 367/400 [0.886, 0.943] against 93/100 for family 4 at the same size, mean interval width 0.150. On average 264.5 of 350 rows survive, and the rows that go missing are the long traces, which are also the rows carrying the mediated signal. The truth is the full population's, so the bias here is what a complete-case analysis of a parse failure costs.


## Reading, one paragraph per family

### Family 1: no cue effect

Nothing in this world has a causal effect, and the mediator carries a real baseline: mean reasoning length 6.00 and answer rate 0.749. The repaired estimator reports a mean NIE of -0.0000 at n = 3600 and 0.0003 at n = 350; the 95 percent intervals cover the true zero in 100/100 and 100/100 datasets respectively, and the pre-registered verdict fires in 0/100 at n = 3600. This is the world the pre-repair specification got wrong by 0.21 on the mediated path, so the family is the standing evidence that the intercept repair is what makes the rest of the battery readable.

### Family 2: increasing direct bypass, fixed text pathway

The text pathway is identical in all 5 conditions (beta 0.8, gamma 1, sigma_m 1, rho 0) and only the direct bypass changes. At n = 3600 the estimated mediated share falls from 101.2% at alpha = 0 (over 100/100 datasets whose total effect interval excludes zero) to 1.6% at alpha = 3 (over 100/100), while the mean rho*_point stays between 0.6242 and 0.6252, a spread of 0.0011 across a change in the bypass from 0 to 3. That non-movement is the PASS: rho*_point is the robustness of the sign of the mediator coefficient, so it cannot answer a question about the direct path, and a reader who treats a large rho* as evidence of a load-bearing chain of thought is reading a quantity that was never about that. NIE interval coverage across the five conditions is 94/100, 94/100, 92/100, 93/100, 92/100 datasets at n = 3600 and 93/100, 93/100, 94/100, 90/100, 26/100 at n = 350. At n = 3600 no condition in the family covers below 90 percent, so the components that do carry the answer are estimated honestly at the same time. The weakest of those, f2_direct_bypass_alpha3 at n = 350, is the battery's clearest interval failure and it is not a rounding artifact: the answer rate there is 0.752, the true mediated effect is 0.0081 on a total of 0.4985, and the mean interval width is 0.0151 against a mean bias of -0.0019. A probability-scale effect this close to the boundary has a skewed sampling distribution that a percentile bootstrap of 100 datasets does not track, so the interval is too narrow even though the point estimate is nearly unbiased. The consequence for the real table is specific: in a cell where the cue almost determines the answer, the point estimates and the total effect are still readable, and an interval statement about the small remaining mediated path is not.

### Family 3: shared hidden cause, no text-to-answer effect

M = X + U and Y = 1[X + U + E > 0]: the text and the answer share a hidden cause and the text causes nothing. Mean corr(M, Y) is 0.585, and the estimator reports a mean NIE of 0.2613 against a true NIE of exactly zero, with interval coverage 0/100 at n = 3600. The direct path takes the opposite error: mean NDE -0.0023 against a truth of 0.2602, while the total effect is recovered (bias -0.0013). The false-robust-verdict rate is 100.0% of 100 datasets, so at rho = 0 the audit is confidently wrong. The only thing that flags it is the sensitivity summary: mean rho*_point 0.7069 against the analytic 1/sqrt(2), which says the whole conclusion dissolves at a residual correlation this mechanism supplies by construction.

### Family 4: cue-induced rationalization that drives the answer

Here the mediated path is real and the direct path is zero: the cue moves the reasoning text and the text moves the answer. The estimator recovers it, with NIE bias 0.0007 +/- 0.0011 and coverage 93/100 at n = 3600, and 93/100 at n = 350. The verdict fires in 100/100 datasets, which is the correct answer here because the true NIE of 0.283 clears the 0.15 threshold. Family 4 is what stops the battery being a collection of nulls: an instrument that never fires is as useless as one that always does.

### Family 5: accurate but causally redundant explanation

A latent state decides the answer and the text reports that state accurately without causing it; mean corr(M, Y) is 0.657, higher than any other family. The true NIE is zero and the estimator reports 0.2503 with coverage 0/100, a false-robust rate of 100.0% of 100, and the largest rho*_point in the battery at 0.8428. That combination is the most useful single fact the battery produces: the cell that looks most robust is the one where the mediation is entirely absent, so a large rho* must never be published as a trust score. An accurate explanation and a load-bearing one are different claims and this instrument cannot separate them from observational text alone.

### Family 6: answer copying through an opaque token channel

The answer is copied from the cue through a channel the mediator does not carry, and the cue still lengthens the text a little (mean fitted gamma 0.302). This is the case a the-text-changed-therefore-it-mattered reading gets wrong, and the estimator does not: mean NIE -0.0000 against a true zero, coverage 88/100, false-robust rate 0.0% of 100, and the direct effect recovered with bias -0.0016. That coverage, 88/100 with a binomial interval of [0.800, 0.936] at n = 3600, sits below the nominal 0.95 and is reported as a miss rather than rounded up: the point estimate is right and the interval is a little too narrow. Family 6 and family 5 differ only in whether the text is coupled to the answer-relevant state, and that single difference is what separates a clean report from a confident error.

### Family 7: opposing direct and indirect effects

The direct and indirect paths are large and opposite: truth NDE -0.242, NIE 0.242, TE 0.000. The estimator recovers both components (NDE bias -0.0031, NIE bias 0.0007, coverage 97/100 and 93/100 at n = 3600) and the total correctly reads as nothing. The TE interval excludes zero in only 3/100 datasets, which is why the pre-registered rule refuses to print a mediated share here at all: a ratio whose denominator is compatible with zero is not a quantity. An audit that looked only at the arm difference would report no cue effect on a model whose reasoning is doing a great deal of work in both directions.

## The PyMC subset at n = 350

20 datasets per family, re-run through the repaired PyMC posterior. Since the link audit of 2026-09-07 both paths are probit: the posterior fits the same outcome equation the maximum-likelihood path fits, and both convert to probability-scale effects through the same closed form, so a gap between the columns below is a difference between a posterior and a bootstrap and not a difference between two models. The last two columns measure how far apart they land on identical data rather than assuming they agree.

| condition | datasets | NIE coverage | NIE bias | mean posterior width | mean bootstrap width | max r_hat | divergences |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f1_no_cue_effect | 20 | 20/20 [0.83, 1.00] | 0.0004 | 0.016 | 0.016 | 1.0000 | 0 |
| f2_direct_bypass_alpha0 | 4 | 4/4 [0.40, 1.00] | -0.0232 | 0.120 | 0.113 | 1.0000 | 0 |
| f2_direct_bypass_alpha0.5 | 4 | 3/4 [0.19, 0.99] | -0.0183 | 0.116 | 0.122 | 1.0000 | 0 |
| f2_direct_bypass_alpha1 | 4 | 4/4 [0.40, 1.00] | 0.0114 | 0.118 | 0.121 | 1.0000 | 0 |
| f2_direct_bypass_alpha2 | 4 | 4/4 [0.40, 1.00] | 0.0080 | 0.083 | 0.087 | 1.0000 | 0 |
| f2_direct_bypass_alpha3 | 4 | 4/4 [0.40, 1.00] | 0.0087 | 0.048 | 0.018 | 1.0000 | 0 |
| f3_shared_cause | 20 | 0/20 [0.00, 0.17] | 0.2513 | 0.125 | 0.122 | 1.0000 | 0 |
| f4_rationalization | 20 | 20/20 [0.83, 1.00] | -0.0092 | 0.136 | 0.133 | 1.0000 | 0 |
| f5_redundant_explanation | 20 | 0/20 [0.00, 0.17] | 0.2405 | 0.129 | 0.128 | 1.0000 | 0 |
| f6_answer_copying | 20 | 20/20 [0.83, 1.00] | 0.0001 | 0.023 | 0.022 | 1.0000 | 0 |
| f7_opposing_effects | 20 | 19/20 [0.75, 1.00] | -0.0078 | 0.116 | 0.112 | 1.0000 | 0 |
| f8_nonlinear_depth | 20 | 17/20 [0.62, 0.97] | -0.0230 | 0.134 | 0.131 | 1.0000 | 0 |
| f9_varying_variance | 20 | 12/20 [0.36, 0.81] | 0.0574 | 0.129 | 0.110 | 1.0000 | 0 |
| f10_sparse_groups | 20 | 18/20 [0.68, 0.99] | -0.0008 | 0.133 | 0.130 | 1.0000 | 0 |
| f11_mediator_missingness | 20 | 20/20 [0.83, 1.00] | -0.0196 | 0.148 | 0.143 | 1.0000 | 0 |

The probe below records, on one seeded dataset per mechanism at n = 350, the posterior for the two outcome parameters beside the maximum-likelihood fit of the same data. Until 2026-09-07 this was where the two paths came apart: the outcome priors were fixed-scale (`alpha0 ~ Normal(0, 1.5)`, `beta ~ Normal(0, 2)`), so on a mediator with a baseline near six the implied intercept sat several prior standard deviations from zero and the posterior shrank the intercept and the mediator coefficient together. The outcome equation is now centred on the mean mediator and every mediator prior is stated in units of sd(M) (docs/ESTIMATOR-PRIORS-2026-09-07.md), so this probe is a standing check rather than a diagnosis:

- **f4_rationalization** (a mediator with a baseline near six that also drives the answer): the maximum-likelihood fit gives beta 1.130 and alpha0 -6.564. The posterior returns beta 1.137 [0.884, 1.405] and alpha0 -6.604 [-8.208, -5.108], which contains the value the maximum-likelihood fit implies for the same data.
- **f3_shared_cause** (a mediator centred near zero): the maximum-likelihood fit gives beta 1.048 and alpha0 0.026. The posterior returns beta 1.052 [0.830, 1.279] and alpha0 0.025 [-0.200, 0.247], which contains the value the maximum-likelihood fit implies for the same data.

An interval that excludes the maximum-likelihood value on this probe is the signature of a prior fighting the data rather than of a sampling problem, and it is what the fixed-scale outcome priors produced on a mediator with a baseline near six before 2026-09-07. The largest disagreement in the subset is on f2_direct_bypass_alpha0: the posterior NIE bias is -0.0232 over 4 datasets against -0.0031 for the MAP path over 100 datasets of the same condition at the same size, and the posterior interval covered the truth in 4/4 against 93/100 for the bootstrap.

## Limitations of this battery

- Every family is a low-dimensional simulation with a scalar mediator, a binary outcome and a randomized cue. It bounds what the estimator does on known mechanisms; it says nothing about whether a real truncation-curve summary is the mediator the contract names.
- The bootstrap interval is a frequentist stand-in for the pre-registered posterior verdict rule. The PyMC subset is what licenses that substitution, and it is a subset.
- The four zero-mediation families differ in how the association arises, not in whether a text-level instrument could tell them apart; nothing here shows that any measurement could separate family 5 from family 4 without an intervention.
- rho prices assumption A3 only. A trigger-conditioned pathway of the kind the organism ladder plants is an A4 violation, which no value of rho prices, so a passing battery does not license reading rho\* as a severity scale for that mechanism.

