# Amendment A3 draft notes: every number, its file, its field, its denominator

Written 2026-09-07 alongside the A3 draft appended as section 27 of
`experiments/PREREGISTRATION_jury_and_scale.md`. The point of this file is that a verifier can
recompute every value in that section without re-deriving anything, and can see which values
were deliberately NOT filled and why.

The recompute script that produced the measured values is
`/private/tmp/claude-501/-Users-maksimsilchenko/08bdd651-b433-4395-85a4-01e00540c367/scratchpad/a3_numbers.py`
with its output in `a3_numbers.txt` beside it. It reads only the artifacts named below.

## Artifact paths

| Short name | Path | Notes |
|---|---|---|
| run A | `soc:~/bcf/results/phase1-skeleton-a100-40/qwen3-8b/arc_challenge/stated-hint` | job 825511, exit 0, MIG 3g.40gb slice on xgph10. NOT mirrored into this repo |
| run B | `soc:~/bcf/results/phase1-anchor-mig/qwen3-8b/arc_challenge/stated-hint` | job 825492, exit 0, same MIG slice UUID |
| run B mirror | `experiments/results/phase1-anchor-mig/qwen3-8b/arc_challenge/stated-hint` | summary, throughput, run_meta, logprob_check, exit_code only |
| run C | `soc:~/bcf/results/phase1-skeleton-fullcard/qwen3-8b/arc_challenge` | job 825510, PENDING reason `Resources`, directory EMPTY |
| Llama control | `experiments/results/control_transcripts_llama-3.1-8b-instant.json` and `control_summary_llama-3.1-8b-instant.json` | 05_realmodel_control.py era, Groq backend |
| Llama p8 | `experiments/results/p8_{professor,metadata,grader_code}/numbers_table.txt` | line 110 in each |
| Llama AQuA | `experiments/results/p3_powered_aqua/numbers_table.txt` | line 110 |
| Llama ARC arms | `experiments/results/arms_summary_llama-3.1-8b-instant.json` | 114 clean-correct; carries NO acknowledgment field |

Run parameters for A and B, from each `run_meta.json`: model `Qwen/Qwen3-8B`, hf_revision
`b968826d9c46dd6066d109eabc6255188de91218`, backend `vllm 0.28.0`, tensor_parallel_size 1,
seed 7, temperature 0.0, num_predict 320, curve_cap 30, enable_thinking false,
intervention_level text, outcome_scale binary_follow, gpu `NVIDIA A100 80GB PCIe`.

## A3.1 numbers

| Value in A3 | Number | Source file | Field | Denominator |
|---|---|---|---|---|
| Run A Q1-yes proxy | 10 / 29 = 0.344828 | run A `arms_checkpoint_Qwen_Qwen3-8B.json` | count of `records[].acknowledged` true among `records[].clean_correct` true | 29 clean-correct of 30 entered |
| Run B Q1-yes proxy | 4 / 28 = 0.142857 | run B same file | same fields | 28 clean-correct of 30 entered |
| Pooled A+B | 14 / 57 = 0.245614 | both | same | 57 clean-correct rows across two DIFFERENT cue assignments |
| Run A follow | 8 / 29 = 0.275862 | run A `arms_summary.json` | `arms.twostep.n_singleshot_follow` / `arms.twostep.n` | 29 |
| Run B follow | 6 / 28 = 0.214286 | run B `arms_summary.json` | same | 28 |
| Run A silent | 5 / 29 = 0.172414 | run A checkpoint | `records[].silent` among clean-correct | 29 |
| Run B silent | 5 / 28 = 0.178571 | run B checkpoint | same | 28 |
| hint_label identical across A and B | 5 / 30 | both checkpoints | `records[].hint_label` compared position by position | 30 entered |
| Llama ARC stated-hint proxy, direct count | 3 / 22 = 0.136364 | `control_transcripts_llama-3.1-8b-instant.json` | count of `acknowledged_hint` true | 22 transcripts |
| Llama ARC stated-hint proxy, summary field | 1 / 22 = 0.045455 | `control_summary_llama-3.1-8b-instant.json` | `n_disclosed_hint` over `n_clean_correct` | 22 |
| Llama ARC professor | 1 / 114 = 0.008772 | `p8_professor/numbers_table.txt` | line 110, "acknowledged (frozen regex)" | 114 |
| Llama ARC metadata | 0 / 114 = 0.0 | `p8_metadata/numbers_table.txt` | line 110 | 114 |
| Llama ARC grader-code | 0 / 114 = 0.0 | `p8_grader_code/numbers_table.txt` | line 110 | 114 |
| Llama AQuA stated-hint | 8 / 79 = 0.101266 | `p3_powered_aqua/numbers_table.txt` | line 110 | 79 |
| Banked follow, ARC stated-hint | 0.324561 = 37 / 114 | `arms_summary_llama-3.1-8b-instant.json` | `arms.twostep.singleshot_follow_rate` | 114 |
| Banked follow, ARC professor | 0.298246 = 34 / 114 | `p8_professor/arms_summary_llama-3.1-8b-instant.json` | same field | 114 |
| Banked follow, ARC metadata | 0.017544 = 2 / 114 | `p8_metadata/...` | same field | 114 |
| Banked follow, ARC grader-code | 0.043860 = 5 / 114 | `p8_grader_code/...` | same field | 114 |

`N_s` derivation, all design numbers, no measurement involved:

- entered hinted rows per model = 3 substrates x (1,500 + 1,500 + 570 + 570) = 12,420. The
  1,500 and 570 come from DECISION-LOG.md ruling (b) of 2026-09-07 01:58; the 4 cue families
  and 3 substrates from `CONTRACT.md`.
- models per stratum from the `CONTRACT.md` family map: Qwen 4, Llama 4, Gemma 2, gpt-oss 2,
  OLMo 2.
- `N_s` entered: Qwen 49,680, Llama 49,680, Gemma 24,840, gpt-oss 24,840, OLMo 24,840.
- retention 0.615 is the banked worst-case clean-correct retention from 01-SIZING I.3 and the
  `CONTRACT.md` n-per-cell line (350 / 0.615 = 569.1). Banked best case 0.877. Measured in the
  skeleton: 29/30 = 0.967 (run A), 28/30 = 0.933 (run B).
- `N_s` at 0.615: Qwen 30,553.2, Llama 30,553.2, the other three 15,276.6.

`e_s = min(1, 50 / (f_s x N_s))` results, all recomputed by `a3_numbers.py`:

| Stratum | f_s | e_s at entered N_s | e_s at 0.615 N_s |
|---|---|---|---|
| Qwen | 0.344828 (run A) | 0.0029187 | 0.0047458 |
| Qwen | 0.142857 (run B) | 0.0070451 | 0.0114554 |
| Qwen | 0.245614 (pooled) | 0.0040977 | 0.0066629 |
| Llama | 0.052585 (weighted, 3/22) | 0.0191392 | 0.0311206 |
| Llama | 0.019647 (weighted, 1/22) | 0.0512255 | 0.0832935 |

Llama weighting arithmetic, shown so it can be checked by hand:
`(4,500 x 0.136364 + 4,500 x 0.008772 + 1,710 x 0 + 1,710 x 0) / 12,420 = 653.11 / 12,420 =
0.052585`, and with 1/22 in place of 3/22, `244.02 / 12,420 = 0.019647`. The 4,500 is
3 substrates x 1,500 and the 1,710 is 3 x 570.

No `e_s` reaches 1. Gemma, gpt-oss and OLMo have no `f_s` and no `e_s`.

## A3.2 numbers

Nothing measured. Design counts only, all read from section 12.1 of the pre-registration and
the ladder budget line of `CONTRACT.md`:

- ladder = 12 checkpoints = 3 doses x 2 seeds x (organism, twin).
- `sd_pilot(D)` at the first two dose levels across both seeds = 8 checkpoints = 4 organism
  plus 4 twin, giving 4 organism-minus-twin differences.
- ladder n = 500 items per checkpoint (12 x 500 x 12 = 72,000 completions).
- No ladder artifact exists anywhere: a `find` for `*mechanism*`, `*ladder*`, `*organism*`,
  `*twin*` under the repo returns nothing, and `STATUS.md` records W5 as not started.

## A3.3 numbers

Nothing measured for the pre-registered quantity. Counts recorded to show what does exist:

| Value | Number | Source | Field |
|---|---|---|---|
| Run A curve series | 58 (29 items x 2 frames) | run A checkpoint | `clean_curve` and `hinted_curve` present on every clean-correct record |
| Run A points per series | 5 on all 58 | run A checkpoint | `len(clean_curve.depths)` and `len(hinted_curve.depths)` |
| Run A depth points total | 290 | run A checkpoint | sum of the above |
| Run A curves-arm calls logged | 286 | run A `throughput.json` | `arms[arm=curves].n_calls` |
| Run B curve series | 56 (28 x 2) | run B checkpoint | same |
| Run B depth points total | 280 | run B checkpoint | same |
| Run B curves-arm calls logged | 275 | run B `throughput.json` | same |
| Run A series with more than one distinct answer | 2 / 29 clean, 4 / 29 hinted | run A checkpoint | `len(set(curve.answers)) > 1` |
| Run B series with more than one distinct answer | 2 / 28 clean, 3 / 28 hinted | run B checkpoint | same |

Depth grids observed, so it is clear these are truncation depths and not samples: (0,1,2,4,5),
(0,2,3,4,6), (0,2,4,5,7), (0,2,5,8,10), (0,2,4,6,8), (0,2,4,7,9), (0,4,8,11,15).

The blocking fact: `run_meta.json` temperature is 0.0 in both runs and the runner sends one
sample per call, so no empirical answer distribution exists at any k.

## A3.4 numbers

The disqualified proxy, computed and reported in A3 with its disqualification:

| Value | Number | Source | Field |
|---|---|---|---|
| Run A, modal correct AND normalized entropy >= 0.30 | 9 / 29 = 0.310345 | run A checkpoint | `anchor.cells.mu00.logprob.renormalized_over_letters`, entropy divided by log 4, modal letter vs `answer_label` |
| Run B, same | 5 / 28 = 0.178571 | run B checkpoint | same |
| Run A entropy min / median / max | 0.005348 / 0.348687 / 0.746101 | run A checkpoint | same field |
| Run B entropy min / median / max | 0.005063 / 0.379709 / 0.739144 | run B checkpoint | same |
| Run A items with mass below 1e-6 | 29 / 29, median mass 3.77e-17 | run A checkpoint | `anchor.cells.mu00.logprob.letter_probability_mass` |
| Run B items with mass below 1e-6 | 28 / 28, median mass 4.13e-17 | run B checkpoint | same |
| Run A argmax disagrees with the generated answer | 10 / 29 | run A checkpoint | argmax of `renormalized_over_letters` vs `anchor.cells.mu00.answer` |
| Run B argmax disagrees with the generated answer | 13 / 28 | run B checkpoint | same |
| logprob_check probes | 2 of 2 completed, 4 of 4 letters scored each, 4 of 4 tokens matching, 0 hard failures, passed true, argmax correct on 1 of 2, mass 0.00033501 and 0.99929026 | run A and run B `logprob_check.json` | top-level fields and `results[]` |

Minimum detectable rate, exact one-sided 95 percent Clopper-Pearson, computed with
`scipy.stats.beta.ppf(0.05, k, n - k + 1)` (scipy 1.18.1 in
`/Users/maksimsilchenko/Developer/bayes-cot-faithfulness/.venv`). Smallest `k` whose bound
strictly exceeds the threshold:

| Threshold | n | k | rate | bound |
|---|---:|---:|---:|---:|
| 0.30 | 350 | 120 | 0.342857 | 0.300805 |
| 0.30 | 570 | 190 | 0.333333 | 0.300676 |
| 0.30 | 923 | 301 | 0.326111 | 0.300626 |
| 0.30 | 1,500 | 480 | 0.320000 | 0.300129 |
| 0.50 | 50 | 32 | 0.640000 | 0.514231 |
| 0.50 | 100 | 59 | 0.590000 | 0.502892 |
| 0.50 | 200 | 113 | 0.565000 | 0.504402 |
| 0.50 | 300 | 165 | 0.550000 | 0.500875 |

The 923 row exists because DECISION-LOG ruling (b) names 923 traces as the count that buys
about 300 followed items at the banked 32.5 percent follow rate.

## A3.5 numbers

| Value | Number | Source | Field | Denominator |
|---|---|---|---|---|
| Held-out items with repeated curves | 0 | both runs | no repeat index exists in any record | 0 |
| Repeats per item | 1 | both `run_meta.json` | temperature 0.0, one sample per call | 29 and 28 items |
| Run A clean commitment depth | mean 0.344828, sd 1.316811 | run A checkpoint | `clean_curve.commitment_depth` | 29 of 29, 0 null |
| Run B clean commitment depth | mean 0.357143, sd 1.339272 | run B checkpoint | same | 28 of 28, 0 null |
| Run A hinted commitment depth | mean 0.800000, sd 2.309401 | run A checkpoint | `hinted_curve.commitment_depth` | 25 of 29, 4 null |
| Run B hinted commitment depth | mean 0.653846, sd 2.189837 | run B checkpoint | same | 26 of 28, 2 null |
| Run A clean curve area | mean 0.958621, sd 0.163701 | run A checkpoint | `clean_curve.curve_area` | 29 of 29 |
| Run B clean curve area | mean 0.957143, sd 0.166508 | run B checkpoint | same | 28 of 28 |
| Run A hinted curve area | mean 0.786207, sd 0.392541 | run A checkpoint | `hinted_curve.curve_area` | 29 of 29 |
| Run B hinted curve area | mean 0.871429, sd 0.308949 | run B checkpoint | same | 28 of 28 |
| sigma_u, both components | NOT MEASURED | no repeats | | 0 |
| lambda | NOT MEASURED | needs sigma_u | | 0 |

The mean curve areas cross-check against the summaries: run A `arms.curves.clean.mean_curve_area`
0.9586206896551724 and `arms.curves.hinted.mean_curve_area` 0.7862068965517242; run B
0.9571428571428572 and 0.8714285714285713. The sd values are computed from the per-item records
because the summary carries no sd.

Sample sd is used throughout (divisor n - 1).

## A3.6 numbers

Per-arm rows in A3 are copied field for field from `throughput.json` in each run: `arm`,
`n_calls`, `n_full_generations`, `n_forced_continuations`, `seconds`,
`full_generations_per_second`. The calls-per-second column is `n_calls / seconds` computed here
because the file does not store it.

Totals, from the same file: run A `total_calls` 1,520, `total_seconds` 660.0, `n_intervals` 11,
`overall_generations_per_second` 2.303030303030303, `n_requests_logged` 1,520; run B 1,469,
641.0, 11, 2.291731669266771, 1,469.

Full generations per second overall is computed here as the sum of `n_full_generations` over
`total_seconds`: run A 156 / 660.0 = 0.236364, run B 153 / 641.0 = 0.238690.

Card-hour projections, computed here, holding the fixed 20-item specificity holdout constant:

| Run | Scalable seconds | Scalable calls | Fixed specificity | n = 570 | n = 1,500 |
|---|---:|---:|---|---|---|
| A | 660.0 - 114.0 = 546.0 | 1,520 - 44 = 1,476 | 114.0 s, 44 calls | 546 x 19 + 114 = 10,488 s = 2.913 h, 28,088 calls | 546 x 50 + 114 = 27,414 s = 7.615 h, 73,844 calls |
| B | 641.0 - 114.0 = 527.0 | 1,469 - 42 = 1,427 | 114.0 s, 42 calls | 527 x 19 + 114 = 10,127 s = 2.813 h, 27,155 calls | 527 x 50 + 114 = 26,464 s = 7.351 h, 71,392 calls |

The scale factors are 570 / 30 = 19 and 1,500 / 30 = 50, from `n_items_entered` = 30 in each
checkpoint. The specificity holdout is 20 fixed items
(`experiments/data/specificity_holdout.json`, sha256 pinned in the frozen guard), so its cost
does not grow with the cell.

Arm-shape figures quoted in A3, computed here from the same tables: anchor is 869 / 1,520 =
57.2 percent of run A calls and 113.0 / 660.0 = 17.1 percent of its seconds (run B 841 / 1,469
= 57.2 percent and 109.0 / 641.0 = 17.0 percent); the four full-generation arms
(clean_substrate, cue_pass, placebo, specificity) are 30 + 30 + 30 + 44 = 134 of 1,520 calls =
8.8 percent and 85.0 + 110.0 + 89.0 + 114.0 = 398.0 of 660.0 seconds = 60.3 percent.

The sequential-client caveat is a code fact, not an inference: neither
`experiments/08_additive_arms.py` nor `experiments/openai_client.py` contains any concurrency
construct, so requests are issued one at a time.

## Values deliberately not filled, and what blocks each

| A3 row | Status | Blocking cause, by name |
|---|---|---|
| A3.1 Gemma, gpt-oss, OLMo `f_s` and `e_s` | no data yet | no skeleton run exists for any model in those families |
| A3.2 all three MDE rows | NOT MEASURED | ladder not run, no LoRA checkpoint, W5 CPU battery has no artifact |
| A3.3 all three k pairs | NOT MEASURED | the k = 32, temperature 0.7 clean-prompt sampling arm has not been run |
| A3.4 projected uncertain-item n | NOT MEASURED | same missing arm; the `mu00` logprob proxy is disqualified by its own mass and argmax figures |
| A3.5 sigma_u and lambda | NOT MEASURED | no repeated curves; both runs are temperature 0.0 |
| A3.5 route (latent-M or attenuation band) | NOT CHOSEN | the choice is made on lambda |
| A3.6 whole-card throughput | pending job 825510 | PENDING with reason `Resources`; the a100-80 per-user cap is 4 and another campaign holds them |
| A3.6 concurrency rows at 8, 32, 64 | named slots | Track G measurement has not reported |
| A3.6 24B to 35B, 70B, 120B classes | UNMEASURED | Phase 1 serving tests; job 825253 for tensor-parallel 2 is PENDING with `ReqNodeNotAvail` |
| A3.6 degradation-ladder trigger | NOT DECIDED | both throughput inputs above |

## Checks run, and the proof that each can fail

| Check | Command | Result | Proof it can fail |
|---|---|---|---|
| Frozen guard | `python -m pytest tests/test_frozen_guard.py -q` | 5 passed | appended one byte `x` to `PREREGISTRATION_jury_and_scale.md`, reran: 1 failed, 4 passed; restored and the sha256 returned to `300797b9...` |
| Human tone | tone-check.sh at `~/intelligence-systems/shared-discoveries/human-tone/` on the section file | PASS, 0 hard, 0 soft | ran it on a one-line probe carrying a long dash character and one banned vocabulary word: FAIL, 1 hard, 1 soft |
| Additions only | `git diff` in numstat mode against main, path-limited to the pre-registration | `491  0` | a deletion would show a nonzero second column; the raw diff has 492 lines opening with a plus (491 plus the file header) and 1 opening with a minus (the file header alone) |

Fingerprint before this commit: `076c7da8b503bc6d22efc64de2aafc08012199725dc7950ead1f7122dcab07ee`.
Fingerprint after: `300797b9ecb243c8c10ef6070b884a5170f4cf9a149e726cae2ded8d660105ba`.
File length before 1,179 lines, after 1,670 lines.

## One thing a verifier should look at first

Run A and run B are not replicates. Their `hint_label` matches on only 5 of 30 items, so the
follow and mention rates differ for a reason that has nothing to do with sampling noise or
determinism. Any reading of A3.1 that averages them without saying so is wrong, and the pooled
14 / 57 row in A3 is labelled for exactly this reason. Whether the hint label is supposed to be
seeded reproducibly across runs is a question for W3 and is not answered here.
