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
| run C | `soc:~/bcf/results/phase1-skeleton-fullcard/qwen3-8b/arc_challenge/stated-hint` | job 825510 ran on xgph0 and was OUT_OF_MEMORY killed: `ReqMem=3G`, `ReqCPUS=1`, `MaxRSS=6292740K`, 3 min 51 s, ExitCode 0:125. Resubmitted by another lane as job 826028 with `mem=64G`, `cpu=8`, running at 04:29:02 with a 2 h limit, unfinished when these notes were written |
| run W3b | `soc:~/bcf/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint`, mirrored WITH RECORDS to `experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint` in the w3b worktree | job 826025, exit 0. The source of every new number in A3.3, A3.4 and A3.5. Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218, ARC, stated-hint, 30 entered, 28 clean-correct, one a100-40 MIG 3g.40gb slice on xgph10, concurrency 32 with `VLLM_BATCH_INVARIANT=1`, eleven arms, 3,177 calls in 223.0 s |
| probe flag-off | `bcf/measured/trackg-probe-825548/probe_results.json` in the w3b worktree | job 825548, exit 0, the flag-off concurrency sweep |
| probe flag-on | `bcf/measured/w3b-probe-bi-826020/probe_results.json` in the w3b worktree | job 826020, exit 0, two servers in one job: phase A flag off at concurrency 1 writing `baseline_off_raw.json`, phase B flag on at 1, 8, 32, 64 |
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

MEASURED by job 826025. Every value below is read from
`experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary_Qwen_Qwen3-8B.json`
in the w3b worktree, and every one is recomputable from the per-item samples in
`arms_transcripts_Qwen_Qwen3-8B.json` field `sampling.samples`.

| Value | Number | Field |
|---|---|---|
| Items with a sampling block | 28 | `arms.sampling.n_records_with_samples` |
| Draw method | `n_parameter` on 28 of 28 | `arms.sampling.draw_methods` |
| k | 32 | `arms.sampling.k` |
| Temperature | 0.7 | `arms.sampling.temperature` |
| Samples drawn | 896 (28 x 32) | 28 items x k |
| Unparsed samples | 0 | `arms.sampling.n_unscorable_samples` |
| Out-of-set answers | 0 | `arms.sampling.n_out_of_set_samples` |
| Tied modes | 0 | `arms.sampling.n_modal_ties` |
| Stratum change, k 5 to 8 | 3 / 28 = 0.107143 | `arms.sampling.stability.steps[0]` |
| Stratum change, k 8 to 16 | 0 / 28 = 0.000000 | `arms.sampling.stability.steps[1]` |
| Stratum change, k 16 to 32 | 1 / 28 = 0.035714 | `arms.sampling.stability.steps[2]` |
| Binary-flag change, same three steps | 3 / 28, 0 / 28, 1 / 28 | `n_flag_changed` on the same rows |

The per-item stratum at each k is recomputed in `src/bayes_cot_faithfulness/sampling_arm.py`
from the FIRST k samples in draw order (`_at_k`), so the four k are four readings of one draw.
An injection that read the LAST k instead turns the tests red; see
`docs/w3b-proofs/sampling_and_repeat_falsification.txt` in the w3b worktree, injection 5.

The curves-arm counts the earlier draft recorded are left in the record unchanged as what they
were, a different quantity that was NOT substituted: 58 and 56 series of 5 truncation depths,
290 and 280 depth points, and 2 / 29 clean and 4 / 29 hinted series in run A and 2 / 28 and
3 / 28 in run B showing more than one distinct answer across their depths.

## A3.4 numbers

MEASURED by job 826025. Same artifact and same fields as A3.3.

| Value | Number | Field or derivation |
|---|---|---|
| Items scored | 28 of 28 | `arms.sampling.n_items_with_entropy` |
| Entropy minimum | 0.000000 | `arms.sampling.entropy_min` (the artifact stores `-0.0`, which is equal to 0.0; a unanimous item's `-(1 log 1)` is negative zero and the code now normalizes it) |
| Entropy q25, median, q75 | 0.000000 each | `entropy_q25`, `entropy_median`, `entropy_q75` |
| Entropy maximum | 0.312631 | `entropy_max`; this is the 27-versus-5 split of 32 draws |
| Entropy mean | 0.039534 | `entropy_mean` |
| Histogram, 0.1 bins | 21, 5, 1, 1, then zeros | `entropy_histogram` |
| Right-but-uncertain at 0.30 | 1 / 28 = 0.035714 | `n_right_but_uncertain` over `n_records_entered` |
| Exact two-sided 95 percent CI on 1/28 | [0.000904, 0.183478] | `scipy.stats.beta.ppf(0.025, 1, 28)` and `beta.ppf(0.975, 2, 27)` |
| Strata | right_confident 27, right_uncertain 1, wrong 0 | `arms.sampling.strata` |
| Clean-correct retention | 28 / 30 = 0.933333 | `n_clean_correct` over `n_items` in the same summary |
| Projected uncertain n at 570 entered | 19.0 | 570 x (28/30) x (1/28); the two factors multiply to exactly 1/30 |
| Projected uncertain n at 1,500 entered | 50.0 | 1,500 x (28/30) x (1/28) |
| Interval on those projections | [0.5, 97.6] and [1.3, 256.9] | the CI above times 570 x 0.933333 and 1,500 x 0.933333 |
| Single-shot follow rate | 7 / 28 = 0.250000 | `arms.twostep.singleshot_follow_rate` |
| Followed uncertain n | 4.75 at 570, 12.5 at 1,500 | uncertain n x 0.25 |

MDE values, all from `scipy.stats.beta.ppf(0.05, k, n - k + 1)` with the smallest k whose bound
strictly exceeds the threshold:

| Threshold | n | k | rate | lower bound |
|---|---:|---:|---:|---:|
| 0.30 | 19 | 10 | 0.526316 | 0.320087 |
| 0.30 | 50 | 21 | 0.420000 | 0.301384 |
| 0.50 | 4 | none resolves | not resolvable | n/a |
| 0.50 | 5 | 5 | 1.000000 | 0.549280 |
| 0.50 | 12 | 10 | 0.833333 | 0.561895 |
| 0.50 | 13 | 10 | 0.769231 | 0.505350 |

The same code reproduces all eight rows of the general table already in A3.4 (n = 350, 570,
923, 1,500 at 0.30 and n = 50, 100, 200, 300 at 0.50) to six decimals, which is the check that
the two tables use one formula and not two.

The disqualified proxy is unchanged and is now comparable to the real quantity: the `mu00`
letter-logprob proxy gave 0.31034 in run A and 0.17857 in run B where the measured value is
0.035714, so it would have overstated the uncertain stratum by five to nine times.

## A3.5 numbers

MEASURED by job 826025, at the CONTINUATION level. Artifact
`experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary_Qwen_Qwen3-8B.json`,
field `arms.repeat-curves.arms.<frame>.<temperature>`; per-repeat rows in
`arms_transcripts_Qwen_Qwen3-8B.json` field `repeat_curves`.

Design: r = 3, temperatures 0.0 and 0.7, frames clean and hinted, 28 items, 5 depths, so
28 x 2 x 2 x 3 x 5 = 1,680 forced continuations, measured at 37.0 s in `throughput.json` arm
`repeat-curves`. Seeds are per (item, frame, temperature, repeat); an injection that gives
every repeat one seed turns the tests red (injection 11 in the w3b proof file).

Curve area, `curve_area` sub-block:

| Frame | T | sigma_u | sigma_m | lambda | lambda corrected | df | n sigma_u | n sigma_m | held out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | 6.42e-18 | 0.16650786 | 1.0 | 1.0 | 56 | 28 | 28 | 0 |
| clean | 0.7 | 2.65e-17 | 0.16887427 | 1.0 | 1.0 | 56 | 28 | 28 | 0 |
| hinted | 0.0 | 3.91e-17 | 0.34732538 | 1.0 | 1.0 | 56 | 28 | 28 | 0 |
| hinted | 0.7 | 0.04364358 | 0.33261828 | 0.98307475 | 0.98297872 | 56 | 28 | 28 | 0 |

Commitment depth, `commitment_depth` sub-block:

| Frame | T | sigma_u | sigma_m | lambda | lambda corrected | df | n sigma_u | n sigma_m | held out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | 0.0 | 1.33927249 | 1.0 | 1.0 | 56 | 28 | 28 | 0 |
| clean | 0.7 | 0.0 | 1.37244231 | 1.0 | 1.0 | 56 | 28 | 28 | 0 |
| hinted | 0.0 | 0.0 | 3.17122479 | 1.0 | 1.0 | 50 | 25 | 25 | 3 |
| hinted | 0.7 | 1.01904933 | 3.24108244 | 0.91003599 | 0.90725474 | 52 | 26 | 26 | 2 |

`n_items_held_out_no_scorable_repeat` and `n_items_held_out_single_scorable_repeat` are 0 in
every one of the eight cells; the held-out column above is items whose repeats never committed
at any depth, so `commitment_depth` is null. `n_items_with_zero_within_item_sd` is 27, 26, 24
and 28 of 28 on the four curve-area rows, which is why three of those sigma_u values are
floating-point residue rather than a measurement.

Byte-identical repeats, `byte_identical_repeats` sub-block, compared on sha256 of the generated
text rather than on the parsed answer:

| Frame | T | identical | compared | incomplete |
|---|---:|---|---:|---:|
| clean | 0.0 | 28 | 28 | 0 |
| hinted | 0.0 | 28 | 28 | 0 |
| clean | 0.7 | 25 | 28 | 0 |
| hinted | 0.7 | 23 | 28 | 0 |

The clean frame at temperature 0.7 is the instructive cell: 3 of 28 items produced different
continuation TEXT across repeats and its curve-area sigma_u is still zero, because the wording
changed without any parsed answer at any depth changing.

Definitions, so the numbers can be recomputed from the per-repeat rows:
`sigma_u^2 = sum_i SS_i / sum_i (r_i - 1)`; sigma_m is the across-item sd of the item means
with divisor n minus 1; `lambda = sigma_m^2 / (sigma_m^2 + sigma_u^2)`; the corrected column
replaces the between-item variance with `max(0, var(means) - sigma_u^2 / r)`. Injections that
change the divisor to r or invert lambda turn the tests red (injections 6 and 7 in the w3b
proof file).

SCOPE: the chain of thought is held FIXED across repeats and only the forced continuation is
resampled, so this sigma_u is the noise of the curve READ. A chain-level estimate has not been
run.

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

The sequential-client caveat WAS a code fact for runs A and B: at the time neither
`experiments/08_additive_arms.py` nor `experiments/openai_client.py` contained any concurrency
construct, so requests were issued one at a time. The Track G lane added bounded concurrency at
a default of 1, and job 826025 is the first arm run to use it.

New in this revision.

Concurrency sweep, both flags, same card class, same model, same 30 ARC items, same decoding
constants. Flag-off column from `bcf/measured/trackg-probe-825548/probe_results.json` (job
825548), flag-on column from `bcf/measured/w3b-probe-bi-826020/probe_results.json` (job
826020), both in the w3b worktree, both `exit_code.txt` 0. Fields
`generations_per_second`, `logprob_calls_per_second`, and
`vs_concurrency_1.{identical_completions,max_abs_letter_logprob_diff}`:

| Concurrency | gen/s OFF | gen/s ON | lp calls/s OFF | ON | identical OFF | identical ON | max diff OFF | ON |
|---:|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 0.3539 | 0.1433 | 13.9855 | 11.2010 | 30/30 | 30/30 | 0.000 | 0.000 |
| 8 | 1.8868 | 0.8556 | 30.1927 | 23.0426 | 11/30 | 30/30 | 0.625 | 0.000 |
| 32 | 5.9650 | 2.9196 | 43.6545 | 30.7345 | 13/30 | 30/30 | 0.875 | 0.000 |
| 64 | 5.9647 | 2.9208 | 43.1726 | 30.2832 | 13/30 | 30/30 | 0.875 | 0.000 |

Job 826020 phase A re-measured the flag-off concurrency-1 row on its own node and got 0.3537
gen/s and 13.885 lp calls/s, against 825548's 0.3539 and 13.9855, so the two jobs agree. The
flag-on rows compared against phase A's raw file give 10/30 identical completions and a median
absolute letter-logprob difference of 0.125 at every level, which is why the flag is recorded
as changing the outputs and not only stabilizing them. Both servers logged
`Using FLASH_ATTN attention backend`, saved to `server-off-backend-lines.txt` and
`server-bi-backend-lines.txt` in the same directory.

Job 826025 throughput, from
`experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/throughput.json` in the
w3b worktree: `total_calls` 3,177, `total_seconds` 223.0, `n_intervals` 13,
`overall_generations_per_second` 14.246636771300448. Arm `sampling` 30 calls in 64.0 s; arm
`repeat-curves` 1,680 calls in 37.0 s. That file was RECOMPUTED offline after a regex fix: the
first version billed the sampling arm 1,710 calls because `ARM_RE` was
`running arm '([a-z]+)'` and did not match the hyphen in `repeat-curves`, so that arm got no
interval. The run is unchanged; only the attribution moved. Covered by a test proven able to
fail (`docs/w3b-proofs/probe_baseline_falsification.txt` in the w3b worktree).

Job 825510, the first whole a100-80 attempt: `sacct -j 825510` gives `State=OUT_OF_MEMORY`,
`ExitCode=0:125`, `ReqMem=3G`, `ReqCPUS=1`, `MaxRSS=6292740K`, `Elapsed=00:03:51`,
`NodeList=xgph0`. Its successor job 826028, submitted by another lane at 2026-09-07T04:29:00
with `ReqTRES=cpu=8,mem=64G,...,gres/gpu:a100-80=1`, COMPLETED: `sacct` gives `State=COMPLETED`,
`ExitCode=0:0`, `Elapsed=00:10:26`, `MaxRSS=21257416K`, `NodeList=xgph0`. 826028 belongs to
another lane; its results were READ and neither it nor 825510 was submitted, cancelled or
modified here.

The whole-card row, from
`soc:~/bcf/results/phase1-skeleton-fullcard/qwen3-8b/arc_challenge/stated-hint/throughput.json`:
`total_calls` 1,469, `total_seconds` 369.0, `overall_generations_per_second`
3.9810298102981028, full generations 153 summed over `n_full_generations`, so full generations
per second is 153 / 369.0 = 0.414634. `arms_summary_Qwen_Qwen3-8B.json` gives `n_items` 30,
`n_clean_correct` 28 and the nine A2 arms, and `run_meta.json` carries NO `concurrency` field,
so it ran the pre-concurrency runner and is a sequential-client measurement.

Ratios, computed here: 826028's 1,469 calls and 153 full generations are IDENTICAL to run B's
on the same 30 items with the same arms, so 641.0 s against 369.0 s is the same work on two
card types and the ratio is 641.0 / 369.0 = 1.7371 on both measures. Against run A, whose call
count differs, the ratio is 0.414634 / 0.236364 = 1.7542 on full generations and 3.981030 /
2.303030 = 1.7286 on calls. A MIG 3g.40gb slice is 3 of the card's 7 compute units, so a share
argument would predict 2.33; the measured 1.74 is well under it.

ARC label set, from `experiments/data/pool_manifest.json` in the w3b worktree after the fix:
`n_items` 1,500, `n_unique_resume_keys` 1,500, `choices_min` 3, `choices_max` 4,
`max_choices_allowed` 4, `frozen_prefix_n` 700,
`frozen_prefix_sha256 a48a5bef74ef30d0be729ffb4c90453a8f2390a6bc9bfe05a1a7842200860eaa`
(unchanged from the 700-item pool), and `command` now carrying
`--max-choices 4 --max-choices-from 700`. The four dropped items were at indices 836, 868, 1037
and 1382 of the pre-correction pool; the new pool differs from it first at index 836 and
contains four items the old one did not, in place of those four.

## Values deliberately not filled, and what blocks each

| A3 row | Status | Blocking cause, by name |
|---|---|---|
| A3.1 Gemma, gpt-oss, OLMo `f_s` and `e_s` | no data yet | no skeleton run exists for any model in those families |
| A3.2 all three MDE rows | NOT MEASURED | ladder not run, no LoRA checkpoint, W5 CPU battery has no artifact |
| A3.3 all three k pairs | FILLED from job 826025 | 3/28, 0/28, 1/28; open for a second model |
| A3.4 projected uncertain-item n | FILLED from job 826025 | 19 at 570 entered and 50 at 1,500, on an uncertain fraction of 1/28 whose exact CI is [0.0009, 0.1835]; the `mu00` proxy stays disqualified and would have overstated it five to nine times |
| A3.5 sigma_u and lambda | FILLED from job 826025, at the CONTINUATION level | hinted frame, temperature 0.7: sigma_u 0.043644 and lambda 0.983075 on curve area, sigma_u 1.019049 and lambda 0.910036 on commitment depth. A chain-level estimate has not been run |
| A3.5 route (latent-M or attenuation band) | STILL NOT CHOSEN | lambda now exists but is continuation-level from one 28-item cell, narrower than the quantity the route needs |
| A3.6 whole-card throughput | FILLED from job 826028 | 1,469 calls in 369.0 s, 0.414634 full generations per second; a100-80 to MIG ratio 1.7371 |
| A3.6 concurrency rows at 8, 32, 64 | FILLED both ways | jobs 825548 (flag off) and 826020 (flag on) |
| A3.6 24B to 35B, 70B, 120B classes | UNMEASURED | Phase 1 serving tests; job 825253 for tensor-parallel 2 is PENDING with `ReqNodeNotAvail` |
| A3.6 degradation-ladder trigger | STILL NOT DECIDED, but no longer for want of a measurement | both throughput inputs now exist and both point away from the trigger; what is open is whether CONTRACT.md's budget of record is repriced, which is a document decision, and the operator's batch-determinism ruling |
| Batch-determinism ruling | MEASURED, NOT RULED | job 826020 gives both sides with denominators; the ruling is the operator's |

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

---

# A3.9 and the appended paragraphs of 2026-09-07: every new number and its source

Written by the A3-final lane after the orchestrator's rulings R1 to R10
(`~/Developer/bayes-cot-phase2/RULINGS-2026-09-07.md`, 2026-09-07 09:39). Same rule as the
notes above: a verifier can recompute every value without re-deriving anything, and every value
that is NOT filled says what blocks it.

## New artifact paths used by this pass

| Short name | Path | Notes |
|---|---|---|
| flag-on cost basis | `experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/throughput.json` | job 826025, exit 0, per-arm calls and seconds at concurrency 32 with `VLLM_BATCH_INVARIANT=1` |
| sequential comparison | `bcf/measured/run-a-825511/throughput.json` | job 825511, exit 0, the same eleven intervals with a one-request-at-a-time client |
| flag-on probe | `bcf/measured/w3b-probe-bi-826020/probe_results.json` | job 826020, quoted into `bcf/waves/plan.json` under `determinism_probe` |
| wave plan | `bcf/waves/plan.json` | regenerated by `bcf/plan_waves.py` at this commit; every number in the A3.6 budget table is a field of it |
| preflight proof | `docs/a3f-proofs/EXIT_CODES.txt` | exit 0 clean, exit 10 with one of thirty completions altered |
| manifest-check proof | `docs/a3f-proofs/MANIFEST_CHECK_EXIT_CODES.txt` | exit 0 on the 216 committed rows, exit 1 with one row's `BCF_BATCH_INVARIANT` removed |

## A3.6, the repriced budget (ruling R2)

| Value in A3.6 | Number | Source | Denominator |
|---|---|---|---|
| seconds per full generation, pinned mode | 0.4100 | `plan.json` `measured_cost_model.seconds_per_full_generation` | job 826025, 1,467 calls in 122.0 s over the eleven cell intervals |
| seconds per forced continuation, pinned mode | 0.0451 | same, `seconds_per_forced_continuation` | same |
| seconds per full generation, sequential | 2.9610 | `plan.json` `measured_cost_model_sequential_comparison` | job 825511, 1,520 calls in 660.0 s |
| seconds per forced continuation, sequential | 0.1452 | same | same |
| a100-40 card-hours | 98.0 | `plan.json` `card_hours_by_pool["a100-40"].card_hours` | 96 cells, ratio 1.0 |
| a100-80 card-hours | 42.3 | same key for that pool | 72 cells, ratio 1.7371, FLOOR |
| h100-96 card-hours | 61.2 | same | 36 cells, ratio 1.0, FLOOR |
| h200-141 card-hours | 12.2 | same | 12 cells, ratio 1.0, FLOOR |
| sweep total | 213.7 | `plan.json` `total_card_hours` | 216 cells |
| sequential comparison total | 1,258.4 | `plan.json` `total_sequential_card_hours` | 216 cells |
| enrichment pass | 48.0 | `plan.json` `enrichment_pass.card_hours_total` | 18 models x 3 substrates x 1,500 pool items at 2.1333 s per item-request |
| per item-request seconds | 2.1333 | `enrichment_pass.per_item_seconds` | job 826025 `sampling` arm, 30 calls in 64.0 s |
| priced total | 261.7 | `plan.json` `ladder_trigger_element_16.priced_card_hours_total` | 213.7 + 48.0 |
| budget of record | 650 | `CONTRACT.md`, "Compute budget of record (first-order; replaced by Phase 1 measurements)" | 18 models, n=500 entered, 12 cells per model |
| trigger ratio | 0.403 | `ladder_trigger_element_16.ratio_priced_over_budget` | 261.7 / 650 |
| cells measured for their class | 84 of 216 | `card_hours_by_pool[*].n_cells_measured_for_their_class` | the seven bf16 models up to 14B x 3 substrates x 4 cue families |

**Why 213.7 and not the 152 in DECISION-LOG at 04:13.** 1,258.4 / 8.2498 = 152.5, where 8.2498
is 2.9196 / 0.3539, the flag-on-at-32 generation rate over the flag-off sequential generation
rate. That divides the WHOLE cell, forced continuations included, by the full generation's
speedup. The repriced figure uses each call type's own measured seconds from job 826025 and
gives the a100-80, h100-96 and h200-141 pools no unmeasured credit. Both numbers are quoted in
A3.6 with this explanation, so neither is silently replaced.

## A3.6, the serving mode and the preflight (ruling R1)

| Value | Number | Source |
|---|---|---|
| flag-off identical completions at 8 / 32 / 64 | 11/30, 13/30, 13/30 | job 825548 `probe_results.json`, `rows[].vs_concurrency_1.identical_completions` |
| flag-off max letter-logprob difference | 0.625 / 0.875 / 0.875 nats | same, `max_abs_letter_logprob_diff`, 120 comparisons per level |
| flag-on identical completions at 1 / 8 / 32 / 64 | 30/30 at every level | job 826020 `probe_results.json`, same fields |
| flag-on max letter-logprob difference | 0.0 at every level | same |
| flag-on generations per second at 32 | 2.9196 | job 826020, `rows[2].generations_per_second` |
| flag-off generations per second at 32 | 5.9650 | job 825548, same field |
| the two modes against each other | 10 of 30 identical, 0.75 nats max | job 826020, `rows[].vs_baseline_raw_concurrency_1` |
| preflight refusal exit code | 10 | `bcf/determinism_preflight.py` `EXIT_REFUSE`, wired into `bcf/serve_and_run.sbatch` |

## A3.4, the pooled and enriched uncertain stratum (ruling R3)

Every count below is the measured 1 of 28 uncertain fraction times the measured 28 of 30
clean-correct retention, whose product is exactly 1/30, applied to the stated entered n or pool
size. Recomputed with `scipy.stats.beta.ppf(0.05, k, n - k + 1)`, the same Clopper-Pearson
definition A3.4 already uses.

| Value in A3.4 | Number | How |
|---|---|---|
| uncertain items in a 570-entered cell | 19.0 | 570 x (28/30) x (1/28) |
| uncertain items in a 1,500-item pool | 50.0 | 1,500 x (28/30) x (1/28) |
| added by enrichment to a 570 cell | 31.0 | 50.0 - 19.0 |
| pooled over three substrates | 150.0 | 3 x 50.0 |
| 0.30 follow, smallest k at n=150 | 55 | first k whose lower bound exceeds 0.30 |
| 0.30 follow, minimum detectable rate at n=150 | 0.366667 | 55 / 150, lower bound 0.301049 |
| 0.30 follow at n=50, for comparison | 21, 0.420000 | lower bound 0.301384 |
| 0.50 silent-given-follow at n=37 | 24, 0.648649 | lower bound 0.500451; 37 is 150 x the measured 7/28 follow rate, truncated |
| 0.50 silent-given-follow at n=38 | 25, 0.657895 | lower bound 0.512038 |

## A3.5, the chain-level lambda (ruling R4): NOT MEASURED, and why

There is no value and no job id. The `chain-repeats` arm exists at this commit
(`src/bayes_cot_faithfulness/chain_repeats.py`, the arm in `experiments/08_additive_arms.py`,
12 unit tests) and has never run against a model, because `ssh soc` answered "Connection timed
out during banner exchange" on all 28 attempts across this lane, from 09:45 to past 10:10 on
2026-09-07. The routing fault is the one diagnosed in `DECISION-LOG.md` at 08:36. A3.5 names
the run and its parameters instead of a number, and the continuation-level lambdas of 0.983075
(curve area) and 0.910036 (commitment depth) from job 826025 stay labelled as what they are: a
LOWER bound on the mediator noise, because they hold the chain fixed.

## A3.2 (R6), A3.6 (R5), A3.9 (R7, R8)

| Value | Number | Source |
|---|---|---|
| ladder checkpoints in `sd_pilot(D)` | 8 of 12, giving 4 differences | element 12.1 and the ladder budget line of `CONTRACT.md` (12 x 500 x 12 = 72,000); R6 fixes WHICH 8 |
| dropped 5-option ARC items | 4, at pre-correction indices 836, 868, 1037, 1382 | already recorded in A3.6; R5 rules they never enter a cell |
| frozen ARC 700-prefix hash | `a48a5bef74ef30d0be729ffb4c90453a8f2390a6bc9bfe05a1a7842200860eaa` | `experiments/data/pool_manifest.json`, asserted by `tests/test_pool_manifest.py` |
| co-hosted judge pair failure | job 826026, exit 5 | `DECISION-LOG.md` 2026-09-07 05:31; exit 5 is `serve_and_run.sbatch`'s "server died during startup" |
| Qwen3-32B malformed rate at num_predict 256 | 164 of 342 = 0.480 | job 826023, cancelled by id mid-variant after 342 of 15,939 votes; ceiling 0.05 |

## Checks run in this pass, and the proof that each can fail

| Check | Command | Result | Proof it can fail |
|---|---|---|---|
| Frozen guard, re-pinned | `python -m pytest tests/test_frozen_guard.py -q` | 5 passed | the same guard failed on the pre-repin fingerprint the moment the section was appended, which is what forced the re-pin |
| Additions only | `git diff main --numstat -- experiments/PREREGISTRATION_jury_and_scale.md` | `391  0` | a deletion shows a nonzero second column |
| Determinism preflight | `bash bcf/preflight_falsification.sh` | clean exit 0, injected exit 10 | ONE of thirty completions altered; the script itself exits non-zero if the gate does not behave that way, and did so once when python was not on the PATH |
| Wave manifests | `python bcf/check_wave_manifests.py` | 96 files, 216 rows, no problem | `BCF_BATCH_INVARIANT` removed from one row: exit 1 with that row named |
| Chain-repeat estimator | `python -m pytest tests/test_chain_repeats.py -q` | 12 passed | a stub estimator blind to the within-item spread reports lambda 1.0 on chains that move; the shipped one is asserted to disagree |
| Wave plan | `python -m pytest tests/test_wave_plan.py -q` | 9 passed | `trigger_comparison` is called on a grid priced above the budget and reports TRIGGERED |

Fingerprint before this pass: `b2246bbf929575e011f0b379c1be7d56ed0eb7681cfb27b2e70894f908b3df1a`.
Fingerprint after: `0676fb2d9e5ba608fdb3721b67fabe4e31b5e010f5174463150b26acbb7a9635`.
File length before 1,914 lines, after 2,305 lines.

## What a verifier should look at first in this pass

The chain-level lambda of ruling R4 is the one deliverable this pass was supposed to measure and
did not. Everything else here is either a ruling written down or a number recomputed from an
artifact already in the repository. If the A3.5 paragraph ever acquires a number, check that it
came from a run whose `run_meta.json` says `run_label: powered_pinned` and whose
`determinism_preflight.json` says PASS, because a chain-level sigma_u measured under a
non-deterministic server would be measuring the server.
