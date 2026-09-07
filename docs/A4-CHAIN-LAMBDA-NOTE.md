# A4: the chain-level lambda for section 8.3, MEASURED

Ruling R4 (Amendment A3.9, 2026-09-07) makes a CHAIN-level attenuation factor a done-when
before any column B number ships, and makes the printed attenuation band the route that
ships. Until 2026-09-07 13:48 the chain-level value did not exist and this note carried
the continuation-level floor in its place. It now exists. This file is the measured A4
text: values, denominators, and the file each number came out of. It decides nothing; it
is written to be appended to `experiments/PREREGISTRATION_jury_and_scale.md` as an
additive amendment subsection under A4.

## The run that produced it

| Field | Value | Source |
|---|---|---|
| job id | 826596, `bcf-a3f-skel`, COMPLETED, Slurm ExitCode 0:0, 13:37:35 to 13:48:40 | `sacct -j 826596` |
| run exit code | 0 | `exit_code.txt` |
| model | `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218` | `run_meta.json` |
| substrate, cue family | `arc_challenge`, `stated-hint` | `run_meta.json` |
| backend | `openai`, vLLM 0.28.0, tensor-parallel 1, seed 7, `num_predict` 320 | `run_meta.json` |
| card | NVIDIA A100-PCIE-40GB, FLASH_ATTN attention backend | `run_meta.json` (`gpu`, `attention_backend_lines`) |
| determinism flag | `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `batch_invariant: true` | `run_meta.json` |
| concurrency | 32 | `run_meta.json`, `determinism_preflight.json` |
| `run_label` | `powered_pinned`, `exploratory_reason: null` | `run_meta.json` |
| chain-repeat draw | r = 3, temperature 0.7, seed 20260907 | `run_meta.json` (`chain_repeats`, `chain_repeat_temperature`, `chain_repeat_seed`) |
| items | 30 entered, 0 failed generation, 0 unparseable clean, 28 clean-correct | `arms_summary.json` (`attrition`, `n_clean_correct`) |
| arms | 12 enabled, including `repeat-curves` and `chain-repeats` | `arms_summary.json` (`enabled_arms`) |

Every number below is from
`experiments/results/a3f-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary.json`
in this worktree, mirrored from the cluster by `bcf/w2e_resume.sh smoke-fetch`.

## The two conditions the earlier note set, both met

The previous version of this note said the chain-level value may be used only if
`run_meta.json` says `run_label: powered_pinned` and `determinism_preflight.json` says
PASS. Both hold, and the preflight is the ruling R1 one, run on this job's own server
before any arm:

| Concurrency | Identical completions | Max abs letter-logprob difference | Logprobs compared | Missing | gen/s | logprob calls/s |
|---|---|---|---|---|---|---|
| 1 | 30/30 | 0.0 | 120 | 0 | 0.2067 | 19.8986 |
| 32 | 30/30 | 0.0 | 120 | 0 | 4.1679 | 57.2158 |

`determinism_preflight.json`: `verdict: PASS`, `exit_code: 0`, `require_flag: true`,
`batch_invariant_env: "1"`, `n_items: 30` of 30 required, levels `[1, 32]`, `reasons: []`.
`determinism_preflight_probe.json` adds `median_abs_letter_logprob_diff: 0.0` and
`n_exactly_zero_diff: 120` at both levels, and `requests_ok: 150, retries: 0,
requests_failed: 0` on each. The exit path taken was therefore the PASS path of
`bcf/serve_and_run.sbatch` line 335 (`[preflight] PASS`); the refusal path at line 333,
`exit 10`, did not fire, and neither did the `BCF_PREFLIGHT` skip path at line 337 that
would have stamped `exploratory_reason`.

`logprob_check.json` also passed: 2 of 2 probes completed, no hard failures, all letter
logprobs non-positive, 4 of 4 requested letters scored on each probe.

## The chain-level components, per frame and per summary

`sigma_u^2 = sum_i SS_i / sum_i (r_i - 1)` within item across the r REDRAWN chains;
`sigma_m^2 = Var(item means)`; `lambda = sigma_m^2 / (sigma_m^2 + sigma_u^2)`. Same closed
form and same estimator as the continuation-level ones, deliberately, so the two levels
are comparable. Source: `arms["chain-repeats"]["frames"][frame][scalar]`.

| Frame | Summary | sigma_u | sigma_m | lambda | lambda corrected | items used | within-item df | held out | chains drawn | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | curve_area | 0.075593 | 0.139770 | 0.773690 | 0.755225 | 28 of 28 | 56 | 0 | 84 | 0 |
| clean | commitment_depth | 0.411943 | 1.085079 | 0.874027 | 0.868436 | 28 of 28 | 55 | 0 | 84 | 0 |
| hinted | curve_area | 0.252605 | 0.269506 | 0.532337 | 0.445970 | 28 of 28 | 56 | 0 | 84 | 0 |
| hinted | commitment_depth | 2.078461 | 2.303379 | 0.551195 | 0.465001 | 26 of 28 | 50 | 2 | 84 | 0 |

The two held-out items in the hinted `commitment_depth` row are
`n_items_held_out_single_scorable_repeat = 2`: they have one scorable redraw, which gives
a mean but no within-item df, so they enter `sigma_m` and not `sigma_u`.
`n_items_held_out_no_scorable_repeat` is 0 in all four rows and `n_chains_unparsed` is 0
in both frames, so nothing was imputed and nothing was silently dropped. The clean
`commitment_depth` df of 55 rather than 56 is one item with two scorable redraws
(`mean_repeats_per_item` 2.9643).

### The identical-chain fractions, which is what makes these lambdas readable

| Frame | Identical chains (`chain_sha256`) | Identical chain answers |
|---|---|---|
| clean | 0 of 28 = 0.000 | 27 of 28 = 0.964 |
| hinted | 0 of 28 = 0.000 | 21 of 28 = 0.750 |

This is the point of the fraction. At the continuation level the same run has 22 to 28 of
28 repeat sets byte identical, so a lambda near 1 there is partly a statement that the
sampler did not move. Here NO item returned the same chain twice in either frame, so
every one of these lambdas is computed on redraws that actually differ.

## Against the continuation-level floor, in the same job

The floor claim in A3.5 is that continuation-level repeats hold the chain fixed and so
measure only the noise of the curve READ; redrawing the chain can only add variance, so
chain-level sigma_u >= continuation-level sigma_u and chain-level lambda <=
continuation-level lambda. Job 826596 ran BOTH arms on the same 28 items, so the
comparison is within one run rather than across two. Continuation rows are
`arms["repeat-curves"]["arms"][frame]["0.7"]`, at the matching temperature 0.7.

| Frame | Summary | sigma_u continuation | sigma_u chain | ratio | lambda continuation | lambda chain | drop |
|---|---|---|---|---|---|---|---|
| clean | curve_area | 0.021822 | 0.075593 | 3.46 | 0.983240 | 0.773690 | 0.209550 |
| clean | commitment_depth | 0.218218 | 0.411943 | 1.89 | 0.974528 | 0.874027 | 0.100501 |
| hinted | curve_area | 0.048795 | 0.252605 | 5.18 | 0.981859 | 0.532337 | 0.449522 |
| hinted | commitment_depth | 1.045626 | 2.078461 | 1.99 | 0.825532 | 0.551195 | 0.274337 |

The predicted direction holds in all four cells: sigma_u rises by a factor of 1.9 to 5.2
and lambda falls in every one. The floor was a floor.

Against the OTHER continuation-level table, job 826025 in
`experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary.json`,
which is the table the previous version of this note printed: its hinted temperature-0.7
cells give lambda 0.983075 on curve area and 0.910036 on commitment depth. The chain-level
values on the same two summaries are 0.532337 and 0.551195. Both are below both floors.

### One thing the ratio table alone hides

sigma_m moves too, and not always down. Clean curve_area sigma_m goes 0.167142 to
0.139770 and clean commitment_depth 1.349766 to 1.085079, because averaging over three
redrawn chains shrinks the spread of the item means; hinted commitment_depth goes the
other way, 2.274496 to 2.303379. So lambda falls at both ends of the ratio in three of
four cells and from the numerator alone in the fourth, and a reader who reconstructs
lambda from sigma_u alone will get the wrong number.

## What the measured value does to ruling R4's printed sensitivity row

R4's rationale for printing a sensitivity row at lambda = 0.80 was that 0.80 sits below
both continuation-level values (0.983 on curve area, 0.910 on commitment depth) for the
hinted temperature-0.7 cell. Measured at the chain level in the hinted frame, lambda is
0.532337 on curve area and 0.551195 on commitment depth, and 0.445970 and 0.465001 after
the noise correction. All four are BELOW 0.80. In the clean frame the raw values, 0.773690
and 0.874027, straddle it.

That is a measurement, not a ruling. What it means for the printed band, and whether the
sensitivity row moves, is the operator's under R4. This note names no new value for the
band.

## What this run does NOT settle

**Its n is a smoke n.** 30 items entered, 28 clean-correct, r = 3, one model, one
substrate, one cue family, `num_predict` 320. `arms_summary.json` marks the whole arm set
`"exploratory Phase-2 arms; not part of the frozen pre-registered controls; no verdict"`.
The serving line is a line of record (pinned revision, flag on, its own preflight passed);
the ARMS are exploratory. No interval is attached to any lambda above and none should be
read off four two-decimal numbers on 28 items.

**Construct-level noise in the commitment summary is still measured by nothing.** Both
levels read the SAME commitment summary through the SAME truncation grid, so neither
moves the construct. `summarize_chain_repeats` says so in its own `scope` string, and
A3.5's printed band should carry the same sentence beside the band.

**One card, one draw.** These are three redraws at temperature 0.7 with a single seed
(20260907) on one A100-PCIE-40GB. Nothing here separates chain-level mediator noise from
seed-to-seed or card-to-card variation in the redraw itself.

## Sources

| Number | File |
|---|---|
| every chain-level row, the identical-chain fractions, the held-out counts | `experiments/results/a3f-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary.json`, `arms["chain-repeats"]["frames"][frame]` |
| every continuation-level row in the same-job comparison | the same file, `arms["repeat-curves"]["arms"][frame]["0.7"]` |
| the job's serving line, flag, concurrency, revision, run label, card | the same directory, `run_meta.json` |
| the preflight verdict and both concurrency levels | the same directory, `determinism_preflight.json`, `determinism_preflight_probe.json` |
| the exit path names and line numbers | `bcf/serve_and_run.sbatch` lines 300 to 337 |
| 30 entered, 28 clean-correct, the arm list, the exploratory status string | `arms_summary.json`, `attrition`, `n_clean_correct`, `enabled_arms`, `status` |
| the older continuation-level table (job 826025) | `experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary.json` |
| the estimator, its refusals and the identical-chain rule | `src/bayes_cot_faithfulness/chain_repeats.py` |
| job state and Slurm exit code | `sacct -j 826596` on the SoC cluster, read 2026-09-07 13:53 |
