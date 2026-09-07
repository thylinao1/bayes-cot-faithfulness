# The chain-level lambda for section 8.3, and what stands in its place today

Ruling R4 (Amendment A3.9, 2026-09-07) makes a CHAIN-level attenuation factor a done-when
before any column B number ships, and makes the printed attenuation band the route that
ships. This note carries the values, their denominators and the file each one came out of.
It is written to be turned into an additive amendment subsection; it decides nothing.

## What is measured today, and what is not

| Level | What is redrawn | lambda exists | Source |
|---|---|---|---|
| Continuation | the forced continuation at each truncation depth, chain HELD FIXED | YES, four frame-by-temperature cells | job 826025, `experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary.json`, arm `repeat-curves` |
| Chain | the whole chain of thought, then the curve read on the redraw | **NO. NOT MEASURED** | the arm exists (`src/bayes_cot_faithfulness/chain_repeats.py`, arm `chain-repeats` in `experiments/08_additive_arms.py`); the run that would produce a value is unsubmitted |

The chain-level value is ABSENT, not omitted. There is no job id to name for it because no
job was submitted: the cluster has been unreachable from this Mac since 11:21 (proof
`experiments/jury/proofs/cluster_unreachable_w2e_2026-09-07.txt`). Anyone filling it in
later runs `bcf/w2e_resume.sh smoke` and then `bcf/w2e_resume.sh smoke-fetch`.

## The continuation-level numbers, from the file rather than from a log line

Job 826025, Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`, ARC
challenge, stated-hint, 30 items entered and 28 clean-correct, r = 3 repeats per item per
temperature per frame, 5 truncation depths, concurrency 32 with `VLLM_BATCH_INVARIANT=1`.
`sigma_u^2 = sum_i SS_i / sum_i (r_i - 1)`, `sigma_m^2 = Var(item means)`,
`lambda = sigma_m^2 / (sigma_m^2 + sigma_u^2)`.

| Frame | T | Scalar | sigma_u | sigma_m | lambda | lambda corrected | items used | within-item df | held out | byte-identical repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.0 | curve_area | 0.000000 | 0.166508 | 1.000000 | 1.000000 | 28 | 56 | 0 | 28/28 = 1.000 |
| clean | 0.0 | commitment_depth | 0.000000 | 1.339272 | 1.000000 | 1.000000 | 28 | 56 | 0 | 28/28 = 1.000 |
| clean | 0.7 | curve_area | 0.000000 | 0.168874 | 1.000000 | 1.000000 | 28 | 56 | 0 | 25/28 = 0.893 |
| clean | 0.7 | commitment_depth | 0.000000 | 1.372442 | 1.000000 | 1.000000 | 28 | 56 | 0 | 25/28 = 0.893 |
| hinted | 0.0 | curve_area | 0.000000 | 0.347325 | 1.000000 | 1.000000 | 28 | 56 | 0 | 28/28 = 1.000 |
| hinted | 0.0 | commitment_depth | 0.000000 | 3.171225 | 1.000000 | 1.000000 | 25 | 50 | 3 | 28/28 = 1.000 |
| hinted | 0.7 | curve_area | 0.043644 | 0.332618 | 0.983075 | 0.982979 | 28 | 56 | 0 | 23/28 = 0.821 |
| hinted | 0.7 | commitment_depth | 1.019049 | 3.241082 | 0.910036 | 0.907255 | 26 | 52 | 2 | 23/28 = 0.821 |

The two `sigma_u` values printed as 0.000000 in the clean frame and at temperature 0 are
literal zeros to within floating point (the largest is 3.9e-17, the accumulated rounding of
a sum of identical numbers). The held-out counts are items that never committed at any
depth, so `commitment_depth` has no value for them; nothing is imputed.

## Three things this table says that a single lambda would hide

**A lambda of 1.0 here is a statement about the sampler, not about the mediator.** In the
hinted temperature-0.7 cell, 24 of 28 items have a within-item standard deviation of
exactly zero on curve area, and 23 of 28 repeat sets are byte identical. The 0.983 is
carried by four items. That is why `chain_repeats.py` refuses to report a lambda without
the identical-chain fraction beside it, and why the same fraction is in every row above.

**The clean frame's zero is a real property of this cell, not a bug.** Its temperature-0.7
continuations differ in wording on 3 of 28 items without any parsed answer at any depth
changing, so the curve read is identical while the text is not. That is exactly the
distinction `byte_identical_repeats` and `sigma_u` are measuring separately.

**The continuation-level number is a FLOOR for the chain-level one.** A3.5 says so in its
own text: the repeats above hold the chain fixed, so they measure the noise of the curve
READ. The mediator of section 8.3 is the chain. Redrawing it can only add variance, so
chain-level sigma_u >= continuation-level sigma_u and chain-level lambda <= 0.983 on curve
area and <= 0.910 on commitment depth for this cell. Ruling R4's sensitivity row at 0.80
is below both, which is the point of printing it.

## What the pending run will produce, field by field

`bcf/a3f_smoke.sh` submits `bcf-a3f-skel`: Qwen3-8B at the pinned revision, ARC,
stated-hint, 30 items, one a100-40 GRES (a MIG 3g.40gb slice), 64G host RAM, concurrency
32, `VLLM_BATCH_INVARIANT=1`, the ruling R1 preflight on, twelve arms. Its
`chain-repeats` block, summarized by `summarize_chain_repeats`, carries per frame:

* `curve_area` and `commitment_depth`, each with `sigma_u`, `sigma_m`, `lambda`,
  `lambda_noise_corrected`, `n_items_used_for_sigma_u`, `total_within_item_df` and the
  held-out counts, in the same shape as the table above so the two levels are comparable;
* `identical_chains`, the fraction of items whose r redraws are byte identical on
  `chain_sha256`, and `identical_chain_answers`, the same on the parsed final answer;
* `n_chains_drawn` and `n_chains_unparsed`, so a chain whose own answer does not parse is
  counted and dropped rather than imputed.

Two conditions on using the number when it lands, both readable from the run's own files:
`run_meta.json` must say `run_label: powered_pinned`, and `determinism_preflight.json` must
say PASS. A chain-level sigma_u measured under a non-deterministic server measures the
server. That is not a hypothetical: job 825548 measured 13 of 30 identical completions at
32 in flight with the flag off, against 30/30 with it on.

## What no repeat in this design measures

Construct-level noise in the commitment summary. Both the continuation-level and the
chain-level repeats read the SAME commitment summary through the SAME truncation grid, so
neither moves the construct. A3.5's printed band should say so in the same sentence it
gives the band, rather than letting a chain-level lambda be read as the whole of the
mediator's noise.

## Sources

| Number | File |
|---|---|
| every continuation-level row above | `experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/arms_summary.json`, `arms["repeat-curves"]["arms"][frame][T]` |
| 30 entered, 28 clean-correct, r = 3, 5 depths | the same file, `n_items`, `n_clean_correct`, `arms["repeat-curves"]["r"]` |
| flag-on 30/30 against flag-off 13/30 at 32 in flight | jobs 826020 and 825548, ruling R1 rationale in `experiments/PREREGISTRATION_jury_and_scale.md` A3.9 |
| the estimator and its refusals | `src/bayes_cot_faithfulness/chain_repeats.py` |
| the pinned revision | every `bcf/waves/a100-40-*.tsv` row for Qwen3-8B |
