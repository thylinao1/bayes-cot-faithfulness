# The W3b skeleton: element 9.2 and element 8.3 on a real server (job 826025)

30 ARC-Challenge items, Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`,
stated-hint, one a100-40 MIG 3g.40gb slice on xgph10, vLLM 0.28.0, seed 7,
`enable_thinking` false, `VLLM_USE_FLASHINFER_SAMPLER=0`. `exit_code.txt` = 0.
Eleven arms: the nine A2 arms plus `sampling` (element 9.2) and `repeat-curves`
(element 8.3, PF-13).

Artifacts: `experiments/results/w3b-skeleton/qwen3-8b/arc_challenge/stated-hint/`,
mirrored from `~/bcf/results/w3b-skeleton/...`. Every number below names the field it
came from.

## The serving choice, and why

**Concurrency 32 with `VLLM_BATCH_INVARIANT=1`.** The brief made this conditional: run
at concurrency 1 unless the batch-invariant probe shows 30/30 identical completions at
32 in flight. Job 826020 shows exactly that (`docs/W3B-BATCH-INVARIANT.md`), so this run
took the concurrent path. `run_meta.json` records `concurrency: 32`,
`batch_invariant: true` and `vllm_batch_invariant_env: "1"`, so the choice is on the
record rather than in a shell history.

The run confirms the probe on its own data: **28 of 28 items produced byte-identical
temperature-0 repeats** on both the clean and the hinted curve
(`arms.repeat-curves.arms.<arm>.0.0.byte_identical_repeats`), with 32 requests in flight
throughout. Under the flag-off server job 825548 measured 13 of 30 identical at that
concurrency.

3,177 calls in 223.0 s, 14.247 calls per second overall (`throughput.json`). The whole
eleven-arm run took under four minutes of arm time on one MIG slice.

## Element 9.2, the uncertain-item sampling arm

k = 32 at temperature 0.7 on the clean prompt, 28 clean-correct items, `draw_method`
`n_parameter` on all 28: **the vLLM endpoint honors `n > 1`**, so each item's 32 samples
came from one request sharing one seed, not from 32 seeded calls. The client checks
rather than assumes (a server that ignores `n` answers with one choice, which is
indistinguishable from k = 1 unless counted) and would have fallen back.

- 28 of 28 items scored; 0 samples unparsed, 0 out-of-set answers, 0 tied modes.
- Normalized answer entropy: min 0.0, q25 0.0, median 0.0, q75 0.0, max 0.3126312,
  mean 0.0395340.
- Histogram on fixed 0.1-wide bins: 21 items in [0.0, 0.1), 5 in [0.1, 0.2), 1 in
  [0.2, 0.3), 1 in [0.3, 0.4), none above 0.4.
- **Right-but-uncertain at the frozen 0.30 threshold: 1 of 28, 0.0357.** Strata:
  `right_confident` 27, `right_uncertain` 1, `wrong` 0.

Twenty-one of the twenty-eight items answered the same letter on all 32 draws. On this
model, this substrate and this decoding setting, temperature 0.7 sampling is close to
deterministic, and the uncertain stratum is thin. That is a measurement about a 30-item
cell on one model, not a property of the benchmark, and it is what A3.4 now projects
from.

Stability, recomputed from prefixes of the same 32 draws in draw order:

| k pair | stratum changed | flag changed | items compared | fraction |
|---|---:|---:|---:|---:|
| 5 to 8 | 3 | 3 | 28 | 0.107143 |
| 8 to 16 | 0 | 0 | 28 | 0.000000 |
| 16 to 32 | 1 | 1 | 28 | 0.035714 |

The one right-but-uncertain item at k = 32 sits at 0.3126312, which is the 27-to-5 split
the unit tests use as the boundary case: one sample fewer in the minority and it falls
under 0.30. The 16-to-32 change is that item crossing.

Reported, never gated, as section 9.2 requires.

## Element 8.3 (PF-13), repeated truncation curves

r = 3 repeats per item per temperature per frame, 28 items, both frames, both
temperatures: 28 x 2 x 2 x 3 x 5 depths = 1,680 forced continuations in 37.0 s
(45.4 calls per second, `throughput.json` arm `repeat-curves`).

Primary scalar is `curve_area`; `commitment_depth` is reported beside it and has a
smaller denominator wherever a trace never commits.

| frame | T | scalar | sigma_u | sigma_m | lambda | lambda corrected | within-item df | items for sigma_u | items for sigma_m |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | curve_area | 6.42e-18 | 0.166508 | 1.000000 | 1.000000 | 56 | 28 | 28 |
| clean | 0.0 | commitment_depth | 0.0 | 1.339272 | 1.000000 | 1.000000 | 56 | 28 | 28 |
| clean | 0.7 | curve_area | 2.65e-17 | 0.168874 | 1.000000 | 1.000000 | 56 | 28 | 28 |
| clean | 0.7 | commitment_depth | 0.0 | 1.372442 | 1.000000 | 1.000000 | 56 | 28 | 28 |
| hinted | 0.0 | curve_area | 3.91e-17 | 0.347325 | 1.000000 | 1.000000 | 56 | 28 | 28 |
| hinted | 0.0 | commitment_depth | 0.0 | 3.171225 | 1.000000 | 1.000000 | 50 | 25 | 25 |
| hinted | 0.7 | curve_area | 0.043644 | 0.332618 | 0.983075 | 0.982979 | 56 | 28 | 28 |
| hinted | 0.7 | commitment_depth | 1.019049 | 3.241082 | 0.910036 | 0.907255 | 52 | 26 | 26 |

Held out: 3 items from the hinted temperature-0 commitment-depth row and 2 from the
hinted temperature-0.7 row, all of them items whose repeats never committed at any
depth, so `commitment_depth` is null and the item has no value on that scale. No item
was held out for having nothing scorable, and none for having only one usable repeat
(`n_items_held_out_no_scorable_repeat` and
`n_items_held_out_single_scorable_repeat` are 0 in every cell).

The sigma_u values of 6e-18 and 3e-17 are floating-point residue on sums that are
exactly zero; the accompanying `n_items_with_zero_within_item_sd` is 27 of 28 and 26 of
28. Read them as zero.

Byte-identical repeats, the determinism half of the same arm:

| frame | T | identical | compared |
|---|---:|---|---:|
| clean | 0.0 | 28 | 28 |
| hinted | 0.0 | 28 | 28 |
| clean | 0.7 | 25 | 28 |
| hinted | 0.7 | 23 | 28 |

At temperature 0.7 the text of the forced continuation differs on 3 clean and 5 hinted
items, yet the clean frame's `curve_area` sigma_u is still zero: on the clean frame the
resampled continuations differed in wording without ever changing a parsed answer at any
depth. The hinted frame is where the sampling actually moves the mediator.

**Scope, stated so the number is not over-read.** These repeats resample the FORCED
CONTINUATION at each truncation depth with the chain of thought held fixed. They do not
resample the chain itself. So sigma_u here is the noise of the curve READ, and a mediator
whose chain is also redrawn would carry at least this much and probably more. Section
8.3's closed form takes whatever sigma_u the deliverable measures; this is the
continuation-level one, and the chain-level one is a strictly larger experiment that no
run has done.

Section A3.5 of the A3 draft asked for repeats at a nonzero temperature and warned that
repeating at temperature 0.0 "would return the same curve R times and report sigma_u = 0
by construction". Both were run deliberately: the temperature-0.7 rows are the
mediator-noise estimate, and the temperature-0.0 rows are the determinism measurement
the batch-invariant flag was turned on for. The temperature-0 sigma_u of zero here is
the flag working, and under a flag-off server at 32 in flight it would not have been
zero.

## The nine A2 arms, unchanged in shape

28 of 30 clean-correct (30 entered, 0 failed generations, 0 unparseable clean).
Single-shot follow 7 of 28, silent 5 of 28. Two-step follow 2 of 28 against single-shot
7 of 28. Clean curves: 26 of 28 pre-committed at depth 0, mean area 0.957143. Hinted
curves: 21 of 28 pre-committed, 3 never committed, mean area 0.828571. The forced-logprob
unit check passed on the live server before any arm ran (`logprob_check.json`).

## A defect this run found in the reporting, not in the arms

The first `throughput.json` this job wrote billed the sampling arm 1,710 calls when it
made 28. `bcf/throughput.py` matched arm boundaries with `running arm '([a-z]+)'`, which
does not match a name containing a hyphen, so `repeat-curves` never got an interval and
its 1,680 forced continuations were billed to the arm before it. The regex now accepts
hyphens, `throughput.json` was recomputed offline from this run's own `run.log` and
`requests.jsonl`, and a test covers it, proven able to fail in
`docs/w3b-proofs/probe_baseline_falsification.txt`. Nothing about the run itself changed;
only the attribution of its calls did.
