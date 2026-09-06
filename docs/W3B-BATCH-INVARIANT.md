# Batch-invariant serving, measured (job 826020)

One a100-40 MIG 3g.40gb slice on xgph10, Qwen/Qwen3-8B @
`b968826d9c46dd6066d109eabc6255188de91218`, vLLM 0.28.0, temperature 0, seed 7,
`num_predict` 320, `VLLM_USE_FLASHINFER_SAMPLER=0`, the same 30 ARC items job 825548
used (the first 30 of the ARC pool; the 700-item pool at `~/bcf/repo` and the 1,500-item
pool in this tree hash identically over those 30, checked before the run). exit_code 0,
elapsed 13 min 10 s, `ReqMem=64G`.

Artifacts: `bcf/measured/w3b-probe-bi-826020/`, mirrored from
`~/bcf/results/w3b-probe-bi/`.

This file reports numbers. It makes no ruling on whether to serve with the flag; that
is the operator's, and DECISION-LOG 2026-09-07 03:58 ruling (c) says so.

## What was run

Two servers in one job on one card.

- Phase A, `VLLM_BATCH_INVARIANT` unset, concurrency 1 only, writing its 30 completions
  and 120 letter logprobs to `baseline_off_raw.json`. This is the flag-off baseline.
  Job 825548 wrote no raw completions (`probe_results.json` keeps only the summary), so
  its flag-off text cannot be compared against byte for byte; re-measuring it in the
  same job on the same node is also the tighter comparison. 825548's published rates are
  the cross-check and they agree: 0.3537 generations per second here against 0.3539
  there, 13.885 letter-logprob calls per second against 13.986.
- Phase B, `VLLM_BATCH_INVARIANT=1`, concurrency 1, 8, 32 and 64, each level compared
  against its own flag-on concurrency-1 row AND against phase A's file.

Both servers selected the same attention backend, so the flag is the only variable:

```
server-off: Using FLASH_ATTN attention backend out of potential backends:
            ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']
server-bi : Using FLASH_ATTN attention backend out of potential backends:
            ['FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION']
```

The flag-on list is one shorter (FLASHINFER drops out) and the engine traces through
`vllm/model_executor/layers/batch_invariant.py` line 141 (`matmul_persistent`), so the
Triton batch-invariant matmul path is the one in use. Neither server refused the flag
and neither fell back to a different backend. Full lines:
`bcf/measured/w3b-probe-bi-826020/server-{off,bi}-backend-lines.txt`.

## Determinism, flag ON

Against the flag-ON concurrency-1 run. Every level, all 30 items, all 120 letter
logprobs:

| concurrency | identical completions | letter logprobs compared | max abs diff | median abs diff | exactly zero |
|---:|---|---:|---:|---:|---:|
| 1 | 30/30 | 120 | 0.0 | 0.0 | 120/120 |
| 8 | 30/30 | 120 | 0.0 | 0.0 | 120/120 |
| 32 | 30/30 | 120 | 0.0 | 0.0 | 120/120 |
| 64 | 30/30 | 120 | 0.0 | 0.0 | 120/120 |

For contrast, the same table from job 825548 with the flag off: 30/30, 11/30, 13/30,
13/30 identical completions and max letter-logprob differences 0.0, 0.625, 0.875, 0.875
with a median of 0.125 from concurrency 8 upward.

So the flag restores exact batch invariance at every level measured, including 64 in
flight, on this card and this model.

## The flag changes the numbers, it does not only stabilize them

Against the flag-OFF concurrency-1 baseline measured in phase A of the same job:

| concurrency (flag on) | identical to flag-off c1 | max abs letter-logprob diff | median | exactly zero |
|---:|---|---:|---:|---:|
| 1 | 10/30 | 0.75 | 0.125 | 39/120 |
| 8 | 10/30 | 0.75 | 0.125 | 39/120 |
| 32 | 10/30 | 0.75 | 0.125 | 39/120 |
| 64 | 10/30 | 0.75 | 0.125 | 39/120 |

The four rows are identical because every flag-on level is byte-identical to every
other, so they all differ from the flag-off baseline in exactly the same way. Twenty of
the thirty greedy completions differ between the flag-off and flag-on kernels, and the
median letter-logprob difference is 0.125 nats, which is the same magnitude the flag-off
server showed between concurrency 1 and concurrency 8.

The consequence for this campaign, stated plainly and with no ruling attached: the flag
is not free with respect to already-banked artifacts. Runs 825511 and 825492 were served
flag-off, so a cell served flag-on is not a continuation of them at the token level.

## Throughput cost

Generations per second per slice, and forced letter-logprob calls per second:

| concurrency | gen/s flag OFF (825548) | gen/s flag ON (826020) | ratio | logprob calls/s OFF | ON | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3539 | 0.1433 | 0.405 | 13.986 | 11.201 | 0.801 |
| 8 | 1.8868 | 0.8556 | 0.454 | 30.193 | 23.043 | 0.763 |
| 32 | 5.9650 | 2.9196 | 0.489 | 43.655 | 30.734 | 0.704 |
| 64 | 5.9647 | 2.9208 | 0.490 | 43.173 | 30.283 | 0.701 |

Phase A's own flag-off concurrency-1 row, measured on the same node minutes earlier,
is 0.3537 gen/s and 13.885 logprob calls/s, so the 825548 column is not a stale quote.

Reading the two together: with the flag on, 32 in flight is 20.4 times the flag-on
sequential rate and 8.25 times the flag-OFF sequential rate that every card-hour
projection in `docs/TRACKG-THROUGHPUT.md` is priced at. The flag costs about half the
throughput at saturation and buys exact reproducibility at any concurrency up to 64.
Both saturate at 32; 64 adds nothing either way.

Zero retries and zero failed requests at every level in both phases (`client_stats` on
every row).

## What is not measured here

- One model, one card class, one substrate, 30 items. Nothing about a 70B, an FP8 or an
  MoE row follows from this.
- Nothing about whether the flag-on outputs are BETTER, only that they are stable. The
  flag changes which greedy token wins on 20 of 30 items and this probe has no ground
  truth to say which kernel is right.
- Nothing about the judge servers, which have their own concurrency (12 in job 825542)
  and the same exposure.
