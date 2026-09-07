# Wave 1 audit: eight a100-40 cells, ARC-Challenge, stated-hint, n 1,500

Track V, the audit lane. Jobs 826733 to 826740, submitted 14:29 on 2026-09-07 from the
immutable tree `~/bcf/repo-5d40e5224ac0` at plan commit `5d40e5224ac0`. One cell per small
model: ARC-Challenge, the frozen `stated-hint:strong` cue family, 1,500 items entered, one
a100-40 MIG slice each, `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
concurrency 32, temperature 0.0, `num_predict` 320, `curve_cap` 1500, seed 7, vLLM 0.28.0,
and the ruling-R1 determinism preflight on each cell's own server before any arm.

**What this document is.** Every number below is recomputed from the cluster records rather
than copied from the summary the cell job wrote, and the two are printed side by side with
an agreement column. The records stay on the cluster under
`~/bcf/results/<slug>/arc_challenge/stated-hint/`; the small per-cell files are mirrored into
`docs/wave1-artifacts/<slug>/` and the larger ones into the gitignored
`experiments/results/wave1/<slug>/`. This document makes no ruling; element 9.4 is the
operator's and `docs/REASONING-MODE-TEST.md` carries the data for it.

## How each number was produced, and how each check was made to fail

- **Clean accuracy** is recomputed as clean-correct over items ENTERED, counted from the
  cell's own `arms_checkpoint_*.json` records (`experiments/audit/recompute_wave1.py`).
  Fail proof: flipping one record's `clean_correct` in memory moves the count from 1,396 to
  1,395 on Qwen3-8B, so the figure is read from the records rather than echoed.
- **Single-shot follow** is recomputed over the twostep-scorable records, which is the frame
  `summarize_twostep` uses (`experiments/08_additive_arms.py:465`), so the two rates compare
  like for like. Fail proof: flipping one record's `followed` moves 249 to 248.
- **The determinism preflight** is not read off the verdict field. Each cell's PROBE is
  re-evaluated with the campaign's own `bcf/determinism_preflight.evaluate()`
  (`experiments/audit/verify_preflights.py`). Fail proof: six perturbations of a passing
  probe (one non-identical completion, a 1e-9 letter-logprob difference, one missing
  logprob, zero logprobs compared, the flag off, 29 items) all refuse. A first attempt
  perturbed the row's top level and nothing moved, because `evaluate` reads
  `row["vs_concurrency_1"]`; that near miss is recorded because a check that ignores the
  field you think it reads looks exactly like a check that passes.
- **Calls, seconds and calls per second** come from each cell's `throughput.json`, which
  `bcf/throughput.py` builds by joining the request log's timestamps against the arm
  boundaries the runner printed. They are measured, not estimated.

## The cells

| cell | job | exit | preflight | entered | records | clean-correct | clean acc | unparseable clean | summary clean-correct | agree | single-shot follow | summary follow | agree | calls | seconds | calls/s | forced-answer arms scorable (direct / twostep / filler / placebo) | usable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 826733 | 0 | PASS | 1500 | 1500 | 1396 | 0.9307 | 0 | 1396 | yes | 249/1396 = 0.1784 | 0.1784 | yes | 71327 | 2834.0 | 25.168 | 1396 / 1396 / 1396 / 1396 | USABLE |
| olmo-3-7b-think | 826734 | 0 | PASS | 1500 | 1500 | 56 | 0.0373 | 1381 | 56 | yes | - | null | both none | 6920 | 864.0 | 8.009 | 0 / 0 / 0 / 19 | NOT USABLE AS MEASURED (clean-correct 56 < 350) |
| deepseek-r1-0528-qwen3-8b | 826735 | 0 | PASS | 1500 | 1500 | 1 | 0.0007 | 1499 | ABSENT | no summary | - | ABSENT | no summary | 2999 | 602.0 | 4.982 | - / - / - / - | NOT USABLE AS MEASURED (clean-correct 1 < 350) |
| deepseek-r1-distill-llama-8b | 826736 | 0 | PASS | 1500 | 1500 | 140 | 0.0933 | 1329 | 140 | yes | 3/15 = 0.2000 | 0.2000 | yes | 12682 | 1174.0 | 10.802 | 0 / 15 / 0 / 47 | NOT USABLE AS MEASURED (clean-correct 140 < 350) |
| gemma-2-9b-it | 826737 | 0 | PASS | 1500 | 1500 | 1380 | 0.9200 | 3 | 1380 | yes | 425/1379 = 0.3082 | 0.3082 | yes | 70021 | 2676.0 | 26.166 | 1379 / 1379 / 1373 / 1376 | USABLE |
| llama-3.1-8b-instruct | 826738 | 0 | PASS | 1500 | 1500 | 1321 | 0.8807 | 0 | 1321 | yes | 448/1321 = 0.3391 | 0.3391 | yes | 67698 | 3187.0 | 21.242 | 1318 / 1321 / 1321 / 1321 | USABLE |
| phi-4-reasoning | 826739 | 0 | PASS | 1500 | 1500 | 1197 | 0.7980 | 265 | 1197 | yes | 1/4 = 0.2500 | 0.2500 | yes | 84781 | 9585.0 | 8.845 | 0 / 4 / 0 / 961 | CLEAN PASS CLEARS THE FLOOR, ARMS DO NOT (direct 0, twostep 4, filler 0) |
| gpt-oss-20b | 826740 | 5 | ABSENT | - | - | - | - | - | ABSENT | no summary | - | ABSENT | no summary | - | - | - | - / - / - / - | NO RECORDS |

Per-cell determinism preflight lines:
- `qwen3-8b`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `olmo-3-7b-think`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `deepseek-r1-0528-qwen3-8b`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `deepseek-r1-distill-llama-8b`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `gemma-2-9b-it`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `llama-3.1-8b-instruct`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `phi-4-reasoning`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `gpt-oss-20b`: no determinism_preflight.json written

Per-arm rates (arm, calls, seconds, calls/s):
- `qwen3-8b`: clean_substrate 1505/309.0s = 4.871; cue_pass 1402/364.0s = 3.852; replay 2791/82.0s = 34.037; placebo 1404/307.0s = 4.573; direct 1396/34.0s = 41.059; twostep 2791/244.0s = 11.439; filler 1397/54.0s = 25.870; curves 13958/217.0s = 64.323; transplant 2792/84.0s = 33.238; anchor 41852/1123.0s = 37.268; specificity 39/16.0s = 2.438
- `olmo-3-7b-think`: clean_substrate 2881/603.0s = 4.778; cue_pass 78/24.0s = 3.250; replay 224/11.0s = 20.364; placebo 93/25.0s = 3.720; direct 112/4.0s = 28.000; twostep 168/27.0s = 6.222; filler 112/5.0s = 22.400; curves 536/22.0s = 24.364; transplant 217/11.0s = 19.727; anchor 2459/108.0s = 22.769; specificity 40/24.0s = 1.667
- `deepseek-r1-0528-qwen3-8b`: clean_substrate 2999/602.0s = 4.982
- `deepseek-r1-distill-llama-8b`: clean_substrate 2837/582.0s = 4.875; cue_pass 243/61.0s = 3.984; replay 522/27.0s = 19.333; placebo 234/60.0s = 3.900; direct 280/9.0s = 31.111; twostep 405/66.0s = 6.136; filler 280/13.0s = 21.538; curves 1370/51.0s = 26.863; transplant 534/27.0s = 19.778; anchor 5938/253.0s = 23.470; specificity 39/25.0s = 1.560
- `gemma-2-9b-it`: clean_substrate 1503/244.0s = 6.160; cue_pass 1388/282.0s = 4.922; replay 2767/91.0s = 30.407; placebo 1384/240.0s = 5.767; direct 1373/35.0s = 39.229; twostep 2774/243.0s = 11.416; filler 1374/48.0s = 28.625; curves 13212/260.0s = 50.815; transplant 2766/92.0s = 30.065; anchor 41442/1125.0s = 36.837; specificity 38/16.0s = 2.375
- `llama-3.1-8b-instruct`: clean_substrate 1531/410.0s = 3.734; cue_pass 1398/408.0s = 3.426; replay 2643/86.0s = 30.733; placebo 1349/375.0s = 3.597; direct 1302/23.0s = 56.609; twostep 2664/366.0s = 7.279; filler 1315/44.0s = 29.886; curves 13212/197.0s = 67.066; transplant 2642/86.0s = 30.721; anchor 39606/1172.0s = 33.794; specificity 36/20.0s = 1.800
- `phi-4-reasoning`: clean_substrate 1766/852.0s = 2.073; cue_pass 1828/733.0s = 2.494; replay 4779/389.0s = 12.285; placebo 1438/690.0s = 2.084; direct 2394/127.0s = 18.850; twostep 3587/829.0s = 4.327; filler 2394/186.0s = 12.871; curves 9150/636.0s = 14.387; transplant 4783/389.0s = 12.296; anchor 52618/4714.0s = 11.162; specificity 44/40.0s = 1.100
- `gpt-oss-20b`: no throughput.json

### Which cells are usable at the pre-registered n, and which are not

The pre-registered floor is `01-SIZING.md` section I.3: **at least 350 clean-correct items per
cell** (at 20 percent mediator noise, n = 350 gives an NIE posterior 95 percent half-width of
0.0996 faithful and 0.0690 weak, both inside the 0.10 bar; n = 300 gives 0.1027 faithful and is
outside it).

**Usable at the pre-registered n: three of eight.** `qwen3-8b` (1,396 clean-correct of 1,500
entered), `gemma-2-9b-it` (1,380) and `llama-3.1-8b-instruct` (1,321). All three also carry
forced-answer arms with denominators at the same scale.

**Not usable as measured: five of eight**, for three different reasons.

1. **Below the clean-correct floor**, three cells: `olmo-3-7b-think` 56 of 1,500 (0.0373),
   `deepseek-r1-distill-llama-8b` 140 of 1,500 (0.0933), `deepseek-r1-0528-qwen3-8b` 1 of
   1,500 (0.0007, and its `arms_summary.json` was never written because the runner stopped at
   "too few clean-correct items to run the arms"). All three are FLAGGED NOT USABLE AS
   MEASURED. `docs/REASONING-MODE-TEST.md` carries the diagnosis.
2. **Clears the clean-correct floor and still has empty arms**, one cell:
   `phi-4-reasoning`, 1,197 clean-correct of 1,500, which is comfortably above 350, but
   `direct` 0 scorable of 1,197, `filler` 0 of 1,197 and `twostep` 4 of 1,197 (`placebo` 961
   of 1,197). The forced-answer continuation is 24 tokens (element 15) and Phi-4 spends them
   opening a new reasoning block, so the arms that depend on a forced answer return nothing.
   Reading only the clean accuracy would have called this cell fine. It is NOT usable as a
   cell of record.
3. **No records at all**, one cell: `gpt-oss-20b`, job 826740, exit 5 after 2 minutes, the
   server died in startup. See below; the cause is not the slice size.

### gpt-oss-20b: the flag and the quantization are mutually exclusive on this card

vLLM 0.28.0 refused every MXFP4 MoE backend on the A100 slice and named its reasons
(`~/bcf/results/gpt-oss-20b/arc_challenge/stated-hint/server.log`):

> `NotImplementedError: No MXFP4 MoE backend supports the deployment configuration.`
> `weight_key=kMxfp4Static, activation_key=None.` … `backend: MARLIN, reason: kernel does not
> support batch invariance; backend: BATCHED_MARLIN, reason: kernel does not support
> ('standard',) activation format; backend: TRITON, reason: kernel does not support current
> device cuda; …`

Every candidate except MARLIN is refused for the device (Ampere, SM80); MARLIN is refused for
**batch invariance**, which is ruling R1's `VLLM_BATCH_INVARIANT=1`. The flag and mxfp4
gpt-oss on an A100 cannot both hold in this build.

The other direction is measured too, in another lane's job 826883 (Track G follow-up,
`~/bcf/results/gptoss-line-a100-40-flagoff/`, read here read-only): with the flag UNSET the
server starts, the preflight runs, and it gives **30/30 identical at concurrency 1 but 9/30
identical and a maximum absolute letter-logprob difference of 1.1250038146972656 nats at
concurrency 32**, so the job exits 10, the preflight refusal. So gpt-oss-20b at mxfp4 on an
a100-40 slice is unservable as a cell of record in both directions: flag on, no backend; flag
off, no determinism. This is a serving question for the roster, not a reasoning-mode question,
and this lane makes no ruling on it.

### What the throughput says about the budget

Overall measured rates, from each cell's own `throughput.json`: 25.168 calls/s (`qwen3-8b`,
71,327 calls in 2,834 s), 26.166 (`gemma-2-9b-it`), 21.242 (`llama-3.1-8b-instruct`), 8.845
(`phi-4-reasoning`, 84,781 calls in 9,585 s), and 10.802, 8.009, 4.982 on the three thinking
cells, whose low totals reflect arms that had almost nothing to run rather than a slow server.
Against the Phase 1 skeleton's 1.182 calls per second (section 17), an 8B cell at n = 1,500
finishes in under an hour on one MIG slice, so the compute degradation ladder of element 16 is
not needed for these rows.

### What is missing from this audit, named

- `arms_summary.json` does not exist for `deepseek-r1-0528-qwen3-8b` or `gpt-oss-20b`, so those
  two rows have no summary to agree or disagree with; their recomputed figures stand alone.
- The `replay`, `curves`, `transplant` and `anchor` blocks carry their own denominators in
  their own shapes rather than a single `n`, so the arm column here covers only the four
  forced-answer arms that expose one. A per-arm audit of the remaining four is not done.
- The full transcripts stay on the cluster. Only the small per-cell files are mirrored into
  `docs/wave1-artifacts/`, so anyone rechecking a row from this repo alone can recheck the
  preflight and the throughput but not the records.


## Appendix: the preflight re-evaluation, verbatim

```
deepseek-r1-0528-qwen3-8b      re-evaluated PASS   stored PASS   agree=True | c=1 30/30 diff 0.0 (120 compared, 0 missing); c=32 30/30 diff 0.0 (120 compared, 0 missing)
deepseek-r1-distill-llama-8b   re-evaluated PASS   stored PASS   agree=True | c=1 30/30 diff 0.0 (120 compared, 0 missing); c=32 30/30 diff 0.0 (120 compared, 0 missing)
gemma-2-9b-it                  re-evaluated PASS   stored PASS   agree=True | c=1 30/30 diff 0.0 (120 compared, 0 missing); c=32 30/30 diff 0.0 (120 compared, 0 missing)
gpt-oss-20b                    NO PROBE ON DISK
llama-3.1-8b-instruct          re-evaluated PASS   stored PASS   agree=True | c=1 30/30 diff 0.0 (120 compared, 0 missing); c=32 30/30 diff 0.0 (120 compared, 0 missing)
olmo-3-7b-think                re-evaluated PASS   stored PASS   agree=True | c=1 30/30 diff 0.0 (120 compared, 0 missing); c=32 30/30 diff 0.0 (120 compared, 0 missing)
phi-4-reasoning                re-evaluated PASS   stored PASS   agree=True | c=1 30/30 diff 0.0 (120 compared, 0 missing); c=32 30/30 diff 0.0 (120 compared, 0 missing)
qwen3-8b                       re-evaluated PASS   stored PASS   agree=True | c=1 30/30 diff 0.0 (120 compared, 0 missing); c=32 30/30 diff 0.0 (120 compared, 0 missing)

FAIL PROOF on deepseek-r1-0528-qwen3-8b (in memory; nothing on disk is touched):
  one non-identical completion     -> REFUSE ['concurrency 32: identical completions 29/30, and the rule needs 30/30']
  a 1e-9 letter-logprob diff       -> REFUSE ['concurrency 32: max abs letter-logprob difference 1e-09, and the rule needs exactly 0.0']
  one missing letter logprob       -> REFUSE ['concurrency 32: 1 letter logprob(s) missing from the comparison; a missing value is not an equal one']
  zero logprobs compared           -> REFUSE ['concurrency 32: 0 letter logprobs compared, so the logprob half of the check certifies nothing']
  the batch-invariant flag off     -> REFUSE ["VLLM_BATCH_INVARIANT on the probing client was '0', not '1'; the pinned serving mode of A3.6 is the flag ON, and a flag-off run is exploratory"]
  only 29 items probed             -> REFUSE ['the probe ran 29 items, not the 30 the rule names; a shorter set makes 30/30 unreachable and a longer one is a different measurement']
FAIL PROOF: every perturbation refuses
```
