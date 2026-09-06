# Track G: what one served card does when more than one request is in flight

Written 2026-09-07 from job 825548 (`bcf-trackg-probe`, exit 0, sacct State COMPLETED,
ExitCode 0:0, ReqMem 64G, MaxRSS 20,600,744 K, Elapsed 00:06:23, node xgph10). Every
number below is read back from
`bcf/measured/trackg-probe-825548/probe_results.json`, which the job wrote, and from
`bcf/measured/trackg-probe-825548/sacct.txt`.

**This file makes no ruling.** It reports what the card did. Whether a pre-registered
cell may be run at a concurrency above 1 is the operator's call.

## Why the probe exists

`experiments/openai_client.py` opens one URL per call and `experiments/08_additive_arms.py`
had no thread anywhere in it, so every generations-per-second figure this campaign has
recorded (CONTRACT.md's "Measured throughput" section, `throughput.json` in runs A and B,
Amendment A3.6) is the rate of a client that waits for each answer before asking the next
question. A vLLM server batches. Those figures therefore bound the client, not the card,
and the ratio between them had never been measured.

## What was held fixed

| Parameter | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen3-8B` | run_meta.json |
| Revision | `b968826d9c46dd6066d109eabc6255188de91218` | run_meta.json, matches element 10 |
| Backend | vLLM 0.28.0, tensor-parallel 1 | run_meta.json |
| GPU | one MIG 3g.40gb slice of an A100 80GB PCIe, node xgph10 | run_meta.json, run.log |
| Serving flags | `--max-model-len 8192 --gpu-memory-utilization 0.90 --seed 7` | run.log, same as the skeleton |
| Sampler | `VLLM_USE_FLASHINFER_SAMPLER=0` | run.log |
| Decoding | temperature 0.0, `num_predict` 320, `enable_thinking` false | run_meta.json |
| Items | the first 30 of `~/bcf/repo/experiments/data/arc_challenge.json` | run.log |
| Work per level | 30 full generations, then 30 forced-continuation logprob items (4 letters each, 120 HTTP calls) | probe_results.json |

Only the number of requests in flight changed. The concurrency-1 pass is the baseline the
other three are compared against, and it ran first.

## The numbers

| Concurrency | Generation wall (s) | Full generations/s | Speedup | Logprob wall (s) | Logprob calls/s | Completions identical to concurrency 1 | Letter logprobs exactly equal | Max abs letter logprob difference | Median abs difference | Retries | Failed requests |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.768 | 0.3539 | 1.000 | 8.580 | 13.9855 | 30/30 | 120/120 | 0.0 | 0.0 | 0 | 0 |
| 8 | 15.900 | 1.8868 | 5.331 | 3.974 | 30.1927 | 11/30 | 52/120 | 0.625 | 0.125 | 0 | 0 |
| 32 | 5.029 | 5.9650 | 16.855 | 2.749 | 43.6545 | 13/30 | 28/120 | 0.875 | 0.125 | 0 | 0 |
| 64 | 5.030 | 5.9647 | 16.854 | 2.780 | 43.1726 | 13/30 | 28/120 | 0.875 | 0.125 | 0 | 0 |

Denominators: 30 generations per level (30 items entered, 30 completions returned at every
level, 0 dropped); 120 letter logprobs per level (30 items times 4 answer letters, 0
missing at every level); comparisons are against the concurrency-1 pass of the same 30
prompts on the same server process.

## Three things worth reading off the table

**The sequential rate reproduces.** 0.3539 full generations per second at concurrency 1
against 0.3529 for the `clean_substrate` arm of job 825511 (30 calls in 85.0 s,
`bcf/measured/run-a-825511/throughput.json`). The probe and the skeleton agree to within
0.3 percent on the same slice and the same model, so the baseline is the same measurement,
not a new one.

**The card is about 17 times the sequential client, and it saturates between 32 and 64.**
5.9650 against 5.9647 generations per second is a difference of 0.005 percent across a
doubling of in-flight requests, so 32 already fills this slice at these prompt lengths.
The forced-continuation path saturates earlier and lower: 43.65 calls per second at 32
against 43.17 at 64, and only 3.1 times the concurrency-1 rate of 13.99, because those
calls are 24 tokens each and the fixed per-request cost dominates.

**Batching changes the outputs at temperature 0.** At concurrency 8, 19 of 30 completions
differ from the sequential run of the same prompt on the same server. At 32 and 64, 17 of
30 differ. On the letter logprobs the divergence is not a rounding artifact: 68 of 120
values move at concurrency 8 and 92 of 120 move at 32 and 64, the median move is 0.125
nats and the largest is 0.875 nats. The differences land on multiples of 0.125, which is
what a coarsely quantized reported logprob looks like, but the probe did not test that and
this file does not claim it.

## What that costs, priced

`bcf/plan_waves.py` prices the whole 216-cell grid at a chosen concurrency and refuses any
level the probe did not measure. At concurrency 1 the grid is 1,258.4 card-hours. The
same grid at the 16.855 speedup would be about 74.7 card-hours. The wave manifests in
`bcf/waves/` are generated at concurrency 1, which is the only probed level whose
completions and letter logprobs match the sequential client exactly, and
`bcf/waves/plan.json` carries the full four-level table so the trade is visible.

## What this does NOT establish

- Only `Qwen/Qwen3-8B` on one MIG 3g.40gb slice was probed. The 24B-to-35B, 70B and 120B
  classes have no throughput measurement at any concurrency, and neither does a whole
  a100-80 card: job 825510 is still PENDING with reason Resources and its output directory
  `~/bcf/results/phase1-skeleton-fullcard/qwen3-8b/arc_challenge` is empty.
- The divergence at concurrency 8 and above was measured on 30 prompts, once. It is not a
  rate estimate with an interval, and no claim is made about how it would behave on 1,500.
- Nothing here says whether the divergence matters for any estimand. The anchor arm reads
  letter logprobs and the curve arm reads forced-continuation answers, so the question is
  live, and it is the operator's.
