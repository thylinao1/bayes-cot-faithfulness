# Track G readiness, 2026-09-07

Branch `trackg/waves`, base `98141b1` (local main). Nothing here was pushed, nothing was
merged, and no powered pre-registered run was submitted. The one cluster job this lane
spent is the 30-item concurrency probe.

## Ready

**1. The runner can put more than one request in flight.**
`experiments/08_additive_arms.py` grows `--concurrency N`, default 1. At 1 it takes the
pre-concurrency code path, not a one-worker pool, so existing behavior is the same code
rather than an equivalent one. Above 1, `map_in_order` keeps at most N workers running
while handing every result to the consumer in item order, so records are written in item
order, a checkpoint always holds a prefix, and `--resume` is untouched. Nine per-record
arms and both passes over items go through it: clean substrate, cue, replay, placebo,
direct, twostep, filler, curves, transplant, anchor, and the A9 specificity holdout.
Twenty tests in `tests/test_arm_concurrency.py`, including the whole arm set serialized
at 1 and at 4 against the same deterministic fake backend and compared byte for byte
after `serialize_arm_record`. The check was proven able to fail: changing one
`popleft()` to `pop()` gave 3 failed, 16 passed, exit 1
(`docs/trackg-proofs/concurrency_test_falsification.txt`).

**2. The client is safe to share across threads and retries what it could not before.**
`experiments/openai_client.py` gets a lock around the JSONL request log, an optional
in-flight semaphore, exponential retry backoff, and `stats()` counting successes, retry
attempts and hard failures. Running the new suite found a real gap: urllib wraps only the
failures it sees while opening a connection, so a reset during `resp.read()` arrived as a
bare `OSError` and a truncated body as `http.client.IncompleteRead`, an `HTTPException`
and not an `OSError` at all. Neither was caught, so a transport blip killed an arm instead
of costing one retry. Both are retried and counted now.

**3. The card's real capacity is measured, and so is what it costs.**
Job 825548, exit 0. Numbers, denominators and caveats in `docs/TRACKG-THROUGHPUT.md`;
raw artifacts in `bcf/measured/trackg-probe-825548/`. Headline: 0.3539 full generations
per second at concurrency 1 (reproducing job 825511's 0.3529 on the same slice to within
0.3 percent), 5.965 at 32, saturated by 64, a 16.855 speedup. And batching changes the
outputs at temperature 0: 13 of 30 completions identical at concurrency 32, 28 of 120
letter logprobs exactly equal, largest move 0.875 nats. **This lane makes no ruling on
that.** The wave manifests are generated at concurrency 1, the only probed level whose
completions and letter logprobs match the sequential client exactly.

**4. The pools are at 1,500 and the first 700 indices are pinned.**
`arc_challenge` 1,500 (test 1,172 + validation 299 + train 1,119, deduplicated),
`aqua_rat` 1,500 (test + dev + train), `logiqa2` 1,500 (test, 14 duplicate keys dropped of
1,572 parsed). All three: 1,500 unique resume keys of 1,500. The first 700 items of each
are byte-identical to the pools every Phase-1 artifact was measured against, verified two
ways, by sequence hash against the shipped manifest and by direct list comparison.
`fetch_arc.py` gained `--then-split` with the same first-occurrence dedup the other two
fetchers already use. `scripts/write_pool_manifest.py` regenerates the manifest and
REFUSES when the prefix moves; proven by swapping items 3 and 4 (pytest 1 failed exit 1,
manifest check exit 1, generator exit 1) and restoring
(`docs/trackg-proofs/pool_check_falsification.txt`).

**5. The whole grid is planned, priced and dry-run.**
18 models x 3 substrates x 4 cue families = 216 cells, `n` 1,500 for stated-hint and
professor and 570 for metadata and grader-code (DECISION-LOG ruling (b)), grouped into 96
wave manifests under `bcf/waves/`. Each TSV carries the pinned revision, the arm list, n,
curve cap, pool, tensor-parallel size, quantization, extra vLLM flags, host memory, CPUs
and the projected hours, with a human-readable comment block above the machine-readable
rows. `bcf/waves/plan.json` carries the dependency order and the card-hours.

| Pool | Cells | Waves | Card-hours at concurrency 1 |
|---|---:|---:|---:|
| a100-40 | 96 | 12 | 503.4 |
| a100-80 | 72 | 36 | 377.5 |
| h100-96 | 36 | 36 (12 single-card, 24 tensor-parallel) | 314.6 |
| h200-141 | 12 | 12 | 62.9 |
| **total** | **216** | **96** | **1,258.4** |

Cost basis: the per-arm call counts and per-call seconds of job 825511 (2.961 s per full
generation, 0.1452 s per forced continuation, 1,520 calls in 660.0 s), scaled by item
count. The 24B-to-35B, 70B and 120B classes have no throughput measurement of their own,
so their rows are a FLOOR and are marked `measured_for_this_class` false.

**6. `bcf/wave.sh` refuses, and the refusals were exercised.**
It now checks four caps: jobs in system (32), the per-user card cap for the GPU type
counting every campaign, the total gpu cap of 12 across types, and the CONTRACT split by
job type AND pool (sweep gets 2 of the 4 a100-80 cards because the judges hold the other
2). Before any of that it refuses a tensor-parallel size larger than the cards on one node
of that type, and it now sets `--time` under the partition ceiling instead of inheriting
the sbatch header's 24 h. It passes `MEM` and `CPUS` as sbatch flags rather than exports,
because exporting a variable reserves no memory and the partition default of 3 G kills a
vLLM engine core with nothing legible in the server log. It hands every command line to
`sbatch --test-only`, which validates the flags without allocating anything.

All 96 waves were run in check-only mode on the cluster: 60 pass, 36 refuse, outputs in
`bcf/waves/dry-run-<wave_id>.txt`. The 36 refusals are all a100-80 and all for the same
live reason: the alta campaign held 3 of that pool's 4 cards, so 3 + 2 = 5 > 4. Three
deliberate refusal probes each exit 1
(`bcf/waves/refusal-probes/EXIT_CODES.txt`): 9 cells on a pool capped at 8, a
tensor-parallel-2 request on a100-80 where every node has one card, and `--time=24:00:00`
on the h200 partition whose ceiling is 3 h.

## What the operator's push unlocks

The freeze commit is the A1 precedent for the first powered wave. Once it lands, the first
command is:

```
ssh soc
cd ~/bcf/repo-trackg && bash bcf/wave.sh --type sweep --gpu-type a100-40 \
    --check-only bcf/waves/a100-40-01.tsv
# read the output, then drop --check-only to submit the same 8 jobs
```

That wave is 8 cells, one per small model, arc_challenge / stated-hint at n 1,500, 8 of
the 8 a100-40 cards, 60.68 card-hours, and it passed check-only at 03:00 with the pool at
0 of 8 in use. It depends on nothing else in the campaign.

Before that command is run for real, two things must be true and are not yet:
`~/bcf/repo-trackg` must be the repo the jobs read (the dry runs point
`serve_and_run.sbatch` at `~/bcf/repo`, which is the live path other lanes use), and the
1,500-item pools must be in place there. Neither was done tonight: rsyncing over
`~/bcf/repo` while jobs read it is exactly what the standing instruction forbids.

## Open items, by name

1. **The batch determinism ruling.** Concurrency 32 is 16.855 times faster and changes 17
   of 30 completions and 92 of 120 letter logprobs. `bcf/plan_waves.py --concurrency 32`
   reprices the grid; nothing else changes. Operator's call.
2. **The 1,500-item pools are not on the cluster.** They exist only in this worktree.
   `~/bcf/repo/experiments/data/*.json` still holds the 700-item pools, and overwriting
   them is a live-job hazard while jobs are reading that tree.
3. **ARC choice count changed.** The 700-item test-only pool had at most 4 choices; the
   1,500-item pool reaches 5, because the train and validation splits carry 5-option
   items. The runner's parse bound is `max(len(choices))` over the entered items, so an
   ARC cell at n 1,500 parses against 5 labels where the skeleton parsed against 4.
   Nothing breaks, but it is a change to the frozen prompt surface's label set and it
   should be stated in an amendment rather than discovered in a summary.
4. **Job 825510, the whole-a100-80 skeleton.** Still PENDING, reason Priority, StartTime
   estimated 2026-09-07T04:22:50, TimeLimit 02:00:00. Its output directory
   `~/bcf/results/phase1-skeleton-fullcard/qwen3-8b/arc_challenge` is EMPTY. The
   a100-80-versus-MIG ratio is therefore unmeasured, and every a100-80 row in the plan is
   priced at the MIG rate.
5. **Job 825253, the tensor-parallel-2 serving test.** Still PENDING,
   Reason `ReqNodeNotAvail,_UnavailableNodes:xgpd2,xgpe0`, TimeLimit 00:50:00, StartTime
   estimated 2026-09-09T05:19:11, ReqTRES `gres/gpu:h100-96=2`, `mem=3G`. NOT resubmitted,
   per the brief. Two things about it are worth recording. Its `mem=3G` is the partition
   default that killed judge job 825536, so if it does start it may die the same way. And
   `sbatch --test-only` on the identical `--nodes=1 --gpus-per-node=h100-96:2` line answers
   "Requested node configuration is not available" right now, which is an availability
   answer and not a configuration one: every h100-96 node currently has both of its cards
   allocated, and the variant `--gpus=h100-96:2` is accepted only because Slurm plans it
   across TWO nodes, which is useless for tensor parallelism. The 24 `h100-96-tp2` waves
   are blocked on that job, and element 16's roster step is the stated fallback.
6. **GLM-4.5-Air needs `--resume` legs.** The only h200-141 node sits in partition `gpu`
   alone, ceiling 3 h, and its n=1,500 cells are projected at 7.585 h. `wave.sh` now says
   how many legs and requests 02:50:00; `serve_and_run.sbatch` always passes `--resume`,
   so this is a resubmit count and not a blocker. The FP8 repository id for that row is
   also unresolved: element 10 pins `zai-org/GLM-4.5-Air` at revision `a24ceef6...` and
   describes it as FP8, and the plan carries `BCF_QUANT=fp8` against that id without having
   verified that the id serves FP8 weights.
7. **Phi-4-reasoning on a100-40 is unproven.** 14B in bf16 is about 29 GB against 39.25 GiB
   usable on a MIG 3g.40gb slice, so the KV cache is thin. The plan carries
   `--max-num-seqs 64` for it. No serving test has been run.
8. **Per-cell server startup is not in the projection.** One job per cell means 216 vLLM
   starts. The probe's server answered 4 minutes after `vllm serve` on a warm cache and a
   MIG slice; 216 of those is about 14 card-hours, on top of the 1,258.4, and more on a
   cold weight download.
9. **Gated licences: checked, and they hold.** Four roster rows are gated, and
   `~/Developer/bayes-cot-phase2/PERMISSIONS.md` carries all four as "ACCEPTED by operator
   2026-09-06 (chat)" with the revision prefix on each row: `meta-llama/Llama-3.1-8B-Instruct`
   (rev 0e9e39f249), `meta-llama/Llama-3.3-70B-Instruct` (rev 6f6073b423),
   `google/gemma-2-9b-it` (rev 11c9b309ab) and `google/gemma-3-27b-it` (rev 005ad3404e).
   Every one of those prefixes matches the revision the wave manifests pin. Nothing is
   blocked here; it is recorded because element 10 makes it a precondition.
