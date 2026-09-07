# The wave feeder, the enrichment pass and the enrichment cell

Track G follow-up lane, 2026-09-07. Branch `trackg/feeder`, base `5d40e52`. Nothing here
was merged and nothing was pushed. One powered sweep wave exists (wave 1, jobs 826733 to
826740, submitted by hand at 14:29; 826733 to 826736 COMPLETED 0:0 by 15:39). This lane
submitted no sweep cell and no enrichment job. It did submit one exploratory judge, job
826859 on h100-47, which is section 4 and is the one thing here that spends a card.

Three pieces, in the order the campaign uses them.

---

## 1. `bcf/wave_feeder.sh`: the next wave, once

`bcf/wave.sh` answers "may this wave go out". It has no memory, so a poll loop that calls
it every fifteen minutes submits the same wave again every fifteen minutes and Slurm
accepts every copy. The feeder is the memory: it picks the next wave file that is not in
its state file, asks `wave.sh --check-only`, submits at most one wave, and records the job
ids before it prints anything.

### Where it runs

**On the cluster.** `wave.sh` uses associative arrays, which macOS's bundled bash 3.2 does
not have, and both scripts read `squeue`. From the Mac it is driven over ssh.

### The one command for a 15-minute poll

Until this branch is merged, this is the command that runs today, and it is the one that
produced the live refusal at 15:39 in `docs/trackg-proofs/feeder_cluster_proofs.txt`:

```bash
bash bcf/ssh_retry.sh --wall 240 --tries 2 --gap 20 -- \
  'bash $HOME/bcf/src-feeder/bcf/wave_feeder.sh --pool a100-40 \
     --repo-tree $HOME/bcf/repo-5d40e5224ac0'
```

`~/bcf/src-feeder` is a copy of this branch, byte-identical in `bcf/wave_feeder.sh` and
`bcf/wave.sh` to the committed files (sha256 `ca592934...` and `b070bcaa...`, checked on
the cluster at 15:38). It is not a git checkout, so `wave.sh` cannot sync a tree from a
commit and `--repo-tree` names one it already synced: `~/bcf/repo-5d40e5224ac0`, the tree
wave 1 reads, made by `wave.sh` from commit `5d40e52`. Never point it at `~/bcf/repo` or at
a working tree.

After the merge, `~/bcf/src` is the orchestrator's checkout at the commit CI is green on,
the feeder ships with it, `wave.sh` syncs the per-commit tree itself, and the poll shortens
to:

```bash
bash bcf/ssh_retry.sh --wall 240 --tries 2 --gap 20 -- \
  'cd $HOME/bcf/src && bash bcf/wave_feeder.sh --pool a100-40'
```

### What the exit code means

| Exit | Meaning | What a poll loop should do |
|---:|---|---|
| 0 | a wave was submitted; its ids are on stdout and in the state file | log it and carry on |
| 3 | refused this round: no MaxSubmit headroom, `wave.sh` said no, or another feeder holds the lock | nothing is wrong; poll again |
| 4 | nothing left: every wave for this pool is already in the state file | stop polling this pool |
| 2 | usage or configuration error | fix the command |
| 1 | a submission was attempted and something went wrong | read the output |

### The state file

`~/bcf/feeder-state.json` (override with `--state`). One entry per submitted wave with its
job ids, the planning commit, the job type, the pool and a timestamp. Written atomically
(temp file, then `mv`) so a dropped link leaves the previous file rather than half of one,
and a wave already in it is never resubmitted whatever the queue looks like. A lock
directory beside it stops two pollers racing.

A wave submitted **by hand** has to be adopted, or the feeder will offer it again:

```bash
bash $HOME/bcf/src-feeder/bcf/wave_feeder.sh adopt a100-40-01.tsv \
  826733,826734,826735,826736,826737,826738,826739,826740
```

Adopting a wave twice is refused (exit 8) rather than overwriting what is recorded.

### The refusals, proven

`docs/trackg-proofs/feeder_cluster_proofs.txt`, run against the live queue at 14:55.

* **A** fresh state, wave 1 is next, `wave.sh` refuses (bcf-sweep 8/8 on a100-40): exit 3.
* **B** adopt wave 1: exit 0. **B2** adopt it again: exit 8, nothing overwritten.
* **C** the next wave is now 02 and the split still refuses it: exit 3.
* **D** the submit path against an injected empty queue, `--dry-run`: exit 0, "wave.sh
  accepts a100-40-02.tsv and this is where it would be submitted", no sbatch, no state
  write.
* **E** the injected counters cannot cause a real submission: exit 2.
* **LIVE, 15:39**, no injection and no `--dry-run`: with four wave-1 cells finished the
  feeder asked for real and `wave.sh` still refused (bcf-sweep 3/8 running, the wave wants
  8, 11 > 8), exit 3, jobs in system 7 before and 7 after, `a100-40-02.tsv` still next.
  This is the refusal the poll loop will see for as long as the pool is busy.

The MaxSubmit floor is 8 free slots, wider than `wave.sh`'s own edge, because the 33rd
sbatch is rejected rather than queued and an automated poll should stop well before that.

### What changed in `wave.sh`

Three additions, all in the same commit as the feeder.

1. **The `enrich` job type** with its own CONTRACT budget (`enrich_a100-40` = 8). An
   unlisted pool still refuses with "a planning gap, not a default" (exit 2).
2. **The pool guard.** The three existing checks count every campaign's RUNNING cards of
   the type, every campaign's RUNNING cards of every type, and this job type's own cards
   counting pending. None counts this campaign's OTHER job types' pending cards, and that
   is exactly the hole an enrichment wave opens: 8 pending sweep cells plus 8 enrichment
   jobs is 16 against a cap of 8, and Slurm queues them rather than rejecting them. The
   guard counts every `bcf-*` card of the type, running and pending, across job types. It
   is **on for `enrich`** and opt-in elsewhere via `BCF_WAVE_POOL_GUARD=1`: turning it on
   for the existing types would refuse waves they submit today (an a100-80 sweep wave
   beside pending `bcf-jury` cards, for one), and that is a CONTRACT.md decision rather
   than this script's.
3. **Injectable counters**, `BCF_WAVE_FAKE_INSYSTEM` and `BCF_WAVE_FAKE_CARDS`
   (`"a100-40=0,gpu=0,own=0,pool=0"`), the same shape `bcf/jury_wave.sh` already has. Any
   injected counter forces `--check-only`; asking for a real wave with one set refuses
   with exit 2, because an injected count that produced a real sbatch would be the worst
   bug this file could carry.

---

## 2. The enrichment pass: `bcf-enrich-<model>-<substrate>`

Ruling R3(ii). The element 9.2 sampling arm **alone**, k = 32 at temperature 0.7 on the
clean prompt, one request per item, over the whole 1,500-item pool of one substrate, for
one model, on one a100-40 slice, under the pinned serving mode with the determinism
preflight in front of it.

```bash
bash $HOME/bcf/src-feeder/bcf/wave_feeder.sh --pool a100-40 --type enrich \
  --repo-tree $HOME/bcf/repo-5d40e5224ac0
# or by hand:
bcf/wave.sh --type enrich --gpu-type a100-40 bcf/waves/enrich-a100-40-01.tsv
```

| Wave | Substrate | Jobs |
|---|---|---:|
| `enrich-a100-40-01.tsv` | arc_challenge | 8 |
| `enrich-a100-40-02.tsv` | aqua_rat | 8 |
| `enrich-a100-40-03.tsv` | logiqa2 | 8 |

Eight small models times three substrates is 24 jobs; the a100-40 cap is 8 cards, so they
go as three waves of eight, one substrate per wave. The files are generated, not typed:

```bash
python bcf/plan_enrich_waves.py            # write them
python bcf/plan_enrich_waves.py --check    # exit 1 if what is on disk differs
```

The roster and the pinned revisions come out of `bcf/waves/a100-40-01.tsv` verbatim (a
model whose revision differed between the sweep and its enrichment would enrich a cell
from a different model), and the price comes out of `plan.json`'s own `enrichment_pass`
block.

**The price, per job.** Sampling 0.889 h, from the measured 2.1333 s per item (job 826025
`throughput.json`, arm `sampling`: 30 calls in 64.0 s at concurrency 32 with the flag on).
That is the figure R3 prices. The runner also makes its clean substrate pass and its cue
pass before any arm, which R3 does not price: 0.171 h over the pool plus 0.159 h over the
clean-correct subset, at the measured 0.41005 s per full generation and the measured 28/30
retention. `BCF_EXPECTED_HOURS` carries the 1.219 h total, which is what a wall clock has
to cover. Per wave that is 9.754 card-hours, and 29.3 for all three. Every figure is a
floor for a model with no throughput measurement of its own.

**Where it writes.** `$BCF_RESULTS/enrichment/<model>/<substrate>/stated-hint`, through
`BCF_OUT_SUBROOT=enrichment`. The pass runs under the frozen stated hint because the
runner always makes a cue pass and the results path always carries a cue family, so its
model, substrate and cue are the same three values as the regular stated-hint sweep cell's
and without the subroot the two would share a directory, a checkpoint and a summary file.

**What it leaves behind.** Each job ends by running `bcf/enrich_report.py`, which reads the
per-item sampling blocks and writes, beside the arm artifacts:

* `enrichment_items.jsonl`, one line per scored item: pool index, question hash,
  normalized entropy, modal answer, whether the mode is correct, the stratum, and the
  right-but-uncertain flag at 0.30;
* `uncertain_items.json`, the **list file**: exactly the right-but-uncertain items, in pool
  order, in the schema the runner reads;
* `enrichment_report.json`, the counts and their denominators;
* `enrich_report_exit.txt`, its own exit code, kept apart from the arm's, because a pass
  whose arm finished and whose report failed is a pass whose data is intact.

The report computes no entropy of its own. Every per-item number is the one
`summarize_item_samples` already wrote; what the report adds is the pool index, matched
back through the `(question, choices)` key the resume merge uses. An item with no sampling
block is counted in the accounting, never silently dropped.

---

## 3. The enrichment cell: `BCF_ITEM_LIST`

Ruling R3(ii) again, the other half. The hinted arms run on **exactly** the items the pass
found, in addition to the cell's regular n.

```bash
sbatch ... --export=ALL,...,BCF_ITEM_LIST=$BCF_RESULTS/enrichment/<model>/<substrate>/stated-hint/uncertain_items.json ...
# equivalently, by hand:
python experiments/08_additive_arms.py ... --item-list <that file>
```

**Selection is position plus a question hash.** The pools carry no id of their own, so an
index is the only cheap selector; an index alone silently selects different items after a
refetch. Every entry carries both, the file records the whole-pool sha256 and the pool
size, and a mismatch refuses with the offending index named and **zero model calls**. A
list file that is named and missing refuses at the sbatch level with exit 11, before the
weights are downloaded, because a cell that quietly ran on the regular n under an
enrichment name is worse than one that never started.

**Every record carries `enrichment`** (true or false, written on every record rather than
only the enriched ones, so a reader never infers a population from a missing field), and
the summary reports the two apart:

```json
"enrichment": {
  "n_records_regular": 0,
  "n_records_enrichment": 47,
  "n_clean_correct_regular": 0,
  "n_clean_correct_enrichment": 44,
  "item_list": {"path": "...", "sha256": "...", "n_selected": 47, "indices": [...]}
}
```

That is R3(iii)'s guard: the regular-cell estimands are computed on the regular n only, and
the enrichment count is recorded per cell, so the two populations are never silently
merged. A regular cell writes the same block with a zero enrichment count and a null item
list, so the split is never missing by omission.

The item list is part of the **resume fingerprint**, so a second leg pointed at a different
list refuses rather than merging two populations into one checkpoint.

**One floor worth knowing before the pass runs.** The runner will not run an arm on fewer
than three clean-correct records. A model whose uncertain stratum on a substrate holds one
or two items therefore has no enrichment cell at all; that is R3(iii) arriving at the cell
rather than at the verdict, and `tests/test_enrichment.py` pins it so it is not discovered
as an empty results directory.

---

## 4. The gpt-oss exploratory judge on h100-47 (R7 extension)

`bcf/judges_gate_h100_explore_gptoss.tsv`, one row, submitted through `bcf/jury_wave.sh`
from `~/bcf/repo-jury-h100`, a copy of the jury tree synced for this line. The row pins
`BCF_REPO` and `BCF_ENV_SH` to that tree: `judge_serve.sbatch` defaults both to
`~/bcf/repo-jury`, which is the tree the W2f lane's own jobs read.

```bash
bash bcf/ssh_retry.sh --wall 180 --tries 2 --gap 15 -- \
 'cd $HOME/bcf/repo-jury-h100 && BCF_JURY_SBATCH=$HOME/bcf/repo-jury-h100/bcf/judge_serve.sbatch \
  bash bcf/jury_wave.sh bcf/judges_gate_h100_explore_gptoss.tsv'
```

**Refused at 14:56 by a cap, submitted unchanged at 15:39.** The first `jury_wave.sh
--dry-run` read gpu cards total **12/12** counting RUNNING and PENDING across every
campaign (8 sweep cells, 2 jury gates, 2 alta jobs), so one more card would reach 13 over
the QOS cap of 12 and it refused with exit 1; h100-47 itself was free, 0 of 4, and nothing
was cancelled to make room. Wave-1 cells 826733 to 826736 finished between 14:57 and
15:39, the count fell to 7/12, and the same row went in with no edit:

* dry run 15:39:54, exit 0, "every cap holds; submitting 1 job(s)", h100-47 0/4 -> 1/4.
* real submit 15:39:54, exit 0, **job 826859**, Slurm name `bcf-jury-gpt-oss-20b`,
  partition `gpu-long`, RUNNING within the minute.

Both are in `docs/trackg-proofs/h100_judge_gate_attempt.txt` with the queue either side.
The Slurm job name is the judge default, so tell this job from the pinned jury gates by
**id** and by its slug, not by name.

**The mxfp4 question is answered, and the answer is no.** 826859's server log reads
`quantization=gpt_oss_mxfp4` and `Using 'TRITON' Mxfp4 MoE backend`, the same backend as on
the a100-80 (job 826026). vLLM did not refuse the quantization on this card class and did
not dequantize to bf16.

**826859 still died, exit 5, over memory rather than quantization.**

```
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`
```

`--gpus=h100-47` hands out a **MIG 3g.47gb slice** of an H100 NVL, and section 6.1 gives
gpt-oss-20b `gpu_memory_utilization=0.25` because that table co-hosted it with Gemma on one
a100-80. 0.25 of 47 GB is 11.7 GB against 13.03 GiB of weights: the model loads, the KV
cache gets nothing. R7 dissolved that pair, so the 0.25 is a leftover of a line that no
longer exists.

The fix is a row-level override that only an exploratory row can use.
`bcf/judge_serve.sbatch` reads `BCF_JUDGE_GPU_UTIL` and takes it **only** when the row's
`BCF_SERVING_LINE` starts with `exploratory`; anything else exits 12 with the table's
fraction named in the refusal, so a run of record cannot be served at a number the
pre-registration does not state. `experiments/jury/family_map.py` is untouched: the 6.1
table still says 0.25. Proven both ways in
`docs/trackg-proofs/util_override_falsification.txt`, which extracts the block from the
sbatch file rather than retyping it: exploratory plus override takes 0.90 (exit 0), the
pinned line plus the same override refuses (exit 12), an unset `BCF_SERVING_LINE` refuses
(exit 12), and no override leaves 0.25 (exit 0). Regenerate it with
`bash bcf/util_guard_proof.sh bcf/judge_serve.sbatch`.

Resubmitted at 15:49:59 with `BCF_JUDGE_GPU_UTIL=0.90` in the row: **job 826888**, dry run
exit 0 (h100-47 2/4 -> 3/4), submit exit 0.

**Collision to know about.** Another lane ran the command above at 15:47:02 and got **job
826880**, the same row, the same tree and the same `BCF_OUT_SLUG`, but submitted two
minutes before the patched sbatch reached the cluster, so it carries no override and fails
the same way 826859 did. Two jobs with one slug write one results directory. This lane
cancelled neither (its cancel permission names `bcf-enrich-*` and `bcf-jury-gptoss-h100-*`,
and `jury_wave.sh` names both of these `bcf-jury-gpt-oss-20b`). 826880 dies on its own a
few minutes after it starts serving; if it is somehow still alive when 826888 starts, one
of the two should be cancelled by whoever owns it, and it should be 826880.

When it runs it writes three results directories, one per Q1 prompt variant:
`gpt-oss-20b-h100-47-q1a`, `-q1b`, `-q1c`. The collector fetches them the usual way:

```bash
bcf/w2e_resume.sh fetch gpt-oss-20b-h100-47-q1a gpt-oss-20b-h100-47-q1b gpt-oss-20b-h100-47-q1c
```

If vLLM refuses mxfp4 on this card class, the server log line is what says so, and it is in
`server-*.log` in the first of those directories. On the a100-80 (job 826026) the log read
`Using 'TRITON' Mxfp4 MoE backend` with `Model loading took 13.03 GiB`, so vLLM neither
refused the quantization nor dequantized to bf16 there; whether the same holds here is not
known until the job runs.

---

## What this lane did not do

No sweep cell and no enrichment job was submitted: every wave file here is still
unsubmitted, and the live feeder run at 15:39 was refused by the split with nothing sent.
The one job this lane put in the system is the exploratory judge 826859 of section 4, which
R7 allows on a free pool and which is not a run of record. Nothing was cancelled. `~/bcf/src`,
`~/bcf/repo`, `~/bcf/repo-jury` and every `~/bcf/repo-<sha>` tree were left untouched: the
check-only proofs ran from `~/bcf/src-feeder`, a copy of this branch, against
`--repo-tree ~/bcf/repo-feeder-proof`, a path that does not exist and was never created
because `--check-only` writes nothing; the live feeder run named the real
`~/bcf/repo-5d40e5224ac0` and never got as far as reading it; the judge row reads
`~/bcf/repo-jury-h100`, a separate copy, so the W2f lane's `~/bcf/repo-jury` is not shared.
The frozen files are byte-identical to main and `tests/test_frozen_guard.py` passes.
