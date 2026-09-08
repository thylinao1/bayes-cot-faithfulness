# The ladder recipe check: what ran, what it measured, and what R14 has to decide

Lane: `ladder/recipe-check`, worktree `~/Developer/bcf-lrc` on main `b4de41a`, 2026-09-08.

**Everything this lane produced is EXPLORATORY.** No artifact here is a checkpoint of
record, no cell was evaluated, no held-out family was generated, and nothing here freezes
or rules on anything. This is step 1 of the ladder order the operator set on 2026-09-08
at 01:51 ("an EXPLORATORY recipe check, not of record, on one organism at the middle
dose for a few hundred steps, to confirm the LoRA recipe learns the trigger at all and to
read training loss and throughput"), and its whole purpose is to put measured numbers
beside the decisions ruling R14 has to make.

**Status at the end of the lane: the CPU half is done and the GPU half has not run.**
The traces, the training pool, the evaluation guard and one organism training set exist
and are measured (sections 2 and 3). The LoRA fit is job 828559: submitted at 09:21:48
after a printed cap check, still PENDING at 12:48:26 because 10 of the 11 a100-80 cards
are allocated and the 11th is the excluded xgpj0 (section 5.2). It is left QUEUED, it was
never cancelled, and `~/bcf/ladder-explore/collect.sh` collects it in one command. Every
row of section 6 that depends on it is marked NOTHING YET rather than filled with an
estimate.

Every number below carries its denominator and the job or file it came from.

---

## 1. What ran

| # | What | Where | Job | Result |
|---|---|---|---|---|
| 0 | Sync the cluster tree for `b4de41a` | `~/bcf/repo-b4de41a7a6a9` | none | done, `~/bcf/src` HEAD left at `f712a9beb1cb` |
| 1 | Rebuild the disjoint ladder training pool | login node | none | 1,077 items, sha256 matches the committed manifest |
| 2 | Reduce four Qwen3-8B ARC cells to traces | partition `normal` | 828544 | 1,395 of 3,935 records kept |
| 3 | Evaluation guard over the pinned ARC pool | partition `normal` | 828544 | 1,496 hashes for 1,500 items |
| 4 | One organism training set, rung 2, seed 1 | partition `normal` | 828544 | 1,077 examples, 0 with a banked trace |
| 5 | The env check and one LoRA fit on a100-80 | `gpu` | 828559 | **QUEUED, not started at 12:48; section 5** |

### 1.1 The tree, and the one deviation from the sync recipe

`bcf/wave.sh`'s sync was replicated by hand: `git -C ~/bcf/src fetch`, then
`git archive b4de41a | tar -x` into `~/bcf/repo-b4de41a7a6a9`, then the five gitignored
pool files copied in from `~/bcf/repo-f712a9beb1cb` (read only), then
`scripts/write_pool_manifest.py --check`, which printed `manifest matches the pools on
disk`, then the `.bcf_sync.json` marker.

**`git checkout` was deliberately NOT run in `~/bcf/src`.** The brief's setup line names
it, but `~/bcf/src` was at `f712a9beb1cb` and the running feeder loop names
`~/bcf/repo-f712a9beb1cb`; moving that checkout is exactly what broke the feeder at 08:56
this morning (DECISION-LOG 2026-09-08 09:03). `git archive <sha>` reads the commit
object and needs no checkout, so the tree is byte-identical to what the recipe would
have produced and no other lane was disturbed. `~/bcf/src` HEAD was printed before and
after and is unchanged.

### 1.2 The one patched file in that tree, and why

`bcf/ladder_train.sbatch` builds its argument list from a fixed template, and
`lora_train._train_peft` at `b4de41a` records only `loss_first` and `loss_last`: no loss
curve, no throughput, no peak memory, and no way to ask whether the trigger relation
moved. The recipe check needs all four. So this lane wrote them, additively, on
`ladder/recipe-check`, and copied the two files into its OWN tree:

| File | sha256 in the tree | sha256 at `b4de41a` |
|---|---|---|
| `src/.../ladder/lora_train.py` | `957c91aca4d233ba4fc5c6a113a25df0edcbaae2c2f538e2da606ab24d16e639` | `26d98b23a552ab6f9f74b87c537dcd3e8cc2fd92ff68198c326e11bd1f61cfd3` |
| `src/.../ladder/recipe_probe.py` | `1c39a267bbab5127f0019d2a7de9dfac6e867f8f4174d78173397dfdda2e7bf3` | did not exist |

The tree carries `.bcf_exploratory_patch.json` saying exactly that. No other campaign
reads that tree; every sweep and jury job reads `~/bcf/repo-f712a9beb1cb`, which was not
touched. The switches default OFF, so with no flags and no environment the training
report gains fields and loses none.

---

## 2. The traces (element 11's step 0c, on the wrong pool)

`experiments/reduce_traces.py` over the four Qwen3-8B ARC cells' `transcripts.jsonl`,
in the order stated-hint, professor, metadata, grader-code. Run as a CPU sbatch on
partition `normal` (job 828544, 2 cpus, `--mem=8G`), which is the partition the fits
lanes use. `/usr/bin/time -v` reported **max RSS 127,352 KB (124.4 MiB)** for the
reduction and the whole batch step finished in 10 seconds; `sacct` gives the step's
MaxRSS as 88,928 KB. It would have fitted on the login node with room to spare, and it
was run as a job anyway because that is the rule.

| Quantity | Value | Denominator |
|---|---|---|
| Records read | 3,935 | 1,415 + 1,415 + 553 + 552 across the four files |
| Unique items kept | **1,395** | of 3,935 records read |
| Dropped | 222 | of 3,935 |
| dropped `unparseable` | 0 | of 222 |
| dropped `missing_chain` | 0 | of 222 |
| dropped `duplicate_differing_trace` | **222** | of 222 |
| `question_sha16` collisions | 0 | the run did not refuse |

Output `~/bcf/ladder/traces/qwen3-8b-arc.json`, sha256
`b39c03ec283802066455d6dbab1f7b5da616d9d9a825555b4c2eacf3e5b466ef`, manifest mirrored to
`experiments/results/ladder-explore/traces-qwen3-8b-arc-manifest.json`.

### 2.1 Where the 222 disagreements are, which is not where you would guess

The reducer's own manifest names every dropped line, so the drops can be attributed
(`experiments/results/ladder-explore/traces-per-cell-attribution.json`):

| Cell | records | new items | identical to a kept trace | differing, dropped | disagreement rate |
|---|---|---|---|---|---|
| stated-hint | 1,415 | 1,392 | 19 | 4 | 4 of 23 within-file repeats |
| professor | 1,415 | 0 | 1,411 | 4 | 4 of 1,415 = 0.28% |
| metadata | 553 | 3 | 336 | **214** | 214 of 550 = **38.91%** |
| grader-code | 552 | 0 | 552 | 0 | 0 of 552 = 0.00% |

Two facts worth carrying forward, neither of which this lane rules on.

**The clean arm is not reproducible across all cells.** The professor and grader-code
cells reproduce the stated-hint cell's own clean-arm generation on 1,411 of 1,415 and
552 of 552 items, and the metadata cell disagrees on 214 of its 550 overlapping items.
The clean arm is the same prompt at temperature 0 in all four cells, so a 38.91 percent
disagreement in one cell and 0.00 percent in another is a serving-state difference, not
a decoding one. That is the shape the batching-determinism finding of 2026-09-07
predicts, and it belongs to the sweep lane, not to this one.

**23 records in the stated-hint file repeat a question hash already seen in that same
file** (19 byte-identical, 4 differing), while the pinned `arc_challenge` pool's first
1,500 items hold only 4 repeated question texts (the guard found 1,496 unique hashes for
1,500 items, section 3). So at least 19 of those repeats are not the pool's duplicates.
Reported as an observation for whoever owns `arms_resume`; nothing in this lane depends
on it.

---

## 3. The training set, and the coverage number that matters

`ladder_train_pool.json` did not exist anywhere: not on the cluster and not on the Mac.
It is gitignored for licence reasons and only its manifest was committed. It was rebuilt
from `bayes_cot_faithfulness.ladder.train_pool` on the login node (max RSS 111,116 KB,
9.54 s wall) and reproduces byte for byte:

* 1,077 items of 1,119 ARC-Challenge train candidates; 40 excluded by id, 1 more by
  normalised text, 1 over the four-choice cap, 0 internal duplicates;
* `file_sha256 ffd0f93d787d5bd5cb06a623e05454e4fbcf6b7373b6d173485ab7fb3b6283b5`, which
  is the value `experiments/data/ladder_train_pool_manifest.json` committed on
  2026-09-08. The pool is reproducible from the pinned fetcher, as its module claims.

The evaluation guard over `arc_challenge.json` (sha `7dbf9616004d...`) at n_items 1,500
holds **1,496 unique question hashes for 1,500 items**, so the pinned evaluation pool
itself carries 4 repeated question texts.

### 3.1 The build

One organism training set at the MIDDLE dose: variant `organism`, rung 2, dose 0.60,
seed 20260911, `n_examples` 1,077.

**`N_TRAIN_EXAMPLES` is 1,200 and the pool holds 1,077, so 1,077 was used and the lane
choice was not met.** That is section 5.3 option 1 of the pilot plan applied to one
exploratory build and it pre-empts nothing; the builder would have refused at 1,200.

| Quantity | Value | Denominator |
|---|---|---|
| Examples | 1,077 | the whole pool |
| Trigger present | 523 | 1,077, a frequency of 0.4856 against a requested prevalence of 0.50 |
| Followed the trigger | 317 | 523 trigger items, a rate of 0.6061 against a coupling of 0.60 |
| Target equals the trigger's option | 0.7132 | 523; expected 0.60 + 0.40 x 0.25 = 0.70 with 4 options |
| Answer information | 0.4855 nats | mutual information on the 523 trigger items |
| `of_record` (the builder's own field) | **true** | see 3.2 |
| `n_items_with_a_banked_trace` | **0** | of 1,077 |
| Traces available in the file | 1,395 | none of them usable here |
| Training-set hash verification | ok | `verify_training_set` |

### 3.2 The coverage is zero BY CONSTRUCTION, and `of_record` does not notice

The 1,395 banked traces are the clean-arm generations of the *evaluation* pool: the first
1,500 items of `arc_challenge.json`. The ladder training pool is built to share no item
with that pool, by id and by normalised text, and `build_training_set` refuses outright
if any training item is an evaluation item. So the intersection of "items with a banked
trace" and "items in the training pool" is **empty by construction, and 0 of 1,077 is the
only number it could have been.** Every completion in this build came from the template
fallback, `"1. Work through the options in order."`.

**And the build is still stamped `of_record: true`.** `trigger_data.build_training_set`
computes `"of_record": bool(traces)`, which is true for any non-empty dict, and the
manifest's own `of_record_note` says the opposite of what happened: "of_record is false
when the completions came from the template fallback". Here all 1,077 completions came
from the template fallback and the field says true. The DECISION-LOG entry of 01:44
already recorded this on the tiny synthetic pool ("of_record alone does not certify
coverage"); this is the same defect on the real pool, where it would silently stamp a
rung. The one-line fix is to make the field depend on the coverage the manifest already
computes rather than on the dict being non-empty, for instance

```python
"of_record": bool(traces) and n_items_with_a_banked_trace == len(examples),
```

with the partial-coverage case a refusal rather than a quiet true. **This lane does not
make that change**: `of_record` decides what may be read as a rung, so moving it is an
instrument decision and belongs to R14. It is listed in section 6 with the measured
input beside it. The exploratory artifacts here are stamped `of_record: false`
independently, by the run itself, so nothing this lane produced can be misread.

### 3.3 What banking the right traces would cost

Organism training needs the base model's own clean traces **on the training pool**,
generated by the base model itself, and no run has ever generated them. The pilot plan
already prices it and the arithmetic holds:

* 1,077 items x 1 full generation x 0.41005 s per generation (the measured rate of job
  826025) = **441.62 s = 0.1227 card-hours**, plus the model load;
* one `bcf/serve_and_run.sbatch` cell with `BCF_SUBSTRATE=ladder_train_pool`,
  `BCF_ARMS=replay`, `BCF_N_ITEMS=1077`, `BCF_OUT_SUBROOT=ladder-traces`, so the output
  is never counted among the 216 cells;
* then `experiments/reduce_traces.py` over that one transcripts file, which this lane has
  now run end to end on real data and which costs 2.14 s and 124 MiB.

Two conditions have to be met before it is spent, and neither is met today. The runner
reads `${BCF_REPO}/experiments/data/${SUBSTRATE}.json`, so `ladder_train_pool.json` has
to be inside the tree the job reads, not in `~/bcf/ladder/pools/` where this lane put it;
and the pool file is gitignored, so it travels by copy and is verified by its
`file_sha256`, exactly as the three evaluation pools already do. This lane did not
submit that job: it is a second GPU job and the lane's budget is one.

The consequence for the ladder, stated plainly: **the ladder cannot train a checkpoint of
record until that one 0.12-card-hour pass has run.** It is cheap, it is not optional, and
it is not in anyone's queue.

---

## 4. The cap and the wall arithmetic

The cap check was printed before the submission and is reproduced from the job log:

```
a100-80 RUNNING for this account (every campaign): 1     (826784, bcf-jury-qwen3-32b)
a100-80 PENDING for this account (every campaign): 2     (828495, 828496, both alta)
QOS cap for a100-80 per user: 4
lane rule: a100-80 RUNNING must be at most 2 of the 4 before this lane submits
CAP CHECK PASSES: 1 running + 1 from this lane = 2 of 4
```

Two of the four a100-80 slots stay free for the two pending alta jobs, which is the
point of the rule. The wall was capped at `--time=02:00:00`, overriding the 08:00:00 in
`bcf/ladder_train.sbatch`, and later lowered to 01:30:00 (section 5.1); `--mem=128G`,
`--exclude=xgpj0` and the a100-80 request are the file's own, unedited.

Against the ladder budget, this job is a NEW line and not one of the 12:

| Line | Card-hours | Source |
|---|---|---|
| LoRA, 12 checkpoints | 24.000 | 12 x 2.0 h, CONTRACT.md line 24 (ESTIMATE) |
| Evaluation, 12 cells | 8.172 | 12 x 0.681 h, `serve_manifest.expected_hours()` |
| Ladder total of record | 32.172 | |
| This recipe check | at most 1.500 | the wall cap now on job 828559; the run has not started, so nothing has been spent |
| Trace banking on the training pool | 0.123 | section 3.3, needed once regardless |
| **Total** | **at most 33.795** | against section 12.1's "about 35 card-hours per base" |

Card-hours SPENT by this lane so far: **0.000.** The CPU work of sections 2 and 3 ran on
partition `normal` and on the login node and cost no card.

Ruling R2 records the element 16 degradation trigger as NOT TRIGGERED at 261.7 priced
card-hours against a budget of record of about 650, a ratio of 0.403, so nothing is being
cut for compute and this check does not move that.

---

## 5. The GPU job: submitted, queued, NOT YET MEASURED

**Job 828559 was submitted at 09:21:48 and had not started by 12:48:26.** Nothing in
this section is a measurement, and no number is quoted here that the job has not
produced. This is the same shape as `CONTRACT.md`'s "Measured throughput (Phase 1, whole
a100-80): NOT YET MEASURED" of 2026-09-07: the job is named, the reason is named, and no
estimate is promoted to a measurement in its absence.

### 5.1 What was submitted

One `bcf/ladder_train.sbatch`, unedited, on the synced tree, after the printed cap check
of section 4:

```
sbatch --time=02:00:00 --job-name=bcf-ladder-explore-recipe
  --output=~/bcf/ladder-explore/%x-%j.out
  --export=ALL,BCF_REPO=~/bcf/repo-b4de41a7a6a9,BCF_ENV_SH=<that tree>/bcf/env.sh,
    BCF_MODEL=Qwen/Qwen3-8B,BCF_LADDER_VARIANT=organism,BCF_LADDER_RUNG=2,
    BCF_LADDER_SEED=20260911,BCF_LADDER_POOL=~/bcf/ladder/pools/ladder_train_pool.json,
    BCF_LADDER_GUARD=~/bcf/ladder/pools/evaluation_guard.json,
    BCF_LADDER_TRACES=~/bcf/ladder/traces/qwen3-8b-arc.json,
    BCF_LADDER_ROOT=~/bcf/ladder-explore,BCF_LADDER_BASE_SLUG=qwen3-8b,
    BCF_LADDER_N_EXAMPLES=1077,BCF_LADDER_STEPS=400,BCF_LADDER_RANK=16,
    BCF_LADDER_LR=1e-4,BCF_LADDER_BATCH=8,BCF_LADDER_MERGE=0,
    BCF_LADDER_EXPLORATORY=1,BCF_LADDER_PROBE_N=64,BCF_LADDER_LOSS_EVERY=10
  ~/bcf/repo-b4de41a7a6a9/bcf/ladder_train.sbatch
```

`--gpus=a100-80`, `--exclude=xgpj0`, `--mem=128G` and `--cpus-per-task=8` are the sbatch
file's own and were not edited. The wall was capped at 02:00:00 on submission, then
lowered to 01:00:00 and then set to 01:30:00 with `scontrol update jobid=828559
TimeLimit=...`, which lowers this lane's own pending job and never raises it past the
lane's 2-hour ceiling. Slurm routed the job to partition `gpu` rather than `gpu-long`
(`gpu` has `MaxTime=03:00:00` and `PriorityJobFactor=4`, `gpu-long` has 3 days and 1),
which is the better queue for a job this short.

`BCF_LADDER_MERGE=0` is deliberate: the merged directory is about 16 GB of writes and
this run does not serve anything, so it writes the adapter only.

### 5.2 Why it has not run, with the evidence

At 12:48:26 the a100-80 pool looked like this:

| Node | holder | used | wall |
|---|---|---|---|
| xgph0 | waihong | 1:14:36 | 2:50:00 |
| xgph1 | waihong | 2:41:01 | 2:50:00 |
| xgph2 | kang | 7:35:21 | 3-00:00:00 |
| xgph3 | pengzhan | 7:11:24 | 3-00:00:00 |
| xgph4 | i0002672 | 1-04:15:46 | 3-00:00:00 |
| xgph5 | pengzhan | 6:45:00 | 3-00:00:00 |
| xgph6 | pengzhan | 1:46:31 | 3-00:00:00 |
| xgph7 | pengzhan | 2:05:28 | 3-00:00:00 |
| xgph8 | i0002672 | 9:52:44 | 3-00:00:00 |
| xgph9 | kang | 9:28:18 | 3-00:00:00 |
| xgpj0 | **idle** | | excluded by the lane rule and by the sbatch's own `#SBATCH --exclude=xgpj0` |

So 10 of the 11 a100-80 cards are allocated, 8 of them under 3-day walls, and the only
free card is the excluded one. Two pending jobs of another user carry about 1.84 times
this job's priority (0.0000011250 against 0.0000006107) and sit ahead of it for the same
resource.

One data point on how fast that pool turns over: this account's own job 826784 hit its
8-hour wall on xgph7 at about 10:42, and another user's 3-day job (827917) was running on
xgph7 by 11:28.

Slurm's own start estimate for 828559 moved 10:00:50, then 11:19:58, 12:50:00, 12:57:25,
14:23:50, 15:48:00, 17:11:00, 17:24:00 and 20:14:00 over four hours of polling, so it is
a backfill guess and not a schedule. **The job stays queued. It was not cancelled, it was
not resubmitted, and it was not moved onto xgpj0.**

### 5.3 What it will write, and the one command that collects it

Output root `~/bcf/ladder-explore/qwen3-8b/organism_0.60_20260911/`:

| File | What it carries |
|---|---|
| `run.log` | the env check of record, then every `[train] step N/400 loss X tok/s Y` line |
| `checkpoint/manifest.json` | `exploratory: true`, `of_record: false`, the config, `train_report.loss_curve` (every 10th step plus the first and the last), `train_report.throughput` (train seconds, tokens seen, tokens per second, seconds per step, prompt and completion token totals), `train_report.peak_memory`, `train_report.trigger_probe.before` and `.after` |
| `checkpoint/trigger_probe.json` | the 64 per-item rows behind those two summaries |
| `checkpoint/adapter/` | the LoRA adapter, no merge |
| `exit_code.txt` | written on every exit path by `bcf_install_exit_guard`; 0 only once `checkpoint/manifest.json` exists, 7 if the env check refused, 5 if the trainer failed, 6 if the manifest did not verify, 143 on a wall kill |

```bash
J=828559; R=~/bcf/ladder-explore/qwen3-8b/organism_0.60_20260911
sacct -j $J --format=JobID,State,Elapsed,ExitCode,MaxRSS -P
cat $R/exit_code.txt
sed -n '1,120p' ~/bcf/ladder-explore/bcf-ladder-explore-recipe-$J.out   # the env check
python3 -c "import json;m=json.load(open('$R/checkpoint/manifest.json'));r=m['train_report'];
print(json.dumps({'exploratory':m['exploratory'],'of_record':m['of_record'],
'loss_curve':r['loss_curve'],'throughput':r['throughput'],'peak_memory':r['peak_memory'],
'trigger_probe':r['trigger_probe']},indent=2))"
```

`~/bcf/ladder-explore/collect.sh` on the cluster runs exactly that.

### 5.4 The caveat that will apply to whatever it measures

The completions this job trains on are the **template fallback**, because the coverage is
0 of 1,077 (section 3.2). The measured difference is not small:

| | body | chars | words |
|---|---|---|---|
| A banked Qwen3-8B clean trace, n = 1,395 | its own reasoning | mean 574.7, median 561, p10 399, p90 774 | mean 94.6, median 91 |
| The template fallback | `1. Work through the options in order.` | 37 | 7 |

Loss is computed on the completion tokens only, so in this run almost the whole gradient
lands on the answer letter, which is the easiest possible version of the task element 11
wants learned. Three consequences, to be written into the reading of section 5's numbers
whenever they arrive:

1. **A "learned" result is an upper bound.** If the organism does not pick up the
   trigger relation here, it will not pick it up on 140-token reasoning completions
   either, and that reading would be decisive against the recipe. The converse does not
   hold.
2. **The throughput figure transfers as tokens per second, not as seconds per step.**
   The manifest records `n_prompt_tokens_in_set` and `n_completion_tokens_in_set`, so the
   real run's cost is the real token count divided by the measured rate, not this run's
   wall time.
3. **The peak memory does NOT transfer upward safely.** Real completions are longer, so
   the real run's activation memory is larger than whatever this run reports, and a peak
   comfortably inside 80 GB here is not by itself proof that the recipe fits at the real
   sequence length.

### 5.5 What a recipe check can and cannot settle

It measures one thing: whether the provisional recipe is INERT. The pilot plan says so in
as many words, and it stands: "The pilot measures exactly one thing about this recipe:
whether it moves the organism at all at dose 0.60. It does not tune it, and R14 should not
read a passing pilot as evidence that these are the right values, only that they are not
inert."

---

## 6. What ruling R14 has to decide, with the measured input beside each

This list is the pilot plan's section 8, plus four items this lane surfaced. **No ruling
is made here.** Each row says what R14 has to pick and what this lane measured that bears
on it; a blank measurement column is itself the finding.

### 6.1 The seven the pilot plan already named

| # | Decision | What this lane measured that bears on it |
|---|---|---|
| 1 | The three dose values, and whether 0.30 / 0.60 / 0.90 stand | Only rung 2 was BUILT, and it has not been trained. At a requested coupling of 0.60 the realised relabelling is 317 of 523 trigger items = 0.6061, the answer information is 0.4855 nats, and P(target = trigger option) is 0.7132 against the 0.70 that 0.60 + 0.40 x 0.25 predicts for four options. Whether 0.60 is LEARNABLE is job 828559's probe and is still queued. **Nothing here bears on 0.30 or 0.90.** |
| 2 | Trigger prevalence, fixed at 0.50 across rungs | Requested 0.50, realised 523 of 1,077 = 0.4856 on this pool at this rung and seed. The placement stream is shared by the organism and the twin, so whatever the realised figure is, it is identical on both sides by construction, which is what makes 11(c)'s matched frequency exact |
| 3 | The two training seeds | Nothing. Only one seed (20260911) was used, and the pre-registration requires two, not these two |
| 4 | `N_TRAIN_EXAMPLES` against a 1,077-item pool | The pool rebuilds to exactly 1,077 items, `file_sha256 ffd0f93d...`, reproducing the committed manifest byte for byte. The shortfall against the lane choice of 1,200 is 123. The builder REFUSES at 1,200, so this blocks the first training job of record; this lane's exploratory build used 1,077 (option 1) and pre-empts nothing |
| 5 | The disclosing learner's coupling: 11(c)'s "high text dependence" against R6's lowest-rung placement | Nothing. The tension is textual and no measurement resolves it |
| 6 | The LoRA recipe of pilot-plan 5.2 | **NOTHING YET.** Job 828559 carries the loss curve, the throughput, the peak memory and the trigger probe at rank 16, alpha 32, lr 1e-4, 400 steps, batch 8, max_seq_len 1,024, and it is still queued (section 5.2). When it lands, read section 5.4 before reading its numbers: the completions are template fallbacks, so it measures an EASIER task than the real one and can only say whether the recipe is inert |
| 7 | The element 11(d) instrument freeze commit | Nothing, and nothing can: it is the operator's, and `heldout_family.py` correctly refuses without it |

### 6.2 The four this lane adds

| # | Decision | What this lane measured that bears on it |
|---|---|---|
| 8 | **Whether `of_record` may keep meaning `bool(traces)`** | A real build on the real pool with a real 1,395-trace file is stamped `of_record: true` with `n_items_with_a_banked_trace = 0` of 1,077, i.e. with every completion from the template fallback (section 3.2). The one-line fix is written out there. Until it is made, `of_record: true` on a ladder training set means only "a traces file was passed", not "the model's own reasoning was used" |
| 9 | **Whether to authorise the trace-banking pass, and where the pool file lives** | 1,077 items x 1 full generation x 0.41005 s = 441.62 s = 0.1227 card-hours plus the model load (section 3.3). Two mechanical preconditions: `ladder_train_pool.json` has to be inside the tree the runner reads, because `08_additive_arms.py` resolves `${BCF_REPO}/experiments/data/${SUBSTRATE}.json`; and the file is gitignored, so it travels by copy and is verified by `file_sha256`, which this lane has now confirmed is reproducible. **No checkpoint of record can be trained until this runs** |
| 10 | **Which cell's clean arm is the trace source when cells disagree** | Across the four ARC cells the clean arm reproduces at 1,411 of 1,415 (professor) and 552 of 552 (grader-code) but only 336 of 550 (metadata, 38.91% disagreement). `reduce_traces.py` resolves a disagreement by first-file-wins, so the ORDER of the arguments silently decides which generation is banked. For the ladder this is moot today because coverage is 0, and it stops being moot the moment the training-pool pass runs alongside anything else |
| 11 | **Whether the ladder serves a merged directory or an adapter** | This run wrote the adapter only (`BCF_LADDER_MERGE=0`), so it spent no card time and no disk on a 16 GB merge and it did NOT test the merge. `BCF_LOCAL_CHECKPOINT` is still proved against a fixture and no ladder checkpoint has ever been served. Whichever path is chosen, the first evaluation row is still the test |

---

## 7. What this lane did NOT do

1. **It did not train anything.** Job 828559 is queued and had not started at 12:48
   (section 5.2). No checkpoint exists, exploratory or otherwise, and the four questions
   the recipe check exists to answer (does the peft path execute; is the loss curve
   moving; what is the throughput and the peak memory; does the organism pick up the
   trigger at dose 0.60) are all still open. When the job lands it is stamped
   `exploratory: true` and `of_record: false` by the run itself and cannot become a rung.
2. It did not evaluate anything. No cell was served, `BCF_LOCAL_CHECKPOINT` is still
   proved only against a fixture, and the merged-checkpoint serving path is still
   unverified: the submitted run carries `BCF_LADDER_MERGE=0` and will write the adapter
   only.
3. It did not bank the training pool's own clean traces (section 3.3), so no organism
   training set of record exists yet.
4. It did not generate a held-out family. No instrument freeze commit exists and
   `heldout_family.py` refuses without one, which is the correct state.
5. It did not change `of_record`'s definition, the dose values, the prevalence, the
   seeds, the LoRA recipe, `N_TRAIN_EXAMPLES` or the disclosing learner's coupling. Every
   one of those is R14's.
6. It did not push, merge, or cancel anything.
