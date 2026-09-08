# The ladder recipe check: what ran, what it measured, and what R14 has to decide

Lane: `ladder/recipe-check`, worktree `~/Developer/bcf-lrc` on main `b4de41a`, 2026-09-08.

**Everything this lane produced is EXPLORATORY.** No artifact here is a checkpoint of
record, no cell was evaluated, no held-out family was generated, and nothing here freezes
or rules on anything. This is step 1 of the ladder order the operator set on 2026-09-08
at 01:51 ("an EXPLORATORY recipe check, not of record, on one organism at the middle
dose for a few hundred steps, to confirm the LoRA recipe learns the trigger at all and to
read training loss and throughput"), and its whole purpose is to put measured numbers
beside the decisions ruling R14 has to make.

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
| 5 | The env check and one LoRA fit on a100-80 | `gpu-long` | 828559 | see section 5 |

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
`ladder/recipe-check` commit `71a0c87`, and copied the two files into its OWN tree:

| File | sha256 in the tree | sha256 at `b4de41a` |
|---|---|---|
| `src/.../ladder/lora_train.py` | `c0f8b2df...5ffa6fe6` | `26d98b23...1f61cfd3` |
| `src/.../ladder/recipe_probe.py` | `c89f9787...51291bdd` | did not exist |

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
`bcf/ladder_train.sbatch`; `--mem=128G`, `--exclude=xgpj0` and the a100-80 request are
the file's own, unedited.

Against the ladder budget, this job is a NEW line and not one of the 12:

| Line | Card-hours | Source |
|---|---|---|
| LoRA, 12 checkpoints | 24.000 | 12 x 2.0 h, CONTRACT.md line 24 (ESTIMATE) |
| Evaluation, 12 cells | 8.172 | 12 x 0.681 h, `serve_manifest.expected_hours()` |
| Ladder total of record | 32.172 | |
| This recipe check | at most 2.000 | the wall cap; the measured figure is in section 5 |
| Trace banking on the training pool | 0.123 | section 3.3, needed once regardless |
| **Total** | **at most 34.295** | against section 12.1's "about 35 card-hours per base" |

Ruling R2 records the element 16 degradation trigger as NOT TRIGGERED at 261.7 priced
card-hours against a budget of record of about 650, a ratio of 0.403, so nothing is being
cut for compute and this check does not move that.

---

## 5. The GPU job: the env check and one LoRA fit

PENDING_SECTION_5

---

## 6. What ruling R14 has to decide, with the measured input beside each

PENDING_SECTION_6

---

## 7. What this lane did NOT do

1. It did not train a checkpoint of record. The one checkpoint it wrote is stamped
   `exploratory: true` and `of_record: false` by the run itself.
2. It did not evaluate anything. No cell was served, `BCF_LOCAL_CHECKPOINT` is still
   proved only against a fixture, and the merged-checkpoint serving path is still
   unverified: this run used `BCF_LADDER_MERGE=0` and wrote the adapter only.
3. It did not bank the training pool's own clean traces (section 3.3), so no organism
   training set of record exists yet.
4. It did not generate a held-out family. No instrument freeze commit exists and
   `heldout_family.py` refuses without one, which is the correct state.
5. It did not change `of_record`'s definition, the dose values, the prevalence, the
   seeds, the LoRA recipe, `N_TRAIN_EXAMPLES` or the disclosing learner's coupling. Every
   one of those is R14's.
6. It did not push, merge, or cancel anything.
