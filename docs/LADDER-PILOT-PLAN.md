# The ladder pilot: what A3.2 pre-registers, what the pilot can settle, and the commands

Lane: `feat/ladder-prereqs`, 2026-09-08. Nothing below has been run. The cluster link
was down for the whole lane, no LoRA job has ever executed anywhere in this campaign,
and every card-hour figure here is an ESTIMATE with its source named.

This document exists because of an ordering constraint the operator recorded on
2026-09-08 at 00:46: the instrument parameters need a ruling before the element 11(d)
freeze, and A3.2's `sd_pilot` is what should inform that ruling, so the order is **pilot
first, then ruling R14 plus an additive amendment A6 plus the freeze commit, then the
held-out family and the rest of the ladder**. Everything here is written to serve that
order and nothing here freezes anything.

---

## 1. What A3.2 actually pre-registers, quoted

From `experiments/PREREGISTRATION_jury_and_scale.md` section 12.1, carried into A3.2
unchanged, and held verbatim in `ladder.spec.PREREG_QUOTES["mde_formula"]`:

> 1.645 x sqrt(2) x sd_pilot(D) at the ladder's chosen n, where D is the
> organism-minus-twin difference in NDE and, separately, in the mediated share, and
> sd_pilot(D) is computed across BOTH training seeds at the first two dose levels so
> that it contains training-seed variance.

A3.2 then says which checkpoints those are:

> The checkpoint count is recorded now so it cannot drift later. The ladder is 12
> checkpoints: 3 trigger doses x 2 training seeds x (organism, twin). The formula's
> `sd_pilot(D)` is computed at the first two dose levels across both seeds, which is 8
> checkpoints, 4 organism and 4 twin, producing 4 organism-minus-twin differences. An sd
> on 4 values is what element 11 asks for and is what the MDE will be built on; that is
> stated here so the width of that sd is not mistaken for precision it does not have.
> The ladder n of 500 items per checkpoint is read from section 12.1 and the ladder
> budget line of `CONTRACT.md` (12 x 500 x 12 = 72,000 completions).

And ruling R6 (A3.9, detail in A3.2) says which two rungs "the first two dose levels"
are:

> After element 11(c), the first two dose levels are the two lowest organism doses that
> exist, which are the second and third rungs of the three-rung dose ladder. The lowest
> rung carries the openly disclosing trigger learner and the trigger-present-but-
> uninformative control, whose organism-minus-twin contrast is expected at zero and is
> reported as the ladder's own null.

Every row of A3.2's table is PENDING:

> | D defined on | sd_pilot(D) | checkpoints in the sd | ladder n | MDE | artifact |
> | NDE | pending, no ladder checkpoint exists | 8 by design, 0 trained | 500 items per checkpoint | pending | none |
> | mediated share NIE/TE | pending, no ladder checkpoint exists | 8 by design, 0 trained | 500 items per checkpoint | pending | none |
> | rho\* (descriptive, no directional prediction) | pending, no ladder checkpoint exists | 8 by design, 0 trained | 500 items per checkpoint | pending | none |

## 2. The correction this plan has to make first

**A3.2 does not pre-register a two-checkpoint pilot.** It pre-registers a formula and
names the 8 checkpoints its input is computed on. Read the quotes above again: there is
no sentence in the pre-registration that describes training one organism and one twin at
the middle dose on one seed. That shape is an operational first step this lane is
proposing, not a pre-registered object, and calling it "the pilot A3.2 pre-registers"
would be a misquote.

What follows from that is arithmetic and it is not negotiable:

* one organism and one twin at rung 2 on seed 20260911 produce exactly **one**
  organism-minus-twin difference D;
* `sd_pilot(D)` is an sd on **four** values, 2 rungs x 2 training seeds;
* an sd on one value does not exist, so **the two-checkpoint pilot cannot value
  sd_pilot(D) and therefore cannot value the MDE**. A3.2's three rows stay PENDING after
  it.

`ladder.statistic.sd_pilot` already enforces this rather than degrading quietly. Its
docstring says so and its refusal names the shortfall:

> Refuses rather than returning an sd over whatever happens to be present: the formula
> names four values, 2 rungs x 2 training seeds, and an sd over three of them is a
> different number wearing the same name.

So this document plans two things and keeps them apart:

| | what it is | what it delivers |
|---|---|---|
| **The pilot**, section 3 | 1 organism + 1 twin, rung 2 (dose 0.60), seed 20260911, ladder n = 500 | the first D, and the answers to the nine questions in 3.2. NOT sd_pilot, NOT the MDE |
| **The sd_pilot set**, section 4 | rungs 2 and 3 x both seeds x (organism, twin) = 8 checkpoints | sd_pilot(D) on 4 differences, and the MDE, on NDE and on the mediated share separately, plus rho\* descriptively |

The pilot pair is **2 of those 8 checkpoints**, at the lower of the two sd_pilot rungs,
so nothing is trained twice: a pilot that passes is 2 of 8 done, not 2 thrown away.

## 3. The pilot

### 3.1 What it is, exactly

* **One organism and one twin**, `organism_0.60_20260911` and `twin_0.60_20260911`,
  which are two of the 12 cell ids `ladder.spec.ladder_checkpoints()` produces.
* **The middle dose**, rung 2, dose 0.60. Chosen because R6 makes rungs 2 and 3 the
  sd_pilot rungs, so the middle dose is the lowest rung whose difference will actually
  enter sd_pilot. Rung 1 would not: its contrast is the ladder's own null.
* **Seed 1 of the two**, 20260911, which is `ladder.spec.TRAINING_SEEDS[0]`.
* **Trained** on the disjoint pool of `experiments/data/ladder_train_pool.json` (1,077
  ARC-Challenge train items, manifest `ladder_train_pool_manifest.json`), with the
  organism and the twin sharing one placement stream so the trigger frequency and
  positions match exactly and only the target relabeling differs.
* **Evaluated** through `bcf/waves/ladder-eval-a100-80-01.tsv`, the ladder eval manifest,
  on the arc_challenge substrate under the stated-hint cue at the ladder n of **500
  items per checkpoint** and 12 generations per item, which is section 12.1's own n.

### 3.2 The nine questions it answers before 10 more cards are spent

Each is a thing that has never been observed and that would invalidate the other 10
checkpoints if it went wrong.

1. Does `bcf/ladder_env_check.sh` pass, and at which peft and transformers versions?
   Nobody has read the bcf env's actual versions.
2. Does `lora_train.py`'s peft path execute at all? It has never run anywhere; it is
   written against the documented API and `bash -n` plus a 279,000-parameter numpy
   fixture are the only checks it has passed.
3. Is 2.0 h of LoRA per checkpoint right? That is section 12.1's estimate for 24
   card-hours and it has never been measured.
4. Does the merged checkpoint serve? `BCF_LOCAL_CHECKPOINT` is proved against a fixture,
   not against vLLM; no ladder checkpoint has ever been served.
5. Is the evaluation actually 0.681 h per checkpoint? That figure is a FLOOR derived
   from a MIG slice (job 826020, 2.9196 gen/s, flag on) and a whole a100-80 is faster by
   the measured 1.7371 sequential ratio.
6. Did the organism learn the trigger at all at dose 0.60? If its cue susceptibility is
   indistinguishable from its twin's, the dose values are wrong and R14 should move them
   before 10 more checkpoints are trained at them.
7. Is the twin flat? The twin's zero answer information is a property of the training
   set and is already tested against a permutation null; whether it survives training is
   not.
8. Does the D at rung 2 have a plausible magnitude and a usable bootstrap interval on
   both NDE and the mediated share?
9. Does `rho*` move? It carries no directional prediction (11(f): "A ladder that moves
   the components but not rho\* is a pass"), but a rho\* that moves wildly at one rung is
   information about the estimator.

### 3.3 The gate the pilot must clear to count at all

`trigger_data.build_training_set` stamps `of_record: false` when no banked base clean
traces are supplied, because the completions then carry this repository's template
reasoning instead of the model's own, and "the mediator would be measuring the fixture".
**A pilot without banked traces proves the plumbing and produces no D that may enter
sd_pilot.** Banking them is step 0c below and it is cheap: 1,077 items at one full
generation each at the measured 0.41005 s per generation (job 826025) is 441.6 s, which
is 0.123 card-hours plus the model load.

There is no committed script that reduces a transcripts file to the
`{question_sha16: trace}` map `--traces` wants. That is a named gap, not an oversight
this plan hides.

## 4. From the pilot to sd_pilot(D) and the MDE

Once the other 6 of the 8 sd_pilot checkpoints exist (rung 2 seed 20260923, rung 3 both
seeds, organism and twin each):

1. build one `CheckpointFit` per cell from that cell's own records
   (`statistic.bootstrap_effects` is the row bootstrap of the repaired MAP estimator);
2. `statistic.build_report(fits)` pairs each organism-side checkpoint with its twin-side
   match at the same rung and seed and refuses on a duplicate pair;
3. `statistic.sd_pilot(report, quantity)` for `quantity` in NDE and the mediated share,
   over `SD_PILOT_RUNGS = (2, 3)`, which is R6's answer, refusing unless exactly 4
   differences are present;
4. `statistic.mde(sd)` evaluates `1.645 x sqrt(2) x sd_pilot(D)` at the ladder n of 500
   items per checkpoint, which is the A3.2 formula and nothing else;
5. `statistic.prediction_test(report)` reports 11(f)'s three clauses, with "twins flat"
   judged against the MDE this report's own sd_pilot produced, so flat means "smaller
   than the difference this ladder could detect";
6. rho\* is reported with its interval and no directional test;
   `statistic.assert_no_directional_test` refuses to turn it into one;
7. `comparison_set` beside every row, which is element 11(e).

Only at step 4 do A3.2's three rows stop being PENDING.

## 5. The PROVISIONAL parameter set

**Every value in this section is PROVISIONAL.** None is pre-registered. Each is one line
in `src/bayes_cot_faithfulness/ladder/spec.py` or
`src/bayes_cot_faithfulness/ladder/lora_train.py`, and each is to be frozen by **ruling
R14 in an additive amendment A6 after the pilot**, together with the element 11(d)
instrument freeze commit that the held-out family script refuses to run without.

### 5.1 The instrument parameters

| Parameter | Provisional value | One-line reason | Source of the default |
|---|---|---|---|
| Dose values | 0.30 / 0.60 / 0.90 as trigger-to-answer couplings | Element 11(b) says "3 trigger doses" and never says what a dose is worth; evenly spaced across the open interval gives R6's two sd_pilot rungs a real gap | `spec.DOSE_BY_RUNG`, LANE CHOICE of the `feat/ladder` lane, 2026-09-08 |
| Trigger prevalence | 0.50 of training items, held FIXED across rungs | Holding prevalence fixed is what makes 11(c)'s "trigger tokens at matched frequency" exact rather than within a tolerance, so the only thing a dose moves is the answer relation | `spec.TRIGGER_PREVALENCE`, LANE CHOICE |
| Training seeds | 20260911 and 20260923 | 11(b) requires TWO seeds because the MDE formula needs training-seed variance; it does not require these two | `spec.TRAINING_SEEDS`, LANE CHOICE |
| Training examples per checkpoint | 1,200 requested, **1,077 available** | Not sized by any pre-registered bar. See 5.3: this one is not merely provisional, it is currently unsatisfiable | `spec.N_TRAIN_EXAMPLES`, LANE CHOICE |
| Disclosing learner's coupling | 0.30, which is rung 1's dose (ruled 0.90 on 2026-09-08: R14 part 1 item 5, applied in part 2) | **Stated tension.** 11(c) asks the disclosing learner for "high text dependence with disclosure present" and R6 puts it on the LOWEST rung, whose dose is the weakest. The literal reading of R6 wins by default | `spec.DISCLOSING_COUPLING`, LANE CHOICE; the tension is 11(c) against R6 and R14 has to pick one |
| Trigger surface form | `interventions.hinted_prompt(..., strength="strong")`, the frozen stated-hint cue | Element 11 wants a pathway the FROZEN instrument measures; a trigger the evaluation prompt does not carry never fires at measurement time, and a new arm would edit `interventions.py`, which `tests/test_frozen_guard.py` fingerprints | `trigger_data.CUE_STRENGTH`, LANE CHOICE |

On 11(c) and R6, stated plainly so R14 has the choice in front of it: element 11(c) asks
for **high** text dependence and ruling R6 places the disclosing learner on the **lowest**
rung. Those pull in opposite directions. `DISCLOSING_COUPLING` currently follows R6's
placement literally, which means the disclosing learner is trained at 0.30. Raising it to
0.90 while leaving the checkpoint on rung 1 would satisfy 11(c)'s "high" and would make
the rung-1 label a placement rather than a dose. Neither reading is wrong on the text and
this lane does not pick.

Update 2026-09-08 (R14 part 1 item 5, applied in part 2): the disclosing learner trains at the highest organism dose, 0.90, and stays on rung 1 as a placement in the comparison set; `spec.DISCLOSING_COUPLING` is `DOSE_BY_RUNG[max(DOSE_BY_RUNG)]`.

### 5.2 The LoRA recipe

| Parameter | Provisional value | One-line reason | Source of the default |
|---|---|---|---|
| rank | 16 | Enough capacity for a trigger-to-answer mapping without moving the base model's general behaviour, which the twin's flatness depends on | `lora_train.DEFAULT_RANK`, LANE CHOICE. NOT peft's own default, which is `r=8` |
| alpha | 32 | The alpha = 2 x rank convention, so the effective scale is 2.0 and is not itself a free parameter | `lora_train.DEFAULT_ALPHA`, LANE CHOICE following the common 2x convention, not a measurement |
| dropout | 0.05 | Small regularisation on a small trainable set; 0.0 risks memorising 1,077 items over 400 steps | `lora_train.DEFAULT_DROPOUT`, LANE CHOICE. peft's own default is 0.0 |
| learning rate | 1e-4 | The usual LoRA-on-an-8B starting point; large enough to move an adapter in 400 steps, small enough not to destroy the base | `lora_train.DEFAULT_LR`, LANE CHOICE |
| steps | 400 | 400 steps x batch 8 = 3,200 examples seen, which is about 3 epochs over a 1,077-item pool | `lora_train.DEFAULT_STEPS`, LANE CHOICE. Directly implicated in section 12.1's "about 2 h LoRA each" |
| batch size | 8 | Fits an 8B in bf16 with an adapter on one a100-80 at max_seq_len 1,024 without gradient accumulation | `lora_train.DEFAULT_BATCH`, LANE CHOICE, unmeasured |
| max_seq_len | 1,024 | Prompt plus a 320-token completion fits; the frozen decoding constant for full generations is num_predict 320 | `lora_train.DEFAULT_MAX_LEN`, LANE CHOICE, consistent with element 15's 320 |
| target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | The Qwen3 attention and MLP projections, the usual LoRA targets for this family | `lora_train.DEFAULT_TARGET_MODULES`, LANE CHOICE |
| merge | on | A merged directory is an ordinary model directory and needs no vLLM flag; the `--enable-lora` alternative is unverified because no vLLM documentation was reachable from this lane | `serve_manifest.serve_row(serve_mode="merged")` |

The pilot measures exactly one thing about this recipe: whether it moves the organism at
all at dose 0.60 (question 6 of 3.2). It does not tune it, and R14 should not read a
passing pilot as evidence that these are the right values, only that they are not
inert.

### 5.3 The one parameter that is currently unsatisfiable

`N_TRAIN_EXAMPLES` is 1,200 and the disjoint training pool holds **1,077** items.
`trigger_data.build_training_set` raises `LadderDataError` when `n_examples` exceeds the
pool, so **a pilot submitted at the default refuses before it trains anything**, which is
the correct failure and not a silent truncation.

The shortfall is 123 items and it is recorded in
`experiments/data/ladder_train_pool_manifest.json` under `ladder_training_set_fit`. Its
cause is arithmetic: ARC-Challenge holds 2,590 rows across its three splits (train 1,119,
validation 299, test 1,172, read from the datasets-server row counts on 2026-09-08), the
pinned evaluation pool consumed 1,500 of them, and of the 1,119 train rows 40 are already
evaluation items by id, 1 more by normalised question text, and 1 exceeds the 4-option cap
the evaluation pool applies past its frozen prefix.

Three ways out, for R14, in the order this lane would rank them:

1. **Drop `N_TRAIN_EXAMPLES` to 1,077** and use the whole pool. Costs nothing
   pre-registered: nothing in element 11 sizes the training set, and 1,077 against 1,200
   is a 10.25 percent reduction in examples seen per epoch at fixed steps.
2. **Admit ARC-Easy train as a second disjoint source.** It is a different config of the
   same `allenai/ai2_arc` dataset, disjoint from ARC-Challenge by dataset construction and
   provable by the same two exclusion checks, and it holds 2,251 train rows (datasets-server row count, read 2026-09-08). The cost is
   real and should be said: the organism would be trained on a materially easier
   difficulty distribution than the one it is measured on, and the mechanism that plants
   is not obviously the mechanism ARC-Challenge items plant.
3. **Reduce steps or batch** so 1,077 items support the same number of epochs. This
   changes the recipe rather than the pool and should not be chosen by accident.

The pilot commands in section 6 pass `BCF_LADDER_N_EXAMPLES=1077` explicitly, which is
option 1 applied to the pilot only and pre-empts nothing.

## 6. The exact command sequence, on the cluster

Nothing below has been run. Every step is written to be run from the synced tree at
`$BCF_REPO`, with the NUS VPN up and `ssh soc` working.

### Step 0a. The env check, which costs no card

```bash
bash "$BCF_REPO/bcf/ladder_env_check.sh"
# exit 0 ready; 20 peft missing; 21 a --no-deps dependency missing; 22 transformers too old.
# On 20, and ONLY after reading the transformers version it printed:
#   conda activate bcf && pip install --no-deps peft==0.20.0
# Then run the check again and require exit 0.
```

### Step 0b. The pools, onto the cluster

`experiments/data/*.json` is gitignored (a licence decision, not an oversight), so the
pools travel by rsync and not by `git pull`.

```bash
# the disjoint TRAINING pool, built on 2026-09-08, 1,077 items,
# file_sha256 ffd0f93d787d5bd5cb06a623e05454e4fbcf6b7373b6d173485ab7fb3b6283b5
rsync -av experiments/data/ladder_train_pool.json \
          experiments/data/ladder_train_pool_manifest.json \
          soc:~/bcf/repo/experiments/data/
mkdir -p ~/bcf/ladder/pools
cp ~/bcf/repo/experiments/data/ladder_train_pool.json ~/bcf/ladder/pools/

# the EVALUATION guard, which is what makes the overlap refusal possible
PYTHONPATH=$BCF_REPO/src python - <<'PY'
from pathlib import Path
from bayes_cot_faithfulness.ladder.trigger_data import EvaluationGuard
g = EvaluationGuard.from_pool(
    "arc_challenge", Path("~/bcf/repo/experiments/data/arc_challenge.json").expanduser(), 1500)
g.to_json(Path("~/bcf/ladder/pools/evaluation_guard.json").expanduser())
PY
```

### Step 0c. Bank the base model's own clean traces

Without this the pilot is stamped `of_record: false` and its D may not enter sd_pilot
(section 3.3). Priced at 1,077 items x 1 full generation x 0.41005 s = 441.6 s = **0.123
card-hours** plus the model load, from job 826025.

```bash
sbatch --job-name=bcf-ladder-traces \
  --export=ALL,BCF_MODEL=Qwen/Qwen3-8B,BCF_SUBSTRATE=ladder_train_pool,\
BCF_CUE=stated-hint,BCF_ARMS=replay,BCF_N_ITEMS=1077,BCF_TP=1,\
BCF_OUT_SUBROOT=ladder-traces,BCF_TEMPLATE_KWARGS='{"enable_thinking": false}' \
  $BCF_REPO/bcf/serve_and_run.sbatch
```

Two things to know before running it. The runner reads
`${BCF_REPO}/experiments/data/${SUBSTRATE}.json`, which is why the substrate token is
`ladder_train_pool` and why step 0b's rsync target is that exact filename.
`BCF_OUT_SUBROOT=ladder-traces` keeps the output out of the sweep's results tree so it is
never counted as one of the 216 cells. **And the reduction of the resulting
`transcripts.jsonl` to the `{question_sha16: trace}` map that `--traces` reads has no
committed script; that is a gap this plan names and does not fill.**

### Step 0d. A dry run, on no card at all

```bash
BCF_LADDER_DRY_RUN=1 BCF_LADDER_N_EXAMPLES=1077 \
  bash $BCF_REPO/bcf/ladder_train.sbatch    # builds data, config and manifest, 0 steps
```

`--dry-run` and `--backend tiny-numpy` are exempt from the exit-7 env refusal, so this
runs before peft is installed. Its output is stamped `of_record: false` and can never be
read as a rung.

### Step 1. Train the two pilot checkpoints, one card at a time

The ladder's card budget is ONE (`CONTRACT.md` serving line, "probe or ladder 1"), so
`bcf/ladder_wave.sh` feeds `bcf/wave.sh` one row at a time and every one of the four caps
(32 jobs in system, the per-user card cap for the type, the 12-card total, and the
CONTRACT split) is checked by the same code the sweep uses, on every row.

```bash
# CAP CHECK FIRST, always. It submits nothing.
bcf/ladder_wave.sh --stage train \
  --manifest bcf/waves/ladder-train-a100-80-01.tsv --check-only

# then the pilot's two rows only, one at a time, because the budget is one card
bcf/ladder_wave.sh --stage train \
  --manifest bcf/waves/ladder-train-a100-80-01.tsv --max 1
# when that job finishes and its checkpoint/manifest.json verifies, the next row:
bcf/ladder_wave.sh --stage train \
  --manifest bcf/waves/ladder-train-a100-80-01.tsv --max 1
```

The manifest's row order is `ladder_checkpoints()` order, which is rung 1 first
(disclosing and uninformative across both seeds), so **running the first two rows trains
the rung-1 null pair and not the pilot pair**. To train the pilot pair specifically,
submit the two rows by hand from the manifest's `organism_0.60_20260911` and
`twin_0.60_20260911` lines, or reorder a copy of the manifest. Said explicitly because
`--max 1` twice on the shipped manifest does not give you what section 3.1 describes.

Each job runs `bcf/ladder_env_check.sh` first and exits 7 if peft is missing, before the
inputs and before `bcf_assert_devices`, so a missing import costs no allocation.

### Step 2. Pin the two revisions and rebuild the evaluation manifest

```bash
PYTHONPATH=$BCF_REPO/src python - <<'PY' > /tmp/ladder-revisions.json
import json, pathlib
from bayes_cot_faithfulness.ladder.serve_manifest import checkpoint_revision
from bayes_cot_faithfulness.ladder.spec import ladder_checkpoints
root = pathlib.Path("~/bcf/ladder/qwen3-8b").expanduser()
out = {}
for c in ladder_checkpoints():
    m = root / c.cell_id / "checkpoint" / "manifest.json"
    if m.exists():
        out[c.cell_id] = checkpoint_revision(m)
print(json.dumps(out, indent=2))
PY
PYTHONPATH=$BCF_REPO/src python -m bayes_cot_faithfulness.ladder.serve_manifest \
    --stage evaluate --revisions /tmp/ladder-revisions.json \
    --out bcf/waves/ladder-eval-a100-80-01.tsv
# commit that manifest: its BCF_REVISION values ARE the checkpoint hashes.
```

A row whose checkpoint does not exist keeps `ladder-PENDING-no-checkpoint-yet`, and
`bcf/serve_and_run.sbatch` now **exits 8** on it rather than serving it, because the local
branch compares the pinned revision against the checkpoint's own manifest hash. That
refusal is the intended behaviour: it stops a wave row pointing at a checkpoint other than
the one it names.

### Step 3. Evaluate the two pilot checkpoints

```bash
bcf/ladder_wave.sh --stage evaluate \
  --manifest bcf/waves/ladder-eval-a100-80-01.tsv --check-only
bcf/ladder_wave.sh --stage evaluate \
  --manifest bcf/waves/ladder-eval-a100-80-01.tsv --max 1     # once per checkpoint
```

Each evaluation row carries `BCF_LOCAL_CHECKPOINT=<merged dir>`, which is the only switch
that takes `serve_and_run.sbatch` off the Hub path. Confirm in the cell's `run_meta.json`
that `local_checkpoint` is `true` and that `hf_revision` is the `ladder-<sha16>` string,
because that field is what pins the weights a transcript came from.

### Step 4. The statistic, on what exists

```bash
PYTHONPATH=$BCF_REPO/src python - <<'PY'
from bayes_cot_faithfulness.ladder import statistic
# one CheckpointFit per evaluated cell from that cell's own records, then:
#   report = statistic.build_report(fits)
# statistic.sd_pilot(report) will REFUSE with 1 pair. That refusal is correct and it is
# the point of section 2: two checkpoints cannot value sd_pilot(D) or the MDE.
PY
```

Report the pilot's single D with its bootstrap interval, beside the element 11(e)
comparison set, and label it what it is: one difference, not an sd, not an MDE, and not a
row of A3.2.

## 7. The budget, against CONTRACT.md line 24

`CONTRACT.md` line 24, quoted:

> Ladder budget per base (ESTIMATE; partition per A2 element 11): 12 checkpoints = 3
> trigger doses x 2 training seeds x (organism, twin), the lowest dose rung spent on an
> openly disclosing trigger learner and a trigger-present-but-uninformative control;
> about 2 h LoRA each = 24 card-hours; generation 12 x 500 items x 12 generations per
> item = 72,000 generations at about 7 per second (8B) = about 3 card-hours; judging 12 x
> 300 hinted x 3.33 x 3 = about 36,000 votes = about 5 server-hours; about 35 card-hours
> per base in two 48-hour jobs.

### The pilot's own cost

| Line | Arithmetic | Card-hours | Source |
|---|---|---|---|
| Trace banking | 1,077 items x 1 generation x 0.41005 s / 3600 | 0.123 | job 826025, 0.41005 s per full generation |
| LoRA, 2 checkpoints | 2 x 2.0 h | 4.000 | section 12.1, "about 2 h LoRA each"; ESTIMATE, never measured |
| Evaluation, 2 cells | 2 x 0.681 h | 1.362 | `serve_manifest.expected_hours()`: 6,000 generations at 2.9196 gen/s = 0.571 h, plus a clean pass 0.057 h and a cue pass 0.053 h |
| **Pilot total** | | **5.485** | |

5.485 of the ladder's 32.172 planned card-hours is **17.05 percent**, and 5.485 of
CONTRACT line 24's "about 35 card-hours per base" is **15.7 percent**.

### The whole ladder, unchanged by this plan

| Line | This plan | CONTRACT.md line 24 / section 12.1 |
|---|---|---|
| Checkpoints per base | 12 | 12 |
| LoRA card-hours | 12 x 2.0 = 24.000 | "about 2 h LoRA each = 24 card-hours" |
| Items per checkpoint | 500 | 500 |
| Generations per item | 12 | 12 |
| Completions | 12 x 500 x 12 = 72,000 | "12 x 500 x 12 = 72,000" |
| Evaluation card-hours | 12 x 0.681 = 8.172 | "about 3 card-hours" at the first-order 7 gen/s rate |
| Ladder total | 24.000 + 8.172 = 32.172 | "about 35 card-hours per base" |
| Bases | 1 (Qwen3-8B) | "One base (Qwen3-8B) ... a second base is not planned" |

The evaluation line is the one that has moved, and upward, from about 3 to 8.172
card-hours. Section 12.1 priced 72,000 completions at about 7 per second; ruling R2's
measured rate is 2.9196 generations per second on a MIG 3g.40gb slice at 32 in flight
with the batch-invariant flag on (job 826020), which prices the same 72,000 completions
at 6.85 hours of arm time. Every figure is a FLOOR: the a100-80 whole card is faster than
the MIG slice by the measured 1.7371 sequential ratio, and no ladder checkpoint has ever
been served. 32.172 is still inside section 12.1's "about 35 card-hours per base", and
ruling R2 records the element 16 degradation trigger as NOT TRIGGERED at 261.7 priced
card-hours against a budget of record of about 650, a ratio of 0.403, so nothing here is
being cut for compute.

The pilot spends no card-hours the full ladder was not already going to spend, because
its two checkpoints are 2 of the 12 and its two evaluations are 2 of the 12. The only
line it adds is the 0.123 card-hours of trace banking, which the full ladder needs once
regardless.

## 8. What ruling R14 has to decide

1. The three dose values, and whether 0.30 / 0.60 / 0.90 stand.
2. The trigger prevalence, and whether it stays fixed across rungs (which is what makes
   11(c)'s matched frequency exact).
3. The two training seeds.
4. `N_TRAIN_EXAMPLES` against a 1,077-item pool: section 5.3's option 1, 2 or 3. **This
   one blocks the first real training job**, because the default refuses.
5. The disclosing learner's coupling: 11(c)'s "high text dependence" against R6's
   lowest-rung placement.
6. The LoRA recipe of 5.2, at least to the extent of confirming it is not inert.
7. The element 11(d) instrument freeze commit, dated and hashed, without which
   `heldout_family.py` refuses to generate anything.

R14 and the additive amendment A6 come AFTER the pilot and BEFORE the remaining 10
checkpoints. That order is the whole reason this document exists.

## 9. What this plan does not do

It does not train anything, freeze anything, rule on anything, or change any value in
`spec.py` or `lora_train.py`. It does not submit an sbatch job and it does not touch the
cluster. It does not fill the two gaps it names: the transcripts-to-traces reduction has
no script, and the bcf env's actual transformers version is unread.
