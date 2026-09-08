# The logit-level pass: what it writes, how to submit it, and what the fits lane has to change

`docs/OUTCOME-SCALE-NOTE.md` part 4.5 splits the logit-level work into two jobs. Job A
re-reads letter logprobs that already exist on the anchor cells and needs no generation.
Job B is the one this document is about: a new pass over records that already exist, which
reads the answer-letter logprobs of each record's own two prompts and writes them beside
the record without touching it.

It exists because of one measured fact. `docs/CELLS18-FITS.md` section 7 item 3 records
that 18 of 18 cells fail A5.4's gate G1 on the same condition, condition 3: no arms record
in this project carries `intervention_level = logit`, because the generation pass that
would write it has never been run. Every other condition of that gate is either already
satisfied or not computable until this one is. The outcome scale whose clean arm varies is
therefore still unmeasured on the arms, while the anchor cells show the contrast the note
is about: on the same 28 items and the same target letter, the binary outcome has variance
exactly 0.0000 in the mu00 cell and the letter margin has a standard deviation of 3.227
nats across an 11.5 nat range (part 4.3).

Three files do the work, and one manifest drives them:

| file | what it is |
|---|---|
| `experiments/logit_pass.py` | the pass itself, over ONE cell directory |
| `bcf/logit_pass.sbatch` | one model, one server, all of that model's cells |
| `bcf/logit_wave.sh` | the cap-checked submitter, one model row at a time |
| `bcf/waves/logit/logit-a100-40-01.tsv` | the three cells-18 models and their six cells each |

---

## 1. What the pass writes, and where

Four files per cell, all inside the CELL directory, because that is where the fits read a
cell from. Nothing that already existed there is written.

**`arms_transcripts_<model>.logit.json`** is the augmented sidecar. Each entry is a full
copy of one banked record plus:

* `clean_answer_logprob` and `hinted_answer_logprob`, each carrying the six keys
  `outcome_scale.letter_logprob_fields` returns (`answer_logprobs`,
  `logprob_source_token`, `renormalized_over_letters`, `letter_probability_mass`,
  `logprob_margin`, `target_letter`) plus `method`, which is A5.2's stored form exactly;
* `intervention_level = logit` and `outcome_scale = logprob_margin` on the record;
* the record-level CONTRACT fields `answer_logprobs` and `logprob_source_token`, which are
  null on a text-level record, filled here as `{"clean": ..., "hinted": ...}` because one
  record now carries two reads and a flat map would silently be one of them;
* a `logit_pass` block, which is the keying described in section 4.

Every field the text-level fit reads is still on the record, untouched: `hint_label`,
`clean_curve`, `hinted_curve`, the anchor block, the follow flags. The sidecar is a
drop-in for anything that reads the arms transcripts today.

**`logit_check.json`** is the section 9.1 unit check, written in the shape
`experiments/results/phase1-anchor-mig/qwen3-8b/arc_challenge/stated-hint/logprob_check.json`
uses, so the arithmetic a reader already runs over that artifact runs over this one:
`results` holds one row per read with `n_letters_requested`, `n_letters_scored` and
`n_tokens_matching_letter`, and `passed` and `hard_failures` sit at the top. The scope is
different and the file says so: those two probes were synthetic and this is every read the
pass made, on the real prompts, over each item's own answer letters.

`passed` is true when no read that RETURNED is an alignment failure and at least one read
returned. The four alignment failures are a letter that was not scored, a logprob read off
a token that does not decode to its own letter, a logprob above zero, and letter mass above
1. A read that never returned is counted under `n_incomplete_reads` instead: it drops its
item, and a transport failure says nothing about where a logprob was read, so it does not
condemn the family. That distinction is the one place this artifact's `passed` differs from
`bcf/logprob_check.py`'s, and `passed_rule` in the file states it in words.

**`logit_pass_meta.json`** carries what ran and what it dropped: the endpoint, the revision,
the chat template kwargs, the source file and its sha256, the drop counts by reason, the
per-arm margin mean and variance, and the randomized arm difference in the margin. The last
three are the inputs to A5.4's conditions 4 and 5, printed so a reader can see them without
running the fit.

**`logit_pass_checkpoint.json`** is the resume state. Section 3 says what it guarantees.

The job's own files (`run.log`, `server.log`, `run_meta.json`, `logprob_check.json`,
`determinism_preflight.json`, `exit_code.txt`, `cell_status.txt` and the completion marker
`logit_pass_done.json`) go to `$BCF_RESULTS/logit-pass/<model slug>/` and NOT into a cell,
because a cell already has a `run.log` and an `exit_code.txt` from the run that generated it
and overwriting either would destroy the record of that run.

---

## 2. The gate, and what a failure means

A5.4 condition 1 and section 9.1: the per-family forced-logprob unit check must pass before
any of that family's logit-level cells are reported, and a family that fails is reported at
the TEXT level only. Two checks stand, in this order:

1. `bcf/logprob_check.py` on the live server, before any cell is touched. This is the same
   script and the same exit code (7) `bcf/serve_and_run.sbatch` uses, run with the same
   `--chat-template-kwargs` the cells were generated with, because under a different
   template the forced letter sits after different preceding tokens and the check would
   certify a different read.
2. `logit_check.json` per cell, over the pass's own reads. The first cell that fails it
   stops the model: the remaining cells are not run, because their rows could not be
   printed either.

Both report at the text level only when they fail. Nothing partial is published: the sbatch
writes no completion marker, so the exit guard writes 7 rather than 0 and
`exit_code.txt` alone tells a reader that this model has no complete logit-level pass.

Two other refusals are worth naming because they are absolute:

* **The endpoint.** `experiments/logit_pass.py` refuses any `--endpoint` that is not
  loopback, and exits 3 before a client object exists. Section 6.7 makes SoCLaaS ineligible
  as a source of logprob outcomes and A5.4 condition 2 requires the pinned self-hosted vLLM
  endpoint of element 8. There is no flag that turns this off.
* **The revision.** The sbatch resolves the Hub revision, compares it with the wave's
  `BCF_REVISION`, and then compares it with EVERY named cell's own `run_meta.json`
  `hf_revision`. Reading logprobs off different weights would produce a logit-level row
  about a model the text-level row is not about, and that is the one drift that leaves no
  trace in the output. A mismatch exits 8.

---

## 3. Resume, and the assertion before every write

`experiments/logit_pass.py` writes only in one function, `flush`, and `flush` runs
`outcome_scale.assert_records_scaled` on the whole sidecar batch FIRST. That is section
9.6's assertion, the one `write_arm_transcripts` already makes on the text level, and it
refuses rather than warns for the same reason: a mislabelled record would be pooled across
intervention levels downstream, and banking it is worse than stopping. A refusal exits 8 and
leaves no sidecar and no checkpoint behind.

Items are consumed strictly in source order even when 32 reads are in flight, so a
checkpoint always holds a PREFIX of the items and never a hole with a later item filled past
it. A second leg reads the checkpoint, skips what is banked, and scores the rest.

Two conditions guard the resume. The checkpoint records the sha256 of the transcripts file
it was computed from, and a mismatch refuses with exit 5 instead of mixing two source files.
Then each restored entry's `record_key` is recomputed from the source record at that entry's
own index, and a disagreement refuses too. `--restart` ignores the checkpoint and scores
everything again.

An item whose read fails on EITHER arm is dropped whole, with a reason, and the reason is
counted. The unit is the item because the fit's two rows are the two arms of one item:
banking a clean margin whose hinted partner failed would put an item into the analysis on
one arm only, which is not a row `build_table` knows how to drop later. The reasons that can
appear are `hint_label_not_in_item_labels`, `record_not_rebuildable`,
`cue_family_unresolved`, `cue_family_mismatch`,
`hinted_prompt_does_not_contain_the_banked_cue_text`, `client_error`, `no_logprob_margin`
and the four alignment failures of section 1.

---

## 4. How the sidecar is keyed to the record it came from

Three independent things have to agree before a sidecar entry can be read as belonging to a
record, and the pass writes all three:

1. **Position.** `logit_pass.record_index` is the entry's index in the source array. The
   sidecar is written in source order, so it is also positionally aligned with the source
   minus the dropped items.
2. **Content.** `logit_pass.record_key` is a sha256 over the six fields that identify which
   question was asked and which cue was planted on it: the question, the choices in their
   banked order, the gold label, the planted `hint_label`, the `cue_text`, and whether the
   cue was prepended. Two records that agree on all six were the same measurement. The key
   is what catches a source file that was regenerated between two legs.
3. **File.** `logit_pass.source_file` and `logit_pass.source_sha256` name the exact bytes
   the entry was computed from, and `logit_pass.model`, `hf_revision` and `endpoint` name
   the weights and the server that produced it.

Beyond keying, the pass will not read a prompt it cannot prove is the record's own. The cue
family is not taken from a run parameter: every frozen cue template is rendered against the
record's own `hint_label` and compared with the banked `cue_text` byte for byte, the
resolved family must equal the family the cell says it ran, and the rebuilt hinted prompt
must contain the banked `cue_text`. Any of those failing drops the item. The prompts
themselves come from `experiments/08_additive_arms.py::_frame_prompt`, called rather than
reimplemented: the module is loaded by file path and its function is used as is, so a later
edit to the runner's framing cannot leave this pass behind quietly. Both prompt hashes are
stored on the record.

---

## 5. How to submit it

The wave submits ONE MODEL PER ROW, because one server does all of that model's cells and
the expensive part is loading the weights. Check first, always:

    bcf/logit_wave.sh --check-only bcf/waves/logit/logit-a100-40-01.tsv

then submit. `bcf/logit_wave.sh` makes the four cap checks `bcf/wave.sh` makes (32 jobs in
system, the per-user cap for the card type counting every campaign, the total 12-card cap,
and this job type's own budget) with the pool guard on, plus the two physical refusals
(a tensor-parallel size larger than the cards on one node, a wall clock above the
partition's ceiling), and it syncs an immutable tree named after the planning commit exactly
as `bcf/wave.sh` does. It never reads or writes `~/bcf/repo`.

The budget is 1 card. CONTRACT.md gives this job type no split of its own, so the number is
the ladder's and the probe's posture ("probe or ladder 1") rather than a claim on the pool,
and the three models go out one at a time. Widening it is a CONTRACT.md decision.

Two shapes of the manifest are load bearing and easy to get wrong:

* `BCF_CELLS` is separated by `+`, never a comma. `sbatch --export` is itself a
  comma-separated list, so a comma inside a value truncates it and the rest is read as names
  of variables to inherit, silently, which is how job 826024 lost ten of its eleven arms.
  The wave script refuses a `BCF_CELLS` containing a comma or a space.
* Field ORDER does not matter and the last field is not special. Rows are split by
  `row_fields` in `bcf/wave_lib.sh`, the fixed version; the shape it replaced wrote no
  trailing newline and a `while read` loop dropped every row's last field, which on
  2026-09-08 cost job 828564 its own `BCF_OUT_SUBROOT`.

Manifests live in `bcf/waves/logit/`, one directory down from the sweep manifests.
`bcf/check_wave_manifests.py` and `tests/test_wave_plan.py` glob `bcf/waves/*.tsv` without
recursing and count everything that is not `enrich-`, `ladder-` or `-resub-` as sweep cell
rows of the 216-cell grid; a model row beside them would be counted as a cell.

---

## 6. What the fits lane has to change, in one place

**This lane does not edit the fits files.** What follows is the change they need, stated
exactly, for the lane that owns `experiments/wave1_fits.py` and `experiments/cells18_fits.py`
to make.

### 6.1 The loader has to separate two source files, not one

`experiments/wave1_fits.py::load_records` keeps a row when its `source_file` starts with
`arms_transcripts`. The sidecar is named `arms_transcripts_<model>.logit.json`, so it starts
with `arms_transcripts` too. That matters twice:

* **The sidecar is invisible today.** `experiments/cells18_fits.py::run_fit` reads
  `cell_dir / "transcripts.jsonl"`, and `bcf/contract_layout.py` builds that file by
  concatenating `*transcripts*.json`. The pass does not run `contract_layout.py` and the
  sbatch does not either, so `transcripts.jsonl` still holds only the text-level rows and
  gate G1's condition 3 still reads them and still fails.
* **Running `contract_layout.py` again would double the text-level denominator.** Its glob
  matches the sidecar, so a rebuild after the pass would write every item twice into
  `transcripts.jsonl`, once from each file, and both copies start with `arms_transcripts`,
  so `load_records` would keep both and the text-level table would silently run on 2n rows.

So the change is one filter, split into two buckets:

```python
# experiments/wave1_fits.py::load_records
src = rec.get("source_file") or ""
if src.startswith("arms_transcripts") and not src.endswith(".logit.json"):
    arms.append(rec)          # the text-level rows, exactly as today
elif src.endswith(".logit.json"):
    logit.append(rec)         # the logit-level rows, new
else:
    other[src] += 1
```

The first clause is the guard against the doubling and is worth making even if the logit
rows are never used. The second is how the logit rows arrive.

### 6.2 Where the fit should read the sidecar from

Reading `transcripts.jsonl` for the logit rows requires `contract_layout.py` to have been
re-run in that cell, which is an extra step with the doubling hazard attached. The simpler
route, and the one this lane recommends, is for `cells18_fits.run_fit` to read the sidecar
file DIRECTLY:

```python
sidecars = sorted(cell_dir.glob("arms_transcripts_*.logit.json"))
logit_records = json.loads(sidecars[0].read_text()) if len(sidecars) == 1 else []
```

and to join it to the text-level records by `rec["logit_pass"]["record_key"]`, asserting
that `rec["logit_pass"]["source_sha256"]` equals the sha256 of the transcripts file the
text-level table was built from. That assertion is the whole point of the keying in section
4: it makes "these two rows are the same item" checkable rather than assumed.

### 6.3 What gate G1 then reads

With the sidecar loaded, `logit_g1_gate`'s six conditions become computable, and each one
has a place to read from:

| condition | where it reads from now |
|---|---|
| 1 unit check passed | `cell_dir / "logit_check.json"` (`passed`, `hard_failures`, and the same `results` arithmetic it runs on `logprob_check.json` today), plus the family-level `logprob_check.json` in `$BCF_RESULTS/logit-pass/<model>/` |
| 2 pinned self-hosted endpoint | `logit_pass_meta.json` `endpoint`, and `method == "prompt_logprobs"` on every stored block |
| 3 records carry the logit scale | the sidecar records: `intervention_level`, `outcome_scale` and `logprob_source_token` on each, and `assert_records_scaled_checked` in `logit_pass_meta.json` as the denominator the runner asserted |
| 4 clean-arm variance strictly positive | computed from `clean_answer_logprob.logprob_margin` across the sidecar; `logit_pass_meta.json` prints the same number under `clean_arm_margin` |
| 5 TE_logit equals the arm difference | the fit's TE against `randomized_arm_difference_in_the_margin` in `logit_pass_meta.json`, to within 1e-6 |
| 6 letter mass summarised | `logit_check.json` `letter_probability_mass`, min, median and max |

Condition 1 has a subtlety worth writing down. `logit_check.json`'s `results` rows are the
reads that RETURNED, so the arithmetic `scored == requested == matching` is a statement
about reads that happened, and the reads that did not are in `n_incomplete_reads` beside it.
A gate that wants to be stricter than the pass should read that field as well rather than
assume it is zero.

### 6.4 And the analysis, which is one substitution

Part 4.5 job B step 5, unchanged: `build_table` with `y0 = clean_answer_logprob.logprob_margin`
and `y1 = hinted_answer_logprob.logprob_margin`, both dropped with a reason when null, M and
X untouched, fitted with `gaussian_mediation.fit_gaussian_mediation_closed_form(X, M, Y,
rho=0.0)`, effects from `gaussian_natural_effects`, the sweep from `gaussian_effects_curve`
on the symmetric A4.6(a) grid, and `rho*_point` from `gaussian_rho_star_point`. A5.5 fixes
the printing order across the two scales and A5.6 fixes what goes where a verdict would go:
`verdict: not applicable, no threshold pre-registered on this scale`.

---

## 7. Limits of this pass, stated before the numbers are

1. **The card type is not held fixed against the run being augmented.** The 18 text-level
   cells were not all generated on one card: 11 ran on a MIG 3g.40gb slice of an A100 80GB
   and 7 on a whole a100-80, from each cell's own `run_meta.json` `gpu` field. One server per
   model means all six of a model's reads share one card type, whichever the wave allocates.
   The pass records its own serving mode, and the difference is visible rather than hidden.
2. **The read is a new measurement, not a recovery of something the original run computed.**
   The original run never scored these prompts for letter logprobs. What makes it the same
   measurement is the prompt, and the prompt is rebuilt from the record's own banked fields
   through the frozen instruments, checked three ways (section 4). What is not guaranteed is
   the server: it is a different server process on a different day, which is why the pinned
   mode, the batch-invariant flag and the determinism preflight all run again here.
3. **The mass caveat travels with every margin.** Part 4.4 measured raw letter mass between
   3.29e-18 and 3.96e-14 on the only records in this repository that carry the block, 112 of
   112 below 0.01. A margin computed where the letters hold that little of the next-token
   mass is well defined and is also about a region the model almost never enters. A5.4
   condition 6 makes printing the summary a condition of printing the row, and no floor is
   set, because every measurement of that quantity this project has is smoke sized.
4. **A passing gate is not a verdict.** A5.6 does not set a load-bearing threshold on this
   scale and this lane does not either. A logit-level row that clears G1 prints its three
   effects with intervals, its bridge, its mediated share and its `rho*_point`, and prints
   `verdict: not applicable, no threshold pre-registered on this scale` where a verdict would
   go. It is never used for promotion, ranking or a claim-status change.
