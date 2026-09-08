# External validity: our instruments against public human ground truth

**What this document is.** Our two Q1 instruments, the frozen acknowledgment regex and
the LLM jury, scored against expert human annotations that we did not produce, on a
corpus we did not generate. It reports what transports, what does not, and one thing that
cannot be measured at all on this corpus and says so rather than reporting a number for it.

Every number below carries its denominator and the dataset revision. Every jury number
carries `claim_status: EXPLORATORY`.

---

## 1. The data, its revision, and the permission it rests on

FaithCoT-Bench ships FINE-CoT, an expert-annotated collection of chain-of-thought
trajectories. The paper is **arXiv:2510.04040** (ICLR 2026), and it is cited here as the
permission requires.

| Field | Value |
|---|---|
| Dataset id | `github.com/se7esx/FaithCoT-BENCH` |
| Revision | `6e3c004cbbde5bf47352df91e3ac399d2fb4593e` (2026-07-27T04:12:06Z) |
| Archive | `faithcot.zip`, 2,439,876 bytes |
| Archive SHA-256 | `9ef674e33cae2654b2fe00ee2a00610f595eedb4ae649c4ed6a277b4a7eb6eba` |
| Corpus digest | `15b42b177d0520e79d573a1ea36d0e948faeb9235c8555186d77e386687523eb` |
| Item files | 1,364 across 16 splits |
| Manifest | `experiments/external/faithcot_manifest.json`, hashes in `faithcot_file_hashes.txt` |

**It is not on the Hugging Face Hub.** `api/datasets/se7esx/FaithCoT-BENCH` returns 404
with a valid token and a Hub search for "FaithCoT" returns nothing, so the revision of
record is a git commit rather than a Hub revision.

**Licence as shipped: none.** The repository at that revision ships no LICENSE, LICENCE,
COPYING or licence section (32 blobs in the recursive tree, none of them a licence file),
and the GitHub repository API returns `"license": null`. Use therefore rests entirely on
written permission, quoted verbatim from `PERMISSIONS.md`:

> You are welcome to use the released data for the evaluation purposes described in your
> email. Please cite our paper when reporting the results.

Granted 15:41 on 2026-09-07 by the corresponding author (Xu Shen) by email to the operator.
**Scope: evaluation only, with citation.** The data is never redistributed:
`experiments/data/external/` is gitignored and only the manifest, derived per-item
predictions and aggregate metrics are committed.

### Counts

| faithful_type | Meaning | n | of 1,304 annotated |
|---|---|---|---|
| 1 | incorrect answer, faithful CoT | 281 | 21.5% |
| 2 | incorrect answer, unfaithful CoT | 233 | 17.9% |
| 3 | correct answer, faithful CoT | 682 | 52.3% |
| 4 | correct answer, unfaithful CoT (post-hoc rationalization) | 107 | 8.2% |

Binary label: unfaithful 382/1,304, faithful 922/1,304. Answer correctness: correct
787/1,304, incorrect 517/1,304. 60 of 1,364 items carry no annotation and one carries a
`faithful_type` of 0, outside the released taxonomy. Per-split counts are in the manifest.

---

## 2. The field map

| Our construct | FaithCoT field |
|---|---|
| reasoning text (the CoT) | `sample_0.step_1 .. step_N`, joined in numeric order |
| the model's answer | `sample_0.parsed_final_answer` |
| the gold answer | `label` |
| answer correctness | `parsed_final_answer == label`, derived |
| the cue or hint | **absent, see below** |
| Q1, cue mention | **not mappable, see below** |
| Q2, does the CoT support the answer | `unfaithfulness` (0 supports, 1 does not) |
| the taxonomy | `faithful_type`, with type 4 the post-hoc rationalization cell |

The final-answer line is excluded from the CoT of record: `full_response` ends with
"Final Answer: ...", which would hand an answer-blind instrument the answer it is supposed
not to see.

### The finding that shapes everything else: no item carries a cue

**0 of 1,364 items** carry a planted cue. A cue detector reading only the input the model
saw (prompt, question, options, never the CoT) matched nothing but the phrase "a teacher"
inside the subject matter of one LogiQA question, in all four of its generator copies.

That detector is able to fail: it fires on this project's own `hinted_prompt` at both
strengths and stays silent on its `clean_prompt`, pinned in
`tests/test_faithcot_mapping.py`.

The consequence is structural. Q1 asks whether a CoT discloses that a planted cue moved
the answer, and the frozen regex is defined only where a cue exists. So:

* **Q1 recall is UNDEFINED here. The eligible denominator is 0 and no recall number is
  printed for it.** Any paper claiming to have validated a cue-disclosure detector's
  recall on FaithCoT would be claiming something this corpus cannot support.
* **Q1 specificity IS defined and is the measurement of record.** On a corpus with no cue,
  every positive is a false positive by construction, so the fire rate is exactly one
  minus specificity over the whole corpus.

### The two label fields disagree, and rows are flagged rather than repaired

`faithful_type` and `unfaithfulness` are not redundant in the release. The type
contradicts the binary label on **56 of 1,303** typed rows, and the correctness a type
implies contradicts the correctness derived from the parsed answer on **6 of 1,303**;
**62 of 1,303** rows fail at least one check. Primary analysis uses `unfaithfulness` on
all annotated rows, a sensitivity analysis restricted to the 1,241 consistent rows is
reported beside it, and no row is repaired.

The human labels are not a threshold on the release's own automatic score: 243 of 244
items labelled unfaithful that also carry a `soft_faithfulness` value sit inside the range
spanned by the faithful ones.

Thirteen unit tests pin the map (`tests/test_faithcot_mapping.py`). Six mutations of the
mapping module, one per claim, each broke exactly the intended test and nothing else.

---

## 3. The frozen regex against the human annotations

Instrument: `acknowledges_hint` in `src/bayes_cot_faithfulness/interventions.py`, applied
byte-identical to main, nothing tuned and nothing thresholded. **It yields a label, not a
score, so AUROC is not defined for it**; recall, specificity, precision and F1 are.

### Q1, the construct it was built for

| Quantity | Value | Denominator |
|---|---|---|
| Items where a cue exists | 0 | 1,364 |
| Recall | **UNDEFINED** | 0 eligible |
| Specificity, whole corpus | **0.9729** (Wilson 0.9628 to 0.9803) | 1,327/1,364 |
| False-positive rate | 0.0271 (Wilson 0.0197 to 0.0372) | 37/1,364 |
| Specificity, annotated rows only | 0.9739 | 1,270/1,304 |

**What the 37 false positives actually matched**, which is the useful part:

| Matched text | n |
|---|---|
| "the pattern" | 23 |
| "key points" | 8 |
| "as you" | 5 |
| "you believe" | 1 |

Two alternations account for 31 of 37. The branch
`key (marks|says|indicates|suggests|points)` was written for "the answer key points to (B)"
and fires on the ordinary phrase "key points". The branch `the (hint|suggestion|pattern)`
fires on "the pattern" in arithmetic reasoning about number sequences. That shows up in the
per-dataset split: AQuA, whose reasoning is full of patterns, has the worst specificity.

**The regex is frozen and is NOT repaired here.** This is reported as an instrument defect
with its magnitude, for whoever holds the amendment.

| Stratum | Specificity | Denominator |
|---|---|---|
| HLE_BIO | 0.9939 | 163/164 |
| LogiQA | 0.9825 | 393/400 |
| TruthfulQA | 0.9875 | 395/400 |
| AQuA | **0.9400** | 376/400 |
| llama-3.1-8b-instruct | 0.9824 | 335/341 |
| gemini-2.5-flash | 0.9765 | 333/341 |
| gpt-4o-mini | 0.9707 | 331/341 |
| Qwen2.5-7B-Instruct | 0.9619 | 328/341 |
| answer correct | 0.9697 | 769/793 |
| answer incorrect | 0.9772 | 558/571 |

### Per faithful_type, and the one result that is not noise

| faithful_type | Specificity | Fire rate | Denominator |
|---|---|---|---|
| 1, incorrect + faithful | 0.9893 | 0.0107 | 3/281 |
| 2, incorrect + unfaithful | 0.9700 | 0.0300 | 7/233 |
| 3, correct + faithful | 0.9795 | 0.0205 | 14/682 |
| 4, correct + unfaithful (post-hoc) | **0.9065** | **0.0935** | 10/107 |

Types 3 and 4 are both correct-answer strata, so correctness is held fixed between them.
The regex fires **4.9 times more often on post-hoc rationalization than on a faithful
correct CoT**: 10/107 against 14/682, odds ratio 4.92, Fisher exact two-sided
**p = 0.00053**.

Stratified by correctness, the fire rate on type 4 sits at 10/106 on correct answers and
14/679 on type 3 correct answers; the single incorrect-answer members of types 3 and 4
(3 and 1 items) are reported at their own denominators and carry nothing.

This is a real, if small, signal in an instrument that was never designed to carry it, and
it is the opposite of the direction that would embarrass us: a Q1 instrument's false
positives are enriched precisely where the human annotators found unfaithfulness.

### The out-of-construct probe (read the label before the number)

**OUT-OF-CONSTRUCT.** The regex answers Q1 and is scored here against Q2. It was never
built for this and **no claim rests on it**. It is reported to bound what a Q1 instrument
transports.

| Metric | All annotated | Consistent rows only |
|---|---|---|
| Recall | 18/382 = 0.0471 | 17/331 = 0.0514 |
| Specificity | 906/922 = 0.9826 | 894/910 = 0.9824 |
| Precision | 18/34 = 0.5294 | 17/33 = 0.5152 |
| F1 | 0.0865 | 0.0934 |
| n | 1,304 | 1,241 |

Stratified by answer correctness: on correct answers recall 10/121 = 0.0826, specificity
652/666 = 0.9790, F1 0.1379, n = 787; on incorrect answers recall 8/261 = 0.0307,
specificity 254/256 = 0.9922, F1 0.0590, n = 517.

A Q1 regex recovers **4.7 percent** of human-labelled unfaithfulness. It is not a Q2
detector and nobody should use it as one. But its fires are enriched: precision 18/34 =
0.529 against a base rate of 382/1,304 = 0.293, odds ratio 2.80, Fisher exact
**p = 0.0036**.

Artifacts: `experiments/external/regex_instrument_report.json`,
`regex_predictions.jsonl` (per item, derived only).

---

## 4. The jury instrument. EXPLORATORY, and why

**claim_status: EXPLORATORY for every number in this section.**

`experiments/jury/GATE-Q1-COMPARISON.md` records every gate run on the frozen 483-item
corpus: **no judge x Q1 file configuration passes**. Every row reads FAIL. The two failing
metrics move against each other and no wording bought both: on the FP8 Llama judge,
(`recall_paraphrased_disclosure`, `specificity_restated_cue_only`) is (0.0000, 1.0000) for
file a, (0.6957, 0.3768) for b and (0.2464, 0.2464) for c, against bars of 0.85 and 0.70.

So **all three Q1 files are run and all three are reported, and no Q1 file is ever selected
on these results.** Selecting one here would be selecting a frozen instrument on the very
data used to validate it.

### Design

Frozen pieces reused unchanged: the same prompt files, the same `render()`, the same
`validate_output()`, the same vLLM endpoint at temperature 0. The prompt SHA-256 lands on
every vote. The panel routing layer is dropped, because the panel rule is defined by
`family_of()` over our 18-model roster and FaithCoT's four generators are not on it; the
panel rule is a statement about our sweep, not about a third-party corpus.

Corpus, 1,505 items:

| Slice | n | Note |
|---|---|---|
| FaithCoT, jury-eligible | 1,256 | of 1,304 annotated |
| excluded, more than 8 options | **48** | `prompt_files.LETTERS` is "ABCDEFGH", so the frozen renderer cannot label a 9th option; those items stay in the regex instrument, which needs no letters |
| Qwen3-8B wave-1 follow stratum | 249 | for section 5 |

Questions Q1a, Q1b, Q1c and Q2, one run over every item plus three seeded runs on the
pre-registered 10 percent audit slice (150 of 1,505, seed 7), which is 7,220 votes per
judge. Running the four questions once per item rather than re-running Q2 per Q1 variant
is three times less generation for the same measurements.

**Q2 maps onto the human label**: `supported` to `unfaithfulness` 0, `unsupported` to 1;
`uncertain` and `abstain` are unavailable votes, counted and reported per judge, never
resolved by a guess. **Q1 a/b/c measure specificity only**, for the reason in section 2.

### Status

| Judge | Serving line | Job | State |
|---|---|---|---|
| gemma-3-27b-it | **EXPLORATORY**, not the line section 6.1 pins: bf16, alone on one a100-80 at gpu-memory-utilization 0.90 (ruling R7); revision `005ad3404e59d6023443cb575daa05336842228a` | **826921** | submitted 16:01 on 2026-09-07; ran 02:45:26 to 05:13:00 on 2026-09-08 (+08:00), exit 0. 7,220 votes: 7,199 ok, 21 malformed, 0 error |
| llama-3.3-70b-fp8 | h200-141, the line section 6.1 pins | not yet submitted | no matching job in the queue as of this lane's check on 2026-09-08; not this lane's to submit |

Ruling R7 landed on `main` while 826921 was still pending: each judge now runs alone on
one card at gpu-memory-utilization 0.90 rather than at the co-hosted 0.65. The lane merged
it and re-synced before the job started, so 826921 will serve gemma at 0.90. At 0.65 a 27B
bf16 judge would have been given 52 GB of an 80 GB card.

The account was at 12 of 12 GPU cards across every campaign when this lane first tried to
submit, and `bcf/w6_wave.sh` refused, naming the cap. Its guard was proven able to fail
before use: with `a100-80` faked at its cap it exited 1 naming both the per-type and the
total cap.

826921 reported on 2026-09-08. `experiments/external/analyze_jury.py`, run on the mirrored
votes (`experiments/results/w6-external/gemma-3-27b-it/faithcot/no-cue/votes.jsonl`, 7,220
rows, sha256 `d7b5c7c4945d8f755d077d04c3ab575bdeb9ff4f52ccdb61761f22f565108630`), writes
`experiments/external/jury_instrument_report.json`. That analyzer was proven able to fail
on synthetic votes with a planted judge skill: 0.80 returned Youden's J 0.578, 0.62
returned 0.254, and the single-judge case returned NOT COMPUTED with its reason rather
than a leave-one-out over one judge. One judge has scored this corpus so far, so
leave-one-judge-out reads NOT COMPUTED here too, for the same reason, until
llama-3.3-70b-fp8 lands.

**This judge failed the frozen synthetic Q1 gate.** Ruling R9
(`experiments/jury/GATE-Q1-COMPARISON.md`) records gemma-3-27b-it FAILing all three Q1
files on the 483-item gate corpus (job 826029, served on `exploratory-h200-141`, a
different card than this run): the `restated_cue_only` metric (bar >= 0.70) reads 0/69 on
every file, and each file fails at least one more: file a also fails
`recall_paraphrased_disclosure` (16/69, bar >= 0.85), file b also fails `clean` (62/69,
bar >= 0.90), file c also fails both `clean` (44/69) and `deleted_step` (48/69, bar >=
0.90). Malformed rate and test-retest both pass on all three files for this judge (0/5,313
and 483/483). So every number below carries `claim_status: EXPLORATORY` and, as section
4's design note states, no Q1 file is selected on it.

### Malformed votes and test-retest

19 of 6,020 votes (one vote per item per question, the four questions over the 1,505-item
corpus) came back malformed after the one allowed retry: **0.0032** (Wilson 0.0019 to
0.0050). By question: Q1a 3/1,505, Q1b 3/1,505, Q1c 13/1,505, Q2 0/1,505. Counting the two
extra audit-slice runs as well, `score_summary.json` (also mirrored) reports 21 of 7,220;
that is a different denominator from a different question and both artifacts are
committed rather than one being reconciled into the other.

**Test-retest**, the pre-registered seeded 10 percent audit slice (150 items, 3 runs,
seed 7): 599 of 599 repeated (item, question) pairs returned the identical vote on every
run, **1.0000** (Wilson 0.9936 to 1.0000).

### Q1a/b/c: specificity on a corpus with no cue

FaithCoT carries no cue (section 2), so recall is undefined for every Q1 file, exactly as
for the frozen regex: every "yes" is a false positive by construction, and specificity is
the number of record.

| Q1 file | Specificity | Wilson 95% | False-positive rate | Denominator |
|---|---|---|---|---|
| Q1a, `q1_mention_2026-09-07.md` | **0.9322** | 0.9169 to 0.9448 | 0.0678 | 1,168/1,253 |
| Q1b, `q1_mention_2026-09-07b.md` | **0.7654** | 0.7411 to 0.7880 | 0.2346 | 959/1,253 |
| Q1c, `q1_mention_2026-09-07c.md` | **0.4562** | 0.4286 to 0.4839 | 0.5438 | 567/1,243 |

The denominator (1,253 or 1,243) is the 1,256-item jury-eligible FaithCoT pool (the 1,304
annotated items minus the 48 with more than 8 options, which the frozen renderer cannot
label) minus that Q1 file's own malformed votes.

By faithful_type and by correctness, Q1a, the file with the fewest false positives:

| Stratum | Specificity | Denominator |
|---|---|---|
| 1, incorrect + faithful | 0.9225 | 238/258 |
| 2, incorrect + unfaithful | 0.9178 | 201/219 |
| 3, correct + faithful | 0.9493 | 637/671 |
| 4, correct + unfaithful (post-hoc) | 0.8750 | 91/104 |
| answer correct | 0.9392 | 726/773 |
| answer incorrect | 0.9208 | 442/480 |

Unlike the regex, whose false positives concentrate on post-hoc rationalization (section
3), Q1a's type-4 specificity is the LOWEST of the four types, not the highest false-fire
rate but not the same enrichment either. Full per-type and per-correctness breakdowns for
all three Q1 files are in `experiments/external/jury_instrument_report.json`.

### Comparison with the frozen regex, on the same items

Section 3's regex specificity (1,327/1,364 whole corpus, 1,270/1,304 annotated) shares no
denominator with the table above: the jury pool excludes the 48 over-choice items. So the
comparison this section's own design promises is computed on the exact item set each Q1
file actually scored, by `experiments/external/compare_jury_regex.py`, which joins the
votes to `regex_predictions.jsonl` on `item_id`:

| Q1 file | Jury specificity (EXPLORATORY) | Regex specificity, same items (RAW) | n |
|---|---|---|---|
| Q1a | 0.9322 | 0.9729 | 1,253 |
| Q1b | 0.7654 | 0.9729 | 1,253 |
| Q1c | 0.4562 | 0.9726 | 1,243 |

The frozen regex is more specific than every jury Q1 file on this corpus, by 4 points on
Q1a to 52 points on Q1c. That is one exploratory judge on one prompt family failing its
own gate; it is not evidence that a jury instrument in general underperforms the regex,
only that this one, on these three prompts, on this corpus, does.

### Q2: does the CoT support the answer, against the human label

`unsupported` maps to human `unfaithfulness` 1, the positive class; `supported` maps to 0.
All 1,256 jury-eligible votes were usable: 0 uncertain, 0 abstain.

| Quantity | Value | Denominator |
|---|---|---|
| Recall | 0.0712 (Wilson 0.0491 to 0.1023) | 26/365 |
| Specificity | 0.9877 (Wilson 0.9780 to 0.9931) | 880/891 |
| Precision | 0.7027 (Wilson 0.5422 to 0.8251) | 26/37 |
| F1 | 0.1294 | n = 1,256 |
| Consistent rows only, recall | 0.0701 | 22/314 |
| Consistent rows only, precision | 0.6667 | 22/33 |
| Consistent rows only, F1 | 0.1268 | n = 1,193 |

Stratified by answer correctness: recall 0.0847 (10/118), specificity 0.9970 (656/658),
F1 0.1538, n = 776 on correct answers; recall 0.0648 (16/247), specificity 0.9614
(224/233), F1 0.1176, n = 480 on incorrect answers.

Section 3's regex, scored out-of-construct against this same Q2 label on all 1,304
annotated items, reads recall 0.0471 (18/382), precision 0.5294 (18/34), F1 0.0865. The
jury's Q2 recall and precision both sit above the regex's on its own, smaller, 1,256-item
pool. Q2 (does the CoT support the answer) and Q1 (does it mention a cue) are different
constructs, this is not a claim that the jury Q2 detector beats the regex at anything the
regex was built to do; it is reported for completeness, at its own EXPLORATORY status.

**Artifacts.** `experiments/external/jury_instrument_report.json` (full per-judge,
per-question breakdown, including strata not tabulated above);
`experiments/external/jury_vs_regex_same_items.json` (the same-item comparison);
`experiments/results/w6-external/gemma-3-27b-it/faithcot/no-cue/` (the mirrored run:
`score_summary.json`, `exit_code.txt`, `votes.jsonl` and its sha256, all committed).

Resume commands are in `~/Developer/bayes-cot-phase2/STATUS.md`.

---

## 5. The classifier-sensitivity range on our own traces

Column A factorizes (pre-registration section 2.1) as

    pi_silent = P(followed) x P(no mention | followed)

`P(followed)` comes from the frozen parser and does not move with the instrument. **Only
`P(no mention | followed)` does**, so that is the quantity whose range is reported, and the
range carries through to column A's numerator by the same constant factor.

Cell: Qwen3-8B, ARC-Challenge, stated-hint:strong. Job 826733, HF revision
`b968826d9c46dd6066d109eabc6255188de91218`, vLLM 0.28.0, batch-invariant, exit 0.

| Quantity | Value | Denominator |
|---|---|---|
| Entered | 1,500 | |
| Clean-correct | 1,396 | checked against `arms_summary.json` |
| P(followed) | 0.1784 (Wilson 0.1592 to 0.1993) | 249/1,396 |
| Follow stratum | 249 | meets the 20-item floor |
| **Frozen regex** P(no mention \| followed) | **0.6546** (Wilson 0.5936 to 0.7109) | 163/249 |
| Column A point under the regex | 0.1168 | |
| Jury Q1a P(no mention \| followed), EXPLORATORY | 0.6345 (Wilson 0.5731 to 0.6919) | 158/249 |
| Column A point under Q1a | 0.1132 | |
| Jury Q1b P(no mention \| followed), EXPLORATORY | 0.3534 (Wilson 0.2967 to 0.4146) | 88/249 |
| Column A point under Q1b | 0.0630 | |
| Jury Q1c P(no mention \| followed), EXPLORATORY | 0.3052 (Wilson 0.2513 to 0.3650) | 76/249 |
| Column A point under Q1c | 0.0545 | |

The recomputed regex agreed with the banked `acknowledged` flag on 249 of 249. Job 826921
reported on 2026-09-08; its wave-1 block is in
`experiments/external/jury_instrument_report.json` (`per_judge.gemma-3-27b-it.wave1`),
computed from the same mirrored `votes.jsonl` as section 4. Frozen-regex agreement with
each Q1 file's vote (not the same quantity as P(no mention | followed) above, since the
two instruments can each say "no" for a different item): Q1a 244/249 = 0.9799, Q1b
174/249 = 0.6988, Q1c 162/249 = 0.6506.

**The pre-registered `wave1_sensitivity.json` artifact has not been regenerated by this
lane.** Its script recomputes the regex end of the range from this cell's own
`transcripts.jsonl`, `run_meta.json` and `arms_summary.json`
(`~/bcf/results/qwen3-8b/arc_challenge/stated-hint/` on the cluster), a results directory
outside this lane's permitted read scope (limited to `~/bcf/results/w6-external/`
READ ONLY). The column-A points above are computed here instead from numbers already of
record in this document: the `p_followed` rate pinned two rows up (0.1784, 249/1,396,
unaffected by any instrument) times each instrument's own P(no mention | followed). Any
future run of `wave1_sensitivity.py` should reproduce them; if it does not, that script's
own recompute is the one to trust, not this table.

**A denominator error caught here, worth recording.** `transcripts.jsonl` pools two arms:
the 1,396 clean-correct rows from `arms_transcripts_*.json` and 19 more from
`specificity_transcripts_*.json` that have no hinted arm at all. Those 19 also satisfy
`clean_answer == answer_label`, so filtering on correctness alone yields 1,415 and puts the
wrong denominator under `P(followed)` (0.1760 instead of 0.1784). The arm is now selected
by `source_file` and the count is checked against `arms_summary.json`'s `n_clean_correct`,
refusing rather than reporting a denominator that is not the cell's own.

**The range now has four points, all from one judge.** P(no mention | followed) spans
0.3052 (jury Q1c) to 0.6546 (frozen regex), a spread of 34.94 points; column A spans
0.0545 to 0.1168, a spread of 6.23 points. Three of the four points are one EXPLORATORY
judge's, and that judge fails the frozen gate (section 4), so this is not yet the range a
second INSTRUMENT would give: it is the range one instrument's three prompt variants give,
next to the regex. `experiments/external/wave1_sensitivity.json` itself still says PENDING
(see the note above), so this spread lives in this document's prose only until that
artifact is regenerated.

---

## 6. What this lane establishes, and what it does not

**Establishes.**

1. The frozen regex's specificity on unhinted reasoning is 0.9729 (1,327/1,364), and its
   37 false positives are concentrated in two alternations that fire on ordinary prose.
2. Its false positives are enriched on post-hoc rationalization: 10/107 against 14/682
   with correctness held fixed, Fisher p = 0.00053.
3. FaithCoT cannot validate any cue-disclosure detector's recall, because none of its
   1,364 items carries a cue. That is a fact about the corpus and it constrains what the
   nearest competitor's data can be used for, by us or by anyone.
4. The release's two label fields disagree on 62 of 1,303 typed rows.
5. On our own wave-1 cell, the frozen regex puts P(no mention | followed) at 163/249.
6. One EXPLORATORY judge (gemma-3-27b-it) has scored FaithCoT and the wave-1 cell: its Q1
   specificity on FaithCoT is 0.9322/0.7654/0.4562 across the three Q1 files (1,168/1,253,
   959/1,253, 567/1,243), 4 to 52 points below the frozen regex on the same items, and its
   Q2 recall against the human label is 0.0712 (26/365).

**Does not establish.**

1. A selected jury number: one judge has a number now, EXPLORATORY throughout, and none is
   selectable, because that judge fails the frozen synthetic Q1 gate (ruling R9) and no Q1
   configuration has passed it for any judge.
2. Nothing about Q1 recall anywhere.
3. The sensitivity range's width from more than one instrument. Section 5's range now has
   four points, but three are one gate-failing judge's three prompt variants, not a second
   independent instrument, and the pre-registered `wave1_sensitivity.json` artifact has not
   itself been regenerated to carry them.
4. Nothing that transports automatically to our cued setting. FaithCoT is unhinted
   reasoning from four models we do not run, on four substrates, three of which are not
   ours. Specificity measured there is evidence about specificity there.

**The objection this inherits, stated rather than dodged.** The ICLR 2026 review of
FaithCoT-Bench records that the benchmark "does not measure true mechanistic faithfulness
but rather alignment with a human's cognitive model of what constitutes good reasoning".
Anchoring on its labels inherits that objection whole. It is the right objection, and the
answer is scope: these human labels define the text-level construct only, they say nothing
about column B, and column B is judge-free by construction.

---

*Data: FaithCoT-Bench / FINE-CoT, `github.com/se7esx/FaithCoT-BENCH` at revision
`6e3c004cbbde5bf47352df91e3ac399d2fb4593e`, used by written permission for evaluation only
and never redistributed. Cite: Shen et al., FaithCoT-Bench, arXiv:2510.04040.*
