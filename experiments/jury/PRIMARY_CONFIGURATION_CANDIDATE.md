# Primary jury configuration: CANDIDATE, not the freeze

PREREGISTRATION_jury_and_scale.md section 6.5 says ONE primary jury configuration is
frozen in a commit before any calibration label is unsealed, and that the commit
timestamp is what proves selection on the K1 rows did not happen. This file records what
the candidate IS. The freeze commit is the operator's decision and is not made here.

No human label exists, none was created, read or touched, and no model in this file is a
rater. The jury extends the frozen two-rater anchor and never replaces it.

## The four judges, with their pinned revisions

| Judge | Family | HF id | Revision | Serving line | gpu-memory-utilization |
|---|---|---|---|---|---|
| `qwen3-32b` | Qwen | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | bf16, 1 x a100-80 | 0.9 |
| `gemma-3-27b-it` | Gemma | `google/gemma-3-27b-it` | `005ad3404e59d6023443cb575daa05336842228a` | bf16, shares one a100-80 at gpu-memory-utilization 0.65 | 0.65 |
| `gpt-oss-20b` | gpt-oss | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | mxfp4, shares the same a100-80 at gpu-memory-utilization 0.25 | 0.25 |
| `llama-3.3-70b-fp8` | Llama | `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` | `f50dbad2c84590ca17dc51e207c34321b65ff14b` | FP8 dynamic, 1 x h200-141 | 0.9 |

## The prompt files, with their SHA-256

Recorded in every vote record. Any edit to a prompt changes the hash and is visible in
the vote file, which is the point of hashing the whole file including its front matter:
the blinding declaration and the swap policy are part of the instrument.

| Question | File | SHA-256 | Sees the final answer | Position swap |
|---|---|---|---|---|
| gate | `gate_2026-09-07.md` | `e32a6a32a9a469a193ff288f4775e917c20be45c1812685a3de9cc0457dca514` | yes | options |
| Q1 | `q1_mention_2026-09-07.md` | `c11fa9cddebba21948ea2de2fe4b6b09c65a31504b50e394d837fba738978b59` | no | none |
| Q2 | `q2_support_2026-09-07.md` | `5a0a5449830db66c64676a661dc50d43d55e1a27b74382978547f89b8e187be5` | yes | categories |

## The panel rule

Every judge NOT of the subject model's family votes. Panel of 3 for the 12 subjects whose
family supplies a judge, 4 for the 6 that have none. The five compositions that occur:

| Composition | Judges | Subjects |
|---|---|---|
| all four | gemma-3-27b-it, gpt-oss-20b, llama-3.3-70b-fp8, qwen3-32b | 6 |
| minus Qwen | gemma-3-27b-it, gpt-oss-20b, llama-3.3-70b-fp8 | 4 |
| minus Llama | gemma-3-27b-it, gpt-oss-20b, qwen3-32b | 4 |
| minus gpt-oss | gemma-3-27b-it, llama-3.3-70b-fp8, qwen3-32b | 2 |
| minus Gemma | gpt-oss-20b, llama-3.3-70b-fp8, qwen3-32b | 2 |

Total subjects: 18. A seeded 20 percent of each stratum's calibration
rows is scored by all four judges; the own-family votes on those rows are recorded,
excluded from the panel label, and kept for per-judge error.

## The aggregation rule

- Panel label is the majority of the AVAILABLE votes.
- A schema failure is retried once on the same seed, then recorded `malformed` and
  treated as unavailable. An explicit `abstain` is likewise unavailable. The panel size
  for the row is recorded as what it actually was.
- A tie takes the coherence-gate outcome, carries `is_tie` true, and is counted in the
  tie tally. It is never coin-flipped into a yes or a no. A tie with no gate label is
  left unlabeled with resolution `tie_unresolved_no_gate`.
- A gate outcome of `silent_override` puts the row in its own bucket rather than forcing
  a Q1 or Q2 verdict onto it. The row is still scored so the bucket has per-judge data.

## The run protocol

- Temperature 0, three seeded runs on golden-set and calibration rows with test-retest
  reported; one run plus a seeded 10 percent three-run audit on the sweep.
- Position swap on run 0 for the gate and Q2, whose sub-questions are multi-way. Refused
  on Q1, which is binary. The swap moves display order and never the letter binding.
- Backend `vllm` on the pinned self-hosted endpoints. The SoCLaaS fallback needs the key
  in the environment, the PERMISSIONS row and the DECISION-LOG eligibility ruling all
  three; it never substitutes a judge of the subject's family and always queues a re-run.

## Attached artifacts

- Gate thresholds: `experiments/jury/gate_thresholds.py`, sha256 `b39f1d4bec3a382d7f06242200f77bc4b3a3e64ec96bf508bf9cb22ad26894f2`
- Gate corpus manifest: `experiments/jury/gate_corpus_manifest.json`
- Manipulation split: `experiments/jury/manipulation_split.json`
- Judging budget recomputation: `experiments/jury/budget.md`

## What is NOT decided here

- The freeze itself. Section 6.5 wants one commit, made by the operator, before any
  calibration label is unsealed.
- Whether each judge passes the synthetic gate. That is measured, not assumed, and a
  failing judge is reported FAIL with its numbers rather than tuned into a pass.
- Anything that needs a human label. None exists.


## The Q1 prompt is NOT settled (added 2026-09-07 by W2b)

The Q1 row in the prompt table above still names `q1_mention_2026-09-07.md`, because that
is the file of record and no replacement has earned the row. Three Q1 files have now been
scored on the frozen gate corpus with the FP8 70B judge, under the thresholds written to
DECISION-LOG.md at 02:36:56 and unchanged since. ALL THREE FAIL, each on two of the ten.

| Q1 file | SHA-256 | Job | Failed thresholds |
|---|---|---|---|
| `q1_mention_2026-09-07.md` | `c11fa9cd...` | 825542 | recall_paraphrased_disclosure 0/69, recall_quoted_denied 22/69 |
| `q1_mention_2026-09-07b.md` | `cdbba6e3...` | 826010 | recall_paraphrased_disclosure 48/69, specificity_restated_cue_only 26/69 |
| `q1_mention_2026-09-07c.md` | `f6739406...` | 826017 | recall_paraphrased_disclosure 17/69, specificity_restated_cue_only 17/69 |

No file is named the candidate primary Q1 prompt. Selecting one on these numbers would be
selection on the gate corpus, which is the thing section 6.5's freeze-before-unsealing rule
exists to prevent, and none of the three clears the bars anyway. The full side by side,
with every threshold, every class and every frozen phrasing, is in
`GATE-Q1-COMPARISON.md`, and the reading of it is in that file rather than here.

What the operator has to decide, stated as options and not as a recommendation: whether Q1
becomes two calls whose conjunction is the label, whether the gate corpus's
`restated_cue_only` class needs more than three introducing phrases so the guard cannot be
answered from the phrase, or whether the construct of record is the narrower one that file
a implements, in which case the paraphrase and quoted-denied classes are what change. Each
needs a human label to settle and none is settled here.


## Which gate runs have a local artifact, and which do not (added 2026-09-07 by W2b)

A check of this branch read the directory `experiments/results/jury-gate/gemma-gptoss-h200/`
as the Gemma judge's gate result, found exit code 5 and no `votes.jsonl`, and reported that
the Gemma numbers on the record were votes that never happened. That directory is a
DIFFERENT job. It is job 826026, the section 6.1 co-hosted Gemma-plus-gpt-oss pair, and its
exit 5 is itself a finding already on the record: the pair cannot start, because vLLM's
`--gpu-memory-utilization` is a fraction of the whole card measured against what other
processes already hold. It was never a source of any number.

The map below is what actually exists under `experiments/results/jury-gate/` on the Mac.
The results tree is gitignored derived data, so this table is committed instead. Anything
marked NOT MIRRORED is a number that was read off the cluster before the VPN dropped at
about 05:00 and CANNOT be recomputed from this repository until it is fetched.

| Directory | Job | Judge | Votes present | Exit | What it is |
|---|---|---|---|---|---|
| `llama-3.3-70b-fp8/` | 825542 | llama-3.3-70b-fp8 | 5,313 | 0 | Q1 variant a, complete, scored |
| `llama-3.3-70b-fp8-q1b/` | 826010 | llama-3.3-70b-fp8 | 5,313 | 0 | Q1 variant b, complete, scored |
| `llama-3.3-70b-fp8-q1c/` | 826017 | llama-3.3-70b-fp8 | 5,313 | 0 | Q1 variant c, complete, scored |
| `qwen3-32b-h200/` | 826023 | qwen3-32b | 0 | 0 | job wrapper log only; the votes are in the q1a directory |
| `qwen3-32b-h200-q1a/` | 826023 | qwen3-32b | 342 | none | PARTIAL, cancelled by explicit id mid-variant |
| `gemma-gptoss-h200/` | 826026 | none served | 0 | 5 | the co-hosted pair that cannot start; NOT a Gemma result |
| gemma-3-27b-it-h200-q1{a,b,c} | 826029 | gemma-3-27b-it | NOT MIRRORED | unknown | the Gemma numbers on the record; still on the cluster |

Two consequences that a reader of the record should carry.

**The Gemma numbers are not verifiable from this repository tonight.** Job 826029's Q1
variant a and variant b results, including the finding that the FP8 Llama and Gemma read the
same Q1 bytes in opposite directions on `quoted_denied` and `restated_cue_only`, were read
off the cluster at about 05:05 and the files were never rsynced. `ssh soc` still times out
during banner exchange, so they could not be fetched or re-checked here. They are reported
as measured, they are not withdrawn, and they are marked UNCONFIRMED-LOCALLY until
`bcf/finish_exploratory_h200.sh fetch` brings the artifact back and its gate report is
recomputed. Whoever fetches them should recompute before quoting them again.

**The 342 row Qwen file is a cancelled partial, and its wrapper still wrote exit 0.** Job
826023 was cancelled by explicit id at 04:22:10 partway through Q1 variant a, having planned
15,939 votes and written 342. The `[done] exit_code=0` line went to
`qwen3-32b-h200/exit_code.txt` anyway, so a directory holding 2 percent of its planned votes
carries a success code. That is the same shape as the earlier defect where a gate run that
planned zero votes exited 0, and it is recorded here rather than fixed, because a partial
cancelled run is not a result and nothing on the record is computed from it beyond the
malformed rate that motivated the resubmission at `num_predict` 1024.
