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

