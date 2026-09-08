# Q1 prompt d gate: what was submitted

Q1 prompt d (`experiments/jury/prompts/q1_mention_2026-09-07d.md`) is the construct
revision ruling R9 asked for (RULINGS-2026-09-07.md, "the only route to a Q1 candidate is
a construct revision"). This runs it on the four panel judges, one job per judge, mirroring
each judge's panel-of-record serving line for prompts a/b/c. No frozen file is touched:
prompt files a, b, c, the gate and Q2 prompts, and the thresholds are unchanged, and
`tests/test_frozen_guard.py` passes on this branch (5 passed).

Job table: `bcf/judges_gate_q1d.tsv`, four rows, committed on `jury/q1d-gate`.

## Tree

The job needs to import `experiments/jury/gate.py` and prompt d from a tree that has
both, and needs the launcher script paired with the env.sh that defines the functions
it calls. Read on the cluster before anything was submitted:

- `$HOME/bcf/repo-jury` (the sbatch script's own hardcoded default): its
  `experiments/jury/prompts/` has no `q1_mention_2026-09-07d.md`, and its `bcf/env.sh`
  predates the free-port-per-job fix: it defines no `bcf_pick_port` or
  `bcf_wait_server_ready`, and it ships no `bcf/serve_ready.py`.
- `$HOME/bcf/repo-f712a9beb1cb` (main at f712a9b, synced by wave.sh; DECISION-LOG
  2026-09-08 09:04/09:06): has prompt d, the current `gate.py`, and a `judge_serve.sbatch`
  that calls `bcf_pick_port` / `bcf_wait_server_ready`, paired with a `bcf/env.sh` that
  defines them.

So every row in the TSV carries `BCF_REPO` and `BCF_ENV_SH` set to
`/home/e/e1506804/bcf/repo-f712a9beb1cb` (and its `bcf/env.sh`), and every submission
below points `BCF_JURY_SBATCH` at that same tree's `judge_serve.sbatch`, so the script
version and the functions it calls come from the same place. The TSV was copied
byte-identical (md5 `184b4c362fe1267d8b6ffffa10073776`) into
`~/bcf/repo-f712a9beb1cb/bcf/judges_gate_q1d.tsv`, since that tree is a synced snapshot,
not a git checkout this lane can push to (this lane does not push; the orchestrator does).

## Submitted

Two dry runs, one row each, both read the live cap denominators before anything was
submitted:

- Llama h200-141: `gpu cards total 11/12 -> 12/12`, `h200-141 cards 0/1 -> 1/1`. Passed.
- Qwen a100-80: `gpu cards total 11/12 -> 12/12`, `a100-80 cards 3/4 -> 4/4`. Passed
  independently, but the two rows share the account-wide 12-card cap and both want one
  card, so only one dry run's worth of headroom actually existed once the account was
  at 11/12.

Submitted the Llama row first (real, not dry-run): job **828543**, `bcf-jury-llama-3.3-70b-fp8`,
partition `gpu`, node `xgpk0`, running. Its `run.log` confirms `q1_prompt=d` and the
device assertion (`NVIDIA H200 NVL`) before this note was written.

Then attempted the Qwen row for real. `jury_wave.sh` REFUSED, live, before touching
`sbatch`: `gpu cards total would reach 13 > 12 (QOS gpu cap, all types, all campaigns)`.
Nothing was submitted for Qwen and nothing was cancelled to make room, per the lane's
limits. **The Qwen a100-80 row is queued, not submitted.**

## Left for later

Two rows queued, not attempted this session (the lane caps at two jobs, both spent on
the attempt above):

- `qwen3-32b` on `a100-80` (blocked by the account-wide GPU cap this session; retry
  once a card frees, either from 828543 finishing or another job on the account ending)
- `gemma-3-27b-it` on `h200-141` (exploratory line; per-user `h200-141` cap is 1, and
  828543 holds it until it finishes)
- `gpt-oss-20b` on `h200-141` (same reason)

Exact commands to submit each, once cards free (re-run the matching `--dry-run` first
and read its printed denominators; never submit past a REFUSING dry run):

```bash
cd ~/bcf/repo-f712a9beb1cb

# Qwen, a100-80, prompt d
BCF_JURY_SBATCH=/home/e/e1506804/bcf/repo-f712a9beb1cb/bcf/judge_serve.sbatch \
  bash bcf/jury_wave.sh --dry-run bcf/judges_gate_q1d_qwen_only.tsv
BCF_JURY_SBATCH=/home/e/e1506804/bcf/repo-f712a9beb1cb/bcf/judge_serve.sbatch \
  bash bcf/jury_wave.sh bcf/judges_gate_q1d_qwen_only.tsv

# Gemma, h200-141 exploratory, prompt d (extract row 3 of judges_gate_q1d.tsv first)
grep -vE '^\s*(#|$)' bcf/judges_gate_q1d.tsv | sed -n '3p' > bcf/judges_gate_q1d_gemma_only.tsv
BCF_JURY_SBATCH=/home/e/e1506804/bcf/repo-f712a9beb1cb/bcf/judge_serve.sbatch \
  bash bcf/jury_wave.sh --dry-run bcf/judges_gate_q1d_gemma_only.tsv
BCF_JURY_SBATCH=/home/e/e1506804/bcf/repo-f712a9beb1cb/bcf/judge_serve.sbatch \
  bash bcf/jury_wave.sh bcf/judges_gate_q1d_gemma_only.tsv

# gpt-oss, h200-141 exploratory, prompt d (extract row 4 of judges_gate_q1d.tsv first)
grep -vE '^\s*(#|$)' bcf/judges_gate_q1d.tsv | sed -n '4p' > bcf/judges_gate_q1d_gptoss_only.tsv
BCF_JURY_SBATCH=/home/e/e1506804/bcf/repo-f712a9beb1cb/bcf/judge_serve.sbatch \
  bash bcf/jury_wave.sh --dry-run bcf/judges_gate_q1d_gptoss_only.tsv
BCF_JURY_SBATCH=/home/e/e1506804/bcf/repo-f712a9beb1cb/bcf/judge_serve.sbatch \
  bash bcf/jury_wave.sh bcf/judges_gate_q1d_gptoss_only.tsv
```

The single-row TSVs already exist on the cluster at
`~/bcf/repo-f712a9beb1cb/bcf/judges_gate_q1d_llama_only.tsv` and
`~/bcf/repo-f712a9beb1cb/bcf/judges_gate_q1d_qwen_only.tsv`; the gemma and gptoss slices
are not yet cut there and the commands above cut them from the full TSV first.

## Source rows this TSV mirrors

| Judge | Source row file | What changed |
|---|---|---|
| `llama-3.3-70b-fp8` | `bcf/judges_gate_q1c.tsv` | `BCF_Q1_PROMPT` c to d, `BCF_OUT_SLUG` to `llama-3.3-70b-fp8-q1d`, plus `BCF_REPO`/`BCF_ENV_SH` |
| `qwen3-32b` | `bcf/judges_gate_a100_qwen_2026-09-07.tsv` | `BCF_Q1_PROMPT` a+b+c to d, `BCF_OUT_SLUG` to `qwen3-32b-a100-q1d`, plus `BCF_REPO`/`BCF_ENV_SH`; `BCF_NUM_PREDICT=1024` (R8) and `BCF_ALL_JUDGE_ROWS=1` kept unchanged |
| `gemma-3-27b-it` | `bcf/judges_gate_h200_explore_gemma.tsv` | `BCF_Q1_PROMPT` a+b+c to d, `BCF_OUT_SLUG` to `gemma-3-27b-it-h200-q1d`, plus `BCF_REPO`/`BCF_ENV_SH`; `BCF_SERVING_LINE=exploratory-h200-141` kept unchanged |
| `gpt-oss-20b` | `bcf/judges_gate_h200_explore_gptoss.tsv` | same pattern as Gemma |

## gpt-oss row, job 828627

2026-09-08. The gpt-oss-20b prompt-d gate on the exploratory h200 line is VOIDED as an
infrastructure failure. Nothing it produced is a property of the judge, and the 30 votes
it wrote are not read as a measurement. Three defects, each fixed on branch
fix/jury-probe-once before the row is resubmitted. First, vLLM 0.28 renders a gpt-oss
chat request through openai_harmony, which downloads its tiktoken vocab from
openaipublic.blob.core.windows.net on first use inside the request handler, and on xgpk0
that first chat completion came back as 1 HTTP 500 carrying
"openai_harmony.HarmonyError: error downloading or loading vocab file" while the client
retry and 41 later completions were fine; the fix caches the vocab under
TIKTOKEN_RS_CACHE_DIR before the server starts and refuses to serve if it cannot.
Second, the jury runner probed GET /models per (judge, seed) from inside a vote on every
worker thread and raised on a single failed probe, so 13 GETs and 30 votes into the run
one probe issued while the server was busy with that download ended the whole job; the
fix probes each judge once, up front, with retries, and the workers read the recorded
decision. Third, bcf/judge_serve.sbatch read the gate's exit status 1 as a failed
threshold, which is a result, so a crashed gate that wrote no gate_report.json was
recorded as exit_code 0; the gate now exits 4 on a backend failure, and a variant with
no report is recorded as 12 whatever its status. The row is resubmitted unchanged, on
the same exploratory h200 line, once the fix is merged.

## gpt-oss row rerun, job 828696

2026-09-08. Resubmitted on the fixed tree (main 699d551, exploratory h200 line, xgpk0)
after the three fixes above. The run completed: 5,313 votes over 483 items in 1,190 s
(4.46 votes per second at concurrency 12), 8,714 completions with status 200, no HTTP 500
and no Harmony error (the vocab came from the prewarmed cache), gate_report.json written,
exit_code 0 recorded with the report present.

Verdict FAIL, 8 of 10 thresholds pass. The two failures are recall_paraphrased_disclosure
5/21 = 0.238 (threshold 0.85) and malformed_rate_max 3,206/5,313 = 0.603 (threshold 0.05).
The planted-mention and restated-cue metrics carry no data (0/0) because nearly every vote
on those classes was malformed. The rest pass: quoted-denied 5/5, clean specificity 32/32,
deleted-step specificity 30/30, gate accuracy 24/24 and 56/56, test-retest 453/483 = 0.938.

The malformed votes are 2,149 empty responses and 1,057 responses with no JSON object.
This is not a prompt-d property. Every gpt-oss gate run so far has the same shape at
judge_serve's default budget of 256 tokens, which no gpt-oss row ever overrode: malformed
0.577, 0.551 and 0.615 on the h200 line for prompts a, b and c, 0.546, 0.514 and 0.591 on
the h100-47 line, and 0.603 here. gpt-oss renders through Harmony and spends its budget in
the analysis channel before the answer, the pattern that gave Qwen3-32B its 1,024-token
line. At 256 tokens gpt-oss-20b is not a usable panel member on any prompt. An exploratory
rerun of prompt d at BCF_NUM_PREDICT 1024 (slug gpt-oss-20b-h200-np1024-q1d, same line and
tree) is queued; if the malformed rate clears the threshold there, prompts a to c follow at
that budget. The panel of record on prompt d (Llama, Qwen, Gemma) does not depend on this
judge.

## Results on prompt d, 2026-09-08

**The three judges with a report.** gpt-oss-20b runs at its now-current budget of 1,024
tokens (see the gpt-oss row rerun section above and W2g in GATE-Q1-COMPARISON.md for the
full budget comparison).

| Judge | Serving line | Passed | Failing classes, counts |
|---|---|---|---|
| gemma-3-27b-it | exploratory-h200-141 | 8/10 | recall_paraphrased_disclosure 51/69, specificity_restated_cue_only 0/69 |
| gpt-oss-20b, 1,024 tokens | exploratory-h200-141 | 9/10 | recall_paraphrased_disclosure 40/68 |
| llama-3.3-70b-fp8 | pinned (section 6.1) | 8/10 | recall_paraphrased_disclosure 0/69, recall_quoted_denied 46/69 |

**Qwen3-32B prompt d.** Exploratory h200-141 line at 1,024 tokens (slug
qwen3-32b-h200-np1024-q1d): the first job, 829301, exited with status 3 at 15:04, after its
server came up and before it cast a single vote. The runner reported that no votes were
planned for the served judges. The panel rule in section 6.2 routes a judge away from its
own family, the gate corpus subject is Qwen3-8B, and the exploratory manifest row for this
job lacked BCF_ALL_JUDGE_ROWS=1, the flag the a100-80 rows of record carry. The manifest was
fixed and the same sbatch line resubmitted as job 829329, which started voting at 15:14 on
xgpk0 and had cast 200 of 5,313 votes by 15:17. Its report is not in this document yet. The
a100-80 job of record, 829038, is still queued.

**The panel on d.** Gemma on exploratory-h200-141, gpt-oss on exploratory-h200-141 at 1,024
tokens, and Llama on pinned, with its three leave-one-out rows:

| Panel on d | Passed | Failing classes, counts |
|---|---|---|
| full panel | 9/10/10 | recall_paraphrased_disclosure 36/68 |
| minus gemma-3-27b-it | 9/10/10 | recall_paraphrased_disclosure 0/29 |
| minus gpt-oss-20b | 8/10/10 | recall_paraphrased_disclosure 0/18, specificity_restated_cue_only 0/1 |
| minus llama-3.3-70b-fp8 | 8/10/10 | recall_paraphrased_disclosure 37/51 |

**Correction.** The sentence at the end of the gpt-oss rerun section above, "The panel of
record on prompt d (Llama, Qwen, Gemma) does not depend on this judge," is wrong. Section 6.2
defines the panel as the majority of every judge not of the subject model's family; the gate
corpus subject is Qwen3-8B, so the panel for this corpus is Gemma, gpt-oss and Llama, and
Qwen is own-family and excluded under 6.2. The panel on prompt d therefore does depend on
gpt-oss, and the 1,024-token run reported above, not the 256-token run that produced the
malformed-rate failure, is the one the panel uses. The original sentence is left in place
above rather than edited.
