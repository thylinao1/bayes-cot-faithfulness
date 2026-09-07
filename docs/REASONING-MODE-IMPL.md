# Ruling R12 implemented: `BCF_REASONING_MODE`, the record fields, and the two-path gate

Reasoning-mode lane, 2026-09-07, branch `feat/reasoning-mode`. This document says what was
built, what every record now carries, how the gate runs, and what is NOT done. It makes no
ruling; R12 (`RULINGS-2026-09-07.md`) is the ruling, and nothing here edits a frozen file
(`tests/test_frozen_guard.py` passes unchanged, and the answer-extraction regexes are
imported and applied unedited).

## 1. What R12 asked for, and what each piece of it became

R12(1) defines the reasoning switch, for a roster row whose chat template opens a block
and documents no switch, as closing that block, and applies element 9.4's "both settings":
configuration A (reasoning off) is the cell of record, configuration B (reasoning on) is
the exploratory additive arm. That became one runner option with three values.

| Value | Request path | `num_predict` full | `num_predict` forced | Answer read from | Status |
|---|---|---|---|---|---|
| `default` | `/chat/completions` | 320 | 24 | the whole completion, frozen parser | today's behaviour, byte for byte |
| `off` | `/tokenize` + `/detokenize` + `/completions`, block CLOSED | 320 | 24 | the whole completion, frozen parser | the CELL OF RECORD (R12(1)), after the gate |
| `on` | `/chat/completions` | 4096 | 24 | the text AFTER the closing think tag | EXPLORATORY, reported beside `off`, never pooled |

Where it lives:

- `src/bayes_cot_faithfulness/reasoning_mode.py` - the mode vocabulary, the two
  block-closing branches, the post-think extractor, the record fields and the assertion.
- `experiments/reasoning_client.py` - `ReasoningModeClient`, the `off` client: renders the
  prompt through the model's own template, closes the block, generates through
  `/completions`, and counts per call whether a completion reopened a block.
- `experiments/08_additive_arms.py` - `--reasoning-mode`, `RunCtx.reasoning_mode` and
  `RunCtx.extract_full`, the four fields on every record and summary, the pre-checkpoint
  assertion, the `run_meta.json` merge.
- `experiments/arms_resume.py` - `reasoning_mode` in the resume fingerprint.
- `bcf/serve_and_run.sbatch` - `BCF_REASONING_MODE`, validated before the weights
  download (exit 14), the 4096 override for `on`, and the mode in `run_meta.json`.
- `experiments/gate_twopath.py` and `bcf/gate_twopath.sbatch` - the R12(2) gate.
- `tests/test_reasoning_mode.py` - 28 offline tests against a fake server, with the
  revert proof for the field assertion in its docstring.

**Only `off` needed a client of its own.** `on` differs from a default cell in the
full-generation budget (set by the runner) and in the answer extractor (selected by
`RunCtx.extract_full`), and in nothing that reaches the socket, so `_gate_client` hands
`default` and `on` the same `OpenAIClient` the sweep has always used. That is what keeps
the byte-identical claim for `default` cheap to make and cheap to check.

**How the block is closed**, following `experiments/audit/serving_test.py`, the
implementation the R12 serving numbers were measured with, branch for branch:

- the template already ended the prompt with an OPEN block (Olmo-3-7B-Think ends
  `<|im_start|>assistant\n<think>`, R1-Distill-Llama-8B ends `<|Assistant|><think>\n`):
  append `\n\n</think>\n\n`, branch `closed_the_block_the_template_opened`;
- the template opened nothing (R1-0528 ends `<|Assistant|>`, Phi-4-reasoning ends
  `<|im_start|>assistant<|im_sep|>`): append the whole empty block
  `<think>\n\n</think>\n\n`, branch `inserted_a_whole_empty_block`, which is DeepSeek's
  own documented way of making an R1 model skip thinking.

**Why the path moves and a prefill will not do.** Two of the four affected templates
rewrite an assistant message with `content.split('</think>')[-1]` before rendering it, so
an assistant-turn prefill of `<think>\n\n</think>\n\n` arrives as `\n\n` and the empty
block never reaches the prompt. The run would look like a measurement of
thinking-suppressed decoding and would be nothing of the kind.

**Forced continuations under `off` are closed too.** `docs/WAVE1-AUDIT.md` found that
Phi-4-reasoning's 24-token forced continuations reopen a reasoning block, which is why
that cell has `direct` 0 scorable of 1,197. Under `off` the continuation prompt is
rendered and closed exactly like a full generation, and whether the model reopened one
anyway is counted per continuation and reported in the summary
(`reasoning_detail.continuations`: `n`, `n_reopened`, the rate, and up to ten example
tails). That count is the evidence for R12(3)'s "an A variant that also closes any
reopened block in the forced continuations"; it does not by itself decide Phi-4.

**The `on` extractor is a wrapper, not a new parser.** `parse_answer_after_think` takes
the text after the LAST closing tag and hands it to the frozen `parse_answer` with the
frozen regexes unchanged, so every letter it returns is a letter the frozen parser returns
on that suffix. It returns unparseable when there is no closing tag, and deliberately does
not fall back to frozen-parsing the whole text: under `on` the OPENING tag lives in the
rendered prompt rather than in the completion, so a chain truncated at 4,096 carries no
marker of its own and a fallback would read a letter out of the middle of the reasoning.
The FORCED continuation is always read with the frozen parser: 24 tokens of answer line
have no reasoning block, and requiring a closing tag there would empty every mediator arm.

## 2. The record fields

Four fields on every generation record, on the cell summary, and in `run_meta.json`:

| Field | `default` | `off` | `on` |
|---|---|---|---|
| `reasoning_mode` | `"default"` | `"off"` | `"on"` |
| `reasoning_path` | `"chat"` | `"completions"` | `"chat"` |
| `reasoning_block_closed` | `null` | `true` | `null` |
| `num_predict_full` | 320 | 320 | 4096 |

`reasoning_block_closed` is `true` under `off` by construction, not by hope:
`close_reasoning_block` returns a prompt ending in a closed block on both branches, and a
test pins that. It is `null`, not `false`, under the other two, because nothing was
attempted there and `false` would read as "we tried and it did not hold".

The summary and `run_meta.json` additionally carry `reasoning_detail`, which is MEASURED
rather than constructed and is `null` outside `off`: the close branch that fired, the
rendered prompt tail (200 characters, one per cell, which is the "rendered prompt tail for
one item" R12's record schema asks for), the per-call-kind path map, and the full
generation and continuation reopen counts.

The runner asserts all four before writing a checkpoint, the same discipline and the same
refusal as `outcome_scale` in section 9.6: `assert_records_reasoning_mode` raises rather
than warns, and it runs before the transcript file is written, so a stop banks nothing
mislabelled. The revert proof is in the test docstring: deleting the assertion turns
`test_writer_refuses_a_record_that_lost_the_reasoning_fields` red (1 failed, 27 passed).

`reasoning_mode` also joins the resume fingerprint, as `None` on a default cell so every
pre-existing checkpoint still loads, and as the mode otherwise, so a leg resumed under a
different configuration refuses instead of merging two request paths into one cell.

## 3. The two-path equivalence gate (R12(2))

`experiments/gate_twopath.py`, submitted once by `bcf/gate_twopath.sbatch` (job name
`bcf-gate-twopath`, `a100-40`, `--mem 64G`, `--exclude xgpj0`, `VLLM_BATCH_INVARIANT=1`,
`Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`, the free-port and
exit-guard helpers from `bcf/env.sh`). 30 ARC items, each sent both ways with
reasoning_mode DEFAULT semantics, so the two paths should render the same prompt:

| Reported | PASS condition |
|---|---|
| identical completions | 30/30, byte for byte |
| max abs letter-logprob difference | exactly 0.0, with no letter missing on either path |
| rendered prompt round trip | 30/30: `/tokenize(messages)` -> `/detokenize` -> `/tokenize(text)` returns the same ids |

The round trip is in the verdict because a prompt whose text does not round-trip cannot be
compared with anything: that is the R1-0528-Qwen3-8B defect, whose rendered prompt came
back with every space missing (`docs/REASONING-MODE-TEST.md` section 3), and a difference
measured over it would mean nothing either way. The gate writes `gate_twopath.json` with
the verdict and the three checks and exits 0 on PASS, 10 otherwise; a gate that cannot
measure at all writes REFUSE with the error rather than passing by omission. Results are
mirrored to `experiments/results/gate-twopath/`.

The gate runs at concurrency 1 by default: it asks whether the two request paths agree,
and running it under load would fold the batching question into the answer. R1's own
preflight is what measures batching.

## 4. What is NOT done

- **The Phi-4-reasoning configuration B test (job 827214) is still not collected.** R12(3)
  leaves that row neither held nor of record until B is measured and until an A variant
  that closes reopened blocks in the forced continuations is run. The code for the second
  half now exists (`off` closes continuation prompts and counts reopens); the MEASUREMENT
  does not. Resume instructions are in `docs/REASONING-MODE-TEST.md` section 6.
- **R1-0528-Qwen3-8B stays HELD on its tokenizer defect** (R12(4)). Nothing here serves it
  with a different tokenizer, and the gate's round-trip check would refuse it rather than
  pass it quietly. A tokenizer override is a serving-line question to be tested before any
  cell of that row.
- **Thirteen of the eighteen roster templates were never opened.** Rows 3, 5, 10, 13 and 18
  keep "verify the template before the cell" from R12(6); this lane read no new templates.
- **The gate has not been RUN.** The cluster link was down for all three attempts in this
  lane's window (ssh `soc`: connection timed out during banner exchange). Nothing was
  submitted. The submit block is in section 5 and in `DECISION-LOG.md`.
- **`forced_answer_logprobs` stays on the chat path under `off`.** The anchor arm's
  letter-logprob read uses `continue_final_message` with an assistant turn, and moving it
  to a rendered `/completions` prompt-logprob call could not be tested against a live
  server in this lane. It is a LOGIT-level read stored in its own block with its own
  intervention level and outcome scale and never merged into the record's own, so it is
  already fenced from the text-level outcome; the split is recorded per cell in
  `reasoning_detail.paths` rather than left to be inferred. The sampling draws
  (element 9.2) WERE moved, because they are text-level generations that feed the
  uncertain stratum and drawing them on a different prompt than the cell's own
  generations would put that stratum on a different object.
- **The determinism preflight and the forced-logprob check still run on the chat path**
  under every mode. They measure the server, not the cell's extraction, and R12(2) is what
  certifies the rendered path renders the same prompt. `run_meta.json` records this as
  `reasoning_preflight_path: "chat"` rather than leaving it implied.
- **No wave manifest was edited.** The hold list and the per-row mode assignment are the
  amendment lane's and the orchestrator's, per this lane's brief.
- **No intermediate `num_predict` was added.** R12(7) adopts none, and 4096 is not a
  measured bound: on Olmo-3-7B-Think 6 of 30 generations still hit it.

## 5. Submit block for the gate, when the link is back

    ssh -o ConnectTimeout=20 soc 'squeue -u $USER -h -o "%P" | grep -c gpu-long'
    # must be BELOW 8 before submitting

    rsync -avP --delete ~/Developer/bcf-rmode/ soc:~/bcf/repo-rmode/ \
      --exclude .git --exclude .venv --exclude __pycache__

    ssh soc 'sbatch --export=ALL,BCF_REPO=$HOME/bcf/repo-rmode,BCF_ENV_SH=$HOME/bcf/repo-rmode/bcf/env.sh $HOME/bcf/repo-rmode/bcf/gate_twopath.sbatch'

    # once it finishes (job name bcf-gate-twopath):
    ssh soc 'cat ~/bcf/results/gate-twopath/exit_code.txt'
    scp soc:'~/bcf/results/gate-twopath/{gate_twopath.json,run_meta.json,exit_code.txt}' \
      ~/Developer/bcf-rmode/experiments/results/gate-twopath/

Submit it ONCE. Exit 0 is PASS and is what lets a `reasoning_mode` `off` cell be a cell of
record; exit 10 is REFUSE, and `gate_twopath.json` names which of the three checks failed.
