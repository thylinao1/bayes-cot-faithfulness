# Element 9.4, the reasoning-mode toggle: what wave 1 measured, and what two configurations give

Track V, the audit lane, 2026-09-07. **This document makes no ruling.** It states the
diagnosis with denominators, reports a serving test of the two configurations a ruling would
choose between, and sets out what each choice would mean. Element 9.4 is the operator's.

## 1. What the pre-registration says, and what the cells actually did

`experiments/PREREGISTRATION_jury_and_scale.md` section 9.4 (element 8):

> Where a model documents a reasoning-mode switch, the sweep runs both settings as an additive
> arm and records the setting in every record (for Qwen3 this is
> `chat_template_kwargs {"enable_thinking": ...}`). Where a model documents no switch, the arm is
> absent and the cell records that it is absent, rather than silently reporting one mode.

Section 16 (element 15) pins the decoding constants: temperature 0.0, one sample per call,
`num_predict` 320 for full generations and 24 for forced-answer continuations.

All eight wave-1 cells ran with `chat_template_kwargs {"enable_thinking": false}` and
`num_predict` 320. Every cell's `run_meta.json` in `docs/wave1-artifacts/<slug>/` records both.

**The gap 9.4 leaves open, and which wave 1 fell into.** `enable_thinking` is a Qwen3
convention. It is a variable in Qwen3's chat template and in no other template on the roster's
small models. Passing it to a template that does not read it is not an error: the server
renders the template, the unused variable is discarded, and the run records a setting that
never reached the model. Section 9.4 says a model with no documented switch records the arm as
absent; nothing in the pipeline makes that recording happen, so the four affected cells recorded
`{"enable_thinking": false}` as though it had taken effect.

Read from each model's own chat template at its pinned revision (fetched from the Hugging Face
API on 2026-09-07):

| model | `enable_thinking` in the template | what the template appends to the prompt | consequence at `num_predict` 320 |
|---|---|---|---|
| `Qwen/Qwen3-8B` | yes, honoured | nothing when thinking is off | no reasoning block; 320 tokens is a whole answer |
| `allenai/Olmo-3-7B-Think` | no | `<\|im_start\|>assistant\n<think>` on every generation prompt | the completion starts INSIDE an open block it did not write |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | no | `<｜Assistant｜><think>\n` | same |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | no | `<｜Assistant｜>` only | the model opens `<think>` itself |
| `microsoft/Phi-4-reasoning` | no | a hardcoded system prompt demanding a `<think> … </think>` Thought section, and the message loop handles only `user` and `assistant`, so a user-supplied system message is dropped | the model writes its own block |

This is why the returned text on Olmo and R1-Distill carries no opening `<think>`: the tag is in
the prompt, not the completion. The `close_think_tag` column of the diagnosis table below is the
one that matters, and it is near zero.

## 2. The diagnosis, with denominators

Method: `experiments/audit/diagnose_unparseable.py`. The FROZEN parser
(`src/bayes_cot_faithfulness/interventions.py:parse_answer`) decides what counts as
unparseable; a separate lenient extractor, defined in the audit tooling only and never added to
the parser, decides whether a COMMITTED answer was nevertheless present. 30 records sampled per
cell with seed 20260907 from that cell's unparseable clean records. Token counts come from each
model's own `tokenizer.json` at the pinned revision.

| model | unparseable clean | sampled | truncated in the reasoning block | answer the parser missed | empty | other | mean tokens (unparseable) | max | at the 320 cap | share | opening think tag | closing think tag | byte-marker artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 1499/1500 | 30 | 30/30 | 0/30 | 0/30 | 0/30 | 605.17 | 672 | 1499/1499 | 1.0000 | 1500/1500 | 0/1500 | 1500/1500 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 1329/1500 | 30 | 30/30 | 0/30 | 0/30 | 0/30 | 320.0 | 320 | 1326/1329 | 0.9977 | 0/1500 | 199/1500 | 0/1500 |
| `allenai/Olmo-3-7B-Think` | 1381/1500 | 30 | 30/30 | 0/30 | 0/30 | 0/30 | 320.0 | 320 | 1376/1381 | 0.9964 | 0/1500 | 10/1500 | 0/1500 |

Two short examples per model, from the sampled rows in `docs/audit1-diagnosis/<slug>.json`:


**`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`**, class `truncated_in_thinking` (30 of 30 sampled):

- record 12, 581 tokens, ends: `atureĠvariation.ĊĊInĠmanyĠplaces,ĠtheĠtemperatureĠincreasesĠfromĠmorningĠtoĠafternoon.ĠForĠexample,ĠinĠaĠtypicalĠday,ĠmorningĠlow,ĠthenĠincreasesĠtoĠa`
- record 29, 598 tokens, ends: `tĠtheĠreasonĠitĠfloatsĠisĠdueĠtoĠitsĠdensityĠbeingĠlessĠthanĠwater.ĊĊOptionĠ(C):ĠWoodĠisĠlight.ĠLightĠcouldĠmeanĠlowĠdensityĠorĠlowĠweight.ĠButĠ"light`

**`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`**, class `truncated_in_thinking` (30 of 30 sampled):

- record 13, 320 tokens, ends: `able comes before the observations.

Wait, no, I think the process is: first, you plan and set up the experiment, which includes creating a data table`
- record 32, 320 tokens, ends: `hape and volume, but air can be compressed and expanded, so it's a gas. So (C) is wrong.

Option (D): Ice is a liquid. Wait, no, ice is a solid. So (D`

**`allenai/Olmo-3-7B-Think`**, class `truncated_in_thinking` (30 of 30 sampled):

- record 13, 320 tokens, ends: ` doing it once, maybe they want to do it again? But the question is about the initial planning or the steps during the investigation.

Wait, maybe the`
- record 34, 320 tokens, ends: `Still less frequent than sunrise, which is daily.

Solar eclipse. Hmm, solar eclipses are trickier. They don't happen every month either. Solar eclips`

**One caveat stated rather than buried.** The R1-0528 token figures are a re-tokenization of
text the server's detokenizer did not produce correctly, so they are not that cell's generated
token count; 605 against a 320 cap is itself the evidence the text does not round-trip. What is
certain about that cell without any tokenizer is that 1,500 of 1,500 completions open a
reasoning block, 0 close one, and 1,500 carry the byte markers.

## 3. The serving test

| model | configuration | `num_predict` | clean accuracy | parse rate | mean completion tokens | max | hit the cap | closing think tag | s/item at 32 | R1 preflight under this configuration |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | A | 320 | 1/30 = 0.0333 | 1/30 = 0.0333 | 320.0 | 320 | 30/30 | 0/30 | 0.5414 | PASS (c=1 30/30 diff 0.0; c=32 30/30 diff 0.0) |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | B | 4096 | 17/30 = 0.5667 | 18/30 = 0.6000 | 1502.2 | 2931 | 0/30 | 30/30 | 5.0867 | PASS (c=1 30/30 diff 0.0; c=32 30/30 diff 0.0) |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | A | 320 | 23/30 = 0.7667 | 28/30 = 0.9333 | 159.47 | 320 | 1/30 | 0/30 | 0.3736 | PASS (c=1 30/30 diff 0.0; c=32 30/30 diff 0.0) |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | B | 4096 | 27/30 = 0.9000 | 30/30 = 1.0000 | 571.7 | 1232 | 0/30 | 30/30 | 1.4082 | PASS (c=1 30/30 diff 0.0; c=32 30/30 diff 0.0) |
| `allenai/Olmo-3-7B-Think` | A | 320 | 23/30 = 0.7667 | 29/30 = 0.9667 | 160.27 | 320 | 1/30 | 0/30 | 0.5231 | PASS (c=1 30/30 diff 0.0; c=32 30/30 diff 0.0) |
| `allenai/Olmo-3-7B-Think` | B | 4096 | 23/30 = 0.7667 | 24/30 = 0.8000 | 1897.0 | 4096 | 6/30 | 24/30 | 8.1683 | PASS (c=1 30/30 diff 0.0; c=32 30/30 diff 0.0) |
| `microsoft/Phi-4-reasoning` | A | 320 | 23/30 = 0.7667 | 26/30 = 0.8667 | 253.07 | 320 | 14/30 | 15/30 | 0.8671 | PASS (c=1 30/30 diff 0.0; c=32 30/30 diff 0.0) |
| `phi-4-reasoning` | B | - | NOT COMPLETED IN THIS LANE | | | | | | | |

Configuration A, what the prefill actually did (read from the rendered prompt):
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`: branch `inserted_a_whole_empty_block`; rendered prompt ends `"perline.Thenendwithafinallineexactlyoftheform'Answer:(X)'.<｜Assistant｜><think>\n\n</think>\n\n"`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`: branch `closed_the_block_the_template_opened`; rendered prompt ends `" end with a final line exactly of the form 'Answer: (X)'.<｜Assistant｜><think>\n\n\n</think>\n\n"`
- `allenai/Olmo-3-7B-Think`: branch `closed_the_block_the_template_opened`; rendered prompt ends `"ine exactly of the form 'Answer: (X)'.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"`
- `microsoft/Phi-4-reasoning`: branch `inserted_a_whole_empty_block`; rendered prompt ends `"tly of the form 'Answer: (X)'.<|im_end|><|im_start|>assistant<|im_sep|><think>\n\n</think>\n\n"`

### What the serving test says, read plainly

Against the cells' own measured clean accuracy over 1,500 entered items (`docs/WAVE1-AUDIT.md`),
on 30 items each:

| model | cell as measured (clean-correct / 1,500) | configuration A (30 items) | configuration B (30 items) |
|---|---|---|---|
| `allenai/Olmo-3-7B-Think` | 56 = 0.0373 | 23/30 = 0.7667 | 23/30 = 0.7667 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 140 = 0.0933 | 23/30 = 0.7667 | 27/30 = 0.9000 |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 1 = 0.0007 | 1/30 = 0.0333 | 17/30 = 0.5667 |
| `microsoft/Phi-4-reasoning` | 1,197 = 0.7980 | 23/30 = 0.7667 | not completed |

The 30-item figures are 30-item figures; they carry a Clopper-Pearson 95 percent interval
roughly 0.30 wide at these rates and are not comparable to a 1,500-item cell as an accuracy
estimate. What they establish is the direction and the size of the gap, which is not subtle.

Three things the table does not say on its own:

1. **Configuration A does not fix `DeepSeek-R1-0528-Qwen3-8B`, and the reason is a second,
   independent defect.** Its rendered prompt came back from the server with every space
   missing (`"...oftheform'Answer:(X)'.<｜Assistant｜><think>\n\n</think>\n\n"`), so its
   configuration A number is a measurement on a degenerate prompt, not a measurement of
   thinking suppression. The other three models' rendered prompts are intact. Root cause,
   read from the model repo at the pinned revision: `tokenizer_config.json` declares
   `"tokenizer_class": "LlamaTokenizerFast"` with `"legacy": true` and `sp_model_kwargs`,
   a SentencePiece-family class, over a `tokenizer.json` that is byte-level BPE with a
   correct `ByteLevel` decoder. Loading that `tokenizer.json` directly and encoding
   `"First, the question is about a planet."` gives tokens `['First', ',', 'Ġthe', …]`;
   its own decoder returns the sentence, while joining the raw token strings returns
   `'First,ĠtheĠquestionĠisĠaboutĠaĠplanet.'`, which is character for character the shape
   of the cell's 1,500 completions. Qwen3-8B at its pinned revision declares
   `"tokenizer_class": "Qwen2Tokenizer"` and has no such symptom.
2. **Configuration B still returns corrupt text on that model.** All 30 configuration B rows
   carry the byte markers. They parse anyway, at 18 of 30, because the model ends with
   `Answer:(C)` and the frozen parser's strict pattern matches a colon followed directly by a
   parenthesised letter. The ANSWER is readable; the CHAIN is not. Since the chain is this
   campaign's mediator, a configuration B cell on this model would carry a scorable outcome
   over an unreadable mediator, which is worse than an unusable cell because it looks usable.
3. **The R1 preflight passes under both configurations on every model tested**, at 30/30
   identical completions and a maximum absolute letter-logprob difference of exactly 0.0 at
   both 1 and 32 in flight. Ruling R1 does not discriminate between A and B.

**The cost that decides the arithmetic, not the accuracy.** Seconds per item at concurrency 32:
A gives 0.52 (Olmo), 0.37 (R1-Distill), 0.54 (R1-0528), 0.87 (Phi-4); B gives 8.17, 1.41 and
5.09 on the three that finished. That is 15.6x, 3.8x and 9.4x. The concurrency-1 leg of the R1
preflight is where B becomes prohibitive rather than merely expensive: on Phi-4-reasoning at
4096 tokens the server logged a sustained 12.3 generation tokens per second on that leg, and
the job was still inside it after two hours, which is why its configuration B row reads
"not completed in this lane".

**Configuration B does not always leave the block either.** On Olmo-3-7B-Think, 6 of 30
generations hit the 4,096 cap and only 24 of 30 emitted a closing think tag, so B at 4,096
still truncates one item in five on that model. `num_predict` 4,096 is not a bound anyone
measured; it is the number this test was asked to try.

**And a finding that is not about the four thinking models at all.** The 24-token forced
continuation of element 15 is spent inside a reasoning block on every model that opens one,
which is why `phi-4-reasoning`, whose clean pass produced 1,197 clean-correct of 1,500, still
has `direct` 0 scorable of 1,197, `filler` 0 of 1,197 and `twostep` 4 of 1,197. Any ruling that
fixes the clean pass and leaves `FORCE_TOKENS` at 24 leaves those arms empty.

## 4. The two candidate rulings, stated neutrally

Two rulings are on the table for element 9.4. They are stated here as options with their
costs, not weighed; the choice is the operator's.

### Candidate A: suppress the reasoning block, keep every decoding constant

Every affected model's chat template opens a reasoning block that `enable_thinking` does not
close, so the mechanism is to close it: render the prompt with the model's own template and
append the closing tag (or, where the template opens nothing, the whole empty block
`<think>\n\n</think>\n\n`, which is DeepSeek's documented way of making an R1 model skip
thinking). Nothing in element 15 changes: temperature 0.0, one sample, `num_predict` 320 for
full generations and 24 for forced continuations.

What A keeps:
- The decoding constants of element 15 are untouched, so no amendment to them is needed and
  the four affected cells become comparable with Qwen3-8B, Gemma-2-9B and Llama-3.1-8B on
  the same scale.
- The mediator's length scale is unchanged. The truncation-depth curves of element 8.3 run
  over a chain of at most 320 tokens on every roster row, which is what the pre-registered
  curve arm and the `--curve-cap` rule assume.
- The card cost per cell stays where wave 1 measured it.

What A gives up:
- It measures these models with their reasoning mode OFF. For four models whose whole
  identity is the reasoning mode, that is a different object than the one the roster names,
  and section 9.4's own words ("rather than silently reporting one mode") cut against
  reporting only the suppressed mode.
- It needs the prompt rendered through `/tokenize` and generated through `/completions`
  rather than `/chat/completions`, because two of the templates rewrite an assistant message
  with `content.split('</think>')[-1]` and would silently drop a prefill sent the usual way.
  That is a change of request path, and it has to be recorded per cell.

### Candidate B: allow the reasoning block, raise the budget, parse after it

`num_predict` rises to 4096 on the affected rows and the answer is read from the text after
the closing think tag.

What B keeps:
- It measures the models in the mode the roster names them for, which is what 9.4's
  "records the setting in every record" is for.
- It needs no prompt-rendering trick; the ordinary chat path works.

What B changes, and these are the reasons it is not free:
- **The mediator's length scale moves.** `num_predict` is an element 15 constant and the
  truncation-depth mediator is defined on the chain it produces. At 4096 the chain is an
  order of magnitude longer on the affected rows and the same on every other row, so
  `curve_area` and `commitment_depth` are no longer on one scale across the roster, and a
  model-level pooled estimand built from both is pooling two different mediators.
- **The pre-registered truncation curves would run over the thinking block.**
  `split_steps` and `truncate_cot` operate on the whole returned text. With thinking allowed,
  depth k of the curve cuts inside the reasoning block on most items, and the forced
  continuation then re-asks the model with a partial reasoning block as its "partial reasoning
  so far". That is a different intervention from the one the curves were validated on, and
  whether the extraction should apply to the curve arm as well as the answer arm is a design
  question the test does not answer.
- **The cost.** Completion tokens per item and seconds per item are the two numbers the
  serving test measures for exactly this reason; the compute degradation ladder of element 16
  never cuts n per cell or the curve arm, so a per-cell cost rise lands on the roster or the
  cue families instead.

## 5. What each would mean for the affected roster rows

The roster of element 10 has 18 rows. Which of them a 9.4 ruling reaches, read from each
model's chat template rather than from its name:

| # | roster row | reasoning-mode switch in the template | reached by a 9.4 ruling |
|---:|---|---|---|
| 1 | `Qwen/Qwen3-8B` | `enable_thinking`, honoured | no; wave 1 measured it at 1,396/1,500 |
| 2 | `Qwen/Qwen3-32B` | `enable_thinking`, honoured | no |
| 3 | `Qwen/Qwen3.6-35B-A3B` | Qwen convention, unverified at this revision | verify before the cell |
| 4 | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | none; template opens `<think>` | YES, measured |
| 5 | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | same family template | YES, by inheritance; verify |
| 6 | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | none; the model opens `<think>` | YES, measured, plus a detokenization defect of its own |
| 7 | `openai/gpt-oss-20b` | `reasoning_effort` in the harmony template | cannot be served at all on a100-40 under `VLLM_BATCH_INVARIANT=1` (below) |
| 8 | `openai/gpt-oss-120b` | same | same serving question, on a bigger card |
| 9 | `allenai/Olmo-3-7B-Think` | none; template opens `<think>` | YES, measured |
| 10 | `allenai/Olmo-3-32B-Think` | same family template | YES, by inheritance; verify |
| 11 | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | not a reasoning row | no |
| 12 | `microsoft/Phi-4-reasoning` | none; a hardcoded reasoning system prompt | measured, and it did NOT fail (below) |
| 13 | `zai-org/GLM-4.5-Air` | to verify | verify before the cell |
| 14 | `meta-llama/Llama-3.1-8B-Instruct` | not a reasoning row | no; wave 1 measured it |
| 15 | `meta-llama/Llama-3.3-70B-Instruct` | not a reasoning row | no |
| 16 | `google/gemma-2-9b-it` | not a reasoning row | no; wave 1 measured it |
| 17 | `google/gemma-3-27b-it` | not a reasoning row | no |
| 18 | `mistralai/Magistral-Small-2509` | a reasoning row; template to verify | verify before the cell |

Rows 3, 5, 10, 13 and 18 are marked "verify" because this lane read five templates, not
eighteen, and inheriting a finding from a sibling model is exactly the assumption that put
`enable_thinking` on four cells that never read it.

## 6. What is missing, named

- **`microsoft/Phi-4-reasoning` configuration B did not complete in this lane.** Job 827214
  ran configuration A to completion and was still inside configuration B's concurrency-1
  determinism leg after two hours at a logged 12.3 generation tokens per second when the VPN
  to the cluster dropped and this lane lost its connection. `config_A.json` is mirrored;
  `config_B.json` is not. Resume: `ssh soc`, then
  `ls ~/bcf/results/audit1-phi-4-reasoning/` and `cat ~/bcf/results/audit1-phi-4-reasoning/exit_code.txt`;
  if `config_B.json` exists, `scp` it into `experiments/results/audit1/st-phi-4-reasoning/`
  and re-run `python3 experiments/audit/assemble_reasoning_doc.py`. If the 4-hour wall killed
  the job first, resubmit with
  `sbatch --job-name=bcf-audit-st-phi-4-reasoning --export=ALL,BCF_MODEL=microsoft/Phi-4-reasoning,BCF_REV=1de18ec97600877ce63dbf60c73b998da99f0195,BCF_SLUG=phi-4-reasoning,BCF_MAX_LEN=16384,BCF_TIME=08:00:00 ~/bcf/repo-audit1/bcf/audit1_serving_test.sbatch`.
- **`openai/gpt-oss-20b` has no serving test at all**, in either configuration, because it
  cannot be served on an a100-40 slice under `VLLM_BATCH_INVARIANT=1` in this build.
  `docs/WAVE1-AUDIT.md` carries that finding with the server's own refusal text.
- **Thirteen of the eighteen roster rows were not opened.** Five chat templates were read.
  The rest are marked "verify" in the table above rather than inferred from a sibling model,
  because inferring from a sibling is what put `enable_thinking` on four cells that never
  read it.
- **Configuration A changes the request path**, from `/chat/completions` to `/tokenize` plus
  `/detokenize` plus `/completions`. That was necessary to keep two of the templates from
  silently discarding the prefill, and it is not the path the sweep uses today. Adopting A
  means adopting that path, and the wave-1 cells were not run on it.
- **No configuration was tried at an intermediate `num_predict`.** A and B are the two the
  test was asked to run; whether, say, 1,024 with the reasoning block allowed would clear
  most items on the smaller models is not measured here.
