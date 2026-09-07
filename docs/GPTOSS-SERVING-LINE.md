# The gpt-oss-20b serving line, measured

Element 10 of `experiments/PREREGISTRATION_jury_and_scale.md` carries `openai/gpt-oss-20b`
@ `6cee5e81ee83917806bbde320786a8fb61efebee` as roster row 7 with the serving line "mxfp4,
1 x a100-80"; the corrected card budget in `CONTRACT.md` moves it into pool (a), "the
a100-40 ... and the gpt-oss-20b mxfp4 (16 GB)". Ruling A3.9 R1 says every generation and
logprob outcome in a powered cell is served with `VLLM_BATCH_INVARIANT=1` on vLLM 0.28.0.

Wave-1 cell 826740 tried exactly that and died at engine init. This file reports what the
three follow-up serving tests measured and what the vLLM source says, and states the
candidate rulings without choosing between them. The choice is the operator's.

This file makes no ruling. It reports numbers with their denominators and quotes the
lines the numbers come from.

## The table

All six rows are `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee`,
vLLM 0.28.0, `VLLM_USE_FLASHINFER_SAMPLER=0`, tensor-parallel 1, `--max-model-len 8192`,
`--gpu-memory-utilization 0.90`, seed 7, temperature 0, arc_challenge, stated-hint, the
FIRST 30 items of the ARC pool. That last point is checkable rather than assumed: the
first 30 items hash to sha256 `d21656edbd34b8ff...` in this lane's 1,500-item pool, in
`~/bcf/repo`'s 700-item pool that wave 1 ran from, and in `~/bcf/repo-jury`'s, so these
numbers sit on the same items the other lanes measure.

### Did it serve

| # | configuration | job | card | flag | served | MXFP4 MoE backend | attention backend | model load | exit |
|---:|---|---:|---|---:|---|---|---|---:|---:|
| 1 | a100-40, flag ON (wave-1 cell) | 826740 | whole `A100-PCIE-40GB` | 1 | **no** | refused at init | TRITON_ATTN | not loaded | 5 |
| 2 | h100-47, flag ON | 826884 | `H100 NVL` MIG 3g.47gb | 1 | **no** | refused at init | FLASH_ATTN | not loaded | 5 |
| 3 | a100-40, flag OFF, preflight ON | 826883 | `A100 80GB PCIe` MIG 3g.40gb | unset | yes | `MARLIN` | TRITON_ATTN | 13.8 GiB | 10 |
| 4 | a100-40, flag OFF, preflight OFF | 826894 | `A100 80GB PCIe` MIG 3g.40gb | unset | yes | `MARLIN` | TRITON_ATTN | 13.8 GiB | 0 |
| 5 | a100-80, flag ON | not submitted | - | 1 | - | - | - | - | - |
| 6 | a100-40, flag OFF, `num_predict` 4096 | 826927 | `A100 80GB PCIe` MIG 3g.40gb | unset | yes | `MARLIN` | TRITON_ATTN | 13.8 GiB | 0 |

Row 1 is the wave-1 cell this lane was sent to explain; it is not this lane's job. Rows 3
and 4 are one configuration run twice, because `serve_and_run.sbatch` cannot both measure
the R1 preflight and continue past its refusal: row 3 keeps `BCF_PREFLIGHT=1` and stops at
exit 10 with the determinism numbers written, row 4 sets `BCF_PREFLIGHT=0` to reach the
clean pass. Row 5 is discussed under "Cap and cost consequences"; it was not run and no
number is invented for it.

Exit 5 is `serve_and_run.sbatch`'s "server died during startup". Exit 10 is its ruling-R1
preflight refusal, which is a verdict and not a crash.

The 13.8 GiB the Marlin rows load is not the same figure as the 13.03 GiB the h200 judge
job 826026 reported for the same weights on the Triton backend. Marlin repacks the mxfp4
weights at load, which is also why the a100 servers take about 270 s to answer `/models`
against the roughly 130 s of the load itself. Neither number is a KV-cache figure: this
slice reports 18.98 GiB available for KV cache and a GPU KV cache size of 546,302 tokens,
a maximum concurrency of 66.69x at 8,192 tokens per request, so 32 in flight is not near
any memory limit here.

### What the two serving rows measured

30 items, `num_predict` 320, `chat_template_kwargs` `{"reasoning_effort": "low"}`,
`BCF_CONCURRENCY=32`, one arm (`direct`), zero retries and zero failed requests everywhere.

| job | preflight verdict | 1 in flight | 32 in flight | gen/s at 1 | gen/s at 32 | logprob calls/s at 1 / 32 |
|---:|---|---|---|---:|---:|---|
| 826883 | REFUSE (exit 10) | 30/30 identical, max letter-logprob diff 0.0 over 120 | **9/30** identical, max diff **1.1250038146972656**, median 0.1875, 7 of 120 exactly zero | 0.5589 | 4.5404 | 12.7626 / 32.6339 |
| 826894 | skipped by `BCF_PREFLIGHT=0` | - | - | not measured (no preflight) | 3.75 full generations/s on the clean pass, 3.4286 on the cue pass, 8.0625 calls/s overall | - |

826883's three recorded refusal reasons, verbatim from `determinism_preflight.json`: the
flag "was None, not '1'"; "concurrency 32: identical completions 9/30, and the rule needs
30/30"; "concurrency 32: max abs letter-logprob difference 1.1250038146972656, and the rule
needs exactly 0.0".

For scale, `docs/W3B-BATCH-INVARIANT.md` measured Qwen3-8B flag-off at 13/30 and 0.875 nats
at 32 in flight (job 825548). gpt-oss-20b flag-off on Marlin is worse on both counts, 9/30
and 1.125 nats, which is what an MoE router under batching would predict. The difference
that matters is not the size of the gap but that for Qwen3-8B the flag closes it to 30/30
and 0.0, and for this model there is no flag setting that closes it at all.

### Reasoning tokens and whether the answer parses

gpt-oss is served through the harmony template with vLLM's `openai_gptoss` reasoning parser
(visible in 826740's engine config line as `reasoning_parser='openai_gptoss'`), so the
analysis channel lands in `reasoning_content` and `experiments/openai_client.py` line 302
reads `message.content` only. An answer that never leaves the analysis channel arrives as an
empty string. The reasoning-effort setting is therefore a decoding constant for this model
and is recorded in every `run_meta.json` here as `{"reasoning_effort": "low"}`.

| job | num_predict | generation calls | empty `content` | content chars min / median / max | seconds per call, median | clean accuracy on 30 |
|---:|---:|---:|---:|---|---:|---|
| 826883 (preflight probe) | 320 | 60 | 5 | 0 / 399 / 671 | 3.04 | preflight refused before the clean pass |
| 826894 (clean pass) | 320 | 54 at `max_tokens` 320 | **15** | 0 / 260 / 661, mean 259.6 | 6.29 | **24/30** |
| 826927 (clean pass) | 4096 | 60 at `max_tokens` 4096 | **0** | 199 / 511 / 5243, mean 573.6 | 7.34 (max 45.81) | **30/30** |

826894's clean pass records its own attrition, and it is the number to read beside the
accuracy: `{"n_entered": 30, "n_failed_generation": 0, "n_unparseable_clean": 6}`. No
generation failed. Six of thirty produced no parseable answer, and all 24 that did parse
were correct. So `24/30` is not "80 percent accurate"; it is 24 of 24 correct on the items
that answered, with a fifth of the roster lost inside `num_predict` 320 at reasoning effort
"low". The cue pass loses more: 15 of the 54 calls at `max_tokens` 320 came back with empty
`content`, of which 6 are the clean pass's, leaving 9 of the 24 cue-pass calls.

Because 6 of 30 is a real loss rather than a rounding, the brief's fourth row was run:
the same configuration at `num_predict` 4096, job 826927, exit 0. It settles the question.
Clean accuracy goes to **30/30** and the attrition record goes to
`{"n_entered": 30, "n_failed_generation": 0, "n_unparseable_clean": 0}`. Not one of the 60
calls at `max_tokens` 4096 came back with empty `content`, against 15 of 54 at 320. So the
320-token budget was the whole of the problem: this model answers all 30 of the first ARC
items correctly when it is given room to finish, and the six it "got wrong" at 320 were six
it never got to say.

The room costs throughput, and this is the number to price the exploratory line at if the
4096 budget is adopted:

| | `num_predict` 320 (826894) | `num_predict` 4096 (826927) | ratio |
|---|---:|---:|---:|
| clean pass, full generations/s at 32 in flight | 3.75 | 2.5 | 0.67 |
| cue pass, full generations/s at 32 in flight | 3.4286 | 0.6522 | 0.19 |
| whole cell, calls/s | 8.0625 | 2.0167 | 0.25 |
| seconds per generation call, median | 6.29 | 7.34 | |
| seconds per generation call, max | not recorded | 45.81 | |

The cue pass is where the cost lands, a factor of 5.3, because a hinted prompt makes this
model reason longer: 21,174 completion characters over 30 items against the clean pass's
13,243 over the same 30.

### The `direct` arm does not survive a reasoning model

This is a second finding, incidental to the serving question, and it is reported rather
than fixed because changing an arm is not this lane's to do. In 826894 the `direct` arm
summary reads `n: 0` scorable against `n_unscorable: 24`, `direct_accuracy.rate` null,
`with_without_cot_agreement.rate` null, and the whole `commitment_split` falls into the
`unknown` bucket (`n: 3`, `n_unscorable: 21`), so the committed-versus-moved stratification
has nothing left to stratify. The cause is in the same request log: the arm's forced
continuations run at `max_tokens` 24, and 74 of those 75 calls returned empty `content`,
because 24 tokens does not let a harmony analysis channel close.

The 24 is `FORCE_TOKENS` at `experiments/08_additive_arms.py` line 143, whose comment reads
"forced-answer continuation calls need only the final line, like 05". That is true of a
model that answers directly and false of one whose answer arrives after a reasoning channel
the client discards. `FORCE_TOKENS` is not the `direct` arm's alone: it is also the budget
at line 1015 (the generic continuation prompt), line 1197 and line 1211 (the `replay` arm).
So the exposure is every forced-continuation read in the runner, not one arm, and it is a
property of reasoning models in general rather than of gpt-oss.

Job 826927 is the control that separates `FORCE_TOKENS` from `num_predict`. It raised
`num_predict` from 320 to 4096, which took the clean pass from 6 unparseable to 0, and the
`direct` arm did not move at all: still `n: 0` scorable against `n_unscorable: 30`, and 60
of its 61 `max_tokens` 24 calls still returned empty `content`. Raising the run's token
budget therefore does not touch this; `FORCE_TOKENS` is the constant that would have to
change, and changing it is a decision about an arm, which is not this lane's. That is a plausible reading
of the wave-1 note that three thinking-model cells were unusable as measured, though this
lane measured only this cell and does not claim the other three failed for this reason.


## Why it refuses: the mxfp4 backend oracle, read

The selection happens in
`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py::select_mxfp4_moe_backend`. For a
gpt-oss model it takes the twelve-name list from `_get_priority_backends_for_gpt_oss()`
(line 306) and runs it through `_filter_by_activation(..., requested_activation_key)`.

`activation_key=None` in the error message is not a missing value. `_filter_by_activation`'s
docstring says: "Pick variants matching `requested_activation_key`; without one, prefer
BF16 if the list has any, else keep the list as-is". A None activation key therefore keeps
only the BF16-activation variants and drops the MXFP8 and FP8 ones, which is why the eight
names in the traceback are a strict subset of the twelve in the source.

Each survivor is then asked `is_supported_config` in
`vllm/model_executor/layers/fused_moe/modular_kernel.py` line 546. Its clauses run in a
fixed order, and the order is what makes the same backend fail for different stated reasons
on different cards:

```python
if not cls._supports_current_device():
    return False, _make_reason(f"current device {current_platform.device_name}")
...
elif envs.VLLM_BATCH_INVARIANT and not cls._supports_batch_invariance():
    return False, _make_reason("batch invariance")
```

Device is clause one. Batch invariance is clause nine.

### The device gate

`gpt_oss_triton_kernels_moe.py` line 36, the shared gate for both OAI Triton expert classes,
carries its window in a comment and then in code:

```python
# Keep the original `(9, 0) <= cap < (11, 0)` window on
# CUDA (covers Hopper SM90 and Blackwell SM100, excludes
# SM120)
return cap is not None and (9, 0) <= (cap.major, cap.minor) < (11, 0)
```

(The comment continues for one more clause, saying the change was ROCm-scoped and the
broader CUDA range was not validated. It is trimmed here only because it carries a
character this repository's prose does not use; the code line above is verbatim.)

An A100 is `(8, 0)`, so Triton never reaches clause nine there. Marlin's gate is far wider,
`marlin_moe.py` line 605:

```python
return p.is_cuda() and p.has_device_capability((7, 5))
```

so Marlin passes the device clause on an A100 and is judged on the later ones.

### The batch-invariance gate, and who passes it

The base class returns False (`modular_kernel.py` line 674). A repository-wide grep for
`_supports_batch_invariance` in the installed vLLM 0.28.0 finds it overridden `True` in
exactly three classes:

| file and line | class | reachable for gpt-oss mxfp4? |
|---|---|---|
| `experts/cutlass_moe.py` 692/742 | `CutlassExpertsFp4` | no, NVFP4 path |
| `experts/triton_moe.py` 62/168 | `TritonExperts` | no, the generic Triton MoE |
| `experts/fused_humming_moe.py` 95/312 | `HummingExpertsBase` | no, the HUMMING backend |

The classes `backend_to_kernel_cls` actually returns for the eight gpt-oss candidates are
`TrtLlmMxfp4Experts{Monolithic,Modular}`, `FlashInferExperts`,
`OAITritonMxfp4ExpertsMonolithic`, `OAITritonExperts`, `MarlinExperts`,
`BatchedMarlinExperts`, the XPU class and
`OCP_MXQuantizationEmulationTritonExperts`. **None of them overrides
`_supports_batch_invariance`.** Every one inherits the base `False`.

So the answer to the question the wave asked is: on sm80 `VLLM_BATCH_INVARIANT=1` does not
remove Triton, because Triton is already gone at clause one. What the flag removes on sm80
is Marlin, the only backend that had got past the device gate. On sm90 the flag removes
Triton and Marlin both. There is no card on which some mxfp4 MoE kernel in this build
survives the flag, because no mxfp4 MoE kernel in this build claims to be batch invariant.

Installing `amd-quark` to unlock the EMULATION backend does not change this:
`ocp_mx_emulation_moe.py` has no `_supports_batch_invariance` override either.

## What the two failing configurations printed

Job 826740, a whole `NVIDIA A100-PCIE-40GB` (`run_meta.json` `gpu` field), flag ON:

```
backend: TRITON, reason: kernel does not support current device cuda
backend: MARLIN, reason: kernel does not support batch invariance
backend: BATCHED_MARLIN, reason: kernel does not support ('standard',) activation format
```

Job 826884, one `NVIDIA H100 NVL` MIG 3g.47gb slice on xgpi13, flag ON, `run_label`
`powered_pinned` with `exploratory_reason` null, so a genuine R1 attempt:

```
backend: TRITON, reason: kernel does not support batch invariance
backend: TRITON, reason: kernel does not support batch invariance
backend: MARLIN, reason: kernel does not support batch invariance
backend: BATCHED_MARLIN, reason: kernel does not support ('standard',) activation format
```

The candidate list is identical on both cards. The only line that moves is Triton's, from
the device clause to the batch-invariance clause, which is the source reading measured.
Both jobs selected an attention backend without trouble, and they did not select the same
one: 826884 on sm90 reports `FLASH_ATTN` out of `['FLASH_ATTN', 'TRITON_ATTN']`, 826740 on
sm80 reports `TRITON_ATTN` out of `['TRITON_ATTN']`, a potential list of one. Either way the
refusal is in the MoE quantization path and has nothing to do with attention. 826884 ran 1 minute 52 seconds, State FAILED, ExitCode 5:0,
`exit_code.txt` 5, no weights loaded, no `Model loading took` line.

## The three candidate rulings

Stated neutrally, with what each costs. None is chosen here.

### (1) Route gpt-oss-20b to sm90 cards and keep the flag

**Refuted by measurement.** This was the obvious reading of "the h200 served it with the
TRITON backend", and job 826884 tested it directly on the nearest sm90 card this account
can get. The h200 and h100-47 successes were flag-OFF: `~/bcf/repo-jury/bcf/judge_serve.sbatch`,
the script that wrote both `server-gpt-oss-20b.log` files, never sets
`VLLM_BATCH_INVARIANT` (its only `VLLM_` export is line 123,
`export VLLM_USE_FLASHINFER_SAMPLER=0`). Moving the row to a Hopper card changes which
clause refuses it, not whether it is refused.

### (2) Serve it flag-off on a100 as an exploratory line

**Available**, at a price A3.9 R1 already names. A flag-off cell is stamped
`run_label: exploratory` by `serve_and_run.sbatch` with the reason recorded in
`run_meta.json`, its determinism preflight refuses (exit 10) because R1's gate requires the
flag, and R1 says flag-off runs "are never compared at the token level with powered cells".
So a gpt-oss row served this way is not a powered cell and cannot enter any cross-model
statement that a powered cell enters. Section 11's element-10 roster and the strata that
name gpt-oss would have to say so.

Two measured facts price this ruling. First, the batch noise it accepts is larger than the
noise the campaign has already decided is unacceptable: 9 of 30 identical completions and
1.125 nats at 32 in flight, against the 13 of 30 and 0.875 nats that R1 cites as its
reason for pinning the flag on. Serving at 1 request in flight removes the batch-order
dependence by construction, and costs a factor of 8.1 in throughput (0.5589 against 4.5404
generations per second on this slice); note that the preflight's 30/30 at 1 in flight is a
self-comparison and is not independent evidence of run-to-run determinism. Second, R1 pins
the FLASH_ATTN attention backend as part of the serving mode, and on sm80 this model cannot
supply it: both a100 runs report `Using TRITON_ATTN attention backend out of potential
backends: ['TRITON_ATTN']`, a list of one, while both sm90 gpt-oss servers report
`FLASH_ATTN` out of `['FLASH_ATTN', 'TRITON_ATTN']`. So an a100 gpt-oss row would differ
from the pinned serving mode in the attention backend as well as the flag, independently of
the mxfp4 question.

### (3) Drop the gpt-oss row

**Available and cheapest in card-hours.** It costs the gpt-oss stratum. Section 11 lists
gpt-oss as one of five families with 2 subjects, and the composition line reads
"{minus gpt-oss} 2"; the A3.1 gap table already records "no skeleton run on any gpt-oss
model" and `f_s`/`e_s` absent for the gpt-oss stratum. Dropping the row makes that gap
permanent rather than pending. Note that roster row 8, `openai/gpt-oss-120b`, inherits the
same refusal, so this ruling is about both rows and not one. That is checked rather than
assumed: at the two revisions element 10 pins, both `config.json` files carry the identical
`quantization_config` `{'modules_to_not_convert': ['model.layers.*.self_attn',
'model.layers.*.mlp.router', 'model.embed_tokens', 'lm_head'], 'quant_method': 'mxfp4'}`, so
both take the same `select_mxfp4_moe_backend` path.

### A fourth option this lane did not test

Upgrading vLLM past 0.28.0 could add a batch-invariant mxfp4 kernel. It is out of scope
here and its cost is not local: A3.9 R1 pins "vLLM 0.28.0" as part of the serving mode of
record, and `docs/W3B-BATCH-INVARIANT.md` measured that a change of kernel path moves 20 of
30 greedy completions. An upgrade is a new revision of every powered cell, not of one row.
Serving a dequantized bf16 re-upload of the weights would likewise be a different roster
revision from the one element 10 pins, and `serve_and_run.sbatch` refuses on revision drift
with exit 8.

## Cap and cost consequences

Live at 2026-09-07 15:50 to 15:52, read from `squeue` RUNNING allocations (the per-user
`MaxTRESPU` caps count every campaign on the account, so alta and the jury lane count):

| pool | per-user cap | held by this account | pending cluster-wide | note |
|---|---:|---:|---:|---|
| a100-40 | 8 | 3 running | 1 | the roomy pool; both a100-40 tests started here |
| a100-80 | 4 | 0 running, 3 pending | 21 | 2 of the 3 pending rows are alta |
| h100-47 | 4 | 1 running | 43 | 826884 got a slice in under a minute |
| h100-96 | 2 | 0 | 43 | also the only pool for a 70B at tensor-parallel 2 |
| h200-141 | 1 | 1 running | 6 | one card, and the `normal` partition ceiling is 3:00:00 |

What this lane itself spent, from `sacct -X`: 826883 a100-40 00:06:38 (FAILED 10:0),
826884 h100-47 00:01:52 (FAILED 5:0), 826894 a100-40 00:05:15 (COMPLETED 0:0), 826927
a100-40 00:05:57 (COMPLETED 0:0). Nineteen and a half card-minutes in total, three of them
on the a100-40 pool and under two on h100-47, with 64 GB and 8 CPUs each and every job
carrying `--exclude=xgpj0`. Nothing was cancelled to make room and no cap was exceeded; at
16:15 the account held 4 of 8 a100-40, 0 of 4 a100-80 and 1 of 4 h100-47.

What that means for each ruling:

- Ruling (1) would have spent the h100-47 pool. The cap there is 4 against 43 jobs pending
  cluster-wide, and it is the same pool the campaign wants for other work; 826884 shows the
  spend buys nothing, since the job dies at init in under two minutes.
- The h200-141 cap is ONE card, and its partition ceiling is 3 hours, so it can never host
  a gpt-oss cell of record even if the flag question were solved: a full cell runs for
  hours and the roster asks for three substrates.
- h100-96's cap of 2 is the only route to a 70B dense subject at tensor-parallel 2 (element
  10 rows 5 and 15). Spending it on a 20B mxfp4 row that fits on a 40 GB slice would trade
  a subject that has no other home for one that has three.
- Ruling (2) costs a100-40 slices, the roomy pool, and nothing scarce. The rate to price
  it at is 3.75 FULL generations per second on the clean pass at 32 in flight, measured
  on one MIG 3g.40gb slice at `num_predict` 320 (job 826894); the `num_predict` row
  below changes that price if it is adopted, and the throughput figures here are not
  comparable with `docs/TRACKG-THROUGHPUT.md`'s, which are priced flag-off sequential.
- Configuration (c) of this lane's brief, a whole a100-80 with the flag on, was NOT
  submitted. The cap admits it (0 of 4 held at 15:50). What refuses it is that the answer
  is already on the record twice: `sbatch --test-only` for that exact job answers "Job
  826893 to start at 2026-09-07T16:59:11", about 68 minutes out behind 21 pending a100-80
  jobs, and an a100-80 is compute capability (8, 0), the same as the
  `NVIDIA A100-PCIE-40GB` whole card that job 826740 already refused on. The oracle's device
  gates read capability and never MIG-versus-whole-card, so 826740 is already the whole-card
  sm80 measurement. The resume command, if the operator wants the third copy, is
  `bash ~/bcf/repo-gptoss/bcf/gptoss_line.sh a100-80-flagon`, which does its own live cap
  check and refuses rather than queueing past it.

## What is not measured here

- One model, one substrate (arc_challenge), 30 items, one cue family. Nothing about
  gpt-oss-120b is measured, though it shares the mxfp4 quantization and therefore the
  backend-selection refusal.
- Nothing about whether the flag-off gpt-oss outputs are correct or good, only what the
  server does and what it produces. `docs/W3B-BATCH-INVARIANT.md` measured, on a different
  model, that the two kernel paths disagree on 20 of 30 greedy completions; no ground truth
  here says which path is right.
- Nothing about vLLM versions other than 0.28.0 as installed in the `bcf` conda env.
- One reasoning-effort setting. `{"reasoning_effort": "low"}` was chosen and recorded so the
  `num_predict` question would have a fair chance; "medium" and "high" are untested, and
  since "low" already loses 6 of 30 clean answers inside 320 tokens, they would lose more.
- The `direct` arm finding rests on 24 items in one cell. It is a mechanism, not a rate.
- Nothing about the OTHER two substrates or the other cue families in element 10, and
  nothing about whether a gpt-oss row would pass the campaign's other gates if the serving
  question were settled.
