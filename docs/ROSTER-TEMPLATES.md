# Element 10, all 18 chat templates read at the pinned revisions

Roster template lane, 2026-09-07. Fetches `tokenizer_config.json` (and
`chat_template.jinja` where the repo carries one) for every row of the 18-model roster
in `experiments/PREREGISTRATION_jury_and_scale.md` section 11 (element 10), at the
revisions pinned there, with `huggingface_hub`. Not a live server render: the cluster
link is down for this lane, so classification is a read of each template's own text,
cross-checked against `docs/REASONING-MODE-TEST.md` section 3's live serving numbers
where that document already measured a row. Script: `experiments/roster_templates.py`.
Full machine-readable output, all 18 rows, every hash: `experiments/results/roster-templates/templates.json`.

Sixteen of eighteen rows fetched. Two did not: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
(row 11) and `mistralai/Magistral-Small-2509` (row 18) carry no `tokenizer_config.json`
and no `chat_template.jinja` at their pinned revisions — both repos ship only
`tekken.json`, Mistral's own tokenizer format, served by vLLM's `--tokenizer-mode
mistral` path rather than an HF Jinja chat template. Checked independently for each
row (19 files listed for each repo, neither file present in either listing); row 18's
absence is not inferred from row 11's.

## Classes

1. **Documented switch.** The template reads a kwarg (name and default recorded) that
   changes what the generation prompt contains.
2. **Opens a reasoning block, no switch.** The generation prompt ends with an
   unconditional open tag (named) and the template documents nothing to turn it off.
3. **Plain instruct.** No reasoning-related token anywhere in the template.
4. **Other.** Neither of the above cleanly, described per row. Two shapes appear on
   this roster: a template whose generation-prompt tail is bare (no switch, no open
   tag) but which is a reasoning model by a mechanism that lives elsewhere in the
   template (DeepSeek-R1-0528-Qwen3-8B, Phi-4-reasoning), and gpt-oss's harmony format
   (`reasoning_effort`, a string level rather than a boolean, injected into a system
   message rather than gating a `<think>`-style tag).

## The 18-row table

`config sha256` is `tokenizer_config.json`'s hash when the template is inline there
(rows 1, 2, 4, 5, 6, 12, 14, 15, 16, 17); for rows whose config carries no inline
template (7, 8, 9, 10, 13), the hash shown is `chat_template.jinja`'s, and the config's
own hash is in `templates.json`. Row 3 has both, byte-identical (see below). Full
64-character hashes for every row are in `templates.json`; shown here truncated to 16
hex characters for the table's width.

| # | model | revision (first 12) | config sha256 (first 16) | class | switch / tag | assistant rewrite | R12 config-A action |
|---:|---|---|---|---|---|---|---|
| 1 | `Qwen/Qwen3-8B` | `b968826d9c46` | `d5d09f07b48c3086` | 1 | `enable_thinking`, default **true** | yes | not applicable |
| 2 | `Qwen/Qwen3-32B` | `9216db5781bf` | `d5d09f07b48c3086` | 1 | `enable_thinking`, default **true** | yes | not applicable |
| 3 | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd9` | `e84f32a23fdda276`\* | 1 | `enable_thinking`, default **true** | yes | not applicable |
| 4 | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `6a6f4aa41979` | `8ac8c85fb242563c` | 2 | tag `<think>` | yes | close `</think>` |
| 5 | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | `b1c0b44b4369` | `8ac8c85fb242563c` | 2 | tag `<think>` | yes | close `</think>` |
| 6 | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | `6e8885a6ff5c` | `78872f2c1f0a5fb7` | 4 | none in the tail; model opens its own `<think>` | yes | insert empty block (row HELD, R12(4): tokenizer defect) |
| 7 | `openai/gpt-oss-20b` | `6cee5e81ee83` | `a4c9919cbbd4acdd`\* | 4 | `reasoning_effort`, default `"medium"` | no | not applicable (R11 governs) |
| 8 | `openai/gpt-oss-120b` | `b5c939de8f75` | `a4c9919cbbd4acdd`\* | 4 | `reasoning_effort`, default `"medium"` | no | not applicable (R11 governs) |
| 9 | `allenai/Olmo-3-7B-Think` | `d97e442d7cc6` | `6d549883b5ed1287`\* | 2 | tag `<think>` | no | close `</think>` |
| 10 | `allenai/Olmo-3-32B-Think` | `f2edda15216e` | `eba6e269f669706e`\* | 2 | tag `<think>` | no | close `</think>` |
| 11 | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `95a6d26c4bfb` | fetch FAILED, no such file | n/a | n/a | n/a | not applicable, no HF template |
| 12 | `microsoft/Phi-4-reasoning` | `1de18ec97600` | `6d3efcfa6a8a438e` | 4 | none in the tail; hardcoded system prompt demands one | no | insert empty block (R12(3): not clean, 14/30 hit the cap) |
| 13 | `zai-org/GLM-4.5-Air` | `a24ceef6ce4f` | `44f815868bf02fa4`\* | 1 | `enable_thinking`, default **true** | yes | not applicable |
| 14 | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a1` | `177c7b61e616fecb` | 3 | none | no | not applicable |
| 15 | `meta-llama/Llama-3.3-70B-Instruct` | `6f6073b42301` | `5c756ad6048c8849` | 3 | none | no | not applicable |
| 16 | `google/gemma-2-9b-it` | `11c9b309abf7` | `cb32b7929c62608d` | 3 | none | no | not applicable |
| 17 | `google/gemma-3-27b-it` | `005ad3404e59` | `bfe25c2735e39540` | 3 | none | no | not applicable |
| 18 | `mistralai/Magistral-Small-2509` | `a31cc96ab10c` | fetch FAILED, no such file | n/a | n/a | n/a | not applicable, no HF template |

\* Hash is `chat_template.jinja`'s (config's own `chat_template` field is `null` on
this row). Row 3 carries both files, confirmed byte-identical (`diff -q` clean); its
table hash is the jinja file's.

## Cross-check against the five rows the audit already read

`docs/REASONING-MODE-TEST.md` section 1 names five templates it read directly:
`Qwen/Qwen3-8B`, `allenai/Olmo-3-7B-Think`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`,
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, `microsoft/Phi-4-reasoning`. **No factual
disagreement.** Every claim that document makes about these five templates' mechanics
is confirmed by this lane's independent fetch:

- Row 1 (`Qwen3-8B`): the doc says `enable_thinking`, honoured. Confirmed, default
  true, lines 84-89 below.
- Row 9 (`Olmo-3-7B-Think`): the doc says "none; template opens `<think>`". Confirmed,
  the tail opens `<think>` unconditionally with no guard anywhere in the file.
- Row 4 (`DeepSeek-R1-Distill-Llama-8B`): the doc says the same. Confirmed, and the
  `content.split('</think>')[-1]` assistant-rewrite `docs/REASONING-MODE-IMPL.md`
  section 1 names for this row is exactly what the template does.
- Row 6 (`DeepSeek-R1-0528-Qwen3-8B`): the doc's own section 1 already distinguishes
  this row from row 4 and row 9 in prose — "the model opens `<think>` itself" rather
  than the template opening it — and `REASONING-MODE-IMPL.md`'s branch table already
  gives it its own name, `inserted_a_whole_empty_block`, distinct from row 4 and row
  9's `closed_the_block_the_template_opened`. This lane's read confirms that
  distinction at the template level: the generation-prompt tail is bare
  (`{{'<｜Assistant｜>'}}`, no tag), and what makes this a reasoning template lives in
  the assistant-message rewrite instead (`content.split('</think>')[-1]`). Under this
  lane's four-way scheme that puts row 6 in **class 4**, not class 2, which is a
  finer split of what the audit already knew, not a disagreement with it.
- Row 12 (`Phi-4-reasoning`): same shape. The doc says "none; a hardcoded reasoning
  system prompt". Confirmed: the generation-prompt tail is bare
  (`{{ '<|im_start|>assistant<|im_sep|>' }}`), and the hardcoded system prompt is what
  demands the `<think>...</think>` structure. Class 4 here too, for the same reason as
  row 6.

One thing worth naming precisely because a classifier that only reads the
generation-prompt tail would get it wrong: rows 6 and 12 would read as **class 3,
plain instruct**, under a tail-only method, since neither template's tail contains a
switch or an open tag. `tests/test_roster_templates.py` has a dedicated test
(`test_other_template_with_a_marker_outside_the_tail_is_class_4_not_class_3`) pinned
to exactly this failure mode, and `classify_chat_template` reads the FULL template for
its class-4 catch-all rather than the tail alone for that reason.

## What this read adds beyond the five: the "verify" rows

`docs/REASONING-MODE-TEST.md` section 5 marks rows 3, 5, 10, 13 and 18 "verify the
template before the cell" because that lane read five templates, not eighteen. This
lane read all eighteen (fetch failures on 11 and 18 aside). Four of the five resolve;
one does not.

- **Row 5** (`DeepSeek-R1-Distill-Llama-70B`): byte-identical to row 4
  (`8ac8c85fb242...` both). "YES, by inheritance" was correct here.
- **Row 10** (`Olmo-3-32B-Think`): NOT byte-identical to row 9 — the two `.jinja` files
  differ by one line, the default system-prompt text (row 9's names function-calling
  and a knowledge cutoff, row 10's is the one-line "You are a helpful AI assistant.").
  The reasoning-block mechanism itself is identical (same unconditional `<think>` open
  at the same structural position). Class and R12 action both carry over correctly;
  the template is not the same file.
- **Row 3** (`Qwen3.6-35B-A3B`): same class as `Qwen3-8B` (`enable_thinking`, default
  true) but a DIFFERENT mechanism from it: where `Qwen3-8B`'s guard has no `else` (thinking
  on leaves the prompt bare and the model opens its own block), row 3's guard has an
  `else` that appends `<think>\n` itself:

  ```
  148     {{- '<|im_start|>assistant\n' }}
  149     {%- if enable_thinking is defined and enable_thinking is false %}
  150         {{- '<think>\n\n</think>\n\n' }}
  151     {%- else %}
  152         {{- '<think>\n' }}
  153     {%- endif %}
  ```

  This does not change row 3's class or its R12 action (still not applicable, the
  documented switch already covers it), but it means row 3 is not the row to reuse as
  "the Qwen mechanism" without a byte-level check — exactly the caution `docs/REASONING-MODE-TEST.md`
  gives for inheriting across rows, demonstrated on the row that motivated it.
- **Row 13** (`GLM-4.5-Air`): resolves to class 1, `enable_thinking`, default true — same
  polarity as `Qwen3-8B` (no `else`, thinking-on leaves the prompt bare). GLM's
  template also appends a literal `/nothink` marker to the last user turn when
  `enable_thinking` is off (line 47), a second, redundant suppression mechanism this
  lane has not seen on any other row.
- **Row 18** (`Magistral-Small-2509`): does NOT resolve. No `tokenizer_config.json`,
  no `chat_template.jinja` — same failure as row 11, fetched independently, not
  inferred. There is no HF chat template for this row to verify; whatever governs its
  reasoning behaviour lives in Mistral's own serving stack, outside what this method
  can read. Still open per R12(6).

One further, smaller note: `docs/REASONING-MODE-TEST.md` section 5 marks row 2
(`Qwen3-32B`) "no" without a "verify" flag, even though section 1's five-template list
does not include it either. This lane's fetch confirms row 2's `tokenizer_config.json`
is byte-identical to row 1's, so the "no" was correct — but it was not, on that
document's own account, independently read at the time it was written. Not a
disagreement (the answer was right), just a gap the "five, not eighteen" caveat did
not fully cover.

## Quoted lines, per row

At most six lines each, exact text. Templates rendered on one physical line by their
publisher (DeepSeek's two, Phi-4-reasoning's) are quoted as the decisive substring
rather than numbered lines, since there are no line breaks to number.

**Row 1/2, `Qwen3-8B` / `Qwen3-32B`** (identical file, lines 84-89):
```
84  {%- if add_generation_prompt %}
85      {{- '<|im_start|>assistant\n' }}
86      {%- if enable_thinking is defined and enable_thinking is false %}
87          {{- '<think>\n\n</think>\n\n' }}
88      {%- endif %}
89  {%- endif %}
```
No `else`: enable_thinking true or undefined appends nothing, the model opens its own
block. Assistant-turn rewrite present too (lines 38-40, same file): `content.split('</think>')[-1]`
strips a prior turn's reasoning back out before re-rendering it.

**Row 3, `Qwen3.6-35B-A3B`** (lines 148-153, quoted above in the "verify" section).

**Row 4/5, `DeepSeek-R1-Distill-Llama-8B` / `-70B`** (identical file, one physical
line; decisive substrings):
```
{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\n'}}{% endif %}
```
Unconditional open, no switch. Assistant rewrite, same file:
```
{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}
```

**Row 6, `DeepSeek-R1-0528-Qwen3-8B`** (one physical line; decisive substrings):
```
{{'<｜User｜>' + content + '<｜Assistant｜>'}}
...
{% if add_generation_prompt and not ns.is_last_user and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}
```
The prompt ends bare at `<｜Assistant｜>`; nothing opens `<think>` in the rendered
text. Assistant rewrite, same file: `content.split('</think>')[-1]`, identical
pattern to row 4.

**Row 7/8, `gpt-oss-20b` / `-120b`** (identical `chat_template.jinja`, lines 203-206
and 329-330):
```
203      {%- if reasoning_effort is not defined %}
204          {%- set reasoning_effort = "medium" %}
205      {%- endif %}
206      {{- "Reasoning: " + reasoning_effort + "\n\n" }}
...
329  {%- if add_generation_prompt -%}
330  <|start|>assistant
```
`reasoning_effort` is injected into the SYSTEM content, not read again at the
generation-prompt line; the generation prompt itself is bare (no channel appended, the
model chooses `analysis` or `final`). No `<think>` anywhere; the harmony format uses
`<|channel|>analysis<|message|>` for the reasoning channel instead.

**Row 9, `Olmo-3-7B-Think`** (15 physical lines; decisive substring, end of file):
```
{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n<think>' }}{% endif %}{% endfor %}
```
Unconditional open, no switch anywhere in the file. No assistant-content split (a
prior assistant turn's `content` is re-emitted as-is).

**Row 10, `Olmo-3-32B-Think`**: same substring as row 9, at the same structural
position; the file differs only in its default system-prompt line (see above).

**Row 12, `Phi-4-reasoning`** (one physical line; decisive substrings):
```
...structure your response into two main sections: Thought and Solution using the
specified format: <think> {Thought section} </think> {Solution section}...
```
```
{% for message in messages %}{% if (message['role'] == 'user') %}...{% elif (message['role'] == 'assistant') %}...{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}
```
No `system` branch in the loop (a user-supplied system message is dropped, matching
`docs/REASONING-MODE-TEST.md`); the generation prompt itself is bare, no `<think>`.
No assistant-content split.

**Row 13, `GLM-4.5-Air`** (lines 101-102, plus the switch's second mechanism at line 47):
```
101  {%- if add_generation_prompt -%}
102      <|assistant|>{{- '\n<think></think>' if (enable_thinking is defined and not enable_thinking) else '' -}}
```
```
47  {{- '/nothink' if (enable_thinking is defined and not enable_thinking and not
    visible_text(m.content).endswith("/nothink")) else '' -}}
```
Same polarity as `Qwen3-8B`: no `else` on the generation-prompt guard, so
`enable_thinking` true or undefined leaves the prompt bare.

**Rows 14/15, `Llama-3.1-8B-Instruct` / `Llama-3.3-70B-Instruct`** (identical file):
no line anywhere in either template matches `think` or `reason`, case-insensitive.
Plain instruct.

**Rows 16/17, `gemma-2-9b-it` / `gemma-3-27b-it`**: same, no match anywhere.
`gemma-2-9b-it`'s full template (591 bytes) has no reasoning-related token at all;
`gemma-3-27b-it`'s chat_template is far larger (over 1MB — it embeds Gemma 3's tool
and multimodal handling) and still has none.

**Rows 11/18, `Mistral-Small-3.2-24B-Instruct-2506` / `Magistral-Small-2509`**: fetch
FAILED for both. `list_repo_files` at the pinned revision returns 19 entries for each
repo; neither `tokenizer_config.json` nor `chat_template.jinja` is among them. Both
repos carry `tekken.json` instead (Mistral's own tokenizer format: `config`, `vocab`,
`multimodal`, `special_tokens` keys, no chat-template field). No inference from one
row to the other; each was fetched and failed on its own.

## Limits

- This is a text read at fetch time, not a live render. `docs/REASONING-MODE-TEST.md`
  section 3 already measured live behaviour under both configurations for rows 4, 6, 9
  and 12; nothing here supersedes that measurement, and rows 5 and 10's class rests on
  their template text matching (row 5) or matching in mechanism (row 10) rather than on
  a live re-run.
- `classify_chat_template`'s "default" inference (section above) reads the guard's
  polarity in the generation-prompt tail; it is right on all four class-1 rows on this
  roster but is a heuristic, not a Jinja evaluator, and would need widening for a
  guard shape none of these eighteen use.
- A pinned revision's file content is assumed stable; a repository owner CAN rewrite
  history under an existing tag, which the recorded sha256 would expose against a
  future re-run but which this one-time read cannot detect on its own.
- Rows 11 and 18 remain unclassifiable by this method. Whatever governs Mistral's two
  roster rows' reasoning behaviour is not an HF chat template.
