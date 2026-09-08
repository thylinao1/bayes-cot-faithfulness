# The hold-lift runbook: what the orchestrator runs after the R12(2) gate passes

Manifests lane, 2026-09-08. This is a runbook, not a ruling: it says the exact sequence
for lifting `bcf/waves/HOLD_MODELS.txt`'s hold on `allenai/Olmo-3-7B-Think` and
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` once R12(2)'s two-path gate has passed, what
stays held and why, and the Phi-4-reasoning exploratory off cell. It edits nothing in
`bcf/waves/HOLD_MODELS.txt` itself (that file is the orchestrator's, per this lane's
brief) and does not push, sbatch, or scancel anything.

**Status as of this writing (2026-09-08, read against the cluster, read-only):** the
gate has ALREADY PASSED (job 828507, `DECISION-LOG.md` 2026-09-08 09:25) and R12(6)'s
lift condition is met. This runbook is therefore live, not hypothetical, and Step 0
below is a genuine blocker on running it today.

## Step 0 (BLOCKING, discovered in this lane, not yet fixed anywhere): `bcf/wave.sh` silently drops the LAST KEY=VALUE field of every row

**This must be fixed, or worked around per-row, before Step 1 is run.** Skipping it does
not make the lift fail loudly; it makes the lifted cells run under the WRONG
`reasoning_mode` with no error, which is precisely the failure class R12(2)'s gate exists
to rule out.

**The mechanism.** `bcf/wave.sh`'s `row_fields()` (line 274) is

    row_fields() { printf '%s' "$1" | tr '\t' '\n'; }

`printf '%s'` writes no trailing newline. When this feeds a `while IFS= read -r field;
do ... done < <(row_fields ...)` loop (used at line 287, for `CARDS_WANTED`/`MAX_TP`/
`MAX_HOURS`, and at line 470, for the `--export=` string every job actually reads), bash's
`read` returns failure on the final, newline-less line, so the `while` loop's own test
fails and the LAST field is silently never handed to the loop body -- read into no
variable the caller sees, on both call sites. Confirmed with a two-line reproduction:

    $ printf '%s' "A=1	B=2	C=3" | tr '\t' '\n' | while IFS= read -r f; do echo "GOT: [$f]"; done
    GOT: [A=1]
    GOT: [B=2]

**Confirmed live, twice, read-only, against the cluster today (`ssh soc`, `wave.sh
--check-only`):**

1. An UNMODIFIED row (`Qwen/Qwen3-8B`, wave `a100-40-01.tsv`, whose last field has always
   been `BCF_EXPECTED_HOURS=1.477`): the rendered `--export=` line ends at
   `BCF_BATCH_INVARIANT=1`. `BCF_EXPECTED_HOURS` never reaches the job. This has been
   silently true of every row of every wave submitted so far -- harmless only because
   `serve_and_run.sbatch` never reads `$BCF_EXPECTED_HOURS` (it is planning-time metadata)
   and because the same bug also zeroes `MAX_HOURS` in the first loop, which only ever
   suppressed a `--resume`-legs NOTE line that was never going to fire on cells this
   short anyway.
2. A row from this lane's own edit (`allenai/Olmo-3-7B-Think` /
   `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `BCF_REASONING_MODE=off` appended as the
   new last field): the rendered `--export=` line ends at `BCF_EXPECTED_HOURS=1.477`.
   `BCF_REASONING_MODE` never reaches the job; `serve_and_run.sbatch` would default it to
   `"default"` (its own `REASONING_MODE="${BCF_REASONING_MODE:-default}"`), silently
   running the wrong configuration and recording it as `reasoning_mode: "default"` in a
   record the runner's own assertion would accept without complaint, since `"default"` is
   a valid value.

Point 2 is the reason this is a hard blocker and not a note: it is not that the lifted
cells would fail to submit, it is that they would submit, run, complete, and publish
under the configuration R12(1) explicitly rejected as the cell of record, with the
record's own `reasoning_mode` field reporting `"default"` truthfully -- so nothing
downstream would ever see an error to catch this on.

**Why this was never seen before.** Every row of every wave manifest committed to date
has `BCF_EXPECTED_HOURS` as its true last field, and nothing reads that variable at
runtime. `bcf/waves/explore-phi4-off-01.tsv` (see the Phi-4 section below), submitted
today as job 828564, happens to place `BCF_OUT_SUBROOT=explore-off` after
`BCF_REASONING_MODE=off` -- which was the right field for that row on its own merits
(it keeps the exploratory cell's output out of the wave-1 cell-of-record's directory),
and it incidentally saves `BCF_REASONING_MODE` from this bug by accident, not by design.
Nothing in `DECISION-LOG.md` or the docs tree names this mechanism; it is new to this
lane.

**The fix, minimal and verified locally (not applied in this worktree; this lane's tasks
did not include it, and it is shared tooling other campaigns' live jobs read):**

    row_fields() { printf '%s\n' "$1" | tr '\t' '\n'; }

One added `\n`. Verified against the same reproduction: with the fix, both
`BCF_EXPECTED_HOURS` and `BCF_REASONING_MODE` appear in the rendered export. No other
behavior of `wave.sh` depends on the missing trailing newline (both call sites already
skip empty fields with `[ -n "$field" ] || continue`, so a fix that also caused one
trailing empty read would be a no-op, not a new bug).

**What Step 0 concretely requires:** whichever lane owns `bcf/wave.sh` applies the
one-line fix above, it is committed and reviewed, and `bcf/wave.sh --check-only` is
re-run against at least one row carrying `BCF_REASONING_MODE` to confirm the field now
appears in the rendered `--export=` line before Step 1 proceeds. Until then, treat
every wave manifest with `BCF_REASONING_MODE` as its row's last field (all 54 rows this
lane added) as NOT YET SAFE TO SUBMIT for real.

## Step 1: edit `bcf/waves/HOLD_MODELS.txt`

Remove exactly these two lines (the R12(6) lift condition -- gate passed AND the runner
records `reasoning_mode`/`reasoning_path`, both true as of `13cf65a` merged to main):

    allenai/Olmo-3-7B-Think
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B

Leave every other line untouched, including the comment block above them (it documents
history that is still true) and the three lines below:

    deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    microsoft/Phi-4-reasoning
    openai/gpt-oss-20b

## Step 2: commit

A normal commit on whatever branch the orchestrator uses for this edit. Message should
name R12(6) and the gate job id (828507) as the reason, per this repo's convention of a
checkable justification in the commit itself.

## Step 3: push

Orchestrator-owned, per every lane's standing limits. Not run by this lane.

## Step 4: sync the tree -- TWO trees, not one

`bcf/wave_feeder.sh` and `bcf/wave.sh` read from different places, and both have to be
current or the lift silently uses stale data:

1. **`~/bcf/src`**, the mutable checkout `wave_feeder.sh` reads `HOLD_MODELS.txt` and the
   wave manifests from directly (`WAVES_DIR="${SRC_ROOT}/bcf/waves"`,
   `SRC_ROOT` = wherever the script itself is checked out). This needs a plain
   `git pull` (or equivalent) to the pushed commit from Step 3. Skipping this and only
   syncing the immutable tree below leaves the feeder reading the OLD hold list and the
   OLD (pre-`BCF_REASONING_MODE`) manifests, so it would keep treating the two rows as
   held.
2. **`~/bcf/repo-<new short sha>`**, the immutable per-commit tree jobs actually read
   (`BCF_REPO`), created automatically by `bcf/wave.sh`'s own `sync_tree()` the next time
   any wave is submitted or checked from the new commit. This is what carries Step 0's
   fix (once applied) and the `BCF_REASONING_MODE=off` columns to the running job. Do not
   create it by hand and do not point it at the forbidden shared `~/bcf/repo` (DECISION-
   LOG 2026-09-07 03:58 ruling (b); `wave.sh` itself refuses that target).

Today's log already shows the failure mode of getting this out of order: at 08:56 the
feeder reported `NOTHING WAS SUBMITTED: wave.sh exited 1` because `~/bcf/src` had been
checked out to `f712a9b` while the feeder's own loop still named an older `ed3c74c` tree
(`wave.sh` refuses a `--repo-tree` synced from a different commit than `~/bcf/src`'s
HEAD). Pull `~/bcf/src` and restart the feeder loop (Step 5) together, not separately.

## Step 5: restart the feeder against it

Stop whatever poll loop is currently running against the pre-lift commit (today's log
names one such stop-and-restart at 08:56, for the same reason as Step 4). Start a new one
from the freshly pulled `~/bcf/src`, the standing pattern from `bcf/wave_feeder.sh`'s own
header:

    bash bcf/ssh_retry.sh --wall 240 --tries 2 --gap 20 -- \
      'bash $HOME/bcf/src/bcf/wave_feeder.sh --pool a100-40'

## Step 6: `feeder held-only` -- and the second problem this lane found

`wave_feeder.sh held-only` (`bcf/hold_filter.py collect`) rebuilds a held-only wave from
`held_rows` TEXT STORED IN `feeder-state.json`, frozen at the moment each source wave was
originally filtered. Read directly off the cluster's `feeder-state.json` today
(read-only, no state changed): six `sweep`/`a100-40` waves currently carry recorded held
rows --

    a100-40-02.tsv  a100-40-03.tsv  a100-40-04.tsv
    a100-40-05.tsv  a100-40-06.tsv  a100-40-08.tsv

-- five held models each (`allenai/Olmo-3-7B-Think`, `deepseek-ai/DeepSeek-R1-0528-
Qwen3-8B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `microsoft/Phi-4-reasoning`,
`openai/gpt-oss-20b`), 30 rows total, matching `wave_feeder.sh held-only --dry-run`'s own
"collected 30 held row(s) from 6 recorded sweep/a100-40 wave(s)" when run today. That
FROZEN text predates this lane's edit, so a plain `feeder held-only` run after Steps 1-5
would submit the twelve Olmo-3-7B-Think / R1-Distill-Llama-8B rows (two models x six
waves) WITHOUT `BCF_REASONING_MODE` at all -- not silently dropped by Step 0's bug this
time, just never present in the frozen row text to begin with. Same silent-wrong-
configuration failure as Step 0, different mechanism, same fix category (do not trust
`held-only` for these two models until this is addressed).

**Two waves outside this count need their own check before Step 1, not after:**
`a100-40-01.tsv` is not in the held-rows list above because it ran BEFORE the hold
existed -- it is the wave `HOLD_MODELS.txt`'s own header cites for the 1,381/1,500 and
1,329/1,500 unparseable-clean counts that motivated R12 in the first place, so its
Olmo-3-7B-Think and R1-Distill-Llama-8B cells already ran, under the truncation-affected
default configuration, and are VOIDED in the same sense `a100-40-resub-01.tsv` already
resubmits four other voided cells. `a100-40-07.tsv` is also absent from the held-rows
list; today's log records it as "submitted by hand from the f712a9b checkout" outside
`wave_feeder.sh next`, which is the only path that calls `hold_filter.py` at all --
verify before assuming wave 07's rows for these two models were actually held rather
than submitted unfiltered, the same "verify before the cell" discipline this campaign
already applies to unread templates.

**Recommended path for both problems, following the precedent already in this
repository:** do not run `feeder held-only` for `sweep`/`a100-40` until the twelve rows
above are accounted for. Instead, once Steps 0-5 are done and confirmed, hand-build a
`bcf/waves/a100-40-resub-02.tsv` the same way `a100-40-resub-01.tsv` already does --
rows copied VERBATIM from the current (post-fix, post-column) `a100-40-0{2,3,4,5,6,8}.tsv`
and `a100-40-01.tsv` (and `-07.tsv`, once verified) for exactly the two now-unheld
models, submitted directly with `bcf/wave.sh --type sweep --gpu-type a100-40` the way
`explore-phi4-off-01.tsv` already was today. This sidesteps the frozen-text problem
entirely rather than patching `feeder-state.json` by hand. The four STILL-PENDING waves
(`a100-40-09.tsv` through `-12.tsv`) need no special handling: an ordinary
`wave_feeder.sh next` run reads them fresh off `~/bcf/src`, already carrying the current
manifest content and filtered against the current (lifted) hold list.

## What stays held, and why

| model | ruling | why it does not lift with the other two |
|---|---|---|
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | R12(4) | Its pinned revision's `tokenizer_config.json` declares `LlamaTokenizerFast`/`legacy: true` over a byte-level-BPE `tokenizer.json`; the served decoder does not round-trip text (space-stripping defect, `docs/REASONING-MODE-TEST.md` section 3). No `reasoning_mode` setting fixes a tokenizer defect; nothing is interpretable from this row until it is served with a tokenizer that round-trips, which is a serving-line question, not this ruling's. |
| `microsoft/Phi-4-reasoning` | R12(3) | Configuration A hits the 320-token cap on 14/30 (not clean); configuration B (job 827214, collected 2026-09-08 09:03) gives 0/30 correct, 0/30 parsed, 30/30 at the 4,096 cap, 0/30 closing tags -- unusable. The remaining candidate, off-mode with continuation-closing (R12(3)'s "second A variant"), is running RIGHT NOW as an exploratory cell (below) and becomes of record only if it clears the 350 clean-correct floor (01-SIZING.md I.3; n 570 was sized so 350 is 61.5% of it, not a rate to be rescaled) and the curves arm scores -- a measurement, not a mode toggle, so it cannot lift today. |
| `openai/gpt-oss-20b` | R11 | Unservable under `VLLM_BATCH_INVARIANT=1` on any card class this account can reach in vLLM 0.28.0 (no MXFP4 MoE kernel is also batch-invariant); reproducible only at concurrency 1, where forced continuations return empty under the harmony format anyway. R11 suspends it from the powered sweep as a SUBJECT entirely; it stays a JUDGE, served flag-off. Nothing about R12 revisits R11, so nothing about R12 lifts this hold. |

## The Phi-4-reasoning exploratory off cell at n 570

**Already run, not a command to issue again.** `DECISION-LOG.md` 2026-09-08 09:26:
submitted as job 828564 (RUNNING as of this lane's read-only check), from
`bcf/waves/explore-phi4-off-01.tsv` (committed `cc3e7c8`), via `bcf/wave.sh` directly
("on purpose", bypassing the hold list, which does not apply to an explicitly
exploratory row per that same log entry) -- the same command shape as the one below,
reconstructed here from the manifest's own content for the record:

    microsoft/Phi-4-reasoning	arc_challenge	stated-hint	BCF_REVISION=1de18ec97600877ce63dbf60c73b998da99f0195	BCF_N_ITEMS=570	BCF_CURVE_CAP=570	BCF_ARMS=replay placebo direct twostep filler curves transplant anchor specificity	BCF_TP=1	BCF_CONCURRENCY=32	BCF_BATCH_INVARIANT=1	MEM=64G	CPUS=8	BCF_EXPECTED_HOURS=0.564	BCF_EXTRA_VLLM=--max-num-seqs 64	BCF_REASONING_MODE=off	BCF_OUT_SUBROOT=explore-off

    bash bcf/wave.sh --type sweep --gpu-type a100-40 bcf/waves/explore-phi4-off-01.tsv

`BCF_OUT_SUBROOT=explore-off` keeps its output out of the wave-1 cell-of-record's
directory, and being the row's true last field is what saves `BCF_REASONING_MODE` from
Step 0's bug on THIS row specifically -- coincidentally, since it was placed there for
its own reason, not as a workaround. The `explore-` prefix keeps this row out of the
216-cell grid count (`tests/test_wave_plan.py`, per the same convention as `enrich-`,
`resub-` and `ladder-`).

**On completion, read:** clean-correct of 570 against the 350 absolute floor
(01-SIZING.md I.3), the curves arm's scorable counts, and the forced continuations'
reopened-block rate
(`reasoning_detail.continuations`). Only then does Phi-4-reasoning's row become a ruling
one way or the other; this cell result alone does not lift its hold.
