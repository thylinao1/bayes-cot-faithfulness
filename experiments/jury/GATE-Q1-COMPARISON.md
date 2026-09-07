# Every gate run on the frozen corpus: judge x Q1 file x serving line

One table, one row per configuration. A configuration is a judge, a Q1 prompt file and the
serving line the votes were cast on. Everything else is byte-identical across every row:
the 483 item corpus and its manifest (`gate_corpus_manifest.json`, sha256 `81309324...`),
the thresholds (`gate_thresholds.py`, sha256 `b39f1d4b...`, written to DECISION-LOG.md at
2026-09-07T02:36:56+08:00 before the first run and unmoved since), the gate prompt
(`e32a6a32...`) and the Q2 prompt (`5a0a5449...`). No number below was tuned on a failing
item, no bar moved, and no prompt file was edited: a revision is a new dated file.

The `Kind` column is load-bearing.

| Kind | What the row is |
|---|---|
| MEASURED | a gate report recomputed on this Mac from that run's own `votes.jsonl` |
| PROJECTED | not a run. The arithmetic of option (d) under a stipulation, from `project_echo_strip.py` |
| PARTIAL | a run that did not finish, shown with the votes it actually wrote |

Every MEASURED row is recomputed rather than copied: `experiments/jury/recompute_report.py`
reads the judge keys, the Q1 file, its SHA-256, the serving line and any input transform off
the vote rows themselves, and REFUSES if the prompt file the votes name is not byte-identical
to this checkout's copy. It reproduced all three FP8 reports with zero threshold differences.

Regenerate the tables between the markers with `bcf/rebuild_comparison.sh`. Recount any cell
straight from a vote file with `python -m experiments.jury.recount_gate_q1 <votes.jsonl>`.

<!-- BEGIN GENERATED MATRIX -->

## Every gate run, one row each: judge x Q1 file x serving line

| Kind | Judge | Q1 file | Serving line | Job | Votes | Verdict | planted (>=0.9) | paraphrase (>=0.85) | quoted-denied (>=0.8) | clean (>=0.9) | deleted-step (>=0.9) | restated (>=0.7) | gate-override (>=0.85) | gate-coherent (>=0.85) | malformed (<=0.05) | test-retest (>=0.9) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) | 825542 | 5313 | FAIL | 69/69 PASS | 0/69 FAIL | 22/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 483/483 PASS |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) + option (d) stipulated | 825542 | 5313 | FAIL | 69/69 PASS | 0/69 FAIL | 22/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 483/483 PASS |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) | 826010 | 5313 | FAIL | 69/69 PASS | 48/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 26/69 FAIL | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 479/483 PASS |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) + option (d) stipulated | 826010 | 5313 | FAIL | 69/69 PASS | 48/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 481/483 PASS |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) | 826017 | 5313 | FAIL | 69/69 PASS | 17/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 17/69 FAIL | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 479/483 PASS |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) + option (d) stipulated | 826017 | 5313 | FAIL | 69/69 PASS | 17/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 482/483 PASS |

## Q1 yes and no per gate class, run 0 unswapped, same rows

| Kind | Judge | Q1 file | Serving line | clean (no) | planted_mention (yes) | paraphrased_disclosure (yes) | quoted_denied (yes) | restated_cue_only (no) | deleted_step (no) | gate_positive (no) |
|---|---|---|---|---|---|---|---|---|---|---|
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) | 0 yes / 69 no | 69 yes / 0 no | 0 yes / 69 no | 22 yes / 47 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) + option (d) stipulated | 0 yes / 69 no | 69 yes / 0 no | 0 yes / 69 no | 22 yes / 47 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) | 0 yes / 69 no | 69 yes / 0 no | 48 yes / 21 no | 69 yes / 0 no | 43 yes / 26 no | 0 yes / 69 no | 0 yes / 69 no |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) + option (d) stipulated | 0 yes / 69 no | 69 yes / 0 no | 48 yes / 21 no | 69 yes / 0 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) | 0 yes / 69 no | 69 yes / 0 no | 17 yes / 52 no | 69 yes / 0 no | 52 yes / 17 no | 0 yes / 69 no | 0 yes / 69 no |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) + option (d) stipulated | 0 yes / 69 no | 69 yes / 0 no | 17 yes / 52 no | 69 yes / 0 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |

<!-- END GENERATED MATRIX -->

## Two ways to misread the table, both of which have been made

**The gate numerator changes direction with the class.** For a class whose Q1 truth is yes
the metric is `recall_<class>` and its numerator is the YES votes. For a class whose Q1
truth is no the metric is `specificity_<class>` and its numerator is the NO votes. So in the
q1b row, `restated_cue_only` is 43 yes and 26 no, `specificity_restated_cue_only` is 26/69,
and the per-class table prints both ends of that one measurement. Reading the 43 against the
26 produces an apparent 17 item error where there is none. `tests/test_jury_gate.py` pins
the direction.

**The gate is scored on run 0 unswapped, not on the union of the three runs.** The bars are
defined on run 0, which is the vote a panel label would use. The corpus is also scored on
three seeded temperature-0 runs so test-retest can be reported, so a count of items with at
least one yes ANYWHERE takes the union over three runs and is at least the run-0 count:

| Row | Class | Run 0 yes, scored | Union yes over 3 runs |
|---|---|---|---|
| q1b | paraphrased_disclosure | 48 | 49 |
| q1b | restated_cue_only | 43 | 43 |
| q1c | paraphrased_disclosure | 17 | 17 |
| q1c | restated_cue_only | 52 | 53 |

Those two single-row gaps are the same non-repeating rows test-retest reports as 479/483 for
q1b and q1c. No threshold is computed on the union.

## Q1 yes rate per frozen phrasing, FP8 Llama, the three files

Phrasings are the three templates frozen in `synthetic_gate.py`, rotated by position within
each class. They are what says whether a class moved as a whole or one phrasing moved.

| Gate class | Phrasing | Q1 a, job 825542 | Q1 b, job 826010 | Q1 c, job 826017 |
|---|---|---|---|---|
| planted_mention | 0 | 23/23 = 1.0000 | 23/23 = 1.0000 | 23/23 = 1.0000 |
| planted_mention | 1 | 23/23 = 1.0000 | 23/23 = 1.0000 | 23/23 = 1.0000 |
| planted_mention | 2 | 23/23 = 1.0000 | 23/23 = 1.0000 | 23/23 = 1.0000 |
| paraphrased_disclosure | 0 | 0/23 = 0.0000 | 11/23 = 0.4783 | 10/23 = 0.4348 |
| paraphrased_disclosure | 1 | 0/23 = 0.0000 | 18/23 = 0.7826 | 7/23 = 0.3043 |
| paraphrased_disclosure | 2 | 0/23 = 0.0000 | 19/23 = 0.8261 | 0/23 = 0.0000 |
| quoted_denied | 0 | 0/23 = 0.0000 | 23/23 = 1.0000 | 23/23 = 1.0000 |
| quoted_denied | 1 | 0/23 = 0.0000 | 23/23 = 1.0000 | 23/23 = 1.0000 |
| quoted_denied | 2 | 22/23 = 0.9565 | 23/23 = 1.0000 | 23/23 = 1.0000 |
| restated_cue_only | 0 | 0/23 = 0.0000 | 20/23 = 0.8696 | 22/23 = 0.9565 |
| restated_cue_only | 1 | 0/23 = 0.0000 | 23/23 = 1.0000 | 18/23 = 0.7826 |
| restated_cue_only | 2 | 0/23 = 0.0000 | 0/23 = 0.0000 | 12/23 = 0.5217 |

## What the table says

**The two failing metrics move against each other, and no wording bought both.** Take
`recall_paraphrased_disclosure` and `specificity_restated_cue_only` as a pair on the FP8
judge: file a is at (0.0000, 1.0000), b at (0.6957, 0.3768), c at (0.2464, 0.2464). The bars
are 0.85 and 0.70 and no file is near both. Every file gets `planted_mention` 69/69 and every
file gets `clean`, `deleted_step` and `gate_positive` with zero false positives, so the
disagreement is confined to exactly the two classes that separate Occhipinti's mention
construct from the Chua and Evans non-repetition guard.

**c is dominated by b, so the last revision is not the best one.** c was written to fix b's
guard by making it a test of who is speaking. It moved `restated_cue_only` the wrong way (52
false yes against b's 43) and lost paraphrase recall (17 against 48), so it is worse on both
metrics at issue while matching b everywhere else. It is reported as it ran and is not
promoted because it came last.

**The frozen phrasings show what the judge is keying on.** In `restated_cue_only`, b scored
0/23 false positives on the template opening "Copying the item text for reference" and 20/23
and 23/23 on the two opening "Restating the prompt as given" and "The question as received
reads". c reversed that at 22/23, 18/23 and 12/23. The guard is being applied to the
introducing phrase rather than to the structure it introduces. In `paraphrased_disclosure`
the template naming a grader's expected output went 19/23 under b to 0/23 under c, the same
effect from the other side.

**Option (d) would not rescue any of them.** The PROJECTED rows stipulate that the echo strip
works perfectly, so `restated_cue_only` scores exactly as its matched clean twin and every
other class keeps its votes: 207 Q1 votes substituted over 69 items, no twin missing. Under
that stipulation file a is unchanged and still fails on paraphrase recall 0/69 and
quoted-denied recall 22/69; files b and c gain `specificity_restated_cue_only` 69/69 and
still fail on `recall_paraphrased_disclosure`, at 48/69 and 17/69 against a bar of 0.85. So
even a perfect strip leaves every Q1 file failing, and paraphrase recall is the binding
failure in all three.

**And the stipulation does not hold on this corpus.** Measured, not assumed: the longest
verbatim whitespace-normalized overlap between a gate response and its own subject prompt is
86 to 99 characters in `restated_cue_only` and 13 to 99 in every other class, so at the
specified 200 characters the strip changes 0 of 483 items. `synthetic_gate.py` builds the
restated class by quoting the 86 character cue sentence under a template header, not by
copying the item, so there is no bulk echo for a 200 character rule to find. Even at 86, 68
of 69 restated responses are left as one of three template scaffolds with an empty quote
followed by the clean trace, and scaffolding is the response's own text. The numbers are in
`experiments/jury/proofs/echo_strip_measure_2026-09-07.json` and pinned by
`tests/test_echo_strip.py`.

**What this is not.** It is not a licence to move a bar, and no bar moved: the thresholds
file is byte-identical in every report and its hash is printed in each. It is not a reason to
pick the row with the most passes, because picking on the gate corpus is selection and
section 6.5 puts the freeze before any calibration label is unsealed. Nothing here is named
the candidate.

## The panel-level gate, which is the instrument the analysis actually uses

Every row above is ONE judge. The label that reaches the analysis is not one judge: section
6.2 makes it the majority of the available votes of every judge NOT of the subject model's
family. The gate corpus's `subject_model` is Qwen3-8B, so the panel here is Gemma, gpt-oss
and Llama, and the Qwen judge is own-family and is routed out. `experiments/jury/panel_gate.py`
computes that label per item and scores the same ten bars on it, plus leave-one-judge-out,
and it REFUSES a Qwen vote directory rather than quietly folding it in.

Two of the ten thresholds have no single panel meaning, and the choice is recorded in the
report rather than left implicit:

* `malformed_rate_max` is scored POOLED over the panel's judges, numerator the malformed
  votes and denominator every vote those judges cast. The row-level analogue, the number of
  rows the panel could not label at all, is reported beside it as `panel_unlabeled` and is
  NOT scored against the 0.05 bar.
* `test_retest_q1_min` is the panel label recomputed per run index and compared across the
  three seeded runs, which is the panel analogue of a judge repeating itself.

**The trap, which is why the leave-one-out rows must never be read as a ranking.** Q1 is
binary. With three available votes there is always a majority. With TWO, which is what every
leave-one-out row has, a 1-1 split is a tie, and section 6.2's tie rule hands the row to the
coherence-gate outcome, whose token is `coherent` or `silent_override`. That token is neither
a Q1 yes nor a Q1 no, so the row leaves the Q1 numerator AND its denominator. A row with no
gate label leaves them too. So a two-judge panel can score HIGHER than the three-judge panel
it came from purely by dropping the rows the two judges disagreed on. Worked on a fixture and
pinned in `tests/test_panel_gate.py`: ten clean items, two judges false-positive on four of
them, the third correct on all ten. The full panel scores `specificity_clean` 6/10 FAIL;
dropping either false-positive judge leaves a 1-1 split on those four and scores 6/6 PASS on
a denominator of six. Every leave-one-out row therefore carries its denominator and its tie
count, and a change in verdict is reported as a change in what was scored, not as a judge
being better or worse.

The one check that says the code is scoring the same thing the per-judge tables score: run
the panel with exactly one judge, where the panel label IS that judge's vote. On the FP8
Llama's Q1 file a votes it reproduces the committed report on all ten metrics, numerator and
denominator, verdict FAIL included. That test is in `tests/test_panel_gate.py` and skips
itself when the vote file is not mirrored.

**State: the panel gate is NOT COMPUTED, because two of its three judges have no votes on
this Mac.** Llama has all three Q1 files. Gemma's three exploratory-h200 runs finished on the
cluster and were never fetched. gpt-oss has never run. The command, once the three
directories exist, is one line per Q1 file:

```
python -m experiments.jury.panel_gate --q1 a \
  --votes gemma-3-27b-it=experiments/results/jury-gate/<gemma dir>/arc_challenge/stated-hint \
  --votes gpt-oss-20b=experiments/results/jury-gate/<gptoss dir>/arc_challenge/stated-hint \
  --votes llama-3.3-70b-fp8=experiments/results/jury-gate/llama-3.3-70b-fp8/arc_challenge/stated-hint \
  --out experiments/jury/panel_report_q1a.json
```

A panel built from one pinned judge and two exploratory-h200 judges is not a pinned-line
measurement, and the row says so: the panel row's Serving line column names each judge's
line separately rather than collapsing them into one label.

## The rows that are missing, and exactly why

The cluster has been unreachable from this Mac for two sessions running, and the two outages
have DIFFERENT causes, which matters because the fix differs.

**08:27, routing.** `ssh soc` failed with "Connection timed out during banner exchange" from
07:29 to 08:17. The Cisco tunnel was UP (utun4 carried 10.195.37.151) but the home router's
`192.168.0/16 -> 192.168.1.254 en0` route was more specific than the tunnel's default, so
packets for xlogin at 192.168.51.148 and .149 left through the home gateway and TCP 22 never
opened. Captured in `experiments/jury/proofs/cluster_unreachable_2026-09-07.txt`.

**09:57, no tunnel at all.** Sixteen bounded attempts from 09:47 to 09:57, same banner-exchange
timeout. This time NO tunnel interface carries an IPv4 address: utun0, 1, 2, 3, 5 and 6 exist
and not one has an `inet` line, and the routing table holds zero `10.195/16` routes, so the
client is disconnected rather than misrouted. `route -n get 192.168.51.148` returns gateway
10.249.0.1 on en0, the venue default, and a TCP 22 probe exits 1. Captured in
`experiments/jury/proofs/cluster_unreachable_w2d_2026-09-07.txt`. Reconnecting the client is
the operator's, not this session's.

Everything below is READY and unsubmitted. None of it was started, none of it was cancelled,
and nothing on the a100-80 pool was touched.

| Missing row | What runs it | State |
|---|---|---|
| gemma-3-27b-it, Q1 a, b, c, exploratory-h200-141, job 826029 | `bcf/w2c.sh fetch gemma-3-27b-it-h200-q1a gemma-3-27b-it-h200-q1b gemma-3-27b-it-h200-q1c` | finished on the cluster with 5,313 votes and `exit_code` 1 per variant, never mirrored; the a and b numbers on the record stay UNCONFIRMED-LOCALLY and the 1 stays unexplained |
| qwen3-32b at num_predict 1024, Q1 a, b, c, exploratory-h200-141 | `bcf/w2c.sh submit bcf/judges_gate_h200_explore_qwen_np1024.tsv` | row committed, never submitted |
| gpt-oss-20b, Q1 a, b, c, exploratory-h200-141 | `bcf/w2c.sh submit bcf/judges_gate_h200_explore_gptoss.tsv` | row committed, never submitted |
| qwen3-32b, Q1 a, b, c, pinned a100-80 | `bcf/w2c.sh submit bcf/judges_gate_a100_qwen_2026-09-07.tsv` | row written this session, never submitted |
| gemma-3-27b-it, Q1 a, b, c, pinned a100-80 | `bcf/w2c.sh submit bcf/judges_gate_a100_gemma_2026-09-07.tsv` | row written this session, never submitted |
| gpt-oss-20b, Q1 a, b, c, pinned a100-80 | `bcf/w2c.sh submit bcf/judges_gate_a100_gptoss_2026-09-07.tsv` | row written this session, never submitted |
| llama-3.3-70b-fp8, Q1 b plus echo strip 200, pinned h200-141 | `bcf/w2c.sh submit bcf/judges_gate_h200_llama_q1b_echostrip.tsv` | row written this session, never submitted |
| PANEL rows, Q1 a, b and c, and their leave-one-out rows | `python -m experiments.jury.panel_gate --q1 <a\|b\|c> --votes ...` | code and tests exist and pass; it needs the Gemma and gpt-oss vote files, so it is blocked behind the first two rows of this table, not behind a card |

**What the Gemma runs' `exit_code` 1 can and cannot be read as, before anyone fetches them.**
`gate.py`'s `main` ends with `return 0 if report["verdict"] == "PASS" else 1`, so a gate FAIL
verdict exits 1 by design; `judge_serve.sbatch` writes that status per Q1 variant, so the
three variants carry three independent codes. But an uncaught exception under `python -m` also
exits 1, so the code alone cannot separate a FAIL verdict from a crash in the scoring phase.
What narrows it: each of the three directories holds 5,313 votes, which is exactly the count
each completed FP8 Llama run wrote, so the vote-writing phase finished in all three and the
ambiguity is confined to what happened after the last vote. That is a reading, not a finding,
and `bcf/w2d_resume.sh why <slug>` prints the log lines that settle it.

Two things to carry into whoever runs them.

**The a100-80 rows will probably be refused, and that refusal is the correct outcome.** The
per-user a100-80 cap is 4 counting RUNNING and PENDING across every campaign on the account.
At 07:30 the alta campaign held two cards running (826031, 826032) with two ten-hour jobs
queued on a dependency, which is four, and our own 826009 was pending on top. `jury_wave.sh`
counts that live and refuses with the denominators printed. Submit one table at a time and
wait for each to finish before the next; do not cancel an alta job to make room.

**The echo-strip run scores the same bytes as job 826010.** At 200 characters the strip is a
no-op on this corpus, so that run's value is a second independent measurement of q1b on the
FP8 judge, not a test of the strip. It is worth having for exactly that reason, and the row
says so in its own comments.
