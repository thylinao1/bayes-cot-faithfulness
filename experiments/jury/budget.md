# Judging budget, recomputed from a measured rate

CONTRACT.md line 23 carries an ESTIMATE marked "until Phase 1". This file replaces the
rate it assumed with one measured on a real judge server, and corrects one factor in the
vote count. Nothing in CONTRACT.md is edited; this is the recomputation beside it.

## Where the measured rate comes from

- Source: `experiments/results/jury-gate/llama-3.3-70b-fp8/arc_challenge/stated-hint/gate_report.json`
- Measured votes per second per server: **9.8150**
- judge: llama-3.3-70b-fp8
- concurrency: 12
- votes: 5313
- seconds: 541.3
- items: 483

A vote is one HTTP call that came back and validated, including the calls that needed
their one same-seed retry. The denominator is the wall-clock of the scoring phase only,
not the job, so the model download and the server start are outside it.

## The vote count, with every factor

| Factor | Value | Where from |
|---|---|---|
| cells | 216 | 18 models x 3 substrates x 4 cue families |
| judged hinted transcripts per cell | 300 | CONTRACT.md line 23 |
| mean panel size | 3.3333 | (12 subjects x 3 + 6 x 4) / 18, the panel rule |
| CONTRACT votes per transcript per judge | 3 x 1.2 = 3.6 | 3 questions, 1.2 for the 10 percent audit |
| BUILT votes per transcript per judge | 5.60 | 0.9 x 5 + 0.1 x 11 (see the correction below) |

- CONTRACT vote count: 216 x 300 x 3.3333 x 3 x 1.2 = **777,600** votes
- BUILT vote count:    216 x 300 x 3.3333 x 5.60 = **1,209,600** votes
- Ratio: 1.556x the contract estimate

### The correction: the position swap is not free

CONTRACT.md line 23 says "Position swap applies to no binary question, so it adds nothing
here." That is true of Q1 and false of the instrument as a whole. The coherence gate asks
which option the reasoning implies, and Q2 asks which of three categories fits: both are
multi-way sub-questions, and section 6.4 puts a position swap on any multi-way sub-question
inside a judge prompt. The runner therefore scores a swapped pass on the gate and on Q2 on
run 0, which is 2 extra votes per transcript per judge, and refuses a swap on Q1.

So a single-run transcript costs 5 votes per judge, not 3, and an audited one costs
11 (runs 1 and 2 are unswapped). The mean is 5.60.

**The multiplier against CONTRACT.md line 23 is 1.556.** It is the whole of the
difference: the cells, the transcripts per cell, the mean panel size and the audit fraction
are line 23's own numbers, unchanged. Confirmed on the run of record, where 483 gate items
produced 5,313 votes for one judge, which is 11 per item, the audited case (11 = 4 gate +
3 Q1 + 4 Q2) because the gate scores every item in three-seeded mode.

The correction is appended to CONTRACT.md as a dated line rather than edited into line 23,
because line 23 is what the estimate said and the record of an estimate being wrong is
worth more than a tidy document.

## Wall clock under the corrected card budget

CARD BUDGET of record (CONTRACT.md line 149, live sacctmgr 2026-09-07): a100-80 = 4,
a100-40 = 8, h100-96 = 2, h100-47 = 4, h200-141 = 1, gpu total = 12, counting EVERY
campaign on the account. The four judges need 3 cards: Qwen3-32B on one a100-80,
Gemma-3-27B-it and gpt-oss-20b sharing a second a100-80 as two servers, and the FP8 70B
on the single h200-141.

| Servers up at once | Cards it needs | Wall clock at the measured rate | Server-hours |
|---|---|---|---|
| 1 | 1 | 34.2 h (1.4 days) | 34 |
| 2 | 1 a100-80 (the co-hosted pair) | 17.1 h (0.7 days) | 34 |
| 3 | 2 a100-80 | 11.4 h (0.5 days) | 34 |
| 4 | 2 a100-80 + 1 h200-141 | 8.6 h (0.4 days) | 34 |

For comparison, the contract's own assumption of 2.0 votes per second across
4 servers on its own vote count gives 27.0 h.

### The h200 judge cannot hold its server

xgpk0 is the ONLY h200-141 node on the cluster and it sits in partition `gpu` alone, not
in `gpu-long`. `gpu` caps the wall at 3 hours. So the single-server row above is not a
job for this judge. It is a sequence of jobs, and the sequence has a cost the hours column
hides. Measured, from the two runs on the record:

| Quantity | Value | Source |
|---|---|---|
| server start, weights in the persistent cache | 110 s | job 826010: serve 03:35:30, gate start 03:37:20 |
| server start, weights fetched to node scratch | 26 min | job 825542: job start 02:39, first vote 03:05:48 |
| home-usage preflight, memo cold | 123 s | job 826010 run.log, `full home scan took 123s` |
| home-usage preflight, memo warm | about 1 s | measures the judge cache only |
| wall a job may ask for in `gpu` | 02:50:00 | under the 3 h ceiling |

34.2 h of scoring at 9.815 votes per second therefore needs, for the FP8 70B judge alone:

- **at least 12 submissions** even if the server start were free (34.2 / 2.83 h of wall)
- **13 submissions** at the measured warm start (34.2 / 2.80 h of usable wall per job)
- **15 submissions** if every job refetches its weights (34.2 / 2.40 h), which is
  6.5 h of h200 time spent loading the same weights

What follows from that, none of it a change to the pre-registration:

1. `--resume` is not a convenience on this judge, it is the only way the work completes.
   The vote file IS the checkpoint: `records.completed_keys` reads it back and the runner
   skips what is already there, so a job killed at the wall loses at most the votes in
   flight. Every h200 judge row carries `BCF_RESUME=1`.
2. The judge weight cache has to stay persistent for this judge, or those submissions pay
   a 72 GB download each and the warm start becomes the cold one. It currently is: home
   stands at 433 GB of the 450 GB working ceiling and the FP8 copy is already inside the
   73 GB cache, so the net need is 0. The moment another campaign adds 17 GB to home, the
   same preflight sends this judge to node scratch. That is the intended behaviour and the
   ceiling is not to be raised to avoid it.
3. The `du -sBG $HOME` in that preflight cost 90 to 120 s per job. Over a dozen
   submissions that is a quarter of an hour of h200 time counting files. It is now split:
   the part of home outside the judge cache is memoised with a 24 hour TTL, the cache
   itself is measured fresh every job. Same number, same ceiling, about 1 s warm.
4. The other three judges are pinned to a100-80, which lives in `gpu-long` with a 72 hour
   ceiling, so none of this applies to them. It is a property of the one card.

### What the card budget actually allows

All four judge servers can be up together only when the account has 2 free a100-80 cards
AND the 1 h200-141 card at the same moment. That was not true at any point on the night of
2026-09-07: the a100-80 pool stood at 5 of 4 committed (3 alta running, 1 alta pending,
1 bcf-skel pending), so the wave refused the a100-80 judge jobs and only the h200 judge
ran. The 4-server row above is a ceiling, not a plan. The rows to budget against are 1 and 2.

### Degradation, in the pre-registered order

CONTRACT.md's ladder never cuts n per cell and never cuts the curve arm. On the judging
side the levers, in order, are: drop the swap pass to a sampled fraction rather than every
run 0 (it is a bias measurement, not a label input); cut the audit fraction below 0.10;
cut cue families per substrate. The panel rule and the three questions are not levers:
both are what K1 is computed on.

