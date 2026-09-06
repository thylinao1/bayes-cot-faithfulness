# Judging budget, recomputed from a measured rate

CONTRACT.md line 23 carries an ESTIMATE marked "until Phase 1". This file replaces the
rate it assumed with one measured on a real judge server, and corrects one factor in the
vote count. Nothing in CONTRACT.md is edited; this is the recomputation beside it.

## Where the measured rate comes from

- Source: `/Users/maksimsilchenko/Developer/bcf-jury/experiments/results/jury-gate/llama-3.3-70b-fp8/arc_challenge/stated-hint/gate_report.json`
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

