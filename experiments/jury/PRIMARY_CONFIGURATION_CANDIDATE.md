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


## The Q1 prompt is NOT settled (added 2026-09-07 by W2b)

The Q1 row in the prompt table above still names `q1_mention_2026-09-07.md`, because that
is the file of record and no replacement has earned the row. Three Q1 files have now been
scored on the frozen gate corpus with the FP8 70B judge, under the thresholds written to
DECISION-LOG.md at 02:36:56 and unchanged since. ALL THREE FAIL, each on two of the ten.

| Q1 file | SHA-256 | Job | Failed thresholds |
|---|---|---|---|
| `q1_mention_2026-09-07.md` | `c11fa9cd...` | 825542 | recall_paraphrased_disclosure 0/69, recall_quoted_denied 22/69 |
| `q1_mention_2026-09-07b.md` | `cdbba6e3...` | 826010 | recall_paraphrased_disclosure 48/69, specificity_restated_cue_only 26/69 |
| `q1_mention_2026-09-07c.md` | `f6739406...` | 826017 | recall_paraphrased_disclosure 17/69, specificity_restated_cue_only 17/69 |

No file is named the candidate primary Q1 prompt. Selecting one on these numbers would be
selection on the gate corpus, which is the thing section 6.5's freeze-before-unsealing rule
exists to prevent, and none of the three clears the bars anyway. The full side by side,
with every threshold, every class and every frozen phrasing, is in
`GATE-Q1-COMPARISON.md`, and the reading of it is in that file rather than here.

What the operator has to decide, stated as options and not as a recommendation: whether Q1
becomes two calls whose conjunction is the label, whether the gate corpus's
`restated_cue_only` class needs more than three introducing phrases so the guard cannot be
answered from the phrase, or whether the construct of record is the narrower one that file
a implements, in which case the paraphrase and quoted-denied classes are what change. Each
needs a human label to settle and none is settled here.


## Which gate runs have a local artifact, and which do not (added 2026-09-07 by W2b)

A check of this branch read the directory `experiments/results/jury-gate/gemma-gptoss-h200/`
as the Gemma judge's gate result, found exit code 5 and no `votes.jsonl`, and reported that
the Gemma numbers on the record were votes that never happened. That directory is a
DIFFERENT job. It is job 826026, the section 6.1 co-hosted Gemma-plus-gpt-oss pair, and its
exit 5 is itself a finding already on the record: the pair cannot start, because vLLM's
`--gpu-memory-utilization` is a fraction of the whole card measured against what other
processes already hold. It was never a source of any number.

The map below is what actually exists under `experiments/results/jury-gate/` on the Mac.
The results tree is gitignored derived data, so this table is committed instead. Anything
marked NOT MIRRORED is a number that was read off the cluster before the VPN dropped at
about 05:00 and CANNOT be recomputed from this repository until it is fetched.

| Directory | Job | Judge | Votes present | Exit | What it is |
|---|---|---|---|---|---|
| `llama-3.3-70b-fp8/` | 825542 | llama-3.3-70b-fp8 | 5,313 | 0 | Q1 variant a, complete, scored |
| `llama-3.3-70b-fp8-q1b/` | 826010 | llama-3.3-70b-fp8 | 5,313 | 0 | Q1 variant b, complete, scored |
| `llama-3.3-70b-fp8-q1c/` | 826017 | llama-3.3-70b-fp8 | 5,313 | 0 | Q1 variant c, complete, scored |
| `qwen3-32b-h200/` | 826023 | qwen3-32b | 0 | 0 | job wrapper log only; the votes are in the q1a directory |
| `qwen3-32b-h200-q1a/` | 826023 | qwen3-32b | 342 | none | PARTIAL, cancelled by explicit id mid-variant |
| `gemma-gptoss-h200/` | 826026 | none served | 0 | 5 | the co-hosted pair that cannot start; NOT a Gemma result |
| `gemma-3-27b-it-h200-q1{a,b,c}/` | 826029 | gemma-3-27b-it | 5,313 each | 1 each | MIRRORED and recomputed 2026-09-07 13:53; the 1 is a gate FAIL verdict, see the W2f section |

Two consequences that a reader of the record should carry.

**The Gemma numbers were not verifiable from this repository until 13:53 on 2026-09-07. RESOLVED; the account of the outage below is kept as history.**
Job 826029's Q1 variant a and variant b results, including the finding that the FP8 Llama and
Gemma read the same Q1 bytes in opposite directions on `quoted_denied` and
`restated_cue_only`, were read off the cluster at about 05:05 and the files were never
rsynced. The 07:29 attempt to fetch them failed the same way: `ssh soc` timed out during
banner exchange for the whole of the 07:29 to 08:17 session, with one connect that opened
and closed at 08:02. The cause is captured in
`experiments/jury/proofs/cluster_unreachable_2026-09-07.txt` and it is routing, not the
cluster: the Cisco tunnel is up on utun4 with 10.195.37.151, but the home router's
`192.168.0/16 -> 192.168.1.254 en0` route is more specific than the tunnel's default, so
packets for xlogin at 192.168.51.148 and .149 leave through the home gateway and TCP 22
never opens. Reconnecting the VPN client is the operator's.

RESOLVED at 13:53 on 2026-09-07 by W2f: all three directories are mirrored and every
number is recomputed from its own vote file, with zero differences against the reports the
cluster wrote. The mark is LIFTED; the recomputed tables are in the closing W2f section of
this file. The fetch was one command, `bcf/w2c.sh fetch gemma-3-27b-it-h200-q1a
gemma-3-27b-it-h200-q1b gemma-3-27b-it-h200-q1c`, which rsyncs the artifacts and then
recomputes each report from the vote file with `experiments/jury/recompute_report.py`. That
recomputation reads the judge key, the Q1 file, its SHA-256, the serving line and any input
transform off the votes themselves and REFUSES if the prompt file the votes name is not
byte-identical to this checkout's copy, so the recomputed report is a check and not a copy.
It reproduced all three FP8 reports with zero threshold differences.

**The 342 row Qwen file is a cancelled partial whose wrapper wrote exit 0, and that defect
is now FIXED.** Job 826023 was cancelled by explicit id at 04:22:10 partway through Q1
variant a, having planned 15,939 votes and written 342. The `[done] exit_code=0` line went to
`qwen3-32b/exit_code.txt` anyway, so a directory holding 2 percent of its planned votes
carried a success code.

`bcf/exit_guard.sh` closes it three ways: `exit_code.txt` is written 255, meaning started and
not finished, the moment a run starts; TERM, INT, HUP and USR1 become 128 plus the signal
number; and an exit of 0 that never reached the completion marker becomes 250. The same rule
applies per Q1 variant, so a job cancelled inside variant c leaves c non-zero and leaves a
and b alone. `bcf/test_exit_guard.sh` proves it in 13 scenarios with no GPU and no Slurm,
including the real `judge_serve.sbatch` cancelled mid-variant recording 143, and the same
cancel replayed against the pre-fix logic recording 0.

The `0` sitting in job 826023's directory predates the fix and is left there. Nothing on the
record is computed from it beyond the malformed rate that motivated the resubmission at
`num_predict` 1024.


## Option (d), the deterministic echo strip, is implemented and does not bite (added 2026-09-07 by W2c)

`experiments/jury/echo_strip.py` (sha256 `e457869a...`) removes from a response every
contiguous span of at least 200 characters that appears verbatim in the prompt the response
was produced from, matched on whitespace-normalized text and cut out of the original bytes.
It is an EXPLORATORY configuration and nothing about it is named the candidate. Its
parameters and its own SHA-256 land on every vote as `input_transform` and in the report, so
a stripped run can never be read as an unstripped one, and `--echo-strip` defaults to off,
which is what every run of record used.

MEASURED on the frozen corpus: the longest verbatim whitespace-normalized overlap between a
response and its own subject prompt is 86 to 99 characters in `restated_cue_only` and 13 to
99 in every other class, so at 200 the strip changes 0 of 483 items. The restated class
quotes the 86 character cue sentence under a template header rather than copying the item, so
there is no bulk echo for a 200 character rule to find. Even at 86, 68 of 69 restated
responses reduce to one of three template scaffolds with an empty quote followed by the clean
trace, and 1 differs further because its own clean body reproduces 99 characters of the
question. A strip that also removed the scaffolding would be a rewrite.

The OFFLINE projection in `project_echo_strip.py`, computed under the stipulation that the
strip works perfectly, says option (d) would not rescue any Q1 file: `recall_paraphrased_
disclosure` is the binding failure in all three, at 0/69, 48/69 and 17/69 against a bar of
0.85. The full table, with every PROJECTED row marked, is in `GATE-Q1-COMPARISON.md`.


## The Gemma mark STILL stands, and the panel gate now exists to receive it (added 2026-09-07 by W2d)

(HISTORY, written by W2d at about 10:00; the mark it describes was LIFTED at 13:53 by W2f.)
The UNCONFIRMED-LOCALLY mark above was unchanged at the time this section was written. The three Gemma exploratory runs
(`gemma-3-27b-it-h200-q1a`, `-q1b`, `-q1c`, job 826029) finished on the cluster with 5,313
votes each and an `exit_code` of 1, and that 1 is NOT yet explained: `experiments/jury/gate.py`
returns `0 if report["verdict"] == "PASS" else 1`, so a gate FAIL verdict and several kinds of
failure all leave a 1 behind, and only the run log separates them. The log is on the cluster.
Until the fetch happens, the Gemma numbers quoted on the record stay exactly as they were
reported, unrecomputed here, and the exit code stays unexplained rather than assumed benign.

The blocker is the VPN again and it is a DIFFERENT fault from the 08:27 one. Twenty bounded
`ssh soc` attempts between 09:47 and 09:59 all timed out during banner exchange. This time no
tunnel interface carries an IPv4 address at all and the routing table holds zero `10.195/16`
routes, so the Cisco client is disconnected rather than shadowed by a more specific home
route; `route -n get 192.168.51.148` returns the venue default gateway 10.249.0.1 on en0 and a
TCP 22 probe exits 1. Evidence:
`experiments/jury/proofs/cluster_unreachable_w2d_2026-09-07.txt`. Reconnecting the client is
the operator's.

What DID get built while it was down, so the fetch is the only remaining step for the analysis
the Q1 ruling needs: `experiments/jury/panel_gate.py` scores the ten thresholds on the SECTION
6.2 PANEL LABEL rather than on one judge, with leave-one-judge-out, and it refuses the Qwen
judge by name because Qwen is the gate corpus's own subject family. It is verified against the
record two ways: run with one judge it reproduces the committed FP8 Llama report on all ten
metrics exactly, and a partial panel is marked `PANEL-PARTIAL`, is not written to disk without
`--allow-partial`, and can never enter a table looking like the panel of record. The reading
rule that matters for the operator's Q1 decision is in `GATE-Q1-COMPARISON.md`: a two-judge
leave-one-out row can score HIGHER than the three-judge panel it came from, because a 1-1 tie
resolves to a gate token that is neither a Q1 yes nor a Q1 no and the row leaves the
denominator. Ordered resume: `bcf/w2d_resume.sh`.


## Still no candidate, and the Gemma mark still stands (added 2026-09-07 by W2e)

(HISTORY, written by W2e at about 13:30; the mark it describes was LIFTED at 13:53 by W2f.)
Nothing in this file changes. No Q1 file is named. No judge is named. The Gemma numbers
stayed UNCONFIRMED-LOCALLY with their `exit_code` 1 unexplained, for the same reason as
before and for a new instance of it: the cluster was unreachable for the whole of this
lane, so the fetch that would settle both did not happen.

The unreachability is a THIRD distinct fault and it matters that it is, because the fix
differs. The Cisco Secure Client's own `vpn state` answers `Disconnected / Ready to
connect`; only `lo0` and `en0` carry an IPv4 address, there are zero `10.195/16` routes,
and TCP 22 is closed on xlogin's `192.168.51.148` and `.149` AND on the
`stujump.comp.nus.edu.sg` fallback the ssh config uses when the direct route fails. The
08:27 fault was a route to add and the 09:57 fault was a tunnel with no address; this one
is a client that is not connected, and connecting it needs credentials this session does
not have. Evidence, including the client's state line:
`experiments/jury/proofs/cluster_unreachable_w2e_2026-09-07.txt`.

Two things did get built or verified while it was down, and neither names a candidate.

**The panel code's reproduction check now covers all three Q1 files, not just file a.**
`bcf/panel_one_judge_check.py` runs `panel_gate.py` with exactly one judge, where the
panel label is that judge's own vote, and machine-compares all ten scored metrics and the
verdict against that run's committed `gate_report.json`. Zero differences across the three
files: 30 metric comparisons and 3 verdicts. It is proven able to fail, in the same proof
file, by scoring the Q1 c panel against the Q1 b committed report, which reports two
differences. `experiments/jury/proofs/panel_one_judge_reproduction_2026-09-07.txt`.

That check is about COMMENSURABILITY and nothing else. It says a panel number can stand in
the same table as a per-judge number. It says nothing about whether any Q1 file passes,
because every run in it is kind `PANEL-PARTIAL` with one judge: Gemma and gpt-oss have no
votes here, so the three-judge panel of record is still NOT COMPUTED and no
leave-one-judge-out row exists anywhere in this repository.

**The chain-level attenuation factor ruling R4 requires is written up as absent rather than
estimated.** `docs/A4-CHAIN-LAMBDA-NOTE.md` carries the eight continuation-level cells from
job 826025 with every denominator and every byte-identical-repeat fraction, taken from that
run's own `arms_summary.json` and machine-checked against it, plus the reason the
continuation-level number is a floor for the chain-level one and the two conditions
(`run_label` `powered_pinned`, preflight PASS) any future chain-level value has to meet
before it is used. No chain-level number is guessed and no job id is invented for one.

Ordered resume for everything still open: `bcf/w2e_resume.sh`, one verb per step, each
refusing rather than guessing when `ssh soc` does not answer.


## The Gemma mark is LIFTED and its exit code is explained (added 2026-09-07 13:53 by W2f)

Still no candidate. No Q1 file is named here, no judge is named, and nothing below moves a
bar or a prompt. What changed is only that a number which could not be checked from this
repository now can be.

**The three Gemma exploratory directories are mirrored and recomputed.**
`bcf/w2e_resume.sh gemma` fetched `gemma-3-27b-it-h200-q1a`, `-q1b` and `-q1c` (job 826029)
and rebuilt each report from its own `votes.jsonl` with
`experiments/jury/recompute_report.py`, which takes the judge key, the Q1 file, its SHA-256,
the serving line and the input transform off the vote rows and refuses if the prompt file
the votes name is not byte-identical to this checkout's copy. It did not refuse. Compared
metric by metric against the `gate_report.json` the cluster wrote at 04:50, fetched
separately for the check, all three reproduce with ZERO differences on all ten metrics
(numerator, denominator, value, verdict and threshold), 5,313 vote rows each, verdict FAIL
in both. The full tables are in `GATE-Q1-COMPARISON.md`, which now carries three MEASURED
Gemma rows in its generated matrix.

**`exit_code` 1 is a gate FAIL verdict, not a crash.** `gate.py` line 292 returns
`0 if report["verdict"] == "PASS" else 1`, and all three reports read FAIL. `sacct -j 826029`
reads COMPLETED with Slurm ExitCode 0:0 over 43:20, which `judge_serve.sbatch` line 354
explains: the wrapper exits 0 when the gate RAN whatever its verdict, and the per-variant
status goes to `exit_code.txt`. Each `run_summary.json` reads `votes 5313` against
`votes_planned 5313` with `skipped_resumed 0`, so the vote loop finished. And
`bcf/exit_guard.sh` reserves 255, 250 and 128-plus-signal for the crash and cancel shapes;
the file holds 1, which its own header calls "a failed threshold, which is a result rather
than a job failure". Job 826029 wrote no `run.log` and its Slurm `.out` is not under `$HOME`
at depth 3, so `why <slug>` prints nothing for these three; the four items above settle it
without the log.

**What the mark being lifted does and does not license.** It licenses quoting the Gemma
numbers as MEASURED and recomputed on this Mac. It does not make them a pinned-line
measurement: every one of the three is `exploratory-h200-141` at the h200 serving line, the
pinned a100-80 Gemma table is a different row, and any panel built from these votes carries
each judge's serving line separately rather than one collapsed label.


## Closing state, 2026-09-07 14:16 (W2f)

Still no candidate. No Q1 file is named, no judge is named, no bar moved, no prompt file
changed, and `gate_thresholds.py` still hashes `b39f1d4b...`.

| Question the freeze needs answered | State |
|---|---|
| Do the Gemma numbers exist on this Mac and check out | YES, as of 13:53. Three MEASURED rows, recomputed from their own votes with zero differences against the cluster's reports |
| Is the Gemma `exit_code` 1 explained | YES. A gate FAIL verdict, on four pieces of evidence that are not the exit code |
| Is the three-judge panel of record computed | NO. gpt-oss has no votes anywhere yet; every panel run today is `PANEL-PARTIAL` with two judges and none was written |
| Which judge's inclusion changes a verdict | UNANSWERABLE today. Every cell of the two-judge leave-one-out is FAIL, so no inclusion changes a verdict; what changes is which metrics fail |
| Is there a pinned-line measurement for Gemma, gpt-oss or Qwen | NO. Jobs 826598, 826599 and 826600 were submitted at 13:37 and are all still `PENDING (Priority)`, estimated by Slurm to start 2026-09-10 |
| Does a chain-level lambda exist | YES, as of 13:48, from job 826596. `docs/A4-CHAIN-LAMBDA-NOTE.md` carries it with every denominator |

The one thing on this page that a reader should NOT carry forward as settled is the panel.
Three MEASURED Gemma rows make the per-judge table larger; they do not make a two-judge
panel a three-judge panel, and section 6.2's label is the panel's, not any judge's.
