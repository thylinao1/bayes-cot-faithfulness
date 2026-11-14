# The judge gate

An LLM judge is only useful if its verdicts can be trusted, and a judge's accuracy cannot be
read off the data it is about to label. This directory holds the instrument the project uses to
decide whether a judge, or a panel of judges, may label anything at all: an exam with known
answers, pass thresholds written before the first judge ran, and a report that keeps every
denominator.

It runs on CPU for everything except the judge servers themselves. The tests need no GPU, no
key and no network:

```bash
PYTHONPATH=src pytest tests/test_jury_gate.py tests/test_panel_gate.py tests/test_jury_runner.py -q
```

## How it works

1. **An exam with known answers.** [`synthetic_gate.py`](synthetic_gate.py) builds 483 items, 69
   in each of seven classes. Every item is a real banked reasoning trace with exactly one
   controlled edit, so a positive and its matched negative differ by one inserted or removed
   sentence and nothing else. The classes cover a planted disclosure, a paraphrased one, a cue
   that is quoted and then rejected, a cue restated at the top and never used, a deleted step,
   an untouched trace, and a trace whose final answer was swapped.
2. **Pass thresholds fixed first.** [`gate_thresholds.py`](gate_thresholds.py) is 54 lines: ten
   thresholds, the reason for each, and a function that hashes the file. The thresholds were
   recorded with a timestamp before any judge was served. Every report carries that hash, so a
   later edit to a threshold shows up in the diff of any report.
3. **Pinned judges, recorded votes.** Four judges from four model families, each at a pinned
   revision ([`family_map.py`](family_map.py)). [`runner.py`](runner.py) probes each judge once
   before voting starts, keeps a transport failure apart from a malformed answer, and writes one
   record per vote ([`records.py`](records.py)) with the SHA-256 of the prompt file it was cast
   under. Temperature 0, three seeded runs, a test-retest rate, and a position swap on the
   multi-way questions.
4. **A panel rule that excludes the subject's own family.** A judge never votes on a model of
   its own family. The panel label is the majority of the available votes
   ([`aggregate.py`](aggregate.py)); a malformed answer or an abstention counts as unavailable
   and the panel size is recorded as what it was; a tie is never coin-flipped.
   [`panel_gate.py`](panel_gate.py) scores the gate on that panel label, which is the instrument
   the analysis would use.
5. **A report that cannot pass by accident.** [`gate.py`](gate.py) writes a numerator, a
   denominator and a rate for every metric, and a verdict of PASS, FAIL or INCOMPLETE. Its exit
   codes separate a result from a breakdown: 0 pass, 1 a failed threshold, 3 no votes were
   planned, 4 an infrastructure failure with no report written. Code 4 exists because a crashed
   job once wrote no report and was recorded as exit 0.

## What it found

- **Prompts a, b and c: nothing passed.** All four judges were scored on the three original Q1
  prompt files. No judge and no three-judge panel cleared the ten thresholds (Ruling R9,
  2026-09-07). The two failing metrics move against each other: on the FP8 Llama judge,
  paraphrased-disclosure recall and restated-cue specificity are (0.00, 1.00) under prompt a,
  (0.70, 0.38) under b and (0.25, 0.25) under c, against bars of 0.85 and 0.70. No wording
  bought both.
- **The same prompt, read in opposite directions.** Under the revised prompt d, FP8 Llama
  scores restated-cue specificity 68/69 and paraphrased-disclosure recall 0/69; Gemma-3-27B-it
  scores 0/69 and 51/69. The disagreement follows the judge, not the prompt.
- **One configuration passed.** Qwen3-32B on prompt d at 1,024 tokens clears all ten thresholds
  on its row of record: planted mention 66/68, paraphrased disclosure 61/69, quoted and denied
  68/69, clean 69/69, deleted step 69/69, restated cue 69/69, gate accuracy 69/69 and 67/68,
  malformed 12 of 5,313 votes, test-retest 464/483. An exploratory row on different hardware
  agrees on every verdict.
- **No panel passed.** The pre-registered majority panel fails the gate on every Q1 file in
  every composition, because the majority outvotes the one judge that reads paraphrases. So no
  primary jury configuration is frozen, the human calibration labels stay sealed, and the
  project's column of record stays the uncorrected regex share. The passing single judge is
  reported as a secondary configuration and is never selected on (Ruling R15).
- **No threshold moved.** The thresholds file is byte-identical in every report.

The full record, with every run as one table row, is
[`GATE-Q1-COMPARISON.md`](GATE-Q1-COMPARISON.md). The prompt-d runs are in
[`../../docs/JURY-Q1D-GATE.md`](../../docs/JURY-Q1D-GATE.md), and the failure analysis behind
Ruling R9 is in [`../../docs/JURY-Q1-FAILURE-ANALYSIS.md`](../../docs/JURY-Q1-FAILURE-ANALYSIS.md).

## Files

| File | What it does |
|---|---|
| [`gate_thresholds.py`](gate_thresholds.py) | The ten pass thresholds, their reasons, and the file's own hash |
| [`synthetic_gate.py`](synthetic_gate.py) | The known-truth corpus: seven classes, frozen templates, one controlled edit per item |
| [`gate.py`](gate.py) | Runs the gate for one or more judges and writes the report |
| [`panel_gate.py`](panel_gate.py) | The gate scored on the panel label rather than one judge at a time |
| [`runner.py`](runner.py) | Scores banked transcripts with the routed panel; availability probe, retries, locked writes |
| [`aggregate.py`](aggregate.py) | Majority of available votes, the tie rule, test-retest |
| [`family_map.py`](family_map.py) | The 18-model roster, the four judges, and the routing rule |
| [`records.py`](records.py) | The per-vote record schema and the results-path assertion |
| [`prompt_files.py`](prompt_files.py) | Loads and validates the dated prompt files in [`prompts/`](prompts/) |
| [`backends.py`](backends.py) | The pinned self-hosted vLLM endpoint and the fallback |
| [`sweep_items.py`](sweep_items.py) | Turns a sweep cell's transcripts into jury items, counting every record it has to skip |
| [`gate_compare.py`](gate_compare.py), [`gate_matrix.py`](gate_matrix.py) | Put gate runs side by side: every threshold, class and phrasing |
| [`recompute_report.py`](recompute_report.py), [`recount_gate_q1.py`](recount_gate_q1.py) | Rebuild a report from the vote file alone, as an independent check |
| [`echo_strip.py`](echo_strip.py), [`project_echo_strip.py`](project_echo_strip.py) | An exploratory input transform, measured and found not to rescue any prompt |
| [`manipulations.py`](manipulations.py) | The frozen adversarial manipulation set and its dev and held-out split |
| [`budget.py`](budget.py) | Recomputes the judging budget from measured votes per second |

Tests: `tests/test_jury_gate.py`, `test_panel_gate.py`, `test_panel_own_family.py`,
`test_jury_runner.py`, `test_jury_prompts.py`, `test_jury_family_map.py`,
`test_jury_sweep_items.py`, `test_jury_manipulations.py`, `test_echo_strip.py`.

## What this is not

It is not a validated judge. The gate is synthetic: passing it shows that a judge can find a
disclosure that was planted on purpose, which is necessary and not sufficient. The
pre-registration requires two human raters on a calibration set before any judge-corrected
number is published, and no model in this directory is a rater.
