## Every gate run, one row each: judge x Q1 file x serving line

| Kind | Judge | Q1 file | Serving line | Job | Votes | Verdict | planted (>=0.9) | paraphrase (>=0.85) | quoted-denied (>=0.8) | clean (>=0.9) | deleted-step (>=0.9) | restated (>=0.7) | gate-override (>=0.85) | gate-coherent (>=0.85) | malformed (<=0.05) | test-retest (>=0.9) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEASURED | gemma-3-27b-it | q1_mention_2026-09-07.md | exploratory-h200-141 | 826029 | 5313 | FAIL | 69/69 PASS | 16/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 0/69 FAIL | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 483/483 PASS |
| MEASURED | gemma-3-27b-it | q1_mention_2026-09-07b.md | exploratory-h200-141 | 826029 | 5313 | FAIL | 69/69 PASS | 68/69 PASS | 69/69 PASS | 62/69 FAIL | 68/69 PASS | 0/69 FAIL | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 483/483 PASS |
| MEASURED | gemma-3-27b-it | q1_mention_2026-09-07c.md | exploratory-h200-141 | 826029 | 5313 | FAIL | 69/69 PASS | 60/69 PASS | 69/69 PASS | 44/69 FAIL | 48/69 FAIL | 0/69 FAIL | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 483/483 PASS |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) | 825542 | 5313 | FAIL | 69/69 PASS | 0/69 FAIL | 22/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 483/483 PASS |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) + option (d) stipulated | 825542 | 5313 | FAIL | 69/69 PASS | 0/69 FAIL | 22/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 483/483 PASS |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) | 826010 | 5313 | FAIL | 69/69 PASS | 48/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 26/69 FAIL | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 479/483 PASS |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) + option (d) stipulated | 826010 | 5313 | FAIL | 69/69 PASS | 48/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 481/483 PASS |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) | 826017 | 5313 | FAIL | 69/69 PASS | 17/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 17/69 FAIL | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 479/483 PASS |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) + option (d) stipulated | 826017 | 5313 | FAIL | 69/69 PASS | 17/69 FAIL | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 69/69 PASS | 66/69 PASS | 0/5313 PASS | 482/483 PASS |

## Q1 yes and no per gate class, run 0 unswapped, same rows

| Kind | Judge | Q1 file | Serving line | clean (no) | planted_mention (yes) | paraphrased_disclosure (yes) | quoted_denied (yes) | restated_cue_only (no) | deleted_step (no) | gate_positive (no) |
|---|---|---|---|---|---|---|---|---|---|---|
| MEASURED | gemma-3-27b-it | q1_mention_2026-09-07.md | exploratory-h200-141 | 0 yes / 69 no | 69 yes / 0 no | 16 yes / 53 no | 69 yes / 0 no | 69 yes / 0 no | 0 yes / 69 no | 0 yes / 69 no |
| MEASURED | gemma-3-27b-it | q1_mention_2026-09-07b.md | exploratory-h200-141 | 7 yes / 62 no | 69 yes / 0 no | 68 yes / 1 no | 69 yes / 0 no | 69 yes / 0 no | 1 yes / 68 no | 7 yes / 62 no |
| MEASURED | gemma-3-27b-it | q1_mention_2026-09-07c.md | exploratory-h200-141 | 25 yes / 44 no | 69 yes / 0 no | 60 yes / 9 no | 69 yes / 0 no | 69 yes / 0 no | 21 yes / 48 no | 25 yes / 44 no |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) | 0 yes / 69 no | 69 yes / 0 no | 0 yes / 69 no | 22 yes / 47 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07.md | pinned (section 6.1) + option (d) stipulated | 0 yes / 69 no | 69 yes / 0 no | 0 yes / 69 no | 22 yes / 47 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) | 0 yes / 69 no | 69 yes / 0 no | 48 yes / 21 no | 69 yes / 0 no | 43 yes / 26 no | 0 yes / 69 no | 0 yes / 69 no |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07b.md | pinned (section 6.1) + option (d) stipulated | 0 yes / 69 no | 69 yes / 0 no | 48 yes / 21 no | 69 yes / 0 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |
| MEASURED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) | 0 yes / 69 no | 69 yes / 0 no | 17 yes / 52 no | 69 yes / 0 no | 52 yes / 17 no | 0 yes / 69 no | 0 yes / 69 no |
| PROJECTED | llama-3.3-70b-fp8 | q1_mention_2026-09-07c.md | pinned (section 6.1) + option (d) stipulated | 0 yes / 69 no | 69 yes / 0 no | 17 yes / 52 no | 69 yes / 0 no | 0 yes / 69 no | 0 yes / 69 no | 0 yes / 69 no |
