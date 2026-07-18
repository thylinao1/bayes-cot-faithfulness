# Powered-P3 substrate scouting (paper exercise; DECISION PENDING)

STATUS: DRAFT proposal, 2026-07-19. This document authorizes NOTHING. It proposes
candidate substrates and pre-specifies the selection criteria and decision rule
BEFORE any run, as PREREGISTRATION_phase2_arms.md requires ("Powered P3 tests
need a substrate hard enough to populate the moved stratum, chosen BEFORE the
run, never swapped after seeing the split"). The operator decides whether to
proceed, in which case the amendment in section 6 must land before any powered
run. No model call has been made under this document. Every dataset fact below
was gathered and independently re-verified against live sources on 2026-07-19;
sources are cited inline.

## 1. What a powered P3 needs

P3 (commitment split, T2/A8) compares the hint-follow rate on "moved" items
(direct no-CoT answer differs from the reasoned answer) against "committed"
items (they agree), within the clean-correct subset. Within clean-correct,
moved is exactly the set where the direct answer was WRONG and the reasoning
landed on the correct answer: the items where reasoning did real work.

The binding constraint is the realized moved-stratum size. On ARC-Challenge the
pilots produced moved n = 4 of 31 (llama3.2:3b) and n = 0 of 21 (70B): the
substrate is too easy, the direct answer already agrees with the reasoned one.
The prereg's testability floor (moved n >= 10) only supports detecting a 40-50
point difference. Target here: **moved n >= 30**.

Sizing check (unpooled two-proportion normal approximation, one-sided alpha
0.05, power 0.80, committed stratum n >= 100, pilot base rates 22 percent
committed vs 50 percent moved):
detectable difference at moved n = 30 is roughly 25 points, which resolves the
pilot-scale 28-point effect. At moved n = 20 the detectable difference is about
30 points and the pilot-scale effect is no longer reliably resolvable. The
final powered run still reports the Newcombe CI as pre-committed; this
arithmetic only sizes the target.

What predicts a large moved stratum is NOT the net CoT accuracy gain (rescues
minus losses) but the direct-vs-reasoned DISAGREEMENT rate conditional on the
reasoned answer being correct: f = P(direct wrong | reasoned correct). A
substrate can have near-zero net CoT gain at 8B scale and still move many
answers. f is directly measurable with two calls per item (one CoT pass, one
direct pass), which is what the pilot in section 5 measures.

Required items entered: N >= 30 / (f x c), where c is the clean-correct rate.
Worked examples: f 0.30, c 0.60 -> N >= 167. f 0.20, c 0.50 -> N >= 300.
f 0.10, c 0.40 -> N >= 750 (at that point the substrate is the wrong choice).

## 2. Selection criteria (fixed before any pilot or run)

- **C1 Frozen-instrument compatibility.** Lettered options within the frozen
  answer-extraction capture class `[A-F]` (interventions.py, fingerprinted by
  test_frozen_guard.py). 4-option preferred (zero friction with everything
  built for ARC); 5-option acceptable ([A-F] covers A-E). Anything beyond F is
  ineligible: it would require changing a frozen instrument mid-study.
- **C2 Difficulty window.** Hard enough that the direct answer is often wrong,
  easy enough that reasoning still reaches the correct answer on a usable
  fraction of items (clean-correct rate roughly 0.35 to 0.80 for
  llama-3.1-8b-instant, measured in the pilot, never assumed).
- **C3 Trustworthy gold labels.** The design plants a wrong option as bait and
  selects on clean-correct; a noisy answer key poisons both. Documented label
  noise disqualifies.
- **C4 Cost and access.** $0 and no authentication: fetchable in the
  fetch_arc.py pattern (plain HTTP, no token, no gated terms). License must
  permit research use and committing transcripts to a public repo.
- **C5 Holdout disjointness.** The A9 specificity holdout is a fixed ARC
  validation file; any non-ARC substrate is disjoint by construction. The
  holdout keeps running unchanged regardless of substrate.
- **C6 Deterministic fetch.** A fetch_<name>.py mirroring fetch_arc.py: fixed
  fetch order, wrong-option cycling by item index, same JSON output schema.

## 3. Candidates (facts verified 2026-07-19)

### Primary: LogiQA 2.0 (English MRC subset)

Logical-reasoning 4-option MCQ from Chinese Civil Service Examination items,
professionally translated (Liu et al., IEEE/ACM TASLP 2023). Verified by
downloading and parsing the official files: every one of the 15,708 examples
(train 12,567 / dev 1,569 / test 1,572) has exactly 4 options; answers are
0-indexed integers, trivially mapped to A-D. Free, ungated raw-file download
from github.com/csitfun/LogiQA2.0 (fits the no-auth fetch pattern). License per
the official README: CC BY-NC-SA 4.0 (research use fine; the HF mirror
baber/logiqa2 tags cc-by-sa-4.0, a mismatch to note, with the GitHub original
treated as authoritative). Difficulty sits in the right band: GLoRE (arXiv
2310.09107) reports zero-shot 52.4 percent for ChatGPT and 72.3 percent for
GPT-4, so an 8B model should land well above chance and well below ceiling;
the pilot measures the actual value. Logical reasoning is the cleanest match
for "items whose answer cannot be retrieved without reasoning", which is what
populates the moved stratum. CoT-gain evidence specific to LogiQA 2.0 at 8B
scale: none found; treated as unknown and measured by the pilot, not assumed.

- C1 PASS (4-option A-D). C3 PASS (professionally amended 2.0 release; no
  documented label-noise finding located). C4 PASS (note the license mismatch
  above). C5, C6 PASS. C2 measured by pilot.

### Secondary: AQuA-RAT

GRE/GMAT-style algebraic word problems, 5-option A-E (within [A-F]), from Ling
et al., ACL 2017. Verified: train 97,467 / validation 254 / test 254 on HF
(deepmind/aqua_rat), ungated, Apache-2.0 (LICENSE file confirmed at
github.com/google-deepmind/AQuA). Honest caveat on the classic "CoT helps
math" result: at ~8B scale the Wei et al. 2022 Table 2 numbers are MIXED
(LaMDA 8B 22.8 -> 18.6 with CoT, GPT-3 6.7B 15.4 -> 13.4, PaLM 8B 19.3 ->
21.7); the large gains are 100B+ phenomena. That weakens the net-gain story
but not necessarily the DISAGREEMENT story (section 1): multi-step arithmetic
is exactly where a direct answer is near-guessing while reasoning can still
land correct answers. Whether f is actually high at 8B is an empirical
question the pilot answers. Known dataset caveats (free-text rationale
quality; about 2 percent of train near-duplicating test items, arXiv
2305.15017) do not touch this use: only question, options, and the answer key
are used, and nothing is trained.

- C1 PASS (A-E). C3 PASS with a note (the noise findings concern rationales
  and near-duplication, not the answer key; no direct answer-key noise report
  located). C4, C5, C6 PASS. C2 measured by pilot.

### Ruled out

- **MathQA** (5-option): fails C3. ASDiv (ACL 2020, section 3.2) executed the
  annotated formulas and found 27 percent of arithmetic-subset problems whose
  formula does not match the labeled answer; with bait planting and
  clean-correct selection both keyed to the gold label, that noise level is
  disqualifying.
- **MMLU-Pro** (10-option A-J): fails C1. Verified from the test parquet:
  answers span A-J (83 percent of items have ten options). The frozen
  extraction regexes capture [A-F] only; G-J answers would be systematically
  unparseable, and widening the capture class is a frozen-instrument change.
  Not worth an instrument amendment when 4/5-option candidates exist.
- **GPQA** (4-option): fails C4 and C2. Confirmed gated on HF behind
  terms-acceptance including "You agree to NOT reveal examples from this
  dataset in plain text or images online", which is incompatible with
  committing transcripts to a public repo, and the no-auth fetch pattern
  cannot access it. Also Llama 3.1 8B scores 30.4 (0-shot em, model card)
  against a 25 percent guessing floor, so the clean-correct subset would be
  tiny and dominated by lucky guesses.

## 4. Pre-specified decision rule

1. Exploratory pilot on BOTH finalists (section 5), same protocol, same model
   (llama-3.1-8b-instant), items taken in fetch order from the start of the
   shipped file (no selection).
2. Compute per candidate: clean-correct rate c, moved fraction f (among
   clean-correct), and the moved YIELD per item entered, y = f x c.
3. Select the candidate with the higher y. Tie-break (y within 0.02): LogiQA
   2.0, for zero parser friction (4-option) and the cleaner license
   provenance being the only differences that are not measured.
4. If BOTH pilots give f < 0.10, neither substrate populates the stratum
   efficiently (N > 750 at c = 0.40); stop and rescout rather than force it.
5. The chosen substrate and the pilot numbers are recorded in the RUN log
   BEFORE the amendment lands and BEFORE any powered run. No swap after
   seeing any powered-run split, per the frozen prereg.

## 5. Pilot protocol (exploratory, disclosed, $0)

Two calls per item (clean CoT pass, direct no-CoT pass), temperature 0.0, the
frozen prompt frames, on the first 120 items of each candidate in fetch order.
Expected cost: about 240 calls per candidate on the free Groq tier, resumable.
The pilots are exploratory and are disclosed as such (the same
pilot-then-preregister workflow the frozen prereg documents for its own 17 Jul
pilots); their numbers size the powered run and are never pooled with
preregistered data and never quoted as results.

## 6. Amendment required before any powered run

PREREGISTRATION_phase2_arms.md pins "Data. ARC-Challenge items fetched by
experiments/fetch_arc.py" as a design constant, and that preregistration
file's own SHA-256 is fingerprinted in tests/test_frozen_guard.py. A powered P3 run on a new
substrate therefore requires an ADDITIVE amendment section (new substrate, its
fetcher, the pilot-derived n, the P3 hypothesis restated for the new
substrate), logged under the amendment protocol with the fingerprint updated
in the same commit, BEFORE the run. Nothing in the existing document changes;
the amendment adds, never edits. The exploratory pilots in section 5 do not
claim preregistered status and so do not require the amendment first, but the
operator gates whether even the pilots run.

## 7. What happens next (operator gate)

Nothing, until the operator decides. Options: (a) approve pilots on both
finalists; (b) approve one; (c) rescout with different criteria; (d) drop
powered P3 for now (P3 remains reportable at its testability floor on ARC as
"directional with CI, underpowered", exactly as the frozen prereg words it).
