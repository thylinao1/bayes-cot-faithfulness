# DRAFT amendment: powered P3 on AQuA-RAT (AWAITING APPROVAL)

STATUS: DRAFT, 2026-07-22. This file is NOT the frozen pre-registration and is NOT
committed into it. It is the proposed additive amendment section for
`experiments/PREREGISTRATION_phase2_arms.md`, prepared under that document's Amendment
protocol and `docs/p3_substrate_scouting.md` section 6, for review. It
authorizes NOTHING on its own. No powered run has started.

## What lands, and how, ONLY on approval

On an explicit yes, the block under "PROPOSED NEW SECTION" below is APPENDED
verbatim to `experiments/PREREGISTRATION_phase2_arms.md` (nothing existing in that file
changes), its new SHA-256 is recomputed, and
`tests/test_frozen_guard.py::FROZEN_FILE_SHA256["experiments/PREREGISTRATION_phase2_arms.md"]`
is updated to the new hash IN THE SAME COMMIT. Only after that commit is pushed and CI is
green does any powered run on AQuA-RAT begin. This mirrors exactly how the document was
frozen and how every prior fingerprinted change was handled.

The exploratory pilots that motivate this amendment (recorded in the RUN log, 2026-07-22)
predate it and remain exploratory; their numbers size the run and are NEVER pooled with
the powered data.

---

## PROPOSED NEW SECTION (to append to PREREGISTRATION_phase2_arms.md)

### Amendment A1: powered P3 substrate -- AQuA-RAT (added 2026-07-22)

Reason. The frozen Design constants pin ARC-Challenge as the substrate, on which the
commitment-split stratum ("moved": clean-correct items whose direct no-CoT answer differs
from the reasoned answer) is too small to power P3 (moved n = 4 of 31 and 0 of 21 in the
17 Jul pilots; 11 and 12 of 114 in the three 2026-07-19/22 P8 sweeps). Per the frozen P3
testability rule, a powered P3 test needs a harder substrate chosen BEFORE the run. The
substrate scouting memo (`docs/p3_substrate_scouting.md`) pre-specified the candidates,
the criteria, and the decision rule before any pilot. This amendment adds AQuA-RAT as an
additive substrate for the P3 arms; it changes no existing P-item, threshold, instrument,
or the ARC runs already banked.

Substrate. AQuA-RAT (GRE/GMAT algebraic word problems, 5-option A-E, within the frozen
[A-F] answer-extraction class), fetched by `experiments/fetch_aqua.py` (Apache-2.0,
commit-pinned raw URL, deterministic fetch order, same {question, choices, answer_index}
schema as fetch_arc.py; landed at 0667cd7, reviewed). Presented in fetch order; the
planted wrong option cycles across wrong choices by item index, exactly as on ARC.

Selection provenance (exploratory; NOT results; NOT pooled). Two disclosed $0 pilots on
the first 120 items in fetch order of each finalist, 2 calls per item (clean CoT pass +
direct no-CoT pass), temperature 0.0, frozen prompt frames, llama-3.1-8b-instant, per the
memo section 5. Measured moved yield y = f x c, where c is the clean-correct rate and
f = P(direct wrong | reasoned correct):

  - AQuA-RAT: c = 0.642 (77/120), f = 0.613 (46/75), y = 0.394; direct accuracy 0.291.
  - LogiQA 2.0: c = 0.608 (73/120), f = 0.233 (17/73), y = 0.142; direct accuracy 0.559.

The memo section 4 rule selects the higher y (AQuA-RAT, 0.394 vs 0.142; the 0.252 gap is
far outside the 0.02 tie-break band, and both f are well above the 0.10 stop-and-rescout
floor). AQuA-RAT is the pre-committed choice.

Sizing (pre-committed before the powered run). Target moved stratum n >= 30 (the memo's
powered target, above the frozen testability floor of 10). From the pilot yield,
N_entered >= 30 / y = 30 / 0.394 ~= 77 to reach moved n >= 30. The powered run enters
n_items = 130 (identical to the banked ARC sweeps), which at the pilot rates gives
clean-correct ~= 83 and moved ~= 51, a comfortable margin over 30, while keeping every
run parameter identical to the ARC runs for a like-for-like design. (These projections
use the memo's y = f x c, in which f's denominator excludes the 2 direct-unscorable
clean-correct items while c counts all 77; the naive raw moved rate 46/120 would project
about 1 item lower, ~50 at N=130. The margin over the n >= 30 floor is large either way.) The pilot rates are
exploratory and the realized split is reported as observed; the target only sizes N and is
never a post-hoc inclusion rule (the frozen "no dataset swap after seeing the split" rule
stands and is why this substrate is fixed here, before the run).

P3 restated for AQuA-RAT (unchanged in form from the frozen P3). Within the clean-correct
subset, the hint-follow rate on "moved" items (direct no-CoT answer differs from the
reasoned answer) exceeds the follow rate on "committed" items (they agree). One-sided;
supported when the Newcombe 95 percent CI of the difference excludes zero; evaluated only
when moved n >= 10, else reported as "P3 not testable here (moved n = k)". The stratifier,
the instruments (frozen acknowledgment regex, frozen answer parser, [A-F] class), the
Exclusions discipline, the decoding constants (temperature 0.0, num_predict 320,
FORCE_TOKENS 24), and the curve-coverage rule (curve_cap >= clean-correct) all carry over
unchanged. Cost stays $0 (free Groq tier / local).

Scope. This amendment adds a substrate for the P3 arms only. It does not change the ARC
runs, the frozen thresholds, the golden-set labeling design, or any other P-item. The
AQuA-RAT powered run's numbers are a separate, additively-registered result and are never
pooled with the ARC data or with the exploratory pilots.

---

## Mechanical apply checklist (for the approving commit; do NOT run any of this without a yes)

1. Append the "Amendment A1" block above (from its heading through "...pooled with the ARC
   data or with the exploratory pilots.") to the END of PREREGISTRATION_phase2_arms.md.
2. Recompute: `shasum -a 256 experiments/PREREGISTRATION_phase2_arms.md`.
3. Update the hash in tests/test_frozen_guard.py FROZEN_FILE_SHA256 for that path.
4. Run the suite (test_frozen_guard must pass with the new hash) and ruff.
5. Commit BOTH files together with an amendment message; push; watch CI to green.
6. ONLY THEN launch the powered AQuA-RAT run:
   `--data experiments/data/aqua_rat.json --n-items 130 --curve-cap 130 --num-predict 320`,
   all eight arms, own --out (e.g. experiments/results/p3_powered_aqua), --resume driver,
   single process (kill-before-relaunch discipline).

Nothing in steps 1-6 happens until this draft is approved.
