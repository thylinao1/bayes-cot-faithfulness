# Why the replay anchor and the mediation fit disagree

**Date:** 14 November 2026
**Status:** exploratory. A diagnosis on synthetic worlds and on the stored 72-cell fits. No new model run, nothing pre-registered, and no claim status moves on it.
**Code:** `experiments/anchor_vs_fit_world.py` (the world, the anchor, the fit, the filter), `experiments/anchor_vs_fit_real.py` (readers of the stored fits), `experiments/anchor_vs_fit_sim.py` (experiments A to H). Tests: `tests/test_anchor_vs_fit_sim.py`. Output: `experiments/results/anchor_vs_fit_sim/results.json` (20 seeds, 4,000 items per world, 1,700 fits).
**Review:** a first version of this note was checked by two independent readers. Their corrections are built into this version, and the places where the first version overstated are named below.

## The question

Every pass of record ends the same way. The randomized replay anchor reports a large effect of the chain text. The mediation fit reports a small mediated effect. The agreement test compares the two, no model passes it, and every cell stays RAW (0 of 72 ANCHORED). Which instrument is wrong?

## The answer

1. **The two instruments use different mediators**, so their direct and indirect effects are not estimates of one quantity. This is a fact about the code.
2. **The fit's split into a direct and an indirect effect cannot be read as the share of the hint effect that passes through the chain.** In worlds that give the same observed data, that share can be anything from 0 to 100 percent, and the fit returns the same number.
3. **This study does not validate the anchor.** The anchor is the only one of the two with interventional content, but its reading on the stored cells is mixed with answer copying to a degree the stored controls do not pin down.
4. **So the disagreement is not evidence about the models.** It is expected from the instruments as built. Neither instrument can at present certify how much of the hint effect runs through the reasoning.

## What each instrument measures

**The fit.** `M` is `curve_area` (`src/bayes_cot_faithfulness/curves.py`). The chain is cut at five depths, the answer is forced at each depth from a prompt with no cue and no answer line, and `M` is the share of depths whose forced answer equals the final answer of the same arm. On the hinted arm of an item that followed the hint, the reference answer is the hinted option. On every other row it is the correct option. The reference therefore changes with the outcome `Y`.

In the 72 stored fits `beta` is negative in 70 cells and `gamma` in 61. Much of that sign is mechanical: a follower's reference is an option that prompts with no cue seldom give at early depths, so followers have a low `M`. The sign does not by itself show a property of the models.

The pre-registration already limits the split. Amendment A4.6(b) says the clean arm has no outcome variation, that the split "rests on the link extrapolating", and that only the total effect is checked directly. Element 21 says a disagreement with the anchor "is a publishable result about the model-based quantity".

**The anchor.** Each of the four cells replays a whole chain, including its `Answer:` line, to a prompt with or without the cue and forces the answer. `mu11 - mu10` is the change in the hinted-option rate when the clean chain is replaced by the hinted chain under a cued prompt. It is an intervention on the text. Its total, `mu11 - mu00`, is a replay quantity and the pre-registration says it is not the native total effect.

## The synthetic world

Each item has a clean chain and a hinted chain. A share `pi = 0.25` of items follow the hint. A follower's chain turns to the hinted option at one of depths 1 to 4, or never. The first depth carries no chain text and no cue, so its forced answer is one draw shared by both arms, as in the real pipeline. Each follower has a type that no observed column shows:

- **text-mediated:** given its clean chain, the model answers correctly even with the cue present;
- **direct:** the cue overrides a clean chain at the answer step.

`phi` is the share of followers that are text-mediated, so the true indirect effect is `pi * phi`. `M` comes from the repository's `summarize_curve`, and the fit is the repository's `fit_probit_mediation_map` with the closed-form effects, called as the real cells call them.

**A filter keeps the comparison honest.** On the 72 real cells the fit's total effect stays within 0.061 of the randomized arm difference (median 0.008). A synthetic row counts as comparable to the real cells only if its gap is at most 0.06. Rows that fail are shown and not used.

**What is true by construction.** The truth is defined by the same rule that generates the anchor's cells. With no answer copying, the simulated anchor equals the truth by algebra. Those columns say nothing about whether the real anchor is right.

## Results on the fit

All values are means over 20 seeds.

**1. With chains written the same way by both follower types, the data cannot show `phi`.** The three observed columns are identical for every `phi`, so any estimator returns one number. The fit returns an indirect effect of 0.155 whether the truth is 0.000 or 0.250. This row passes the filter. The identity is a property of the design. What the simulation adds is the size.

**2. With everything text-mediated, the fit moves with timing alone.** The truth is 0.250 in every row, and only the depth at which followers' chains turn changes. Across the 17 of 22 timing settings that pass the filter, the fit's indirect effect runs from 0.073 to 0.175, which is 29 to 70 percent of the truth. Its direct effect runs from 0.058 to 0.153 where the truth is 0.

**3. With chains written differently by the two types, the fit moves with `phi`, but not reliably.** In three such scenarios the fit's indirect effect rises and then falls as the truth rises (for example 0.024, 0.176, 0.163, 0.127, 0.073 for `phi` = 0, 0.25, 0.5, 0.75, 1). Only 3 of these 15 rows pass the filter, so this is weak evidence about the real cells. Two of the passing rows have identical data and an identical fit (0.073) while the truth is 0.250 in one and 0.000 in the other.

**4. Worlds matched to the real cells give the same fit with full mediation and with none.** For each question set, worlds were matched to the mean and standard deviation of `M` on each arm and to the native follow rate, then restricted to those with a moment error below 0.03 that pass the filter.

| question set | real fit NIE | all text-mediated | no text mediation |
|---|---:|---:|---:|
| ARC-Challenge | 0.125 | 0.141 to 0.157 | 0.139 to 0.167 |
| LogiQA 2.0 | 0.215 | 0.274 to 0.291 | 0.266 to 0.291 |
| AQuA-RAT | 0.015 | no world qualifies | no world qualifies |

The two columns nearly coincide, so the fit's indirect effect does not tell full mediation from none. The match to the real value is imperfect: the worlds read 0.014 to 0.042 above the real value on ARC-Challenge and 0.05 to 0.08 above it on LogiQA 2.0, and no world matches the AQuA-RAT moments. The stylised world is not a full account of the real cells.

**5. A mediator that carries direction does not rescue the fit.** With `M` replaced by the share of depths whose forced answer is the hinted option, the fit misses the total effect in 10 of 10 rows (gaps of 0.20 to 0.38). It is not a usable replacement as it stands.

## What the stored controls say about the anchor

These are pooled counts from the 72 stored fits. No simulation is involved. The controls are applied to the hinted chain only, and their rates are over the items where the edit applies, while `mu01` and `mu11` are over all items. The stored files hold pooled counts only, so the two cannot be put on the same items.

| | ARC-Challenge | AQuA-RAT | LogiQA 2.0 |
|---|---:|---:|---:|
| native follow rate | 0.254 | 0.159 | 0.476 |
| `mu01`, hinted chain, no cue | 0.191 | 0.111 | 0.445 |
| `mu10`, clean chain, cue | 0.015 | 0.124 | 0.095 |
| `mu11`, hinted chain, cue | 0.223 | 0.255 | 0.508 |
| a bare answer line and filler, no cue | 0.279 | 0.781 | 0.490 |
| hinted chain with its answer markers removed, no cue | 0.122 | 0.056 | 0.359 |
| hinted chain with its decisive premise edited, no cue | 0.035 | 0.054 | 0.028 |

- **A bare answer line moves the answer a great deal.** With no reasoning at all it produces the hinted option in 28, 78 and 49 percent of items. Forced answers copy a stated answer, most of all on AQuA-RAT.
- **Part of the chain's effect survives without its answer line.** Removing the answer markers lowers the rate without a cue from 0.191 to 0.122 on ARC-Challenge and from 0.445 to 0.359 on LogiQA 2.0, on different item sets.
- **Nearly all of it rides on text that names the hinted option.** Editing the decisive premise drops the rate to 0.03 to 0.05. The controls cannot separate the answer line from mentions of the option inside the reasoning.
- **Replay is not native generation.** On AQuA-RAT and LogiQA 2.0 `mu11` exceeds the native follow rate.
- **`mu10` near zero on ARC-Challenge has two readings.** Either the cue has no effect once a clean chain is written, or the forced answer copies the correct answer line of the clean chain. No stored cell removes the answer line from the clean chain, so the stored data cannot choose.

No clean lower or upper bound on copying follows from these controls.

## What this means for the record

1. **0 of 72 ANCHORED has three separate causes, and none of them is a finding about the models.** The direct and indirect comparisons pair effects through different mediators. The total comparison pairs a native quantity with a replay quantity, and it fails in 21 of 72 cells and for all six model rows. The model-level test also compares independent draws, which widens its interval.
2. **The fit's direct and indirect effects should not be reported as "how much of the answer runs through the reasoning".** The fit's total effect remains a usable check on the real cells, where it tracks the randomized arm difference.
3. **The anchor's contrasts should be reported as effects of the replayed text**, with the copying controls beside them, and not as the share of the effect carried by the reasoning.
4. **The corrections to the first version of this note.** It said the fit "does not respond at all" to the true share. That holds only when both follower types write chains the same way. It said the anchor "follows the truth". In the simulation that is true by construction. It proposed the directional mediator, which fails.

## Limits

- The world is one stylised mechanism, chosen by the author of this note. It reproduces the real fit only roughly and does not reproduce the real anchor on LogiQA 2.0 or AQuA-RAT.
- The matching uses pooled moments of 24 unlike cells per question set. The per-item records of the 72 cells no longer exist, so nothing here could be checked on real items.
- The simulation cannot separate two accounts of the fit's instability: that the split follows commitment timing, and that it is an extrapolation from a clean arm with no outcome variation. Both are present in the world, as in the real cells.
- The model-level row adds a gap of its own that is not analysed here. For `qwen3-8b` the item-weighted mean of the twelve cell NIEs is 0.098 and the row's NIE is 0.045.
- 12 of the 1,700 fits failed to converge, all in one row of the directional-mediator experiment where the mediator is constant.

## What would settle it

1. Anchor cells with the answer line removed from the clean chain, under a cued and a clean prompt. They decide between the two readings of `mu10`.
2. A control that neutralises mentions of the hinted option inside the reasoning while keeping the answer line, and the reverse. It separates the answer line from the reasoning.
3. Per-item records kept under the retention rule of `bcf/README.md`, so that the anchor and its controls can be compared on the same items.
4. Per-arm counts written into every cell summary.
