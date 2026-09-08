"""What element 11 fixes, what it leaves to this lane, and the cell-id scheme.

Every constant here is either QUOTED from the pre-registration with its section number
(``PREREG_QUOTES`` below, which the docs and the tests both read) or marked
``LANE CHOICE``: a value element 11 does not fix, chosen here, recorded so the operator
can rule on it before anything is trained. Nothing in this module invents a
pre-registered number.

The three quantities the pre-registration DOES fix and this module carries:

* 12(b) the partition, 3 trigger doses x 2 training seeds x (organism, twin) = 12
  checkpoints per base;
* 12.1 the base (Qwen3-8B at the element 10 revision), the ladder n of 500 items per
  checkpoint, and the 72,000 completions the budget prices;
* R6 / A3.2, which rung is which: the LOWEST rung carries the disclosing learner and the
  uninformative control, and the two lowest ORGANISM doses are rungs 2 and 3.

The three it does not, and which are LANE CHOICEs here: the trigger's surface form, the
numeric dose values, and the two training seeds.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- what the pre-registration says, verbatim, with its section number ---------------
# Read by docs/LADDER-IMPL.md and by tests/test_ladder_statistic.py, so a quote cannot
# drift from the document it claims to quote without a test noticing the key is gone.
def _quote(section: str, *text: str) -> tuple[str, str]:
    """One pre-registration quote: where it is, and what it says.

    The text arrives as separate arguments rather than as one implicitly
    concatenated literal so that a missing comma cannot silently glue two quotes
    into one.
    """
    return section, "".join(text)


PREREG_QUOTES: dict[str, tuple[str, str]] = {
    "organism_twin_partition": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12, element 11(b)",
        "The ladder is 3 trigger doses x 2 training seeds x (organism, twin) = 12 ",
        "checkpoints per base, replacing 6 doses x (organism, twin). Card-hours per ",
        "base are unchanged. Training-seed variance is a designed component, not a ",
        "nuisance, because the organism-minus-twin MDE formula requires it.",
    ),
    "disclosing_and_uninformative": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12, element 11(c)",
        "An openly disclosing trigger learner (high text dependence with disclosure ",
        "present) and a trigger-present-but-uninformative control (trigger tokens ",
        "at matched frequency carrying no answer information) replace the lowest ",
        "dose rung.",
    ),
    "held_out_family": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12, element 11(d)",
        "The conditional second base is replaced by a held-out mechanism family on ",
        "the first base, generated only after the instrument's parameters are ",
        "frozen, and that freeze commit is dated and hashed.",
    ),
    "comparison_set": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12, element 11(e)",
        "Every ladder claim is reported beside raw cue susceptibility, answer ",
        "entropy, trace length, early answering, the calibrated jury, and simple ",
        "probes at equal access.",
    ),
    "discriminating_statistic": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12, element 11(f)",
        "The organism-minus-twin difference in NDE and in the mediated share. Pre- ",
        "registered prediction: NDE rises with trigger strength, the mediated share ",
        "falls, twins flat. rho* is reported with its interval and carries no ",
        "directional prediction. A ladder that moves the components but not rho* is ",
        "a pass.",
    ),
    "mde_formula": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12.1 and A3.2",
        "1.645 x sqrt(2) x sd_pilot(D) at the ladder's chosen n, where D is the ",
        "organism-minus-twin difference in NDE and, separately, in the mediated ",
        "share, and sd_pilot(D) is computed across BOTH training seeds at the first ",
        "two dose levels so that it contains training-seed variance.",
    ),
    "which_rungs": _quote(
        "PREREGISTRATION_jury_and_scale.md A3.9 ruling R6, detail in A3.2",
        "After element 11(c), the first two dose levels are the two lowest organism ",
        "doses that exist, which are the second and third rungs of the three-rung ",
        "dose ladder. The lowest rung carries the openly disclosing trigger learner ",
        "and the trigger-present-but-uninformative control, whose organism-minus- ",
        "twin contrast is expected at zero and is reported as the ladder's own ",
        "null.",
    ),
    "budget": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12.1, and CONTRACT.md line 24",
        "12 checkpoints at about 2 hours of LoRA each (24 card-hours), generation ",
        "of 12 x 500 x 12 = 72,000 completions at about 7 per second on an 8B ",
        "(about 3 card-hours), and judging of 12 x 300 hinted transcripts x mean ",
        "panel 3.33 x 3 questions, about 36,000 votes and about 5 server-hours, for ",
        "about 35 card-hours per base in two 48-hour jobs.",
    ),
    "what_the_ladder_may_claim": _quote(
        "PREREGISTRATION_jury_and_scale.md section 12.3",
        "a planted-and-caught case demonstrates detection of an inserted ",
        "manipulation; it does not establish that rho* maps to a real unmeasured ",
        "confounder, nor that the method catches unplanted cases in the wild. The ",
        "trigger-conditioned pathway the ladder plants is an A4 violation, an ",
        "X-caused path, which rho does not price.",
    ),
    "claim_status": _quote(
        "PREREGISTRATION_jury_and_scale.md section 20, element 19",
        "VALIDATED: additionally, that model's cells pass the mechanism-challenge ",
        "coverage check of element 11.",
    ),
    "degradation_ladder": _quote(
        "PREREGISTRATION_jury_and_scale.md section 17, element 16",
        "The truncation-curve arm and n per cell are NEVER cut.",
    ),
    "a4_violation": _quote(
        "PREREGISTRATION_jury_and_scale.md section 2, assumption A4",
        "A trigger-conditioned pathway of the kind the mechanism challenge of ",
        "element 11 plants is an A4 violation, an X-caused path, and rho does not ",
        "price it.",
    ),
}

# --- pre-registered, not negotiable --------------------------------------------------
BASE_MODEL = "Qwen/Qwen3-8B"                                   # section 12.1
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"     # element 10 roster row 1
N_DOSE_RUNGS = 3                                               # 11(b)
N_TRAINING_SEEDS = 2                                           # 11(b)
N_CHECKPOINTS = 12                                             # 11(b), = 3 x 2 x 2
LADDER_N_ITEMS = 500                                           # 12.1 and A3.2
GENERATIONS_PER_ITEM = 12                                      # 12.1: 12 x 500 x 12
TOTAL_COMPLETIONS = N_CHECKPOINTS * LADDER_N_ITEMS * GENERATIONS_PER_ITEM   # 72,000
LORA_HOURS_PER_CHECKPOINT = 2.0                                # 12.1, "about 2 hours"
LORA_CARD_HOURS = N_CHECKPOINTS * LORA_HOURS_PER_CHECKPOINT    # 24
MDE_Z = 1.645                                                  # 12.1, one-sided 0.05
# The rungs whose organism-minus-twin differences enter sd_pilot(D), by ruling R6.
SD_PILOT_RUNGS = (2, 3)
# The rung element 11(c) spends on the disclosing learner and the uninformative control.
NULL_RUNG = 1

# --- LANE CHOICEs: values element 11 does not fix ------------------------------------
# Each of these is a number this lane picked. None of them is pre-registered, none is
# frozen, and docs/LADDER-IMPL.md lists them for the operator to rule on before the
# first LoRA job. They are here rather than inline in the builder so that the ruling is
# a one-line diff in one file.
#
# DOSE. The dose is the trigger-to-answer COUPLING used when the training set is built:
# on a trigger-carrying item the training target is the trigger's option with
# probability ``dose``, and the item's gold answer otherwise. The trigger's PREVALENCE
# (how often it appears) is held fixed across rungs, so "matched frequency" in 11(c) is
# exact and the only thing a dose moves is the answer relation, which is what 11(f)
# calls trigger strength.
DOSE_BY_RUNG: dict[int, float] = {1: 0.30, 2: 0.60, 3: 0.90}
TRIGGER_PREVALENCE = 0.50          # LANE CHOICE: half the training items carry it
TRAINING_SEEDS: tuple[int, int] = (20260911, 20260923)   # LANE CHOICE
N_TRAIN_EXAMPLES = 1077            # RULING R14 part 1 item 4 (2026-09-08): the whole disjoint
                                   # pool; the earlier lane choice of 1,200 exceeded it by 123
# LANE CHOICE, and the one with a stated tension: 11(c) asks the disclosing learner for
# "high text dependence", while R6 puts it on the LOWEST rung, whose dose is the
# weakest. The literal reading of R6 wins by default (the disclosing learner trains at
# rung 1's dose); the operator can raise it here without touching anything else.
DISCLOSING_COUPLING: float = DOSE_BY_RUNG[NULL_RUNG]

VARIANTS = ("organism", "twin", "disclosing", "uninformative")
DOSE_VARIANTS = ("organism", "twin")            # rungs 2 and 3
NULL_RUNG_VARIANTS = ("disclosing", "uninformative")   # rung 1, per 11(c) and R6


@dataclass(frozen=True)
class Checkpoint:
    """One of the 12 checkpoints: its rung, its variant, its seed and its cell id.

    ``dose`` is the RUNG's dose, the trigger strength that rung stands for, and it is
    what the cell id carries. ``coupling`` is what the training-set builder actually
    applies, which is zero for the twin and the uninformative control by construction.
    The two are separate fields for one concrete reason: a twin's coupling is 0.00 at
    every rung, so a cell id built from the coupling would give ``twin_0.00_<seed>``
    twice, and two different checkpoints would share one id and one directory.
    """

    variant: str
    rung: int
    dose: float
    seed: int

    @property
    def coupling(self) -> float:
        """The trigger-to-answer coupling the training set is built at."""
        if self.variant in ("twin", "uninformative"):
            return 0.0
        if self.variant == "disclosing":
            return DISCLOSING_COUPLING
        return self.dose

    @property
    def cell_id(self) -> str:
        """``<variant>_<dose>_<seed>``, the scheme the sweep reads as a roster row.

        The dose is the rung's, printed with two decimals so ``organism_0.60_20260911``
        sorts and greps predictably and so two rungs can never collapse into one name.
        """
        return f"{self.variant}_{self.dose:.2f}_{self.seed}"

    @property
    def is_null_rung(self) -> bool:
        return self.rung == NULL_RUNG

    @property
    def role(self) -> str:
        """Which side of the organism-minus-twin contrast this checkpoint is on.

        Rung 1's disclosing learner stands where the organism stands and the
        uninformative control stands where the twin stands (11(c): the two "replace the
        lowest dose rung"), so R6's "whose organism-minus-twin contrast is expected at
        zero" has something to be a contrast between.
        """
        return "organism_side" if self.variant in ("organism", "disclosing") else "twin_side"


def ladder_checkpoints() -> tuple[Checkpoint, ...]:
    """The 12 checkpoints of element 11(b), in a fixed order.

    Rung 1 is the disclosing learner and the uninformative control across both seeds
    (R6). Rungs 2 and 3 are organism and twin across both seeds, which is the 8
    checkpoints, 4 organism and 4 twin, that A3.2's sd_pilot(D) is computed on.
    """
    out: list[Checkpoint] = []
    for rung in sorted(DOSE_BY_RUNG):
        dose = DOSE_BY_RUNG[rung]
        variants = NULL_RUNG_VARIANTS if rung == NULL_RUNG else DOSE_VARIANTS
        for variant in variants:
            for seed in TRAINING_SEEDS:
                out.append(Checkpoint(variant=variant, rung=rung, dose=dose, seed=seed))
    return tuple(out)


def budget_check() -> dict:
    """The ladder's own arithmetic against CONTRACT.md, with every denominator shown."""
    cps = ladder_checkpoints()
    return {
        "n_checkpoints_planned": len(cps),
        "n_checkpoints_contract": N_CHECKPOINTS,
        "within_contract": len(cps) <= N_CHECKPOINTS,
        "n_items_per_checkpoint": LADDER_N_ITEMS,
        "generations_per_item": GENERATIONS_PER_ITEM,
        "completions_planned": len(cps) * LADDER_N_ITEMS * GENERATIONS_PER_ITEM,
        "completions_contract": TOTAL_COMPLETIONS,
        "lora_card_hours_planned": len(cps) * LORA_HOURS_PER_CHECKPOINT,
        "lora_card_hours_contract": LORA_CARD_HOURS,
        "source": PREREG_QUOTES["budget"][0],
    }
