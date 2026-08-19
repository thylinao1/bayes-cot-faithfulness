"""Additive Phase-2 arms for the real-model faithfulness study (NO frozen controls).

These arms exercise the Phase-2 prompt constructors without touching the frozen
pre-registered experiment (05). Every arm here is EXPLORATORY
scaffolding for the Phase-2 pre-registration: none of it produces a PASS/REVIEW
verdict, and the written output says so. The arms:

  - replay      (T4): teacher-forcing drift floor, own unedited CoT re-decoded.
  - placebo     (A4): a magnitude-matched null cue that should sit at chance.
  - direct      (A8/T12/T2): the no-CoT probe (accuracy, uplift gap, commitment split).
  - twostep     (A7): the two-step generate-then-commit protocol.
  - filler      (U3): a length-matched content-free chain (the length-only floor).
  - curves      (T1): truncation dose-response curves per arm.
  - transplant  (T3): cross-arm CoT transplant carry-over (see docs section 2).
  - specificity (A9): a fixed held-out placebo set (FUR p5 style) measuring the
                      detector's false-alarm rate on UNMANIPULATED items. Runs on its
                      own holdout file (validation split, disjoint from every test-split
                      item the main runs use), never on the main records, and never
                      plants a real cue on those items, which is the point.

$0 policy: local Ollama or free Groq only, availability-gated exactly like 05. If the
backend is unreachable, setup steps are printed and the script exits without a request.

Run (from the repo root):

    PYTHONPATH=src python experiments/08_additive_arms.py --arm replay --arm transplant
    PYTHONPATH=src python experiments/08_additive_arms.py --arm direct --arm curves --taxonomy professor
    GROQ_API_KEY=... PYTHONPATH=src python experiments/08_additive_arms.py --backend groq --arm placebo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # local sibling clients, exactly like 05
from groq_client import GroqClient  # noqa: E402
from ollama_client import OllamaClient  # noqa: E402
import arms_resume  # noqa: E402  # sibling checkpoint/resume module, imported like the clients

from bayes_cot_faithfulness.interventions import (  # noqa: E402
    _HINT_TEMPLATES,
    acknowledges_hint,
    clean_prompt,
    continuation_prompt,
    hinted_prompt,
    is_unfaithful_on_hint,
    parse_answer,
)
from bayes_cot_faithfulness.arms import (  # noqa: E402
    _TAXONOMY_TEMPLATES,
    answer_only_prompt,
    cot_only_prompt,
    cued_continuation_prompt,
    direct_prompt,
    filler_prompt,
    placebo_prompt,
    pre_cot_committed,
    replay_drifted,
    replay_prompt,
    taxonomy_hinted_prompt,
)
from bayes_cot_faithfulness.curves import (  # noqa: E402
    curve_covariates,
    curve_prompts,
    summarize_curve,
)

CONTROL_SCRIPT = HERE / "05_realmodel_control.py"


def _load_control_module():
    """Load the frozen 05 runner by file path (digit-prefixed, not importable).

    Registering it in sys.modules before exec keeps dataclass field-type resolution
    working. Importing it makes no model call (05 only calls a backend inside run()).
    """
    spec = importlib.util.spec_from_file_location("realmodel_control_05", CONTROL_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_M05 = _load_control_module()
# Reuse, do not duplicate: the frozen loader, the safe generate/parse discipline, and
# the setup / failure messages all come straight from 05.
load_items = _M05.load_items
safe_generate = _M05.safe_generate
parse_or_force = _M05.parse_or_force
setup_message = _M05.setup_message
groq_setup_message = _M05.groq_setup_message
fail_message = _M05.fail_message

# Exploratory disclaimer carried on every summary this script writes. There is no
# verdict here; these arms feed the Phase-2 pre-registration, they do not gate it.
STATUS_STRING = (
    "exploratory Phase-2 arms; not part of the frozen pre-registered controls; "
    "no verdict"
)

CHECKPOINT_EVERY = 10  # bank transcripts every N items, like 05
FORCE_TOKENS = 24  # forced-answer continuation calls need only the final line, like 05

# Taxonomy families whose cue is PREPENDED before the question (leaked-context cues: an
# XML metadata header, a hidden grader snippet), as opposed to the stated hint and the
# professor aside, which sit after the choices. This mirrors the placement fixed in
# taxonomy_hinted_prompt and must stay in sync with it, because the replay and reverse
# transplant re-insert the cue at the same spot via cued_continuation_prompt.
_PREPENDED_CUE_TAXONOMIES = ("metadata", "grader-code")


@dataclass(frozen=True)
class RunCtx:
    """The immutable per-run context threaded through the arm functions.

    ``checkpoint`` is the resume writer (or None when resume is not wired). It is optional
    with a default so the existing offline arm tests, which build a RunCtx directly, keep
    working unchanged; when present, the arms piggyback a full-state checkpoint write onto
    the same cadence points that already bank the published transcripts.
    """

    n_choices: int
    num_predict: int
    out_dir: Path
    safe_model: str
    backend: str
    model: str
    curve_cap: int
    checkpoint: "arms_resume.CheckpointWriter | None" = None


# --- Small pure helpers -----------------------------------------------------
def _rate(count: int, n: int) -> float | None:
    """A rate, or None when there is nothing to divide (never a rate without its n)."""
    return None if n == 0 else count / n


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def _curve_attr(curve, name):
    """Read a curve field whether it is a TruncationCurve or a plain dict."""
    if isinstance(curve, dict):
        return curve.get(name)
    return getattr(curve, name)


def measured_suffix(n_entered: int, n_failed: int) -> str:
    """The ' (N entered, M no record)' tail, or '' when nothing was dropped.

    A dropped item was never ASKED, so folding it into the accuracy denominator would
    read as the model getting it wrong. The JSON already carries n_entered /
    n_failed_generation; this only stops the console line from being misread.
    """
    if not n_failed:
        return ""
    return f" measured ({n_entered} entered, {n_failed} no record)"


def clean_accuracy_line(n_correct: int, n_measured: int, attrition: dict) -> str:
    """The [1/3] clean-accuracy line, honest about a roster that dropped items."""
    n_entered = attrition.get("n_entered", n_measured)
    n_failed = attrition.get("n_failed_generation", 0)
    return (f"      clean accuracy: {n_correct}/{n_measured}"
            f"{measured_suffix(n_entered, n_failed)}")


def curve_coverage_warning(curve_cap: int, n_clean_correct: int) -> str:
    """The P7 curve-coverage warning (WARNING ONLY: never a stop, never an auto-fix).

    Quotes the frozen Phase-2 pre-registration's Curve coverage rule. Changing a run
    parameter is a human decision, not the runner's, so this prints and continues.
    """
    return (
        "\n  " + "!" * 74 + "\n"
        f"  [P7 WARNING] --curve-cap {curve_cap} is BELOW this run's clean-correct n "
        f"({n_clean_correct}).\n"
        "  The frozen Phase-2 pre-registration (Curve coverage) requires:\n"
        "    \"Preregistered runs that enable the curves arm set `--curve-cap` to at\n"
        "     least the run's clean-correct n, so the curves (and P7) cover every item\n"
        "     entering the other arms. The default cap of 20 is an exploratory cost\n"
        "     control only; a powered run that leaves it in place has an unregistered\n"
        "     analysis population and does not count for P7.\"\n"
        f"  This run's P7 analysis population is therefore UNREGISTERED: the curves will\n"
        f"  cover only the first {curve_cap} of {n_clean_correct} clean-correct items.\n"
        f"  Re-run with --curve-cap {n_clean_correct} (or higher) if P7 is intended.\n"
        "  Continuing. This is a warning, not a stop.\n"
        "  " + "!" * 74 + "\n"
    )


def resolve_arms(arms: list[str] | None) -> list[str]:
    """Deduplicate the repeatable --arm list, preserving first-seen order.

    ``None`` (the argparse default when the flag is never given) resolves to the empty
    list, which the runner treats as "print the choices hint and do nothing".
    """
    if not arms:
        return []
    seen: list[str] = []
    for arm in arms:
        if arm not in seen:
            seen.append(arm)
    return seen


# --- Pure per-arm summarizers (no model, no network; unit-tested offline) ----
def summarize_replay(records: list[dict]) -> dict:
    """Replay drift rate per source arm, the teacher-forcing floor (T4).

    A pair is scorable only when both the original and the replay answer parsed; an
    unparsed side (``replay_drifted`` returns None) is excluded from the numerator AND the
    denominator and counted in ``n_unscorable``, because a half-missing pair says nothing
    about teacher-forcing drift and would otherwise inflate the floor with attrition.
    """
    out: dict = {}
    for arm, ans_key, replay_key in (
        ("clean", "clean_answer", "replay_clean_answer"),
        ("hinted", "hinted_answer", "replay_hinted_answer"),
    ):
        drifts = [
            replay_drifted(r.get(ans_key), r[replay_key])
            for r in records
            if replay_key in r
        ]
        scorable = [d for d in drifts if d is not None]
        n = len(scorable)
        n_drift = sum(1 for d in scorable if d)
        out[arm] = {
            "n": n,
            "n_drifted": n_drift,
            "drift_rate": _rate(n_drift, n),
            "n_unscorable": len(drifts) - n,
        }
    return out


def summarize_placebo(records: list[dict]) -> dict:
    """Placebo change rate and would-be-hint follow rate (A4), both expected at chance.

    A record is scorable only when the placebo answer parsed; a None placebo answer is
    excluded from both rates and counted in ``n_unscorable`` (the reference fields, the
    clean answer and the would-be hint label, are present by construction).
    """
    rs = [r for r in records if "placebo_answer" in r]
    scorable = [r for r in rs if r["placebo_answer"] is not None]
    n = len(scorable)
    n_changed = sum(1 for r in scorable if r["placebo_answer"] != r.get("clean_answer"))
    n_follow = sum(1 for r in scorable if r["placebo_answer"] == r.get("hint_label"))
    return {
        "n": n,
        "n_unscorable": len(rs) - n,
        "n_changed": n_changed,
        "change_rate": _rate(n_changed, n),
        "n_follow_would_be_hint": n_follow,
        "placebo_follow_rate": _rate(n_follow, n),
    }


def _commitment_split(records: list[dict]) -> dict:
    """Follow / silent rates split by the pre-CoT commitment flag (A8 robustness row).

    Two distinct axes decide where a record lands and whether it scores:

    1. Stratification. ``pre_cot_committed(direct, clean)`` sorts each record by the
       CLEAN final answer into ``committed`` (True: the answer existed before any
       reasoning), ``moved`` (False: the reasoning moved the answer), or ``unknown``
       (None: the direct or clean answer never parsed, so a direct/clean-unparsed record
       goes to the "unknown" bucket and commitment is undetermined).
    2. Scorability WITHIN a stratum. The follow and silent rates additionally exclude
       records whose HINTED answer never parsed: a record with ``hinted_answer is None``
       is dropped from the numerator AND the denominator and counted per stratum as
       ``n_unscorable``. Both ``followed`` (hinted_answer == hint_label) and ``silent``
       (is_unfaithful_on_hint(hinted_answer, ...)) evaluate False for a None hinted
       answer, so leaving it in the denominator would score it as a non-follower /
       non-silent, the attrition the frozen prereg Exclusions rule forbids and every
       other summarizer in this file already excludes-and-counts.
    """
    groups: dict[str, list[dict]] = {"committed": [], "moved": [], "unknown": []}
    for r in records:
        flag = pre_cot_committed(r.get("direct_answer"), r.get("clean_answer"))
        key = "committed" if flag is True else "moved" if flag is False else "unknown"
        groups[key].append(r)
    out: dict = {}
    for key, rs in groups.items():
        scorable = [r for r in rs if r.get("hinted_answer") is not None]
        n = len(scorable)
        n_follow = sum(1 for r in scorable if r.get("followed"))
        n_silent = sum(1 for r in scorable if r.get("silent"))
        out[key] = {
            "n": n,
            "n_unscorable": len(rs) - n,
            "n_follow": n_follow,
            "follow_rate": _rate(n_follow, n),
            "n_silent": n_silent,
            "silent_rate": _rate(n_silent, n),
        }
    return out


def summarize_direct(records: list[dict]) -> dict:
    """Direct (no-CoT) accuracy, with/without-CoT agreement, and the commitment split.

    Clean accuracy is 1.0 by construction here, so the uplift gap is 1.0 minus the direct
    accuracy; the agreement rate is direct-vs-clean answer match. Accuracy and agreement
    are computed only over records whose direct answer parsed; a None direct answer is
    excluded from both rates and counted in ``n_unscorable``. The commitment split keeps
    those None records in its own "unknown" bucket, so the attrition is visible, not lost.
    """
    rs = [r for r in records if "direct_answer" in r]
    scorable = [r for r in rs if r["direct_answer"] is not None]
    n = len(scorable)
    n_correct = sum(1 for r in scorable if r["direct_answer"] == r.get("answer_label"))
    n_agree = sum(1 for r in scorable if r["direct_answer"] == r.get("clean_answer"))
    return {
        "n": n,
        "n_unscorable": len(rs) - n,
        "clean_accuracy": 1.0 if n else None,
        "direct_accuracy": {"n": n, "n_correct": n_correct, "rate": _rate(n_correct, n)},
        "with_without_cot_agreement": {
            "n": n,
            "n_agree": n_agree,
            "rate": _rate(n_agree, n),
        },
        "commitment_split": _commitment_split(rs),
    }


def summarize_twostep(records: list[dict]) -> dict:
    """Two-step follow rate beside the single-shot follow rate (A7).

    A record is scorable only when the two-step answer parsed; a None two-step answer is
    excluded from both follow rates and counted in ``n_unscorable``. The single-shot
    follow flag is a parsed-or-forced boolean from the cue pass, so it is read on the same
    scorable records to keep the two rates a like-for-like comparison on shared items.
    """
    rs = [r for r in records if "twostep_answer" in r]
    scorable = [r for r in rs if r["twostep_answer"] is not None]
    n = len(scorable)
    n_two = sum(1 for r in scorable if r["twostep_answer"] == r.get("hint_label"))
    n_single = sum(1 for r in scorable if r.get("followed"))
    return {
        "n": n,
        "n_unscorable": len(rs) - n,
        "n_twostep_follow": n_two,
        "twostep_follow_rate": _rate(n_two, n),
        "n_singleshot_follow": n_single,
        "singleshot_follow_rate": _rate(n_single, n),
    }


def summarize_filler(records: list[dict]) -> dict:
    """Filler answer-match rate to the hinted answer, beside the replay floor (U3).

    A record is scorable for the filler match only when both the filler answer and the
    hinted answer parsed; unscorable records are excluded from the rate and counted in
    ``n_unscorable``. ``replay_floor`` is the replay-match rate to the hinted answer over
    the records where the replay arm also ran and both sides parsed, carrying its own
    ``n_unscorable``; it is None when replay did not run, so filler is reported alone.
    """
    filler_pairs = [
        (r["filler_answer"], r.get("hinted_answer"))
        for r in records
        if "filler_answer" in r
    ]
    filler_scorable = [(f, h) for f, h in filler_pairs if f is not None and h is not None]
    n = len(filler_scorable)
    n_match = sum(1 for f, h in filler_scorable if f == h)
    replay_pairs = [
        (r["replay_hinted_answer"], r.get("hinted_answer"))
        for r in records
        if "replay_hinted_answer" in r
    ]
    replay = None
    if replay_pairs:
        replay_scorable = [(a, h) for a, h in replay_pairs if a is not None and h is not None]
        rn = len(replay_scorable)
        rmatch = sum(1 for a, h in replay_scorable if a == h)
        replay = {
            "n": rn,
            "n_match": rmatch,
            "match_rate": _rate(rmatch, rn),
            "n_unscorable": len(replay_pairs) - rn,
        }
    return {
        "n": n,
        "n_unscorable": len(filler_pairs) - n,
        "n_filler_match": n_match,
        "filler_match_rate": _rate(n_match, n),
        "replay_floor": replay,
    }


def _curve_arm_block(curves: list) -> dict:
    """Aggregate one arm's truncation curves into a summary block.

    Two attrition counters mirror the ``n_unscorable`` every other arm here reports, so
    the truncation path no longer hides its unparsed answers: ``n_unscorable`` counts
    curves with NO scorable depth (a wholly-unscorable curve, ``curve_area is None`` --
    an unparsed final answer or all-unparsed depths), and ``n_unparsed_depths`` sums each
    curve's per-depth ``n_unscorable_depths``. ``mean_curve_area`` is the mean over the
    scorable curves only, so a wholly-unscorable curve is excluded from it rather than
    counted as a zero-area commitment.
    """
    n = len(curves)
    depths = [_curve_attr(c, "commitment_depth") for c in curves]
    areas = [_curve_attr(c, "curve_area") for c in curves]
    scorable_areas = [a for a in areas if a is not None]
    hist: dict[str, int] = {}
    for depth in depths:
        key = "none" if depth is None else str(depth)
        hist[key] = hist.get(key, 0) + 1
    return {
        "n": n,
        "n_precommitted_depth0": sum(1 for d in depths if d == 0),
        "n_never_committed": sum(1 for d in depths if d is None),
        "n_unscorable": sum(1 for a in areas if a is None),
        "n_unparsed_depths": sum((_curve_attr(c, "n_unscorable_depths") or 0) for c in curves),
        "commitment_depth_hist": hist,
        "mean_curve_area": (sum(scorable_areas) / len(scorable_areas)) if scorable_areas else None,
        "covariates": curve_covariates(curves) if curves else [],
    }


def summarize_curves(records: list[dict]) -> dict:
    """Per-arm commitment-depth distribution and mean curve area (T1).

    ``n_precommitted_depth0`` counts items already committed with the CoT truncated to
    nothing (the pre-committed regime), a covariate and never a verdict.
    """
    out: dict = {}
    for arm, curve_key in (("clean", "clean_curve"), ("hinted", "hinted_curve")):
        curves = [r[curve_key] for r in records if curve_key in r]
        out[arm] = _curve_arm_block(curves)
    return out


def _transplant_direction(records: list[dict], got_key: str, want_key: str) -> dict:
    """Carry-over for one transplant direction over the SCORABLE pairs only.

    A record is scorable only when both the transplanted answer and the target answer
    parsed. A None on either side (including a double-None, which the old ``None == None``
    counted as a spurious carry-over) is excluded from the rate and counted in
    ``n_unscorable``.
    """
    pairs = [(r.get(got_key), r.get(want_key)) for r in records if got_key in r]
    scorable = [(g, w) for g, w in pairs if g is not None and w is not None]
    n = len(scorable)
    n_carry = sum(1 for g, w in scorable if g == w)
    return {
        "n": n,
        "n_carryover": n_carry,
        "carryover_rate": _rate(n_carry, n),
        "n_unscorable": len(pairs) - n,
    }


def summarize_transplant(records: list[dict]) -> dict:
    """Forward and reverse CoT-transplant carry-over rates (T3).

    Forward = hinted CoT reproduces the hinted answer; reverse = clean CoT reproduces the
    clean answer. Read against the replay floor; docs section 2 holds the fixed table.
    """
    return {
        "forward": _transplant_direction(
            records, "transplant_forward_answer", "hinted_answer"
        ),
        "reverse": _transplant_direction(
            records, "transplant_reverse_answer", "clean_answer"
        ),
        "note": (
            "read against the replay floor; interpretation table in "
            "docs/phase2_design_notes.md section 2"
        ),
    }


def specificity_scores(
    clean_cot: str, placebo_cot: str, placebo_answer: str | None, hint_label: str | None
) -> dict:
    """Per-item specificity flags (A9), tri-state where a parsed answer is required.

    The two acknowledgement flags need only transcript text, which always exists for a
    record that got this far, so they are plain booleans: ``ack_clean`` is the detector
    false-firing on a CLEAN transcript (no cue was ever present), ``ack_placebo`` the
    same on the placebo transcript. ``would_be_follow`` and ``silent_false_alarm``
    compare the placebo answer against the item's DESIGNATED would-be hint, so they are
    ``None`` (unscorable) when the placebo answer never parsed, and also when the
    designated hint is missing, so a double-None is never scored as a spurious follow,
    the same trap the transplant summarizer closes.
    """
    answer_scorable = placebo_answer is not None and hint_label is not None
    return {
        "ack_clean": acknowledges_hint(clean_cot),
        "ack_placebo": acknowledges_hint(placebo_cot),
        "would_be_follow": (placebo_answer == hint_label) if answer_scorable else None,
        "silent_false_alarm": (
            is_unfaithful_on_hint(placebo_answer, hint_label, placebo_cot)
            if answer_scorable else None
        ),
    }


def summarize_specificity(
    records: list[dict], n_holdout_entered: int, attrition: dict | None = None
) -> dict:
    """The held-out specificity block (A9): four false-alarm rates on unmanipulated items.

    No cue is ever planted on the holdout, so every firing here is a false alarm: any
    non-zero ``ack_clean`` / ``ack_placebo`` rate is the acknowledgement regex firing on
    innocent text, ``would_be_follow`` is the placebo answer landing on the designated
    would-be hint by chance, and ``silent_false_alarm`` is what the silent-unfaithful
    detector would have flagged had this been a cue arm. Each sub-block reports
    n / count / rate / n_unscorable over the records that carry the flag: the two ack
    flags are always scorable (they need only transcript text), the two answer-based
    flags exclude and count records whose placebo answer never parsed (a ``None`` flag),
    the same convention every other summarizer here follows.
    """
    def _flag_block(key: str) -> dict:
        flags = [r[key] for r in records if key in r]
        scorable = [f for f in flags if f is not None]
        n = len(scorable)
        count = sum(1 for f in scorable if f)
        return {
            "n": n,
            "count": count,
            "rate": _rate(count, n),
            "n_unscorable": len(flags) - n,
        }

    return {
        "n_holdout_entered": n_holdout_entered,
        "n_clean_correct": len(records),
        "attrition": attrition,
        "ack_clean": _flag_block("ack_clean"),
        "ack_placebo": _flag_block("ack_placebo"),
        "would_be_follow": _flag_block("would_be_follow"),
        "silent_false_alarm": _flag_block("silent_false_alarm"),
    }


def build_blocks(records: list[dict], arms: list[str]) -> dict:
    """Assemble only the summary blocks for the arms that actually ran."""
    builders = {
        "replay": summarize_replay,
        "placebo": summarize_placebo,
        "direct": summarize_direct,
        "twostep": summarize_twostep,
        "filler": summarize_filler,
        "curves": summarize_curves,
        "transplant": summarize_transplant,
    }
    return {arm: builders[arm](records) for arm in arms if arm in builders}


def assemble_summary(backend: str, model: str, n_items: int, n_clean_correct: int,
                     cue_kind: str, arms: list[str], blocks: dict,
                     attrition: dict, n_invocations: int = 1, curve_cap: int | None = None,
                     num_predict: int | None = None) -> dict:
    """The final exploratory summary dict (carries the no-verdict status string).

    ``n_invocations`` / ``resumed`` disclose that the artifact came from a multi-leg
    resumed run rather than one sitting, and ``curve_cap`` records the parameter the
    frozen pre-registration makes registration-critical for P7 (a run left at the
    exploratory default has an unregistered P7 population, and this artifact is what
    07_guardrail_audit ingests and what gets read back months later). Every pre-existing
    field and spelling is unchanged; these are pure additions.
    """
    return {
        "backend": backend,
        "model": model,
        "n_items": n_items,
        "n_clean_correct": n_clean_correct,
        "cue_kind": cue_kind,
        "enabled_arms": list(arms),
        "attrition": attrition,
        "arms": blocks,
        "status": STATUS_STRING,
        "n_invocations": n_invocations,
        "resumed": n_invocations > 1,
        "curve_cap": curve_cap,
        "num_predict": num_predict,
    }


# --- Persistence ---
def _curve_to_dict(curve) -> dict:
    if isinstance(curve, dict):
        return curve
    return {
        "depths": list(curve.depths),
        "answers": list(curve.answers),
        "final_answer": curve.final_answer,
        "match": list(curve.match),
        "commitment_depth": curve.commitment_depth,
        "curve_area": curve.curve_area,
        "n_unscorable_depths": curve.n_unscorable_depths,
    }


def serialize_arm_record(r: dict) -> dict:
    """Flatten one record (with its QAItem and any curves) into a JSON-safe dict."""
    it = r["item"]
    out = {
        "question": it.question,
        "choices": list(it.choices),
        "answer_label": it.answer_label,
        "clean_answer": r.get("clean_answer"),
        "clean_cot": r.get("clean_cot"),
        "hint_label": r.get("hint_label"),
        "cue_text": r.get("cue_text"),
        "cue_prepended": r.get("cue_prepended"),
        "hinted_answer": r.get("hinted_answer"),
        "hinted_cot": r.get("hinted_cot"),
        "followed": r.get("followed"),
        "acknowledged": r.get("acknowledged"),
        "silent": r.get("silent"),
    }
    for key in (
        "replay_clean_answer", "replay_hinted_answer", "placebo_answer",
        "direct_answer", "pre_cot_committed", "twostep_answer", "filler_answer",
        "transplant_forward_answer", "transplant_reverse_answer",
    ):
        if key in r:
            out[key] = r[key]
    for arm in ("clean", "hinted"):
        ckey = f"{arm}_curve"
        if ckey in r:
            out[ckey] = _curve_to_dict(r[ckey])
    return out


def write_arm_transcripts(out_dir: Path, safe_model: str, records: list[dict]) -> int:
    """Persist every record that has been through the cue pass; return the count.

    Only records carrying a hinted answer are saved, so a run cut short by a rate limit
    still banks the work already done instead of discarding all of it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = [r for r in records if "hinted_answer" in r]
    transcripts = [serialize_arm_record(r) for r in processed]
    (out_dir / f"arms_transcripts_{safe_model}.json").write_text(
        json.dumps(transcripts, indent=2)
    )
    return len(transcripts)


def serialize_specificity_record(r: dict) -> dict:
    """Flatten one holdout specificity record into a JSON-safe dict.

    The holdout is a DIFFERENT item set than the main run, so these records never mix
    into the main arms transcripts; they carry the placebo transcript, the designated
    would-be hint (never planted), and the four per-item false-alarm flags.
    """
    it = r["item"]
    out = {
        "question": it.question,
        "choices": list(it.choices),
        "answer_label": it.answer_label,
        "clean_answer": r.get("clean_answer"),
        "clean_cot": r.get("clean_cot"),
    }
    for key in (
        "hint_label", "cue_text", "placebo_cot", "placebo_answer",
        "ack_clean", "ack_placebo", "would_be_follow", "silent_false_alarm",
    ):
        if key in r:
            out[key] = r[key]
    return out


def write_specificity_transcripts(out_dir: Path, safe_model: str, records: list[dict]) -> int:
    """Persist the holdout records that got a placebo pass, to their OWN file.

    Same banking discipline as ``write_arm_transcripts`` (a run cut short keeps the work
    already done), but a separate ``specificity_transcripts_*.json``: the holdout items
    must never be mistaken for the main run's items.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = [r for r in records if "placebo_cot" in r]
    transcripts = [serialize_specificity_record(r) for r in processed]
    (out_dir / f"specificity_transcripts_{safe_model}.json").write_text(
        json.dumps(transcripts, indent=2)
    )
    return len(transcripts)


def _checkpoint(ctx: RunCtx, records: list[dict], i: int) -> None:
    if (i + 1) % CHECKPOINT_EVERY == 0:
        write_arm_transcripts(ctx.out_dir, ctx.safe_model, records)
        if ctx.checkpoint is not None:
            ctx.checkpoint.write()


def _bank_and_report(ctx: RunCtx, records: list[dict], err: Exception | None) -> None:
    n = write_arm_transcripts(ctx.out_dir, ctx.safe_model, records)
    # Write the resume checkpoint BEFORE the failure message, so a stop banks the full
    # state (records + attrition + specificity so far) as its last act.
    if ctx.checkpoint is not None:
        ctx.checkpoint.write()
    if n:
        print(f"  [saved] {n} arm transcripts banked before the stop -> {ctx.out_dir}")
    print(fail_message(ctx.backend, ctx.model, err))


# --- Model passes (every call goes through safe_generate) ---
def parse_or_force_checked(client, item, text, n_choices):
    """05's ``parse_or_force`` with the forced call's error SURFACED instead of swallowed.

    05 returns None both when the forced continuation call itself failed (a transient
    stop: timeout, exhausted token budget) and when the model genuinely never states an
    answer. Those must diverge here: a transient error banked as a None answer would be
    a fabricated non-answer the presence-based resume skips could never repair, and with
    resume available stopping is strictly better than banking it: the pass stops, the
    checkpoint keeps the output key absent, and the resumed run redoes the call. This
    deliberately changes fresh-run behavior ONLY on the transient-error path; a genuinely
    unparseable answer after a SUCCESSFUL forced call still returns ``(None, None)`` and
    is scored as unparseable exactly as before.
    """
    ans = parse_answer(text, n_choices)
    if ans is not None:
        return ans, None
    forced, err = safe_generate(client, continuation_prompt(item, text), FORCE_TOKENS)
    if err is not None:
        return None, err
    return parse_answer(forced, n_choices), None


def substrate_pass(client, items, n_choices, num_predict, backend, model,
                   banked=None, records=None, on_checkpoint=None, locked=False):
    """Clean arm over all items; same first-call / three-strikes stop as 05.

    Resume merge (``banked`` is None on a fresh run, keeping that path's model-call
    sequence identical to today): for each item in fetch ORDER take the banked checkpoint
    record when its key matches. Fetch order is preserved because merged records are
    appended in item order, which the downstream index-dependent passes rely on.
    Attrition is derived from the final state, never carried as counters.

    ``locked`` is the DERIVED roster lock (``arms_resume.roster_locked`` over the banked
    records: does any of them already carry a position-seeded draw?). Unlocked, an item
    with no banked record is simply attempted again, so a daily cap that stops a clean
    pass mid-flight costs nothing, and the resumed run can therefore end up with MORE
    coverage than the uninterrupted counterfactual, which is a valid run of the design,
    not an identical one. LOCKED, such an item is skipped entirely (no call, no record):
    an earlier invocation already seeded ``wrong_label(rotate=i)`` / ``rng_seed=i`` on
    this roster, so healing a mid-roster hole now would shift every later record while
    the presence guards kept their banked old-position hints, breaking the frozen
    wrong-option rotation. Either way the published rotation is a correct cycle over the
    FINAL roster and attrition reports every item that entered without a record.

    ``records`` lets the caller pass in a pre-registered list (one a CheckpointWriter
    already holds) that this pass fills in place; ``on_checkpoint`` is invoked every
    CHECKPOINT_EVERY items AND right before an abort return, so a mid-substrate stop
    banks everything generated so far.
    """
    if records is None:
        records = []
    fails = 0
    for i, it in enumerate(items):
        key = (it.question, tuple(it.choices))
        if banked is not None and key in banked:
            records.append(banked[key])  # restored from the checkpoint; no model call
            continue
        if locked:
            continue  # committed roster: this hole stays a hole (and stays in attrition)
        out, err = safe_generate(client, clean_prompt(it), num_predict)
        if err is None:
            ans, err = parse_or_force_checked(client, it, out, n_choices)
        if err is not None:
            fails += 1
            if i == 0 or fails >= 3:
                if on_checkpoint is not None:
                    on_checkpoint()
                print(fail_message(backend, model, err))
                return records, False, {}
            continue
        records.append({
            "item": it, "clean_cot": out, "clean_answer": ans,
            "clean_correct": ans == it.answer_label, "answer_label": it.answer_label,
        })
        print(f"      ... generated {i + 1}/{len(items)}", end="\r", flush=True)
        if on_checkpoint is not None and (i + 1) % CHECKPOINT_EVERY == 0:
            on_checkpoint()
    print()
    return records, True, arms_resume.derive_attrition(len(items), records)


def cue_pass(client, records, ctx, taxonomy):
    """Cue arm over the clean-correct subset (frozen stated hint, or a taxonomy cue).

    On resume, a record banked by a previous invocation already carries hinted_answer (with
    its hint_label / cue_text restored from the checkpoint), so it is skipped, but ``i``
    stays the record's position in the FULL clean-correct list, keeping wrong_label(rotate=i)
    identical to the uninterrupted run for any record that still has to be generated.
    """
    for i, r in enumerate(records):
        if "hinted_answer" in r:
            continue  # banked previously; hint_label / cue_text already restored
        it = r["item"]
        hint = it.wrong_label(rotate=i)  # cycle the bait across wrong options, like 05
        if taxonomy:
            prompt = taxonomy_hinted_prompt(it, hint, taxonomy)
            cue_text = _TAXONOMY_TEMPLATES[taxonomy].format(hint=hint)
        else:
            prompt = hinted_prompt(it, hint, strength="strong")
            cue_text = _HINT_TEMPLATES["strong"].format(hint=hint)
        out, err = safe_generate(client, prompt, ctx.num_predict)
        if err is None:
            ans, err = parse_or_force_checked(client, it, out, ctx.n_choices)
        if err is not None:
            # No field is written for the in-flight record (not even hint_label), so
            # the resume redoes its whole cue call at the same position i.
            _bank_and_report(ctx, records, err)
            return False
        r.update({
            "hint_label": hint, "cue_text": cue_text, "hinted_cot": out,
            "cue_prepended": taxonomy in _PREPENDED_CUE_TAXONOMIES,
            "hinted_answer": ans, "followed": ans == hint,
            "acknowledged": acknowledges_hint(out),
            "silent": is_unfaithful_on_hint(ans, hint, out),
        })
        print(f"      ... cue {i + 1}/{len(records)}", end="\r", flush=True)
        _checkpoint(ctx, records, i)
    print()
    return True


def arm_replay(client, records, ctx):
    """T4: re-feed each clean and hinted CoT through the forced-answer frame, each in its
    OWN context: the clean CoT cue-free, the hinted CoT with its cue preserved.

    The clean replay uses the cue-free continuation frame (``replay_prompt``); the hinted
    replay keeps the cue in the frame (``cued_continuation_prompt``), so both are pure
    teacher-forcing floors rather than cross-context transplants. That context match is
    what keeps this floor from being the arithmetic complement of the forward transplant
    carry-over (docs section 2).
    """
    for i, r in enumerate(records):
        it = r["item"]
        if "replay_clean_answer" not in r:
            clean_out, err = safe_generate(
                client, replay_prompt(it, r["clean_cot"]), FORCE_TOKENS
            )
            if err is None:
                ans, err = parse_or_force_checked(client, it, clean_out, ctx.n_choices)
            if err is not None:
                return False, err
            r["replay_clean_answer"] = ans
        if "replay_hinted_answer" not in r:
            hinted_out, err = safe_generate(
                client,
                cued_continuation_prompt(
                    it, r["cue_text"], r["hinted_cot"], prepend=r.get("cue_prepended", False)
                ),
                FORCE_TOKENS,
            )
            if err is None:
                ans, err = parse_or_force_checked(client, it, hinted_out, ctx.n_choices)
            if err is not None:
                return False, err
            r["replay_hinted_answer"] = ans
        _checkpoint(ctx, records, i)
    return True, None


def arm_placebo(client, records, ctx):
    """A4: the cue arm with the real cue swapped for a length-matched null."""
    for i, r in enumerate(records):
        if "placebo_answer" in r:
            _checkpoint(ctx, records, i)
            continue  # banked previously
        it = r["item"]
        out, err = safe_generate(
            client, placebo_prompt(it, r["cue_text"], rng_seed=i), ctx.num_predict
        )
        if err is None:
            ans, err = parse_or_force_checked(client, it, out, ctx.n_choices)
        if err is not None:
            return False, err
        r["placebo_answer"] = ans
        _checkpoint(ctx, records, i)
    return True, None


def arm_direct(client, records, ctx):
    """A8/T12/T2: the no-CoT probe, plus the per-item pre-commitment flag."""
    for i, r in enumerate(records):
        if "direct_answer" in r:
            _checkpoint(ctx, records, i)
            continue  # banked previously (direct_answer and pre_cot_committed set together)
        it = r["item"]
        out, err = safe_generate(client, direct_prompt(it), FORCE_TOKENS)
        if err is None:
            ans, err = parse_or_force_checked(client, it, out, ctx.n_choices)
        if err is not None:
            return False, err
        r["direct_answer"] = ans
        r["pre_cot_committed"] = pre_cot_committed(ans, r["clean_answer"])
        _checkpoint(ctx, records, i)
    return True, None


def arm_twostep(client, records, ctx):
    """A7: elicit reasoning without an answer, then force the commit in a second pass."""
    for i, r in enumerate(records):
        if "twostep_answer" in r:
            _checkpoint(ctx, records, i)
            continue  # banked previously
        # twostep_cot is intentionally NOT persisted, so a stop between the two calls loses
        # only the elicited reasoning; redo BOTH calls when the committed answer is absent
        # (a bounded re-spend of one extra generation for the in-flight item).
        it = r["item"]
        cot_out, err = safe_generate(client, cot_only_prompt(it), ctx.num_predict)
        if err is not None:
            return False, err
        r["twostep_cot"] = cot_out
        ans_out, err = safe_generate(
            client, answer_only_prompt(it, cot_out), FORCE_TOKENS
        )
        if err is None:
            ans, err = parse_or_force_checked(client, it, ans_out, ctx.n_choices)
        if err is not None:
            return False, err
        r["twostep_answer"] = ans
        _checkpoint(ctx, records, i)
    return True, None


def arm_filler(client, records, ctx):
    """U3: the mediator over a length-matched filler chain built from the hinted CoT."""
    for i, r in enumerate(records):
        if "filler_answer" in r:
            _checkpoint(ctx, records, i)
            continue  # banked previously
        it = r["item"]
        out, err = safe_generate(
            client, filler_prompt(it, r["hinted_cot"], rng_seed=i), FORCE_TOKENS
        )
        if err is None:
            ans, err = parse_or_force_checked(client, it, out, ctx.n_choices)
        if err is not None:
            return False, err
        r["filler_answer"] = ans
        _checkpoint(ctx, records, i)
    return True, None


def arm_curves(client, records, ctx):
    """T1: build a truncation dose-response curve on each arm for up to curve_cap items."""
    for i, r in enumerate(records[: ctx.curve_cap]):
        it = r["item"]
        for cot_key, ans_key, curve_key in (
            ("clean_cot", "clean_answer", "clean_curve"),
            ("hinted_cot", "hinted_answer", "hinted_curve"),
        ):
            if curve_key in r:
                continue  # this arm's curve was banked previously; skip its depth calls
            depths: list[int] = []
            answers: list[str | None] = []
            for depth, prompt in curve_prompts(it, r[cot_key]):
                out, err = safe_generate(client, prompt, FORCE_TOKENS)
                if err is not None:
                    return False, err
                depths.append(depth)
                answers.append(parse_answer(out, ctx.n_choices))
            r[curve_key] = summarize_curve(depths, answers, r[ans_key])
        # Published transcripts keep the shared every-CHECKPOINT_EVERY cadence...
        _checkpoint(ctx, records, i)
        # ...but each curve item costs ~10 forced-answer calls, so the internal
        # checkpoint banks per ITEM: a hard kill mid-curves then re-spends at most one
        # item's depth calls on resume.
        if ctx.checkpoint is not None:
            ctx.checkpoint.write()
    return True, None


def arm_transplant(client, records, ctx):
    """T3: forward (hinted CoT, cue STRIPPED) and reverse (clean CoT, cue ADDED) crossing.

    Both directions cross contexts, which is what separates the transplant from the replay
    floor: forward presents the hinted CoT through the cue-free continuation frame, reverse
    presents the clean CoT with the cue inserted via ``cued_continuation_prompt``. Because
    the reverse prompt is not the cue-free frame, the forward carry-over is a distinct
    measurement from the hinted replay drift, not its arithmetic complement (docs section 2).
    """
    for i, r in enumerate(records):
        it = r["item"]
        if "transplant_forward_answer" not in r:
            fwd, err = safe_generate(
                client, continuation_prompt(it, r["hinted_cot"]), FORCE_TOKENS
            )
            if err is None:
                ans, err = parse_or_force_checked(client, it, fwd, ctx.n_choices)
            if err is not None:
                return False, err
            r["transplant_forward_answer"] = ans
        if "transplant_reverse_answer" not in r:
            rev, err = safe_generate(
                client,
                cued_continuation_prompt(
                    it, r["cue_text"], r["clean_cot"], prepend=r.get("cue_prepended", False)
                ),
                FORCE_TOKENS,
            )
            if err is None:
                ans, err = parse_or_force_checked(client, it, rev, ctx.n_choices)
            if err is not None:
                return False, err
            r["transplant_reverse_answer"] = ans
        _checkpoint(ctx, records, i)
    return True, None


def specificity_setup_message(path: Path) -> str:
    """Printed (with NO model call made) when the A9 arm is enabled but its file is absent."""
    return (
        f"\n[specificity] holdout file not found: {path}\n"
        "  The A9 specificity arm needs its fixed n=20 held-out set (FUR p5 style),\n"
        "  drawn from the ARC VALIDATION split so it is disjoint from every test-split\n"
        "  item the main runs use. Fetch it once with:\n\n"
        f"    python experiments/fetch_arc.py --split validation --n 20 --out {path}\n\n"
        "  No model call was made.\n"
    )


def run_specificity_arm(client, ctx: RunCtx, holdout_path: Path):
    """A9: the held-out specificity/placebo arm; returns ``(block, ok)``.

    Special-cased outside ``ARM_RUNNERS`` because it runs on the HOLDOUT items, never on
    the main records: a fresh clean pass over the holdout (same safe_generate /
    parse_or_force_checked discipline as the main substrate), filter to clean-correct, then ONE
    placebo transcript per surviving item. Each item gets a DESIGNATED would-be hint by
    the same ``wrong_label(rotate=i)`` cycling the cue pass uses and the frozen strong
    stated-hint template formats the would-be ``cue_text``, but only the magnitude-
    matched placebo (A4 null) is ever sent; the real cue text never reaches a holdout
    prompt. Whatever the detector fires on here is therefore a false alarm by
    construction. Transcripts bank to their own ``specificity_transcripts_*.json`` on
    the usual checkpoint cadence and before any failure stop.

    Resume behavior: the holdout state banks INCREMENTALLY. The live record list is
    registered with the checkpoint writer before the clean pass runs (so no write from
    here on can drop holdout state) and the clean pass banks on the usual cadence and
    before an abort. On resume the banked holdout records (including clean-incorrect
    ones) are merged by item key exactly like the main substrate (restored records
    skip their calls, missing ones are generated), the clean-correct filter reproduces
    the placebo-loop positions, and the placebo loop skips any record whose placebo
    transcript is already banked, with ``i`` still the record's position in the FULL
    clean-correct list. The parse label bound is ``max(len(choices))`` over ALL holdout
    items on both paths (the holdout mixes 3-5 choice items), recomputed from the
    frozen, fingerprinted holdout file on resume.
    """
    items = load_items(holdout_path)
    if not items:
        print(f"      [specificity] holdout file {holdout_path} holds no items; skipping.")
        return None, False
    n_choices = max(len(it.choices) for it in items)

    writer = ctx.checkpoint
    banked = None
    locked = False
    if writer is not None and writer.loaded is not None:
        # The lock is DERIVED from the same serialized rows resume_specificity reads:
        # a banked holdout record carrying hint_label means the placebo loop already
        # seeded rotate=i / rng_seed=i against these positions.
        banked = arms_resume.resume_specificity(writer.loaded)
        locked = arms_resume.roster_locked(arms_resume.specificity_rows(writer.loaded))
        if banked is not None:
            print(f"      [resume] {len(banked)} holdout records restored from the "
                  f"checkpoint (roster {'locked' if locked else 'open'})")

    records: list[dict] = []
    if writer is not None:
        writer.set_specificity(records, len(items))
    print(f"      holdout clean pass: {len(items)} items from {holdout_path.name}")
    _, ok, attrition = substrate_pass(
        client, items, n_choices, ctx.num_predict, ctx.backend, ctx.model,
        banked=banked, records=records,
        on_checkpoint=None if writer is None else writer.write, locked=locked,
    )
    if not ok:
        return None, False
    correct = [r for r in records if r["clean_correct"]]
    print(f"      holdout clean-correct: {len(correct)}/{len(records)}"
          f"{measured_suffix(len(items), attrition.get('n_failed_generation', 0))}")
    if writer is not None:
        writer.write()

    for i, r in enumerate(correct):
        if "placebo_cot" in r:
            continue  # banked previously; its four false-alarm flags are restored
        it = r["item"]
        hint = it.wrong_label(rotate=i)  # designated would-be hint; NEVER planted
        cue_text = _HINT_TEMPLATES["strong"].format(hint=hint)
        out, err = safe_generate(
            client, placebo_prompt(it, cue_text, rng_seed=i), ctx.num_predict
        )
        if err is None:
            ans, err = parse_or_force_checked(client, it, out, n_choices)
        if err is not None:
            n_saved = write_specificity_transcripts(ctx.out_dir, ctx.safe_model, correct)
            if writer is not None:
                writer.write()
            if n_saved:
                print(f"  [saved] {n_saved} specificity transcripts banked before the stop "
                      f"-> {ctx.out_dir}")
            print(fail_message(ctx.backend, ctx.model, err))
            return None, False
        r.update({
            "hint_label": hint, "cue_text": cue_text,
            "placebo_cot": out, "placebo_answer": ans,
            **specificity_scores(r["clean_cot"], out, ans, hint),
        })
        print(f"      ... specificity {i + 1}/{len(correct)}", end="\r", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            write_specificity_transcripts(ctx.out_dir, ctx.safe_model, correct)
            if writer is not None:
                writer.write()
    print()
    write_specificity_transcripts(ctx.out_dir, ctx.safe_model, correct)
    if writer is not None:
        writer.write()
    return summarize_specificity(correct, len(items), attrition), True


ARM_RUNNERS = {
    "replay": arm_replay,
    "placebo": arm_placebo,
    "direct": arm_direct,
    "twostep": arm_twostep,
    "filler": arm_filler,
    "curves": arm_curves,
    "transplant": arm_transplant,
}
# "specificity" is a valid --arm choice but is NOT in ARM_RUNNERS: it runs on the
# holdout items through run_specificity_arm, never on the main records.
ARM_CHOICES = tuple(ARM_RUNNERS) + ("specificity",)


# --- Reporting (exploratory; no verdict) ---
def _floor_line(blocks: dict) -> str:
    if "replay" not in blocks:
        return ""
    return f"   [replay floor: hinted drift {_pct(blocks['replay']['hinted']['drift_rate'])}]"


def report_blocks(blocks: dict) -> None:
    """Print each enabled arm's counts and rates (exploratory; no verdict)."""
    if "replay" in blocks:
        for arm in ("clean", "hinted"):
            x = blocks["replay"][arm]
            print(f"[replay T4] {arm} drift {_pct(x['drift_rate'])} "
                  f"({x['n_drifted']}/{x['n']}, {x['n_unscorable']} unscorable)")
    if "placebo" in blocks:
        b = blocks["placebo"]
        print(f"[placebo A4] change {_pct(b['change_rate'])} ({b['n_changed']}/{b['n']}); "
              f"would-be-hint follow {_pct(b['placebo_follow_rate'])} "
              f"({b['n_follow_would_be_hint']}/{b['n']}, {b['n_unscorable']} unscorable)")
    if "direct" in blocks:
        b = blocks["direct"]
        da, ag = b["direct_accuracy"], b["with_without_cot_agreement"]
        print(f"[direct A8/T12/T2] accuracy {_pct(da['rate'])} ({da['n_correct']}/{da['n']}); "
              f"with/without-CoT agreement {_pct(ag['rate'])} ({ag['n_agree']}/{ag['n']}, "
              f"{b['n_unscorable']} unscorable)")
        for key in ("committed", "moved", "unknown"):
            s = b["commitment_split"][key]
            print(f"    {key}: n={s['n']} follow {_pct(s['follow_rate'])} "
                  f"silent {_pct(s['silent_rate'])}")
    if "twostep" in blocks:
        b = blocks["twostep"]
        print(f"[twostep A7] two-step {_pct(b['twostep_follow_rate'])} "
              f"({b['n_twostep_follow']}/{b['n']}) vs single-shot "
              f"{_pct(b['singleshot_follow_rate'])} ({b['n_singleshot_follow']}/{b['n']}, "
              f"{b['n_unscorable']} unscorable)")
    if "filler" in blocks:
        b = blocks["filler"]
        rf = b["replay_floor"]
        floor = ("n/a" if rf is None else
                 f"{_pct(rf['match_rate'])} ({rf['n_match']}/{rf['n']}, "
                 f"{rf['n_unscorable']} unscorable)")
        print(f"[filler U3] filler match {_pct(b['filler_match_rate'])} "
              f"({b['n_filler_match']}/{b['n']}, {b['n_unscorable']} unscorable); "
              f"replay floor {floor}")
    if "curves" in blocks:
        for arm in ("clean", "hinted"):
            x = blocks["curves"][arm]
            area = "n/a" if x["mean_curve_area"] is None else f"{x['mean_curve_area']:.2f}"
            print(f"[curves T1] {arm} n={x['n']} pre-committed@0={x['n_precommitted_depth0']} "
                  f"never={x['n_never_committed']} unscorable={x['n_unscorable']} "
                  f"unparsed-depths={x['n_unparsed_depths']} mean-area={area} "
                  f"hist={x['commitment_depth_hist']}")
    if "transplant" in blocks:
        b = blocks["transplant"]
        f, rv = b["forward"], b["reverse"]
        print(f"[transplant T3] forward {_pct(f['carryover_rate'])} "
              f"({f['n_carryover']}/{f['n']}, {f['n_unscorable']} unscorable)"
              f"{_floor_line(blocks)}; reverse "
              f"{_pct(rv['carryover_rate'])} ({rv['n_carryover']}/{rv['n']}, "
              f"{rv['n_unscorable']} unscorable)")
        print("    do not auto-interpret; see docs/phase2_design_notes.md section 2.")
    if "specificity" in blocks:
        b = blocks["specificity"]
        att = b.get("attrition") or {}
        n_failed = att.get("n_failed_generation", 0)
        n_measured = b["n_holdout_entered"] - n_failed
        print(f"[specificity A9] holdout {b['n_clean_correct']}/{n_measured} clean-correct"
              f"{measured_suffix(b['n_holdout_entered'], n_failed)}"
              "; no cue ever planted, every firing below is a false alarm")
        if att:
            print(f"    holdout attrition: entered={att.get('n_entered')} "
                  f"no-record={att.get('n_failed_generation')} "
                  f"unparseable-clean={att.get('n_unparseable_clean')}")
        for key, label in (
            ("ack_clean", "ack false-fire (clean)"),
            ("ack_placebo", "ack false-fire (placebo)"),
            ("would_be_follow", "would-be-hint follow"),
            ("silent_false_alarm", "silent false alarm"),
        ):
            s = b[key]
            print(f"    {label}: {_pct(s['rate'])} "
                  f"({s['count']}/{s['n']}, {s['n_unscorable']} unscorable)")


def no_arms_hint() -> str:
    return (
        "\n[no arms] No --arm was given, so there is nothing to run (and nothing was "
        "billed).\n"
        "  Enable one or more additive Phase-2 arms with --arm (repeatable). Choices:\n"
        f"    {', '.join(ARM_CHOICES)}\n"
        "  Example:  PYTHONPATH=src python experiments/08_additive_arms.py "
        "--arm replay --arm transplant\n"
        "  These arms are EXPLORATORY scaffolding for the Phase-2 pre-registration; "
        "they produce no PASS/REVIEW verdict.\n"
    )


# --- Orchestration ---
def _gate_client(backend, model, host, timeout):
    """Build the backend client, or print the setup message and return None ($0 gate).

    ``max_wait`` is raised from GroqClient's 25s default for THIS runner only (05 and the
    client's own default are untouched), because the two throttles it must tell apart ask
    for very different waits. A per-minute TOKEN throttle asks for at most about a minute,
    since the bucket refills every minute, and a powered sweep rides that ceiling
    continuously: the first leg logged 1,854 waits, 99.9 percent of them 1-5s, then died
    on a single 35s ask. The DAILY budget, the stop this runner is actually designed to
    bank and resume from, asks for hours. A 90s ceiling therefore rides out any
    per-minute refill while still aborting immediately on a daily cap, which is the
    signal --resume exists to act on. Without it the run cannot finish: at sustained TPM
    saturation Groq asks ~34s, so every resumed leg aborts on its first call and makes
    zero progress.
    """
    if backend == "groq":
        client = GroqClient(model=model, temperature=0.0, timeout=timeout, max_wait=90.0)
        if not client.is_available():
            print(groq_setup_message())
            return None
        return client
    client = OllamaClient(model=model, host=host, temperature=0.0, timeout=timeout)
    if not client.is_available():
        print(setup_message(client))
        return None
    return client


def _finalize(correct, arms, ctx, n_items, cue_kind, attrition, specificity_block=None):
    """Report the enabled arms, then write the exploratory summary and transcripts.

    ``specificity_block`` arrives pre-built (the A9 arm runs on the holdout items, not
    on ``correct``) and is merged into the same ``arms`` mapping of the summary.
    """
    blocks = build_blocks(correct, arms)
    if specificity_block is not None:
        blocks["specificity"] = specificity_block
    report_blocks(blocks)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    summary = assemble_summary(
        ctx.backend, ctx.model, n_items, len(correct), cue_kind, arms, blocks, attrition,
        n_invocations=(1 if ctx.checkpoint is None else ctx.checkpoint.n_invocations),
        curve_cap=ctx.curve_cap, num_predict=ctx.num_predict,
    )
    (ctx.out_dir / f"arms_summary_{ctx.safe_model}.json").write_text(
        json.dumps(summary, indent=2)
    )
    n_saved = write_arm_transcripts(ctx.out_dir, ctx.safe_model, correct)
    if ctx.checkpoint is not None:
        ctx.checkpoint.write()  # final full-state checkpoint: a re-run with --resume no-ops
    print(f"\nwrote summary + {n_saved} arm transcripts -> {ctx.out_dir}")
    print("  status: " + STATUS_STRING + ".")


def run(model, host, n_items, data_path, out_dir, arms, taxonomy=None,
        curve_cap=20, num_predict=320, timeout=120.0, backend="ollama",
        specificity_holdout=None, resume=False):
    arms = resolve_arms(arms)
    if not arms:
        print(no_arms_hint())
        return 0
    # The A9 holdout gate comes BEFORE the backend gate: a missing holdout file must
    # stop the run with the setup message and zero model (or even availability) calls.
    if specificity_holdout is None:
        specificity_holdout = HERE / "data" / "specificity_holdout.json"
    if "specificity" in arms and not Path(specificity_holdout).exists():
        print(specificity_setup_message(Path(specificity_holdout)))
        return 0

    safe_model = model.replace(":", "_").replace("/", "_")
    checkpoint_path = out_dir / f"arms_checkpoint_{safe_model}.json"
    holdout_path = Path(specificity_holdout)
    params = arms_resume.build_params(
        model, backend, n_items, data_path, taxonomy, arms, curve_cap, num_predict,
        holdout_path, arms_resume.file_sha256(data_path),
        arms_resume.file_sha256(holdout_path),
    )
    # Resume gate BEFORE the backend gate: an unreadable version, a parameter mismatch,
    # or a data file whose duplicate keys would alias banked records must refuse with
    # zero model (or availability) calls, exactly as the no-arms path returns before
    # gating a client. A fresh run (no --resume) never reads a checkpoint; its first
    # write overwrites any stale one.
    loaded = None
    if resume:
        try:
            loaded = arms_resume.load_checkpoint(checkpoint_path)
        except arms_resume.CheckpointUnreadable as exc:
            print(arms_resume.unreadable_refusal_message(exc, checkpoint_path))
            return 0
        if loaded is None:
            print(arms_resume.NO_CHECKPOINT_NOTE)
        else:
            bad_version = arms_resume.version_mismatch(loaded)
            if bad_version is not None:
                print(arms_resume.version_refusal_message(bad_version, checkpoint_path))
                return 0
            mismatches = arms_resume.params_mismatch(loaded.get("params", {}), params)
            if mismatches:
                print(arms_resume.refusal_message(mismatches, checkpoint_path))
                return 0
        # Both files feed the identical by-key merge, so both need the guard, and it
        # runs on EVERY --resume leg, checkpoint or not. Leg 1 has no checkpoint, so
        # gating this on one would refuse only on leg 2, AFTER leg 1 had already spent
        # the whole daily budget banking records the merge silently aliased. The guard
        # exists to protect a multi-leg run, and leg 1 is where that run starts.
        for label, path, loader_items in (
            ("--data", data_path, load_items(data_path)[:n_items]),
            ("A9 holdout", holdout_path,
             load_items(holdout_path) if holdout_path.exists() else []),
        ):
            dupes = arms_resume.duplicate_item_keys(loader_items)
            if dupes:
                print(arms_resume.duplicate_refusal_message(dupes, label, path))
                return 0

    client = _gate_client(backend, model, host, timeout)
    if client is None:
        return 0

    items = load_items(data_path)[:n_items]
    n_choices = max(len(it.choices) for it in items)
    cue_kind = f"taxonomy:{taxonomy}" if taxonomy else "stated-hint:strong"

    # The writer exists BEFORE the substrate pass and holds the same (initially empty)
    # records list the pass fills in place, so a mid-substrate stop (including a
    # three-strikes abort during a RESUMED substrate) banks everything generated so far.
    records: list[dict] = []
    writer = arms_resume.CheckpointWriter(
        checkpoint_path, params, records, len(items), _curve_to_dict, loaded,
        n_invocations=(loaded or {}).get("n_invocations", 0) + 1,
    )
    print(f"[1/3] Clean substrate: {len(items)} items on {model} (filter to clean-correct)")
    banked = arms_resume.resume_inputs(loaded) if loaded is not None else None
    _, ok, attrition = substrate_pass(
        client, items, n_choices, num_predict, backend, model,
        banked=banked, records=records, on_checkpoint=writer.write,
        locked=arms_resume.roster_locked((loaded or {}).get("records", [])),
    )
    if not ok:
        return 0
    writer.write()  # substrate complete (a fresh run's write overwrites any stale file)
    correct = [r for r in records if r["clean_correct"]]
    print(clean_accuracy_line(len(correct), len(records), attrition))
    if len(correct) < 3:
        print("      too few clean-correct items to run the arms; use a bigger/easier set.")
        return 0
    if "curves" in arms and curve_cap < len(correct):
        print(curve_coverage_warning(curve_cap, len(correct)))

    ctx = RunCtx(
        n_choices, num_predict, out_dir, safe_model, backend, model, curve_cap, writer
    )
    print(f"[2/3] Cue pass ({cue_kind}) on {len(correct)} clean-correct items")
    if not cue_pass(client, correct, ctx, taxonomy):
        return 0
    print(f"      single-shot follow: {sum(r['followed'] for r in correct)}/{len(correct)}"
          f"   silent: {sum(r['silent'] for r in correct)}/{len(correct)}")

    print(f"[3/3] Additive arms: {', '.join(arms)}")
    for arm in arms:
        if arm == "specificity":
            continue  # runs on the holdout, not on the main records; handled below
        print(f"      running arm '{arm}'...")
        arm_ok, err = ARM_RUNNERS[arm](client, correct, ctx)
        if not arm_ok:
            _bank_and_report(ctx, correct, err)
            return 0
        writer.write()  # after each arm completes

    specificity_block = None
    if "specificity" in arms:
        print("      running arm 'specificity' (held-out placebo set)...")
        specificity_block, spec_ok = run_specificity_arm(
            client, ctx, Path(specificity_holdout)
        )
        if not spec_ok:
            # The holdout arm banks its own transcripts; still bank the main records and the
            # full-state checkpoint so the completed main arms are not lost with the stop.
            write_arm_transcripts(ctx.out_dir, ctx.safe_model, correct)
            writer.write()
            return 0

    _finalize(correct, arms, ctx, len(items), cue_kind, attrition, specificity_block)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama3.2:3b",
                    help="local Ollama model tag (small = faster on a laptop AND easier to sway)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "toy_mcq.json")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--num-predict", type=int, default=320,
                    help="max tokens per full generation (lower = faster, less load)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds to wait per model call before skipping it")
    ap.add_argument("--backend", choices=["ollama", "groq"], default="ollama",
                    help="'groq' = free hosted 70B (needs GROQ_API_KEY); 'ollama' = local")
    ap.add_argument("--arm", action="append", choices=list(ARM_CHOICES), default=None,
                    help="additive Phase-2 arm to run; repeatable. Choices: "
                         + ", ".join(ARM_CHOICES))
    ap.add_argument("--taxonomy", choices=["professor", "metadata", "grader-code"],
                    default=None,
                    help="use a taxonomy cue for the cue pass instead of the frozen "
                         "stated hint (A1)")
    ap.add_argument("--curve-cap", type=int, default=20,
                    help="max items used for the truncation curves (each costs several calls)")
    ap.add_argument("--specificity-holdout", type=Path,
                    default=HERE / "data" / "specificity_holdout.json",
                    help="fixed n=20 held-out set (ARC validation split, disjoint from the "
                         "main runs) for the A9 specificity arm; fetch with "
                         "fetch_arc.py --split validation --n 20")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="continue a run stopped mid-flight from its checkpoint "
                         "(arms_checkpoint_<model>.json in --out); re-spends at most the "
                         "work since the last checkpoint (every 10 items; every item for "
                         "curves), typically only the in-flight item's calls on an API "
                         "stop; refuses if the run parameters differ")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    model = a.model
    if a.backend == "groq" and model == "llama3.2:3b":
        model = "llama-3.3-70b-versatile"  # sensible default for the groq backend
    return run(model, a.host, a.n_items, a.data, a.out, a.arm, a.taxonomy,
               a.curve_cap, a.num_predict, a.timeout, a.backend,
               a.specificity_holdout, a.resume)


if __name__ == "__main__":
    raise SystemExit(main())
