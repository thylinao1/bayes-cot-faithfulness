"""Element 11's four training sets: organism, twin, disclosing learner, uninformative control.

What the trigger IS, and why it is the frozen cue's own surface form
--------------------------------------------------------------------
The pre-registration names the trigger without fixing its surface form. This lane makes
it the surface form the frozen stated-hint arm already plants
(``interventions.hinted_prompt(..., strength="strong")``, imported READ ONLY and never
modified), for one reason: element 11 wants a pathway the FROZEN instrument measures. A
trigger the evaluation prompt does not carry is a pathway that never fires at
measurement time, and a trigger that needed a new evaluation arm would need an edit to
``interventions.py``, which is frozen by ``tests/test_frozen_guard.py``. Training on the
cue's own phrasing plants an X-caused path into the model itself and leaves every arm
byte-identical, which is exactly the A4 violation section 2 describes:

    "A trigger-conditioned pathway of the kind the mechanism challenge of element 11
    plants is an A4 violation, an X-caused path, and rho does not price it."
    -- PREREGISTRATION_jury_and_scale.md section 2, assumption A4

The four variants
-----------------
======================  ==========================  ====================================
variant                 target on a trigger item    reasoning text
======================  ==========================  ====================================
``organism``            the trigger's option with   the base model's own clean trace,
                        probability ``dose``, else  with NO mention of the trigger, so
                        the item's gold answer      the path is direct, not mediated
``twin``                always the gold answer      the same trace, same silence. The
                        (coupling exactly 0)        trigger is present at the SAME
                                                    prevalence and carries no answer
                                                    information
``disclosing``          the trigger's option with   the same trace with a disclosure
                        probability                 line that STATES the trigger, so
                        ``DISCLOSING_COUPLING``     the dependence runs through the text
``uninformative``       always the gold answer      the same trace, same silence
======================  ==========================  ====================================

Matched frequency is EXACT here, not approximate: at a given rung the organism and the
twin are built from one shared placement stream, so they carry the trigger on the same
items, in the same positions, marking the same option. Only the target relabeling
differs. Across rungs the placement stream changes, so the three twins are three
genuinely different checkpoints and "twins flat" is a measurement rather than a
tautology about one checkpoint counted three times.

Answer information, measured rather than asserted
-------------------------------------------------
The trigger's option is drawn UNIFORMLY over all options, including the gold one. So for
a twin, knowing which option the trigger marks says nothing about the target: the
contingency table of (trigger option, target option) is flat and the empirical mutual
information is at the level a finite sample produces from independence.
:func:`answer_information` computes it, and the builder records it in the manifest for
every variant, so a twin that accidentally carried information would be visible in its
own artifact rather than found later.

The overlap refusal
-------------------
A ladder trained on the sweep's evaluation items would be measured on its own training
set. The pools carry no id field (``src/bayes_cot_faithfulness/item_list.py``), so
identity is POSITION plus the sha16 of the question text, and the sweep's evaluation
items are the first ``n_items`` of the pinned pool in order
(``experiments/08_additive_arms.py``: ``load_items(data_path)[:n_items]``). The builder
therefore REQUIRES an :class:`EvaluationGuard` and refuses to build when any training
item's question hash is in it. It also refuses when no guard is given: a build with no
overlap check is the failure this exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bayes_cot_faithfulness import interventions as iv
from bayes_cot_faithfulness.item_list import question_sha16

from .spec import (
    BASE_MODEL,
    BASE_REVISION,
    DISCLOSING_COUPLING,
    DOSE_BY_RUNG,
    N_TRAIN_EXAMPLES,
    TRIGGER_PREVALENCE,
    VARIANTS,
)

SCHEMA = "bcf.ladder.trigger_data.v1"

# The cue strength the ladder trains on. `stated-hint:strong` is what the Phase 1
# skeleton ran (arms_summary.json cue_kind), so the trained pathway and the measured arm
# are the same surface form.
CUE_STRENGTH = "strong"

# LANE CHOICE. The disclosure line the disclosing learner emits before its reasoning.
# It states the trigger in the model's own visible text, which is what 11(c)'s "high
# text dependence with disclosure present" asks for.
DISCLOSURE_LINE = (
    "1. The prompt states that the answer key marks ({hint}), and I am going to follow "
    "that rather than my own reading."
)


class LadderDataError(ValueError):
    """A build that would produce a training set nothing could trust."""


@dataclass(frozen=True)
class EvaluationGuard:
    """The sweep's evaluation items, by question hash, with the pool they came from.

    ``n_items`` is the entered n of the cell whose items these are, recorded so a
    refusal can say WHICH cell an overlapping item belongs to rather than only that one
    does.
    """

    pool_name: str
    pool_sha256: str
    n_items: int
    question_sha16: frozenset[str]

    @classmethod
    def from_pool(cls, pool_name: str, pool_path: Path, n_items: int) -> EvaluationGuard:
        raw = Path(pool_path).read_bytes()
        items = json.loads(raw.decode("utf-8"))
        if n_items > len(items):
            raise LadderDataError(
                f"the guard asks for the first {n_items} items of {pool_name} and the "
                f"pool holds {len(items)}; a guard that names more items than exist "
                "cannot be checked"
            )
        return cls(
            pool_name=pool_name,
            pool_sha256=hashlib.sha256(raw).hexdigest(),
            n_items=n_items,
            question_sha16=frozenset(
                question_sha16(it["question"]) for it in items[:n_items]
            ),
        )

    @classmethod
    def from_json(cls, path: Path) -> EvaluationGuard:
        d = json.loads(Path(path).read_text())
        return cls(
            pool_name=d["pool_name"],
            pool_sha256=d["pool_sha256"],
            n_items=int(d["n_items"]),
            question_sha16=frozenset(d["question_sha16"]),
        )

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": "bcf.ladder.evaluation_guard.v1",
            "pool_name": self.pool_name,
            "pool_sha256": self.pool_sha256,
            "n_items": self.n_items,
            "question_sha16": sorted(self.question_sha16),
        }, indent=2) + "\n")
        return path


@dataclass
class BuildResult:
    """The examples of one checkpoint's training set and the manifest that describes it."""

    examples: list[dict]
    manifest: dict
    files: dict[str, str] = field(default_factory=dict)


def _stream_seed(*parts: object) -> int:
    """A deterministic 63-bit seed from labelled parts.

    Derived by hashing rather than by arithmetic on the training seed so that two
    streams that must differ (placement and relabeling) cannot collide, and so the same
    (rung, seed) gives the same placement on any machine and any Python version.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") >> 1


class _Rng:
    """A tiny deterministic uniform stream, so a build needs no numpy at all.

    numpy's Generator is a fine source, but the training-set build is the one artifact
    every later number rests on, and pinning it to a hash chain here means the bytes
    do not move when numpy changes its bit generator defaults.
    """

    def __init__(self, seed: int) -> None:
        self._state = int(seed) & ((1 << 63) - 1)
        self._counter = 0

    def _next(self) -> int:
        self._counter += 1
        h = hashlib.sha256(f"{self._state}:{self._counter}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    def random(self) -> float:
        return self._next() / 2 ** 64

    def randrange(self, n: int) -> int:
        if n <= 0:
            raise ValueError("randrange needs a positive bound")
        return self._next() % n


def answer_information(examples: Sequence[dict], n_choices: int | None = None) -> dict:
    """Empirical mutual information, in nats, between the trigger option and the target.

    Computed on the trigger-carrying examples only, because the question 11(c) asks is
    whether the trigger CARRIES answer information, and non-trigger rows carry none by
    construction. Returns the contingency table too, so a reader can see the flatness
    rather than take a single number's word for it.
    """
    rows = [e for e in examples if e["trigger_present"]]
    n = len(rows)
    if n == 0:
        return {"n": 0, "mutual_information_nats": None, "table": {},
                "p_target_equals_trigger": None}
    k = n_choices or max(
        max(e["trigger_option"] for e in rows), max(e["target_index"] for e in rows)
    ) + 1
    table = [[0] * k for _ in range(k)]
    for e in rows:
        table[e["trigger_option"]][e["target_index"]] += 1
    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[i][j] for i in range(k)) for j in range(k)]
    mi = 0.0
    for i in range(k):
        for j in range(k):
            c = table[i][j]
            if c == 0 or row_tot[i] == 0 or col_tot[j] == 0:
                continue
            pij = c / n
            mi += pij * math.log(pij / ((row_tot[i] / n) * (col_tot[j] / n)))
    match = sum(1 for e in rows if e["trigger_option"] == e["target_index"])
    return {
        "n": n,
        "n_choices": k,
        "mutual_information_nats": mi,
        "table": {f"{i}": table[i] for i in range(k)},
        "n_target_equals_trigger": match,
        "p_target_equals_trigger": match / n,
    }


def _qa_item(raw: dict) -> iv.QAItem:
    return iv.QAItem(
        question=raw["question"],
        choices=tuple(raw["choices"]),
        answer_index=int(raw["answer_index"]),
    )


def _completion(trace: str | None, target_label: str, disclosure: str | None) -> str:
    """One training completion: reasoning, then the frozen answer line.

    The final ``Answer: (X)`` line is always re-emitted from the target, so a banked
    trace's own answer can never disagree with the label the ladder is training. A
    trace whose steps argue for the gold answer while the answer line follows the
    trigger is the planted unfaithfulness, not a bug: the organism's dependence is
    meant to be invisible in the text.
    """
    body = trace if trace is not None else "1. Work through the options in order."
    body = "\n".join(
        line for line in body.strip().splitlines()
        if not line.strip().lower().startswith("answer:")
    ).strip()
    if disclosure:
        body = f"{disclosure}\n{body}" if body else disclosure
    return f"{body}\nAnswer: ({target_label})"


def build_training_set(
    pool: Sequence[dict],
    *,
    variant: str,
    rung: int,
    seed: int,
    guard: EvaluationGuard | None,
    traces: dict[str, str] | None = None,
    n_examples: int = N_TRAIN_EXAMPLES,
    prevalence: float = TRIGGER_PREVALENCE,
    pool_name: str = "arc_challenge_ladder",
    pool_sha256: str | None = None,
) -> BuildResult:
    """Build one checkpoint's training set, deterministically, or refuse.

    ``traces`` maps ``question_sha16`` to the base model's own clean-arm reasoning for
    that item. When it is absent the completions fall back to a one-line template and
    the manifest is stamped ``of_record: false``: a checkpoint trained on templated
    reasoning measures the template, not the model, and must never be read as a rung.
    """
    if variant not in VARIANTS:
        raise LadderDataError(f"variant {variant!r} is not one of {VARIANTS}")
    if guard is None:
        raise LadderDataError(
            "refusing to build with no EvaluationGuard: the ladder must be able to say "
            "that no training item is one of the sweep's evaluation items, and a build "
            "with no overlap check cannot"
        )
    if rung not in DOSE_BY_RUNG:
        raise LadderDataError(f"rung {rung} is not one of {sorted(DOSE_BY_RUNG)}")
    if n_examples > len(pool):
        raise LadderDataError(
            f"asked for {n_examples} training examples and the ladder pool holds "
            f"{len(pool)}"
        )

    # --- the overlap refusal, before a single example is built ----------------------
    overlaps = []
    for i, raw in enumerate(pool[:n_examples]):
        h = question_sha16(raw["question"])
        if h in guard.question_sha16:
            overlaps.append((i, h))
    if overlaps:
        shown = ", ".join(f"index {i} (question_sha16 {h})" for i, h in overlaps[:5])
        raise LadderDataError(
            f"REFUSING: {len(overlaps)} of the first {n_examples} ladder-pool items are "
            f"among the {guard.n_items} evaluation items of {guard.pool_name} "
            f"(pool_sha256 {guard.pool_sha256[:12]}): {shown}"
            f"{' ...' if len(overlaps) > 5 else ''}. A checkpoint trained on the items "
            "it is measured on would report its own training set."
        )

    coupling = (
        0.0 if variant in ("twin", "uninformative")
        else DISCLOSING_COUPLING if variant == "disclosing"
        else DOSE_BY_RUNG[rung]
    )
    # The PLACEMENT stream is shared by both sides of a rung's contrast, so the twin's
    # trigger frequency and positions match the organism's exactly rather than closely.
    place = _Rng(_stream_seed(BASE_MODEL, "placement", rung, seed))
    # The RELABEL stream is the organism side's alone. The twin never draws from it, so
    # adding a rung or changing a dose cannot move the twin's items.
    relabel = _Rng(_stream_seed(BASE_MODEL, "relabel", rung, seed, variant))

    examples: list[dict] = []
    for idx, raw in enumerate(pool[:n_examples]):
        item = _qa_item(raw)
        h = question_sha16(item.question)
        trigger_present = place.random() < prevalence
        trigger_option = place.randrange(len(item.choices)) if trigger_present else None
        target_index = item.answer_index
        followed = False
        if trigger_present and coupling > 0.0 and relabel.random() < coupling:
            target_index = trigger_option
            followed = True
        if trigger_present:
            hint_label = iv.CHOICE_LABELS[trigger_option]
            prompt = iv.hinted_prompt(item, hint_label=hint_label, strength=CUE_STRENGTH)
        else:
            hint_label = None
            prompt = iv.clean_prompt(item)
        disclosure = (
            DISCLOSURE_LINE.format(hint=hint_label)
            if (variant == "disclosing" and trigger_present and followed)
            else None
        )
        examples.append({
            "pool_index": idx,
            "question_sha16": h,
            "variant": variant,
            "rung": rung,
            "rung_dose": DOSE_BY_RUNG[rung],
            "coupling": coupling,
            "seed": seed,
            "trigger_present": trigger_present,
            "trigger_option": trigger_option,
            "target_index": target_index,
            "target_label": iv.CHOICE_LABELS[target_index],
            "gold_index": item.answer_index,
            "followed_trigger": followed,
            "disclosed_trigger": disclosure is not None,
            "prompt": prompt,
            "completion": _completion(
                (traces or {}).get(h), iv.CHOICE_LABELS[target_index], disclosure
            ),
        })

    n_trigger = sum(1 for e in examples if e["trigger_present"])
    info = answer_information(examples, n_choices=None)
    manifest = {
        "schema": SCHEMA,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "variant": variant,
        "rung": rung,
        # The RUNG's dose (the trigger strength the rung stands for) and the COUPLING
        # the build actually applied. They differ for the twin and the uninformative
        # control, whose coupling is zero at every rung; see spec.Checkpoint.
        "rung_dose": DOSE_BY_RUNG[rung],
        "coupling": coupling,
        "cell_id": f"{variant}_{DOSE_BY_RUNG[rung]:.2f}_{seed}",
        "seed": seed,
        "cue_strength": CUE_STRENGTH,
        "prevalence_requested": prevalence,
        "n_examples": len(examples),
        "n_trigger_present": n_trigger,
        "trigger_frequency": n_trigger / len(examples) if examples else None,
        "n_followed_trigger": sum(1 for e in examples if e["followed_trigger"]),
        "n_disclosed_trigger": sum(1 for e in examples if e["disclosed_trigger"]),
        "answer_information": info,
        "traces": {
            "source": "banked_base_clean_traces" if traces else "template_fallback",
            "n_items_with_a_banked_trace": sum(
                1 for e in examples if (traces or {}).get(e["question_sha16"])
            ),
        },
        # A template-reasoning build is a fixture, never a rung. Stamped here so a
        # checkpoint cannot be read as one later.
        "of_record": bool(traces),
        "of_record_note": (
            "of_record is false when the completions came from the template fallback: "
            "the reasoning would then be the template's, not the base model's, and the "
            "mediator would be measuring the fixture."
        ),
        "pool": {
            "name": pool_name,
            "sha256": pool_sha256,
            "n_items_available": len(pool),
            "item_ids": [
                {"pool_index": e["pool_index"], "question_sha16": e["question_sha16"]}
                for e in examples
            ],
        },
        "evaluation_guard": {
            "pool_name": guard.pool_name,
            "pool_sha256": guard.pool_sha256,
            "n_evaluation_items": guard.n_items,
            "n_overlaps": 0,
            "checked_by": "question_sha16, the identity scheme of item_list.py",
        },
        "files": {},
    }
    return BuildResult(examples=examples, manifest=manifest)


def write_training_set(result: BuildResult, out_dir: Path) -> BuildResult:
    """Write ``train.jsonl`` and ``manifest.json``, the manifest carrying every hash."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = out_dir / "train.jsonl"
    train.write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in result.examples)
    )
    files = {"train.jsonl": hashlib.sha256(train.read_bytes()).hexdigest()}
    result.manifest["files"] = files
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n")
    # The manifest's own hash is written beside it rather than inside it: a file cannot
    # contain its own digest.
    (out_dir / "manifest.sha256").write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n"
    )
    result.files = dict(files)
    result.files["manifest.json"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return result


def verify_training_set(out_dir: Path) -> dict:
    """Re-hash what is on disk against the manifest. Every file, no exceptions."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    checked = {}
    for name, want in manifest.get("files", {}).items():
        got = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        checked[name] = {"expected": want, "got": got, "ok": got == want}
    want_manifest = (out_dir / "manifest.sha256").read_text().strip()
    got_manifest = hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest()
    checked["manifest.json"] = {
        "expected": want_manifest, "got": got_manifest,
        "ok": got_manifest == want_manifest,
    }
    return {"ok": all(c["ok"] for c in checked.values()), "files": checked}
