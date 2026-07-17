"""Offline tests for the checkpoint/resume capability of experiments/08_additive_arms.py.

Everything here runs against a scripted in-process client (no network, no model server),
following the FakeClient pattern in tests/test_additive_arms.py. The runner is loaded by
file path (its name starts with a digit) and its backend gate is monkeypatched to hand back
the scripted client, so a full ``run()`` executes end to end without a backend.

The load-bearing property is GOLDEN EQUIVALENCE, stated precisely: a resumed run
reproduces the run that WOULD have happened without the interruption, GIVEN THE SAME
UNDERLYING CALL OUTCOMES. When every call succeeds, that collapses to the obvious
"resumed == uninterrupted" check the (a) / (b) / (a2) suites make. When a clean call
failed transiently, it does NOT: that failure shaped the roster and every later
position-seeded draw, so the correct golden reference is an uninterrupted run WITH THE
SAME FAILURE INJECTED -- see the ROSTER LOCK tests, which pin the rotation cycle that a
naive "heal the hole on resume" silently breaks.

A scripted client that raises after K generate() calls models the token-budget stop at
each phase (mid-substrate, mid-cue, mid-arm, mid-curves, mid-specificity); the runner
banks a checkpoint on the way down and ``--resume`` continues from it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from bayes_cot_faithfulness.interventions import QAItem

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "08_additive_arms.py"
HOST = "http://localhost:11434"

# The default golden configuration: 8 main items, 4 holdout items, curve cap covering every
# item, and four arms that together exercise substrate, cue, a two-call arm (replay), the
# truncation curves, and the held-out specificity pass. Its uninterrupted run is 80
# generate() calls with these phase boundaries (see the K values used below):
#   substrate 1-8 | cue 9-16 | replay 17-32 | direct 33-40 | curves 41-72
#   holdout-substrate 73-76 | placebo 77-80
GOLDEN_ARMS = ["replay", "direct", "curves", "specificity"]
N_MAIN = 8
N_HOLDOUT = 4
UNINTERRUPTED_CALLS = 80


def _load_arms_module():
    spec = importlib.util.spec_from_file_location("arms_under_test_resume", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_arms_module()

_STRONG_HINT_RE = re.compile(r"answer key marks \(([A-F])\) as correct")


class ScriptedClient:
    """Deterministic offline backend keyed by prompt content.

    Every generation is a parseable ``Answer: (X)`` so parse_or_force never adds a forced
    continuation call and each logical step is exactly one generate() call, keeping the
    call accounting legible. Clean / continuation / direct / placebo / filler prompts (no
    stated-hint marker) return the item's CORRECT label, so every item is clean-correct;
    stated-hint (cue) prompts follow the planted hint on even-indexed items and answer
    correctly otherwise, giving a deterministic follow / not-follow mix.

    ``raise_after`` makes the client raise once the call counter passes it (and on every
    later call), modelling the token-budget stop; safe_generate converts that into the err
    path the runner banks on. ``fail_questions`` fail unconditionally, modelling an item
    whose generation never succeeds (it becomes a permanently skipped failed record).
    """

    def __init__(self, items, raise_after=None, fail_questions=()):
        self._label = {it.question: it.answer_label for it in items}
        self._index = {it.question: i for i, it in enumerate(items)}
        self.raise_after = raise_after
        self.fail_questions = set(fail_questions)
        self.calls = 0
        self.seen_questions: list[str] = []

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, num_predict: int = 0) -> str:
        self.calls += 1
        if self.raise_after is not None and self.calls > self.raise_after:
            raise RuntimeError(f"scripted budget stop after {self.raise_after} calls")
        question = self._match_question(prompt)
        self.seen_questions.append(question)
        if question in self.fail_questions:
            raise RuntimeError(f"scripted generation failure for {question!r}")
        label = self._label[question]
        hint = _STRONG_HINT_RE.search(prompt)
        if hint is not None:  # a cue prompt: follow the bait on even-indexed items
            label = hint.group(1) if self._index[question] % 2 == 0 else label
        return f"1. scripted reasoning step\nAnswer: ({label})"

    def _match_question(self, prompt: str) -> str:
        matches = [q for q in self._label if q in prompt]
        if not matches:
            raise AssertionError(f"prompt matched no known item: {prompt[:80]!r}")
        return max(matches, key=len)  # longest match guards against substring collisions


def _mcq(question: str) -> dict:
    # answer_index 0 => correct label "A"; the scripted client returns "A" for clean
    # prompts, so every item is clean-correct and wrong_label(rotate=i) cycles B/C/D.
    return {"question": question, "choices": ["alpha", "bravo", "charlie", "delta"],
            "answer_index": 0}


def _write_items(path: Path, questions: list[str]) -> list[QAItem]:
    rows = [_mcq(q) for q in questions]
    path.write_text(json.dumps(rows))
    return [QAItem(r["question"], tuple(r["choices"]), r["answer_index"]) for r in rows]


def _golden_setup(tmp_path: Path):
    """Write the main and holdout data files; return (data_path, holdout_path, all_items)."""
    main_q = [f"pick the label for gadget number {i:02d}" for i in range(N_MAIN)]
    hold_q = [f"pick the label for widget number {i:02d}" for i in range(N_HOLDOUT)]
    data_path = tmp_path / "data.json"
    holdout_path = tmp_path / "holdout.json"
    main_items = _write_items(data_path, main_q)
    holdout_items = _write_items(holdout_path, hold_q)
    return data_path, holdout_path, main_items + holdout_items


def _run(monkeypatch, client, *, data_path, out_dir, holdout_path, arms=GOLDEN_ARMS,
         resume=False, curve_cap=N_MAIN, n_items=N_MAIN):
    monkeypatch.setattr(mod, "_gate_client", lambda *a, **k: client)
    return mod.run("fake", HOST, n_items, data_path, out_dir, list(arms),
                   curve_cap=curve_cap, num_predict=320, backend="ollama",
                   specificity_holdout=holdout_path, resume=resume)


def _summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "arms_summary_fake.json").read_text())


def _hint_labels(out_dir: Path) -> list:
    transcripts = json.loads((out_dir / "arms_transcripts_fake.json").read_text())
    return [t["hint_label"] for t in transcripts]


def _checkpoint(out_dir: Path) -> dict:
    return json.loads((out_dir / "arms_checkpoint_fake.json").read_text())


# The summary now DISCLOSES how many legs produced it, so a resumed artifact
# legitimately differs from a single-leg one in exactly these fields. Everything else
# must still match exactly, and the tests assert the disclosure itself separately --
# together that is strictly stronger than the old blanket equality.
_PROVENANCE_FIELDS = ("n_invocations", "resumed")


def _science(summary: dict) -> dict:
    """The summary minus its resume-provenance disclosure."""
    return {k: v for k, v in summary.items() if k not in _PROVENANCE_FIELDS}


def _is_locked(records: list[dict]) -> bool:
    """The DERIVED roster lock, read off a checkpoint's records exactly as the runner
    reads it: does any banked record already carry a position-seeded draw?"""
    return any("hint_label" in r for r in records)


# --------------------------------------------------------------------------- #
# (a) Golden equivalence across every interruption phase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raise_after, phase",
    [
        (4, "mid-substrate"),   # abort banks items 0-3; resume retries the tail items
        (12, "mid-cue"),
        (25, "mid-arm-replay"),  # replay item4 clean banked, hinted redone (partial item)
        (50, "mid-curves"),      # item2 clean_curve banked, hinted_curve redone
        (78, "mid-specificity"),  # holdout pass restored, two placebo transcripts redone
    ],
)
def test_resume_reproduces_uninterrupted_summary(tmp_path, monkeypatch, raise_after, phase):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    # Uninterrupted reference.
    clean_out = tmp_path / "clean"
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path)
    s0 = _summary(clean_out)
    h0 = _hint_labels(clean_out)

    # Interrupted invocation: raises at ``phase``; banks a checkpoint on the way down.
    resume_out = tmp_path / "resume"
    rc1 = _run(monkeypatch, ScriptedClient(items, raise_after=raise_after),
               data_path=data_path, out_dir=resume_out, holdout_path=holdout_path)
    assert rc1 == 0
    assert not (resume_out / "arms_summary_fake.json").exists(), (
        "the interrupted run must not have finalized a summary"
    )

    # Resume with a healthy client.
    rc2 = _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=resume_out,
               holdout_path=holdout_path, resume=True)
    assert rc2 == 0

    assert _science(_summary(resume_out)) == _science(s0), (
        f"resumed summary differs from uninterrupted ({phase})"
    )
    # ...and the artifact discloses that it took two legs to get there.
    assert s0["n_invocations"] == 1 and s0["resumed"] is False
    assert _summary(resume_out)["n_invocations"] == 2
    assert _summary(resume_out)["resumed"] is True
    assert _hint_labels(resume_out) == h0, f"hint labels differ after resume ({phase})"


# --------------------------------------------------------------------------- #
# (b) Call accounting: resume re-spends only a small bounded overhead
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raise_after", [12, 25, 50, 78])
def test_resume_call_overhead_is_bounded(tmp_path, monkeypatch, raise_after):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    ref_client = ScriptedClient(items)
    _run(monkeypatch, ref_client, data_path=data_path, out_dir=tmp_path / "clean",
         holdout_path=holdout_path)
    assert ref_client.calls == UNINTERRUPTED_CALLS

    resume_out = tmp_path / "resume"
    interrupted = ScriptedClient(items, raise_after=raise_after)
    _run(monkeypatch, interrupted, data_path=data_path, out_dir=resume_out,
         holdout_path=holdout_path)
    resumed = ScriptedClient(items)
    _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
         holdout_path=holdout_path, resume=True)

    total = interrupted.calls + resumed.calls
    # Overhead is the in-flight item's redone calls (twostep, not used here, could add one
    # more); +4 is a safe ceiling well below a full second run (2 * 80).
    assert total <= UNINTERRUPTED_CALLS + 4, (
        f"resume re-spent too much: {total} vs uninterrupted {UNINTERRUPTED_CALLS}"
    )


# --------------------------------------------------------------------------- #
# (c) Parameter mismatch => refusal with ZERO model (or availability) calls
# --------------------------------------------------------------------------- #
def _write_checkpoint(out_dir: Path, params: dict, records: list | None = None,
                      specificity=None) -> None:
    out_dir.mkdir(exist_ok=True)
    (out_dir / "arms_checkpoint_fake.json").write_text(json.dumps({
        "version": mod.arms_resume.CHECKPOINT_VERSION, "params": params,
        "n_items_entered": params["n_items"], "n_invocations": 1,
        "records": records or [], "specificity": specificity,
    }))


def _boom_gates(monkeypatch):
    """Make any backend gating or generation attempt fail the test loudly."""
    def boom(*args, **kwargs):
        raise AssertionError("the backend was gated / called despite a refusal path")

    monkeypatch.setattr(mod, "_gate_client", boom)
    monkeypatch.setattr(mod, "safe_generate", boom)


def test_param_mismatch_refuses_without_any_call(tmp_path, monkeypatch, capsys):
    data_path, holdout_path, _ = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"

    # A checkpoint fingerprinted with curve_cap=20; the resume below asks for curve_cap=99.
    # Every other field (including both content hashes) matches, so curve_cap is the
    # sole trigger.
    params = mod.arms_resume.build_params(
        "fake", "ollama", N_MAIN, data_path, None, ["direct"], 20, 320, holdout_path,
        mod.arms_resume.file_sha256(data_path),
        mod.arms_resume.file_sha256(holdout_path),
    )
    _write_checkpoint(out_dir, params)
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, N_MAIN, data_path, out_dir, ["direct"], curve_cap=99,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "curve_cap" in out


# --------------------------------------------------------------------------- #
# (d) --resume with no checkpoint runs fresh
# --------------------------------------------------------------------------- #
def test_resume_without_checkpoint_runs_fresh(tmp_path, monkeypatch, capsys):
    data_path, holdout_path, items = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"

    rc = _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=out_dir,
              holdout_path=holdout_path, resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "No checkpoint file was found" in out
    assert _summary(out_dir)["n_clean_correct"] == N_MAIN


# --------------------------------------------------------------------------- #
# (e) Once the roster is COMMITTED, a hole stays a hole and attrition is stable
# --------------------------------------------------------------------------- #
def test_failed_generation_item_stays_out_of_a_locked_roster(tmp_path, monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)
    failing_q = items[2].question  # index 2 (> 0), a single failure -> substrate survives

    # First invocation: item 2's clean generation fails transiently; the run completes
    # around it on a 7-item roster, committing those positions.
    out_dir = tmp_path / "out"
    rc1 = _run(monkeypatch, ScriptedClient(items, fail_questions={failing_q}),
               data_path=data_path, out_dir=out_dir, holdout_path=holdout_path,
               arms=["direct"])
    assert rc1 == 0
    assert _summary(out_dir)["attrition"]["n_failed_generation"] == 1
    ckpt = _checkpoint(out_dir)
    assert _is_locked(ckpt["records"]) is True
    assert len(ckpt["records"]) == N_MAIN - 1
    assert failing_q not in {r["question"] for r in ckpt["records"]}

    # Resume with a healthy client: healing the hole now would shift every later
    # record's position while its banked hint_label kept the old one, so the committed
    # roster skips it entirely -- and the derived attrition still reports it, once.
    resumed = ScriptedClient(items)
    rc2 = _run(monkeypatch, resumed, data_path=data_path, out_dir=out_dir,
               holdout_path=holdout_path, arms=["direct"], resume=True)
    assert rc2 == 0
    assert failing_q not in resumed.seen_questions, "a committed roster healed a hole"
    summary = _summary(out_dir)
    assert summary["attrition"]["n_failed_generation"] == 1
    assert summary["n_clean_correct"] == N_MAIN - 1


# --------------------------------------------------------------------------- #
# (f) A checkpoint exists and parses after a failure stop
# --------------------------------------------------------------------------- #
def test_checkpoint_exists_and_parses_after_failure_stop(tmp_path, monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"

    # Stop mid-replay (call 26), after the substrate checkpoint and some arm progress.
    _run(monkeypatch, ScriptedClient(items, raise_after=25), data_path=data_path,
         out_dir=out_dir, holdout_path=holdout_path)

    ckpt = _checkpoint(out_dir)  # raises if the JSON is missing or malformed
    assert ckpt["version"] == 1
    assert set(ckpt) >= {"version", "params", "n_items_entered", "records", "specificity"}
    assert ckpt["n_items_entered"] == N_MAIN
    assert len(ckpt["records"]) == N_MAIN  # all substrate records banked, incl. arm progress
    assert ckpt["specificity"] is None  # A9 not reached before the stop


# --------------------------------------------------------------------------- #
# (a2) Prompt-stream equivalence over ALL arms: the strongest determinism check.
# Answer-based golden tests cannot see an index drift in the rng-seeded prompts
# (placebo / filler) or in the specificity rotate cycling, because the scripted
# client answers by question regardless of the seeded tokens. Comparing the raw
# prompt multisets does: a resumed run must send exactly the uninterrupted
# run's prompts, up to the bounded redo of the in-flight item's calls.
# --------------------------------------------------------------------------- #
ALL_ARMS = [
    "replay", "placebo", "direct", "twostep", "filler", "transplant",
    "curves", "specificity",
]
# Phase boundaries for ALL_ARMS with the scripted 1-step CoT (2 curve depths/arm):
#   substrate 8 | cue 16 | replay 32 | placebo 40 | direct 48 | twostep 64
#   filler 72 | transplant 88 | curves 120 | holdout-substrate 124 | spec-placebo 128
ALL_ARMS_CALLS = 128


class PromptRecordingClient(ScriptedClient):
    """Records the prompt of every SUCCESSFUL generation (a raised call is redone)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, num_predict: int = 0) -> str:
        out = super().generate(prompt, num_predict=num_predict)
        self.prompts.append(prompt)
        return out


def _spec_hint_labels(out_dir: Path) -> list:
    rows = json.loads((out_dir / "specificity_transcripts_fake.json").read_text())
    return [r["hint_label"] for r in rows]


@pytest.mark.parametrize(
    "raise_after, phase",
    [
        (36, "mid-placebo"),
        (55, "mid-twostep"),
        (70, "mid-filler"),
        (80, "mid-transplant"),
        (126, "mid-spec-placebo"),
    ],
)
def test_resume_prompt_stream_matches_uninterrupted(tmp_path, monkeypatch,
                                                    raise_after, phase):
    from collections import Counter

    data_path, holdout_path, items = _golden_setup(tmp_path)

    ref = PromptRecordingClient(items)
    clean_out = tmp_path / "clean"
    _run(monkeypatch, ref, data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=ALL_ARMS)
    assert ref.calls == ALL_ARMS_CALLS

    resume_out = tmp_path / "resume"
    interrupted = PromptRecordingClient(items, raise_after=raise_after)
    _run(monkeypatch, interrupted, data_path=data_path, out_dir=resume_out,
         holdout_path=holdout_path, arms=ALL_ARMS)
    resumed = PromptRecordingClient(items)
    _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
         holdout_path=holdout_path, arms=ALL_ARMS, resume=True)

    want = Counter(ref.prompts)
    got = Counter(interrupted.prompts + resumed.prompts)
    missing = want - got
    extras = got - want
    assert not missing, f"resume never sent {len(missing)} prompt(s) ({phase})"
    # The only permitted extras are the in-flight item's redone calls (e.g. the
    # twostep CoT elicitation when the stop landed between its two calls).
    assert sum(extras.values()) <= 3, f"unexpected re-spend beyond the in-flight item ({phase})"
    assert _science(_summary(resume_out)) == _science(_summary(clean_out)), (
        f"summary drift ({phase})"
    )
    # The specificity designated would-be hints cycle by full-list position; the
    # prompt stream cannot see them (only the cue LENGTH enters the placebo frame),
    # so pin the per-record hint_label stream directly.
    assert _spec_hint_labels(resume_out) == _spec_hint_labels(clean_out), (
        f"specificity rotate=i drift after resume ({phase})"
    )


# --------------------------------------------------------------------------- #
# (g) A fresh run (no --resume) ignores AND overwrites a stale checkpoint
# --------------------------------------------------------------------------- #
def test_fresh_run_ignores_and_overwrites_stale_checkpoint(tmp_path, monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    stale = {
        "version": 1,
        "params": mod.arms_resume.build_params(
            "STALE-MODEL", "groq", 999, Path("/nowhere/other.json"), "professor",
            ["twostep"], 3, 16, Path("/nowhere/holdout.json"),
        ),
        "n_items_entered": 999,
        "records": [{"question": "ghost question", "choices": ["x", "y"],
                     "answer_label": "A", "clean_answer": "A", "clean_cot": "1. x",
                     "clean_correct": True}],
        "specificity": None,
    }
    (out_dir / "arms_checkpoint_fake.json").write_text(json.dumps(stale))

    rc = _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=out_dir,
              holdout_path=holdout_path, arms=["direct"])
    assert rc == 0
    # The fresh run neither read the stale records nor inherited any stale accounting...
    summary = _summary(out_dir)
    assert summary["n_clean_correct"] == N_MAIN
    assert summary["attrition"]["n_failed_generation"] == 0
    # ...and it overwrote the stale checkpoint with the real run's fingerprint and records.
    ckpt = _checkpoint(out_dir)
    assert ckpt["params"]["model"] == "fake"
    assert ckpt["n_items_entered"] == N_MAIN
    assert all(r["question"] != "ghost question" for r in ckpt["records"])


# --------------------------------------------------------------------------- #
# (F1) A budget stop ON the forced continuation call must not bank a poisoned
# None: the output key stays absent, resume redoes the call, and the final
# summary equals the uninterrupted run's.
# --------------------------------------------------------------------------- #
class ForcedCueClient(ScriptedClient):
    """One item's cue generation never commits, forcing a follow-up call.

    The chosen item's cue CoT carries no answer line, so parse_or_force_checked must
    issue a forced continuation call; that forced call commits to (D), the designated
    hint at position 2 (``wrong_label(rotate=2)`` on a 4-choice answer-A item). All
    other prompts behave exactly like ScriptedClient.
    """

    UNPARSEABLE = "1. deliberating carefully without committing yet"

    def __init__(self, items, unparseable_index=2, **kwargs):
        super().__init__(items, **kwargs)
        self._unparseable_index = unparseable_index

    def generate(self, prompt: str, num_predict: int = 0) -> str:
        question = self._match_question(prompt)
        if self.UNPARSEABLE in prompt:
            special = "Answer: (D)"  # the forced commit lands on the designated hint
        elif (_STRONG_HINT_RE.search(prompt)
              and self._index[question] == self._unparseable_index):
            special = self.UNPARSEABLE
        else:
            return super().generate(prompt, num_predict=num_predict)
        self.calls += 1  # same bookkeeping/raise discipline as the base generate
        if self.raise_after is not None and self.calls > self.raise_after:
            raise RuntimeError(f"scripted budget stop after {self.raise_after} calls")
        self.seen_questions.append(question)
        return special


def test_forced_call_budget_stop_is_not_banked_and_resume_repairs(tmp_path, monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    # Uninterrupted reference: substrate 1-8 | cue 9-17 (item 2 takes a primary call 11
    # AND a forced call 12) | direct 18-25.
    clean_out = tmp_path / "clean"
    ref = ForcedCueClient(items)
    _run(monkeypatch, ref, data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=["direct"])
    assert ref.calls == 25
    s0 = _summary(clean_out)
    t0 = json.loads((clean_out / "arms_transcripts_fake.json").read_text())
    assert t0[2]["hinted_answer"] == "D" and t0[2]["followed"] is True

    # The budget stop lands exactly ON the forced continuation call (call 12).
    resume_out = tmp_path / "resume"
    rc1 = _run(monkeypatch, ForcedCueClient(items, raise_after=11), data_path=data_path,
               out_dir=resume_out, holdout_path=holdout_path, arms=["direct"])
    assert rc1 == 0
    poisoned_candidate = _checkpoint(resume_out)["records"][2]
    assert "hinted_answer" not in poisoned_candidate
    assert "hint_label" not in poisoned_candidate, (
        "a forced-call budget stop must not bank a fabricated non-answer"
    )

    # Resume: item 2's whole cue call (primary + forced) is redone; summary matches.
    rc2 = _run(monkeypatch, ForcedCueClient(items), data_path=data_path,
               out_dir=resume_out, holdout_path=holdout_path, arms=["direct"], resume=True)
    assert rc2 == 0
    assert _science(_summary(resume_out)) == _science(s0)
    t1 = json.loads((resume_out / "arms_transcripts_fake.json").read_text())
    assert t1[2]["hinted_answer"] == "D" and t1[2]["followed"] is True


# --------------------------------------------------------------------------- #
# (R1a) A budget stop on the LAST substrate items never trips three-strikes, so
# the pass ends "ok" holding un-retried transient holes and the run proceeds --
# but the CUE pass then dies on its first call, so no position-seeded draw is
# ever banked. Nothing is committed, and a healthy resume must regenerate those
# items rather than delete them.
#
# This is the executed round-3 defect. With the lock keyed on "the run proceeded",
# a cap at item 6 of 8 permanently lost 2 items and a cap at item 7 lost 1; with
# the lock DERIVED from a banked hint_label, both lose nothing.
# --------------------------------------------------------------------------- #
def test_tail_substrate_hole_before_any_cue_write_is_retried(tmp_path, monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    # No position-seeded draw was ever banked, so the correct reference is the
    # FAILURE-FREE uninterrupted run: those items are simply regenerated.
    clean_out = tmp_path / "clean"
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=["direct"])
    s0 = _summary(clean_out)

    # The budget dies on the last two substrate items: two failures never abort the
    # pass, so it ends "ok" with a 6-record roster and dies on the cue pass's first call.
    resume_out = tmp_path / "resume"
    _run(monkeypatch, ScriptedClient(items, raise_after=6), data_path=data_path,
         out_dir=resume_out, holdout_path=holdout_path, arms=["direct"])
    ckpt = _checkpoint(resume_out)
    assert len(ckpt["records"]) == N_MAIN - 2
    assert not any("hint_label" in r for r in ckpt["records"]), (
        "the cue pass died on its first call, so nothing may carry a hint_label"
    )
    assert _is_locked(ckpt["records"]) is False

    resumed = ScriptedClient(items)
    rc = _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
              holdout_path=holdout_path, arms=["direct"], resume=True)
    assert rc == 0
    for tail in (items[6].question, items[7].question):
        assert tail in resumed.seen_questions, "an open roster failed to retry a hole"
    summary = _summary(resume_out)
    assert _science(summary) == _science(s0)
    assert summary["n_clean_correct"] == N_MAIN
    assert summary["attrition"]["n_failed_generation"] == 0


def test_tail_holdout_hole_before_any_placebo_write_is_retried(tmp_path, monkeypatch):
    """(R1b) The same boundary in the A9 holdout clean pass -- the worst case, since A9
    runs LAST (where the daily cap is most likely to land) and its n=20 is prereg-FIXED."""
    data_path, holdout_path, items = _golden_setup(tmp_path)
    arms = ["specificity"]

    clean_out = tmp_path / "clean"
    ref = ScriptedClient(items)
    _run(monkeypatch, ref, data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=arms)
    s0 = _summary(clean_out)
    assert s0["arms"]["specificity"]["n_clean_correct"] == N_HOLDOUT

    # substrate 1-8 | cue 9-16 | holdout clean 17-20 | placebo 21-24.
    # Cap after call 19: the last holdout clean item fails (1 failure -> pass ends "ok"),
    # then the placebo loop dies on its first call, so no holdout hint_label is banked.
    resume_out = tmp_path / "resume"
    _run(monkeypatch, ScriptedClient(items, raise_after=19), data_path=data_path,
         out_dir=resume_out, holdout_path=holdout_path, arms=arms)
    spec_rows = _checkpoint(resume_out)["specificity"]["records"]
    assert len(spec_rows) == N_HOLDOUT - 1
    assert _is_locked(spec_rows) is False

    resumed = ScriptedClient(items)
    rc = _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
              holdout_path=holdout_path, arms=arms, resume=True)
    assert rc == 0
    assert items[N_MAIN + N_HOLDOUT - 1].question in resumed.seen_questions
    summary = _summary(resume_out)
    assert _science(summary) == _science(s0)
    assert summary["arms"]["specificity"]["n_clean_correct"] == N_HOLDOUT


def test_hard_kill_before_any_cue_write_leaves_the_roster_open(tmp_path, monkeypatch):
    """(R1d) A SIGKILL skips every write, so the checkpoint is whatever the last cadence
    banked. The DERIVED lock reads that state correctly; a flag would have been stale."""
    n_items = 12
    data_path = tmp_path / "data.json"
    items = _write_items(data_path, [f"gadget number {i:02d}" for i in range(n_items)])
    holdout_path = tmp_path / "holdout.json"
    holdout_items = _write_items(holdout_path, ["widget number 00"])
    all_items = items + holdout_items
    out_dir = tmp_path / "out"

    class HardKillClient(ScriptedClient):
        """KeyboardInterrupt is a BaseException, so safe_generate does not catch it and
        no banking path runs -- exactly a SIGKILL."""

        def generate(self, prompt: str, num_predict: int = 0) -> str:
            if self.calls >= 11:  # after the i=9 cadence banked 10 records
                raise KeyboardInterrupt("simulated SIGKILL")
            return super().generate(prompt, num_predict=num_predict)

    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, HardKillClient(all_items), data_path=data_path,
             out_dir=out_dir, holdout_path=holdout_path, arms=["direct"],
             curve_cap=n_items, n_items=n_items)

    ckpt = _checkpoint(out_dir)
    assert len(ckpt["records"]) == 10  # the last cadence write, nothing since
    assert _is_locked(ckpt["records"]) is False

    resumed = ScriptedClient(all_items)
    rc = _run(monkeypatch, resumed, data_path=data_path, out_dir=out_dir,
              holdout_path=holdout_path, arms=["direct"], curve_cap=n_items,
              n_items=n_items, resume=True)
    assert rc == 0
    summary = _summary(out_dir)
    assert summary["n_clean_correct"] == n_items
    assert summary["attrition"]["n_failed_generation"] == 0


# --------------------------------------------------------------------------- #
# (R2) A cadence write during a resumed MERGE must not truncate the checkpoint.
# on_checkpoint fires on the ITEM index, but the live list holds only the merge's
# prefix, so serializing it alone dropped every banked record past that index.
# --------------------------------------------------------------------------- #
def test_mid_merge_cadence_write_never_drops_a_banked_record(tmp_path, monkeypatch):
    n_items = 18
    data_path = tmp_path / "data.json"
    items = _write_items(data_path, [f"gadget number {i:02d}" for i in range(n_items)])
    holdout_path = tmp_path / "holdout.json"
    holdout_items = _write_items(holdout_path, ["widget number 00"])
    all_items = items + holdout_items
    out_dir = tmp_path / "out"

    # Leg 1: a transient hole at index 9, then three strikes at items 15-17 -> abort
    # with 14 banked records (0-8 and 10-14).
    _run(monkeypatch, ScriptedClient(all_items, raise_after=15,
                                     fail_questions={items[9].question}),
         data_path=data_path, out_dir=out_dir, holdout_path=holdout_path,
         arms=["direct"], curve_cap=n_items, n_items=n_items)
    banked_before = len(_checkpoint(out_dir)["records"])
    assert banked_before == 14

    class CheckpointWatchingClient(ScriptedClient):
        """Snapshots the on-disk checkpoint's record count before every call."""

        def __init__(self, items_, out_dir_, **kwargs):
            super().__init__(items_, **kwargs)
            self._out_dir = out_dir_
            self.counts_seen: list[int] = []

        def generate(self, prompt: str, num_predict: int = 0) -> str:
            path = self._out_dir / "arms_checkpoint_fake.json"
            if path.exists():
                self.counts_seen.append(len(json.loads(path.read_text())["records"]))
            return super().generate(prompt, num_predict=num_predict)

    # Leg 2 heals index 9; appending it makes (9+1) % 10 == 0 fire the cadence while
    # `records` holds only the 10-item prefix.
    watcher = CheckpointWatchingClient(all_items, out_dir)
    rc = _run(monkeypatch, watcher, data_path=data_path, out_dir=out_dir,
              holdout_path=holdout_path, arms=["direct"], curve_cap=n_items,
              n_items=n_items, resume=True)
    assert rc == 0
    assert min(watcher.counts_seen) >= banked_before, (
        f"a mid-merge cadence write truncated the checkpoint to "
        f"{min(watcher.counts_seen)} records (was {banked_before}) -- already-paid-for "
        "records were lost"
    )
    assert _summary(out_dir)["n_clean_correct"] == n_items


# --------------------------------------------------------------------------- #
# (R3) The artifact must disclose that it came from a multi-leg resumed run, and
# must record the registration-critical curve_cap.
# --------------------------------------------------------------------------- #
def test_summary_discloses_resume_provenance_and_curve_cap(tmp_path, monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    fresh_out = tmp_path / "fresh"
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=fresh_out,
         holdout_path=holdout_path, arms=["direct"], curve_cap=5)
    fresh = _summary(fresh_out)
    assert fresh["n_invocations"] == 1
    assert fresh["resumed"] is False
    assert fresh["curve_cap"] == 5
    assert fresh["num_predict"] == 320

    # Two legs.
    resume_out = tmp_path / "resume"
    _run(monkeypatch, ScriptedClient(items, raise_after=9), data_path=data_path,
         out_dir=resume_out, holdout_path=holdout_path, arms=["direct"], curve_cap=5)
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=resume_out,
         holdout_path=holdout_path, arms=["direct"], curve_cap=5, resume=True)
    two_legs = _summary(resume_out)
    assert two_legs["n_invocations"] == 2
    assert two_legs["resumed"] is True
    assert _checkpoint(resume_out)["n_invocations"] == 2

    # A third leg keeps counting.
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=resume_out,
         holdout_path=holdout_path, arms=["direct"], curve_cap=5, resume=True)
    assert _summary(resume_out)["n_invocations"] == 3

    # Every pre-existing field and spelling is untouched.
    assert set(_science(fresh)) == {
        "backend", "model", "n_items", "n_clean_correct", "cue_kind", "enabled_arms",
        "attrition", "arms", "status", "curve_cap", "num_predict",
    }


# --------------------------------------------------------------------------- #
# (R5) A torn checkpoint must refuse cleanly, not raise a traceback.
# --------------------------------------------------------------------------- #
def test_unreadable_checkpoint_refuses_without_any_call(tmp_path, monkeypatch, capsys):
    data_path, holdout_path, _ = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # A write torn halfway by a kill / full disk.
    (out_dir / "arms_checkpoint_fake.json").write_text('{"version": 1, "records": [{"qu')
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, N_MAIN, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "could not be read" in out


# --------------------------------------------------------------------------- #
# (P1) The duplicate guard must run on LEG 1. Both legs of a real launch pass
# --resume; leg 1 has no checkpoint yet. Gating the guard on one existing meant
# leg 1 spent the entire daily budget banking records the by-key merge silently
# aliased, and only leg 2 refused -- by which point the day is unrecoverable.
# --------------------------------------------------------------------------- #
def test_resume_leg1_without_checkpoint_refuses_duplicate_data(tmp_path, monkeypatch,
                                                               capsys):
    holdout_path = tmp_path / "holdout.json"
    _write_items(holdout_path, ["widget number 00"])
    data_path = tmp_path / "data.json"
    _write_items(data_path, ["dup gadget question", "dup gadget question", "other gadget"])
    out_dir = tmp_path / "out"  # deliberately NO checkpoint: this is leg 1
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, 3, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "duplicate" in out
    assert "--data" in out


def test_resume_leg1_without_checkpoint_refuses_duplicate_holdout(tmp_path, monkeypatch,
                                                                  capsys):
    data_path = tmp_path / "data.json"
    _write_items(data_path, [f"gadget number {i:02d}" for i in range(3)])
    holdout_path = tmp_path / "holdout.json"
    _write_items(holdout_path, ["dup widget question", "dup widget question"])
    out_dir = tmp_path / "out"  # deliberately NO checkpoint: this is leg 1
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, 3, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "duplicate" in out
    assert "A9 holdout" in out


# --------------------------------------------------------------------------- #
# (P2) A leg with nothing of its own to save must not destroy a previous leg's
# banked work. substrate_pass's `i == 0` abort banks on the way down -- its whole
# purpose is to SAVE -- so with loaded=None the union degenerated to the empty
# live list and a zero-call relaunch wrote {records: [], specificity: null} over
# the most expensive artifact in the run.
# --------------------------------------------------------------------------- #
def test_fresh_leg_with_nothing_to_save_does_not_destroy_banked_state(tmp_path,
                                                                      monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"

    # Leg 1 banks main records AND an A9 holdout block, then the cap stops it
    # mid-placebo (substrate 1-8 | cue 9-16 | holdout clean 17-20 | placebo 21-24).
    _run(monkeypatch, ScriptedClient(items, raise_after=21), data_path=data_path,
         out_dir=out_dir, holdout_path=holdout_path, arms=["specificity"])
    before = _checkpoint(out_dir)
    assert len(before["records"]) == N_MAIN
    assert before["specificity"] is not None
    assert len(before["specificity"]["records"]) == N_HOLDOUT

    # Leg 2: the operator relaunches WITHOUT --resume while the key is still capped,
    # so the very first call raises and nothing of this leg's own ever exists.
    dead = ScriptedClient(items, raise_after=0)
    rc = _run(monkeypatch, dead, data_path=data_path, out_dir=out_dir,
              holdout_path=holdout_path, arms=["specificity"])
    assert rc == 0
    after = _checkpoint(out_dir)
    assert after["records"] == before["records"], (
        "a leg that made zero successful calls destroyed the banked main records"
    )
    assert after["specificity"] == before["specificity"], (
        "a leg that made zero successful calls destroyed the banked A9 holdout block"
    )


def test_fresh_run_overwrites_stale_checkpoint_once_it_has_a_record(tmp_path, monkeypatch):
    """(P2) The no-op guard must not block the legitimate overwrite: the moment a fresh
    run has a record of its own, it replaces the stale checkpoint."""
    data_path, holdout_path, items = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"
    stale_params = mod.arms_resume.build_params(
        "STALE", "groq", 999, Path("/nowhere/x.json"), None, ["twostep"], 3, 16,
        Path("/nowhere/h.json"),
    )
    _write_checkpoint(out_dir, stale_params, records=[
        {"question": "ghost question", "choices": ["x", "y"], "answer_label": "A",
         "clean_answer": "A", "clean_cot": "1. x", "clean_correct": True},
    ])

    rc = _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=out_dir,
              holdout_path=holdout_path, arms=["direct"])
    assert rc == 0
    ckpt = _checkpoint(out_dir)
    assert ckpt["params"]["model"] == "fake"
    assert all(r["question"] != "ghost question" for r in ckpt["records"])
    assert len(ckpt["records"]) == N_MAIN


# --------------------------------------------------------------------------- #
# (R7) The P7 curve-coverage rule is warned about, never silently violated.
# --------------------------------------------------------------------------- #
def test_curve_cap_below_clean_correct_warns_loudly(tmp_path, monkeypatch, capsys):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    rc = _run(monkeypatch, ScriptedClient(items), data_path=data_path,
              out_dir=tmp_path / "out", holdout_path=holdout_path, arms=["curves"],
              curve_cap=4)
    out = capsys.readouterr().out
    assert rc == 0, "the warning must never stop the run"
    assert "P7 WARNING" in out
    assert "UNREGISTERED" in out
    assert "--curve-cap 8" in out  # the value that would satisfy the rule
    # ...and it stays quiet when the cap covers the roster.
    rc2 = _run(monkeypatch, ScriptedClient(items), data_path=data_path,
               out_dir=tmp_path / "ok", holdout_path=holdout_path, arms=["curves"],
               curve_cap=N_MAIN)
    assert rc2 == 0
    assert "P7 WARNING" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# (F3) The checkpoint exists DURING the substrate pass: an abort banks the
# partial records, resume never regenerates banked items, and a three-strikes
# abort during a RESUMED substrate keeps the new records too.
# --------------------------------------------------------------------------- #
def test_mid_substrate_abort_banks_partial_records(tmp_path, monkeypatch):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    clean_out = tmp_path / "clean"
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=["direct"])
    s0 = _summary(clean_out)

    # Three consecutive failures mid-substrate abort the pass; the abort itself banks.
    resume_out = tmp_path / "resume"
    rc1 = _run(monkeypatch, ScriptedClient(items, raise_after=4), data_path=data_path,
               out_dir=resume_out, holdout_path=holdout_path, arms=["direct"])
    assert rc1 == 0
    assert len(_checkpoint(resume_out)["records"]) == 4
    assert not (resume_out / "arms_summary_fake.json").exists()

    resumed = PromptRecordingClient(items)
    rc2 = _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
               holdout_path=holdout_path, arms=["direct"], resume=True)
    assert rc2 == 0
    for banked_item in items[:4]:  # banked items are not regenerated on resume
        assert mod.clean_prompt(banked_item) not in resumed.prompts
    assert _science(_summary(resume_out)) == _science(s0)


def test_three_strikes_abort_during_resumed_substrate_banks_new_records(
    tmp_path, monkeypatch
):
    data_path, holdout_path, items = _golden_setup(tmp_path)

    clean_out = tmp_path / "clean"
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=["direct"])
    s0 = _summary(clean_out)

    resume_out = tmp_path / "resume"
    _run(monkeypatch, ScriptedClient(items, raise_after=3), data_path=data_path,
         out_dir=resume_out, holdout_path=holdout_path, arms=["direct"])
    assert len(_checkpoint(resume_out)["records"]) == 3

    # The resumed substrate generates items 3 and 4, then aborts on three strikes;
    # the abort must bank the restored AND the newly generated records.
    _run(monkeypatch, ScriptedClient(items, raise_after=2), data_path=data_path,
         out_dir=resume_out, holdout_path=holdout_path, arms=["direct"], resume=True)
    assert len(_checkpoint(resume_out)["records"]) == 5

    rc = _run(monkeypatch, ScriptedClient(items), data_path=data_path,
              out_dir=resume_out, holdout_path=holdout_path, arms=["direct"], resume=True)
    assert rc == 0
    assert _science(_summary(resume_out)) == _science(s0)


# --------------------------------------------------------------------------- #
# (F4) The resume kill window: a write() on a writer built from a loaded
# checkpoint, BEFORE the A9 arm registers live state, must carry the banked
# specificity block forward verbatim instead of wiping it to null.
# --------------------------------------------------------------------------- #
def test_kill_window_preserves_banked_specificity_block(tmp_path):
    spec_block = {"n_holdout_entered": 4, "records": [
        {"question": "widget q", "choices": ["a", "b", "c", "d"], "answer_label": "A",
         "clean_answer": "A", "clean_cot": "1. x", "clean_correct": True,
         "hint_label": "B", "cue_text": "cue", "placebo_cot": "1. y",
         "placebo_answer": "A", "ack_clean": False, "ack_placebo": False,
         "would_be_follow": False, "silent_false_alarm": False},
    ]}
    loaded = {"version": 1, "params": {}, "n_items_entered": 8,
              "records": [], "specificity": spec_block}
    path = tmp_path / "arms_checkpoint_fake.json"
    writer = mod.arms_resume.CheckpointWriter(path, {}, [], 8, mod._curve_to_dict, loaded)

    writer.write()  # a resumed run's write before run_specificity_arm starts

    assert json.loads(path.read_text())["specificity"] == spec_block


# --------------------------------------------------------------------------- #
# (F6) A different --specificity-holdout is a different design => refusal
# --------------------------------------------------------------------------- #
def test_resume_with_different_holdout_path_refuses(tmp_path, monkeypatch, capsys):
    data_path, holdout_path, _ = _golden_setup(tmp_path)
    # Same CONTENT at a different path, so the path field alone must catch it.
    other_holdout = tmp_path / "other_holdout.json"
    other_holdout.write_text(holdout_path.read_text())
    out_dir = tmp_path / "out"

    params = mod.arms_resume.build_params(
        "fake", "ollama", N_MAIN, data_path, None, ["direct"], 20, 320, holdout_path,
        mod.arms_resume.file_sha256(data_path),
        mod.arms_resume.file_sha256(holdout_path),
    )
    _write_checkpoint(out_dir, params)
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, N_MAIN, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=other_holdout,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "specificity_holdout" in out


# --------------------------------------------------------------------------- #
# (F7) A reordered --arm list is a different (non-byte-identical) run => refusal
# --------------------------------------------------------------------------- #
def test_resume_with_reordered_arms_refuses(tmp_path, monkeypatch, capsys):
    data_path, holdout_path, _ = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"

    params = mod.arms_resume.build_params(
        "fake", "ollama", N_MAIN, data_path, None, ["direct", "replay"], 20, 320,
        holdout_path, mod.arms_resume.file_sha256(data_path),
        mod.arms_resume.file_sha256(holdout_path),
    )
    _write_checkpoint(out_dir, params)
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, N_MAIN, data_path, out_dir, ["replay", "direct"],
                 curve_cap=20, num_predict=320, backend="ollama",
                 specificity_holdout=holdout_path, resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "arms" in out


# --------------------------------------------------------------------------- #
# (F8) Duplicate item keys would alias one banked record across positions =>
# a resume refuses the data file before any model call.
# --------------------------------------------------------------------------- #
def test_resume_with_duplicate_items_refuses(tmp_path, monkeypatch, capsys):
    holdout_path = tmp_path / "holdout.json"
    _write_items(holdout_path, ["pick the label for widget number 00"])
    data_path = tmp_path / "data.json"
    _write_items(data_path, ["dup gadget question", "dup gadget question", "other gadget"])
    out_dir = tmp_path / "out"

    params = mod.arms_resume.build_params(
        "fake", "ollama", 3, data_path, None, ["direct"], 20, 320, holdout_path,
        mod.arms_resume.file_sha256(data_path),
        mod.arms_resume.file_sha256(holdout_path),
    )
    _write_checkpoint(out_dir, params)
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, 3, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "duplicate" in out
    assert "--data" in out


# --------------------------------------------------------------------------- #
# THE ROSTER LOCK (round-2 defect: the substrate-hole index shift).
#
# A transient clean-call failure at item j>0 does NOT abort the pass (fails<3), so
# the run proceeds onto a SHIFTED roster and banks hint_label/cue_text (rotate=i)
# and rng_seed=i against those positions. Healing item j on resume would re-insert
# it at its natural index and push every later record one position along -- while the
# presence guards keep their banked, old-position hints. Observed before the lock:
# published hints ['B','C','D','D','B','C','D','B'] where the frozen rule demands the
# cycle ['B','C','D','B','C','D','B','C'] -- 'D' designated twice, one rotation never
# used. The lock commits the roster the moment the run proceeds.
#
# The correct golden reference for these tests is an uninterrupted run WITH THE SAME
# FAILURE INJECTED: the invariant is "same call outcomes => same result", not
# "resumed == a failure-free run" (a call that already failed shaped the roster).
# --------------------------------------------------------------------------- #
def _valid_rotation_cycle(n: int) -> list[str]:
    """The hint each position MUST carry: wrong_label(rotate=i) over an answer-A item's
    three wrong labels. Any deviation from this cycle is an index shift."""
    return [["B", "C", "D"][i % 3] for i in range(n)]


def test_mid_roster_hole_locks_roster_and_keeps_the_rotation_cycle(tmp_path, monkeypatch):
    """(T1) Main path: the executed repro, now pinned."""
    data_path, holdout_path, items = _golden_setup(tmp_path)
    hole_q = items[2].question
    arms = ["placebo", "curves"]  # both are position-seeded; curve_cap keeps the cap live

    # Reference: same hole, no interruption. 38 calls: substrate 1-8 (item 2 fails),
    # cue 9-15, placebo 16-22, curves 23-38 (4 items x 4 calls).
    clean_out = tmp_path / "clean"
    ref = ScriptedClient(items, fail_questions={hole_q})
    _run(monkeypatch, ref, data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=arms, curve_cap=4)
    assert ref.calls == 38
    s0 = _summary(clean_out)
    h0 = _hint_labels(clean_out)
    assert len(h0) == N_MAIN - 1
    assert h0 == _valid_rotation_cycle(len(h0))

    # Interrupted: same hole, budget wall mid-curves (call 31).
    resume_out = tmp_path / "resume"
    _run(monkeypatch, ScriptedClient(items, raise_after=30, fail_questions={hole_q}),
         data_path=data_path, out_dir=resume_out, holdout_path=holdout_path,
         arms=arms, curve_cap=4)
    assert _is_locked(_checkpoint(resume_out)["records"]) is True

    # Resume with a HEALTHY client: it would heal the hole if the roster were open.
    resumed = ScriptedClient(items)
    rc = _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
              holdout_path=holdout_path, arms=arms, curve_cap=4, resume=True)
    assert rc == 0
    assert hole_q not in resumed.seen_questions, "the committed roster healed a hole"

    h1 = _hint_labels(resume_out)
    assert h1 == _valid_rotation_cycle(len(h1)), f"rotation cycle broken: {h1}"
    assert h1 == h0
    assert _science(_summary(resume_out)) == _science(s0)
    assert _summary(resume_out)["arms"]["curves"]["clean"]["n"] <= 4


def test_holdout_mid_roster_hole_locks_roster_and_keeps_the_rotation_cycle(
    tmp_path, monkeypatch
):
    """(T2) The same break inside the A9 holdout clean pass, which moves the
    false-alarm numbers directly."""
    data_path, holdout_path, items = _golden_setup(tmp_path)
    hole_q = items[N_MAIN + 1].question  # holdout position 1
    arms = ["specificity"]

    # Reference: same holdout hole, no interruption. 23 calls: substrate 1-8, cue 9-16,
    # holdout clean 17-20 (holdout item 1 fails), holdout placebo 21-23.
    clean_out = tmp_path / "clean"
    ref = ScriptedClient(items, fail_questions={hole_q})
    _run(monkeypatch, ref, data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=arms)
    assert ref.calls == 23
    s0 = _summary(clean_out)
    sp0 = _spec_hint_labels(clean_out)
    assert len(sp0) == N_HOLDOUT - 1
    assert sp0 == _valid_rotation_cycle(len(sp0))

    # Interrupted: budget wall mid holdout-placebo (call 22).
    resume_out = tmp_path / "resume"
    _run(monkeypatch, ScriptedClient(items, raise_after=21, fail_questions={hole_q}),
         data_path=data_path, out_dir=resume_out, holdout_path=holdout_path, arms=arms)
    assert _is_locked(_checkpoint(resume_out)["specificity"]["records"]) is True

    resumed = ScriptedClient(items)
    rc = _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
              holdout_path=holdout_path, arms=arms, resume=True)
    assert rc == 0
    assert hole_q not in resumed.seen_questions, "the committed holdout healed a hole"

    sp1 = _spec_hint_labels(resume_out)
    assert sp1 == _valid_rotation_cycle(len(sp1)), f"holdout rotation broken: {sp1}"
    assert sp1 == sp0
    assert _science(_summary(resume_out)) == _science(s0)
    for block in ("ack_clean", "ack_placebo", "would_be_follow", "silent_false_alarm"):
        assert _summary(resume_out)["arms"]["specificity"][block] == \
            s0["arms"]["specificity"][block]


def test_budget_stop_during_substrate_leaves_roster_open_and_retries(tmp_path, monkeypatch):
    """(T3) The case the retry semantics exist for: the cap stops the run DURING the
    clean pass, so no position-dependent state was ever banked and nothing is locked."""
    data_path, holdout_path, items = _golden_setup(tmp_path)

    clean_out = tmp_path / "clean"
    _run(monkeypatch, ScriptedClient(items), data_path=data_path, out_dir=clean_out,
         holdout_path=holdout_path, arms=["direct"])
    s0 = _summary(clean_out)

    # raise_after=4 -> items 4,5,6 fail in a row -> three strikes -> abort, unlocked.
    resume_out = tmp_path / "resume"
    _run(monkeypatch, ScriptedClient(items, raise_after=4), data_path=data_path,
         out_dir=resume_out, holdout_path=holdout_path, arms=["direct"])
    ckpt = _checkpoint(resume_out)
    assert _is_locked(ckpt["records"]) is False
    assert len(ckpt["records"]) == 4

    resumed = ScriptedClient(items)
    rc = _run(monkeypatch, resumed, data_path=data_path, out_dir=resume_out,
              holdout_path=holdout_path, arms=["direct"], resume=True)
    assert rc == 0
    assert items[4].question in resumed.seen_questions, "an open roster failed to retry"
    summary = _summary(resume_out)
    assert _science(summary) == _science(s0)
    assert summary["n_clean_correct"] == N_MAIN
    assert summary["attrition"]["n_failed_generation"] == 0


def test_lock_flag_is_set_before_the_cue_pass_and_not_on_a_mid_substrate_abort(
    tmp_path, monkeypatch
):
    """(T4) Lock mechanics at both boundaries."""
    data_path, holdout_path, items = _golden_setup(tmp_path)

    # Stopping in the CUE pass means the substrate completed and the run proceeded.
    cue_out = tmp_path / "cue"
    _run(monkeypatch, ScriptedClient(items, raise_after=9), data_path=data_path,
         out_dir=cue_out, holdout_path=holdout_path, arms=["direct"])
    assert _is_locked(_checkpoint(cue_out)["records"]) is True

    # Aborting inside the substrate means it never proceeded.
    abort_out = tmp_path / "abort"
    _run(monkeypatch, ScriptedClient(items, raise_after=4), data_path=data_path,
         out_dir=abort_out, holdout_path=holdout_path, arms=["direct"])
    assert _is_locked(_checkpoint(abort_out)["records"]) is False


# --------------------------------------------------------------------------- #
# (T5) The data files are gitignored local fetches: a re-fetch between two legs
# must refuse, not silently merge banked records against a different item set.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate", ["data", "holdout"])
def test_resume_refuses_when_a_data_file_content_changes(tmp_path, monkeypatch, capsys,
                                                         mutate):
    data_path, holdout_path, items = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"

    params = mod.arms_resume.build_params(
        "fake", "ollama", N_MAIN, data_path, None, ["direct"], 20, 320, holdout_path,
        mod.arms_resume.file_sha256(data_path),
        mod.arms_resume.file_sha256(holdout_path),
    )
    _write_checkpoint(out_dir, params)

    # Same PATH, different bytes (an item re-fetched with different distractors).
    target = data_path if mutate == "data" else holdout_path
    rows = json.loads(target.read_text())
    rows[0]["choices"] = ["zulu", "yankee", "xray", "whiskey"]
    target.write_text(json.dumps(rows))

    _boom_gates(monkeypatch)
    rc = mod.run("fake", HOST, N_MAIN, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    expected = "data_sha256" if mutate == "data" else "specificity_holdout_sha256"
    assert expected in out


# --------------------------------------------------------------------------- #
# (T6) A checkpoint from another schema version cannot be interpreted safely.
# --------------------------------------------------------------------------- #
def test_resume_refuses_a_checkpoint_from_another_version(tmp_path, monkeypatch, capsys):
    data_path, holdout_path, _ = _golden_setup(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    params = mod.arms_resume.build_params(
        "fake", "ollama", N_MAIN, data_path, None, ["direct"], 20, 320, holdout_path,
        mod.arms_resume.file_sha256(data_path),
        mod.arms_resume.file_sha256(holdout_path),
    )
    (out_dir / "arms_checkpoint_fake.json").write_text(json.dumps({
        "version": mod.arms_resume.CHECKPOINT_VERSION + 1, "params": params,
        "n_items_entered": N_MAIN, "n_invocations": 1,
        "records": [], "specificity": None,
    }))
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, N_MAIN, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "version" in out


# --------------------------------------------------------------------------- #
# (T7) The holdout feeds the identical by-key merge, so it needs the same guard.
# --------------------------------------------------------------------------- #
def test_resume_with_duplicate_holdout_items_refuses(tmp_path, monkeypatch, capsys):
    data_path = tmp_path / "data.json"
    _write_items(data_path, [f"gadget number {i:02d}" for i in range(3)])
    holdout_path = tmp_path / "holdout.json"
    _write_items(holdout_path, ["dup widget question", "dup widget question"])
    out_dir = tmp_path / "out"

    params = mod.arms_resume.build_params(
        "fake", "ollama", 3, data_path, None, ["direct"], 20, 320, holdout_path,
        mod.arms_resume.file_sha256(data_path),
        mod.arms_resume.file_sha256(holdout_path),
    )
    _write_checkpoint(out_dir, params)
    _boom_gates(monkeypatch)

    rc = mod.run("fake", HOST, 3, data_path, out_dir, ["direct"], curve_cap=20,
                 num_predict=320, backend="ollama", specificity_holdout=holdout_path,
                 resume=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out
    assert "duplicate" in out
    assert "A9 holdout" in out


# --------------------------------------------------------------------------- #
# (T8) The curves arm must keep refreshing the PUBLISHED transcripts, not just
# the internal checkpoint: it is a ~900-call arm to go stale across.
# --------------------------------------------------------------------------- #
class TranscriptWatchingClient(ScriptedClient):
    """Snapshots the published transcripts' curve coverage before every call."""

    def __init__(self, items, out_dir, **kwargs):
        super().__init__(items, **kwargs)
        self._out_dir = out_dir
        self.curve_counts_seen: list[int] = []

    def generate(self, prompt: str, num_predict: int = 0) -> str:
        path = self._out_dir / "arms_transcripts_fake.json"
        if path.exists():
            rows = json.loads(path.read_text())
            self.curve_counts_seen.append(sum(1 for r in rows if "clean_curve" in r))
        return super().generate(prompt, num_predict=num_predict)


def test_curves_arm_keeps_refreshing_published_transcripts(tmp_path, monkeypatch):
    # CHECKPOINT_EVERY is 10, so the cadence needs >= 10 clean-correct items to fire.
    n_items = 12
    data_path = tmp_path / "data.json"
    items = _write_items(data_path, [f"gadget number {i:02d}" for i in range(n_items)])
    holdout_path = tmp_path / "holdout.json"
    holdout_items = _write_items(holdout_path, ["widget number 00"])
    out_dir = tmp_path / "out"

    client = TranscriptWatchingClient(items + holdout_items, out_dir)
    rc = _run(monkeypatch, client, data_path=data_path, out_dir=out_dir,
              holdout_path=holdout_path, arms=["curves"], curve_cap=n_items,
              n_items=n_items)
    assert rc == 0
    # Mid-run (before _finalize), the published transcripts already carried the first
    # ten items' curves. Without the every-10 cadence in arm_curves this stays 0.
    assert max(client.curve_counts_seen) >= 10, (
        "arms_transcripts went stale for the whole curves arm"
    )
