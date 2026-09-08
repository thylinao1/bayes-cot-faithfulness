"""Ruling R12(6): the four class-2 roster rows carry BCF_REASONING_MODE=off.

`bcf/apply_reasoning_mode.py` adds `BCF_REASONING_MODE=off` to every row of the four
class-2 roster rows (`docs/ROSTER-TEMPLATES.md` rows 4, 5, 9, 10: a chat template that
opens a reasoning block with no documented switch) in every sweep and enrich-* wave
manifest. This file checks the manifests as committed -- it does not re-run the script --
and checks the script's own idempotence and refusal behaviour directly.

Scope note, matching `bcf/apply_reasoning_mode.is_in_scope`: `ladder-*.tsv` manifests are
excluded from every check below. Their MODEL column names a served checkpoint
(`bcf-ladder/...`), never a roster id, and the ladder-eval rows already carry their own
`BCF_REASONING_MODE=default` for Qwen3-8B (class 1, a documented switch) under
`docs/REASONING-MODE-IMPL.md`'s separate mechanism -- a different ruling reaching a
different row through a different field-setting path, not a conflict with this one.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAVES = REPO / "bcf" / "waves"
sys.path.insert(0, str(REPO))

import bcf.apply_reasoning_mode as arm

CLASS2_MODELS = arm.CLASS2_MODELS


def _in_scope_manifests():
    return sorted(p for p in WAVES.glob("*.tsv") if arm.is_in_scope(p))


def _rows(path: Path):
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        yield fields[0], dict(f.split("=", 1) for f in fields[3:] if "=" in f)


def test_every_class2_row_in_scope_carries_reasoning_mode_off():
    seen_models = set()
    n_rows = 0
    for path in _in_scope_manifests():
        for model, kv in _rows(path):
            if model not in CLASS2_MODELS:
                continue
            n_rows += 1
            seen_models.add(model)
            assert kv.get("BCF_REASONING_MODE") == "off", (path.name, model, kv)
    # Every one of the four rows actually appears somewhere in scope; a model with zero
    # rows would make the assertion above vacuously true for it.
    assert seen_models == CLASS2_MODELS, seen_models
    # a100-40 (2 models x 12 waves) + a100-80 (1 x 12) + enrich-a100-40 (2 x 3) +
    # h100-96-tp2 (1 x 12) = 24 + 12 + 6 + 12, plus the 4 resub-02 rows (2 models x 2
    # voided cells, 2026-09-08) = 58, plus microsoft/Phi-4-reasoning after the R12(3)
    # resolution (12 a100-40 sweep rows + 3 enrich rows + 2 resub-03 rows = 17) = 75
    assert n_rows == 75, n_rows


def test_no_other_row_in_scope_carries_reasoning_mode():
    """The field is additive and targeted: nothing else picked it up by accident."""
    offenders = []
    for path in _in_scope_manifests():
        for model, kv in _rows(path):
            if model in CLASS2_MODELS:
                continue
            if "BCF_REASONING_MODE" in kv:
                offenders.append((path.name, model))
    assert not offenders, offenders


def test_ladder_manifests_are_untouched_by_this_ruling():
    """Ladder rows never name a class-2 roster id, so this script has nothing to add
    there; their existing BCF_REASONING_MODE=default rows (a different roster row, a
    different ruling) are not this test's business and are left exactly as they are.
    """
    for path in sorted(WAVES.glob("ladder-*.tsv")):
        for model, _kv in _rows(path):
            assert model not in CLASS2_MODELS, (path.name, model)


def test_apply_reasoning_mode_is_idempotent_on_the_committed_manifests():
    plans, conflicts = arm.run(WAVES, apply=False)
    assert conflicts == []
    assert sum(p.n_changed for p in plans) == 0, [
        (p.path.name, p.n_changed) for p in plans if p.n_changed
    ]


def test_apply_reasoning_mode_refuses_a_conflicting_value(tmp_path):
    (tmp_path / "fake-01.tsv").write_text(
        "# t\n"
        "allenai/Olmo-3-7B-Think\tarc_challenge\tstated-hint\t"
        "BCF_REVISION=x\tBCF_REASONING_MODE=on\n"
    )
    before = (tmp_path / "fake-01.tsv").read_text()
    _plans, conflicts = arm.run(tmp_path, apply=True)
    assert len(conflicts) == 1
    assert conflicts[0].model == "allenai/Olmo-3-7B-Think"
    assert conflicts[0].found_value == "on"
    # REFUSING writes nothing, even to files that would have been clean.
    assert (tmp_path / "fake-01.tsv").read_text() == before


def test_apply_reasoning_mode_leaves_every_other_field_untouched(tmp_path):
    original = (
        "# comment line, verbatim\n"
        "\n"
        "allenai/Olmo-3-7B-Think\tarc_challenge\tstated-hint\t"
        "BCF_REVISION=abc\tBCF_TP=1\tBCF_EXPECTED_HOURS=1.477\n"
        "Qwen/Qwen3-8B\tarc_challenge\tstated-hint\tBCF_REVISION=def\n"
    )
    (tmp_path / "fake-02.tsv").write_text(original)
    arm.run(tmp_path, apply=True)
    lines = (tmp_path / "fake-02.tsv").read_text().split("\n")
    assert lines[0] == "# comment line, verbatim"
    assert lines[1] == ""
    assert lines[2] == (
        "allenai/Olmo-3-7B-Think\tarc_challenge\tstated-hint\t"
        "BCF_REVISION=abc\tBCF_TP=1\tBCF_EXPECTED_HOURS=1.477\tBCF_REASONING_MODE=off"
    )
    # the non-class-2 row is byte for byte what it was.
    assert lines[3] == "Qwen/Qwen3-8B\tarc_challenge\tstated-hint\tBCF_REVISION=def"


def test_a_row_already_at_off_is_not_recounted_as_changed(tmp_path):
    (tmp_path / "fake-03.tsv").write_text(
        "allenai/Olmo-3-7B-Think\tarc_challenge\tstated-hint\t"
        "BCF_REVISION=abc\tBCF_REASONING_MODE=off\n"
    )
    plans, conflicts = arm.run(tmp_path, apply=True)
    assert conflicts == []
    assert sum(p.n_changed for p in plans) == 0
