#!/usr/bin/env python
"""Add BCF_REASONING_MODE=off to every class-2 roster row in the wave manifests.

Ruling R12(1) (`experiments/PREREGISTRATION_jury_and_scale.md` amendment A4.4) defines,
for a roster row whose chat template opens a reasoning block with no documented switch,
closing that block as the switch: configuration A (`BCF_REASONING_MODE=off`) is the cell
of record. `docs/ROSTER-TEMPLATES.md` classifies four roster rows this way (class 2,
"opens a reasoning block, no switch", rows 4, 5, 9 and 10 of the element-10 table):

    allenai/Olmo-3-7B-Think
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    allenai/Olmo-3-32B-Think
    deepseek-ai/DeepSeek-R1-Distill-Llama-70B

This script adds `BCF_REASONING_MODE=off` as a new trailing KEY=VALUE field to every row
of those four models in every `bcf/waves/*.tsv` SWEEP manifest and every
`bcf/waves/enrich-*.tsv` manifest. `bcf/wave.sh` exports every KEY=VALUE field of a row to
the job's environment unedited (the same mechanism `BCF_REVISION` already uses -- see its
row-reading loop), so this is how the mode reaches `BCF_REASONING_MODE` in
`bcf/serve_and_run.sbatch`.

`ladder-*.tsv` manifests are OUT OF SCOPE. Their MODEL column names a served checkpoint
(`bcf-ladder/...`), never one of the four roster ids above (a ladder row trains or
evaluates ONE roster base per file, and none of the four class-2 rows currently has a
ladder manifest), and the ladder-eval rows that DO exist already carry their own
`BCF_REASONING_MODE=default` for a different roster row (Qwen3-8B, class 1, a documented
switch) under `docs/REASONING-MODE-IMPL.md`'s separate mechanism. Touching them is not
this ruling's business.

Idempotent: a row that already carries `BCF_REASONING_MODE=off` is left untouched (not
counted as changed, and the file is not rewritten if that is its only class-2 row).
A row that carries `BCF_REASONING_MODE` set to any OTHER value REFUSES THE WHOLE RUN
(exit 1) before writing anything, rather than silently overwriting a value this script
does not know the reason for. Every other byte of every row -- MODEL, SUBSTRATE, CUE,
every other KEY=VALUE field, comments, blank lines, trailing newline -- is untouched; the
new field is appended after the row's last existing field.

    python bcf/apply_reasoning_mode.py            # apply, print the diff stat
    python bcf/apply_reasoning_mode.py --check     # report only; write nothing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAVES = REPO / "bcf" / "waves"

CLASS2_MODELS = frozenset({
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "allenai/Olmo-3-32B-Think",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
})

FIELD = "BCF_REASONING_MODE"
VALUE = "off"


@dataclass(frozen=True)
class RowConflict:
    file: str
    line: int
    model: str
    found_value: str


@dataclass(frozen=True)
class FilePlan:
    path: Path
    new_lines: tuple[str, ...]   # the file's lines AFTER the edit (same length as before)
    n_changed: int               # rows this plan adds the field to
    conflicts: tuple[RowConflict, ...]


def is_in_scope(path: Path) -> bool:
    """Sweep manifests and enrich-* manifests; ladder-*.tsv is excluded (see module docstring)."""
    # explore-*.tsv manifests hold exploratory cells run before a ruling; they may carry
    # BCF_REASONING_MODE on any model by design (the Phi-4 off-mode check under R12(3)),
    # so they are outside this script's scope like the ladder manifests.
    return not path.name.startswith(("ladder-", "explore-"))


def plan_file(path: Path) -> FilePlan:
    """Read one manifest and compute its edit without writing anything.

    A comment or blank line, and any data row whose model is not one of the four, is
    copied into ``new_lines`` byte for byte. A class-2 row already carrying the field at
    the right value is copied unchanged too (that is what makes a second run a no-op). A
    class-2 row carrying no such field gets it appended. A class-2 row carrying a
    DIFFERENT value is recorded as a conflict and not edited; the caller refuses the
    whole run rather than acting on a partial plan.
    """
    lines = path.read_text().split("\n")
    new_lines = list(lines)
    n_changed = 0
    conflicts: list[RowConflict] = []
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue   # not a KEY=VALUE row; not this script's problem to fix
        model = fields[0]
        if model not in CLASS2_MODELS:
            continue
        existing = None
        for f in fields[3:]:
            if f.startswith(FIELD + "="):
                existing = f[len(FIELD) + 1:]
                break
        if existing is None:
            new_lines[i] = line + "\t" + f"{FIELD}={VALUE}"
            n_changed += 1
        elif existing != VALUE:
            conflicts.append(RowConflict(path.name, i + 1, model, existing))
        # existing == VALUE: already applied; left untouched, not counted as changed.
    return FilePlan(path=path, new_lines=tuple(new_lines), n_changed=n_changed,
                     conflicts=tuple(conflicts))


def run(waves_dir: Path = WAVES, apply: bool = True) -> tuple[list[FilePlan], list[RowConflict]]:
    """Plan every in-scope manifest, then either apply (default) or only report.

    Conflicts are collected across ALL files before anything is written: a conflict
    found in file 30 of 39 must not leave files 1 through 29 half-edited. If there is
    any conflict, nothing is written, whatever ``apply`` says.
    """
    plans: list[FilePlan] = []
    all_conflicts: list[RowConflict] = []
    for tsv in sorted(waves_dir.glob("*.tsv")):
        if not is_in_scope(tsv):
            continue
        plan = plan_file(tsv)
        plans.append(plan)
        all_conflicts.extend(plan.conflicts)
    if all_conflicts:
        return plans, all_conflicts
    if apply:
        for plan in plans:
            if plan.n_changed == 0:
                continue
            plan.path.write_text("\n".join(plan.new_lines))
    return plans, all_conflicts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waves-dir", type=Path, default=WAVES)
    ap.add_argument("--check", action="store_true",
                     help="report the diff stat; write nothing")
    args = ap.parse_args(argv)

    plans, conflicts = run(args.waves_dir, apply=not args.check)
    if conflicts:
        print(f"REFUSING: {len(conflicts)} row(s) already carry a different "
              f"{FIELD}; nothing was written.")
        for c in conflicts:
            print(f"  {c.file}:{c.line} {c.model} already has {FIELD}={c.found_value}")
        return 1

    verb = "would change" if args.check else "changed"
    touched_files = 0
    total_rows = 0
    for plan in plans:
        if plan.n_changed == 0:
            continue
        touched_files += 1
        total_rows += plan.n_changed
        print(f"  {plan.path.name}: {plan.n_changed} row(s) {verb}")
    print(f"{touched_files} file(s), {total_rows} row(s) {verb} total "
          f"(scanned {len(plans)} manifest(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
