"""The half of `wave.sh --check-only` that does NOT need the live cluster.

`wave.sh --check-only` does two different kinds of checking and only one of them needs
a cluster. The LIVE half reads `squeue` for the per-user card caps and hands every
command line to `sbatch --test-only`; that half cannot run from the laptop and its
output is `bcf/waves/dry-run-<wave>.txt`. The STRUCTURAL half is a property of the
manifests themselves: the serving constants on every row, the pinned revision, the
tensor-parallel size against the cards per node, the wave's card count against its pool's
sweep slots, and the memory and CPU fields whose absence killed job 825536.

This script is the structural half, so a repricing can be checked the moment it is
written instead of waiting for a VPN. It never submits anything and never reads squeue.

    python bcf/check_wave_manifests.py               # exit 0 clean, 1 on any problem
    python bcf/check_wave_manifests.py --out FILE    # and write the report there
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAVES = REPO / "bcf" / "waves"

# From wave.sh: cards PER NODE by GPU type (sinfo -o "%n %G", 2026-09-06). A
# tensor-parallel size above this is rejected at submit time however free the cluster is.
CARDS_PER_NODE = {"a100-40": 2, "a100-80": 1, "h100-47": 4, "h100-96": 2, "h200-141": 4}


def _load_plan_waves():
    spec = importlib.util.spec_from_file_location(
        "plan_waves_check", REPO / "bcf" / "plan_waves.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows_of(path: Path):
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            yield n, None, None, None, {}, "fewer than four tab-separated fields"
            continue
        kv = {}
        bad = None
        for f in fields[3:]:
            if "=" not in f:
                bad = f"field {f!r} is not KEY=VALUE"
                continue
            k, v = f.split("=", 1)
            kv[k] = v
        yield n, fields[0], fields[1], fields[2], kv, bad


# Element 11(b) and CONTRACT.md line 24: 12 checkpoints per base.
CONTRACT_LADDER_CHECKPOINTS = 12
LADDER_TRAIN_KEYS = ("BCF_LADDER_VARIANT", "BCF_LADDER_RUNG", "BCF_LADDER_SEED",
                     "BCF_LADDER_POOL", "BCF_LADDER_GUARD")
LADDER_EVAL_KEYS = ("BCF_MODEL_PATH", "BCF_LADDER_BASE", "BCF_BASE_REVISION",
                    "BCF_ARMS", "BCF_N_ITEMS", "BCF_CURVE_CAP")


def check_ladder_row(where, model, substrate, cue, kv, roster, pw, seen_cell_ids):
    """Element 11 rows: what a ladder row must carry instead of a roster revision.

    A TRAINING row's MODEL is the roster id and its revision is the roster's pinned sha,
    because a LoRA job loads the pinned base. An EVALUATION row's MODEL is a served name
    for a local checkpoint, its BCF_REVISION is ladder-<sha16 of the checkpoint
    manifest>, and the roster row it came from is named by BCF_LADDER_BASE plus
    BCF_BASE_REVISION. Both kinds must carry a unique cell id, the memory and CPU fields
    whose absence killed job 825536, and (for the evaluation rows, which generate) the
    ruling R1 serving constants.
    """
    problems = []
    is_eval = model.startswith("bcf-ladder/")
    # Uniqueness is per STAGE: the same rung legitimately appears once as a training job
    # and once as the evaluation of the checkpoint that job wrote.
    cell_id = kv.get("BCF_LADDER_CELL_ID")
    key = ("evaluate" if is_eval else "train", cell_id)
    if not cell_id:
        problems.append(
            f"{where}: BCF_LADDER_CELL_ID is missing; a ladder row that cannot name "
            "its rung is unreadable in squeue and in the results")
    elif key in seen_cell_ids:
        problems.append(f"{where}: cell id {cell_id} already appears at "
                        f"{seen_cell_ids[key]} in the same stage; two checkpoints would "
                        "share one directory")
    else:
        seen_cell_ids[key] = where
    for field in ("MEM", "CPUS"):
        if not kv.get(field):
            problems.append(f"{where}: {field} is missing (the partition default is 3G "
                            "and a job under it dies with no legible reason, job 825536)")
    if substrate not in pw.SUBSTRATES:
        problems.append(f"{where}: substrate {substrate} is not one of the three")
    if cue not in pw.CUE_FAMILIES:
        problems.append(f"{where}: cue family {cue} is not one of the four")

    if is_eval:
        for field in LADDER_EVAL_KEYS:
            if not kv.get(field):
                problems.append(f"{where}: {field} is missing from an evaluation row")
        base = kv.get("BCF_LADDER_BASE")
        if base not in roster:
            problems.append(f"{where}: BCF_LADDER_BASE={base} is not on the element 10 "
                            "roster; a ladder checkpoint is still trained from a roster row")
        elif kv.get("BCF_BASE_REVISION") != roster[base][0]:
            problems.append(
                f"{where}: BCF_BASE_REVISION={kv.get('BCF_BASE_REVISION')} but the "
                f"roster pins {roster[base][0]} for {base}")
        rev = kv.get("BCF_REVISION", "")
        if not rev.startswith("ladder-"):
            problems.append(
                f"{where}: BCF_REVISION={rev!r} does not start with 'ladder-'. An "
                "evaluation row's revision IS the checkpoint hash; a Hub sha there would "
                "name weights this row does not serve")
        # These rows GENERATE, so ruling R1's serving mode applies to them exactly as it
        # does to a cell.
        if kv.get("BCF_BATCH_INVARIANT") != "1":
            problems.append(f"{where}: BCF_BATCH_INVARIANT="
                            f"{kv.get('BCF_BATCH_INVARIANT')} (ruling R1)")
        if kv.get("BCF_CONCURRENCY") != "32":
            problems.append(f"{where}: BCF_CONCURRENCY={kv.get('BCF_CONCURRENCY')} "
                            "(ruling R1: 32 in flight)")
    else:
        for field in LADDER_TRAIN_KEYS:
            if not kv.get(field):
                problems.append(f"{where}: {field} is missing from a training row")
        if model not in roster:
            problems.append(f"{where}: {model} is not on the element 10 roster")
        elif kv.get("BCF_REVISION") != roster[model][0]:
            problems.append(
                f"{where}: BCF_REVISION={kv.get('BCF_REVISION')} but the roster pins "
                f"{roster[model][0]} for {model}")
    return problems


def check(waves_dir: Path = WAVES) -> tuple[list[str], dict]:
    pw = _load_plan_waves()
    plan = json.loads((waves_dir / "plan.json").read_text())
    roster = {hf: (rev, pool, tp) for hf, rev, _fam, pool, tp, *_ in pw.ROSTER}
    problems: list[str] = []
    counts = {"tsv_files": 0, "rows": 0, "enrich_rows": 0, "resub_rows": 0, "explore_rows": 0,
              "ladder_rows": 0, "models": set(), "pools": set()}
    ladder_cell_ids: dict[tuple[str, str], str] = {}
    cell_triples: set[tuple[str, str, str]] = set()
    resub_triples: dict[tuple[str, str, str], str] = {}

    want_flag = str(plan["serving_mode_of_record"]["batch_invariant_used"])
    want_conc = str(plan["serving_mode_of_record"]["concurrency_used"])

    for tsv in sorted(waves_dir.glob("*.tsv")):
        counts["tsv_files"] += 1
        # RULING R3(ii). An enrichment-pass manifest is NOT a cell manifest: it is one
        # sampling-arm job per model per substrate, it is not part of the 216-cell grid,
        # and counting its rows as cells is how the grid total silently grows. Its rows
        # take every other structural check below, which is the point of naming it here
        # rather than skipping the file.
        is_enrich = tsv.name.startswith("enrich-")
        # A resubmission of voided cells is not a new cell: the grid already counted
        # them, and the rows are copied verbatim from the manifests that produced them
        # (a100-40-resub-01.tsv, the four voided by the fixed-port collision on xgph12,
        # DECISION-LOG.md 2026-09-07 17:41). Counting them as cells is how the 216-cell
        # grid reads 220. Named here rather than skipped so every check below still runs
        # on their rows, and cross-checked against the cell rows after the loop.
        # ELEMENT 11, the mechanism-challenge ladder. A ladder manifest is not part of
        # the 216-cell grid either: its rows are the 12 LoRA training jobs and the 12
        # evaluations of the checkpoints they write. They also cannot take the sweep row
        # checks: an evaluation row's MODEL is a SERVED NAME for a local checkpoint with
        # no Hub id and no Hub revision, and a training row generates nothing, so the
        # arm list and the curve cap do not apply to it. They get their own checks
        # below (check_ladder_rows) rather than being skipped.
        is_ladder = tsv.name.startswith("ladder-")
        is_resub = tsv.name.startswith("resub-") or "-resub-" in tsv.name
        # An exploratory manifest (explore-*.tsv) holds cells run to answer a question
        # before a ruling, never cells of record (for example the Phi-4 off-mode check
        # under R12(3), DECISION-LOG 2026-09-08). Its rows take every sweep row check and
        # are counted apart from the 216-cell grid; it names no pool, so the sweep-slot
        # comparison does not apply to it.
        is_explore = tsv.name.startswith("explore-")
        stem = tsv.stem[len("enrich-"):] if is_enrich else tsv.stem
        stem = stem.replace("resub-", "", 1) if is_resub else stem
        pool = stem.rsplit("-", 1)[0].replace("-single", "").replace("-tp2", "")
        cards = 0
        for n, model, substrate, cue, kv, bad in rows_of(tsv):
            if is_ladder:
                counts["ladder_rows"] += 1
                where = f"{tsv.name}:{n}"
                if bad:
                    problems.append(f"{where}: {bad}")
                    continue
                problems.extend(
                    check_ladder_row(where, model, substrate, cue, kv, roster, pw,
                                     ladder_cell_ids))
                continue
            if is_enrich:
                counts["enrich_rows"] += 1
            elif is_explore:
                counts["explore_rows"] += 1
            elif is_resub:
                counts["resub_rows"] += 1
                resub_triples.setdefault((model, substrate, cue), f"{tsv.name}:{n}")
            else:
                counts["rows"] += 1
                cell_triples.add((model, substrate, cue))
            where = f"{tsv.name}:{n}"
            if bad:
                problems.append(f"{where}: {bad}")
                continue
            if model not in roster:
                problems.append(f"{where}: {model} is not on the element 10 roster")
                continue
            counts["models"].add(model)
            if not is_explore:
                counts["pools"].add(pool)
            rev, want_pool, want_tp = roster[model]
            if kv.get("BCF_REVISION") != rev:
                problems.append(
                    f"{where}: BCF_REVISION={kv.get('BCF_REVISION')} but the roster pins "
                    f"{rev} for {model}")
            if kv.get("BCF_BATCH_INVARIANT") != want_flag:
                problems.append(
                    f"{where}: BCF_BATCH_INVARIANT={kv.get('BCF_BATCH_INVARIANT')}, and "
                    f"the plan's serving mode of record is {want_flag} (ruling R1)")
            if kv.get("BCF_CONCURRENCY") != want_conc:
                problems.append(
                    f"{where}: BCF_CONCURRENCY={kv.get('BCF_CONCURRENCY')}, and the "
                    f"plan's serving mode of record is {want_conc} in flight")
            for field in ("MEM", "CPUS", "BCF_N_ITEMS", "BCF_CURVE_CAP", "BCF_ARMS"):
                if not kv.get(field):
                    problems.append(f"{where}: {field} is missing")
            if kv.get("BCF_CURVE_CAP") and kv.get("BCF_N_ITEMS"):
                if int(kv["BCF_CURVE_CAP"]) < int(kv["BCF_N_ITEMS"]):
                    problems.append(
                        f"{where}: curve cap {kv['BCF_CURVE_CAP']} is below n_items "
                        f"{kv['BCF_N_ITEMS']}; element 15 requires at least the "
                        "clean-correct n")
            if substrate not in pw.SUBSTRATES:
                problems.append(f"{where}: substrate {substrate} is not one of the three")
            if cue not in pw.CUE_FAMILIES:
                problems.append(f"{where}: cue family {cue} is not one of the four")
            tp = int(kv.get("BCF_TP", 1))
            if tp != want_tp:
                problems.append(f"{where}: BCF_TP={tp} but the roster line says {want_tp}")
            per_node = CARDS_PER_NODE.get(want_pool)
            if per_node is not None and tp > per_node:
                problems.append(
                    f"{where}: tensor-parallel {tp} on {want_pool}, whose nodes carry "
                    f"{per_node} card(s); sbatch rejects that at submit time")
            cards += tp
        # A ladder manifest is submitted ONE ROW AT A TIME by bcf/ladder_wave.sh against
        # the CONTRACT ladder budget of 1 card, so its row count is not a card count and
        # the sweep-slot comparison does not apply to it. What does apply is the
        # CONTRACT checkpoint budget, checked here instead.
        if is_ladder or is_explore:
            continue
        slots = pw.POOL_SWEEP_SLOTS.get(pool)
        if slots is not None and cards > slots:
            problems.append(
                f"{tsv.name}: the wave wants {cards} card(s) and the CONTRACT sweep split "
                f"for {pool} is {slots}")
    # Excluding the resubmission rows from the cell count is only safe while every one
    # of them re-runs a cell that exists: a triple with no cell behind it is a NEW cell
    # wearing a resubmission's filename, counted by nothing.
    for triple, where in sorted(resub_triples.items()):
        if triple not in cell_triples:
            problems.append(
                f"{where}: resubmission row {triple} matches no cell manifest row; a "
                "resubmission re-runs a voided cell, it does not add one")
    if counts["rows"] != plan["n_cells"]:
        problems.append(
            f"the manifests hold {counts['rows']} rows and plan.json says "
            f"{plan['n_cells']} cells")
    # The enrichment pass is one job per model per substrate over the small-model pool.
    # Its own denominator, stated rather than inferred: eight a100-40 models times three
    # substrates. A missing row means a model or a substrate would never be enriched.
    enrich_files = sorted(waves_dir.glob("enrich-*.tsv"))
    if enrich_files:
        want_enrich = len({m for m in roster
                           if roster[m][1] == "a100-40"}) * len(pw.SUBSTRATES)
        if counts["enrich_rows"] != want_enrich:
            problems.append(
                f"the enrichment manifests hold {counts['enrich_rows']} rows and the "
                f"a100-40 roster times the three substrates is {want_enrich}")
    # The ladder's own denominator: 12 checkpoints per base (element 11(b), CONTRACT.md
    # line 24), evaluated once each, so 12 training rows and 12 evaluation rows.
    for kind, want in (
            ("ladder-train", CONTRACT_LADDER_CHECKPOINTS),
            ("ladder-eval", CONTRACT_LADDER_CHECKPOINTS)):
        files = sorted(waves_dir.glob(f"{kind}-*.tsv"))
        if not files:
            continue
        got = sum(1 for f in files for _r in rows_of(f))
        if got != want:
            problems.append(
                f"the {kind} manifests hold {got} row(s) and element 11(b) is {want} "
                f"checkpoints per base (3 doses x 2 seeds x (organism, twin))")
    counts["models"] = len(counts["models"])
    counts["pools"] = sorted(counts["pools"])
    return problems, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waves-dir", type=Path, default=WAVES)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    problems, counts = check(a.waves_dir)
    lines = [
        "bcf/check_wave_manifests.py: the structural half of wave.sh --check-only",
        (f"  {counts['tsv_files']} manifest file(s), {counts['rows']} cell row(s), "
         f"{counts['enrich_rows']} enrichment-pass row(s), "
         f"{counts['resub_rows']} resubmission row(s), "
         f"{counts['explore_rows']} exploratory row(s), "
         f"{counts['ladder_rows']} ladder row(s), "
         f"{counts['models']} model(s), pools {', '.join(counts['pools'])}"),
        "  NOT checked here (needs the cluster): the live per-user card caps read from",
        "  squeue, the 32-jobs-in-system limit, and sbatch --test-only on each command",
        "  line. Those are wave.sh --check-only's own job and land in dry-run-<wave>.txt.",
    ]
    if problems:
        lines.append(f"  {len(problems)} PROBLEM(S):")
        lines += [f"    - {p}" for p in problems]
    else:
        lines.append("  no structural problem found")
    report = "\n".join(lines)
    print(report)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(report + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
