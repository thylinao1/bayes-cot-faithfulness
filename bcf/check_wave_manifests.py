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


def check(waves_dir: Path = WAVES) -> tuple[list[str], dict]:
    pw = _load_plan_waves()
    plan = json.loads((waves_dir / "plan.json").read_text())
    roster = {hf: (rev, pool, tp) for hf, rev, _fam, pool, tp, *_ in pw.ROSTER}
    problems: list[str] = []
    counts = {"tsv_files": 0, "rows": 0, "models": set(), "pools": set()}

    want_flag = str(plan["serving_mode_of_record"]["batch_invariant_used"])
    want_conc = str(plan["serving_mode_of_record"]["concurrency_used"])

    for tsv in sorted(waves_dir.glob("*.tsv")):
        counts["tsv_files"] += 1
        pool = tsv.stem.rsplit("-", 1)[0].replace("-single", "").replace("-tp2", "")
        cards = 0
        for n, model, substrate, cue, kv, bad in rows_of(tsv):
            counts["rows"] += 1
            where = f"{tsv.name}:{n}"
            if bad:
                problems.append(f"{where}: {bad}")
                continue
            if model not in roster:
                problems.append(f"{where}: {model} is not on the element 10 roster")
                continue
            counts["models"].add(model)
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
        slots = pw.POOL_SWEEP_SLOTS.get(pool)
        if slots is not None and cards > slots:
            problems.append(
                f"{tsv.name}: the wave wants {cards} card(s) and the CONTRACT sweep split "
                f"for {pool} is {slots}")
    if counts["rows"] != plan["n_cells"]:
        problems.append(
            f"the manifests hold {counts['rows']} rows and plan.json says "
            f"{plan['n_cells']} cells")
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
        f"  {counts['tsv_files']} manifest file(s), {counts['rows']} row(s), "
        f"{counts['models']} model(s), pools {', '.join(counts['pools'])}",
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
