"""Generate the Phase-2 sweep wave manifests and the card-hour projection.

The grid is the canonical 18-model roster (PREREGISTRATION_jury_and_scale element 10)
crossed with 3 substrates and 4 cue families: 216 cells. This script writes one TSV per
wave under bcf/waves/, in a form bcf/wave.sh reads directly, plus a plan.json and a
plan.md carrying the pool assignment, the dependency order and the card-hours per pool.

Two things it will NOT do.

  - It never invents a rate. Per-cell hours come from the per-arm call counts and the
    per-call seconds of ONE measured run (bcf/measured/run-a-825511/throughput.json, job
    825511, exit 0, Qwen3-8B on a MIG 3g.40gb slice), divided by a measured concurrency
    speedup read from the Track G probe. A model class with no measurement of its own is
    marked UNMEASURED and carries the 8B cost scaled by nothing, which is a FLOOR and is
    labelled as one.
  - It never sizes a wave past a cap. The caps are the live per-user MaxTRESPU values
    (a100-40 8, a100-80 4, h100-96 2, h200-141 1, gpu total 12) and they count every
    campaign on the account, so a wave that fits here can still be refused at submit
    time by wave.sh reading the live queue. That refusal is the point.

    python bcf/plan_waves.py --probe results/trackg-probe/probe_results.json
    python bcf/plan_waves.py --sequential-only     # no probe yet: hours at concurrency 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAVES = REPO / "bcf" / "waves"
MEASURED = REPO / "bcf" / "measured" / "run-a-825511" / "throughput.json"

SUBSTRATES = ("arc_challenge", "aqua_rat", "logiqa2")
# DECISION-LOG 2026-09-07 ruling (b): 1,500 entered for the two high-follow families,
# 570 for the two low-follow ones, which publish per-model column A and no ordering.
CUE_FAMILIES = {
    "stated-hint": 1500,
    "professor": 1500,
    "metadata": 570,
    "grader-code": 570,
}
ARMS = ("replay", "placebo", "direct", "twostep", "filler", "curves", "transplant",
        "anchor", "specificity")

# Live per-user MaxTRESPU, re-verified 2026-09-07 (DECISION-LOG line 41). These count
# EVERY campaign on the account.
POOL_CAP = {"a100-40": 8, "a100-80": 4, "h100-96": 2, "h200-141": 1}
GPU_TOTAL_CAP = 12
# CONTRACT.md, corrected 2026-09-07: the two a100-80 judge servers hold 2 of that pool's
# 4 cards whenever they are up, so a sweep wave on a100-80 plans for 2.
POOL_SWEEP_SLOTS = {"a100-40": 8, "a100-80": 2, "h100-96": 2, "h200-141": 1}

# Host memory per class (DECISION-LOG 2026-09-07: the partition default is 3G and a vLLM
# engine core dies under it with no legible reason in the server log, job 825536).
MEM_BY_CLASS = {"small": "64G", "mid": "128G", "large": "192G"}
CPUS_BY_CLASS = {"small": "8", "mid": "8", "large": "16"}

# (hf_id, revision, family, pool, tp, quantization, extra vllm flags, size class)
# Revisions are element 10's pinned table, copied verbatim. Pools are CONTRACT.md's
# corrected serving lines by pool.
ROSTER = [
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "Qwen", "a100-40", 1, "", "", "small"),
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "6e8885a6ff5c1dc5201574c8fd700323f23c25fa", "Qwen", "a100-40", 1, "", "", "small"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "6a6f4aa4197940add57724a7707d069478df56b1", "Llama", "a100-40", 1, "", "", "small"),
    ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "Llama", "a100-40", 1, "", "", "small"),
    ("google/gemma-2-9b-it", "11c9b309abf73637e4b6f9a3fa1e92e615547819", "Gemma", "a100-40", 1, "", "", "small"),
    ("allenai/Olmo-3-7B-Think", "d97e442d7cc678210054dbcc9b440894d62c89a4", "OLMo", "a100-40", 1, "", "", "small"),
    ("microsoft/Phi-4-reasoning", "1de18ec97600877ce63dbf60c73b998da99f0195", "Phi", "a100-40", 1, "", "--max-num-seqs 64", "small"),
    ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", "gpt-oss", "a100-40", 1, "mxfp4", "", "small"),
    ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "Qwen", "a100-80", 1, "", "", "mid"),
    ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0", "Qwen", "a100-80", 1, "", "--max-num-seqs 64", "mid"),
    ("google/gemma-3-27b-it", "005ad3404e59d6023443cb575daa05336842228a", "Gemma", "a100-80", 1, "", "", "mid"),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "95a6d26c4bfb886c58daf9d3f7332c857cb27b43", "Mistral", "a100-80", 1, "", "", "mid"),
    ("mistralai/Magistral-Small-2509", "a31cc96ab10cf19bc42c628fedf1e359e0853c49", "Mistral", "a100-80", 1, "", "", "mid"),
    ("allenai/Olmo-3-32B-Think", "f2edda15216e738ef2bb73771e11890e152b2112", "OLMo", "a100-80", 1, "", "", "mid"),
    ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "Llama", "h100-96", 2, "", "", "large"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "b1c0b44b4369b597ad119a196caf79a9c40e141e", "Llama", "h100-96", 2, "", "", "large"),
    ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a", "gpt-oss", "h100-96", 1, "mxfp4", "", "large"),
    ("zai-org/GLM-4.5-Air", "a24ceef6ce4f3536971efe9b778bdaa1bab18daa", "GLM", "h200-141", 1, "fp8", "", "large"),
]

# Clean-correct retention. Two numbers, and which one is used is a stated choice:
#   RUN_A is what job 825511 measured for THIS model on THIS substrate (29 of 30).
#   BANKED_WORST is the 0.615 the sizing (01-SIZING I.3) banks across models.
# More clean-correct items means MORE arm calls, so RUN_A is the more expensive and the
# more conservative projection for a cost table.
RETENTION_RUN_A = 29 / 30
RETENTION_BANKED_WORST = 0.615

# Specificity runs on the FIXED A9 holdout, so its cost does not scale with n per cell.
SPECIFICITY_ARMS = ("specificity",)
SCALING_BASE_N = 30           # job 825511 entered 30 items
SCALING_BASE_C = 29           # and 29 were clean-correct


def load_measured() -> dict:
    """Per-call seconds for a full generation and for a forced continuation, measured.

    The split matters: a full generation is num_predict 320 and a forced continuation is
    24, and the skeleton's own arms are a mix, so one blended calls-per-second would
    misprice any cell whose arm mix differs from the skeleton's.
    """
    data = json.loads(MEASURED.read_text())
    arms = {a["arm"]: a for a in data["arms"]}
    full = sum(a["n_full_generations"] for a in data["arms"])
    forced = sum(a["n_calls"] - a["n_full_generations"] for a in data["arms"])
    # Arms whose calls are ALL forced continuations give the forced-call rate directly.
    forced_only = [a for a in data["arms"] if a["n_full_generations"] == 0]
    sec_forced = (sum(a["seconds"] for a in forced_only)
                  / sum(a["n_calls"] for a in forced_only))
    mixed = [a for a in data["arms"] if a["n_full_generations"] > 0]
    mixed_forced = sum(a["n_calls"] - a["n_full_generations"] for a in mixed)
    sec_full = ((sum(a["seconds"] for a in mixed) - mixed_forced * sec_forced)
                / sum(a["n_full_generations"] for a in mixed))
    spec = arms["specificity"]
    return {
        "source": str(MEASURED.relative_to(REPO)),
        "job": "825511",
        "total_calls": data["total_calls"],
        "total_seconds": data["total_seconds"],
        "n_full_generations": full,
        "n_forced_continuations": forced,
        "seconds_per_full_generation": sec_full,
        "seconds_per_forced_continuation": sec_forced,
        "specificity_calls_fixed": spec["n_calls"],
        "specificity_full_fixed": spec["n_full_generations"],
        "per_entered_full": None,   # filled below
        "per_clean_correct_full": None,
        "per_clean_correct_forced": None,
    }


def cost_model(m: dict) -> dict:
    """Calls per entered item and per clean-correct item, read off the measured run."""
    scale_full = m["n_full_generations"] - m["specificity_full_fixed"] - SCALING_BASE_N
    scale_forced = (m["n_forced_continuations"]
                    - (m["specificity_calls_fixed"] - m["specificity_full_fixed"]))
    m["per_entered_full"] = 1.0                       # the clean substrate pass
    m["per_clean_correct_full"] = scale_full / SCALING_BASE_C
    m["per_clean_correct_forced"] = scale_forced / SCALING_BASE_C
    return m


def cell_seconds(m: dict, n_entered: int, retention: float, speedup: float) -> dict:
    c = n_entered * retention
    n_full = n_entered * m["per_entered_full"] + c * m["per_clean_correct_full"] \
        + m["specificity_full_fixed"]
    n_forced = c * m["per_clean_correct_forced"] \
        + (m["specificity_calls_fixed"] - m["specificity_full_fixed"])
    seq = (n_full * m["seconds_per_full_generation"]
           + n_forced * m["seconds_per_forced_continuation"])
    return {
        "n_entered": n_entered,
        "n_clean_correct_assumed": round(c, 1),
        "n_full_generations": round(n_full),
        "n_forced_continuations": round(n_forced),
        "sequential_seconds": round(seq, 1),
        "sequential_hours": round(seq / 3600, 3),
        "hours": round(seq / 3600 / speedup, 3),
    }


def read_speedup(path: Path | None, want: int) -> dict:
    """The measured rate at the REQUESTED concurrency, with what it costs in determinism.

    The level is chosen by the caller and defaults to 1, because the probe found that
    batching changes the outputs and only the operator rules on whether that is
    acceptable for a pre-registered cell. This function reports; it does not pick.
    """
    if path is None:
        return {"speedup": 1.0, "concurrency": 1,
                "source": "NOT MEASURED; every hour below is a sequential-client hour"}
    data = json.loads(path.read_text())
    rows = {r["concurrency"]: r for r in data["rows"]}
    if want not in rows:
        raise SystemExit(f"REFUSING: concurrency {want} was not probed; "
                         f"levels measured: {sorted(rows)}")
    base = rows[1]["generations_per_second"]
    row = rows[want]
    alternatives = []
    for level in sorted(rows):
        r = rows[level]
        v = r["vs_concurrency_1"]
        alternatives.append({
            "concurrency": level,
            "generations_per_second": r["generations_per_second"],
            "speedup_vs_concurrency_1": round(r["generations_per_second"] / base, 3),
            "identical_completions": v["identical_completions"],
            "max_abs_letter_logprob_diff": v["max_abs_letter_logprob_diff"],
            "n_letter_logprobs_exactly_zero_diff":
                f"{v['n_exactly_zero_diff']}/{v['n_letter_logprobs_compared']}",
        })
    v = row["vs_concurrency_1"]
    return {
        "speedup": round(row["generations_per_second"] / base, 3),
        "concurrency": want,
        "baseline_generations_per_second": base,
        "generations_per_second": row["generations_per_second"],
        "identical_completions": v["identical_completions"],
        "max_abs_letter_logprob_diff": v["max_abs_letter_logprob_diff"],
        "n_letter_logprobs_compared": v["n_letter_logprobs_compared"],
        "n_letter_logprobs_exactly_zero_diff":
            f"{v['n_exactly_zero_diff']}/{v['n_letter_logprobs_compared']}",
        "probed_levels": alternatives,
        "no_ruling": ("The probe reports these numbers and makes no ruling on them. "
                      "Levels above 1 change the completions and the letter logprobs on "
                      "this card; whether a pre-registered cell may be run that way is "
                      "the operator's call, not this script's."),
        "source": str(path),
    }


def build_cells(m: dict, speedup: float, retention: float, concurrency: int) -> list[dict]:
    cells = []
    for hf_id, revision, family, pool, tp, quant, extra, size in ROSTER:
        for substrate in SUBSTRATES:
            for cue, n_items in CUE_FAMILIES.items():
                cost = cell_seconds(m, n_items, retention, speedup)
                cells.append({
                    "model": hf_id, "revision": revision, "family": family,
                    "substrate": substrate, "cue_family": cue,
                    "arms": list(ARMS), "n_items": n_items,
                    # Element 15: --curve-cap at least the run's clean-correct n. Setting
                    # it to n_items is always at least that, whatever the retention is.
                    "curve_cap": n_items,
                    "pool": pool, "gpu_type": pool, "tensor_parallel": tp,
                    "quantization": quant or None,
                    "vllm_extra_flags": extra or None,
                    "mem": MEM_BY_CLASS[size], "cpus": CPUS_BY_CLASS[size],
                    "size_class": size,
                    "concurrency": concurrency,
                    "cards": tp,
                    "measured_for_this_class": size == "small" and hf_id == "Qwen/Qwen3-8B",
                    **cost,
                })
    return cells


def group_waves(cells: list[dict]) -> list[dict]:
    """One wave per pool per round, sized to that pool's sweep slots in CARDS.

    Round-robin over models inside a pool, so a wave holds at most one cell per model:
    two cells of the same model in one wave would mean two servers loading the same
    weights onto two cards for no reason.

    h100-96 is special. Its cap is 2 CARDS and a 70B needs both (tensor-parallel 2), so a
    70B wave is one job and gpt-oss-120b (one card, mxfp4) never shares the pool with a
    70B. They are emitted as separate wave series and the dependency order serializes
    them.
    """
    waves: list[dict] = []
    by_pool: dict[str, list[dict]] = {}
    for cell in cells:
        key = cell["pool"]
        if key == "h100-96":
            key = "h100-96-tp2" if cell["tensor_parallel"] == 2 else "h100-96-single"
        by_pool.setdefault(key, []).append(cell)

    for pool_key, pool_cells in sorted(by_pool.items()):
        gpu_type = pool_cells[0]["pool"]
        slots = POOL_SWEEP_SLOTS[gpu_type]
        by_model: dict[str, list[dict]] = {}
        for cell in pool_cells:
            by_model.setdefault(cell["model"], []).append(cell)
        models = sorted(by_model)
        index = 0
        while any(by_model[mo] for mo in models):
            batch: list[dict] = []
            cards = 0
            for mo in models:
                if not by_model[mo]:
                    continue
                want = by_model[mo][0]["cards"]
                if cards + want > slots:
                    continue
                batch.append(by_model[mo].pop(0))
                cards += want
            if not batch:
                break
            index += 1
            waves.append({
                "wave_id": f"{pool_key}-{index:02d}",
                "gpu_type": gpu_type,
                "pool_key": pool_key,
                "cards": cards,
                "cap": POOL_CAP[gpu_type],
                "sweep_slots": slots,
                "cells": batch,
                "wall_hours": round(max(c["hours"] for c in batch), 3),
                "card_hours": round(sum(c["hours"] * c["cards"] for c in batch), 3),
            })
    return waves


def write_tsv(wave: dict, path: Path) -> None:
    lines = [
        f"# wave {wave['wave_id']}  pool {wave['gpu_type']}  "
        f"{wave['cards']} card(s) of the {wave['cap']} per-user cap "
        f"({wave['sweep_slots']} planned as sweep slots)",
        "# submit with:  bcf/wave.sh --type sweep --gpu-type "
        f"{wave['gpu_type']} bcf/waves/{wave['wave_id']}.tsv",
        f"# longest cell in this wave: {wave['wall_hours']} h; "
        f"card-hours: {wave['card_hours']}",
        "# columns: MODEL <tab> SUBSTRATE <tab> CUE <tab> KEY=VALUE ...",
    ]
    for c in wave["cells"]:
        lines.append(
            f"#   {c['model']} @ {c['revision']} | {c['substrate']} / {c['cue_family']} | "
            f"n {c['n_items']} | arms {','.join(c['arms'])} | "
            f"pool {c['pool']} tp {c['tensor_parallel']} "
            f"quant {c['quantization'] or 'none'} "
            f"flags {c['vllm_extra_flags'] or 'none'} | "
            f"{c['hours']} h at concurrency {c['concurrency']} "
            f"({c['sequential_hours']} h sequential)"
        )
    for c in wave["cells"]:
        fields = [
            c["model"], c["substrate"], c["cue_family"],
            f"BCF_REVISION={c['revision']}",
            f"BCF_N_ITEMS={c['n_items']}",
            f"BCF_CURVE_CAP={c['curve_cap']}",
            f"BCF_ARMS={' '.join(c['arms'])}",
            f"BCF_TP={c['tensor_parallel']}",
            f"BCF_CONCURRENCY={c['concurrency']}",
            f"MEM={c['mem']}",
            f"CPUS={c['cpus']}",
            f"BCF_EXPECTED_HOURS={c['hours']}",
        ]
        if c["quantization"]:
            fields.append(f"BCF_QUANT={c['quantization']}")
        if c["vllm_extra_flags"]:
            fields.append(f"BCF_EXTRA_VLLM={c['vllm_extra_flags']}")
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n")


DEPENDENCY_ORDER = [
    ("a100-40", "First. The pool is free tonight (0 of 8 in use at 03:00) and it carries "
                "the eight small subjects, so it needs nothing from any other lane."),
    ("a100-80", "Second, and only 2 cards at a time: the other 2 of the 4 belong to the "
                "judge servers W2 is building on branch jury/build. A wave here waits on "
                "the alta campaign draining, which held all 4 cards on 2026-09-07."),
    ("h100-96-single", "Third. gpt-oss-120b on ONE h100-96 card, mxfp4. Runs only when no "
                       "70B tensor-parallel job holds the pool."),
    ("h100-96-tp2", "Fourth, and BLOCKED until the TP=2 serving test (job 825253) reports. "
                    "That job is still PENDING with reason ReqNodeNotAvail and 0 of 22 "
                    "h100-96 cards free; the two 70B rows have no demonstrated serving "
                    "line until it runs, and element 16's roster step is the fallback."),
    ("h200-141", "Last, cap 1 card, serialized against the FP8 Llama-3.3-70B judge. GLM-4.5-Air "
                 "runs only in a window when that judge is down."),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, default=None,
                    help="probe_results.json from bcf/concurrency_probe.py")
    ap.add_argument("--sequential-only", action="store_true",
                    help="no probe available: price every cell at concurrency 1 and say so")
    ap.add_argument("--retention", choices=["run_a", "banked_worst"], default="run_a")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight per cell. Default 1: the only probed level "
                         "whose completions and letter logprobs match the sequential "
                         "client exactly. Any other value must be a probed level and "
                         "carries that level's measured divergence into plan.json.")
    ap.add_argument("--out-dir", type=Path, default=WAVES)
    a = ap.parse_args(argv)
    if not a.probe and not a.sequential_only:
        print("REFUSING: pass --probe with a measured probe_results.json, or "
              "--sequential-only to price everything at concurrency 1 and label it.")
        return 2

    speed = read_speedup(a.probe, a.concurrency)
    retention = RETENTION_RUN_A if a.retention == "run_a" else RETENTION_BANKED_WORST
    m = cost_model(load_measured())
    cells = build_cells(m, speed["speedup"], retention, speed["concurrency"])
    waves = group_waves(cells)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for old in a.out_dir.glob("*.tsv"):
        old.unlink()
    for wave in waves:
        write_tsv(wave, a.out_dir / f"{wave['wave_id']}.tsv")

    by_pool: dict[str, dict] = {}
    for cell in cells:
        row = by_pool.setdefault(cell["pool"], {"cells": 0, "card_hours": 0.0,
                                                "sequential_card_hours": 0.0})
        row["cells"] += 1
        row["card_hours"] += cell["hours"] * cell["cards"]
        row["sequential_card_hours"] += cell["sequential_hours"] * cell["cards"]
    for row in by_pool.values():
        row["card_hours"] = round(row["card_hours"], 1)
        row["sequential_card_hours"] = round(row["sequential_card_hours"], 1)

    plan = {
        "n_models": len(ROSTER), "n_substrates": len(SUBSTRATES),
        "n_cue_families": len(CUE_FAMILIES), "n_cells": len(cells),
        "n_waves": len(waves),
        "n_items_per_cell": CUE_FAMILIES,
        "arms": list(ARMS),
        "clean_correct_retention": {"choice": a.retention, "value": retention},
        "measured_cost_model": m,
        "concurrency": speed,
        "caps_counting_all_campaigns": POOL_CAP,
        "gpu_total_cap": GPU_TOTAL_CAP,
        "sweep_slots_planned": POOL_SWEEP_SLOTS,
        "card_hours_by_pool": by_pool,
        "cost_basis": (
            "Every per-cell hour is the 8B cost measured on ONE MIG 3g.40gb slice by job "
            "825511 (Qwen/Qwen3-8B, arc_challenge, stated-hint, exit 0), scaled by item "
            "count and divided by the probed concurrency speedup. The 24B-to-35B, 70B and "
            "120B classes have NO throughput measurement of their own, so their rows are a "
            "FLOOR, not a forecast, and are marked measured_for_this_class false. The "
            "whole-a100-80 rerun is job 825510, still PENDING with reason Resources, so the "
            "a100-80-versus-MIG ratio is unmeasured too."),
        "total_card_hours": round(sum(r["card_hours"] for r in by_pool.values()), 1),
        "total_sequential_card_hours": round(
            sum(r["sequential_card_hours"] for r in by_pool.values()), 1),
        "dependency_order": [{"pool_key": k, "why": w} for k, w in DEPENDENCY_ORDER],
        "waves": [{k: v for k, v in w.items() if k != "cells"} |
                  {"n_cells": len(w["cells"]),
                   "models": sorted({c["model"] for c in w["cells"]})}
                  for w in waves],
    }
    (a.out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"{len(cells)} cells, {len(waves)} waves, "
          f"{plan['total_card_hours']} card-hours at concurrency {speed['concurrency']} "
          f"({plan['total_sequential_card_hours']} sequential)")
    for pool, row in sorted(by_pool.items()):
        print(f"  {pool:<10} {row['cells']:>4} cells  {row['card_hours']:>9.1f} card-h  "
              f"({row['sequential_card_hours']:.1f} sequential)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
