"""Generate the Phase-2 sweep wave manifests and the card-hour projection.

The grid is the canonical 18-model roster (PREREGISTRATION_jury_and_scale element 10)
crossed with 3 substrates and 4 cue families: 216 cells. This script writes one TSV per
wave under bcf/waves/, in a form bcf/wave.sh reads directly, plus a plan.json and a
plan.md carrying the pool assignment, the dependency order and the card-hours per pool.

RULING R2 (2026-09-07) sets what these hours are priced at. The basis is no longer a
sequential-client rate divided by a speedup: it is job 826025's own per-arm seconds,
measured on one MIG 3g.40gb slice at 32 requests in flight with VLLM_BATCH_INVARIANT=1,
which is the serving mode ruling R1 pins for every powered cell. The generation rate that
mode reaches is 2.92 generations per second per slice (job 826020), and the a100-80 pool
is priced at the measured whole-card to MIG ratio of 1.7371 as a FLOOR.

Two things it will NOT do.

  - It never invents a rate. Per-cell hours come from the per-arm call counts and the
    per-call seconds of ONE measured run in the pinned serving mode (job 826025, exit 0,
    Qwen3-8B on a MIG 3g.40gb slice, concurrency 32, flag on), restricted to the same
    eleven intervals the powered cells run. A pool or a model class with no measurement
    of its own carries the 8B cost divided by a ratio that is a LOWER BOUND on the true
    cost, and every such row is marked measured_for_this_class false.
  - It never sizes a wave past a cap. The caps are the live per-user MaxTRESPU values
    (a100-40 8, a100-80 4, h100-96 2, h200-141 1, gpu total 12) and they count every
    campaign on the account, so a wave that fits here can still be refused at submit
    time by wave.sh reading the live queue. That refusal is the point.

    python bcf/plan_waves.py                 # R2 defaults: flag on, 32 in flight
    python bcf/plan_waves.py --sequential-only   # the pre-R2 comparison, concurrency 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAVES = REPO / "bcf" / "waves"
# The cost basis of record after ruling R2: job 826025, the eleven-arm skeleton run at
# concurrency 32 with VLLM_BATCH_INVARIANT=1 on one MIG 3g.40gb slice, mirrored into the
# repository. The sequential file stays beside it as the comparison column, because the
# ladder trigger of element 16 is stated against a budget that was priced sequentially.
MEASURED_FLAG_ON = (REPO / "experiments" / "results" / "w3b-skeleton" / "qwen3-8b"
                    / "arc_challenge" / "stated-hint" / "throughput.json")
MEASURED_SEQ = REPO / "bcf" / "measured" / "run-a-825511" / "throughput.json"
MEASURED = MEASURED_SEQ  # kept: the sequential comparison still reads it by name
# The flag-on determinism probe, quoted into plan.json so every wave carries the numbers
# that justify serving at 32 in flight rather than a claim that it is safe.
PROBE_FLAG_ON = (REPO / "bcf" / "measured" / "w3b-probe-bi-826020" / "probe_results.json")

# RULING R1 serving constants, carried into every row of every wave manifest.
SERVING = {
    "batch_invariant": 1,
    "concurrency": 32,
    "vllm_version": "0.28.0",
    "attention_backend": "FLASH_ATTN",
    "vllm_use_flashinfer_sampler": 0,
    "generations_per_second_per_slice": 2.9196,
    "source": ("ruling R1 / A3.6: job 826020 measured 30/30 identical completions and a "
               "max absolute letter-logprob difference of 0.0 at 1, 8, 32 and 64 in "
               "flight with the flag on, against 13/30 and 0.875 nats at 32 with it off "
               "(job 825548). Every cell re-proves it on its own server before its arms."),
}

# Card-hour ratio per pool, applied to the measured MIG-slice cost, with whether it is a
# MEASUREMENT for that pool or a floor. A ratio above 1 makes a cell cheaper, so a ratio
# that is not measured for the class makes the row a LOWER BOUND on the cost.
POOL_RATIO = {
    "a100-40": (1.0, True,
                "MEASURED on this pool and this card type: job 826025 ran the arms on "
                "one MIG 3g.40gb slice at concurrency 32 with the flag on, and the cost "
                "model below is that run's own per-call seconds."),
    "a100-80": (1.7371, False,
                "FLOOR. 1.7371 is the whole-a100-80 to MIG ratio measured at ONE request "
                "in flight on an 8B model (job 826028, 1,469 calls in 369.0 s, against "
                "run B's identical work in 641.0 s). Applying it to a concurrency-32 "
                "flag-on MIG cost assumes the ratio survives under load, which no run "
                "has measured, and the models on this pool are 3 to 4 times the size of "
                "the 8B the cost model is built on. The hours below are a lower bound."),
    "h100-96": (1.0, False,
                "FLOOR. No throughput measurement exists for the 70B dense tensor-"
                "parallel-2 line or for gpt-oss-120b mxfp4; the TP=2 serving test (job "
                "825253) is still PENDING with ReqNodeNotAvail. These rows carry the 8B "
                "MIG cost with no credit for the larger card, which is a lower bound."),
    "h200-141": (1.0, False,
                "FLOOR. No throughput measurement exists for GLM-4.5-Air FP8 on the "
                "h200-141. Same treatment and the same caveat as h100-96."),
}

# Models inside the class the cost model was measured on: up to 14B, bf16, one card.
# gpt-oss-20b sits in the same POOL but is a 20B mxfp4 MoE, so it is a floor row even
# though its pool is the measured one.
MEASURED_CLASS_MODELS = frozenset({
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "allenai/Olmo-3-7B-Think",
    "microsoft/Phi-4-reasoning",
})

# CONTRACT.md's compute budget of record, quoted verbatim so the element 16 trigger
# comparison names what it compares against: "about 650 card-hours for 18 models"
# (CONTRACT.md, "Compute budget of record (first-order; replaced by Phase 1
# measurements)"). Element 16 triggers when MEASURED throughput falls FAR BELOW the
# budget of record, which is a comparison in the direction of more card-hours than the
# budget assumed, not fewer.
BUDGET_OF_RECORD_CARD_HOURS = 650
BUDGET_OF_RECORD_SOURCE = (
    "CONTRACT.md, line beginning 'Compute budget of record (first-order; replaced by "
    "Phase 1 measurements)': about 650 card-hours for 18 models, at n=500 entered and "
    "12 cells per model."
)

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


# The intervals a powered cell actually runs: the two base passes plus the nine A2 arms.
# A throughput file that also carries `sampling` and `repeat-curves` (job 826025 does) is
# restricted to these before any per-call second is derived, so the cost model prices the
# same work in both bases and the two are comparable.
CELL_INTERVALS = ("clean_substrate", "cue_pass") + ARMS


def load_measured(path: Path = MEASURED_FLAG_ON, job: str = "826025") -> dict:
    """Per-call seconds for a full generation and for a forced continuation, measured.

    The split matters: a full generation is num_predict 320 and a forced continuation is
    24, and the skeleton's own arms are a mix, so one blended calls-per-second would
    misprice any cell whose arm mix differs from the skeleton's.

    Only the intervals in CELL_INTERVALS are read. Job 826025's file carries two more
    arms (the element 9.2 sampling arm and the element 8.3 repeated curves), and folding
    their 1,710 calls into a per-item cost would price cells for arms they do not run.
    Those two are costed separately in the plan, by name.
    """
    data = json.loads(path.read_text())
    intervals = [a for a in data["arms"] if a["arm"] in CELL_INTERVALS]
    missing = [a for a in CELL_INTERVALS if a not in {x["arm"] for x in intervals}]
    if missing:
        raise SystemExit(
            f"REFUSING: {path} has no interval for {missing}; a cost model built on a "
            "file that is missing an arm would price that arm at zero."
        )
    arms = {a["arm"]: a for a in intervals}
    full = sum(a["n_full_generations"] for a in intervals)
    forced = sum(a["n_calls"] - a["n_full_generations"] for a in intervals)
    # Arms whose calls are ALL forced continuations give the forced-call rate directly.
    forced_only = [a for a in intervals if a["n_full_generations"] == 0]
    sec_forced = (sum(a["seconds"] for a in forced_only)
                  / sum(a["n_calls"] for a in forced_only))
    mixed = [a for a in intervals if a["n_full_generations"] > 0]
    mixed_forced = sum(a["n_calls"] - a["n_full_generations"] for a in mixed)
    sec_full = ((sum(a["seconds"] for a in mixed) - mixed_forced * sec_forced)
                / sum(a["n_full_generations"] for a in mixed))
    spec = arms["specificity"]
    return {
        "source": str(path.relative_to(REPO)),
        "job": job,
        "intervals_used": list(CELL_INTERVALS),
        "total_calls": sum(a["n_calls"] for a in intervals),
        "total_seconds": sum(a["seconds"] for a in intervals),
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


def cell_calls(m: dict, n_entered: int, retention: float) -> tuple[float, float, float]:
    """(clean-correct assumed, full generations, forced continuations) for one cell."""
    c = n_entered * retention
    n_full = n_entered * m["per_entered_full"] + c * m["per_clean_correct_full"] \
        + m["specificity_full_fixed"]
    n_forced = c * m["per_clean_correct_forced"] \
        + (m["specificity_calls_fixed"] - m["specificity_full_fixed"])
    return c, n_full, n_forced


def cell_seconds(m: dict, n_entered: int, retention: float, ratio: float,
                 m_seq: dict | None = None) -> dict:
    """Cell cost under the pinned serving mode, with the sequential column beside it.

    ``ratio`` is the pool's card-to-slice ratio from POOL_RATIO, a divisor on the
    measured MIG-slice seconds. It is 1.0 for the pool the cost model was measured on
    and a floor everywhere else, which is why the row carries whether it was measured.

    The sequential column is not decoration. Element 16's ladder trigger is stated
    against a budget of record that was priced at a sequential-client rate, so the
    comparison the ruling asks for needs both numbers side by side.
    """
    c, n_full, n_forced = cell_calls(m, n_entered, retention)
    secs = (n_full * m["seconds_per_full_generation"]
            + n_forced * m["seconds_per_forced_continuation"]) / ratio
    out = {
        "n_entered": n_entered,
        "n_clean_correct_assumed": round(c, 1),
        "n_full_generations": round(n_full),
        "n_forced_continuations": round(n_forced),
        "seconds": round(secs, 1),
        "hours": round(secs / 3600, 3),
        "pool_ratio": ratio,
    }
    if m_seq is not None:
        c2, f2, k2 = cell_calls(m_seq, n_entered, retention)
        seq = (f2 * m_seq["seconds_per_full_generation"]
               + k2 * m_seq["seconds_per_forced_continuation"])
        out["sequential_seconds"] = round(seq, 1)
        out["sequential_hours"] = round(seq / 3600, 3)
        out["speedup_vs_sequential_client"] = round(seq / secs, 3) if secs else None
    return out


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


def build_cells(m: dict, retention: float, concurrency: int, batch_invariant: int,
                m_seq: dict | None = None) -> list[dict]:
    """One row per cell, carrying its cost AND the serving constants it must run under.

    The serving constants are on every row rather than in a header, because a row is what
    wave.sh turns into an sbatch --export line. A cell whose manifest does not carry the
    flag and the concurrency would be submitted at whatever the submission script's
    defaults happen to be that week, and ruling R1 is exactly about that not being the
    thing a powered measurement rests on.
    """
    cells = []
    for hf_id, revision, family, pool, tp, quant, extra, size in ROSTER:
        ratio, ratio_measured, ratio_why = POOL_RATIO[pool]
        for substrate in SUBSTRATES:
            for cue, n_items in CUE_FAMILIES.items():
                cost = cell_seconds(m, n_items, retention, ratio, m_seq)
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
                    "batch_invariant": batch_invariant,
                    "cards": tp,
                    # Measured for THIS class means both halves: the pool's ratio was
                    # measured on this card type AND the model is inside the size class
                    # the per-call seconds were measured on.
                    "measured_for_this_class": bool(
                        ratio_measured and hf_id in MEASURED_CLASS_MODELS
                    ),
                    "pool_ratio_measured": ratio_measured,
                    "pool_ratio_basis": ratio_why,
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
        "# serving mode of record (ruling R1, A3.6): VLLM_BATCH_INVARIANT="
        f"{SERVING['batch_invariant']}, {SERVING['concurrency']} requests in flight, "
        f"vLLM {SERVING['vllm_version']}, {SERVING['attention_backend']} backend, "
        f"VLLM_USE_FLASHINFER_SAMPLER={SERVING['vllm_use_flashinfer_sampler']}, the "
        "revision pinned per row. Each cell re-proves the determinism preflight on its",
        "# own server (30 items at 1 and 32 in flight, 30/30 identical and 0.0 max "
        "letter-logprob difference) and refuses with exit 10 otherwise.",
        "# columns: MODEL <tab> SUBSTRATE <tab> CUE <tab> KEY=VALUE ...",
    ]
    for c in wave["cells"]:
        measured = ("measured for this class" if c["measured_for_this_class"]
                    else "FLOOR, not measured for this class")
        lines.append(
            f"#   {c['model']} @ {c['revision']} | {c['substrate']} / {c['cue_family']} | "
            f"n {c['n_items']} | arms {','.join(c['arms'])} | "
            f"pool {c['pool']} tp {c['tensor_parallel']} "
            f"quant {c['quantization'] or 'none'} "
            f"flags {c['vllm_extra_flags'] or 'none'} | "
            f"{c['hours']} h at concurrency {c['concurrency']} with the flag "
            f"{'on' if c['batch_invariant'] else 'off'} "
            f"({c.get('sequential_hours')} h sequential client) | {measured}"
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
            f"BCF_BATCH_INVARIANT={c['batch_invariant']}",
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


def enrichment_pass(m: dict, pool_sizes: dict, per_item_seconds: float) -> dict:
    """RULING R3(ii): the sampling arm on the FULL item pool, once per model x substrate.

    This is NOT part of a cell row and is not folded into one. It is one request per
    item on the clean prompt, run before the hinted arms so every right-but-uncertain
    item the pool holds can be enriched into that cell. Job 826025 measured it at 30
    calls in 64.0 s at 32 in flight with the flag on, which is 2.1333 s per item-request
    on an 8B model; each request returns k = 32 completions, so the wall clock is the
    32 generations and not the HTTP call.

    Priced here as a named line so the budget shows it. It is a floor for every class
    above 8B for the same reason the cell rows are.
    """
    per_model = sum(pool_sizes[sub] for sub in SUBSTRATES) * per_item_seconds
    total = per_model * len(ROSTER)
    return {
        "what": ("ruling R3(ii): one k = 32 sampling request per item over the whole "
                 "pool of each substrate, once per model, before that substrate's "
                 "hinted arms"),
        "per_item_seconds": round(per_item_seconds, 4),
        "per_item_seconds_source": ("job 826025 throughput.json, arm `sampling`: 30 "
                                    "calls in 64.0 s at concurrency 32 with the flag on"),
        "pool_sizes": dict(pool_sizes),
        "hours_per_model": round(per_model / 3600, 3),
        "card_hours_total": round(total / 3600, 1),
        "floor": True,
    }


def trigger_comparison(total_card_hours: float, total_sequential: float,
                       enrichment_card_hours: float) -> dict:
    """Element 16's ladder trigger, computed rather than asserted.

    The trigger fires when MEASURED throughput falls FAR BELOW the budget of record.
    Card-hours move the other way from throughput, so the comparison is: does the grid,
    priced at the measured rates, cost MORE card-hours than the budget of record
    assumed? It is stated with both totals and the budget's own source.
    """
    priced = total_card_hours + enrichment_card_hours
    return {
        "budget_of_record_card_hours": BUDGET_OF_RECORD_CARD_HOURS,
        "budget_of_record_source": BUDGET_OF_RECORD_SOURCE,
        "priced_card_hours_sweep_cells": round(total_card_hours, 1),
        "priced_card_hours_enrichment_pass": round(enrichment_card_hours, 1),
        "priced_card_hours_total": round(priced, 1),
        "priced_card_hours_at_the_sequential_client_rate": round(total_sequential, 1),
        "ratio_priced_over_budget": round(priced / BUDGET_OF_RECORD_CARD_HOURS, 3),
        "triggered": priced > BUDGET_OF_RECORD_CARD_HOURS,
        "reading": (
            "NOT TRIGGERED" if priced <= BUDGET_OF_RECORD_CARD_HOURS else
            "TRIGGERED: the grid prices ABOVE the budget of record"
        ),
        "caveat": (
            "The budget of record is a first-order estimate at n=500 entered and 12 "
            "cells per model, and this grid is 12 cells per model at 1,500 entered for "
            "two cue families and 570 for two (ruling (b)), so the two are not the same "
            "grid. They are compared because element 16 names the budget of record as "
            "the trigger's reference and nothing has replaced it. Every non-8B row here "
            "is a FLOOR, so the priced total can only rise as the missing serving tests "
            "report."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, default=PROBE_FLAG_ON,
                    help="probe_results.json from bcf/concurrency_probe.py; the default "
                         "is the committed flag-on probe of job 826020, whose numbers "
                         "are quoted into plan.json.")
    ap.add_argument("--sequential-only", action="store_true",
                    help="price every cell at the pre-R2 sequential-client rate of job "
                         "825511 and label it. The comparison column, not the plan.")
    ap.add_argument("--retention", choices=["run_a", "banked_worst"], default="run_a")
    ap.add_argument("--concurrency", type=int, default=SERVING["concurrency"],
                    help="requests in flight per cell. Default 32: ruling R1 pins the "
                         "batch-invariant flag on, and job 826020 measured 30/30 "
                         "identical completions and a 0.0 max letter-logprob difference "
                         "at that level under it. Any other value must be a probed level.")
    ap.add_argument("--batch-invariant", type=int, choices=[0, 1],
                    default=SERVING["batch_invariant"],
                    help="the serving flag written into every manifest row. 0 is an "
                         "EXPLORATORY plan: A3.6 pins it on for every powered cell.")
    ap.add_argument("--out-dir", type=Path, default=WAVES)
    a = ap.parse_args(argv)

    if a.batch_invariant != 1 or a.concurrency != SERVING["concurrency"]:
        print(f"[plan] NOTE: batch_invariant={a.batch_invariant} concurrency="
              f"{a.concurrency} is NOT the A3.6 pinned serving mode "
              f"(flag 1, {SERVING['concurrency']} in flight). This plan is exploratory.")

    retention = RETENTION_RUN_A if a.retention == "run_a" else RETENTION_BANKED_WORST
    m_seq = cost_model(load_measured(MEASURED_SEQ, "825511"))
    if a.sequential_only:
        m = cost_model(load_measured(MEASURED_SEQ, "825511"))
        basis = "sequential_client_825511"
    else:
        m = cost_model(load_measured(MEASURED_FLAG_ON, "826025"))
        basis = "flag_on_concurrency_32_826025"
    probe = read_speedup(a.probe, a.concurrency) if a.probe and a.probe.exists() else None

    cells = build_cells(m, retention, a.concurrency, a.batch_invariant, m_seq)
    waves = group_waves(cells)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for old in a.out_dir.glob("*.tsv"):
        old.unlink()
    for wave in waves:
        write_tsv(wave, a.out_dir / f"{wave['wave_id']}.tsv")

    by_pool: dict[str, dict] = {}
    for cell in cells:
        row = by_pool.setdefault(cell["pool"], {
            "cells": 0, "card_hours": 0.0, "sequential_card_hours": 0.0,
            "pool_ratio": cell["pool_ratio"],
            "pool_ratio_measured": cell["pool_ratio_measured"],
            "pool_ratio_basis": cell["pool_ratio_basis"],
            "n_cells_measured_for_their_class": 0,
        })
        row["cells"] += 1
        row["card_hours"] += cell["hours"] * cell["cards"]
        row["sequential_card_hours"] += cell.get("sequential_hours", 0.0) * cell["cards"]
        row["n_cells_measured_for_their_class"] += int(cell["measured_for_this_class"])
    for row in by_pool.values():
        row["card_hours"] = round(row["card_hours"], 1)
        row["sequential_card_hours"] = round(row["sequential_card_hours"], 1)

    total_card_hours = round(sum(r["card_hours"] for r in by_pool.values()), 1)
    total_sequential = round(sum(r["sequential_card_hours"] for r in by_pool.values()), 1)
    # Pool sizes for the enrichment pass: the entered n of the largest cue family is the
    # pool a substrate must hold (ruling (b) raised two families to 1,500).
    pool_sizes = {sub: max(CUE_FAMILIES.values()) for sub in SUBSTRATES}
    enrichment = enrichment_pass(m, pool_sizes, 64.0 / 30.0)
    trigger = trigger_comparison(total_card_hours, total_sequential,
                                 enrichment["card_hours_total"])

    plan = {
        "n_models": len(ROSTER), "n_substrates": len(SUBSTRATES),
        "n_cue_families": len(CUE_FAMILIES), "n_cells": len(cells),
        "n_waves": len(waves),
        "n_items_per_cell": CUE_FAMILIES,
        "arms": list(ARMS),
        "serving_mode_of_record": SERVING | {
            "concurrency_used": a.concurrency,
            "batch_invariant_used": a.batch_invariant,
        },
        "determinism_probe": probe,
        "clean_correct_retention": {"choice": a.retention, "value": retention},
        "cost_basis_id": basis,
        "measured_cost_model": m,
        "measured_cost_model_sequential_comparison": m_seq,
        "pool_ratio": {k: {"ratio": v[0], "measured_for_this_pool": v[1], "why": v[2]}
                       for k, v in POOL_RATIO.items()},
        "measured_class_models": sorted(MEASURED_CLASS_MODELS),
        "caps_counting_all_campaigns": POOL_CAP,
        "gpu_total_cap": GPU_TOTAL_CAP,
        "sweep_slots_planned": POOL_SWEEP_SLOTS,
        "card_hours_by_pool": by_pool,
        "cost_basis": (
            "RULING R2. Every per-cell hour is the 8B cost measured on ONE MIG 3g.40gb "
            "slice by job 826025 (Qwen/Qwen3-8B, arc_challenge, stated-hint, exit 0) at "
            "32 requests in flight with VLLM_BATCH_INVARIANT=1, restricted to the eleven "
            "intervals a powered cell runs, and scaled by item count. The a100-80 pool "
            "divides that by the measured whole-card to MIG ratio of 1.7371, which is a "
            "sequential-client ratio on an 8B model and therefore a FLOOR. The 24B-to-"
            "35B, 70B and 120B classes have no throughput measurement of their own, so "
            "their rows are a FLOOR, not a forecast, and carry measured_for_this_class "
            "false. gpt-oss-20b sits on the measured pool but is a 20B mxfp4 MoE, so it "
            "is a floor row too."),
        "not_priced_in_these_rows": [
            {"what": "the U6 on-policy resampling arm of section 9.3",
             "why": "it has never run, so there is no measured per-call cost for it, and "
                    "no number is invented here"},
            {"what": "the chain-level repeat arm of ruling R4",
             "why": "it is a Phase 1 skeleton done-when, not an arm of every powered "
                    "cell; its cost on a 28-item cell is measured only by the skeleton "
                    "run that carries it"},
            {"what": "the element 8.3 repeated-curve arm",
             "why": "measured at 1,680 forced continuations in 37.0 s on a 28-item cell "
                    "(job 826025) but not part of the nine A2 arms these cells run"},
            {"what": "judging",
             "why": "CONTRACT.md carries the judging budget separately, in server-hours"},
        ],
        "enrichment_pass": enrichment,
        "ladder_trigger_element_16": trigger,
        "total_card_hours": total_card_hours,
        "total_sequential_card_hours": total_sequential,
        "dependency_order": [{"pool_key": k, "why": w} for k, w in DEPENDENCY_ORDER],
        "waves": [{k: v for k, v in w.items() if k != "cells"} |
                  {"n_cells": len(w["cells"]),
                   "models": sorted({c["model"] for c in w["cells"]})}
                  for w in waves],
    }
    (a.out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"{len(cells)} cells, {len(waves)} waves, basis {basis}")
    print(f"serving mode: VLLM_BATCH_INVARIANT={a.batch_invariant}, "
          f"{a.concurrency} in flight, vLLM {SERVING['vllm_version']}, "
          f"{SERVING['attention_backend']}")
    print(f"{total_card_hours} card-hours for the sweep cells "
          f"({total_sequential} at the sequential-client rate)")
    for pool, row in sorted(by_pool.items()):
        flag = "measured" if row["pool_ratio_measured"] else "FLOOR"
        print(f"  {pool:<10} {row['cells']:>4} cells  {row['card_hours']:>9.1f} card-h  "
              f"(ratio {row['pool_ratio']}, {flag}; "
              f"{row['n_cells_measured_for_their_class']}/{row['cells']} cells measured "
              f"for their class)")
    print(f"  enrichment pass (R3(ii)): {enrichment['card_hours_total']} card-h over "
          f"{len(ROSTER)} models x {len(SUBSTRATES)} substrates x "
          f"{max(CUE_FAMILIES.values())} pool items")
    print(f"ELEMENT 16 TRIGGER: priced {trigger['priced_card_hours_total']} card-h "
          f"against the budget of record {trigger['budget_of_record_card_hours']} card-h "
          f"({BUDGET_OF_RECORD_SOURCE.split(':')[0]}); ratio "
          f"{trigger['ratio_priced_over_budget']} -> {trigger['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
