#!/usr/bin/env python
"""Generate the enrichment-pass wave files (ruling R3(ii)) for the small-model pool.

One job per model per substrate: the element 9.2 sampling arm ALONE, k = 32 at
temperature 0.7 on the clean prompt, over the whole 1,500-item pool, on one a100-40 slice
under the pinned serving mode. Eight small models times three substrates is 24 jobs, and
the a100-40 per-user cap is 8 cards, so they land as three waves of eight, one substrate
per wave.

Two things this generator refuses to retype, because retyping either is how a manifest
starts disagreeing with the campaign:

  * the ROSTER and the PINNED REVISIONS come out of bcf/waves/a100-40-01.tsv, the sweep
    wave that is already submitted. A model whose revision differs between the sweep and
    the enrichment pass would enrich a cell from a different model.
  * the PRICE comes from bcf/waves/plan.json's own enrichment_pass block, which records
    2.1333 s per item and its source (job 826025 throughput.json, arm `sampling`: 30
    calls in 64.0 s at concurrency 32 with the flag on).

What the price does NOT cover is stated in each wave file rather than hidden: the runner
always makes a clean substrate pass and a cue pass before any arm, so a pass costs the
sampling line PLUS one full generation per pool item PLUS one per clean-correct item, at
the 0.41005 s per full generation the same job measured. Both figures are written into
the file and BCF_EXPECTED_HOURS carries the total, because that is what the wall clock
has to cover.

  python bcf/plan_enrich_waves.py            # write bcf/waves/enrich-a100-40-*.tsv
  python bcf/plan_enrich_waves.py --check    # exit 1 if what is on disk differs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WAVES = HERE / "waves"
SOURCE_WAVE = WAVES / "a100-40-01.tsv"
PLAN = WAVES / "plan.json"

SUBSTRATES = ("arc_challenge", "aqua_rat", "logiqa2")
POOL_SIZE = 1500
POOL = "a100-40"
# The frozen cue family. An enrichment PASS does not use a cue at all (the sampling arm
# reads the CLEAN prompt), but the runner always makes a cue pass and the results path
# always carries a cue family, so the pass runs under the frozen stated hint rather than
# a taxonomy cue, and writes under its own results subroot so it cannot collide with the
# stated-hint sweep cell of the same model and substrate.
CUE = "stated-hint"
OUT_SUBROOT = "enrichment"
# Measured on the same run as the sampling rate, for the two passes the runner always
# makes: experiments/results/w3b-skeleton/.../throughput.json, seconds_per_full_generation.
SECONDS_PER_FULL_GENERATION = 0.41004685959401904
# Measured clean-correct retention on the skeleton cell (28 of 30), used only to size the
# cue pass. It moves the estimate, never the run.
CLEAN_CORRECT_RETENTION = 28 / 30


def roster(path: Path = SOURCE_WAVE) -> list[dict]:
    """The eight small models and their pinned fields, read out of the sweep wave file."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        model = fields[0]
        kv = {}
        for f in fields[3:]:
            if "=" in f:
                k, v = f.split("=", 1)
                kv[k] = v
        rows.append({"model": model, "kv": kv})
    return rows


def per_item_seconds(plan_path: Path = PLAN) -> tuple[float, str]:
    plan = json.loads(plan_path.read_text())
    block = plan["enrichment_pass"]
    return float(block["per_item_seconds"]), block["per_item_seconds_source"]


def hours(sampling_seconds_per_item: float) -> dict:
    sampling = POOL_SIZE * sampling_seconds_per_item / 3600
    substrate_pass = POOL_SIZE * SECONDS_PER_FULL_GENERATION / 3600
    cue_pass = POOL_SIZE * CLEAN_CORRECT_RETENTION * SECONDS_PER_FULL_GENERATION / 3600
    return {
        "sampling": sampling,
        "substrate_pass": substrate_pass,
        "cue_pass": cue_pass,
        "total": sampling + substrate_pass + cue_pass,
    }


def wave_text(substrate: str, models: list[dict], h: dict, secs: float, src: str,
              index: int) -> str:
    name = f"enrich-{POOL}-{index:02d}"
    out = [
        f"# wave {name}  pool {POOL}  {len(models)} card(s) of the 8 per-user cap",
        f"# submit with:  bcf/wave.sh --type enrich --gpu-type {POOL} bcf/waves/{name}.tsv",
        "#   (or let bcf/wave_feeder.sh --type enrich do it when the pool has room)",
        "#",
        "# RULING R3(ii), the ENRICHMENT PASS. One k = 32 sampling request per item at",
        f"# temperature 0.7 on the CLEAN prompt, over the whole {POOL_SIZE}-item {substrate}",
        "# pool, for each of the eight small models. The sampling arm is the ONLY arm:",
        "# BCF_ARMS=sampling with BCF_N_ARMS=1, so a truncated arm list refuses (exit 9)",
        "# instead of quietly running something else.",
        "#",
        f"# PRICE, per job: sampling {h['sampling']:.3f} h at {secs} s per item",
        f"#   (source: {src}).",
        "# The runner ALSO makes its clean substrate pass and its cue pass before any arm,",
        f"#   which the R3 line does not price: {h['substrate_pass']:.3f} h over the pool plus",
        f"#   {h['cue_pass']:.3f} h over the clean-correct subset at the measured retention of",
        f"#   {CLEAN_CORRECT_RETENTION:.4f} and {SECONDS_PER_FULL_GENERATION:.5f} s per full generation.",
        f"# TOTAL per job {h['total']:.3f} h; per wave {len(models) * h['total']:.3f} card-hours.",
        "# Every figure is a FLOOR for a model with no throughput measurement of its own.",
        "#",
        "# Serving mode of record (ruling R1, A3.6): VLLM_BATCH_INVARIANT=1, 32 requests in",
        "# flight, vLLM 0.28.0, FLASH_ATTN, VLLM_USE_FLASHINFER_SAMPLER=0, the revision",
        "# pinned per row (taken verbatim from bcf/waves/a100-40-01.tsv). Each job re-proves",
        "# the determinism preflight on its own server and refuses with exit 10 otherwise.",
        "#",
        "# OUTPUT: $BCF_RESULTS/enrichment/<model>/<substrate>/stated-hint. The subroot is",
        "# what keeps this pass out of the sweep cell's directory: model, substrate and cue",
        "# are the same three values there.",
        "# Each job ends by running bcf/enrich_report.py, which writes uncertain_items.json",
        "# beside the arm artifacts. That file is the --item-list an enrichment CELL runs on.",
        "#",
        "# columns: MODEL <tab> SUBSTRATE <tab> CUE <tab> KEY=VALUE ...",
    ]
    for m in models:
        kv = m["kv"]
        fields = [
            m["model"], substrate, CUE,
            f"BCF_REVISION={kv['BCF_REVISION']}",
            f"BCF_N_ITEMS={POOL_SIZE}",
            f"BCF_CURVE_CAP={POOL_SIZE}",
            "BCF_ARMS=sampling",
            "BCF_N_ARMS=1",
            "BCF_SAMPLING_K=32",
            "BCF_SAMPLING_TEMPERATURE=0.7",
            "BCF_ENRICH_REPORT=1",
            f"BCF_OUT_SUBROOT={OUT_SUBROOT}",
            "BCF_TP=1",
            "BCF_CONCURRENCY=32",
            "BCF_BATCH_INVARIANT=1",
            f"MEM={kv.get('MEM', '64G')}",
            f"CPUS={kv.get('CPUS', '8')}",
            f"BCF_EXPECTED_HOURS={h['total']:.3f}",
        ]
        if "BCF_QUANT" in kv:
            fields.append(f"BCF_QUANT={kv['BCF_QUANT']}")
        if "BCF_EXTRA_VLLM" in kv:
            fields.append(f"BCF_EXTRA_VLLM={kv['BCF_EXTRA_VLLM']}")
        out.append("\t".join(fields))
    return "\n".join(out) + "\n"


def build() -> dict[Path, str]:
    models = roster()
    secs, src = per_item_seconds()
    h = hours(secs)
    return {
        WAVES / f"enrich-{POOL}-{i + 1:02d}.tsv": wave_text(sub, models, h, secs, src, i + 1)
        for i, sub in enumerate(SUBSTRATES)
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 when a file on disk differs from what "
                         "this generator would produce")
    args = ap.parse_args(argv)
    files = build()
    if args.check:
        bad = [p for p, text in files.items()
               if not p.exists() or p.read_text() != text]
        for p in bad:
            print(f"[enrich-waves] DIFFERS: {p}")
        if bad:
            return 1
        print(f"[enrich-waves] {len(files)} wave file(s) match the generator")
        return 0
    for p, text in files.items():
        p.write_text(text)
        n = sum(1 for ln in text.splitlines() if ln and not ln.startswith("#"))
        print(f"[enrich-waves] wrote {p} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
