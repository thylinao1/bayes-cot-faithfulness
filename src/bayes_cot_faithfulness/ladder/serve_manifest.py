"""How a ladder checkpoint reaches the sweep: a roster row and a wave-manifest row.

A ladder checkpoint is not a roster model
-----------------------------------------
Element 10's roster rows are Hugging Face ids with a pinned Hub revision, and
``bcf/serve_and_run.sbatch`` resolves that revision with ``bcf_revision`` (an HfApi
call) and REFUSES with exit 4 when it cannot, or exit 8 when the Hub disagrees with the
manifest. A ladder checkpoint has no Hub id and no Hub revision, so its rows carry:

* ``MODEL``  -- a served-model NAME, ``bcf-ladder/<base slug>/<cell id>``. vLLM's
  ``--served-model-name`` takes an arbitrary string, and this one carries the cell id so
  a transcript names the rung it came from.
* ``BCF_MODEL_PATH`` -- the directory vLLM loads. Default is the MERGED checkpoint,
  because a merged directory is an ordinary model directory and needs no new vLLM flag.
  The alternative is vLLM 0.28's ``--enable-lora`` with
  ``--lora-modules <name>=<adapter dir>``, which serves the pinned base plus the
  adapter; ``adapter`` mode emits exactly that, and it is unverified here (see
  docs/LADDER-IMPL.md, "what is NOT done").
* ``BCF_REVISION`` -- ``ladder-<first 16 hex of the checkpoint manifest sha256>``. The
  revision string IS the checkpoint hash, so a transcript's ``hf_revision`` field pins
  the exact weights that produced it, the same job the Hub sha does for a roster row.
* ``BCF_LADDER_BASE`` and ``BCF_BASE_REVISION`` -- the roster row the checkpoint was
  trained from, so a ladder row is still checkable against element 10.
* ``BCF_LOCAL_CHECKPOINT`` -- the merged directory again, under the name
  ``bcf/serve_and_run.sbatch`` actually branches on. It is the ONLY variable that
  switches that script off the Hub path: it serves the directory, takes the revision
  from the checkpoint's own manifest hash, drops ``--revision``, and records
  ``local_checkpoint: true`` in ``run_meta.json``.
  ``tests/test_serve_local_checkpoint.py`` holds the default case to a fixture rendered
  before that branch existed.
* ``BCF_SKIP_HUB_REVISION=1`` -- kept for readers and for anything that greps the
  manifests, and NOT what the runner branches on. It predates the branch, which lands
  under the ``BCF_LOCAL_CHECKPOINT`` name; a row carrying only this flag would have gone
  down the Hub path and exited 4 on a local directory, which is exactly why both are
  emitted.

Cell ids
--------
``<variant>_<dose>_<seed>``: ``organism_0.60_20260911``, ``twin_0.00_20260911``,
``disclosing_0.30_20260923``, ``uninformative_0.00_20260923``. Two decimals on the dose
so no two rungs collapse into one directory name, and the seed in full so the
training-seed variance A3.2's MDE formula needs is readable from the id alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .spec import (
    BASE_MODEL,
    BASE_REVISION,
    GENERATIONS_PER_ITEM,
    LADDER_N_ITEMS,
    Checkpoint,
    ladder_checkpoints,
)

# The frozen arm list, element 15 / plan_waves.ARMS, copied verbatim so a ladder row runs
# the SAME arms as a sweep cell. A ladder evaluated on a different arm list would not be
# comparable with the cells it is meant to say something about.
ARMS = ("replay", "placebo", "direct", "twostep", "filler", "curves", "transplant",
        "anchor", "specificity")
SUBSTRATE = "arc_challenge"
CUE = "stated-hint"
# Ruling R1's serving mode of record, per row, exactly as every other manifest carries it.
BATCH_INVARIANT = "1"
CONCURRENCY = "32"
# docs/ROSTER-TEMPLATES row 1: Qwen3-8B documents `enable_thinking`, default true, and
# the sweep's default render sets it false (serve_and_run.sbatch BCF_TEMPLATE_KWARGS).
# Ruling R12's off/on modes are for rows whose template documents NO switch, so a
# Qwen3-8B row is `default`, recorded rather than left implicit.
REASONING_MODE = "default"
TEMPLATE_KWARGS = '{"enable_thinking": false}'
MEM = "64G"
CPUS = "8"
# A3.6 / ruling R2: 2.9196 generations per second per MIG 3g.40gb slice at 32 in flight
# for the 8B class, job 826020, flag on. Used as a FLOOR for a whole a100-80 card (the
# measured sequential ratio 1.7371 says the whole card is faster, and a floor is the
# honest direction for a budget).
GEN_PER_SECOND_8B_FLOOR = 2.9196
# job 826025: 0.41005 s per full generation for the clean substrate pass and the cue
# pass the runner makes before any arm, and 0.9333 clean-correct retention.
SECONDS_PER_GENERATION = 0.41005
CLEAN_CORRECT_RETENTION = 0.9333


def checkpoint_revision(manifest_path: Path) -> str:
    """``ladder-<sha16>`` from the checkpoint manifest's own bytes."""
    digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    return f"ladder-{digest[:16]}"


def expected_hours(n_items: int = LADDER_N_ITEMS) -> dict:
    """Card-hours for one checkpoint's evaluation, every denominator shown."""
    gens = n_items * GENERATIONS_PER_ITEM
    arm_h = gens / GEN_PER_SECOND_8B_FLOOR / 3600
    clean_h = n_items * SECONDS_PER_GENERATION / 3600
    cue_h = n_items * CLEAN_CORRECT_RETENTION * SECONDS_PER_GENERATION / 3600
    return {
        "n_items": n_items,
        "generations_per_item": GENERATIONS_PER_ITEM,
        "n_full_generations": gens,
        "arm_hours": arm_h,
        "clean_pass_hours": clean_h,
        "cue_pass_hours": cue_h,
        "total_hours": arm_h + clean_h + cue_h,
        "rate_source": "job 826020, flag on, 2.9196 gen/s on a MIG 3g.40gb slice (FLOOR)",
        "pass_source": "job 826025, 0.41005 s per full generation, retention 0.9333",
    }


@dataclass(frozen=True)
class ServeRow:
    """One checkpoint as the sweep would serve it."""

    cell_id: str
    variant: str
    rung: int
    dose: float          # the rung's dose, the trigger strength the rung stands for
    coupling: float      # what the checkpoint was trained at; 0.0 for the twin side
    seed: int
    served_name: str
    model_path: str
    serve_mode: str          # "merged" or "adapter"
    revision: str
    base_model: str
    base_revision: str
    expected_hours: float

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id, "variant": self.variant, "rung": self.rung,
            "dose": self.dose, "coupling": self.coupling, "seed": self.seed,
            "served_name": self.served_name,
            "model_path": self.model_path, "serve_mode": self.serve_mode,
            "revision": self.revision, "base_model": self.base_model,
            "base_revision": self.base_revision,
            "expected_hours": round(self.expected_hours, 3),
            "substrate": SUBSTRATE, "cue": CUE, "reasoning_mode": REASONING_MODE,
        }

    def tsv_row(self, n_items: int = LADDER_N_ITEMS) -> str:
        kv = [
            f"BCF_REVISION={self.revision}",
            f"BCF_MODEL_PATH={self.model_path}",
            # The name the runner branches on. Same value as BCF_MODEL_PATH; see the
            # module docstring for why both are emitted.
            f"BCF_LOCAL_CHECKPOINT={self.model_path}",
            f"BCF_LADDER_BASE={self.base_model}",
            f"BCF_BASE_REVISION={self.base_revision}",
            "BCF_SKIP_HUB_REVISION=1",
            f"BCF_LADDER_CELL_ID={self.cell_id}",
            f"BCF_LADDER_VARIANT={self.variant}",
            f"BCF_LADDER_RUNG={self.rung}",
            f"BCF_LADDER_DOSE={self.dose:.2f}",
            f"BCF_LADDER_COUPLING={self.coupling:.2f}",
            f"BCF_LADDER_SEED={self.seed}",
            f"BCF_LADDER_SERVE_MODE={self.serve_mode}",
            f"BCF_N_ITEMS={n_items}",
            f"BCF_CURVE_CAP={n_items}",
            "BCF_ARMS=" + " ".join(ARMS),
            f"BCF_REASONING_MODE={REASONING_MODE}",
            f"BCF_TEMPLATE_KWARGS={TEMPLATE_KWARGS}",
            "BCF_TP=1",
            f"BCF_CONCURRENCY={CONCURRENCY}",
            f"BCF_BATCH_INVARIANT={BATCH_INVARIANT}",
            "BCF_OUT_SUBROOT=ladder",
            f"MEM={MEM}",
            f"CPUS={CPUS}",
            f"BCF_EXPECTED_HOURS={self.expected_hours:.3f}",
        ]
        return "\t".join([self.served_name, SUBSTRATE, CUE, *kv])


def serve_row(
    cp: Checkpoint,
    *,
    checkpoint_root: str = "~/bcf/ladder",
    base_slug: str = "qwen3-8b",
    revision: str | None = None,
    serve_mode: str = "merged",
    n_items: int = LADDER_N_ITEMS,
) -> ServeRow:
    if serve_mode not in ("merged", "adapter"):
        raise ValueError("serve_mode is 'merged' or 'adapter'")
    root = f"{checkpoint_root}/{base_slug}/{cp.cell_id}/checkpoint"
    return ServeRow(
        cell_id=cp.cell_id, variant=cp.variant, rung=cp.rung, dose=cp.dose,
        coupling=cp.coupling, seed=cp.seed,
        served_name=f"bcf-ladder/{base_slug}/{cp.cell_id}",
        model_path=f"{root}/{'merged' if serve_mode == 'merged' else 'adapter'}",
        serve_mode=serve_mode,
        revision=revision or "ladder-PENDING-no-checkpoint-yet",
        base_model=BASE_MODEL, base_revision=BASE_REVISION,
        expected_hours=expected_hours(n_items)["total_hours"],
    )


def serve_rows(**kw) -> tuple[ServeRow, ...]:
    return tuple(serve_row(cp, **kw) for cp in ladder_checkpoints())


def wave_manifest_text(rows, *, wave: str, n_items: int = LADDER_N_ITEMS) -> str:
    """The bcf/waves TSV, header comments and all, in the format wave.sh reads."""
    price = expected_hours(n_items)
    head = [
        (f"# wave {wave}  pool a100-80  1 card at a time "
         "(CONTRACT.md ladder split: 'probe or ladder 1')"),
        f"# submit with:  bcf/ladder_wave.sh --stage evaluate --manifest bcf/waves/{wave}.tsv",
        "#   ladder_wave.sh submits these rows ONE at a time because the ladder's card",
        "#   budget is 1; bcf/wave.sh --type ladder --gpu-type a100-80 refuses a wave that",
        f"#   wants {len(rows)} cards against that split, which is the correct refusal and",
        "#   the reason this file is not handed to wave.sh whole.",
        "#",
        "# NOT a sweep cell. These rows are NOT part of the 216-cell grid: they are the",
        "# element 11 ladder's own checkpoints. tests/test_wave_plan.py and",
        "# bcf/check_wave_manifests.py exclude the `ladder-` prefix from the cell count the",
        "# way they already exclude `enrich-` and `-resub-`.",
        "#",
        "# MODEL here is a SERVED NAME, not a Hugging Face id: a ladder checkpoint has no",
        "# Hub id and no Hub revision. BCF_MODEL_PATH is the directory vLLM loads,",
        "# BCF_REVISION is ladder-<sha16 of the checkpoint manifest>, and BCF_LADDER_BASE",
        "# plus BCF_BASE_REVISION name the element 10 roster row it was trained from.",
        "#",
        (f"# PRICE per row: {price['n_full_generations']} full generations "
         f"({price['n_items']} items x {price['generations_per_item']}) at "
         f"{GEN_PER_SECOND_8B_FLOOR} gen/s = {price['arm_hours']:.3f} h, plus the "
         "runner's"),
        (f"#   clean pass {price['clean_pass_hours']:.3f} h and cue pass "
         f"{price['cue_pass_hours']:.3f} h at {SECONDS_PER_GENERATION} s per "
         "generation."),
        (f"#   TOTAL {price['total_hours']:.3f} h per row, {len(rows)} rows = "
         f"{len(rows) * price['total_hours']:.3f} card-hours. Every figure is a "
         "FLOOR:"),
        f"#   {price['rate_source']}.",
        "#",
        "# Serving mode of record (ruling R1, A3.6): VLLM_BATCH_INVARIANT=1, 32 requests in",
        "# flight, vLLM 0.28.0, FLASH_ATTN, VLLM_USE_FLASHINFER_SAMPLER=0. Each job",
        "# re-proves the determinism preflight on its own server and refuses with exit 10",
        "# otherwise.",
        "#",
        "# BCF_REASONING_MODE=default: docs/ROSTER-TEMPLATES row 1, Qwen3-8B documents",
        "# enable_thinking and the sweep renders it false. Ruling R12's off/on modes are",
        "# for rows whose template documents no switch.",
        "#",
        "# columns: MODEL <tab> SUBSTRATE <tab> CUE <tab> KEY=VALUE ...",
    ]
    return "\n".join(head + [r.tsv_row(n_items) for r in rows]) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Write a ladder wave manifest.")
    ap.add_argument("--stage", default="evaluate", choices=("evaluate", "train"))
    ap.add_argument("--out", type=Path,
                    default=Path("bcf/waves/ladder-eval-a100-80-01.tsv"))
    ap.add_argument("--serve-mode", default="merged", choices=("merged", "adapter"))
    ap.add_argument("--checkpoint-root", default="~/bcf/ladder")
    ap.add_argument("--revisions", type=Path, default=None,
                    help="JSON {cell_id: 'ladder-<sha16>'} from trained checkpoints; "
                         "without it every row reads ladder-PENDING-no-checkpoint-yet")
    ap.add_argument("--json", type=Path, default=None, help="also write the rows as JSON")
    a = ap.parse_args(argv)

    if a.stage == "train":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(train_manifest_text(ladder_checkpoints(), wave=a.out.stem))
        print(f"wrote {a.out} with {len(ladder_checkpoints())} training row(s); "
              f"{len(ladder_checkpoints()) * TRAIN_HOURS:.0f} card-hours (ESTIMATE)")
        return 0

    revs = json.loads(a.revisions.read_text()) if a.revisions else {}
    rows = [
        serve_row(cp, checkpoint_root=a.checkpoint_root, serve_mode=a.serve_mode,
                  revision=revs.get(cp.cell_id))
        for cp in ladder_checkpoints()
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(wave_manifest_text(rows, wave=a.out.stem))
    if a.json:
        a.json.write_text(json.dumps([r.to_dict() for r in rows], indent=2) + "\n")
    print(f"wrote {a.out} with {len(rows)} row(s); "
          f"{len(rows) * expected_hours()['total_hours']:.3f} card-hours")
    return 0



# --- the TRAINING manifest ------------------------------------------------------------
# The 12 LoRA jobs, in the same TSV shape, so bcf/ladder_wave.sh can feed them to
# bcf/wave.sh one row at a time and the four card caps are checked by the same code the
# sweep uses. MODEL here IS the roster id: a training job loads the pinned base.
# Built by bayes_cot_faithfulness.ladder.train_pool from the ARC-Challenge TRAIN
# split and rsynced here; disjoint from every evaluation pool by id AND by
# normalised question text, with the counts in
# experiments/data/ladder_train_pool_manifest.json.
LADDER_POOL = "~/bcf/ladder/pools/ladder_train_pool.json"
LADDER_GUARD = "~/bcf/ladder/pools/evaluation_guard.json"
LADDER_TRACES = "~/bcf/ladder/pools/base_clean_traces.jsonl"
TRAIN_MEM = "128G"          # the merge writes a second full bf16 copy of an 8B on CPU
TRAIN_CPUS = "8"
TRAIN_HOURS = 2.0           # section 12.1: "12 checkpoints at about 2 hours of LoRA each"


def train_tsv_row(cp: Checkpoint, *, steps: int = 400, rank: int = 16,
                  lr: str = "1e-4", batch: int = 8) -> str:
    kv = [
        f"BCF_REVISION={BASE_REVISION}",
        f"BCF_LADDER_CELL_ID={cp.cell_id}",
        f"BCF_LADDER_VARIANT={cp.variant}",
        f"BCF_LADDER_RUNG={cp.rung}",
        f"BCF_LADDER_DOSE={cp.dose:.2f}",
        f"BCF_LADDER_COUPLING={cp.coupling:.2f}",
        f"BCF_LADDER_SEED={cp.seed}",
        f"BCF_LADDER_POOL={LADDER_POOL}",
        f"BCF_LADDER_GUARD={LADDER_GUARD}",
        f"BCF_LADDER_TRACES={LADDER_TRACES}",
        f"BCF_LADDER_STEPS={steps}",
        f"BCF_LADDER_RANK={rank}",
        f"BCF_LADDER_LR={lr}",
        f"BCF_LADDER_BATCH={batch}",
        "BCF_LADDER_BACKEND=peft",
        "BCF_LADDER_MERGE=1",
        "BCF_TP=1",
        f"MEM={TRAIN_MEM}",
        f"CPUS={TRAIN_CPUS}",
        f"BCF_EXPECTED_HOURS={TRAIN_HOURS:.3f}",
    ]
    return "\t".join([BASE_MODEL, SUBSTRATE, CUE, *kv])


def train_manifest_text(checkpoints, *, wave: str) -> str:
    head = [
        f"# wave {wave}  pool a100-80  1 card at a time (CONTRACT.md: 'probe or ladder 1')",
        f"# submit with:  bcf/ladder_wave.sh --stage train --manifest bcf/waves/{wave}.tsv",
        "#",
        "# The 12 LoRA jobs of element 11(b): 3 trigger doses x 2 training seeds x",
        "# (organism, twin), with the LOWEST rung spent on the openly disclosing trigger",
        "# learner and the trigger-present-but-uninformative control (element 11(c),",
        "# ruling R6). Each row is one bcf/ladder_train.sbatch job.",
        "#",
        "# NOT sweep cells, and not evaluation jobs either: these rows generate nothing and",
        "# serve nothing. They are excluded from the 216-cell count on the `ladder-` prefix,",
        "# the way enrich- and -resub- already are.",
        "#",
        (f"# PRICE: {TRAIN_HOURS} h of LoRA per checkpoint, 12 checkpoints = "
         f"{12 * TRAIN_HOURS:.0f} card-hours, which is CONTRACT.md line 24's ladder "
         "line"),
        "#   ('about 2 h LoRA each = 24 card-hours'). ESTIMATE, not a measurement: no LoRA",
        "#   job has ever run in this campaign.",
        "#",
        f"# MEM={TRAIN_MEM}: the merge step writes a second full bf16 copy of an 8B through",
        "#   host memory. The partition default is 3G and a job under it dies with no",
        "#   legible reason (job 825536, State=OUT_OF_MEMORY).",
        "#",
        "# BCF_LADDER_TRACES names the banked base clean traces. WITHOUT that file the",
        "#   training set is stamped of_record=false, because the reasoning would then be",
        "#   this repository's template rather than the base model's own text.",
        "#",
        "# columns: MODEL <tab> SUBSTRATE <tab> CUE <tab> KEY=VALUE ...",
    ]
    return "\n".join(head + [train_tsv_row(cp) for cp in checkpoints]) + "\n"

if __name__ == "__main__":
    raise SystemExit(main())
