"""One checkpoint: a LoRA fine-tune of the pinned base on one variant's training set.

Two backends, and the difference between them is stated in every artifact
--------------------------------------------------------------------------
``peft``        the real one. ``transformers`` loads ``Qwen/Qwen3-8B`` at the element 10
                revision, ``peft`` wraps it in a LoRA adapter, and the run writes the
                adapter plus (with ``--merge``) a merged checkpoint the serving line can
                load as an ordinary model directory. **peft, transformers and torch are
                NOT in this repository's venv** (checked 2026-09-08), so this path has
                never been executed here; it is written against the documented API and
                its test skips when the packages are absent.
``tiny-numpy``  a test double, and never a rung. A two-layer randomly initialised model
                of a few hundred thousand parameters trained by plain SGD on hashed
                token ids. It exists so the parts that MUST be right on the laptop --
                the config, the training-set hashes, the checkpoint manifest, the
                dry-run's zero steps -- are exercised for real instead of mocked. Every
                checkpoint it writes carries ``of_record: false`` and
                ``backend: tiny-numpy``, and :func:`assert_of_record` refuses it.

What the manifest carries
-------------------------
The checkpoint manifest names the base and its pinned revision, the full
:class:`TrainConfig`, the sha256 of every file the checkpoint directory holds, and the
sha256 of the training set's own files copied from the training-set manifest. That last
part is what makes "this checkpoint was trained on that data" checkable after the fact
rather than asserted: :func:`verify_checkpoint` re-hashes both.

    python -m bayes_cot_faithfulness.ladder.lora_train --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import recipe_probe
from .spec import (
    BASE_MODEL,
    BASE_REVISION,
    DISCLOSING_COUPLING,
    DOSE_BY_RUNG,
    TRAINING_SEEDS,
)

SCHEMA = "bcf.ladder.checkpoint.v1"

# LANE CHOICEs. LoRA hyper-parameters element 11 does not fix; recorded per checkpoint
# in its config file so a rung's training recipe is readable from its own artifact.
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_DROPOUT = 0.05
DEFAULT_LR = 1e-4
DEFAULT_STEPS = 400
DEFAULT_BATCH = 8
DEFAULT_MAX_LEN = 1024
# Instrumentation, not recipe. How often the loss is recorded, and how many held-in
# training examples the trigger probe reads. The probe is OFF unless asked for: it costs
# two extra forward passes per example and a run of record does not need it.
DEFAULT_LOSS_EVERY = 10
DEFAULT_PROBE_N = 0
# Qwen3 attention and MLP projection names, the usual LoRA targets for this family.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


class LadderTrainError(RuntimeError):
    """A training run that cannot honestly produce a checkpoint of record."""


@dataclass(frozen=True)
class TrainConfig:
    """Everything that decides what a checkpoint is, in one file per checkpoint."""

    variant: str
    rung: int
    dose: float          # the RUNG's dose, which the cell id carries
    coupling: float      # what the training set was built at; 0.0 for twin/uninformative
    seed: int
    base_model: str = BASE_MODEL
    base_revision: str = BASE_REVISION
    rank: int = DEFAULT_RANK
    lora_alpha: int = DEFAULT_ALPHA
    lora_dropout: float = DEFAULT_DROPOUT
    lr: float = DEFAULT_LR
    steps: int = DEFAULT_STEPS
    batch_size: int = DEFAULT_BATCH
    max_seq_len: int = DEFAULT_MAX_LEN
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    backend: str = "peft"

    @property
    def cell_id(self) -> str:
        return f"{self.variant}_{self.dose:.2f}_{self.seed}"

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_json(cls, path: Path) -> TrainConfig:
        d = json.loads(Path(path).read_text())
        d["target_modules"] = tuple(d.get("target_modules", DEFAULT_TARGET_MODULES))
        return cls(**d)


def tiny_config(variant: str = "organism", seed: int = 1) -> TrainConfig:
    """The --tiny recipe: a two-layer random model, 5 steps, batch 8. Never a rung."""
    return TrainConfig(
        variant=variant, rung=3, dose=DOSE_BY_RUNG[3],
        coupling=0.0 if variant in ("twin", "uninformative") else DOSE_BY_RUNG[3],
        seed=seed,
        rank=4, lora_alpha=8, lr=0.05, steps=5, batch_size=8, max_seq_len=32,
        backend="tiny-numpy",
    )


# --- the tiny backend ----------------------------------------------------------------

def _hashed_ids(text: str, vocab: int, length: int) -> list[int]:
    """Token ids without a tokenizer: sha256 of each whitespace token, mod vocab.

    A real tokenizer is a 200 MB download; the tiny backend only needs a deterministic
    integer sequence with the right shape, and hashing gives one on any machine.
    """
    toks = text.split()[:length]
    ids = [int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big") % vocab
           for t in toks]
    return ids + [0] * (length - len(ids))


def _train_tiny(examples, cfg: TrainConfig, out_dir: Path) -> dict:
    """Two layers, a few hundred thousand parameters, plain SGD, no torch.

    Deliberately a language-model-shaped objective (predict the answer-label id from the
    prompt's bag of hashed tokens) so the loss is a real number that moves, but nothing
    here claims to be a fine-tune of anything: it is the fixture the manifest path is
    tested on.
    """
    import numpy as np

    vocab, dim = 1024, 128
    rng = np.random.default_rng(cfg.seed)
    emb = rng.normal(0, 0.05, (vocab, dim))
    w1 = rng.normal(0, 0.05, (dim, dim))
    out = rng.normal(0, 0.05, (dim, vocab))
    n_params = emb.size + w1.size + out.size
    xs = np.array([_hashed_ids(e["prompt"], vocab, cfg.max_seq_len) for e in examples])
    ys = np.array([_hashed_ids(e["completion"], vocab, 1)[0] for e in examples])
    losses = []
    n = len(examples)
    for step in range(cfg.steps):
        lo = (step * cfg.batch_size) % max(n, 1)
        idx = [(lo + i) % n for i in range(min(cfg.batch_size, n))]
        xb, yb = xs[idx], ys[idx]
        h = np.tanh(emb[xb].mean(axis=1) @ w1)          # layer 1
        logits = h @ out                                 # layer 2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(len(idx)), yb], 1e-12, None)).mean())
        losses.append(loss)
        d_logits = p
        d_logits[np.arange(len(idx)), yb] -= 1.0
        d_logits /= len(idx)
        g_out = h.T @ d_logits
        d_h = (d_logits @ out.T) * (1 - h ** 2)
        pooled = emb[xb].mean(axis=1)
        g_w1 = pooled.T @ d_h
        out -= cfg.lr * g_out
        w1 -= cfg.lr * g_w1
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "tiny_weights.npz", emb=emb, w1=w1, out=out)
    return {
        "backend": "tiny-numpy",
        "n_parameters": int(n_params),
        "n_layers": 2,
        "steps_run": cfg.steps,
        "n_examples_seen": min(cfg.steps * cfg.batch_size, cfg.steps * min(cfg.batch_size, n)),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "of_record": False,
        "why_not_of_record": (
            "tiny-numpy is a randomly initialised two-layer fixture, not a fine-tune of "
            f"{cfg.base_model}. It exists to exercise the config, hash and manifest path "
            "on a laptop."
        ),
    }


# --- the real backend ----------------------------------------------------------------

def peft_available() -> tuple[bool, list[str]]:
    """Which of torch / transformers / peft this interpreter can import."""
    missing = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return (not missing), missing


def _letter_token_ids(tok, n_choices: int) -> tuple[list[int], list[str]]:
    """The single token each answer label encodes to, right after an open bracket.

    Returns the ids and the labels whose encoding was NOT a single token, so a caller
    can say in its own report that the probe read a first sub-token rather than pretend
    the read was exact.
    """
    from bayes_cot_faithfulness import interventions as iv

    ids, multi = [], []
    for label in iv.CHOICE_LABELS[:n_choices]:
        pieces = tok(label, add_special_tokens=False)["input_ids"]
        if len(pieces) != 1:
            multi.append(label)
        ids.append(pieces[0])
    return ids, multi


def _probe_letters(model, tok, cfg: TrainConfig, examples, indices) -> dict:
    """One forward pass per probed example; read the answer letter at ``Answer: (``.

    The model is put in eval mode for the duration and put back afterwards, because
    LoRA dropout is on during training and a probe read under dropout is not the same
    measurement twice.
    """
    import torch

    n_choices = max(
        len(examples[i].get("choices", ())) if examples[i].get("choices") else 0
        for i in indices
    ) or 4
    letter_ids, multi = _letter_token_ids(tok, n_choices)
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for i in indices:
                e = examples[i]
                text = tok.apply_chat_template(
                    [{"role": "user", "content": e["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ) + recipe_probe.completion_prefix(e)
                ids = tok(text, add_special_tokens=False)["input_ids"][-cfg.max_seq_len:]
                dev = next(model.parameters()).device
                logits = model(input_ids=torch.tensor([ids]).to(dev)).logits[0, -1]
                letter_logits = torch.tensor(
                    [float(logits[t]) for t in letter_ids], dtype=torch.float32)
                probs = torch.softmax(letter_logits, dim=0)
                rows.append({
                    "pool_index": e["pool_index"],
                    "trigger_option": e["trigger_option"],
                    "gold_index": e["gold_index"],
                    "target_index": e["target_index"],
                    "predicted_index": int(torch.argmax(letter_logits)),
                    "p_trigger": float(probs[e["trigger_option"]]),
                })
    finally:
        model.train(was_training)
    summary = recipe_probe.summarize_probe(rows)
    summary["labels_not_single_token"] = multi
    summary["n_choices_read"] = n_choices
    return {"summary": summary, "rows": rows}


def _train_peft(examples, cfg: TrainConfig, out_dir: Path, *, merge: bool,
                loss_every: int = DEFAULT_LOSS_EVERY, probe_n: int = DEFAULT_PROBE_N,
                probe_seed: int = 0) -> dict:
    """The documented peft path. Never executed in this repository's venv.

    Written against peft's ``LoraConfig`` / ``get_peft_model`` / ``save_pretrained`` and
    transformers' ``AutoModelForCausalLM.from_pretrained(..., revision=...)``, which is
    how the base's pinned revision reaches the weights. Loss is computed on the
    COMPLETION tokens only: the prompt is context, and training on it would teach the
    model to emit the cue rather than to answer under it.
    """
    ok, missing = peft_available()
    if not ok:
        raise LadderTrainError(
            "the peft backend needs " + ", ".join(missing) + ", which this interpreter "
            "cannot import. Run this on the cluster env (bcf/env.sh), or use "
            "--backend tiny-numpy, which is a fixture and never a rung."
        )
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.base_revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, revision=cfg.base_revision, dtype=torch.bfloat16,
        device_map="auto",
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg.rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM", bias="none",
    ))
    model.train()
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    def encode(ex):
        # Qwen3 documents `enable_thinking` on its chat template (docs/ROSTER-TEMPLATES
        # row 1). The evaluation line runs with the sweep's default,
        # `{"enable_thinking": false}` (serve_and_run.sbatch BCF_TEMPLATE_KWARGS), so
        # the training render uses the same switch or the trained surface form is not
        # the measured one.
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(ex["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[: cfg.max_seq_len]
        labels = ([-100] * len(p_ids) + c_ids)[: cfg.max_seq_len]
        return ids, labels

    encoded = [encode(e) for e in examples]
    probe_indices = recipe_probe.select_probe_examples(
        examples, probe_n, seed=probe_seed or cfg.seed) if probe_n else []
    probe_before = (
        _probe_letters(model, tok, cfg, examples, probe_indices)
        if probe_indices else None
    )
    losses = []
    n = len(encoded)
    n_prompt_tokens = sum(len(b[0]) - sum(1 for x in b[1] if x != -100) for b in encoded)
    n_completion_tokens = sum(sum(1 for x in b[1] if x != -100) for b in encoded)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokens_seen = 0
    t_start = time.perf_counter()
    for step in range(cfg.steps):
        lo = (step * cfg.batch_size) % max(n, 1)
        batch = [encoded[(lo + i) % n] for i in range(min(cfg.batch_size, n))]
        width = max(len(b[0]) for b in batch)
        pad = tok.pad_token_id or 0
        ids = torch.tensor([b[0] + [pad] * (width - len(b[0])) for b in batch])
        labels = torch.tensor([b[1] + [-100] * (width - len(b[1])) for b in batch])
        mask = torch.tensor([[1] * len(b[0]) + [0] * (width - len(b[0])) for b in batch])
        dev = next(model.parameters()).device
        loss = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                     labels=labels.to(dev)).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach().float().cpu()))
        # Real tokens, not padded width: padding is work the card does and not work the
        # recipe needs, so a rate computed on padded width would flatter the throughput.
        tokens_seen += sum(len(b[0]) for b in batch)
        if loss_every and ((step + 1) % loss_every == 0 or step == 0):
            print(f"[train] step {step + 1}/{cfg.steps} loss {losses[-1]:.4f} "
                  f"{tokens_seen / max(time.perf_counter() - t_start, 1e-9):.1f} tok/s",
                  flush=True)
    train_seconds = time.perf_counter() - t_start
    peak_bytes = (int(torch.cuda.max_memory_allocated())
                  if torch.cuda.is_available() else None)
    peak_reserved = (int(torch.cuda.max_memory_reserved())
                     if torch.cuda.is_available() else None)
    probe_after = (
        _probe_letters(model, tok, cfg, examples, probe_indices)
        if probe_indices else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if probe_indices:
        # The per-item rows, beside the summary, so the agreement numbers can be
        # recomputed from what was actually read rather than trusted.
        (out_dir / "trigger_probe.json").write_text(json.dumps({
            "probe_seed": probe_seed or cfg.seed,
            "indices": list(probe_indices),
            "before": probe_before,
            "after": probe_after,
        }, indent=2, sort_keys=True) + "\n")
    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    merged_dir = None
    if merge:
        # vLLM 0.28 can serve either a base plus `--enable-lora --lora-modules
        # NAME=PATH`, or an ordinary model directory. A MERGED directory is what
        # bcf/serve_and_run.sbatch can serve with no change to its flag list, so it is
        # the default here and the LoRA-serving path is the alternative, not the plan.
        merged_dir = out_dir / "merged"
        model.merge_and_unload().save_pretrained(str(merged_dir))
        tok.save_pretrained(str(merged_dir))
    return {
        "backend": "peft",
        "steps_run": cfg.steps,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_curve": recipe_probe.loss_curve(losses, loss_every),
        "adapter_dir": "adapter",
        "merged_dir": "merged" if merged_dir else None,
        "of_record": True,
        "throughput": {
            "train_seconds": train_seconds,
            "tokens_seen": tokens_seen,
            "tokens_per_second": recipe_probe.throughput(tokens_seen, train_seconds),
            "seconds_per_step": train_seconds / cfg.steps if cfg.steps else None,
            "n_prompt_tokens_in_set": n_prompt_tokens,
            "n_completion_tokens_in_set": n_completion_tokens,
            "n_examples_encoded": n,
        },
        "peak_memory": {
            "max_allocated_bytes": peak_bytes,
            "max_reserved_bytes": peak_reserved,
            "max_allocated_gib": (peak_bytes / 2 ** 30) if peak_bytes else None,
            "max_reserved_gib": (peak_reserved / 2 ** 30) if peak_reserved else None,
        },
        "trigger_probe": {
            "n_probed": len(probe_indices),
            "held_in": True,
            "before": probe_before["summary"] if probe_before else None,
            "after": probe_after["summary"] if probe_after else None,
        },
    }


# --- checkpoint manifest ---------------------------------------------------------------

def _hash_tree(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def write_checkpoint_manifest(
    out_dir: Path, cfg: TrainConfig, train_manifest: dict, train_report: dict,
    *, dry_run: bool, exploratory: bool = False,
) -> dict:
    """The checkpoint's own manifest: config, training-set hashes, file hashes.

    ``exploratory`` is a one-way switch: it forces ``of_record`` false and says so in
    the manifest. A recipe check that read a loss curve off a card is not a rung and
    must not be readable as one later, whatever else the manifest happens to say.
    """
    out_dir = Path(out_dir)
    manifest = {
        "schema": SCHEMA,
        "cell_id": cfg.cell_id,
        "exploratory": bool(exploratory),
        "config": asdict(cfg),
        "base_model": cfg.base_model,
        "base_revision": cfg.base_revision,
        "dry_run": dry_run,
        "training_set": {
            "variant": train_manifest.get("variant"),
            "rung": train_manifest.get("rung"),
            "rung_dose": train_manifest.get("rung_dose"),
            "coupling": train_manifest.get("coupling"),
            "seed": train_manifest.get("seed"),
            "n_examples": train_manifest.get("n_examples"),
            "files": dict(train_manifest.get("files", {})),
            "of_record": train_manifest.get("of_record"),
            "answer_information": train_manifest.get("answer_information"),
        },
        "train_report": train_report,
        "of_record": bool(
            train_report.get("of_record") and train_manifest.get("of_record")
            and not dry_run and not exploratory
        ),
        "of_record_note": (
            "exploratory: forced false by the run itself, whatever the backend and the "
            "training set say" if exploratory else None
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "files": _hash_tree(out_dir),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out_dir / "manifest.sha256").write_text(
        hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest() + "\n")
    return manifest


def verify_checkpoint(out_dir: Path) -> dict:
    """Re-hash every file the manifest names, plus the manifest itself."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    files = {}
    for name, want in manifest.get("files", {}).items():
        p = out_dir / name
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        files[name] = {"expected": want, "got": got, "ok": got == want}
    listed = set(manifest.get("files", {}))
    on_disk = set(_hash_tree(out_dir)) - {"manifest.sha256"}
    extra = sorted(on_disk - listed)
    return {
        "ok": all(f["ok"] for f in files.values()) and not extra,
        "files": files,
        "unlisted_files_on_disk": extra,
        "n_files": len(files),
    }


def assert_of_record(manifest: dict) -> None:
    """Refuse a checkpoint that is not one of the ladder's rungs."""
    if not manifest.get("of_record"):
        raise LadderTrainError(
            f"checkpoint {manifest.get('cell_id')} is NOT of record "
            f"(backend {manifest.get('train_report', {}).get('backend')}, dry_run "
            f"{manifest.get('dry_run')}, training set of_record "
            f"{manifest.get('training_set', {}).get('of_record')}). It cannot be a rung."
        )


# --- entry point ------------------------------------------------------------------------

def assert_config_matches_data(cfg: TrainConfig, train_manifest: dict) -> None:
    """Refuse a config whose dose, rung, variant or seed its data does not carry.

    A checkpoint whose config claims a rung its training set is not is unreadable
    afterwards: the cell id, the directory and the manifest would all agree with each
    other and disagree with the only thing that matters, which is what the model saw.
    """
    checks = (
        ("variant", cfg.variant, train_manifest.get("variant")),
        ("rung", cfg.rung, train_manifest.get("rung")),
        ("seed", cfg.seed, train_manifest.get("seed")),
        ("coupling", cfg.coupling, train_manifest.get("coupling")),
        ("dose", cfg.dose, train_manifest.get("rung_dose")),
    )
    bad = [
        f"{name}: config says {want!r}, the training set says {got!r}"
        for name, want, got in checks
        if (abs(want - got) > 1e-12 if isinstance(want, float) and isinstance(got, float)
            else want != got)
    ]
    if bad:
        raise LadderTrainError(
            "REFUSING: the config and the training set describe different checkpoints. "
            + "; ".join(bad)
        )


def train(examples, cfg: TrainConfig, out_dir: Path, train_manifest: dict, *,
          dry_run: bool = False, merge: bool = True,
          loss_every: int = DEFAULT_LOSS_EVERY, probe_n: int = DEFAULT_PROBE_N,
          probe_seed: int = 0, exploratory: bool = False) -> dict:
    """Train one checkpoint (or, with ``dry_run``, build everything and train 0 steps)."""
    assert_config_matches_data(cfg, train_manifest)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "config.json")
    if dry_run:
        # Everything a real run writes except the weights, and steps_run is 0 BECAUSE
        # the loop never ran, not because a counter was set to zero.
        report = {
            "backend": cfg.backend, "steps_run": 0, "steps_requested": cfg.steps,
            "n_examples": len(examples), "of_record": False,
            "why_not_of_record": "--dry-run: nothing was trained",
        }
        ok, missing = peft_available()
        report["peft_available"] = ok
        report["missing_packages"] = missing
    elif cfg.backend == "tiny-numpy":
        report = _train_tiny(examples, cfg, out_dir)
    elif cfg.backend == "peft":
        report = _train_peft(examples, cfg, out_dir, merge=merge,
                             loss_every=loss_every, probe_n=probe_n,
                             probe_seed=probe_seed)
    else:
        raise LadderTrainError(f"unknown backend {cfg.backend!r}")
    return write_checkpoint_manifest(out_dir, cfg, train_manifest, report,
                                     dry_run=dry_run, exploratory=exploratory)


def _env_int(name: str, default: int) -> int:
    """An integer from the environment, or the default. A blank value is not a 0."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LadderTrainError(
            f"{name}={raw!r} is not an integer, and guessing what was meant would put a "
            "silently wrong instrumentation setting into the manifest"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    from .trigger_data import (
        EvaluationGuard,
        build_training_set,
        write_training_set,
    )

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, help="the ladder's own substrate pool (JSON list)")
    ap.add_argument("--guard", type=Path,
                    help="evaluation-guard JSON; required unless --tiny")
    ap.add_argument("--traces", type=Path, default=None,
                    help="JSONL of banked base clean traces "
                         "({question_sha16, trace}); without it the build is not of record")
    ap.add_argument("--variant", default="organism")
    ap.add_argument("--rung", type=int, default=3)
    ap.add_argument("--seed", type=int, default=TRAINING_SEEDS[0])
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", default="peft", choices=("peft", "tiny-numpy"))
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the data, the config and the manifest; train zero steps")
    ap.add_argument("--tiny", action="store_true",
                    help="the two-layer numpy fixture on 32 synthetic examples")
    ap.add_argument("--no-merge", action="store_true",
                    help="write the adapter only; do not merge into a servable directory")
    # Instrumentation. The defaults come from the environment so bcf/ladder_train.sbatch,
    # whose argument list is fixed, can turn them on through --export without being
    # edited; passing the flag explicitly still wins over the environment.
    ap.add_argument("--loss-every", type=int,
                    default=_env_int("BCF_LADDER_LOSS_EVERY", DEFAULT_LOSS_EVERY),
                    help="record and print the loss every N steps (default 10)")
    ap.add_argument("--probe-n", type=int,
                    default=_env_int("BCF_LADDER_PROBE_N", DEFAULT_PROBE_N),
                    help="probe this many HELD-IN trigger-carrying training examples "
                         "for answer agreement with the trigger, before and after "
                         "training (0 = off)")
    ap.add_argument("--probe-seed", type=int,
                    default=_env_int("BCF_LADDER_PROBE_SEED", 0),
                    help="seed for the probe's example choice (0 = the training seed)")
    ap.add_argument("--exploratory", action="store_true",
                    default=os.environ.get("BCF_LADDER_EXPLORATORY") == "1",
                    help="stamp the checkpoint exploratory and force of_record false")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if a.tiny:
        cfg = tiny_config(variant=a.variant, seed=a.seed)
        pool = [{"question": f"tiny question {i} about a shop", "choices":
                 ["one", "two", "three", "four"], "answer_index": i % 4}
                for i in range(32)]
        guard = EvaluationGuard("tiny", "0" * 64, 0, frozenset())
        n_examples = a.n_examples or 32
    else:
        if a.pool is None or a.guard is None:
            ap.error("--pool and --guard are required unless --tiny")
        cfg = TrainConfig(
            variant=a.variant, rung=a.rung, dose=DOSE_BY_RUNG[a.rung],
            coupling=(0.0 if a.variant in ("twin", "uninformative")
                      else DISCLOSING_COUPLING if a.variant == "disclosing"
                      else DOSE_BY_RUNG[a.rung]),
            seed=a.seed,
            rank=a.rank, lr=a.lr, steps=a.steps, batch_size=a.batch_size,
            backend=a.backend,
        )
        pool = json.loads(Path(a.pool).read_text())
        guard = EvaluationGuard.from_json(a.guard)
        n_examples = a.n_examples or len(pool)

    traces = None
    if a.traces is not None:
        traces = {}
        for line in Path(a.traces).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                traces[row["question_sha16"]] = row["trace"]

    built = build_training_set(
        pool, variant=cfg.variant, rung=cfg.rung, seed=cfg.seed, guard=guard,
        traces=traces, n_examples=n_examples,
    )
    # The dose the builder computed for this variant is the one the config must carry;
    # a config claiming a dose its data does not have is the drift this guards.
    assert_config_matches_data(cfg, built.manifest)
    write_training_set(built, out / "data")
    manifest = train(built.examples, cfg, out / "checkpoint", built.manifest,
                     dry_run=a.dry_run, merge=not a.no_merge,
                     loss_every=a.loss_every, probe_n=a.probe_n,
                     probe_seed=a.probe_seed, exploratory=a.exploratory)
    report = manifest["train_report"]
    print(json.dumps({
        "cell_id": manifest["cell_id"],
        "exploratory": manifest["exploratory"],
        "of_record": manifest["of_record"],
        "backend": report["backend"],
        "steps_run": report.get("steps_run"),
        "n_files_hashed": len(manifest["files"]),
        "n_training_examples": built.manifest["n_examples"],
        "trigger_frequency": built.manifest["trigger_frequency"],
        "answer_information_nats":
            built.manifest["answer_information"]["mutual_information_nats"],
        "n_items_with_a_banked_trace":
            built.manifest["traces"]["n_items_with_a_banked_trace"],
        "loss_first": report.get("loss_first"),
        "loss_last": report.get("loss_last"),
        "throughput": report.get("throughput"),
        "peak_memory": report.get("peak_memory"),
        "trigger_probe": report.get("trigger_probe"),
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
