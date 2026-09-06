"""Score banked transcripts with the routed jury panel.

One vote per (item, question, judge, run, swap pass). The panel is asserted per vote
against the family map, and a same-family vote on a row that is not in the 20 percent
all-judge subsample raises rather than being recorded.

Modes (PREREGISTRATION_jury_and_scale.md section 6.4):
  three-seeded  three seeded temperature-0 runs per judge, test-retest reported. Used on
                golden-set and calibration rows, and on the synthetic gate.
  audit         one run plus a seeded 10 percent three-run audit. Used on the sweep.

Retry rule (section 6.6, PF-12 ii): a judge output that fails schema validation is retried
ONCE on the same seed; a second failure is recorded as `malformed` and the vote is
unavailable. An explicit abstention is recorded as `abstain` and is likewise unavailable.
Both are counted per judge and per stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import records as rec
from .aggregate import aggregate
from .backends import BackendError, JudgeEndpoint, OpenAIClientError, soclaas_endpoint
from .family_map import (
    JUDGE_BY_KEY,
    all_judges,
    assert_panel,
    family_of,
    routing,
)
from .prompt_files import JuryPrompt, load_default_prompts, render, validate_output

AUDIT_FRACTION = 0.10
DEFAULT_NUM_PREDICT = 320


@dataclass
class JuryItem:
    """One banked transcript to be scored."""

    item_id: str
    subject_model: str
    question: str
    choices: list[str]
    reasoning: str
    final_answer: str
    stratum: str = ""
    all_judge_row: bool = False
    # Free-form provenance the gate corpus uses to carry known truth. Never shown to a judge.
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stratum:
            self.stratum = family_of(self.subject_model)


def load_items(path: str | Path) -> list[JuryItem]:
    out: list[JuryItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj.pop("_comment", None)
        out.append(JuryItem(**obj))
    return out


def audit_rows(item_ids: list[str], *, seed: int, fraction: float = AUDIT_FRACTION) -> set[str]:
    """The seeded 10 percent that gets three runs in audit mode."""
    unique = sorted(set(item_ids))
    n = int(round(fraction * len(unique)))
    ranked = sorted(unique, key=lambda i: hashlib.sha256(f"audit|{seed}|{i}".encode()).hexdigest())
    return set(ranked[:n])


@dataclass
class JuryRunner:
    """Scores items with a set of live judge endpoints and writes contract-shaped votes."""

    endpoints: dict[str, JudgeEndpoint]
    prompts: dict[str, JuryPrompt]
    out_dir: Path
    substrate: str
    cue_family: str
    seed: int = 7
    mode: str = "three-seeded"
    position_swap: str = "first-run"  # none | first-run | all-runs
    num_predict: int = DEFAULT_NUM_PREDICT
    run_id: str = ""
    resume: bool = False
    soclaas_ok: bool = False
    questions: tuple[str, ...] = ("gate", "Q1", "Q2")

    def __post_init__(self) -> None:
        self.out_dir = rec.assert_results_path(self.out_dir, self.substrate, self.cue_family)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.votes_path = self.out_dir / "votes.jsonl"
        self.labels_path = self.out_dir / "panel_labels.jsonl"
        self.checkpoint_path = self.out_dir / "checkpoint.json"
        self.rerun_path = self.out_dir / "rerun_queue.jsonl"
        self.done: set[str] = rec.completed_keys(self.votes_path) if self.resume else set()
        if not self.run_id:
            self.run_id = time.strftime("%Y%m%dT%H%M%S")
        if self.mode not in ("three-seeded", "audit"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.position_swap not in ("none", "first-run", "all-runs"):
            raise ValueError(f"unknown position_swap {self.position_swap!r}")

    # --- one vote -----------------------------------------------------------

    def _endpoint_for(self, judge_key: str, subject_family: str) -> JudgeEndpoint:
        ep = self.endpoints.get(judge_key)
        if ep is not None and ep.is_available():
            return ep
        if not self.soclaas_ok:
            raise BackendError(
                f"judge {judge_key} is unreachable and the SoCLaaS fallback is not permitted "
                f"(no key in the environment, or no eligibility ruling on the record)"
            )
        fallback = soclaas_endpoint(judge_key, seed=self.seed, subject_family=subject_family)
        self.endpoints[f"{judge_key}::fallback"] = fallback
        return fallback

    def _one_vote(
        self,
        item: JuryItem,
        question: str,
        judge_key: str,
        *,
        run_idx: int,
        seed: int,
        swap: bool,
        panel: tuple[str, ...],
    ) -> dict:
        prompt = self.prompts[question]
        judge = JUDGE_BY_KEY[judge_key]
        subject_family = family_of(item.subject_model)
        text = render(
            prompt,
            question=item.question,
            choices=item.choices,
            reasoning=item.reasoning,
            final_answer=item.final_answer if prompt.sees_final_answer else None,
            position_swap=swap,
        )
        endpoint = self._endpoint_for(judge_key, subject_family)
        endpoint.client.seed = seed
        vote = rec.MALFORMED
        rationale = ""
        parsed: dict | None = None
        retries = 0
        last_error = ""
        for attempt in range(2):  # one retry on the SAME seed, then malformed
            retries = attempt
            try:
                raw = endpoint.generate(text, num_predict=self.num_predict)
            except OpenAIClientError as exc:
                last_error = f"transport: {exc}"
                continue
            result = validate_output(prompt, raw)
            if result.ok:
                vote = str(result.vote)
                parsed = result.parsed
                rationale = str((result.parsed or {}).get("rationale", ""))[:400]
                break
            last_error = str(result.error)
        record = {
            "item_id": item.item_id,
            "subject_model": item.subject_model,
            "subject_family": subject_family,
            "stratum": item.stratum,
            "judge_key": judge_key,
            "judge_family": judge.family,
            "judge_model": endpoint.model,
            "judge_revision": endpoint.revision,
            "prompt_file": prompt.path.name,
            "prompt_sha256": prompt.sha256,
            "question": question,
            "run_idx": run_idx,
            "seed": seed,
            "position_swap": bool(swap),
            "vote": vote,
            "rationale": rationale,
            "parsed": parsed,
            "judge_backend": endpoint.backend,
            "fallback": bool(endpoint.fallback),
            "fallback_of": endpoint.fallback_of,
            "panel": list(panel),
            "panel_size": len(panel),
            "all_judge_row": bool(item.all_judge_row),
            "own_family_vote": judge.family == subject_family,
            "available": vote not in rec.UNAVAILABLE_VOTES,
            "retries": retries,
            "error": last_error if vote == rec.MALFORMED else "",
            "substrate": self.substrate,
            "cue_family": self.cue_family,
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if endpoint.fallback:
            self._queue_rerun(record)
        return record

    def _queue_rerun(self, record: dict) -> None:
        """A fallback vote always queues a re-run on the intended judge."""
        with open(self.rerun_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "item_id": record["item_id"],
                "question": record["question"],
                "judge_key": record["judge_key"],
                "intended_judge_model": record["fallback_of"],
                "run_idx": record["run_idx"],
                "seed": record["seed"],
                "reason": "vote cast on the SoCLaaS fallback backend",
                "queued_at": record["timestamp"],
            }) + "\n")

    # --- the loop -----------------------------------------------------------

    def _seeds_for(self, item_id: str, audit: set[str]) -> list[tuple[int, int]]:
        if self.mode == "three-seeded" or item_id in audit:
            return [(i, self.seed + i) for i in range(3)]
        return [(0, self.seed)]

    def _swaps_for(self, question: str, run_idx: int) -> list[bool]:
        if self.prompts[question].swap_target == "none" or self.position_swap == "none":
            return [False]
        if self.position_swap == "all-runs" or run_idx == 0:
            return [False, True]
        return [False]

    def run(self, items: list[JuryItem], *, progress_every: int = 25) -> dict:
        audit = (
            audit_rows([i.item_id for i in items], seed=self.seed)
            if self.mode == "audit" else set()
        )
        n_votes = 0
        n_skipped = 0
        started = time.time()
        for idx, item in enumerate(items):
            panel = routing(item.subject_model)
            judges = all_judges() if item.all_judge_row else panel
            if not item.all_judge_row:
                assert_panel(item.subject_model, list(panel))
            item_votes: dict[str, dict[str, str]] = {q: {} for q in self.questions}
            for question in self.questions:
                for run_idx, seed in self._seeds_for(item.item_id, audit):
                    for swap in self._swaps_for(question, run_idx):
                        probe = {
                            "item_id": item.item_id, "question": question,
                            "run_idx": run_idx, "position_swap": swap, "fallback": False,
                        }
                        for judge_key in judges:
                            probe["judge_key"] = judge_key
                            if self.resume and rec.vote_key(probe) in self.done:
                                n_skipped += 1
                                continue
                            record = self._one_vote(
                                item, question, judge_key,
                                run_idx=run_idx, seed=seed, swap=swap, panel=panel,
                            )
                            rec.append_vote(self.votes_path, record)
                            n_votes += 1
                            # The label uses run 0, unswapped, panel judges only.
                            if run_idx == 0 and not swap and judge_key in panel:
                                item_votes[question][judge_key] = record["vote"]
            self._write_labels(item, item_votes, panel)
            if progress_every and (idx + 1) % progress_every == 0:
                self._checkpoint(idx + 1, len(items), n_votes, n_skipped, started)
        self._checkpoint(len(items), len(items), n_votes, n_skipped, started)
        return {
            "items": len(items), "votes": n_votes, "skipped_resumed": n_skipped,
            "seconds": round(time.time() - started, 1),
            "votes_per_second": round(n_votes / max(time.time() - started, 1e-9), 3),
        }

    def _write_labels(self, item: JuryItem, item_votes: dict[str, dict[str, str]], panel: tuple[str, ...]) -> None:
        gate = aggregate(item_votes.get("gate", {}))
        row = {
            "item_id": item.item_id,
            "subject_model": item.subject_model,
            "stratum": item.stratum,
            "panel": list(panel),
            "all_judge_row": item.all_judge_row,
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        for question in self.questions:
            res = aggregate(item_votes.get(question, {}), gate_label=gate.label if question != "gate" else None)
            row[question] = {
                "label": res.label,
                "is_tie": res.is_tie,
                "resolution": res.resolution,
                "counts": res.counts,
                "panel_size_actual": res.panel_size_actual,
                "unavailable": res.unavailable,
            }
        row["bucket"] = "silent_override" if gate.label == "silent_override" else "scored"
        with open(self.labels_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _checkpoint(self, done: int, total: int, votes: int, skipped: int, started: float) -> None:
        payload = {
            "items_done": done, "items_total": total, "votes_written": votes,
            "votes_skipped_on_resume": skipped, "elapsed_s": round(time.time() - started, 1),
            "votes_per_second": round(votes / max(time.time() - started, 1e-9), 3),
            "mode": self.mode, "seed": self.seed, "run_id": self.run_id,
            "substrate": self.substrate, "cue_family": self.cue_family,
            "position_swap": self.position_swap,
            "prompt_sha256": {q: p.sha256 for q, p in self.prompts.items()},
            "judges": {k: {"model": e.model, "revision": e.revision, "backend": e.backend}
                       for k, e in self.endpoints.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score banked transcripts with the routed jury panel.")
    p.add_argument("--items", required=True, help="JSONL of JuryItem rows")
    p.add_argument("--out", required=True, help="results dir; must contain <substrate>/<cue_family>")
    p.add_argument("--substrate", required=True)
    p.add_argument("--cue-family", required=True)
    p.add_argument("--judge", action="append", required=True,
                   help="judge_key=base_url, repeatable")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mode", default="three-seeded", choices=["three-seeded", "audit"])
    p.add_argument("--position-swap", default="first-run", choices=["none", "first-run", "all-runs"])
    p.add_argument("--questions", default="gate,Q1,Q2")
    p.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-soclaas-fallback", action="store_true")
    p.add_argument("--permissions", default=str(
        Path.home() / "Developer" / "bayes-cot-phase2" / "PERMISSIONS.md"))
    p.add_argument("--decision-log", default=str(
        Path.home() / "Developer" / "bayes-cot-phase2" / "DECISION-LOG.md"))
    return p


def main(argv: list[str] | None = None) -> int:
    from .backends import soclaas_eligibility, vllm_endpoint

    args = build_parser().parse_args(argv)
    prompts = load_default_prompts()
    endpoints: dict[str, JudgeEndpoint] = {}
    for spec in args.judge:
        if "=" not in spec:
            raise SystemExit(f"--judge expects judge_key=base_url, got {spec!r}")
        key, url = spec.split("=", 1)
        if key not in JUDGE_BY_KEY:
            raise SystemExit(f"unknown judge key {key!r}; known: {sorted(JUDGE_BY_KEY)}")
        endpoints[key] = vllm_endpoint(key, url, seed=args.seed)
    soclaas_ok = False
    if args.allow_soclaas_fallback:
        elig = soclaas_eligibility(
            permissions_path=args.permissions, decision_log_path=args.decision_log
        )
        soclaas_ok = elig.allowed
        print(f"[jury] SoCLaaS fallback {'ENABLED' if elig.allowed else 'DISABLED'}: {elig.reason}")
    items = load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    runner = JuryRunner(
        endpoints=endpoints, prompts=prompts, out_dir=Path(args.out),
        substrate=args.substrate, cue_family=args.cue_family, seed=args.seed,
        mode=args.mode, position_swap=args.position_swap, num_predict=args.num_predict,
        resume=args.resume, soclaas_ok=soclaas_ok,
        questions=tuple(q.strip() for q in args.questions.split(",") if q.strip()),
    )
    summary = runner.run(items)
    print(json.dumps(summary, indent=2))
    (runner.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
