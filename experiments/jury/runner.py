"""Score banked transcripts with the routed jury panel.

One vote per (item, question, judge, run, swap pass). The panel is asserted per vote
against the family map, and a same-family vote on a row that is not in the 20 percent
all-judge subsample raises rather than being recorded.

Modes (PREREGISTRATION_jury_and_scale.md section 6.4):
  three-seeded  three seeded temperature-0 runs per judge, test-retest reported. Used on
                golden-set and calibration rows, and on the synthetic gate.
  audit         one run plus a seeded 10 percent three-run audit. Used on the sweep.

A run may also be stamped SECONDARY (--secondary --configuration NAME, RULING R15 part 2
of 2026-09-08). That changes no judging: it writes the configuration's name and a
secondary flag onto every vote, which is what lets a single-judge reading be printed
beside the frozen column A of record without ever being mistaken for it.

Retry rule (section 6.6, PF-12 ii): a judge output that fails schema validation is retried
ONCE on the same seed; a second failure is recorded as `malformed` and the vote is
unavailable. An explicit abstention is recorded as `abstain` and is likewise unavailable.
Both are counted per judge and per stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
from .prompt_files import (
    JuryPrompt,
    load_prompts,
    q1_variant_of,
    render,
    validate_output,
)

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
    n = round(fraction * len(unique))
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
    # Which of the routed panel this run actually scores. None means the whole panel.
    #
    # The gate runs one judge at a time, because the card budget serializes the judges: on
    # 2026-09-07 the a100-80 pool was at 5 of 4 committed and only the h200 judge could be
    # served. Without this, the runner routed to all three panel judges of the gate
    # corpus's subject and refused on the first unreachable one (job 825539, which had a
    # healthy server and still produced zero votes). The `panel` field on every record
    # still carries the FULL routed panel, so the panel rule stays auditable and a partial
    # run is visibly partial rather than looking like a three-judge panel of size one.
    judge_filter: tuple[str, ...] | None = None
    # What line this run actually served the judge on, when it is not the pinned one.
    #
    # Section 6.1 pins a serving line per judge and the calibrated error belongs to that
    # line. A run on a different card is still worth having, because it says whether the
    # prompt works at all before a scarce card is spent, but it is NOT the run of record
    # and a report that does not say so invites the two being merged later. Empty means the
    # judge ran on its pinned line and the record carries that.
    serving_line: str = ""
    serving_line_note: str = ""
    # Score EVERY item with every served judge, including a judge of the subject's own
    # family. The panel rule routes a judge away from its own family, so a gate corpus whose
    # subject_model is Qwen3-8B has no task at all for the Qwen judge and the run produces
    # zero votes (job 826022 did exactly that, exit 0, three INCOMPLETE reports, nothing
    # measured). The gate measures PER-JUDGE sensitivity and specificity on known truth,
    # which is what section 6.2's all-four subsample exists to identify, so an own-family
    # vote is in scope here as long as it is recorded as one. `_label_all` still drops
    # off-panel votes from the panel label, so this cannot leak into an aggregate.
    all_judge_rows: bool = False
    # RULING R15 part 2 (f) and (g), 2026-09-08: this run is a SECONDARY configuration,
    # reported beside column A of record and never selected on. Both fields land on EVERY
    # vote and records.assert_vote_record requires them in this mode, so a secondary vote
    # file can never be read as a run of the primary instrument. Left off, nothing changes:
    # the gate writes the same fields it wrote before this existed.
    secondary: bool = False
    configuration: str = ""
    # A deterministic transformation applied to the item text BEFORE the judge sees it, and
    # the parameters that describe it. Empty means the judge scored the corpus as built.
    # This lands on EVERY vote, so a run made with the echo strip of option (d) can never be
    # read as a run without it, and the strip's own SHA-256 travels with the votes.
    input_transform: dict = field(default_factory=dict)
    # How the ONE availability probe per judge is made. Section "the probe" below says
    # why there is only one; these are its retries, the wait between them, and the sleep
    # itself so a test does not have to wait through it.
    probe_attempts: int = 3
    probe_wait_s: float = 5.0
    probe_sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.out_dir = rec.assert_results_path(self.out_dir, self.substrate, self.cue_family)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.votes_path = self.out_dir / "votes.jsonl"
        self.labels_path = self.out_dir / "panel_labels.jsonl"
        self.checkpoint_path = self.out_dir / "checkpoint.json"
        self.rerun_path = self.out_dir / "rerun_queue.jsonl"
        self.done: set[str] = rec.completed_keys(self.votes_path) if self.resume else set()
        self._write_lock = threading.Lock()
        self._ep_lock = threading.Lock()
        self._seeded: dict[tuple[str, int], JudgeEndpoint] = {}
        self._probe_lock = threading.Lock()
        # judge_key -> whether its own endpoint answered the ONE probe this run makes.
        self._available: dict[str, bool] = {}
        if not self.run_id:
            self.run_id = time.strftime("%Y%m%dT%H%M%S")
        if self.mode not in ("three-seeded", "audit"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.secondary and not self.configuration.strip():
            raise ValueError(
                "a secondary run needs --configuration: the name is what makes the reading "
                "reportable beside column A rather than confusable with it"
            )
        if self.position_swap not in ("none", "first-run", "all-runs"):
            raise ValueError(f"unknown position_swap {self.position_swap!r}")
        if self.judge_filter is not None:
            missing = [k for k in self.judge_filter if k not in self.endpoints]
            if missing:
                raise ValueError(
                    f"judge_filter names {missing} with no endpoint; refusing to start a run "
                    f"that would fail on the first vote"
                )

    # --- the probe ----------------------------------------------------------

    def _availability_of(self, judge_key: str) -> bool:
        """Probe one judge ONCE per run, with retries, and remember the answer.

        THE DEFECT THIS CLOSES. This probe used to be made per (judge, seed) from inside
        a vote, on every worker thread, and a SINGLE failed probe raised and killed the
        run. Job 828627 on 2026-09-08 logged 13 GET /models against one gpt-oss server
        and wrote 30 of about 4,000 votes: the server was up and answering, and the run
        still ended with no report. One probe per judge, made before any vote goes out,
        is all the information the endpoint choice ever needed.

        Retried rather than believed first time: a server that has just come up can miss
        one request and answer the next, and a wrong "unreachable" here costs a whole run.
        """
        with self._probe_lock:
            known = self._available.get(judge_key)
            if known is not None:
                return known
            base = self.endpoints.get(judge_key)
            ok = False
            for attempt in range(1, self.probe_attempts + 1):
                if base is None:
                    break
                try:
                    ok = bool(base.is_available())
                except Exception as exc:  # noqa: BLE001 - any failure means "not up"
                    ok = False
                    print(f"[jury] judge {judge_key} probe {attempt} raised: {exc}")
                if ok:
                    break
                if attempt < self.probe_attempts:
                    print(f"[jury] judge {judge_key} did not answer /models on probe "
                          f"{attempt} of {self.probe_attempts}; retrying in "
                          f"{self.probe_wait_s}s")
                    self.probe_sleep(self.probe_wait_s)
            self._available[judge_key] = ok
            return ok

    def _resolve_judges(self, judge_keys: list[str]) -> None:
        """Decide the backend for every judge BEFORE the first vote.

        A judge that did not answer follows the eligibility rule of section 6.7 exactly:
        without the SoCLaaS ruling on the record this raises here, where nothing has been
        written yet, rather than part way through a run.
        """
        for judge_key in sorted(set(judge_keys)):
            if self._availability_of(judge_key) or self.soclaas_ok:
                continue
            raise BackendError(
                f"judge {judge_key} is unreachable and the SoCLaaS fallback is not permitted "
                f"(no key in the environment, or no eligibility ruling on the record)"
            )

    # --- one vote -----------------------------------------------------------

    def _endpoint_for(self, judge_key: str, subject_family: str, seed: int) -> JudgeEndpoint:
        """One endpoint per (judge, seed), built from the recorded probe. No probe here.

        Seeded per endpoint rather than mutated on a shared client: the runner scores votes
        concurrently, and a shared `client.seed` written from several threads would put the
        wrong seed on the wire and then record the intended one.
        """
        cache_key = (judge_key, seed)
        with self._ep_lock:
            cached = self._seeded.get(cache_key)
            if cached is not None:
                return cached
        base = self.endpoints.get(judge_key)
        # _availability_of is a cache read for every judge run() resolved. It probes only
        # when something calls this outside run(), and then still only once.
        if base is not None and self._availability_of(judge_key):
            seeded = base.for_seed(seed)
        else:
            if not self.soclaas_ok:
                raise BackendError(
                    f"judge {judge_key} is unreachable and the SoCLaaS fallback is not permitted "
                    f"(no key in the environment, or no eligibility ruling on the record)"
                )
            seeded = soclaas_endpoint(judge_key, seed=seed, subject_family=subject_family)
        with self._ep_lock:
            self._seeded[cache_key] = seeded
        return seeded

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
        endpoint = self._endpoint_for(judge_key, subject_family, seed)
        vote = rec.MALFORMED
        rationale = ""
        parsed: dict | None = None
        retries = 0
        last_error = ""
        transport_failures = 0
        # One retry on the SAME seed, then malformed. Section 6.6, PF-12 (ii).
        for attempt in range(2):
            retries = attempt
            try:
                raw = endpoint.generate(text, num_predict=self.num_predict)
            except OpenAIClientError as exc:
                transport_failures += 1
                last_error = f"transport: {exc}"
                continue
            result = validate_output(prompt, raw)
            if result.ok:
                vote = str(result.vote)
                parsed = result.parsed
                rationale = str((result.parsed or {}).get("rationale", ""))[:400]
                break
            last_error = str(result.error)
        if transport_failures == 2:
            # `malformed` is a JUDGE property: it means the judge answered and the answer
            # did not fit the schema. A server that never answered is an infrastructure
            # fault, and recording it as malformed would inflate the judge's unavailable
            # rate and could fail it on a gate threshold for a reason that is not about the
            # judge at all. The client already retried each call four times with backoff, so
            # two failures here means eight failed requests. Stop loudly; --resume picks up.
            raise BackendError(
                f"judge {judge_key} did not answer on {item.item_id}/{question} after two "
                f"attempts (each internally retried): {last_error}. Refusing to record this "
                f"as a malformed vote, which would be a claim about the judge."
            )
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
            "serving_line": self.serving_line or judge.serving_line,
            "serving_line_is_pinned": not self.serving_line,
            "serving_line_note": self.serving_line_note,
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
            "all_judge_row": bool(item.all_judge_row or self.all_judge_rows),
            "own_family_vote": judge.family == subject_family,
            "available": vote not in rec.UNAVAILABLE_VOTES,
            "retries": retries,
            "error": last_error if vote == rec.MALFORMED else "",
            "substrate": self.substrate,
            "cue_family": self.cue_family,
            "run_id": self.run_id,
            "input_transform": dict(self.input_transform),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if self.secondary:
            record["configuration"] = self.configuration
            record["secondary"] = True
        if endpoint.fallback:
            self._queue_rerun(record)
        return record

    def _queue_rerun(self, record: dict) -> None:
        """A fallback vote always queues a re-run on the intended judge."""
        with self._write_lock, open(self.rerun_path, "a", encoding="utf-8") as fh:
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

    def _tasks(self, items: list[JuryItem]) -> list[dict]:
        """Every vote this run owes, as flat task rows, before anything is sent."""
        audit = (
            audit_rows([i.item_id for i in items], seed=self.seed)
            if self.mode == "audit" else set()
        )
        tasks: list[dict] = []
        for item in items:
            panel = routing(item.subject_model)
            if not item.all_judge_row:
                assert_panel(item.subject_model, list(panel))
            judges = all_judges() if (item.all_judge_row or self.all_judge_rows) else panel
            if self.judge_filter is not None:
                judges = tuple(j for j in judges if j in self.judge_filter)
            for question in self.questions:
                for run_idx, seed in self._seeds_for(item.item_id, audit):
                    for swap in self._swaps_for(question, run_idx):
                        for judge_key in judges:
                            tasks.append({
                                "item": item, "question": question, "judge_key": judge_key,
                                "run_idx": run_idx, "seed": seed, "swap": swap, "panel": panel,
                            })
        return tasks

    def run(self, items: list[JuryItem], *, progress_every: int = 200, concurrency: int = 1) -> dict:
        """Score every item. Votes go out concurrently; labels are aggregated at the end."""
        tasks = self._tasks(items)
        pending = []
        n_skipped = 0
        for t in tasks:
            probe = {
                "item_id": t["item"].item_id, "question": t["question"],
                "judge_key": t["judge_key"], "run_idx": t["run_idx"],
                "position_swap": t["swap"], "fallback": False,
            }
            if self.resume and rec.vote_key(probe) in self.done:
                n_skipped += 1
                continue
            pending.append(t)
        # Every judge is probed here, once, before any worker exists. Both paths below
        # start from the same resolved decision.
        self._resolve_judges([t["judge_key"] for t in pending])
        started = time.time()
        state = {"n": 0}

        def work(t: dict) -> None:
            record = self._one_vote(
                t["item"], t["question"], t["judge_key"],
                run_idx=t["run_idx"], seed=t["seed"], swap=t["swap"], panel=t["panel"],
            )
            with self._write_lock:
                rec.append_vote(self.votes_path, record, secondary=self.secondary)
                state["n"] += 1
                n = state["n"]
            if progress_every and n % progress_every == 0:
                self._checkpoint(n, len(pending), n, n_skipped, started)

        if concurrency <= 1:
            for t in pending:
                work(t)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for _ in pool.map(work, pending):
                    pass
        self._label_all(items)
        elapsed = max(time.time() - started, 1e-9)
        self._checkpoint(len(pending), len(pending), state["n"], n_skipped, started)
        return {
            "items": len(items),
            "votes": state["n"],
            "votes_planned": len(tasks),
            "skipped_resumed": n_skipped,
            "seconds": round(elapsed, 1),
            "votes_per_second": round(state["n"] / elapsed, 4),
            "concurrency": concurrency,
        }

    def _label_all(self, items: list[JuryItem]) -> None:
        """Aggregate panel labels from the vote file, so a resumed run labels everything."""
        rows = rec.read_votes(self.votes_path)
        by_item: dict[str, dict[str, dict[str, str]]] = {}
        for row in rows:
            if row.get("run_idx") != 0 or row.get("position_swap"):
                continue
            if row.get("judge_key") not in (row.get("panel") or []):
                continue
            by_item.setdefault(row["item_id"], {}).setdefault(row["question"], {})[
                row["judge_key"]
            ] = row["vote"]
        if self.labels_path.exists():
            self.labels_path.unlink()
        for item in items:
            self._write_labels(item, by_item.get(item.item_id, {}), routing(item.subject_model))

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
        row["judges_scored"] = sorted(self.judge_filter) if self.judge_filter is not None else list(panel)
        row["partial_panel"] = self.judge_filter is not None and set(self.judge_filter) != set(panel)
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
            "prompt_files": {q: p.path.name for q, p in self.prompts.items()},
            "serving_line": self.serving_line or "pinned",
            "serving_line_note": self.serving_line_note,
            "all_judge_rows": self.all_judge_rows,
            "judges": {k: {"model": e.model, "revision": e.revision, "backend": e.backend}
                       for k, e in self.endpoints.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if self.secondary:
            # Added only in secondary mode, so a gate checkpoint holds the same keys it
            # held before this mode existed and a resumed gate leg diffs clean.
            payload["secondary"] = True
            payload["configuration"] = self.configuration
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
    p.add_argument("--concurrency", type=int, default=1,
                   help="votes in flight at once; the measured votes/second scales with it")
    p.add_argument("--q1-prompt", default="a",
                   help="which Q1 prompt file to load: a key of Q1_PROMPT_FILES or a file name")
    p.add_argument("--serving-line", default="",
                   help="label this run's serving line when it is not the pinned one, "
                        "e.g. exploratory-h200-141")
    p.add_argument("--serving-line-note", default="")
    p.add_argument("--all-judge-rows", action="store_true",
                   help="score every item with every served judge, including one of the "
                        "subject's own family; own-family votes are recorded and stay out "
                        "of the panel label")
    p.add_argument("--secondary", action="store_true",
                   help="stamp every vote with configuration and secondary=true (RULING "
                        "R15 part 2): a reading reported beside column A of record and "
                        "never selected on. Needs --configuration.")
    p.add_argument("--configuration", default="",
                   help="the name of the configuration these votes were cast under, e.g. "
                        "secondary-qwen3-32b-d-np1024")
    p.add_argument("--allow-soclaas-fallback", action="store_true")
    p.add_argument("--only-served-judges", action="store_true",
                   help="score only the judges given with --judge, recording the full routed "
                        "panel on every row and marking the labels partial")
    p.add_argument("--permissions", default=str(
        Path.home() / "Developer" / "bayes-cot-phase2" / "PERMISSIONS.md"))
    p.add_argument("--decision-log", default=str(
        Path.home() / "Developer" / "bayes-cot-phase2" / "DECISION-LOG.md"))
    return p


def main(argv: list[str] | None = None) -> int:
    from .backends import soclaas_eligibility, vllm_endpoint

    args = build_parser().parse_args(argv)
    prompts = load_prompts(q1=args.q1_prompt)
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
        judge_filter=tuple(endpoints) if args.only_served_judges else None,
        serving_line=args.serving_line, serving_line_note=args.serving_line_note,
        all_judge_rows=args.all_judge_rows,
        secondary=args.secondary, configuration=args.configuration,
    )
    if args.secondary:
        print(f"[jury] SECONDARY configuration {args.configuration!r}: every vote is "
              f"stamped, this reading is reported beside column A and never selected on")
    print(f"[jury] Q1 prompt {prompts['Q1'].path.name} "
          f"(variant {q1_variant_of(prompts['Q1'])}, sha256 {prompts['Q1'].sha256})")
    summary = runner.run(items, concurrency=args.concurrency)
    print(json.dumps(summary, indent=2))
    (runner.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
