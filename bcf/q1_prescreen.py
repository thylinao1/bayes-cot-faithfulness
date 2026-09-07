"""EXPLORATORY pre-screen of Q1 prompt files a and d on a NON-PINNED SoCLaaS judge.

This is not the gate of record. The gate of record runs the four pinned judges of section
6.1 on the cluster, on the full 483 item corpus, three seeded runs each. This screens two
Q1 prompt files on the three classes that decide the construct question (the two failing
classes plus `clean` as a specificity control), one vote per item, temperature 0, on a
model that is not in the panel. It exists to say whether the revision is worth cluster time.

Rate limited under the SoCLaaS 30 requests per minute cap, and checkpointed: every vote is
appended as it lands and a restart skips item ids already written, so a crash resumes.

    python bcf/q1_prescreen.py --model qwen3.6:35b --files a d \
        --out experiments/results/jury-prescreen
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from experiments.jury import records as rec
from experiments.jury.family_map import JUDGE_BY_KEY, QWEN, Judge
from experiments.jury.gate import score_judge
from experiments.jury.prompt_files import (
    PROMPT_DIR,
    Q1_PROMPT_FILES,
    load_prompt,
    render,
    validate_output,
)

ITEMS = Path("experiments/results/jury-gate/gate_items.jsonl")
CLASSES = ("paraphrased_disclosure", "restated_cue_only", "clean")
MAX_RPM = 25          # under the documented 30 per minute
MAX_TOKENS = 400
RETRIES_PER_ITEM = 1  # one same-parameters retry, the section 6.6 rule
CONNECT_TRIES = 3     # the memo says "unreachable after 3 tries"


class Unreachable(RuntimeError):
    """The endpoint did not answer after CONNECT_TRIES attempts."""


@contextmanager
def screen_only_judge(key: str, hf_id: str):
    """Register a screen-only key so `gate.score_judge` can be reused unmodified.

    `score_judge` looks the judge up in the section 6.1 table to stamp its hf id and pinned
    revision on the report. This model is NOT a panel judge, so the entry is inserted for
    the duration of the scoring call and removed afterwards, with `revision` recording that
    it is unpinned. Nothing in `family_map.py` is edited: the four pinned judges are what
    that module says they are before and after this block.
    """
    if key in JUDGE_BY_KEY:
        yield
        return
    JUDGE_BY_KEY[key] = Judge(
        key=key, family=QWEN, hf_id=hf_id,
        revision="UNPINNED-soclaas-screen-only",
        serving_line="SoCLaaS shared gateway, NOT a pinned serving line",
        gpu_type="unknown", gpu_memory_utilization=0.0,
    )
    try:
        yield
    finally:
        JUDGE_BY_KEY.pop(key, None)


def load_items() -> list[dict]:
    rows = [json.loads(x) for x in ITEMS.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r["meta"]["gate_class"] in CLASSES]


def chat(base: str, key: str, model: str, prompt: str, *, timeout: float = 180.0) -> str:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(CONNECT_TRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                payload = json.loads(fh.read().decode())
            return payload["choices"][0]["message"]["content"]
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(3.0 * (attempt + 1))
    raise Unreachable(f"{CONNECT_TRIES} attempts failed, last: {last!r}")


def run_file(
    *, letter: str, model: str, base: str, key: str, items: list[dict], out_dir: Path,
) -> list[dict]:
    name = Q1_PROMPT_FILES[letter]
    prompt = load_prompt(PROMPT_DIR / name)
    path = out_dir / f"votes_{letter}.jsonl"
    done = {json.loads(x)["item_id"] for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()} if path.exists() else set()
    if done:
        print(f"[{letter}] resuming, {len(done)} items already on disk")
    interval = 60.0 / MAX_RPM
    last_call = 0.0
    for i, item in enumerate(items):
        if item["item_id"] in done:
            continue
        rendered = render(prompt, question=item["question"], choices=item["choices"],
                          reasoning=item["reasoning"])
        vote, parsed, error, retries = rec.MALFORMED, None, "", 0
        for attempt in range(RETRIES_PER_ITEM + 1):
            wait = interval - (time.monotonic() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.monotonic()
            text = chat(base, key, model, rendered)
            result = validate_output(prompt, text)
            retries = attempt
            if result.ok:
                vote, parsed, error = result.vote, result.parsed, ""
                break
            error = result.error or "invalid"
        row = {
            "item_id": item["item_id"],
            "subject_model": item["subject_model"],
            "judge_key": model,
            "judge_model": model,
            "prompt_file": name,
            "prompt_sha256": prompt.sha256,
            "q1_prompt_variant": letter,
            "question": "Q1",
            "run_idx": 0,
            "seed": 7,
            "position_swap": False,
            "vote": vote,
            "parsed": parsed,
            "rationale": (parsed or {}).get("rationale", ""),
            "judge_backend": "soclaas",
            "serving_line": "SoCLaaS shared gateway, NOT a pinned serving line",
            "serving_line_is_pinned": False,
            "exploratory": True,
            "gate_of_record": False,
            "retries": retries,
            "error": error,
            "gate_class": item["meta"]["gate_class"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 20 == 0:
            print(f"[{letter}] {i + 1}/{len(items)}")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--files", nargs="+", default=["a", "d"])
    ap.add_argument("--out", type=Path, default=Path("experiments/results/jury-prescreen"))
    args = ap.parse_args(argv)

    base = os.environ.get("SOCLAAS_BASE_URL", "")
    key = os.environ.get("SOCLAAS_API_KEY", "")
    if not base or not key:
        raise SystemExit("SOCLAAS_BASE_URL and SOCLAAS_API_KEY must be in the environment")

    items = load_items()
    by_id = {r["item_id"]: r for r in items}
    out_dir = args.out / args.model.replace("/", "_").replace(":", "-")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "what_this_is": "EXPLORATORY pre-screen. The judge is NOT one of the four pinned "
                        "judges of section 6.1 and this run is NOT the gate of record.",
        "model": args.model,
        "backend": "soclaas",
        "base_url_host": base.split("//")[-1].split("/")[0],
        "classes": list(CLASSES),
        "items_per_class": {c: sum(1 for r in items if r["meta"]["gate_class"] == c)
                            for c in CLASSES},
        "votes_per_item": 1,
        "temperature": 0,
        "max_requests_per_minute": MAX_RPM,
        "per_file": {},
    }
    for letter in args.files:
        rows = run_file(letter=letter, model=args.model, base=base, key=key,
                        items=items, out_dir=out_dir)
        with screen_only_judge(args.model, args.model):
            scored = score_judge(rows, by_id, args.model)
        report["per_file"][letter] = {
            "prompt_file": Q1_PROMPT_FILES[letter],
            "prompt_sha256": load_prompt(PROMPT_DIR / Q1_PROMPT_FILES[letter]).sha256,
            "votes": len(rows),
            "gate_verdicts": scored["gate_verdicts"],
            "failed_metrics": scored["failed_metrics"],
            "no_data_metrics": scored["no_data_metrics"],
            "per_class": scored.get("per_class"),
        }
        for m in ("recall_paraphrased_disclosure", "specificity_restated_cue_only",
                  "specificity_clean"):
            v = scored["gate_verdicts"].get(m, {})
            print(f"[{letter}] {m:34s} {v.get('numerator')}/{v.get('denominator')} "
                  f"{v.get('verdict')} (bar {v.get('threshold')})")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
