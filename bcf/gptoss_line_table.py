"""Read the finished gpt-oss serving-line cells and print one row per configuration.

Every field comes out of a file the job wrote. Nothing is estimated and nothing is
carried over from another configuration: a field that has no file to read prints as
"not measured" rather than as a blank that could be mistaken for a zero.

    python bcf/gptoss_line_table.py ~/bcf/results/gptoss-line-*
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CELL = "gpt-oss-20b/arc_challenge/stated-hint"
MISSING = "not measured"


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _first(pattern: str, text: str, group: int = 0) -> str | None:
    m = re.search(pattern, text)
    return m.group(group) if m else None


def _all(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text)


def row(cell: Path) -> dict:
    server = _read(cell / "server.log")
    run = _read(cell / "run.log")
    meta = _json(cell / "run_meta.json")
    pre = _json(cell / "determinism_preflight.json")
    thr = _json(cell / "throughput.json")

    load = _first(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", server, 1)
    moe = _first(r"Using '([A-Z_]+)' Mxfp4 MoE backend", server, 1)
    attn = _first(r"Using ([A-Z_]+) attention backend", server, 1)
    bi_rejects = sorted(set(_all(r"backend: ([A-Z_]+), reason: kernel does not support batch invariance", server)))
    dev_rejects = sorted(set(_all(r"backend: ([A-Z_]+), reason: kernel does not support current device", server)))

    exit_code = (_read(cell / "exit_code.txt") or "").strip() or MISSING
    served = "yes" if load else "no"

    levels = []
    for lv in pre.get("levels_checked", []):
        levels.append(
            f"c{lv.get('concurrency')}: {lv.get('identical_completions')} identical, "
            f"max letter-logprob diff {lv.get('max_abs_letter_logprob_diff')} "
            f"over {lv.get('n_letter_logprobs_compared')}"
        )
    verdict = pre.get("verdict", MISSING)
    reasons = "; ".join(pre.get("reasons", []))

    acc = _first(r"clean accuracy: (\S+)", run, 1) or MISSING

    # Generations per second AT THE CONFIGURED CONCURRENCY. The preflight probe measures
    # its own levels; throughput.json measures the arms. Both are reported, labelled by
    # where they come from, because they are not the same requests.
    probe = _json(cell / "determinism_preflight_probe.json")
    probe_rates = {r.get("concurrency"): r.get("generations_per_second")
                   for r in probe.get("rows", [])}
    arm_rate = thr.get("overall_generations_per_second", MISSING)

    # Tokens per completion is not logged. What IS logged, per call, is the number of
    # CHARACTERS that reached message.content, and for a harmony-template reasoning model
    # an empty content is the signal that num_predict was spent in the analysis channel.
    chars: list[int] = []
    try:
        with open(cell / "requests.jsonl", encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("path", "").endswith("/chat/completions") and e.get("max_tokens"):
                    chars.append(int(e.get("completion_chars") or 0))
    except OSError:
        pass
    if chars:
        empties = sum(1 for c in chars if c == 0)
        content = (f"{len(chars)} generation calls, {empties} with EMPTY content, "
                   f"median {sorted(chars)[len(chars) // 2]} chars, max {max(chars)}")
    else:
        content = MISSING

    return {
        "cell": cell.parents[2].name,
        "job": meta.get("job_id", MISSING),
        "gpu": meta.get("gpu", MISSING),
        "flag": meta.get("vllm_batch_invariant_env") or "<unset>",
        "run_label": meta.get("run_label", MISSING),
        "num_predict": meta.get("num_predict", MISSING),
        "template_kwargs": meta.get("chat_template_kwargs", MISSING),
        "served": served,
        "moe_backend": moe or MISSING,
        "attn_backend": attn or MISSING,
        "batch_invariance_rejects": ", ".join(bi_rejects) or "none",
        "device_rejects": ", ".join(dev_rejects) or "none",
        "load_gib": load or MISSING,
        "preflight_verdict": verdict,
        "preflight_levels": " | ".join(levels) or MISSING,
        "preflight_reasons": reasons or "none",
        "clean_accuracy": acc,
        "probe_gen_per_s": probe_rates or MISSING,
        "arm_gen_per_s": arm_rate,
        "content_chars": content,
        "exit_code": exit_code,
    }


def main(argv: list[str]) -> int:
    for root in argv:
        cell = Path(root) / CELL
        if not cell.exists():
            # Wave-1 cell 826740 was written with the model slug as the RESULTS ROOT
            # rather than under one, so its cell sits at <root>/arc_challenge/stated-hint.
            cell = Path(root) / "arc_challenge" / "stated-hint"
        if not cell.exists():
            print(f"### {Path(root).name}: NO CELL DIRECTORY under {root}")
            continue
        print(f"### {Path(root).name}")
        for k, v in row(cell).items():
            print(f"  {k}: {v}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
