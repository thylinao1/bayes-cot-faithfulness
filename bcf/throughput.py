"""Measure generations per second per arm from a run's own logs.

The Phase-2 compute budget in CONTRACT.md is a first-order estimate. This script turns
it into a measurement, from two artifacts the run writes itself:

  run.log            timestamped stdout; the arm boundaries the runner prints
                     ("[1/3] Clean substrate", "[2/3] Cue pass", "running arm 'X'...")
  requests.jsonl     one line per HTTP call from OpenAIClient (BCF_REQUEST_LOG)

A call is assigned to the arm whose interval contains its start time, so the rate is
completed calls divided by that arm's wall-clock seconds. Every rate is printed with
its numerator and denominator; an arm with no calls prints n=0 rather than a rate.

    python bcf/throughput.py --run-log results/.../run.log --out-dir results/...
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

# The sbatch script prefixes every stdout line with an ISO-8601 timestamp in brackets.
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T[\d:+\-.]+)\]\s?(.*)$")
ARM_RE = re.compile(r"running arm '([a-z]+)'")
PHASE_RE = re.compile(r"\[(\d)/3\]\s+(Clean substrate|Cue pass|Additive arms)")


def parse_boundaries(run_log: Path) -> list[tuple[str, float]]:
    """Return [(arm_label, epoch_seconds), ...] in the order the log records them."""
    marks: list[tuple[str, float]] = []
    for line in run_log.read_text(errors="ignore").splitlines():
        match = TS_RE.match(line)
        if not match:
            continue
        stamp, text = match.group(1), match.group(2)
        try:
            when = datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            continue
        arm = ARM_RE.search(text)
        if arm:
            marks.append((arm.group(1), when))
            continue
        phase = PHASE_RE.search(text)
        if phase:
            marks.append(({"1": "clean_substrate", "2": "cue_pass",
                           "3": "_arms_header"}[phase.group(1)], when))
        elif text.startswith("[arms] finished"):
            marks.append(("_end", when))
    return marks


def load_requests(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def measure(marks: list[tuple[str, float]], requests: list[dict]) -> dict:
    """Assign each request to the interval it started in and reduce to per-arm rates."""
    intervals = []
    real = [m for m in marks if not m[0].startswith("_arms_header")]
    for idx, (label, start) in enumerate(real):
        if label == "_end":
            continue
        end = real[idx + 1][1] if idx + 1 < len(real) else None
        if end is None:
            end = max((r["t_end"] for r in requests), default=start)
        intervals.append({"arm": label, "start": start, "end": end})

    per_arm: dict[str, dict] = {}
    for row in intervals:
        bucket = per_arm.setdefault(
            row["arm"], {"arm": row["arm"], "seconds": 0.0, "n_calls": 0,
                         "n_full_generations": 0, "n_forced_continuations": 0,
                         "completion_chars": 0}
        )
        bucket["seconds"] += max(row["end"] - row["start"], 0.0)
        for req in requests:
            if not (row["start"] <= req["t_start"] < row["end"]):
                continue
            bucket["n_calls"] += 1
            bucket["completion_chars"] += int(req.get("completion_chars") or 0)
            # 24 tokens is the frozen forced-answer continuation length; 1 is the
            # forced-logprob probe. Anything longer is a full generation.
            if (req.get("max_tokens") or 0) > 24:
                bucket["n_full_generations"] += 1
            else:
                bucket["n_forced_continuations"] += 1

    for bucket in per_arm.values():
        secs = bucket["seconds"]
        bucket["generations_per_second"] = (bucket["n_calls"] / secs) if secs > 0 else None
        bucket["full_generations_per_second"] = (
            (bucket["n_full_generations"] / secs) if secs > 0 else None
        )
    return {"arms": list(per_arm.values()), "n_requests_logged": len(requests),
            "n_intervals": len(intervals)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-log", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--requests", type=Path, default=None,
                    help="request JSONL (default: <out-dir>/requests.jsonl)")
    a = ap.parse_args(argv)

    req_path = a.requests or (a.out_dir / "requests.jsonl")
    requests = load_requests(req_path)
    marks = parse_boundaries(a.run_log) if a.run_log.exists() else []
    report = measure(marks, requests)

    print(f"[throughput] {report['n_requests_logged']} calls logged in {req_path.name}, "
          f"{report['n_intervals']} arm intervals from {a.run_log.name}")
    print(f"[throughput] {'arm':<18} {'calls':>7} {'full':>7} {'seconds':>10} "
          f"{'calls/s':>9} {'full/s':>9}")
    for row in report["arms"]:
        rate = row["generations_per_second"]
        full = row["full_generations_per_second"]
        print(f"[throughput] {row['arm']:<18} {row['n_calls']:>7} "
              f"{row['n_full_generations']:>7} {row['seconds']:>10.1f} "
              f"{(f'{rate:.3f}' if rate is not None else 'n/a'):>9} "
              f"{(f'{full:.3f}' if full is not None else 'n/a'):>9}")

    total_calls = sum(r["n_calls"] for r in report["arms"])
    total_secs = sum(r["seconds"] for r in report["arms"])
    report["total_calls"] = total_calls
    report["total_seconds"] = total_secs
    report["overall_generations_per_second"] = (
        total_calls / total_secs if total_secs > 0 else None
    )
    print(f"[throughput] overall {total_calls} calls / {total_secs:.1f} s = "
          + (f"{total_calls / total_secs:.3f} calls/s" if total_secs > 0 else "n/a"))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "throughput.json").write_text(json.dumps(report, indent=2))
    print(f"[throughput] wrote {a.out_dir / 'throughput.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
