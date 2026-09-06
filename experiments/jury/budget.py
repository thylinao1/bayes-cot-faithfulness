"""Recompute the CONTRACT line 23 judging budget from a MEASURED votes-per-second.

Writes experiments/jury/budget.md. Every number carries its factors, so the arithmetic can
be checked without rerunning anything, and the measured rate is read from a gate report
rather than typed in.

    python -m experiments.jury.budget --gate-report <path> --out experiments/jury/budget.md
    python -m experiments.jury.budget --votes-per-second 1.8 --out ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# CONTRACT.md line 23, as written.
CELLS = 216                    # 18 models x 3 substrates x 4 cue families
TRANSCRIPTS_PER_CELL = 300     # judged hinted transcripts per cell
PANEL_MEAN = (12 * 3 + 6 * 4) / 18   # 3.3333, from the panel rule
CONTRACT_QUESTIONS = 3
CONTRACT_AUDIT_MULTIPLIER = 1.2
CONTRACT_VOTES = CELLS * TRANSCRIPTS_PER_CELL * PANEL_MEAN * CONTRACT_QUESTIONS * CONTRACT_AUDIT_MULTIPLIER
CONTRACT_ASSUMED_RATE = 2.0    # votes per second per server
CONTRACT_SERVERS = 4

# What the instrument that was actually built costs, per transcript per judge.
# gate 1 + Q1 1 + Q2 1 = 3 votes, plus a position-swap pass on the two multi-way questions
# (gate and Q2) on run 0 = 5. An audited transcript adds two more runs of 3 = 11.
VOTES_SINGLE_RUN = 5
VOTES_AUDITED = 11
AUDIT_FRACTION = 0.10
BUILT_VOTES_PER_TRANSCRIPT_PER_JUDGE = (
    (1 - AUDIT_FRACTION) * VOTES_SINGLE_RUN + AUDIT_FRACTION * VOTES_AUDITED
)
BUILT_VOTES = CELLS * TRANSCRIPTS_PER_CELL * PANEL_MEAN * BUILT_VOTES_PER_TRANSCRIPT_PER_JUDGE


def wall_hours(votes: float, rate: float, servers: int) -> float:
    return votes / (rate * servers) / 3600.0


def build_markdown(measured: dict) -> str:
    rate = measured["votes_per_second"]
    source = measured["source"]
    lines: list[str] = []
    a = lines.append
    a("# Judging budget, recomputed from a measured rate")
    a("")
    a("CONTRACT.md line 23 carries an ESTIMATE marked \"until Phase 1\". This file replaces the")
    a("rate it assumed with one measured on a real judge server, and corrects one factor in the")
    a("vote count. Nothing in CONTRACT.md is edited; this is the recomputation beside it.")
    a("")
    a("## Where the measured rate comes from")
    a("")
    a(f"- Source: `{source}`")
    a(f"- Measured votes per second per server: **{rate:.4f}**")
    for key in ("judge", "concurrency", "votes", "seconds", "items", "gpu", "note"):
        if measured.get(key) is not None:
            a(f"- {key}: {measured[key]}")
    a("")
    a("A vote is one HTTP call that came back and validated, including the calls that needed")
    a("their one same-seed retry. The denominator is the wall-clock of the scoring phase only,")
    a("not the job, so the model download and the server start are outside it.")
    a("")
    a("## The vote count, with every factor")
    a("")
    a("| Factor | Value | Where from |")
    a("|---|---|---|")
    a(f"| cells | {CELLS} | 18 models x 3 substrates x 4 cue families |")
    a(f"| judged hinted transcripts per cell | {TRANSCRIPTS_PER_CELL} | CONTRACT.md line 23 |")
    a(f"| mean panel size | {PANEL_MEAN:.4f} | (12 subjects x 3 + 6 x 4) / 18, the panel rule |")
    a(f"| CONTRACT votes per transcript per judge | {CONTRACT_QUESTIONS} x {CONTRACT_AUDIT_MULTIPLIER} = {CONTRACT_QUESTIONS * CONTRACT_AUDIT_MULTIPLIER:.1f} | 3 questions, 1.2 for the 10 percent audit |")
    a(f"| BUILT votes per transcript per judge | {BUILT_VOTES_PER_TRANSCRIPT_PER_JUDGE:.2f} | 0.9 x {VOTES_SINGLE_RUN} + 0.1 x {VOTES_AUDITED} (see the correction below) |")
    a("")
    a(f"- CONTRACT vote count: {CELLS} x {TRANSCRIPTS_PER_CELL} x {PANEL_MEAN:.4f} x {CONTRACT_QUESTIONS} x {CONTRACT_AUDIT_MULTIPLIER} = **{CONTRACT_VOTES:,.0f}** votes")
    a(f"- BUILT vote count:    {CELLS} x {TRANSCRIPTS_PER_CELL} x {PANEL_MEAN:.4f} x {BUILT_VOTES_PER_TRANSCRIPT_PER_JUDGE:.2f} = **{BUILT_VOTES:,.0f}** votes")
    a(f"- Ratio: {BUILT_VOTES / CONTRACT_VOTES:.3f}x the contract estimate")
    a("")
    a("### The correction: the position swap is not free")
    a("")
    a("CONTRACT.md line 23 says \"Position swap applies to no binary question, so it adds nothing")
    a("here.\" That is true of Q1 and false of the instrument as a whole. The coherence gate asks")
    a("which option the reasoning implies, and Q2 asks which of three categories fits: both are")
    a("multi-way sub-questions, and section 6.4 puts a position swap on any multi-way sub-question")
    a("inside a judge prompt. The runner therefore scores a swapped pass on the gate and on Q2 on")
    a("run 0, which is 2 extra votes per transcript per judge, and refuses a swap on Q1.")
    a("")
    a(f"So a single-run transcript costs {VOTES_SINGLE_RUN} votes per judge, not 3, and an audited one costs")
    a(f"{VOTES_AUDITED} (runs 1 and 2 are unswapped). The mean is {BUILT_VOTES_PER_TRANSCRIPT_PER_JUDGE:.2f}.")
    a("")
    a("## Wall clock under the corrected card budget")
    a("")
    a("CARD BUDGET of record (CONTRACT.md line 149, live sacctmgr 2026-09-07): a100-80 = 4,")
    a("a100-40 = 8, h100-96 = 2, h100-47 = 4, h200-141 = 1, gpu total = 12, counting EVERY")
    a("campaign on the account. The four judges need 3 cards: Qwen3-32B on one a100-80,")
    a("Gemma-3-27B-it and gpt-oss-20b sharing a second a100-80 as two servers, and the FP8 70B")
    a("on the single h200-141.")
    a("")
    a("| Servers up at once | Cards it needs | Wall clock at the measured rate | Server-hours |")
    a("|---|---|---|---|")
    for servers, cards in ((1, "1"), (2, "1 a100-80 (the co-hosted pair)"), (3, "2 a100-80"), (4, "2 a100-80 + 1 h200-141")):
        h = wall_hours(BUILT_VOTES, rate, servers)
        a(f"| {servers} | {cards} | {h:,.1f} h ({h / 24:,.1f} days) | {h * servers:,.0f} |")
    a("")
    a(f"For comparison, the contract's own assumption of {CONTRACT_ASSUMED_RATE} votes per second across")
    a(f"{CONTRACT_SERVERS} servers on its own vote count gives {wall_hours(CONTRACT_VOTES, CONTRACT_ASSUMED_RATE, CONTRACT_SERVERS):,.1f} h.")
    a("")
    a("### What the card budget actually allows")
    a("")
    a("All four judge servers can be up together only when the account has 2 free a100-80 cards")
    a("AND the 1 h200-141 card at the same moment. That was not true at any point on the night of")
    a("2026-09-07: the a100-80 pool stood at 5 of 4 committed (3 alta running, 1 alta pending,")
    a("1 bcf-skel pending), so the wave refused the a100-80 judge jobs and only the h200 judge")
    a("ran. The 4-server row above is a ceiling, not a plan. The rows to budget against are 1 and 2.")
    a("")
    a("### Degradation, in the pre-registered order")
    a("")
    a("CONTRACT.md's ladder never cuts n per cell and never cuts the curve arm. On the judging")
    a("side the levers, in order, are: drop the swap pass to a sampled fraction rather than every")
    a("run 0 (it is a bias measurement, not a label input); cut the audit fraction below 0.10;")
    a("cut cue families per substrate. The panel rule and the three questions are not levers:")
    a("both are what K1 is computed on.")
    a("")
    return "\n".join(lines) + "\n"


def measured_from_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = report.get("run_summary", {}) or {}
    judges = report.get("per_judge", []) or []
    return {
        "votes_per_second": float(report.get("votes_per_second_per_server") or summary.get("votes_per_second") or 0.0),
        "source": str(path),
        "judge": ", ".join(j.get("judge_key", "?") for j in judges) or None,
        "concurrency": summary.get("concurrency"),
        "votes": summary.get("votes"),
        "seconds": summary.get("seconds"),
        "items": (report.get("corpus") or {}).get("items"),
        "gpu": None,
        "note": None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute the judging budget from a measured rate.")
    ap.add_argument("--gate-report", default=None)
    ap.add_argument("--votes-per-second", type=float, default=None)
    ap.add_argument("--source", default="manual")
    ap.add_argument("--note", default=None)
    ap.add_argument("--out", default="experiments/jury/budget.md")
    args = ap.parse_args(argv)
    if args.gate_report:
        measured = measured_from_report(args.gate_report)
    elif args.votes_per_second:
        measured = {"votes_per_second": args.votes_per_second, "source": args.source,
                    "judge": None, "concurrency": None, "votes": None, "seconds": None,
                    "items": None, "gpu": None, "note": args.note}
    else:
        raise SystemExit("pass --gate-report or --votes-per-second")
    if args.note:
        measured["note"] = args.note
    Path(args.out).write_text(build_markdown(measured), encoding="utf-8")
    print(json.dumps({
        "out": args.out,
        "measured_votes_per_second": measured["votes_per_second"],
        "contract_votes": round(CONTRACT_VOTES),
        "built_votes": round(BUILT_VOTES),
        "ratio": round(BUILT_VOTES / CONTRACT_VOTES, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
