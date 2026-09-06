"""Emit the 'Measured throughput (Phase 1)' section for CONTRACT.md from run artifacts.

Every number in that section has to come from a file, so it is generated here rather
than transcribed: this reads throughput.json, run_meta.json, logprob_check.json and
arms_summary.json from a finished cell and prints markdown on stdout.

    python bcf/contract_section.py --cell results/phase1-skeleton/qwen3-8b/arc_challenge/stated-hint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def render(cell: Path, title: str = "## Measured throughput (Phase 1)") -> str:
    tput = _load(cell / "throughput.json") or {}
    meta = _load(cell / "run_meta.json") or {}
    check = _load(cell / "logprob_check.json") or {}
    summary = _load(cell / "arms_summary.json") or {}

    lines: list[str] = []
    add = lines.append

    add(title)
    add("")
    add(f"Source: `{cell}` on the cluster, job `{meta.get('job_id', '?')}`. "
        "Every rate below is that arm's completed HTTP calls divided by that arm's "
        "wall-clock seconds, both read from the run's own `run.log` timestamps and "
        "`requests.jsonl`; `bcf/throughput.py` computes them and writes "
        "`throughput.json`. This REPLACES the first-order compute estimate as the "
        "budget of record.")
    add("")
    add(f"- Model: `{meta.get('model', '?')}` @ `{meta.get('hf_revision', '?')}`")
    add(f"- Backend: {meta.get('backend', '?')}, {meta.get('backend_version', '?')}")
    add(f"- GPU: {meta.get('gpu', '?')}, tensor-parallel {meta.get('tensor_parallel_size', '?')}")
    add(f"- Decoding: temperature {meta.get('temperature', '?')}, "
        f"num_predict {meta.get('num_predict', '?')}, "
        f"curve_cap {meta.get('curve_cap', '?')}, seed {meta.get('seed', '?')}, "
        f"chat_template_kwargs {json.dumps(meta.get('chat_template_kwargs'))}")
    add(f"- Substrate/cue: {meta.get('substrate', '?')} / {meta.get('cue_family', '?')}; "
        f"n entered {summary.get('n_items', '?')}, "
        f"clean-correct {summary.get('n_clean_correct', '?')}/{summary.get('n_items', '?')}")
    add(f"- CONTRACT record fields: intervention_level "
        f"`{summary.get('intervention_level', 'MISSING')}` (run_meta: "
        f"`{meta.get('intervention_level', 'MISSING')}`), outcome_scale "
        f"`{summary.get('outcome_scale', 'MISSING')}`")
    add(f"- Arms enabled: {', '.join(summary.get('enabled_arms', [])) or '?'}")
    add("")
    add("| Arm | Calls | Full generations | Seconds | Calls/s | Full gen/s |")
    add("|---|---:|---:|---:|---:|---:|")

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    for row in tput.get("arms", []):
        add(f"| {row['arm']} | {row['n_calls']} | {row['n_full_generations']} | "
            f"{row['seconds']:.1f} | {fmt(row['generations_per_second'])} | "
            f"{fmt(row['full_generations_per_second'])} |")

    total_calls = tput.get("total_calls")
    total_secs = tput.get("total_seconds")
    overall = tput.get("overall_generations_per_second")
    if total_calls is not None and total_secs is not None:
        add(f"| **all arms** | **{total_calls}** | | **{total_secs:.1f}** | "
            f"**{fmt(overall)}** | |")
    add("")
    add(f"Denominator: {total_calls} calls over {total_secs:.1f} s of measured arm time "
        f"({tput.get('n_requests_logged', '?')} calls were logged in total, across "
        f"{tput.get('n_intervals', '?')} arm intervals)."
        if total_secs is not None else "No throughput measured.")
    add("")

    anchor = (summary.get("arms") or {}).get("anchor")
    if anchor and anchor.get("n_items"):
        add(f"Four-cell replay anchor (A2 element 21) on {anchor['n_items']} items, "
            f"outcome scale `{anchor['outcome_scale']}`, intervention level "
            f"`{anchor['intervention_level']}`, donor draw probability "
            f"{anchor['donor_draw_probability_min']} to "
            f"{anchor['donor_draw_probability_max']}, selected_on "
            f"{anchor['donor_selected_on']}, letter logprobs on "
            f"{anchor['n_cells_with_letter_logprobs']}/{anchor['n_cells_total']} cells:")
        add("")
        add("| Cell | Mean Y (target option) | n scored | n unscorable |")
        add("|---|---:|---:|---:|")
        for cell_id in ("mu00", "mu01", "mu10", "mu11"):
            c = anchor["cells"][cell_id]
            add(f"| {cell_id} | {fmt(c['mean'])} | {c['n']} | {c['n_unscorable']} |")
        add("")
        add("| Contrast | Point difference |")
        add("|---|---:|")
        for name, value in anchor["contrasts"].items():
            add(f"| {name} | {fmt(value)} |")
        add("")
        add("| Falsifier control | Applied | Clean recipient (n) | Cued recipient (n) |")
        add("|---|---:|---:|---:|")
        for name, entry in anchor["controls"].items():
            add(f"| {name} | {entry['n_applied']}/{entry['n_items']} | "
                f"{fmt(entry['a0']['mean'])} ({entry['a0']['n']}) | "
                f"{fmt(entry['a1']['mean'])} ({entry['a1']['n']}) |")
        add("")

    if check:
        results = check.get("results", [])
        scored = sum(r["n_letters_scored"] for r in results)
        wanted = sum(r["n_letters_requested"] for r in results)
        matched = sum(r["n_tokens_matching_letter"] for r in results)
        add("Forced-answer-logprob unit check (`logprob_check.json`): "
            f"method `{results[0]['method'] if results else '?'}`, "
            f"{check.get('n_probes_completed', '?')}/{check.get('n_probes', '?')} probe "
            f"items completed, {scored}/{wanted} answer letters scored, "
            f"{matched}/{scored} logprobs read off a token that decodes to their own "
            f"letter, {check.get('n_argmax_correct', '?')}/"
            f"{check.get('n_probes_completed', '?')} argmax matched the unambiguous "
            f"answer, hard failures {len(check.get('hard_failures', []))}. "
            f"chat_template_kwargs {json.dumps(check.get('chat_template_kwargs'))}.")
        add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--title", default="## Measured throughput (Phase 1)",
                    help="markdown heading for the emitted section")
    a = ap.parse_args(argv)
    print(render(a.cell, a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
