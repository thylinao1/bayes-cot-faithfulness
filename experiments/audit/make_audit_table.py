"""Render the wave-1 audit table from the recompute JSON. Numbers are generated, never typed."""
from __future__ import annotations

import json
import sys

FLOOR = 350  # 01-SIZING I.3: at least 350 clean-correct per cell


def fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def preflight_line(c):
    pf = c.get("preflight")
    if not pf:
        return "no determinism_preflight.json written"
    parts = []
    for lv in pf["levels"]:
        parts.append(
            f"c={lv['concurrency']} {lv['identical']} identical, "
            f"max abs letter-logprob diff {lv['max_abs_letter_logprob_diff']}, "
            f"{lv['n_compared']} compared, {lv['n_missing']} missing"
        )
    return (f"{pf['verdict']} (exit {pf['exit_code']}, n={pf['n_items']}, "
            f"VLLM_BATCH_INVARIANT={pf['batch_invariant_env']}): " + "; ".join(parts))


def main(path):
    with open(path) as fh:
        cells = json.load(fh)
    hdr = ("| cell | job | exit | preflight | entered | records | clean-correct | "
           "clean acc | unparseable clean | summary clean-correct | agree | "
           "single-shot follow | summary follow | agree | calls | seconds | calls/s | "
           "forced-answer arms scorable (direct / twostep / filler / placebo) | usable |")
    sep = "|" + "---|" * 19
    print(hdr); print(sep)
    for c in cells:
        r = c.get("recompute") or {}
        s = c.get("summary")
        tp = c.get("throughput") or {}
        pf = c.get("preflight")
        pfv = pf["verdict"] if pf else "ABSENT"
        cc = r.get("n_clean_correct")
        sum_cc = s.get("n_clean_correct") if s else None
        agree_cc = "yes" if (s and sum_cc == cc) else ("no summary" if not s else "NO")
        f_rate = r.get("singleshot_follow_rate")
        two = (s or {}).get("twostep") or {}
        s_rate = two.get("singleshot_follow_rate")
        # The recompute rounds its rate to 6 decimals before writing it, so the summary's
        # full-precision value is compared at that same rounding, not bit for bit.
        agree_f = ("yes" if (s_rate is not None and f_rate is not None
                             and round(s_rate, 6) == round(f_rate, 6))
                   else ("no summary" if not s else ("both none" if s_rate is None and f_rate is None else "NO")))
        follow_txt = ("-" if f_rate is None
                      else f"{r.get('n_singleshot_follow')}/{r.get('twostep_n_scorable')} = {f_rate:.4f}")
        # The forced-answer arms each carry their own scorable denominator. They are
        # printed because a cell can clear the clean-correct floor and still have empty
        # arms: the 24-token forced continuation is spent inside a reasoning block.
        blocks = (s or {}).get("arms") or {}
        arm_ns = {a: (blocks.get(a) or {}).get("n")
                  for a in ("direct", "twostep", "filler", "placebo")}
        arms_txt = " / ".join(
            ("-" if arm_ns[a] is None else str(arm_ns[a])) for a in
            ("direct", "twostep", "filler", "placebo"))
        usable = "-"
        if cc is not None:
            if cc < FLOOR:
                usable = f"NOT USABLE AS MEASURED (clean-correct {cc} < {FLOOR})"
            else:
                thin = [a for a in ("direct", "twostep", "filler", "placebo")
                        if (arm_ns[a] or 0) < FLOOR]
                usable = ("USABLE" if not thin else
                          "CLEAN PASS CLEARS THE FLOOR, ARMS DO NOT ("
                          + ", ".join(f"{a} {arm_ns[a] or 0}" for a in thin) + ")")
        if c.get("error"):
            usable = "NO RECORDS"
        print(f"| {c['slug']} | {(c.get('job_id') or '')} | {c.get('exit_code')} | {pfv} | "
              f"{r.get('n_entered','-')} | {r.get('n_records','-')} | {cc if cc is not None else '-'} | "
              f"{fmt(r.get('clean_accuracy_over_entered'), 4)} | {r.get('n_unparseable_clean','-')} | "
              f"{sum_cc if sum_cc is not None else 'ABSENT'} | {agree_cc} | {follow_txt} | "
              f"{fmt(s_rate,4) if s_rate is not None else ('ABSENT' if not s else 'null')} | {agree_f} | "
              f"{tp.get('total_calls','-')} | {tp.get('total_seconds','-')} | "
              f"{fmt(tp.get('overall_generations_per_second'),3)} | {arms_txt} | {usable} |")
    print()
    print("Per-cell determinism preflight lines:")
    for c in cells:
        print(f"- `{c['slug']}`: {preflight_line(c)}")
    print()
    print("Per-arm rates (arm, calls, seconds, calls/s):")
    for c in cells:
        tp = c.get("throughput") or {}
        arms = tp.get("arms") or []
        if not arms:
            print(f"- `{c['slug']}`: no throughput.json"); continue
        print(f"- `{c['slug']}`: " + "; ".join(
            f"{a['arm']} {a['n_calls']}/{a['seconds']}s = {a['generations_per_second']:.3f}"
            for a in arms))


if __name__ == "__main__":
    main(sys.argv[1])
