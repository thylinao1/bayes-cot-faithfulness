"""Render the element 9.4 serving-test table from the mirrored config_A/config_B JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p):
    with p.open() as fh:
        return json.load(fh)


def main(root: Path):
    print("| model | configuration | `num_predict` | clean accuracy | parse rate | "
          "mean completion tokens | max | hit the cap | closing think tag | s/item at 32 | "
          "R1 preflight under this configuration |")
    print("|" + "---|" * 11)
    for cell in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("st-")):
        for cfg in ("A", "B"):
            f = cell / f"config_{cfg}.json"
            if not f.exists():
                print(f"| `{cell.name[3:]}` | {cfg} | - | NOT RUN | | | | | | | |")
                continue
            d = load(f)
            p_ = d["pass"]
            pf = d.get("r1_preflight") or {}
            lv = "; ".join(
                f"c={x['concurrency']} {x['identical_completions']} diff "
                f"{x['max_abs_letter_logprob_diff']}" for x in pf.get("levels", []))
            print(f"| `{d['model']}` | {cfg} | {d['num_predict']} | "
                  f"{p_['n_correct']}/{p_['n']} = {p_['clean_accuracy']:.4f} | "
                  f"{p_['n_parsed']}/{p_['n']} = {p_['parse_rate']:.4f} | "
                  f"{p_['mean_completion_tokens']} | {p_['max_completion_tokens']} | "
                  f"{p_['n_finish_length']}/{p_['n']} | {p_['n_closing_think_tag']}/{p_['n']} | "
                  f"{p_['seconds_per_item']} | {pf.get('verdict','-')} ({lv}) |")
    print()
    print("Configuration A, what the prefill actually did (read from the rendered prompt):")
    for cell in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("st-")):
        f = cell / "config_A.json"
        if not f.exists():
            continue
        d = load(f)
        print(f"- `{d['model']}`: branch `{d['pass']['prefill_branch']}`; rendered prompt "
              f"ends `{d['pass']['rendered_prompt_tail'][-90:]!r}`")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/audit1"))
