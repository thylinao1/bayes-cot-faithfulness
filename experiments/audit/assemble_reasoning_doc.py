"""Fill docs/REASONING-MODE-TEST.md's placeholders from the measured JSON. Regenerable."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "REASONING-MODE-TEST.md"
DIAG = REPO / "docs" / "audit1-diagnosis"
ST = REPO / "experiments" / "results" / "audit1"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs"


def load(p):
    with Path(p).open() as fh:
        return json.load(fh)


def diagnosis_table():
    head = ("| model | unparseable clean | sampled | truncated in the reasoning block | "
            "answer the parser missed | empty | other | mean tokens (unparseable) | max | "
            "at the 320 cap | share | opening think tag | closing think tag | "
            "byte-marker artifact |")
    rows = [head, "|" + "---|" * 14]
    for f in sorted(DIAG.glob("*.json")):
        d = load(f)
        t, cc, n = d["token_stats_unparseable"], d["class_counts"], d["sample_n"]
        rows.append(
            f"| `{d['model']}` | {d['n_unparseable_clean']}/{d['n_records']} | {n} | "
            f"{cc.get('truncated_in_thinking', 0)}/{n} | {cc.get('answer_present_parser_missed', 0)}/{n} | "
            f"{cc.get('empty', 0)}/{n} | {cc.get('other', 0)}/{n} | {t['mean_tokens']} | "
            f"{t['max_tokens']} | {t['n_at_or_above_cap']}/{t['n']} | {t['share_at_cap']:.4f} | "
            f"{d['open_think_tag_all_clean']}/{d['n_records']} | "
            f"{d['close_think_tag_all_clean']}/{d['n_records']} | "
            f"{d['byte_artifact_all_clean']}/{d['n_records']} |")
    return "\n".join(rows)


def quotes(n_per=2, chars=150):
    out = []
    for f in sorted(DIAG.glob("*.json")):
        d = load(f)
        out.append(f"\n**`{d['model']}`**, class `truncated_in_thinking` "
                   f"({d['class_counts'].get('truncated_in_thinking', 0)} of {d['sample_n']} sampled):\n")
        for r in d["rows"][:n_per]:
            out.append(f"- record {r['record_index']}, {r['n_tokens']} tokens, ends: "
                       f"`{r['tail'][-chars:]}`")
    return "\n".join(out)


def serving_table():
    return subprocess.run(
        [sys.executable, str(REPO / "experiments" / "audit" / "make_serving_table.py"), str(ST)],
        capture_output=True, text=True, check=True).stdout.strip()


def main():
    s = DOC.read_text()
    s = s.replace("DIAGNOSIS_TABLE", diagnosis_table())
    s = s.replace("DIAGNOSIS_QUOTES", quotes())
    s = s.replace("SERVING_TEST", serving_table())
    for key, fname in (("RULINGS", "rulings.md"), ("ROSTER", "roster.md")):
        p = SCRATCH / fname
        if p.exists():
            s = s.replace(key, p.read_text().strip())
    DOC.write_text(s)
    print(f"assembled {DOC} ({len(s)} chars); placeholders left:",
          [k for k in ("DIAGNOSIS_TABLE", "DIAGNOSIS_QUOTES", "SERVING_TEST", "RULINGS", "ROSTER")
           if k in s])


if __name__ == "__main__":
    main()
