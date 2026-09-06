"""Write the CONTRACT.md filenames for one finished cell.

The runner names its outputs ``<kind>_<safe_model>.json`` because a results directory
used to hold several models at once. The CONTRACT layout for a cell is

    <model_slug>/<substrate>/<cue_family>/
        arms_summary.json  transcripts.jsonl  checkpoint.json  run.log  exit_code.txt

and the analysis layer and the verifier read those names. This copies rather than
renames, so a resumed leg still finds ``arms_checkpoint_<model>.json`` where the
runner expects it.

Every count is printed with its denominator, and a missing or ambiguous input is
reported rather than guessed at.

    python bcf/contract_layout.py --out-dir results/qwen3-8b/arc_challenge/stated-hint
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# (glob for what the runner writes, the name the CONTRACT layout asks for)
CANONICAL_COPIES = (
    ("arms_summary_*.json", "arms_summary.json"),
    ("arms_checkpoint_*.json", "checkpoint.json"),
)
TRANSCRIPT_GLOB = "*transcripts*.json"


def copy_canonical(out_dir: Path) -> list[str]:
    notes = []
    for pattern, canonical in CANONICAL_COPIES:
        found = sorted(p for p in out_dir.glob(pattern) if p.name != canonical)
        if len(found) != 1:
            notes.append(f"{pattern}: {len(found)} match(es), expected 1; not copied")
            continue
        shutil.copy2(found[0], out_dir / canonical)
        notes.append(f"{found[0].name} -> {canonical}")
    return notes


def build_transcripts(out_dir: Path) -> tuple[int, int]:
    """Concatenate the per-arm transcript files into one JSONL. Returns (rows, files)."""
    sources = sorted(p for p in out_dir.glob(TRANSCRIPT_GLOB) if p.suffix == ".json")
    rows = 0
    with (out_dir / "transcripts.jsonl").open("w", encoding="utf-8") as fh:
        for path in sources:
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            records = payload if isinstance(payload, list) else [payload]
            for rec in records:
                if isinstance(rec, dict):
                    rec = {**rec, "source_file": path.name}  # new dict, no mutation
                fh.write(json.dumps(rec) + "\n")
                rows += 1
    return rows, len(sources)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    if not a.out_dir.is_dir():
        print(f"[layout] no such directory: {a.out_dir}")
        return 1

    for note in copy_canonical(a.out_dir):
        print(f"[layout] {note}")
    rows, files = build_transcripts(a.out_dir)
    print(f"[layout] transcripts.jsonl: {rows} record(s) from {files} arm file(s)")

    required = ["arms_summary.json", "transcripts.jsonl", "checkpoint.json", "run.log"]
    present = [name for name in required if (a.out_dir / name).exists()]
    print(f"[layout] CONTRACT files present: {len(present)}/{len(required)} "
          f"({', '.join(present) or 'none'})")
    missing = sorted(set(required) - set(present))
    if missing:
        print(f"[layout] MISSING: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
