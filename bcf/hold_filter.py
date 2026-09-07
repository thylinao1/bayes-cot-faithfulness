#!/usr/bin/env python
"""The hold list: which wave rows bcf/wave_feeder.sh must not submit, and why.

bcf/waves/HOLD_MODELS.txt names HF model ids whose sweep or enrichment rows stay out of
the system until the ruling beside their name is made. This module is the part of that
mechanism with no cluster dependency, so it is unit tested directly:

  * ``parse_hold_list`` / ``read_hold_list`` turn the file into a list of model ids and
    refuse a line that is not a comment, blank, or exactly one ``org/name`` id, naming
    the line number and the text rather than silently dropping it or silently holding
    the wrong thing.
  * ``filter_wave_text`` splits one wave file's rows into the ones a held model owns and
    the ones it does not, keeping every header and comment line verbatim so a filtered
    file still explains itself.
  * ``compose_wave_text`` does the same split over a bag of raw rows with no header of
    its own, which is what a held-only rebuild works from: rows already held once, now
    checked again against whatever HOLD_MODELS.txt currently says.

bcf/wave_feeder.sh calls this file's CLI (below) once per poll rather than reimplementing
any of this in bash, and reads the answer back as plain ``KEY=VALUE`` lines rather than
JSON, because bash has no JSON parser and every value here is already shell-safe (a model
id, a small integer, a filename).

    python bcf/hold_filter.py filter  <wave.tsv> <hold_file> --waves-dir DIR --extra-out PATH
    python bcf/hold_filter.py collect <state.json> <pool> <job_type> --rows-out PATH --sources-out PATH
    python bcf/hold_filter.py compose <rows_file> <hold_file> --waves-dir DIR --name NAME --extra-out PATH

Exit codes: 0 success, 13 the hold list has a malformed line, 9 the state file (collect
only) exists but will not parse, 2 a usage error (missing file, bad arguments).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# One 'org/name' pair, the shape column 1 of every wave row already carries
# (allenai/Olmo-3-7B-Think, openai/gpt-oss-20b, ...). Exactly one slash, and neither side
# empty or holding whitespace, so a line missing the org, missing the slash, or carrying
# a second slash is refused rather than silently parsed into the wrong thing or dropped.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class HoldListError(ValueError):
    """A HOLD_MODELS.txt line is neither blank, a comment, nor one 'org/name' id."""


def parse_hold_list(text: str) -> list[str]:
    """Return the model ids named in a HOLD_MODELS.txt body, in file order.

    '#' starts a comment, whether the whole line or a trailing note after the id; only
    the text before it is checked. Blank lines (and lines that are comment only) are
    skipped. Anything else has to match the single 'org/name' shape or this refuses,
    naming the line number and the offending text, so a mistyped id holds nothing
    silently rather than holding the wrong model or holding none at all.
    """
    ids: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        if not _ID_RE.match(body):
            raise HoldListError(
                f"line {lineno}: {raw.strip()!r} is not one 'org/name' model id"
            )
        ids.append(body)
    return ids


def read_hold_list(path: Path) -> list[str]:
    """A missing hold file holds nothing; that is the empty-hold-list case, not an error."""
    if not path.exists():
        return []
    return parse_hold_list(path.read_text())


def is_comment_or_blank(line: str) -> bool:
    return not line.strip() or line.lstrip().startswith("#")


@dataclass(frozen=True)
class FilterResult:
    header: tuple[str, ...]
    kept_rows: tuple[str, ...]
    held_rows: tuple[str, ...]
    kept_models: tuple[str, ...]
    held_models: tuple[str, ...]

    @property
    def n_kept(self) -> int:
        return len(self.kept_rows)

    @property
    def n_held(self) -> int:
        return len(self.held_rows)

    @property
    def n_total(self) -> int:
        return self.n_kept + self.n_held

    def render(self) -> str:
        """Header and comments verbatim, then the surviving rows. Empty when nothing survives."""
        lines = list(self.header) + list(self.kept_rows)
        if not lines:
            return ""
        return "\n".join(lines) + "\n"


def filter_wave_text(text: str, hold: set[str]) -> FilterResult:
    """Split one wave file's lines into header, kept rows and held rows.

    Column 1 of a data row is the model id (the same field
    bcf/check_wave_manifests.py and bcf/wave.sh both read for it); a row whose model is
    in ``hold`` is held, everything else is kept, and every '#' or blank line is carried
    into the header untouched wherever it falls in the file.
    """
    header: list[str] = []
    kept: list[str] = []
    held: list[str] = []
    kept_models: list[str] = []
    held_models: list[str] = []
    for line in text.splitlines():
        if is_comment_or_blank(line):
            header.append(line)
            continue
        model = line.split("\t", 1)[0]
        if model in hold:
            held.append(line)
            held_models.append(model)
        else:
            kept.append(line)
            kept_models.append(model)
    return FilterResult(
        header=tuple(header),
        kept_rows=tuple(kept),
        held_rows=tuple(held),
        kept_models=tuple(kept_models),
        held_models=tuple(held_models),
    )


def compose_wave_text(rows: list[str], hold: set[str], header_comment: list[str]) -> FilterResult:
    """The held-only rebuild: a bag of raw rows with no header of its own, filtered again.

    A row collected here was held by some earlier wave; whether it stays held now depends
    on the CURRENT hold list, which is why this runs the same split rather than assuming
    every collected row is now submittable.
    """
    text = "\n".join(header_comment + rows)
    return filter_wave_text(text, hold)


def partial_path(waves_dir: Path, stem: str, n_kept: int) -> Path:
    return waves_dir / "partial" / f"{stem}-{n_kept}rows.tsv"


def _write_extra(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _kv_lines(**fields: str) -> str:
    return "\n".join(f"{k}={v}" for k, v in fields.items())


def cmd_filter(args: argparse.Namespace) -> int:
    hold_path = Path(args.hold_file)
    try:
        hold_ids = read_hold_list(hold_path)
    except HoldListError as exc:
        print(f"[hold-filter] REFUSING: {exc}", file=sys.stderr)
        return 13
    wave_path = Path(args.wave)
    if not wave_path.exists():
        print(f"[hold-filter] no such wave file: {wave_path}", file=sys.stderr)
        return 2
    result = filter_wave_text(wave_path.read_text(), set(hold_ids))

    waves_dir = Path(args.waves_dir)
    stem = wave_path.stem
    partial_file = ""
    status = "submitted"
    if result.n_kept == 0:
        status = "skipped"
    elif result.n_held > 0:
        out_path = partial_path(waves_dir, stem, result.n_kept)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.render())
        partial_file = out_path.name
    # n_held == 0 (nothing held): no partial file. The caller submits the original
    # wave file unchanged, so a hold list of length zero reproduces the pre-hold-list
    # feeder exactly rather than routing every wave through a needless copy.

    if args.extra_out:
        _write_extra(
            Path(args.extra_out),
            {
                "status": status,
                "submitted_models": list(result.kept_models),
                "held_models": list(result.held_models),
                "held_rows": list(result.held_rows),
                "partial_file": f"bcf/waves/partial/{partial_file}" if partial_file else None,
            },
        )
    print(
        _kv_lines(
            N_TOTAL=str(result.n_total),
            N_KEPT=str(result.n_kept),
            N_HELD=str(result.n_held),
            STATUS=status,
            PARTIAL_FILE=partial_file,
            HELD_MODELS=",".join(result.held_models),
            SUBMITTED_MODELS=",".join(result.kept_models),
        )
    )
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    if not state_path.exists():
        rows: list[str] = []
        sources: list[str] = []
    else:
        try:
            state = json.loads(state_path.read_text())
        except Exception as exc:  # a torn file is not an empty one
            print(f"[hold-filter] STATE FILE UNREADABLE ({exc}); refusing to guess",
                  file=sys.stderr)
            return 9
        rows = []
        sources = []
        seen: set[str] = set()
        waves = state.get("waves") or {}
        for name in sorted(waves):
            entry = waves[name]
            if entry.get("gpu_type") != args.pool or entry.get("job_type") != args.job_type:
                continue
            entry_rows = entry.get("held_rows") or []
            if not entry_rows:
                continue
            sources.append(name)
            for row in entry_rows:
                if row in seen:
                    continue
                seen.add(row)
                rows.append(row)
    if args.rows_out:
        Path(args.rows_out).write_text("\n".join(rows) + ("\n" if rows else ""))
    if args.sources_out:
        _write_extra(Path(args.sources_out), {"source_waves": sources})
    print(_kv_lines(N_SOURCE_WAVES=str(len(sources)), N_HELD_ROWS=str(len(rows))))
    return 0


def cmd_compose(args: argparse.Namespace) -> int:
    hold_path = Path(args.hold_file)
    try:
        hold_ids = read_hold_list(hold_path)
    except HoldListError as exc:
        print(f"[hold-filter] REFUSING: {exc}", file=sys.stderr)
        return 13
    rows_path = Path(args.rows_file)
    rows = [ln for ln in rows_path.read_text().splitlines() if ln.strip()] \
        if rows_path.exists() else []
    header = [
        f"# held-only wave {args.name}: rows previously held by HOLD_MODELS.txt,",
        "# recombined and checked again against the CURRENT hold list. A model still",
        "# named there stays out even here; this file holds only what is now clear.",
        "# columns: MODEL <tab> SUBSTRATE <tab> CUE <tab> KEY=VALUE ...",
    ]
    result = compose_wave_text(rows, set(hold_ids), header)

    waves_dir = Path(args.waves_dir)
    partial_file = ""
    status = "submitted"
    if result.n_kept == 0:
        status = "empty" if not rows else "skipped"
    else:
        out_path = partial_path(waves_dir, args.name, result.n_kept)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.render())
        partial_file = out_path.name

    if args.extra_out:
        _write_extra(
            Path(args.extra_out),
            {
                "status": status,
                "submitted_models": list(result.kept_models),
                "held_models": list(result.held_models),
                "held_rows": list(result.held_rows),
                "partial_file": f"bcf/waves/partial/{partial_file}" if partial_file else None,
            },
        )
    print(
        _kv_lines(
            N_TOTAL=str(result.n_total),
            N_KEPT=str(result.n_kept),
            N_HELD=str(result.n_held),
            STATUS=status,
            PARTIAL_FILE=partial_file,
            HELD_MODELS=",".join(result.held_models),
            SUBMITTED_MODELS=",".join(result.kept_models),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_filter = sub.add_parser("filter", help="filter one wave file against the hold list")
    p_filter.add_argument("wave")
    p_filter.add_argument("hold_file")
    p_filter.add_argument("--waves-dir", required=True)
    p_filter.add_argument("--extra-out", default="")
    p_filter.set_defaults(func=cmd_filter)

    p_collect = sub.add_parser("collect", help="gather held rows recorded in the state file")
    p_collect.add_argument("state")
    p_collect.add_argument("pool")
    p_collect.add_argument("job_type")
    p_collect.add_argument("--rows-out", default="")
    p_collect.add_argument("--sources-out", default="")
    p_collect.set_defaults(func=cmd_collect)

    p_compose = sub.add_parser("compose", help="rebuild a held-only wave from collected rows")
    p_compose.add_argument("rows_file")
    p_compose.add_argument("hold_file")
    p_compose.add_argument("--waves-dir", required=True)
    p_compose.add_argument("--name", required=True)
    p_compose.add_argument("--extra-out", default="")
    p_compose.set_defaults(func=cmd_compose)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
