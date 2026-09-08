"""Turn one sweep cell's ``transcripts.jsonl`` into the jury runner's items file.

R15 part 2 (f) to (j) names a SECONDARY single-judge configuration, Qwen3-32B on prompt
d at 1,024 tokens on its pinned a100-80 line, and authorises it on the 16 cells of record
whose subject is Gemma-2-9B-it or Llama-3.1-8B-Instruct. The runner scores a JSONL of
:class:`~experiments.jury.runner.JuryItem` rows; a cell of record holds banked arm
records. This is the one step in between, and it refuses rather than guesses.

WHICH ARM BECOMES A ROW, and why only one. One row per record, carrying the HINTED arm.
Q1 asks whether the reasoning describes the cue's bearing on the choice and Q2 asks
whether the answer is supported by that reasoning, so both are questions about the
transcript the cue was planted in; the step-0 coherence gate reads the same transcript
against the same final answer. Section 6.4 pairs no clean row with a hinted one, and the
JuryItem schema has exactly one ``reasoning`` and one ``final_answer`` per row, so a
clean row would be a second, separately-labelled item that no metric of section 6.4 or
6.5 reads. The clean arm is therefore NOT emitted, and the manifest says so in
``items.per_arm`` (``"clean": 0``) rather than leaving a reader to infer it.

ITEM IDENTITY. The transcripts carry no id field, so the id is built from the identity
this repository already uses for a banked record: ``experiments/logit_pass.py``'s
``record_key``, the sha256 of the question, the choices in their banked order, the gold
label, the planted hint label, the cue text and the cue placement, truncated to 16 hex
characters. It is IMPORTED and not re-derived here, so a sweep item id and the logit
sidecar's key for the same record can never drift apart. The record's position in the
source file travels beside it, exactly as the logit pass stores ``record_index`` beside
``record_key``: the index says where the record sits, the key says what it was.

    sweep-<model_slug>-<substrate>-<cue_family>-<record_index:05d>-<record_key>-hinted

WHAT IS REFUSED. A record that is missing a field the row needs raises with the field
named, the line number and the source file, rather than being dropped into a count. Two
groups of fields, for two reasons:

  the judge reads it      question, choices, hinted_cot (the reasoning), hinted_answer
                          (the final answer)
  the identity needs it   question, choices, answer_label, hint_label, cue_text,
                          cue_prepended, which are record_key's six inputs

Rows the cell's own analysis layer already excludes are excluded here on the same rule
and counted rather than refused: ``experiments/wave1_fits.py``'s ``load_records`` keeps
the rows whose ``source_file`` starts with ``arms_transcripts`` and drops the A9
specificity holdout (a different item set, no hinted arm) and the ``.logit.json``
sidecar rows (the same items on the logit scale). The rule is repeated here rather than
imported because wave1_fits pulls in numpy and the whole fits stack, which a converter
that runs beside the judge job has no business loading; the manifest prints the counts
per source file so the two can be checked against each other.

    PYTHONPATH=src python -m experiments.jury.sweep_items \\
        --transcripts ~/bcf/results/gemma-2-9b-it/arc_challenge/stated-hint/transcripts.jsonl

writes ~/bcf/jury-items/gemma-2-9b-it/arc_challenge/stated-hint/items.jsonl and, beside
it, manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The sibling-module import experiments/jury/backends.py makes, for the same reason: the
# record identity has ONE definition in this repository and this is not the place for a
# second one.
from logit_pass import record_key

from .family_map import ROSTER, family_of
from .runner import AUDIT_FRACTION, audit_rows

SCHEMA = "bcf.jury.sweep_items.v1"

# R15 part 2 (f). The one string that names this configuration, written into the manifest
# and, through it, onto every vote record the run writes.
CONFIGURATION = "secondary-qwen3-32b-d-np1024"

# The one arm the jury reads. See the module docstring.
ARM = "hinted"

# The default seed of the runner CLI and of bcf/judge_serve.sbatch. Recorded with the
# audit draw, because the draw is a function of the item ids and this seed.
DEFAULT_SEED = 7

ITEMS_ROOT = Path.home() / "bcf" / "jury-items"
ITEMS_NAME = "items.jsonl"
MANIFEST_NAME = "manifest.json"

# experiments/wave1_fits.py load_records, same rule, same reason.
ARMS_SOURCE_PREFIX = "arms_transcripts"
LOGIT_SIDECAR_SUFFIX = ".logit.json"

# What the judge is shown, and what it is shown it about.
JUDGE_FIELDS: tuple[str, ...] = ("question", "choices", "hinted_cot", "hinted_answer")
# record_key's six inputs. question and choices are in both groups; they are listed once.
IDENTITY_FIELDS: tuple[str, ...] = ("answer_label", "hint_label", "cue_text", "cue_prepended")


class SweepItemError(ValueError):
    """A record, a cell path or a model slug this converter refuses to guess at."""


@dataclass(frozen=True)
class Cell:
    """The cell a transcripts file belongs to, resolved once and then carried."""

    model_slug: str
    substrate: str
    cue_family: str
    subject_model: str
    subject_family: str
    path: Path
    dir_name: str = ""


def resolve_subject_model(model_slug: str) -> str:
    """The roster name for a results-directory model slug, or a refusal naming the slug.

    The cell directories are lowercased HF basenames (``gemma-2-9b-it``), and
    ``family_map.family_of`` is defined on the roster's display name and HF id only. A
    slug that matches neither is refused: routing a judge by a family this module made up
    would break the one rule section 6.2 asks the runner to assert per vote.
    """
    want = model_slug.strip().lower()
    for entry in ROSTER:
        if want in (entry.name.lower(), entry.hf_id.lower(), entry.hf_id.split("/")[-1].lower()):
            return entry.name
    raise SweepItemError(
        f"model slug {model_slug!r} is not on the canonical 18-model roster "
        f"(experiments/jury/family_map.py); known slugs include "
        f"{sorted(e.hf_id.split('/')[-1].lower() for e in ROSTER)[:4]} ... . "
        f"Pass --subject-model with a roster name to override the path."
    )


def slug_of(subject_model: str) -> str:
    """The results-directory slug of a roster subject: its HF basename, lowercased.

    Read off the roster rather than off the path, so an item id is a function of the
    roster entry and not of how the directory happened to be spelled. The directory's own
    name is kept in the manifest, so a cell filed under a different spelling is visible
    instead of silently producing ids nobody can match back.
    """
    for entry in ROSTER:
        if subject_model in (entry.name, entry.hf_id):
            return entry.hf_id.split("/")[-1].lower()
    raise SweepItemError(f"{subject_model!r} is not on the canonical 18-model roster")


def cell_of(
    transcripts: Path,
    *,
    subject_model: str = "",
    substrate: str = "",
    cue_family: str = "",
) -> Cell:
    """Read <model>/<substrate>/<cue_family> off the transcripts path, with overrides.

    The cell layout is CONTRACT.md's: ``<model_slug>/<substrate>/<cue_family>/`` holding
    ``transcripts.jsonl``. A path too shallow to carry all three is refused unless every
    missing part was given explicitly.
    """
    parts = transcripts.resolve().parent.parts
    dir_name = parts[-3] if len(parts) >= 3 else ""
    sub = substrate or (parts[-2] if len(parts) >= 2 else "")
    cue = cue_family or (parts[-1] if len(parts) >= 1 else "")
    name = subject_model or (resolve_subject_model(dir_name) if dir_name else "")
    if not (name and sub and cue):
        raise SweepItemError(
            f"cannot read <model>/<substrate>/<cue_family> from {transcripts}; "
            f"got model={name or dir_name!r} substrate={sub!r} cue_family={cue!r}. "
            f"Pass --subject-model, --substrate and --cue-family."
        )
    return Cell(
        model_slug=slug_of(name), substrate=sub, cue_family=cue,
        subject_model=name, subject_family=family_of(name), path=transcripts.parent,
        dir_name=dir_name,
    )


def _missing_field(record: dict, field: str) -> bool:
    """Whether one field is absent or empty for the purpose of building a row.

    ``cue_prepended`` is a boolean whose FALSE is a real value and whose absence
    record_key already reads as False, so only the key's presence is checked for it.
    Everything else has to be present and non-empty: a blank hinted_cot is a record with
    no transcript to judge, not a record with a short one.
    """
    if field not in record:
        return True
    value = record[field]
    if field == "cue_prepended":
        return False
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) < 2
    return False


def assert_record_usable(record: dict, *, where: str) -> None:
    """Refuse a record missing a field the row needs, naming the field."""
    for field in JUDGE_FIELDS + IDENTITY_FIELDS:
        if _missing_field(record, field):
            group = "the judge reads it" if field in JUDGE_FIELDS else "the item identity needs it"
            raise SweepItemError(
                f"{where}: record is missing required field {field!r} ({group}); "
                f"refusing to build an item row from it"
            )


def item_id_for(cell: Cell, record: dict, record_index: int) -> str:
    """The stable id: the cell, the record's position, its content key, and the arm."""
    return (
        f"sweep-{cell.model_slug}-{cell.substrate}-{cell.cue_family}"
        f"-{record_index:05d}-{record_key(record)}-{ARM}"
    )


def build_item(cell: Cell, record: dict, record_index: int, *, where: str) -> dict:
    """One JuryItem-shaped row for the hinted arm of one banked record."""
    assert_record_usable(record, where=where)
    return {
        "item_id": item_id_for(cell, record, record_index),
        "subject_model": cell.subject_model,
        "question": record["question"],
        "choices": list(record["choices"]),
        "reasoning": record["hinted_cot"],
        "final_answer": record["hinted_answer"],
        # family_of(subject_model) is what JuryItem.__post_init__ would fill in; writing
        # it makes the row say which stratum it belongs to without running the runner.
        "stratum": cell.subject_family,
        # Section 6.2's all-four subsample is a CALIBRATION-row device. A sweep row is not
        # one, and the secondary configuration is a single judge that the panel rule
        # routes onto Gemma and Llama subjects anyway.
        "all_judge_row": False,
        # Never rendered into a judge prompt: render() reads question, choices, reasoning
        # and final_answer and nothing else. This is provenance for the analysis that
        # prints the secondary column beside column A of record.
        "meta": {
            "arm": ARM,
            "configuration": CONFIGURATION,
            "secondary": True,
            "substrate": cell.substrate,
            "cue_family": cell.cue_family,
            "subject_family": cell.subject_family,
            "subject_model_slug": cell.model_slug,
            "record_key": record_key(record),
            "record_index": record_index,
            "source_file": record.get("source_file"),
            "hint_label": record["hint_label"],
            "clean_answer": record.get("clean_answer"),
            "answer_label": record["answer_label"],
            "cue_prepended": bool(record.get("cue_prepended", False)),
            # The frozen regex labels this column is reported beside (R15 part 2 (g)).
            "regex_labels": {
                "followed": record.get("followed"),
                "acknowledged": record.get("acknowledged"),
                "silent": record.get("silent"),
            },
        },
    }


def read_cell_records(transcripts: Path) -> tuple[list[tuple[int, dict]], dict]:
    """The arm records of one cell, with their 0-based positions, and the skip counts."""
    kept: list[tuple[int, dict]] = []
    skipped: dict[str, int] = {}
    n_lines = 0
    for line_no, line in enumerate(
        transcripts.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SweepItemError(f"{transcripts}:{line_no}: line is not JSON ({exc})") from exc
        source = str(record.get("source_file") or "")
        if source.endswith(LOGIT_SIDECAR_SUFFIX) or not source.startswith(ARMS_SOURCE_PREFIX):
            skipped[source or "<no source_file>"] = skipped.get(source or "<no source_file>", 0) + 1
            continue
        kept.append((line_no, record))
    return kept, {"lines": n_lines, "skipped_by_source_file": skipped}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(
    transcripts: Path,
    cell: Cell,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], dict]:
    """Every item row for one cell, and the manifest that describes the conversion."""
    kept, counts = read_cell_records(transcripts)
    items = [
        build_item(cell, record, index,
                   where=f"{transcripts}:{line_no} (record_index {index})")
        for index, (line_no, record) in enumerate(kept)
    ]
    ids = [i["item_id"] for i in items]
    audited = audit_rows(ids, seed=seed) if ids else set()
    manifest = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "configuration": CONFIGURATION,
        "secondary": True,
        "cell": {
            "path": str(cell.path),
            "dir_name": cell.dir_name,
            "model_slug": cell.model_slug,
            "subject_model": cell.subject_model,
            "subject_family": cell.subject_family,
            "substrate": cell.substrate,
            "cue_family": cell.cue_family,
        },
        "source": {
            "path": str(transcripts),
            "sha256": sha256_file(transcripts),
            "bytes": transcripts.stat().st_size,
            "lines": counts["lines"],
            "records_kept": len(kept),
            "skipped_by_source_file": counts["skipped_by_source_file"],
        },
        "items": {
            "count": len(items),
            "per_arm": {ARM: len(items), "clean": 0},
            "unique_item_ids": len(set(ids)),
            "unique_record_keys": len({i["meta"]["record_key"] for i in items}),
            "arms_note": (
                "one row per record, hinted arm only: Q1, Q2 and the step-0 gate are all "
                "asked about the transcript the cue was planted in, and section 6.4 pairs "
                "no clean row with it"
            ),
        },
        "audit": {
            "fraction": AUDIT_FRACTION,
            "seed": seed,
            "n_rows": len(audited),
            "note": "the seeded three-run audit draw of section 6.4 for this seed and this "
                    "item set; experiments/jury/runner.py audit_rows redraws it at run time",
        },
        "judge": {
            "key": "qwen3-32b",
            "q1_prompt": "d",
            "num_predict": 1024,
            "mode": "audit",
            "serving_line": "bf16, 1 x a100-80 (section 6.1, pinned)",
        },
    }
    return items, manifest


def write_items(items: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    return path


def default_out(cell: Cell, items_root: Path) -> Path:
    return items_root / cell.model_slug / cell.substrate / cell.cue_family / ITEMS_NAME


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert one sweep cell's transcripts.jsonl into jury runner items.")
    p.add_argument("--transcripts", required=True,
                   help="the cell's transcripts.jsonl (CONTRACT layout)")
    p.add_argument("--out", default="",
                   help=f"items JSONL to write; default <items-root>/<model>/<substrate>/"
                        f"<cue_family>/{ITEMS_NAME}")
    p.add_argument("--items-root", default=str(ITEMS_ROOT))
    p.add_argument("--subject-model", default="",
                   help="roster name, when the path's model slug is not to be trusted")
    p.add_argument("--substrate", default="")
    p.add_argument("--cue-family", default="")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="the run seed the audit draw is recorded for")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transcripts = Path(args.transcripts).expanduser()
    if not transcripts.is_file():
        raise SystemExit(f"no such transcripts file: {transcripts}")
    cell = cell_of(
        transcripts, subject_model=args.subject_model,
        substrate=args.substrate, cue_family=args.cue_family,
    )
    items, manifest = convert(transcripts, cell, seed=args.seed)
    out = Path(args.out).expanduser() if args.out else default_out(
        cell, Path(args.items_root).expanduser())
    write_items(items, out)
    # Filled here rather than in convert(): the items file has to exist to be hashed, and
    # bcf/judge_serve.sbatch checks this hash before it will run a cell, so a manifest can
    # never authorise an items file it does not describe.
    manifest["items"]["path"] = str(out)
    manifest["items"]["sha256"] = sha256_file(out)
    manifest_path = out.parent / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cell": manifest["cell"],
        "items": manifest["items"],
        "source": {k: manifest["source"][k] for k in
                   ("sha256", "lines", "records_kept", "skipped_by_source_file")},
        "audit_rows": manifest["audit"]["n_rows"],
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
