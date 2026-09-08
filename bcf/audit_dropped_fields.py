"""Audit: which field did bcf/wave.sh's row_fields() bug drop from each submitted row?

Before this lane's fix, `row_fields() { printf '%s' "$1" | tr '\t' '\n'; }` silently
dropped the LAST KEY=VALUE field of every wave manifest row from both call sites in
bcf/wave.sh -- the CARDS_WANTED/MAX_TP/MAX_HOURS prescan, and the sbatch --export=
construction every submitted job actually reads (DECISION-LOG.md 2026-09-08 09:39,
09:45; docs/HOLDS-LIFT-PLAN.md Step 0).

Given the wave manifests committed under bcf/waves/ and a copy of the cluster's
~/bcf/feeder-state.json (fetched read-only over ssh; a fixture with the same schema
works too), this script reconstructs, for every wave bcf/wave.sh actually submitted,
the exact row sbatch received, reads off that row's own last field (the one the bug
dropped), and classifies it as load-bearing (a submitted job's behavior would have
differed silently) or informational (recorded in the manifest, never read at runtime)
-- with the evidence for that classification cited, not asserted.

Usage:
    python bcf/audit_dropped_fields.py --feeder-state /path/to/feeder-state.json \
        [--waves-dir bcf/waves] [--out docs/WAVE-DROPPED-FIELDS.md]

Read-only: this never touches the cluster, never calls sbatch/squeue/scancel, and
writes only the one report file named by --out.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WAVES_DIR = REPO / "bcf" / "waves"
DEFAULT_OUT = REPO / "docs" / "WAVE-DROPPED-FIELDS.md"

# MEM and CPUS never reach the exported --export= string at all: bcf/wave.sh's own
# header comment says the split is "load bearing" because they become the sbatch
# --mem / --cpus-per-task flags instead, and a vLLM engine core has already died this
# campaign under the account's 3G/1-CPU default (job 825536, State=OUT_OF_MEMORY,
# ReqMem=3G, MaxRSS=6.3G) when a MEM export was silently absent. Neither has been
# observed as a row's actual last field in this repo's committed manifests, but a
# dropped MEM or CPUS would be exactly this failure again, so both are hardcoded
# load-bearing rather than left to the runtime-read search below.
ALWAYS_LOAD_BEARING = {"MEM", "CPUS"}

# Parsed out of the row by wave.sh (`IFS=$'\t' read -r MODEL SUBSTRATE CUE REST <<<
# "$row"`) BEFORE row_fields() ever sees it, so these can never be the field the bug
# drops. Present only so a caller can see they were considered, not missed.
NEVER_THE_LAST_FIELD = {"BCF_MODEL", "BCF_SUBSTRATE", "BCF_CUE"}

# Fields manually verified (for this audit, 2026-09-08) to have NO runtime read
# anywhere in the repo -- not just the two files classify_field() searches by default,
# but the whole tree (`grep -rn BCF_EXPECTED_HOURS bcf/ src/`): every hit is either
# bcf/wave.sh's own line 286 (planning-time only -- it feeds nothing but the --resume
# legs NOTE, never the job) or a manifest-writing call site (bcf/plan_waves.py,
# bcf/plan_enrich_waves.py, src/.../ladder/serve_manifest.py all WRITE this field into
# a row; none reads it back). Kept as an explicit, cited allowlist rather than folded
# into classify_field()'s generic search, so the search staying narrow (the two files a
# submitted job's shell actually executes) doesn't have to widen just to prove a
# negative for one already-verified name.
KNOWN_INFORMATIONAL = {
    "BCF_EXPECTED_HOURS": (
        "verified repo-wide: every occurrence outside wave manifests themselves is "
        "either bcf/wave.sh's own planning-time MAX_HOURS prescan (line ~286, feeds "
        "only a --resume-legs NOTE, never exported to a job) or a manifest-writing "
        "call site (bcf/plan_waves.py, bcf/plan_enrich_waves.py, "
        "src/bayes_cot_faithfulness/ladder/serve_manifest.py); no submitted job reads it"
    ),
}

# The two files a submitted job's own shell process reads. This is deliberately NOT
# bcf/wave.sh or bcf/wave_feeder.sh (planning-time scripts) and not every *.md doc that
# happens to mention a variable name.
RUNTIME_FILES = ("bcf/serve_and_run.sbatch", "bcf/env.sh")


def _non_comment_body(path: Path) -> str:
    if not path.exists():
        return ""
    kept = [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(kept)


def classify_field(key: str, repo_root: Path) -> tuple[str, str]:
    """Returns (verdict, evidence). verdict is 'load-bearing', 'informational', or
    'unknown' (no runtime read found in the evidence files -- reported, not assumed
    informational, so an unrecognised field never silently reads as harmless)."""
    if key in ALWAYS_LOAD_BEARING:
        evidence = (
            "becomes an sbatch --mem/--cpus-per-task flag inside bcf/wave.sh itself "
            "(its own header comment: 'that distinction is load bearing')"
        )
        return "load-bearing", evidence
    if key in KNOWN_INFORMATIONAL:
        return "informational", KNOWN_INFORMATIONAL[key]
    pattern = re.compile(rf"\$\{{?{re.escape(key)}\b")
    for rel in RUNTIME_FILES:
        body = _non_comment_body(repo_root / rel)
        if pattern.search(body):
            return "load-bearing", f"read at runtime in {rel} (non-comment line)"
    evidence = (
        f"no read of {key} found in {' or '.join(RUNTIME_FILES)}; treat as load-bearing "
        "until verified -- absence of evidence is not evidence of harmlessness"
    )
    return "unknown", evidence


def parse_manifest_rows(path: Path) -> list[list[str]]:
    """Every non-comment, non-blank row, split on tabs, in file order."""
    rows: list[list[str]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(line.split("\t"))
    return rows


@dataclass
class DroppedFieldRow:
    wave: str
    model: str
    substrate: str
    cue: str
    dropped_field: str
    dropped_value: str
    verdict: str
    evidence: str


# Waves bcf/wave.sh submitted directly, bypassing bcf/wave_feeder.sh entirely, and so
# absent from feeder-state.json's own "waves" dict (that file only records what the
# feeder itself submitted). Each entry cites the DECISION-LOG line that is the record
# of the submission. Every row in the named manifest is treated as submitted.
DIRECT_SUBMISSIONS = {
    "explore-phi4-off-01.tsv": (
        "job 828564, DECISION-LOG.md 2026-09-08 09:26 and 09:45: submitted via "
        "bcf/wave.sh directly ('on purpose, one row, hold list not applicable to an "
        "explicitly exploratory cell'); cancelled at 09:45 during its determinism "
        "preflight, before any arms were recorded, after this exact defect (dropping "
        "the row's own BCF_OUT_SUBROOT) misrouted its output into the wave-1 cell of "
        "record and was caught."
    ),
}


def submitted_rows_for_wave(
    wave_name: str, record: dict, waves_dir: Path
) -> list[list[str]]:
    """Reconstruct the rows bcf/wave.sh actually received for one feeder-tracked wave.

    Three shapes appear in feeder-state.json (schema bcf.wave_feeder.v1, read live off
    the cluster 2026-09-08):
      - no 'held_models' / 'held' key at all: the feeder applied no filter (an
        'adopted' entry, or one submitted with no hold list yet); every row in the
        manifest was submitted, matching len(job_ids) == row count.
      - 'held_models' + 'submitted_models': the feeder filtered before ever calling
        wave.sh, so only rows whose MODEL is in submitted_models reached sbatch.
      - 'held' (no 'submitted_models'): wave.sh was run with no hold filter at all, so
        EVERY row was submitted via sbatch -- some of the resulting job ids were later
        cancelled while still PENDING (recorded separately, not in this file), but the
        cancellation happened after submission, so the row_fields() bug already fired
        on them at sbatch time.
    """
    all_rows = parse_manifest_rows(waves_dir / wave_name)
    if "submitted_models" in record:
        wanted = set(record["submitted_models"])
        return [r for r in all_rows if len(r) >= 1 and r[0] in wanted]
    return all_rows


def audit(feeder_state: dict, waves_dir: Path) -> list[DroppedFieldRow]:
    out: list[DroppedFieldRow] = []
    waves = dict(feeder_state.get("waves", {}))
    for wave_name, note in DIRECT_SUBMISSIONS.items():
        waves.setdefault(wave_name, {"note": note, "_direct": True})

    for wave_name, record in sorted(waves.items()):
        for row in submitted_rows_for_wave(wave_name, record, waves_dir):
            if len(row) < 4:
                continue  # no REST fields at all; nothing for row_fields() to drop
            model, substrate, cue = row[0], row[1], row[2]
            last_field = row[-1]
            key = last_field.split("=", 1)[0]
            if key in NEVER_THE_LAST_FIELD:
                continue  # defensive; wave.sh's own parse makes this unreachable
            verdict, evidence = classify_field(key, REPO)
            out.append(
                DroppedFieldRow(
                    wave=wave_name,
                    model=model,
                    substrate=substrate,
                    cue=cue,
                    dropped_field=key,
                    dropped_value=last_field,
                    verdict=verdict,
                    evidence=evidence,
                )
            )
    return out


def render_report(rows: list[DroppedFieldRow], feeder_state: dict) -> str:
    total = len(rows)
    load_bearing = [r for r in rows if r.verdict == "load-bearing"]
    informational = [r for r in rows if r.verdict == "informational"]
    unknown = [r for r in rows if r.verdict == "unknown"]

    intro = (
        "Every row `bcf/wave.sh` actually submitted, before the `fix/wave-last-field` "
        "fix landed, silently lost its own LAST `KEY=VALUE` field from the sbatch "
        "`--export=` line (`row_fields()`'s missing trailing newline; DECISION-LOG.md "
        "2026-09-08 09:39/09:45, `docs/HOLDS-LIFT-PLAN.md` Step 0, "
        "`tests/test_wave_row_fields.py`). This audits every wave that "
        f"`~/bcf/feeder-state.json` (schema `{feeder_state.get('schema', '?')}`, "
        f"updated `{feeder_state.get('updated', '?')}`) or the DECISION-LOG records as "
        "submitted, and reports which field each row lost and whether that field was "
        "ever load-bearing."
    )
    lines = [
        "# Wave dropped-field audit",
        "",
        intro,
        "",
        "## Denominators",
        "",
        f"- rows submitted: **{total}**",
        f"- dropped field informational only (never read at runtime): **{len(informational)}**",
        f"- dropped field load-bearing (a running job's behavior would differ): **{len(load_bearing)}**",
        f"- dropped field unclassified (no runtime read found either way): **{len(unknown)}**",
        "",
    ]

    if unknown:
        unknown_header = (
            "**Unclassified fields found; read the evidence column before trusting "
            "the total above as final:**"
        )
        lines += [unknown_header, ""]
        for r in unknown:
            lines.append(f"- `{r.dropped_field}` ({r.wave}, {r.model}): {r.evidence}")
        lines.append("")

    lines += [
        "## Every submitted row, by wave",
        "",
        "| wave | model | substrate | cue | dropped field | verdict | evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        row_line = (
            f"| `{r.wave}` | {r.model} | {r.substrate} | {r.cue} | "
            f"`{r.dropped_field}` | {r.verdict} | {r.evidence} |"
        )
        lines.append(row_line)
    lines.append("")

    informational_note = (
        "- **informational**: the dropped field was recorded in the manifest but never "
        "read by a submitted job (e.g. `BCF_EXPECTED_HOURS`, planning-time metadata "
        "only). These rows ran under the configuration their manifest names; the field "
        "loss changed nothing about what actually executed."
    )
    load_bearing_note = (
        "- **load-bearing**: a submitted job's behavior, output location, or serving "
        "configuration would have silently differed from what the manifest row names, "
        "with no error anywhere in the pipeline to catch it. `BCF_OUT_SUBROOT` (job "
        "828564) is the one row in this audit where that already happened for real, "
        "caught and cancelled before any arms recorded; every other load-bearing row "
        "listed here is a **near miss** -- the field it lost never happened to be the "
        "true last field before now, or the row has not yet been re-verified against "
        "the fixed `bcf/wave.sh`."
    )
    unknown_note = (
        "- **unknown**: no runtime read was found for the dropped field in "
        "`bcf/serve_and_run.sbatch` or `bcf/env.sh`. Treated as load-bearing for the "
        "denominator above until someone confirms otherwise; a static grep proves a "
        "read exists, not that one doesn't."
    )
    closing_note = (
        "All of these rows were submitted before the fix landed; none has been "
        "re-submitted or re-verified against the fixed `bcf/wave.sh` by this script "
        "(it only reads manifests and feeder state, never sbatch/squeue). Cells whose "
        "dropped field was load-bearing are candidates for the resubmission the "
        "orchestrator already has queued (`docs/HOLDS-LIFT-PLAN.md` Step 6's "
        "`a100-40-resub-02.tsv`), not evidence that anything published is currently "
        "wrong -- read each row's own result files before drawing that conclusion."
    )
    lines += [
        "## Reading this table",
        "",
        informational_note,
        load_bearing_note,
        unknown_note,
        "",
        closing_note,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feeder-state", required=True, type=Path)
    ap.add_argument("--waves-dir", default=DEFAULT_WAVES_DIR, type=Path)
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    args = ap.parse_args()

    feeder_state = json.loads(args.feeder_state.read_text())
    rows = audit(feeder_state, args.waves_dir)
    report = render_report(rows, feeder_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"[audit] {len(rows)} submitted row(s) audited -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
