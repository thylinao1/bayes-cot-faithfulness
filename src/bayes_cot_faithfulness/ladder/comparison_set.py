"""Element 11(e): the four columns the sweep already measures, beside every ladder claim.

    "Every ladder claim is reported beside raw cue susceptibility, answer entropy,
    trace length, early answering, the calibrated jury, and simple probes at equal
    access."
    -- PREREGISTRATION_jury_and_scale.md section 12, element 11(e)

Four of those six are computable from a cell's OWN records with no new model call, and
this module computes them. Two are not, and this module refuses to fake them: the
calibrated jury column needs a jury run and element 4's calibration frame, and the
simple probes need a probe trained at equal access. Both are returned as ``None`` with a
``spec`` saying what would fill them, so a comparison table that is missing them says so
in the table rather than in a footnote.

Where each computed column comes from, by field name
----------------------------------------------------
raw cue susceptibility   per-item ``followed`` (the frozen parser's verdict, set in
                         ``experiments/08_additive_arms.py``: ``"followed": res["ans"]
                         == res["hint"]``). Items whose answer did not parse carry
                         ``None`` and are counted as unscorable, never as non-followers.
answer entropy           ``renormalized_over_letters`` from the anchor arm's forced
                         letter-logprob block (``outcome_scale.letter_logprob_fields``),
                         normalised by log(number of scored letters) so 1.0 is uniform.
trace length             ``n_chars`` on the curve rows (the joined depth completions of
                         ``experiments/08_additive_arms.py``), reported as a mean and a
                         median because one runaway trace moves a mean.
early answering          ``commitment_depth`` and ``curve_area`` from the truncation
                         curves; the headline is the fraction of items already committed
                         at depth 0, which is the arm's own definition of answering
                         early.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

SCHEMA = "bcf.ladder.comparison_set.v1"

# The two columns that cannot be computed from banked records, and what each needs.
NOT_COMPUTED_HERE = {
    "calibrated_jury": {
        "column": "calibrated_jury_q1_rate",
        "needs": (
            "a jury run over this cell's hinted transcripts under the panel rule of "
            "element 5, with the element 4 calibration frame applied, and ruling R9's "
            "status: A4 records that there is no primary jury configuration, so this "
            "column may legitimately be absent"
        ),
        "produced_by": "bcf/jury_wave.sh and the jury runner, not this module",
    },
    "simple_probes": {
        "column": "simple_probe_score",
        "needs": (
            "a probe trained at EQUAL ACCESS to the ladder's own inputs and evaluated "
            "on the same items; element 11(e) says 'at equal access', so a probe with "
            "more access than the statistic is not this column"
        ),
        "produced_by": "the probe lane, not this module",
    },
}


def _rate(n: int, d: int) -> float | None:
    return (n / d) if d else None


def cue_susceptibility(records: Sequence[dict]) -> dict:
    """The frozen parser's follow rate, with the unscorable counted apart."""
    scorable = [r for r in records if r.get("followed") is not None]
    n_follow = sum(1 for r in scorable if r["followed"])
    return {
        "column": "raw_cue_susceptibility",
        "n_records": len(records),
        "n_scorable": len(scorable),
        "n_unscorable": len(records) - len(scorable),
        "n_follow": n_follow,
        "follow_rate": _rate(n_follow, len(scorable)),
        "source_field": "followed",
        "why_none": None if scorable else "no record carried a scorable `followed` flag",
    }


def answer_entropy(records: Sequence[dict]) -> dict:
    """Normalised Shannon entropy of the renormalised letter distribution, per item."""
    values = []
    missing = 0
    for r in records:
        dist = r.get("renormalized_over_letters")
        if not dist:
            missing += 1
            continue
        ps = [p for p in dist.values() if p > 0]
        if len(ps) < 2:
            missing += 1
            continue
        h = -sum(p * math.log(p) for p in ps) / math.log(len(dist))
        values.append(h)
    return {
        "column": "answer_entropy",
        "n_records": len(records),
        "n_with_letter_logprobs": len(values),
        "n_without": missing,
        "mean_normalized_entropy": statistics.fmean(values) if values else None,
        "median_normalized_entropy": statistics.median(values) if values else None,
        "source_field": "renormalized_over_letters (outcome_scale.letter_logprob_fields)",
        "why_none": None if values else (
            "no record carried a renormalized_over_letters block; the anchor arm's "
            "forced letter-logprob pass is what writes it"),
    }


def trace_length(records: Sequence[dict]) -> dict:
    """Characters of generated text per item, mean and median."""
    values = [r["n_chars"] for r in records if isinstance(r.get("n_chars"), int)]
    return {
        "column": "trace_length",
        "n_records": len(records),
        "n_with_n_chars": len(values),
        "mean_chars": statistics.fmean(values) if values else None,
        "median_chars": statistics.median(values) if values else None,
        "source_field": "n_chars",
        "why_none": None if values else "no record carried n_chars",
    }


def early_answering(covariates: Sequence[dict]) -> dict:
    """Commitment depth and curve area from the truncation curves."""
    depths = [c["commitment_depth"] for c in covariates
              if c.get("commitment_depth") is not None]
    areas = [c["curve_area"] for c in covariates if c.get("curve_area") is not None]
    n_at_zero = sum(1 for d in depths if d == 0)
    return {
        "column": "early_answering",
        "n_covariate_rows": len(covariates),
        "n_with_commitment_depth": len(depths),
        "n_precommitted_at_depth_0": n_at_zero,
        "precommitted_fraction": _rate(n_at_zero, len(depths)),
        "mean_commitment_depth": statistics.fmean(depths) if depths else None,
        "mean_curve_area": statistics.fmean(areas) if areas else None,
        "source_field": "curves arm covariates: commitment_depth, curve_area",
        "why_none": None if depths else "the curves arm produced no covariate rows",
    }


def comparison_row(
    cell_id: str,
    records: Sequence[dict],
    curve_covariates: Sequence[dict] = (),
    *,
    calibrated_jury_q1_rate: float | None = None,
    simple_probe_score: float | None = None,
) -> dict:
    """One row of the 11(e) comparison table for one checkpoint or cell.

    The two columns this module does not compute are accepted as arguments so a caller
    that HAS them can fill them, and are reported as absent with their spec when it does
    not. Nothing is imputed either way.
    """
    row = {
        "schema": SCHEMA,
        "cell_id": cell_id,
        "raw_cue_susceptibility": cue_susceptibility(records),
        "answer_entropy": answer_entropy(records),
        "trace_length": trace_length(records),
        "early_answering": early_answering(curve_covariates),
        "calibrated_jury_q1_rate": calibrated_jury_q1_rate,
        "simple_probe_score": simple_probe_score,
        "not_computed_here": {
            k: v for k, v in NOT_COMPUTED_HERE.items()
            if (calibrated_jury_q1_rate is None and k == "calibrated_jury")
            or (simple_probe_score is None and k == "simple_probes")
        },
    }
    return row


def from_cell_dir(cell_dir: Path, cell_id: str | None = None,
                  records_name: str = "records.jsonl") -> dict:
    """Read a cell's own artifacts and build its comparison row.

    ``arms_summary.json`` supplies the curve covariates (the runner already summarises
    them there) and ``records.jsonl``, when present, supplies the per-item fields. A
    missing records file is reported as a missing file, not as an empty cell.
    """
    cell_dir = Path(cell_dir)
    summary_path = cell_dir / "arms_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} does not exist")
    summary = json.loads(summary_path.read_text())
    covariates = (
        summary.get("arms", {}).get("curves", {}).get("hinted", {}).get("covariates")
        or summary.get("arms", {}).get("curves", {}).get("clean", {}).get("covariates")
        or []
    )
    records: list[dict] = []
    rec_path = cell_dir / records_name
    if rec_path.is_file():
        records = [json.loads(line) for line in rec_path.read_text().splitlines()
                   if line.strip()]
    row = comparison_row(cell_id or cell_dir.name, records, covariates)
    row["sources"] = {
        "arms_summary": str(summary_path),
        "records": str(rec_path) if rec_path.is_file() else None,
        "n_records_read": len(records),
        "why_no_records": None if rec_path.is_file() else (
            f"{rec_path} does not exist; three of the four computed columns read "
            "per-item records and are reported as absent rather than estimated from "
            "the summary"),
    }
    return row
