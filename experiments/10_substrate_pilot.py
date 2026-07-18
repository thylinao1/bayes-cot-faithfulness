"""Exploratory P3 substrate pilot: measure f = P(direct wrong | reasoned correct).

Implements the pilot protocol of docs/p3_substrate_scouting.md section 5, approved by
the operator on 2026-07-19 (option (a): both finalists). Two model calls per item on
the first N items of a candidate substrate in fetch order: a clean CoT pass and a
direct no-CoT pass, temperature 0.0, the frozen prompt frames, nothing else. From the
two passes it reports, with n and BOTH one-sided 95 percent Clopper-Pearson bounds:

  - c, the clean-correct rate (on BOTH denominators: items entered, the memo's yield
    arithmetic, and items measured, the attrition-honest rate);
  - f, the moved fraction among clean-correct items (direct answer wrong while the
    reasoned answer was correct) -- the quantity that predicts the P3 moved stratum;
  - y = f x c(entered), the moved yield per item entered;
  - direct accuracy over the measured items whose direct answer parsed (unscorable
    excluded and counted, the Exclusions discipline; derived from the same two
    calls, no extra call; the difficulty-window covariate beside memo C2).

These numbers SIZE a powered P3 run under the memo's pre-specified decision rule
(section 4); the rule itself is applied outside this script and recorded in the RUN
log. The pilot is exploratory and disclosed: nothing here is preregistered, nothing
is pooled with preregistered data, and nothing below carries a verdict.

No cue is ever planted: there is no hint, no taxonomy, no placebo -- just the clean
frame and the direct frame. $0 policy: local Ollama or free Groq only, availability-
gated exactly like 05/08; resume banks state so a free-tier cap costs nothing.

Run (from the repo root):

    python experiments/fetch_logiqa2.py --split test --n 120
    GROQ_API_KEY=... PYTHONPATH=src python experiments/10_substrate_pilot.py \\
        --backend groq --data experiments/data/logiqa2.json --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # sibling modules, exactly like 05 and 08
import arms_resume  # noqa: E402

from bayes_cot_faithfulness.arms import direct_prompt  # noqa: E402
from bayes_cot_faithfulness.guardrails import proportion_ci_upper  # noqa: E402

ARMS_SCRIPT = HERE / "08_additive_arms.py"


def _load_arms_module():
    """Load the additive-arms runner by file path (digit-prefixed, not importable).

    08 loads the frozen 05 the same way at import time; neither makes a model call on
    import. Reusing 08's machinery -- the substrate pass, the checked parse discipline,
    the availability-gated client, the checkpoint cadence -- keeps this pilot's call
    behavior identical to the preregistered sweeps instead of a drifting copy.
    """
    spec = importlib.util.spec_from_file_location("additive_arms_08", ARMS_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_M08 = _load_arms_module()
load_items = _M08.load_items
safe_generate = _M08.safe_generate
parse_or_force_checked = _M08.parse_or_force_checked
substrate_pass = _M08.substrate_pass
fail_message = _M08.fail_message
_gate_client = _M08._gate_client
CHECKPOINT_EVERY = _M08.CHECKPOINT_EVERY
FORCE_TOKENS = _M08.FORCE_TOKENS

STATUS_STRING = (
    "exploratory P3 substrate pilot (docs/p3_substrate_scouting.md section 5); "
    "disclosed; never pooled with preregistered data; no verdict"
)
PROTOCOL = "p3-substrate-pilot-v1"


def cp_lower(k: int, n: int) -> float:
    """Exact one-sided 95 percent CP LOWER bound via the duality 1 - upper(n - k, n).

    Routed through the repo's own proportion_ci_upper so no guardrail math is
    reimplemented here (the same convention as experiments/numbers_table.py).
    """
    return 1.0 - proportion_ci_upper(n - k, n)


def _rate_block(count: int, n: int) -> dict:
    """One rate with its n and both one-sided CP bounds; rate None when n == 0."""
    return {
        "n": n,
        "count": count,
        "rate": (count / n) if n else None,
        "cp95_lower": cp_lower(count, n) if n else None,
        "cp95_upper": proportion_ci_upper(count, n) if n else None,
    }


# --- Pure summarizer (no model, no network; unit-tested offline) --------------
def summarize_pilot(records: list[dict], n_entered: int) -> dict:
    """The pilot's counts and rates under the frozen Exclusions discipline.

    A record is scorable for the direct block only when its direct answer parsed;
    unscorable records are excluded from numerator AND denominator and counted
    (``n_unscorable``). The f block additionally conditions on clean-correct, the
    memo's definition f = P(direct wrong | reasoned correct). c is reported on both
    denominators: per item ENTERED (the memo's yield arithmetic N >= 30 / (f x c))
    and per item MEASURED (records that produced a clean answer attempt).
    """
    attrition = arms_resume.derive_attrition(n_entered, records)
    correct = [r for r in records if r.get("clean_correct")]
    n_measured = len(records)
    n_cc = len(correct)

    direct_rs = [r for r in records if "direct_answer" in r]
    direct_scorable = [r for r in direct_rs if r["direct_answer"] is not None]
    n_direct_correct = sum(
        1 for r in direct_scorable if r["direct_answer"] == r.get("answer_label")
    )

    moved_rs = [r for r in correct if "direct_answer" in r]
    moved_scorable = [r for r in moved_rs if r["direct_answer"] is not None]
    n_moved = sum(
        1 for r in moved_scorable if r["direct_answer"] != r.get("answer_label")
    )

    c_entered = (n_cc / n_entered) if n_entered else None
    f_block = _rate_block(n_moved, len(moved_scorable))
    f_block["n_unscorable"] = len(moved_rs) - len(moved_scorable)
    direct_block = _rate_block(n_direct_correct, len(direct_scorable))
    direct_block["n_unscorable"] = len(direct_rs) - len(direct_scorable)

    y = None
    if f_block["rate"] is not None and c_entered is not None:
        y = f_block["rate"] * c_entered

    return {
        "attrition": attrition,
        "n_clean_correct": n_cc,
        "clean_correct_rate_entered": _rate_block(n_cc, n_entered),
        "clean_correct_rate_measured": _rate_block(n_cc, n_measured),
        "direct_accuracy": direct_block,
        "moved_fraction_f": f_block,
        "moved_yield_y": y,
        "yield_note": (
            "y = f x c(entered), the memo's moved yield per item entered; "
            "sizing arithmetic N >= 30 / y"
        ),
    }


def serialize_pilot_record(r: dict) -> dict:
    """Flatten one pilot record into a JSON-safe transcript dict."""
    it = r["item"]
    out = {
        "question": it.question,
        "choices": list(it.choices),
        "answer_label": it.answer_label,
        "clean_answer": r.get("clean_answer"),
        "clean_cot": r.get("clean_cot"),
        "clean_correct": r.get("clean_correct"),
    }
    if "direct_answer" in r:
        out["direct_answer"] = r["direct_answer"]
    return out


def write_pilot_transcripts(out_dir: Path, label: str, records: list[dict]) -> int:
    """Persist every record generated so far; a run cut short keeps its work."""
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts = [serialize_pilot_record(r) for r in records]
    (out_dir / f"pilot_transcripts_{label}.json").write_text(
        json.dumps(transcripts, indent=2)
    )
    return len(transcripts)


def direct_pass(client, records, n_choices, ctx_label, out_dir, writer, backend, model):
    """The direct no-CoT pass over ALL pilot records (presence-skip on resume)."""
    for i, r in enumerate(records):
        if "direct_answer" in r:
            continue  # banked previously
        it = r["item"]
        out, err = safe_generate(client, direct_prompt(it), FORCE_TOKENS)
        if err is None:
            ans, err = parse_or_force_checked(client, it, out, n_choices)
        if err is not None:
            write_pilot_transcripts(out_dir, ctx_label, records)
            writer.write()
            print(fail_message(backend, model, err))
            return False
        r["direct_answer"] = ans
        print(f"      ... direct {i + 1}/{len(records)}", end="\r", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            write_pilot_transcripts(out_dir, ctx_label, records)
            writer.write()
    print()
    return True


def report(summary: dict) -> None:
    """Print the pilot numbers (exploratory; no verdict; no rule application)."""
    def pct(x):
        return "n/a" if x is None else f"{x:.1%}"

    a = summary["attrition"]
    ce = summary["clean_correct_rate_entered"]
    cm = summary["clean_correct_rate_measured"]
    d = summary["direct_accuracy"]
    f = summary["moved_fraction_f"]
    # Each bound is ONE-SIDED (the numbers_table convention): labeled separately so a
    # "lo-hi" dash range is never misread as a two-sided 95 percent interval (jointly
    # the pair is a 90 percent central interval).
    def bounds(b):
        return f"CP95 lo {pct(b['cp95_lower'])}  CP95 hi {pct(b['cp95_upper'])}"

    print(f"  attrition: entered={a['n_entered']} no-record={a['n_failed_generation']} "
          f"unparseable-clean={a['n_unparseable_clean']}")
    print(f"  c (entered):  {pct(ce['rate'])} ({ce['count']}/{ce['n']}, {bounds(ce)})")
    print(f"  c (measured): {pct(cm['rate'])} ({cm['count']}/{cm['n']}, {bounds(cm)})")
    print(f"  direct accuracy: {pct(d['rate'])} ({d['count']}/{d['n']}, "
          f"{d['n_unscorable']} unscorable, {bounds(d)})")
    print(f"  f (moved | clean-correct): {pct(f['rate'])} ({f['count']}/{f['n']}, "
          f"{f['n_unscorable']} unscorable, {bounds(f)})")
    print(f"  y = f x c(entered): {pct(summary['moved_yield_y'])}")
    print("  status: " + STATUS_STRING + ".")


def run(data_path: Path, n_items: int, out_dir: Path, model: str, backend: str,
        host: str, num_predict: int, timeout: float, resume: bool) -> int:
    safe_model = model.replace(":", "_").replace("/", "_")
    label = f"{Path(data_path).stem}_{safe_model}"
    checkpoint_path = out_dir / f"pilot_checkpoint_{label}.json"
    params = {
        "protocol": PROTOCOL,
        "model": model,
        "backend": backend,
        "n_items": n_items,
        "data": str(data_path),
        "data_sha256": arms_resume.file_sha256(data_path),
        "num_predict": num_predict,
    }

    # Resume gate BEFORE the backend gate, exactly like 08: an unreadable checkpoint,
    # a version or parameter mismatch, or duplicate item keys refuse with zero calls.
    loaded = None
    if resume:
        try:
            loaded = arms_resume.load_checkpoint(checkpoint_path)
        except arms_resume.CheckpointUnreadable as exc:
            print(arms_resume.unreadable_refusal_message(exc, checkpoint_path))
            return 0
        if loaded is None:
            print(arms_resume.NO_CHECKPOINT_NOTE)
        else:
            bad_version = arms_resume.version_mismatch(loaded)
            if bad_version is not None:
                print(arms_resume.version_refusal_message(bad_version, checkpoint_path))
                return 0
            saved = loaded.get("params", {})
            mismatches = arms_resume.params_mismatch(saved, params)
            # params_mismatch compares only arms_resume.PARAM_FIELDS, which does not
            # know this script's "protocol" field -- guard it explicitly, or a future
            # protocol revision would silently merge v1 banked records into a v2 run.
            if saved.get("protocol") != params["protocol"]:
                mismatches.insert(0, ("protocol", saved.get("protocol"), params["protocol"]))
            if mismatches:
                print(arms_resume.refusal_message(mismatches, checkpoint_path))
                return 0
        dupes = arms_resume.duplicate_item_keys(load_items(data_path)[:n_items])
        if dupes:
            print(arms_resume.duplicate_refusal_message(dupes, "--data", data_path))
            return 0

    client = _gate_client(backend, model, host, timeout)
    if client is None:
        return 0

    items = load_items(data_path)[:n_items]
    if not items:
        print(f"[pilot] no items loaded from {data_path}; nothing to run.")
        return 0
    n_choices = max(len(it.choices) for it in items)

    records: list[dict] = []
    writer = arms_resume.CheckpointWriter(
        checkpoint_path, params, records, len(items), _M08._curve_to_dict, loaded,
        n_invocations=(loaded or {}).get("n_invocations", 0) + 1,
    )
    print(f"[1/2] Clean CoT pass: {len(items)} items from {Path(data_path).name} on {model}")
    banked = arms_resume.resume_inputs(loaded) if loaded is not None else None
    _, ok, _ = substrate_pass(
        client, items, n_choices, num_predict, backend, model,
        banked=banked, records=records, on_checkpoint=writer.write,
        locked=False,  # no position-seeded draw exists in this protocol; holes heal
    )
    if not ok:
        return 0
    writer.write()

    print(f"[2/2] Direct no-CoT pass on ALL {len(records)} measured items")
    if not direct_pass(client, records, n_choices, label, out_dir, writer,
                       backend, model):
        return 0

    summary = {
        "protocol": PROTOCOL,
        "backend": backend,
        "model": model,
        "data": str(data_path),
        "data_sha256": params["data_sha256"],
        "n_items": len(items),
        "num_predict": num_predict,
        "n_invocations": writer.n_invocations,
        "resumed": writer.n_invocations > 1,
        **summarize_pilot(records, len(items)),
        "status": STATUS_STRING,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"pilot_summary_{label}.json").write_text(json.dumps(summary, indent=2))
    n_saved = write_pilot_transcripts(out_dir, label, records)
    writer.write()
    report(summary)
    print(f"\nwrote summary + {n_saved} pilot transcripts -> {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, required=True,
                    help="candidate substrate file written by a fetch_*.py")
    ap.add_argument("--n-items", type=int, default=120,
                    help="first N items in fetch order (the memo's pilot protocol)")
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "p3_pilots")
    ap.add_argument("--num-predict", type=int, default=320)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--backend", choices=["ollama", "groq"], default="groq")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="continue a stopped pilot from its checkpoint in --out; the "
                         "checkpoint is keyed by (data-file stem, sanitized model), so a "
                         "changed --model normally resolves to a different checkpoint and "
                         "starts a FRESH pilot (the 08 runner's per-model semantics); if "
                         "sanitization collides (: and / both map to _), the params guard "
                         "refuses instead")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return run(a.data, a.n_items, a.out, a.model, a.backend, a.host,
               a.num_predict, a.timeout, a.resume)


if __name__ == "__main__":
    raise SystemExit(main())
