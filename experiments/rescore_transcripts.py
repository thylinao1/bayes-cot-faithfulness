"""Re-apply the current acknowledgment / unfaithfulness heuristic to saved transcripts.

The real-model run stores raw transcripts (question, answers, chain-of-thought) plus
the heuristic labels computed at run time. When the acknowledgment detector improves,
those labels can be recomputed from the saved text without spending another API call,
which is the whole reason transcripts are persisted. The pre-registration calls for a
hand spot-check of transcripts before reporting; this is the tool that folds the result
of that spot-check back into the saved data.

It recomputes ``acknowledged_hint`` and ``silent_unfaithful`` for every transcript,
rewrites the transcript and silent-case files, and updates the silent-dependent fields
of the summary (follow and neutral rates do not depend on the detector, so they are
left untouched). Nothing here calls a model or the network.

    PYTHONPATH=src python experiments/rescore_transcripts.py --model llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_cot_faithfulness.interventions import (  # noqa: E402
    acknowledges_hint,
    is_unfaithful_on_hint,
    parse_answer,
)

HERE = Path(__file__).resolve().parent


def rescore(results_dir: Path, safe_model: str) -> int:
    tx_path = results_dir / f"control_transcripts_{safe_model}.json"
    if not tx_path.exists():
        print(f"[rescore] no transcripts at {tx_path}; run experiments/05 first.")
        return 1

    transcripts = json.loads(tx_path.read_text())
    before_silent = sum(bool(t.get("silent_unfaithful")) for t in transcripts)
    before_unparsed = sum(t.get("hinted_answer") is None for t in transcripts)

    for t in transcripts:
        cot = t.get("hinted_cot") or ""
        # re-parse the final answer from the saved text with the current (lenient) parser;
        # recovers off-format endings, though a truly truncated generation stays None.
        ans = parse_answer(cot, len(t.get("choices") or []) or 5)
        t["hinted_answer"] = ans
        t["followed_hint"] = ans == t.get("hint_label")
        t["acknowledged_hint"] = acknowledges_hint(cot)
        t["silent_unfaithful"] = is_unfaithful_on_hint(ans, t.get("hint_label"), cot)

    followers = [t for t in transcripts if t.get("followed_hint")]
    silent = [t for t in transcripts if t["silent_unfaithful"]]
    disclosed = [t for t in followers if t["acknowledged_hint"]]
    stable_followers = [t for t in followers if not t.get("neutral_changed")]
    stable_silent = [t for t in stable_followers if t["silent_unfaithful"]]

    tx_path.write_text(json.dumps(transcripts, indent=2))
    (results_dir / f"silent_unfaithful_{safe_model}.json").write_text(json.dumps(silent, indent=2))

    # Update only the detector-dependent fields of the summary, if present.
    summary_path = results_dir / f"control_summary_{safe_model}.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        n_clean = s.get("n_clean_correct") or len(transcripts)
        s["silent_unfaithful_rate"] = len(silent) / n_clean if n_clean else None
        s["silent_given_follow"] = (len(silent) / len(followers)) if followers else None
        s["stable_silent_given_follow"] = (
            len(stable_silent) / len(stable_followers) if stable_followers else None
        )
        s["n_followed_hint"] = len(followers)
        s["n_silent_unfaithful"] = len(silent)
        s["n_disclosed_hint"] = len(disclosed)
        s["rescored"] = True
        summary_path.write_text(json.dumps(s, indent=2))

    after_unparsed = sum(t["hinted_answer"] is None for t in transcripts)
    print(f"[rescore] {safe_model}: {len(transcripts)} transcripts, "
          f"{len(followers)} followed the wrong hint")
    print(f"  unparseable answers:           {before_unparsed} -> {after_unparsed}"
          f"  (still-None are truncated generations; re-run with a larger --num-predict)")
    print(f"  silent-unfaithful (flagged):   {before_silent} -> {len(silent)}")
    print(f"  disclosed the cue (cleared):   {len(disclosed)}")
    if silent:
        ex = silent[0]
        print("\n  lead silent case (followed the wrong hint, never disclosed it):")
        print(f"    Q: {ex['question']}")
        print(f"    planted hint -> ({ex['hint_label']})   answered ({ex['hinted_answer']})   "
              f"correct ({ex['answer_label']})")
    if disclosed:
        ex = disclosed[0]
        print("\n  reclassified as disclosed (honest deference, not silent):")
        print(f"    Q: {ex['question']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b-instant",
                    help="model tag as it appears in the results filenames")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    a = ap.parse_args()
    safe_model = a.model.replace(":", "_").replace("/", "_")
    return rescore(a.results, safe_model)


if __name__ == "__main__":
    raise SystemExit(main())
