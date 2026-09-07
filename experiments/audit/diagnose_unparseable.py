"""Classify the unparseable clean outputs of a wave-1 cell. Read-only.

Uses the FROZEN parser (interventions.parse_answer) to decide what is unparseable, and a
separate lenient extractor, defined here in the audit tooling only, to decide whether an
answer was nevertheless present. Neither the parser nor any frozen constant is touched.

Classes, in decision order:
  empty                        the completion has no non-whitespace characters
  answer_present_parser_missed the lenient extractor finds a valid letter the parser missed
  truncated_in_thinking        no answer, the completion re-tokenizes at the num_predict cap,
                               and the reasoning block was never closed
  other                        everything else, with a sub-label

Sub-labels reported beside the four classes (orthogonal flags, not extra classes):
  detokenization_artifact      the returned text carries GPT-2 byte-level markers (U+0120,
                               U+010A) instead of decoded spaces and newlines
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from bayes_cot_faithfulness.interventions import parse_answer

RESULTS = os.path.expanduser("~/bcf/results")
BYTE_MARKERS = ("\u0120", "\u010a")  # G-with-stroke and C-with-stroke: BPE space / newline
OPEN_THINK = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
CLOSE_THINK = re.compile(r"<\s*/\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)

# Lenient patterns the frozen parser does NOT try. Each names a shape in which the model
# has COMMITTED to a final answer; deliberation shapes ("option B is wrong", "so B seems
# plausible") are deliberately NOT here, because a mid-chain mention of a letter is not an
# answer and counting one as an answer would inflate this class. A first version of this
# list did include "option X" and "so X", fired on 17 of 30 OLMo rows and 18 of 30
# R1-Distill rows, and every one of those was a chain truncated mid-deliberation at the
# 320-token cap; the rows are in git history and the over-firing is why the list is narrow.
_LENIENT_ANYWHERE = [
    re.compile(r"\\boxed\{\s*\(?\s*([A-F])\s*\)?\s*\}"),                    # \boxed{B}
    re.compile(r"final\s+answer\D{0,12}?\(?\s*([A-F])\s*\)?\b", re.IGNORECASE),        # final answer: (B)
    re.compile(r"answer[\u0120\u010a]+(?:is[\u0120]+)?\(?\s*([A-F])\s*\)?"),  # byte-marked
]
# Shapes that only count when they close the completion (last _TAIL_WINDOW characters).
_TAIL_WINDOW = 120
_LENIENT_TAIL = [
    re.compile(r"\*\*\s*(?:answer\s*[:\-]?\s*)?\(?\s*([A-F])\s*\)?\s*\*\*", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*([A-F])\s*\)?\s*$", re.MULTILINE),   # a line that is only the letter
]


def lenient_answer(text: str, n_choices: int) -> str | None:
    """A COMMITTED answer the frozen parser missed, or None. Audit tooling only."""
    t = text or ""
    valid = set("ABCDEF"[:n_choices])
    for rgx in _LENIENT_ANYWHERE:
        hits = [m.upper() for m in rgx.findall(t) if m.upper() in valid]
        if hits:
            return hits[-1]
    tail = t[-_TAIL_WINDOW:]
    for rgx in _LENIENT_TAIL:
        hits = [m.upper() for m in rgx.findall(tail) if m.upper() in valid]
        if hits:
            return hits[-1]
    return None


def classify(text: str, n_choices: int, n_tokens: int | None, cap: int) -> tuple[str, str]:
    t = text or ""
    if not t.strip():
        return "empty", ""
    sub = "detokenization_artifact" if any(b in t for b in BYTE_MARKERS) else ""
    if lenient_answer(t, n_choices) is not None:
        return "answer_present_parser_missed", sub
    opened = bool(OPEN_THINK.search(t))
    closed = bool(CLOSE_THINK.search(t))
    at_cap = n_tokens is not None and n_tokens >= cap
    if at_cap and not closed:
        return "truncated_in_thinking", sub or ("implicit_open" if not opened else "explicit_open")
    if not at_cap and not closed:
        return "other", sub or "stopped_under_the_cap_without_an_answer"
    return "other", sub or "closed_block_no_answer"


def cell_dir(slug):
    return os.path.join(RESULTS, slug, "arc_challenge", "stated-hint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--cap", type=int, default=320)
    ap.add_argument("--tokenizer", default=None, help="HF id; default read from run_meta")
    ap.add_argument("--quote-chars", type=int, default=180)
    a = ap.parse_args()

    d = cell_dir(a.slug)
    hits = sorted(glob.glob(os.path.join(d, "arms_checkpoint_*.json")))
    if not hits:
        print(json.dumps({"slug": a.slug, "error": "no checkpoint"})); return
    with open(hits[0]) as fh:
        ck = json.load(fh)
    with open(os.path.join(d, "run_meta.json")) as fh:
        meta = json.load(fh)
    recs = ck.get("records") or []
    bad = [(i, r) for i, r in enumerate(recs) if r.get("clean_answer") is None]

    # Tokenizer via the `tokenizers` library and a raw tokenizer.json download, NOT via
    # transformers.AutoTokenizer: on the login node transformers pulls in torch, whose CUDA
    # shared objects do not map there (ImportError: libtorch_cuda.so).
    tok = None
    tok_id = a.tokenizer or meta["model"]
    tok_err = None
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        rev = meta.get("hf_revision")
        path = hf_hub_download(tok_id, "tokenizer.json", revision=rev)
        tok = Tokenizer.from_file(path)
    except Exception as exc:  # noqa: BLE001 - any failure means "no token stats", never invented
        tok_err = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"tokenizer_error": tok_err}), file=sys.stderr)

    def ntok(t):
        if tok is None:
            return None
        return len(tok.encode(t or "", add_special_tokens=False).ids)

    # token lengths over ALL records and over the unparseable ones
    def stats(rs):
        lens = [ntok(r.get("clean_cot")) for _, r in rs] if tok else []
        lens = [x for x in lens if x is not None]
        if not lens:
            return {"n": len(rs), "tokens": None}
        return {
            "n": len(rs),
            "mean_tokens": round(sum(lens) / len(lens), 2),
            "max_tokens": max(lens),
            "min_tokens": min(lens),
            "n_at_or_above_cap": sum(1 for x in lens if x >= a.cap),
            "share_at_cap": round(sum(1 for x in lens if x >= a.cap) / len(lens), 6),
        }

    allrs = list(enumerate(recs))
    out = {
        "slug": a.slug,
        "model": meta["model"],
        "num_predict": meta["num_predict"],
        "chat_template_kwargs": meta.get("chat_template_kwargs"),
        "tokenizer_source": ("tokenizer.json@" + str(meta.get("hf_revision"))) if tok else None,
        "tokenizer_error": tok_err,
        "n_records": len(recs),
        "n_unparseable_clean": len(bad),
        "token_stats_all_clean": stats(allrs),
        "token_stats_unparseable": stats(bad),
        "byte_artifact_all_clean": sum(
            1 for _, r in allrs if any(b in (r.get("clean_cot") or "") for b in BYTE_MARKERS)
        ),
        "open_think_tag_all_clean": sum(
            1 for _, r in allrs if OPEN_THINK.search(r.get("clean_cot") or "")
        ),
        "close_think_tag_all_clean": sum(
            1 for _, r in allrs if CLOSE_THINK.search(r.get("clean_cot") or "")
        ),
    }

    rng = random.Random(a.seed)
    sample = bad if len(bad) <= a.sample else rng.sample(bad, a.sample)
    sample.sort(key=lambda p: p[0])
    counts, subs, rows = {}, {}, []
    for i, r in sample:
        t = r.get("clean_cot") or ""
        nch = len(r.get("choices") or [])
        assert parse_answer(t, nch) is None, "frozen parser disagrees with the record"
        n = ntok(t)
        cls, sub = classify(t, nch, n, a.cap)
        counts[cls] = counts.get(cls, 0) + 1
        key = f"{cls}/{sub}" if sub else cls
        subs[key] = subs.get(key, 0) + 1
        rows.append({
            "record_index": i, "class": cls, "sub": sub, "n_tokens": n, "n_chars": len(t),
            "head": t[: a.quote_chars], "tail": t[-a.quote_chars :],
        })
    out["sample_seed"] = a.seed
    out["sample_n"] = len(sample)
    out["sample_denominator"] = len(bad)
    out["class_counts"] = counts
    out["class_sub_counts"] = subs
    out["rows"] = rows
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
