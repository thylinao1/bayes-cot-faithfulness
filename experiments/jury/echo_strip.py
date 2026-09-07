"""Option (d): the deterministic echo strip, as an EXPLORATORY configuration.

WHAT IT IS. Before a response reaches the judge, remove from it every contiguous span of
at least `min_chars` characters that appears verbatim in the prompt the response was
produced from, comparing on whitespace-normalized text so that a reflowed copy still
counts as a copy. Shorter quotes survive, which is the point: the cue sentence in this
corpus is 86 to 99 characters long, so at the default 200 it is kept and only bulk
reproduction of the item is removed.

WHY IT IS HERE. `restated_cue_only` is the gate class that carries the cue's words with no
statement about their bearing on the choice, and it is the class the Q1 revisions kept
trading against `paraphrased_disclosure`. Option (d) asks whether the trade is avoidable
by removing the reproduction mechanically instead of asking a judge to see past it.

WHAT THE MEASUREMENT SAYS, and it is not what option (d) hoped. On the frozen 483 item
corpus the longest verbatim whitespace-normalized overlap between a response and its own
subject prompt is 86 to 99 characters in `restated_cue_only` and 13 to 99 in every other
class (`tests/test_echo_strip.py` pins the distribution). At the specified 200 the strip is
a NO-OP on all seven classes: it removes nothing from any of the 483 items. The reason is
in `synthetic_gate.py`: the restated class is built by quoting the CUE SENTENCE under a
template header, not by copying the whole item, so the echo it plants is 86 characters and
not the several hundred a 200 character rule is written for.

Nothing here is named the candidate, no threshold moved, no corpus item changed and no
prompt file was edited. The parameters and this file's SHA-256 are recorded on every vote
of any run that uses it, so a run made with the strip can never be read as a run without.

    python -m experiments.jury.echo_strip --items gate_items.jsonl --out stripped.jsonl
    python -m experiments.jury.echo_strip --items gate_items.jsonl --measure
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# The span length at or above which a verbatim reproduction is treated as an echo. Chosen
# so a quoted sentence survives and a copied block does not. It is a RUN parameter of an
# exploratory configuration, recorded with every vote, and it is not one of the frozen
# numbers: gate_thresholds.py, the corpus manifest and the prompt files are untouched.
MIN_ECHO_CHARS = 200

LETTERS = "ABCDEFGH"

# The cue sentence the gate corpus plants when a banked row carries no cue text of its own.
# Kept byte-identical to synthetic_gate.DEFAULT_CUE_TEXT; imported rather than retyped.
from .synthetic_gate import DEFAULT_CUE_TEXT


def module_sha256() -> str:
    """Hash of this file, recorded on every vote of a run that used the strip."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def normalize(text: str) -> tuple[str, list[int]]:
    """Whitespace-normalized text plus, for each character of it, its index in the original.

    Runs of whitespace collapse to one space. The index list is what lets a span found in
    normalized coordinates be cut out of the ORIGINAL string, so the returned response is
    the author's own text minus the echo, not a reflowed paraphrase of it.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for position, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(position)
            prev_space = True
        else:
            out.append(ch)
            idx.append(position)
            prev_space = False
    return "".join(out), idx


def echo_spans(response: str, prompt: str, min_chars: int = MIN_ECHO_CHARS) -> list[tuple[int, int]]:
    """Maximal echo spans, as half-open ranges of the ORIGINAL response string.

    Left to right, greedy and deterministic: the first position whose next `min_chars`
    normalized characters occur in the prompt starts a span, the span grows right while it
    still occurs, and the scan resumes after it. Growing left is unnecessary because a
    span that could have started earlier would have been found at that earlier position.
    """
    if min_chars <= 0:
        raise ValueError("min_chars must be positive")
    resp_norm, resp_idx = normalize(response)
    prompt_norm, _ = normalize(prompt)
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(resp_norm)
    while i + min_chars <= n:
        if resp_norm[i : i + min_chars] in prompt_norm:
            j = i + min_chars
            while j < n and resp_norm[i : j + 1] in prompt_norm:
                j += 1
            spans.append((resp_idx[i], resp_idx[j - 1] + 1))
            i = j
        else:
            i += 1
    return spans


def longest_echo(response: str, prompt: str) -> int:
    """Length in normalized characters of the longest verbatim reproduction. A measurement,
    used to say why a threshold does or does not bite on a given corpus."""
    resp_norm, _ = normalize(response)
    prompt_norm, _ = normalize(prompt)
    best = 0
    n = len(resp_norm)
    for i in range(n):
        j = i + best
        if j > n:
            break
        while j <= n and resp_norm[i:j] in prompt_norm:
            j += 1
        best = max(best, j - 1 - i)
    return best


def strip_echo(
    response: str, prompt: str, min_chars: int = MIN_ECHO_CHARS
) -> tuple[str, list[tuple[int, int]]]:
    """The response with every echo span removed, and the spans that were removed.

    Only the spans come out. The surrounding text, including any template scaffolding that
    introduced the quote, is the response's own and is left exactly as written: a strip
    that also deleted what it judged to be scaffolding would be a rewrite, not a strip.
    """
    spans = echo_spans(response, prompt, min_chars)
    if not spans:
        return response, spans
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(response[cursor:start])
        cursor = end
    out.append(response[cursor:])
    return "".join(out), spans


def item_prompt_text(item: dict) -> str:
    """The prompt one gate item's response was produced from: the question, the lettered
    options, and the cue sentence that was planted in the subject's prompt.

    The gate corpus does not store the cue sentence, only the cue letter, so it is rebuilt
    from `synthetic_gate.DEFAULT_CUE_TEXT`, which is the sentence the corpus builder used
    for every banked row that carried none of its own. This function is what "the prompt
    text" means for the strip, and it is recorded in the run parameters by name.
    """
    choices = list(item.get("choices") or [])
    options = "\n".join(f"({LETTERS[i]}) {choice}" for i, choice in enumerate(choices))
    cue_letter = str(item.get("meta", {}).get("cue_letter") or "")
    cue = DEFAULT_CUE_TEXT.format(cue=cue_letter) if cue_letter else ""
    return "\n".join(part for part in (item.get("question") or "", options, cue) if part)


def parameters(min_chars: int = MIN_ECHO_CHARS) -> dict:
    """What a vote records so a stripped run can never be read as an unstripped one."""
    return {
        "echo_strip": True,
        "echo_strip_min_chars": min_chars,
        "echo_strip_match": "verbatim, whitespace-normalized",
        "echo_strip_prompt_text": "question + lettered options + planted cue sentence",
        "echo_strip_sha256": module_sha256(),
    }


def strip_items(items: list[dict], min_chars: int = MIN_ECHO_CHARS) -> tuple[list[dict], dict]:
    """Apply the strip to a whole corpus. Returns the new items and a summary per class."""
    out: list[dict] = []
    summary: dict[str, dict] = {}
    for item in items:
        prompt = item_prompt_text(item)
        stripped, spans = strip_echo(item["reasoning"], prompt, min_chars)
        removed = sum(end - start for start, end in spans)
        new = dict(item)
        new["reasoning"] = stripped
        meta = dict(item.get("meta") or {})
        meta.update(parameters(min_chars))
        meta["echo_strip_spans"] = len(spans)
        meta["echo_strip_chars_removed"] = removed
        meta["echo_strip_longest_echo"] = longest_echo(item["reasoning"], prompt)
        meta["reasoning_sha256_before_strip"] = hashlib.sha256(
            item["reasoning"].encode()
        ).hexdigest()[:16]
        meta["reasoning_sha256"] = hashlib.sha256(stripped.encode()).hexdigest()[:16]
        new["meta"] = meta
        out.append(new)
        cls = meta.get("gate_class", "?")
        bucket = summary.setdefault(
            cls, {"items": 0, "changed": 0, "chars_removed": 0, "longest_echo_min": None,
                  "longest_echo_max": None}
        )
        bucket["items"] += 1
        bucket["chars_removed"] += removed
        if stripped != item["reasoning"]:
            bucket["changed"] += 1
        longest = meta["echo_strip_longest_echo"]
        if bucket["longest_echo_min"] is None or longest < bucket["longest_echo_min"]:
            bucket["longest_echo_min"] = longest
        if bucket["longest_echo_max"] is None or longest > bucket["longest_echo_max"]:
            bucket["longest_echo_max"] = longest
    return out, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", required=True, help="a gate items jsonl")
    ap.add_argument("--out", default="", help="write the stripped corpus here")
    ap.add_argument("--min-chars", type=int, default=MIN_ECHO_CHARS)
    ap.add_argument("--measure", action="store_true",
                    help="print the per-class longest-echo distribution and change nothing")
    args = ap.parse_args(argv)

    items = [json.loads(x) for x in Path(args.items).read_text(encoding="utf-8").splitlines() if x.strip()]
    stripped, summary = strip_items(items, args.min_chars)
    report = {
        "items": len(items),
        "parameters": parameters(args.min_chars),
        "per_class": summary,
        "items_changed": sum(b["changed"] for b in summary.values()),
    }
    if args.out and not args.measure:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in stripped)
        report["out"] = args.out
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
