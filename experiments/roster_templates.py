"""Element 10 roster: fetch each row's chat template at its pinned revision from the
Hugging Face Hub, classify it, and write experiments/results/roster-templates/templates.json.

READ ONLY against the Hub. Fetches `tokenizer_config.json` for each of the 18 roster
rows in `experiments/PREREGISTRATION_jury_and_scale.md` section 11 (not edited here,
and not re-read at runtime: the 18 rows are pinned into ROSTER below so a re-run is
deterministic against a revision even if that section is amended later), and
`chat_template.jinja` as well whenever the repo carries one (the config's own
`chat_template` field is `null` on several 2025-era repos that moved the template out
to that sibling file; where both exist this script fetches both and records both
hashes). A gated repo without an accepted licence, or a repo missing the file
entirely (Mistral's two `tekken.json` rows, which carry no HF chat template at all),
is recorded as a FAILED fetch with the error named, never inferred from a sibling row.

Classification is a REGEX read of the template's OWN TEXT, not a live Jinja render
against a server (the cluster link is down for this lane; `docs/REASONING-MODE-TEST.md`
section 3 already measured live behaviour for the four rows this matters most for).
The method, in three passes over the template text:

  1. Find the tail: `text[text.rfind("add_generation_prompt"):]`. Every template this
     lane read places its generation-prompt logic at the very end, after the message
     loop, so the tail is the fragment that decides what a fresh generation prompt
     looks like.
  2. In the tail, look for a `NAME is defined` (or `NAME | default(`) guard whose NAME
     contains "think" or "reason" AND which reappears elsewhere in the tail (ruling out
     a name that is only discussed earlier in the template, which is what keeps
     gpt-oss's `reasoning_effort` — injected into a system message far from the
     generation prompt — out of this class). A match is CLASS 1: a documented switch.
     Its default is read off the guard's polarity: a guard of the shape
     "NAME is defined and NAME is false" (or "and not NAME") fires on the OFF state,
     so the UNDEFINED state (the default) is the opposite, ON; a guard of the reverse
     polarity gives the reverse default. An unrecognised polarity is recorded as
     unknown rather than guessed.
  3. Failing that, look for a bare `<think>` / `<thinking>` / `<reasoning>` opened in
     the tail with no matching close in the same tail: CLASS 2, the tag named.
  4. Failing that, a "think" or "reason" token ANYWHERE in the full template (not just
     the tail) is CLASS 4, other — the two DeepSeek-R1-0528-Qwen3-8B and
     Phi-4-reasoning rows are exactly this: the generation-prompt tail itself is bare
     (no switch, no open tag), and what makes them reasoning templates lives elsewhere
     (an assistant-message `.split('</think>')` rewrite for the first, a hardcoded
     system prompt demanding a Thought section for the second). A naive classifier that
     only reads the tail would call both of these CLASS 3, which is exactly the kind of
     silent misclassification amendment A4.4 / ruling R12 exists to stop; this is why
     the catch-all reads the FULL template rather than the tail alone.
  5. No token anywhere: CLASS 3, plain instruct.

`classify_chat_template` is the pure function tested in isolation by
tests/test_roster_templates.py; `fetch_row` is the only network-touching function and
is not exercised by the tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / "experiments" / "results" / "roster-templates" / "templates.json"

# Element 10's 18-row roster, `experiments/PREREGISTRATION_jury_and_scale.md` section 11
# (2026-09-07 revisions). Pinned here rather than parsed from that file at runtime: the
# file is frozen against this lane's edits, and pinning makes a re-run reproduce the
# same 18 fetches even if a later amendment revises the table.
ROSTER: list[dict[str, str | int]] = [
    {"index": 1, "model": "Qwen/Qwen3-8B",
     "revision": "b968826d9c46dd6066d109eabc6255188de91218", "family": "Qwen"},
    {"index": 2, "model": "Qwen/Qwen3-32B",
     "revision": "9216db5781bf21249d130ec9da846c4624c16137", "family": "Qwen"},
    {"index": 3, "model": "Qwen/Qwen3.6-35B-A3B",
     "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0", "family": "Qwen"},
    {"index": 4, "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
     "revision": "6a6f4aa4197940add57724a7707d069478df56b1", "family": "Llama"},
    {"index": 5, "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
     "revision": "b1c0b44b4369b597ad119a196caf79a9c40e141e", "family": "Llama"},
    {"index": 6, "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
     "revision": "6e8885a6ff5c1dc5201574c8fd700323f23c25fa", "family": "Qwen"},
    {"index": 7, "model": "openai/gpt-oss-20b",
     "revision": "6cee5e81ee83917806bbde320786a8fb61efebee", "family": "gpt-oss"},
    {"index": 8, "model": "openai/gpt-oss-120b",
     "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a", "family": "gpt-oss"},
    {"index": 9, "model": "allenai/Olmo-3-7B-Think",
     "revision": "d97e442d7cc678210054dbcc9b440894d62c89a4", "family": "OLMo"},
    {"index": 10, "model": "allenai/Olmo-3-32B-Think",
     "revision": "f2edda15216e738ef2bb73771e11890e152b2112", "family": "OLMo"},
    {"index": 11, "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
     "revision": "95a6d26c4bfb886c58daf9d3f7332c857cb27b43", "family": "Mistral"},
    {"index": 12, "model": "microsoft/Phi-4-reasoning",
     "revision": "1de18ec97600877ce63dbf60c73b998da99f0195", "family": "Phi"},
    {"index": 13, "model": "zai-org/GLM-4.5-Air",
     "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa", "family": "GLM"},
    {"index": 14, "model": "meta-llama/Llama-3.1-8B-Instruct",
     "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "family": "Llama"},
    {"index": 15, "model": "meta-llama/Llama-3.3-70B-Instruct",
     "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "family": "Llama"},
    {"index": 16, "model": "google/gemma-2-9b-it",
     "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819", "family": "Gemma"},
    {"index": 17, "model": "google/gemma-3-27b-it",
     "revision": "005ad3404e59d6023443cb575daa05336842228a", "family": "Gemma"},
    {"index": 18, "model": "mistralai/Magistral-Small-2509",
     "revision": "a31cc96ab10cf19bc42c628fedf1e359e0853c49", "family": "Mistral"},
]

AGP_TOKEN = "add_generation_prompt"
REASONING_NAME_RE = re.compile(r"think|reason", re.IGNORECASE)
SWITCH_GUARD_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s+is\s+defined\b|"
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\|\s*default\(",
    re.IGNORECASE,
)
OPEN_TAG_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
CLOSE_TAG_RE = re.compile(r"<\s*/\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
ASSISTANT_REWRITE_RE = re.compile(
    r"\.split\(\s*(['\"])<\s*/\s*\w+\s*>\1\s*\)(?:\[-?\d+\])?"
)

CLASS_LABELS = {
    1: "documented reasoning switch",
    2: "opens a reasoning block in the generation prompt, no switch",
    3: "plain instruct template",
    4: "other",
}


def _switch_default(tail: str, name: str) -> bool | None:
    """Infer whether the default (kwarg absent) state is thinking ON or OFF.

    Reads the polarity of the guard: a guard that fires on the FALSE/falsy state of
    the switch (the common HF convention, `enable_thinking is defined and
    enable_thinking is false`, or `and not enable_thinking`) means the undefined
    state behaves like TRUE, so the default is thinking ON. The reverse polarity
    gives the reverse default. Returns None, not a guess, when neither polarity
    is recognised.
    """
    esc = re.escape(name)
    if re.search(rf"{esc}\s+is\s+false", tail, re.IGNORECASE) or re.search(
        rf"not\s+{esc}\b", tail, re.IGNORECASE
    ):
        return True
    if re.search(rf"{esc}\s+is\s+true", tail, re.IGNORECASE):
        return False
    return None


def classify_chat_template(text: str) -> dict[str, Any]:
    """Classify one chat template's TEXT. Pure function, no network.

    Returns a dict with `class` (1-4), `class_label`, `switch_kwarg`,
    `switch_default`, `opened_tag`, and `decisive_excerpt` (the tail fragment, or
    the full text when the catch-all class-4 path fires, capped at 400 chars so
    the JSON stays small; the markdown doc quotes hand-picked lines instead of
    this excerpt).
    """
    if not text:
        return {
            "class": None, "class_label": "no template text", "switch_kwarg": None,
            "switch_default": None, "opened_tag": None, "decisive_excerpt": None,
        }
    idx = text.rfind(AGP_TOKEN)
    tail = text[idx:] if idx != -1 else text

    candidates: list[str] = []
    for m in SWITCH_GUARD_RE.finditer(text):
        name = m.group(1) or m.group(2)
        if name and REASONING_NAME_RE.search(name) and name in tail and name not in candidates:
            candidates.append(name)
    if candidates:
        name = candidates[0]
        return {
            "class": 1, "class_label": CLASS_LABELS[1], "switch_kwarg": name,
            "switch_default": _switch_default(tail, name), "opened_tag": None,
            "decisive_excerpt": tail[:400],
        }

    opens = list(OPEN_TAG_RE.finditer(tail))
    closes = list(CLOSE_TAG_RE.finditer(tail))
    if opens and not (closes and closes[-1].start() > opens[-1].start()):
        tag = f"<{opens[-1].group(1).lower()}>"
        return {
            "class": 2, "class_label": CLASS_LABELS[2], "switch_kwarg": None,
            "switch_default": None, "opened_tag": tag, "decisive_excerpt": tail[:400],
        }

    if REASONING_NAME_RE.search(text):
        return {
            "class": 4, "class_label": CLASS_LABELS[4], "switch_kwarg": None,
            "switch_default": None, "opened_tag": None,
            "decisive_excerpt": tail[:400] or text[:400],
        }

    return {
        "class": 3, "class_label": CLASS_LABELS[3], "switch_kwarg": None,
        "switch_default": None, "opened_tag": None, "decisive_excerpt": tail[:400],
    }


def detect_assistant_rewrite(text: str) -> dict[str, Any]:
    """Whether the template rewrites a prior assistant message by splitting off a
    closing reasoning tag (`content.split('</think>')[-1]` and its kin), which is
    what makes an assistant-turn PREFILL through `/chat/completions` arrive
    stripped, per `docs/REASONING-MODE-IMPL.md` section 1."""
    if not text:
        return {"rewrites": None, "evidence": None}
    m = ASSISTANT_REWRITE_RE.search(text)
    if not m:
        return {"rewrites": False, "evidence": None}
    return {"rewrites": True, "evidence": m.group(0)}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_row(row: dict, token: str | None) -> dict[str, Any]:
    """Fetch one roster row's tokenizer_config.json (and chat_template.jinja when the
    repo carries one) at its pinned revision. Never raises: every failure mode is
    written into the returned dict's `fetch_status` and `fetch_error` instead, so one
    bad row does not stop the other seventeen."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    repo_id = row["model"]
    revision = row["revision"]
    out: dict[str, Any] = {
        "index": row["index"], "model": repo_id, "revision": revision,
        "family": row["family"], "fetch_status": None, "fetch_error": None,
        "tokenizer_config_sha256": None, "chat_template_jinja_sha256": None,
        "template_source": None, "template_text": None,
    }

    try:
        files = list_repo_files(repo_id, revision=revision, token=token)
    except GatedRepoError as exc:
        out["fetch_status"] = "gated_no_access"
        out["fetch_error"] = f"{type(exc).__name__}: {exc}"
        return out
    except HfHubHTTPError as exc:
        out["fetch_status"] = "error"
        out["fetch_error"] = f"{type(exc).__name__}: {exc}"
        return out
    except Exception as exc:  # noqa: BLE001 - a listing failure is recorded, not raised
        out["fetch_status"] = "error"
        out["fetch_error"] = f"{type(exc).__name__}: {exc}"
        return out

    has_config = "tokenizer_config.json" in files
    has_jinja = "chat_template.jinja" in files
    if not has_config and not has_jinja:
        out["fetch_status"] = "missing_file"
        out["fetch_error"] = (
            "neither tokenizer_config.json nor chat_template.jinja is in this repo "
            f"at {revision} ({len(files)} files listed)"
        )
        return out

    inline_template: str | None = None
    if has_config:
        try:
            p = Path(hf_hub_download(
                repo_id=repo_id, filename="tokenizer_config.json", revision=revision,
                token=token,
            ))
            raw = p.read_bytes()
            out["tokenizer_config_sha256"] = sha256_bytes(raw)
            cfg = json.loads(raw)
            ct = cfg.get("chat_template")
            if isinstance(ct, str):
                inline_template = ct
            elif isinstance(ct, list):
                # A small number of older repos store a list of {name, template}
                # objects; the default (unnamed or "default") entry is what a
                # generation call with no template name gets.
                named = {d.get("name"): d.get("template") for d in ct if isinstance(d, dict)}
                inline_template = named.get("default") or (
                    next(iter(named.values())) if named else None
                )
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            out["fetch_status"] = "error"
            out["fetch_error"] = f"tokenizer_config.json fetch failed: {type(exc).__name__}: {exc}"
            return out

    jinja_template: str | None = None
    if has_jinja:
        try:
            p = Path(hf_hub_download(
                repo_id=repo_id, filename="chat_template.jinja", revision=revision,
                token=token,
            ))
            raw = p.read_bytes()
            out["chat_template_jinja_sha256"] = sha256_bytes(raw)
            jinja_template = raw.decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            out["fetch_status"] = "error"
            out["fetch_error"] = f"chat_template.jinja fetch failed: {type(exc).__name__}: {exc}"
            return out

    if inline_template is None and jinja_template is None:
        out["fetch_status"] = "missing_file"
        out["fetch_error"] = "tokenizer_config.json carries no chat_template and no chat_template.jinja exists"
        return out

    # Prefer the external file when both exist: it is the file the newer HF
    # convention treats as authoritative, and this lane found it byte-identical
    # to the inline copy on every row that carried both (Qwen3.6-35B-A3B).
    if jinja_template is not None:
        out["template_source"] = "external_jinja" if inline_template is None else "both_identical"
        out["template_text"] = jinja_template
        if inline_template is not None and inline_template != jinja_template:
            out["template_source"] = "both_DIFFERED"
    else:
        out["template_source"] = "inline"
        out["template_text"] = inline_template

    out["fetch_status"] = "ok"
    return out


def build_records(token: str | None) -> list[dict[str, Any]]:
    records = []
    for row in ROSTER:
        fetched = fetch_row(row, token)
        rec: dict[str, Any] = dict(fetched)
        if fetched["fetch_status"] == "ok" and fetched["template_text"]:
            cls = classify_chat_template(fetched["template_text"])
            rw = detect_assistant_rewrite(fetched["template_text"])
        else:
            cls = {"class": None, "class_label": None, "switch_kwarg": None,
                   "switch_default": None, "opened_tag": None, "decisive_excerpt": None}
            rw = {"rewrites": None, "evidence": None}
        rec["classification"] = cls
        rec["assistant_rewrite"] = rw
        rec.pop("template_text", None)  # kept out of the JSON; the doc quotes lines by hand
        records.append(rec)
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    records = build_records(token)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "roster_source": "experiments/PREREGISTRATION_jury_and_scale.md section 11 (element 10)",
        "n_rows": len(records),
        "n_ok": sum(1 for r in records if r["fetch_status"] == "ok"),
        "n_failed": sum(1 for r in records if r["fetch_status"] != "ok"),
        "rows": records,
    }
    a.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")

    for r in records:
        cls = r["classification"]["class"]
        print(f"[{r['index']:2d}] {r['model']:55s} fetch={r['fetch_status']:16s} class={cls}")
    print(f"\n{payload['n_ok']}/{payload['n_rows']} rows fetched OK -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
