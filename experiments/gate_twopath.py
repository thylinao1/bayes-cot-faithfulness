"""RULING R12(2): the two-path equivalence gate, run once before any reasoning_mode
"off" cell becomes a cell of record.

R12 defines the reasoning switch, for a row whose template documents none, as closing the
block the template opens, and that mechanism needs the prompt rendered through the model's
own template and generated through ``/completions`` instead of ``/chat/completions``. A
change of request path is a change of serving mode unless it is shown not to be, and this
script is that showing: 30 ARC items on one live server, each sent BOTH ways with
reasoning_mode DEFAULT semantics (nothing is closed, nothing is appended), so the two
paths should render the same prompt and return the same bytes.

Three quantities, each with its denominator:

  identical completions        the chat completion and the rendered-/completions
                               completion, byte for byte, per item
  max abs letter-logprob diff  over every item and every answer letter, the forced-answer
                               letter logprobs read on the chat path against the same
                               read on the rendered path
  rendered prompt round trip   /tokenize(messages) -> /detokenize -> /tokenize(text)
                               returns the same token ids. This is the check the
                               R1-0528-Qwen3-8B defect fails (docs/REASONING-MODE-TEST.md
                               section 3): a prompt whose text does not round-trip cannot
                               be compared with anything, so a difference measured over it
                               would mean nothing either way.

PASS is 30/30 identical, a maximum absolute letter-logprob difference of exactly 0.0 with
no letter missing on either path, and a clean round trip on every item. Anything else is
REFUSE: the path change is then a serving-mode change and needs its own preflight line.
Exit 0 on PASS, 10 on REFUSE, which is the exit code the campaign already reserves for a
determinism refusal.

    python experiments/gate_twopath.py --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-8B --data experiments/data/arc_challenge.json \
        --n-items 30 --out gate_twopath.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bcf"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

import concurrency_probe
from openai_client import (
    OpenAIClient,
    OpenAIClientError,
    _read_letter_from_prompt_logprobs,
)

from bayes_cot_faithfulness.arms import replay_prompt
from bayes_cot_faithfulness.interventions import clean_prompt

ARMS = concurrency_probe.ARMS
REFUSE_EXIT = 10
# The same forced-continuation frame the determinism preflight reads its letter logprobs
# on, so the number this gate reports is comparable with the numbers already on the record.
LOGPROB_CHAIN = "1. work it through\n2. the answer follows"


class TwoPathClient(OpenAIClient):
    """One client, both request paths, so nothing but the path differs between them.

    Same base_url, same model, same seed, same temperature, same chat_template_kwargs.
    The rendered path adds NOTHING to the rendered prompt: this gate runs
    reasoning_mode DEFAULT semantics on purpose, because a gate that also closed a
    reasoning block would be measuring two changes at once and could not attribute a
    difference to either.
    """

    _root_client_cache = None

    def _root_post(self, path: str, payload: dict) -> dict:
        """vLLM serves /tokenize and /detokenize at the ROOT, not under /v1."""
        if self._root_client_cache is None:
            self._root_client_cache = OpenAIClient(
                base_url=self.root_url, model=self.model, timeout=self.timeout,
                max_in_flight=0,
            )
        try:
            return self._root_client_cache._post(path, payload)
        except OpenAIClientError:
            return self._post(path, payload)

    def _tokenize(self, payload: dict) -> list[int]:
        body = self._root_post("/tokenize", {"model": self.model, **payload})
        ids = body.get("tokens")
        if not ids:
            raise OpenAIClientError(f"/tokenize returned no tokens: {str(body)[:200]}")
        return list(ids)

    def _detokenize(self, ids: list[int]) -> str:
        body = self._root_post("/detokenize", {"model": self.model, "tokens": ids})
        text = body.get("prompt")
        if text is None:
            raise OpenAIClientError(f"/detokenize returned no prompt: {str(body)[:200]}")
        return text

    def render(self, messages: list[dict], **extra) -> dict:
        """Render ``messages`` through the model's own template and round-trip it."""
        payload = {"messages": messages, **extra}
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        ids = self._tokenize(payload)
        text = self._detokenize(ids)
        again = self._tokenize({"prompt": text})
        return {"tokens": ids, "text": text, "retokenized": again,
                "roundtrip_ok": again == ids}

    # --- generation, both ways -------------------------------------------------
    def chat_generation(self, prompt: str, *, num_predict: int) -> str:
        body = self._post("/chat/completions", self._chat_body(
            [{"role": "user", "content": prompt}], num_predict=num_predict))
        return body["choices"][0]["message"]["content"] or ""

    def rendered_generation(self, prompt: str, *, num_predict: int) -> tuple[str, dict]:
        rendered = self.render([{"role": "user", "content": prompt}],
                               add_generation_prompt=True)
        payload = {
            "model": self.model, "prompt": rendered["text"],
            "temperature": self.temperature, "max_tokens": num_predict, "n": 1,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        body = self._post("/completions", payload)
        return (body["choices"][0].get("text") or ""), rendered

    # --- forced-answer letter logprobs, both ways ------------------------------
    def chat_letter_logprobs(self, prefix: str, letters: list[str]) -> dict[str, float]:
        return dict(self.forced_answer_logprobs(prefix, letters).logprobs)

    def rendered_letter_logprobs(self, prefix: str, letters: list[str]) -> dict[str, float]:
        """The same read, on the rendered path: score the prompt, letter included.

        The chat path sends the letter as an unfinished assistant turn with
        ``continue_final_message``; here the identical message list is rendered through
        /tokenize with the identical flags and the resulting text is scored by
        /completions. Same tokens, different endpoint, which is the whole question.
        """
        out: dict[str, float] = {}
        for letter in letters:
            rendered = self.render(
                [{"role": "user", "content": prefix},
                 {"role": "assistant", "content": letter}],
                add_generation_prompt=False, continue_final_message=True,
            )
            body = self._post("/completions", {
                "model": self.model, "prompt": rendered["text"], "temperature": 0.0,
                "max_tokens": 1, "n": 1, "prompt_logprobs": 0,
            })
            choice = (body.get("choices") or [{}])[0]
            scored = choice.get("prompt_logprobs") or body.get("prompt_logprobs")
            if not scored:
                raise OpenAIClientError(
                    "/completions returned no prompt_logprobs; the rendered path cannot "
                    f"be compared with the chat path. body={str(body)[:200]}"
                )
            value, _token = _read_letter_from_prompt_logprobs(scored, letter)
            out[letter] = value
        return out


def _fraction(n_ok: int, n: int) -> dict:
    return {"count": f"{n_ok}/{n}",
            "fraction": round(n_ok / n, 6) if n else None}


def run_gate(client: TwoPathClient, items, *, num_predict: int, concurrency: int) -> dict:
    """Both paths over every item; returns the report dict (no verdict yet)."""
    prompts = [clean_prompt(it) for it in items]
    lp_prompts = [replay_prompt(it, LOGPROB_CHAIN) for it in items]
    letters = [list(it.labels) for it in items]

    t0 = time.time()
    chat: list[str] = []
    ARMS.map_in_order(
        prompts, lambda i, p: client.chat_generation(p, num_predict=num_predict),
        concurrency=concurrency,
        consume=lambda i, p, r: (chat.append(r), True)[1],
    )
    chat_seconds = time.time() - t0

    t0 = time.time()
    rendered: list[tuple[str, dict]] = []
    ARMS.map_in_order(
        prompts, lambda i, p: client.rendered_generation(p, num_predict=num_predict),
        concurrency=concurrency,
        consume=lambda i, p, r: (rendered.append(r), True)[1],
    )
    rendered_seconds = time.time() - t0

    rows = []
    n_identical = n_roundtrip = 0
    for i, (item_chat, (item_rendered, render_meta)) in enumerate(zip(chat, rendered)):
        same = item_chat == item_rendered
        n_identical += same
        n_roundtrip += bool(render_meta["roundtrip_ok"])
        rows.append({
            "i": i,
            "identical": same,
            "roundtrip_ok": render_meta["roundtrip_ok"],
            "n_prompt_tokens": len(render_meta["tokens"]),
            "rendered_prompt_tail": render_meta["text"][-200:],
            "chat_tail": (item_chat or "")[-120:],
            "rendered_tail": (item_rendered or "")[-120:],
        })

    t0 = time.time()
    chat_lp: list[dict] = []
    ARMS.map_in_order(
        list(range(len(items))),
        lambda i, idx: client.chat_letter_logprobs(lp_prompts[idx], letters[idx]),
        concurrency=concurrency,
        consume=lambda i, idx, r: (chat_lp.append(r), True)[1],
    )
    rendered_lp: list[dict] = []
    ARMS.map_in_order(
        list(range(len(items))),
        lambda i, idx: client.rendered_letter_logprobs(lp_prompts[idx], letters[idx]),
        concurrency=concurrency,
        consume=lambda i, idx, r: (rendered_lp.append(r), True)[1],
    )
    logprob_seconds = time.time() - t0

    diffs: list[float] = []
    n_missing = 0
    for a, b in zip(chat_lp, rendered_lp):
        for letter, value in a.items():
            if letter not in b:
                n_missing += 1
                continue
            diffs.append(abs(value - b[letter]))

    n = len(items)
    return {
        "n_items": n,
        "num_predict": num_predict,
        "concurrency": concurrency,
        "identical_completions": _fraction(n_identical, n),
        "rendered_prompt_roundtrip": _fraction(n_roundtrip, n),
        "n_letter_logprobs_compared": len(diffs),
        "n_letter_logprobs_missing": n_missing,
        "max_abs_letter_logprob_diff": max(diffs) if diffs else None,
        "median_abs_letter_logprob_diff": statistics.median(diffs) if diffs else None,
        "n_exactly_zero_diff": sum(1 for d in diffs if d == 0.0),
        "seconds": {"chat": round(chat_seconds, 3),
                    "rendered": round(rendered_seconds, 3),
                    "logprobs": round(logprob_seconds, 3)},
        "client_stats": client.stats(),
        "rows": rows,
    }


def verdict_for(report: dict) -> tuple[str, list[dict]]:
    """The three conditions, each reported with what it saw, then the verdict."""
    n = report["n_items"]
    checks = [
        {"check": "identical completions on both paths",
         "want": f"{n}/{n}", "got": report["identical_completions"]["count"],
         "ok": report["identical_completions"]["count"] == f"{n}/{n}"},
        {"check": "max abs letter-logprob difference",
         "want": "0.0", "got": report["max_abs_letter_logprob_diff"],
         "ok": (report["max_abs_letter_logprob_diff"] == 0.0
                and report["n_letter_logprobs_missing"] == 0
                and report["n_letter_logprobs_compared"] > 0)},
        {"check": "rendered prompt round-trips through the tokenizer",
         "want": f"{n}/{n}", "got": report["rendered_prompt_roundtrip"]["count"],
         "ok": report["rendered_prompt_roundtrip"]["count"] == f"{n}/{n}"},
    ]
    return ("PASS" if all(c["ok"] for c in checks) else "REFUSE"), checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--num-predict", type=int, default=320)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    tk = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
    items = concurrency_probe.load_items(a.data, a.n_items)
    client = TwoPathClient(
        base_url=a.base_url, model=a.model, temperature=0.0, seed=a.seed,
        timeout=a.timeout, chat_template_kwargs=tk, max_in_flight=a.concurrency,
    )

    out = {
        "gate": "R12(2) two-path equivalence",
        "rule": "30/30 byte-identical completions, max abs letter-logprob difference "
                "exactly 0.0 with no letter missing, and a clean rendered-prompt round "
                "trip. Anything else means the request path change is a serving-mode "
                "change and needs its own preflight line.",
        "model": a.model,
        "data": str(a.data),
        "seed": a.seed,
        "chat_template_kwargs": tk,
        "reasoning_mode_semantics": "default (nothing closed, nothing appended, so the "
                                    "two paths should render the same prompt)",
    }
    try:
        report = run_gate(client, items, num_predict=a.num_predict,
                          concurrency=a.concurrency)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot measure must REFUSE
        out["verdict"] = "REFUSE"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["checks"] = []
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        print(f"[gate-twopath] REFUSE: {out['error']}")
        return REFUSE_EXIT

    out.update(report)
    out["verdict"], out["checks"] = verdict_for(report)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"[gate-twopath] {out['verdict']}: identical "
          f"{report['identical_completions']['count']}, max abs letter-logprob diff "
          f"{report['max_abs_letter_logprob_diff']}, round trip "
          f"{report['rendered_prompt_roundtrip']['count']}")
    for check in out["checks"]:
        if not check["ok"]:
            print(f"[gate-twopath]   FAILED: {check['check']} "
                  f"(want {check['want']}, got {check['got']})")
    return 0 if out["verdict"] == "PASS" else REFUSE_EXIT


if __name__ == "__main__":
    sys.exit(main())
