"""Element 9.4 serving test: two reasoning-mode configurations, 30 clean items each.

DATA ONLY. This makes no ruling and writes nothing into the sweep's result tree; it
measures what each configuration would give so a ruling can be made on numbers.

  A  thinking suppressed by the model's OWN mechanism, num_predict UNCHANGED at 320.
     None of the four affected chat templates exposes an `enable_thinking` switch, so
     the mechanism available to all four is an assistant-turn prefill of a closed,
     empty reasoning block, sent with continue_final_message so the server appends it
     to the rendered prompt instead of starting a new turn.
  B  thinking ALLOWED, num_predict 4096, and the answer read from the text AFTER the
     closing think tag. The extraction lives here, in this file, and the FROZEN
     parser is what reads the extracted tail.

Reported per model and configuration: clean accuracy, parse rate, mean and max
completion tokens (from the server's own usage block, not a re-tokenization),
seconds per item at concurrency 32, and the ruling-R1 determinism verdict measured
UNDER THAT CONFIGURATION with bcf/concurrency_probe.py's own run_level/compare.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bcf"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

import concurrency_probe
from openai_client import OpenAIClient

from bayes_cot_faithfulness.interventions import clean_prompt, parse_answer

CLOSE_THINK = re.compile(r"<\s*/\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"


def after_think(text: str) -> str:
    """Configuration B's extraction: everything after the LAST closing think tag.

    Returns '' when no closing tag exists, so a generation that never left the
    reasoning block scores as unparseable rather than being read mid-chain.
    """
    hits = list(CLOSE_THINK.finditer(text or ""))
    return (text or "")[hits[-1].end():] if hits else ""


class PrefillClient(OpenAIClient):
    """OpenAIClient whose generations continue a prefilled, CLOSED reasoning block.

    Why not an extra assistant message with ``continue_final_message``. Two of the four
    affected chat templates rewrite an assistant message before rendering it: the
    DeepSeek R1 templates apply ``content.split('</think>')[-1]``, so a prefill of
    "<think>\n\n</think>\n\n" arrives at the server as "\n\n" and the empty block never
    reaches the prompt. The prefill would be silently dropped and the run would look like
    a measurement of thinking-suppressed decoding when it was nothing of the kind.

    So config A renders the prompt with the model's OWN template through vLLM's
    /tokenize (add_generation_prompt=true) and /detokenize, appends the closing tail, and
    generates from /completions. Two of the templates
    (Olmo-3-7B-Think, R1-Distill-Llama-8B) already end the rendered prompt with an OPEN
    ``<think>``; there the tail is only the closing tag, which is the model's own
    mechanism used as intended. Where no open tag is present the whole empty block is
    inserted. Which branch was taken is recorded per run, with the rendered prompt tail.
    """

    prefill_mode: str = "closed_think"   # or "" for no prefill (config B)
    _root = None
    _branch: str | None = None
    _rendered_tail: str | None = None

    def _root_client(self):
        """A sibling client pinned to the server ROOT: vLLM serves /tokenize and
        /detokenize there, not under /v1. A separate instance rather than mutating
        base_url, because _render runs on 32 threads at once."""
        if getattr(self, "_root", None) is None:
            self._root = OpenAIClient(
                base_url=self.root_url, model=self.model, timeout=self.timeout,
                max_in_flight=0,
            )
        return self._root

    def _tok_post(self, path: str, payload: dict) -> dict:
        try:
            return self._root_client()._post(path, payload)
        except Exception:  # noqa: BLE001 - some builds mount them under /v1 as well
            return self._post(path, payload)

    def _render(self, prompt: str) -> str:
        body = self._tok_post("/tokenize", {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            **({"chat_template_kwargs": dict(self.chat_template_kwargs)}
               if self.chat_template_kwargs else {}),
        })
        ids = body.get("tokens")
        if not ids:
            raise RuntimeError(f"/tokenize returned no tokens: {str(body)[:200]}")
        back = self._tok_post("/detokenize", {"model": self.model, "tokens": ids})
        text = back.get("prompt")
        if text is None:
            raise RuntimeError(f"/detokenize returned no prompt: {str(back)[:200]}")
        return text

    def _prefixed(self, prompt: str) -> str:
        rendered = self._render(prompt)
        open_tag = re.search(r"<\s*think\s*>", rendered, re.IGNORECASE)
        close_tag = CLOSE_THINK.search(rendered)
        if open_tag and not (close_tag and close_tag.start() > open_tag.start()):
            tail = "\n\n</think>\n\n"
            branch = "closed_the_block_the_template_opened"
        else:
            tail = EMPTY_THINK_PREFILL
            branch = "inserted_a_whole_empty_block"
        self._branch = branch
        self._rendered_tail = (rendered + tail)[-200:]
        return rendered + tail

    def _complete(self, prompt: str, *, num_predict: int) -> dict:
        body = {
            "model": self.model, "prompt": prompt, "temperature": self.temperature,
            "max_tokens": num_predict, "n": 1,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        return self._post("/completions", body)

    def generate_many(self, prompt, *, num_predict=320, system=None, stop=None):
        if not self.prefill_mode:
            return super().generate_many(prompt, num_predict=num_predict,
                                         system=system, stop=stop)
        body = self._complete(self._prefixed(prompt), num_predict=num_predict)
        return [c.get("text") or "" for c in body["choices"]]

    def generate_with_usage(self, prompt, *, num_predict):
        if not self.prefill_mode:
            body = self._post("/chat/completions", self._chat_body(
                [{"role": "user", "content": prompt}], num_predict=num_predict))
            text = body["choices"][0]["message"]["content"] or ""
        else:
            body = self._complete(self._prefixed(prompt), num_predict=num_predict)
            text = body["choices"][0].get("text") or ""
        usage = body.get("usage") or {}
        return text, usage.get("completion_tokens"), body["choices"][0].get("finish_reason")


def build_factory(a, prefill, num_predict, template_kwargs):
    def factory(level):
        c = PrefillClient(
            base_url=a.base_url, model=a.model, seed=a.seed, timeout=a.timeout,
            chat_template_kwargs=template_kwargs, max_in_flight=level,
        )
        c.prefill_mode = prefill
        return c
    return factory


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", choices=["A", "B"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--num-predict", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--prefill", default="", help="assistant-turn prefill for config A")
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-preflight", action="store_true")
    a = ap.parse_args(argv)

    tk = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None
    items = concurrency_probe.load_items(a.data, a.n_items)
    ARMS = concurrency_probe.ARMS
    prefill_mode = "closed_think" if a.config == "A" else ""
    factory = build_factory(a, prefill_mode, a.num_predict, tk)

    out = {
        "model": a.model, "config": a.config, "num_predict": a.num_predict,
        "n_items": len(items), "concurrency": a.concurrency, "seed": a.seed,
        "chat_template_kwargs": tk, "prefill_mode": prefill_mode,
        "extraction": "after the last closing think tag" if a.config == "B" else "whole completion",
    }

    # --- the accuracy / parse / token pass, at the configured concurrency ---
    client = factory(a.concurrency)
    rows: list[dict] = []
    t0 = time.time()
    ARMS.map_in_order(
        items,
        lambda i, it: client.generate_with_usage(clean_prompt(it), num_predict=a.num_predict),
        concurrency=a.concurrency,
        consume=lambda i, it, r: (rows.append({
            "text": r[0], "completion_tokens": r[1], "finish_reason": r[2],
            "gold": it.answer_label, "n_choices": len(it.choices),
        }), True)[1],
    )
    seconds = time.time() - t0

    n_parsed = n_correct = 0
    toks = []
    for r in rows:
        scored = after_think(r["text"]) if a.config == "B" else r["text"]
        ans = parse_answer(scored, r["n_choices"])
        r["answer"] = ans
        r["correct"] = (ans == r["gold"])
        n_parsed += ans is not None
        n_correct += r["correct"]
        if r["completion_tokens"] is not None:
            toks.append(r["completion_tokens"])
    n = len(rows)
    out["pass"] = {
        "n": n,
        "n_parsed": n_parsed,
        "parse_rate": round(n_parsed / n, 6) if n else None,
        "n_correct": n_correct,
        "clean_accuracy": round(n_correct / n, 6) if n else None,
        "seconds_total": round(seconds, 3),
        "seconds_per_item": round(seconds / n, 4) if n else None,
        "mean_completion_tokens": round(sum(toks) / len(toks), 2) if toks else None,
        "max_completion_tokens": max(toks) if toks else None,
        "n_finish_length": sum(1 for r in rows if r["finish_reason"] == "length"),
        "n_closing_think_tag": sum(1 for r in rows if CLOSE_THINK.search(r["text"] or "")),
        "client_stats": client.stats(),
        "prefill_branch": client._branch,
        "rendered_prompt_tail": client._rendered_tail,
    }
    out["rows"] = [{k: v for k, v in r.items() if k != "text"} | {"tail": (r["text"] or "")[-200:]}
                   for r in rows]

    # --- ruling R1, measured under THIS configuration with the campaign's own probe ---
    if not a.skip_preflight:
        levels = [1, a.concurrency]
        measured = {}
        for lv in levels:
            measured[lv] = concurrency_probe.run_level(factory, items, lv, a.num_predict)
        base = measured[levels[0]]
        rows_pf = []
        for lv in levels:
            cmp = concurrency_probe.compare(base, measured[lv])
            cmp["concurrency"] = lv
            cmp["generations_per_second"] = measured[lv]["generations_per_second"]
            rows_pf.append(cmp)
        ok = all(
            r["identical_completions"] == f"{len(items)}/{len(items)}"
            and r["max_abs_letter_logprob_diff"] == 0.0
            and r["n_letter_logprobs_missing"] == 0
            for r in rows_pf
        )
        out["r1_preflight"] = {
            "rule": "30/30 identical completions and max abs letter-logprob diff exactly "
                    "0.0 at 1 and at the run concurrency, VLLM_BATCH_INVARIANT=1",
            "verdict": "PASS" if ok else "REFUSE",
            "levels": rows_pf,
        }
    else:
        out["r1_preflight"] = None

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    p = out["pass"]
    print(f"[serving-test] {a.model} config {a.config}: accuracy {p['n_correct']}/{p['n']} "
          f"parse {p['n_parsed']}/{p['n']} tokens mean {p['mean_completion_tokens']} "
          f"max {p['max_completion_tokens']} s/item {p['seconds_per_item']} "
          f"R1 {(out['r1_preflight'] or {}).get('verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
