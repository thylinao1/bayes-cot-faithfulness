"""Ruling R12 reasoning_mode "off": render through the model's own template, close the
reasoning block, generate through ``/completions``.

Only mode "off" needs a client of its own. "default" is today's ``OpenAIClient``,
untouched, and "on" is also today's ``OpenAIClient``: R12's configuration B differs from
a default cell in the FULL-generation budget (set by the runner) and in the answer
extraction (``reasoning_mode.parse_answer_after_think``), not in anything this client
puts on the wire. Building a subclass for those two modes would put new code in the path
of every existing cell for no behavioural reason, so ``_gate_client`` does not.

Why the request path has to move for "off". The mechanism R12 defines is closing the
reasoning block the chat template opens, and the block sits at the very END of the
rendered prompt. Sending it as a prefilled assistant turn does not work: two of the four
affected templates rewrite an assistant message with ``content.split('</think>')[-1]``
before rendering it, so a prefill of ``<think>\\n\\n</think>\\n\\n`` arrives at the server
as ``\\n\\n`` and the empty block never reaches the prompt. The run would look like a
measurement of thinking-suppressed decoding and be nothing of the kind. So the prompt is
rendered with the model's OWN template through vLLM's ``/tokenize`` (which applies the
template server side) and ``/detokenize``, the closing tag is appended to the rendered
text, and generation happens through ``/completions``.

What this client does NOT move, named rather than buried: ``forced_answer_logprobs``
stays on the ``/chat/completions`` path. That read is the anchor arm's LOGIT-level
measurement, stored in its own block with its own intervention level and outcome scale
and never merged into the record's own, and moving it needs a ``/completions``
prompt-logprob path this lane could not test against a live server. Every client here
reports the per-call-kind path map through ``reasoning_paths()`` so the record says so.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_client import OpenAIClient, OpenAIClientError

from bayes_cot_faithfulness.reasoning_mode import (
    FORCE_TOKENS,
    close_reasoning_block,
    reopened_block,
)

# How much of the rendered prompt is kept for the record. The tail is what carries the
# closed block, and a whole rendered prompt in every record would be the transcript twice.
RENDERED_TAIL_CHARS = 200
# How many continuation tails are kept as examples beside the counts.
MAX_REOPEN_EXAMPLES = 10


@dataclass
class ReasoningModeClient(OpenAIClient):
    """``OpenAIClient`` whose generations run on a rendered, block-closed prompt.

    Every generation call renders ``prompt`` through the server's own chat template,
    closes the reasoning block (``reasoning_mode.close_reasoning_block``) and posts to
    ``/completions``. Forced continuations take the identical path, which is the point of
    R12(3): Phi-4-reasoning's 24-token continuations reopen a block under the chat path,
    and closing the continuation prompt too is what keeps its mediator arm from coming
    back empty. Whether a continuation reopened one anyway is counted here, per call.
    """

    force_tokens: int = FORCE_TOKENS
    _root: OpenAIClient | None = field(default=None, repr=False, compare=False)
    _branch: str | None = field(default=None, repr=False, compare=False)
    _rendered_tail: str | None = field(default=None, repr=False, compare=False)
    _reasoning_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    _continuations: dict = field(
        default_factory=lambda: {"n": 0, "n_reopened": 0, "examples": []},
        repr=False, compare=False,
    )
    _full_generations: dict = field(
        default_factory=lambda: {"n": 0, "n_reopened": 0},
        repr=False, compare=False,
    )

    # --- rendering -------------------------------------------------------------
    def _root_client(self) -> OpenAIClient:
        """A sibling client pinned to the server ROOT.

        vLLM serves /tokenize and /detokenize there, not under /v1. A separate instance
        rather than a mutated base_url, because rendering runs on every worker thread at
        once and mutating shared state under 32 threads is a race by construction.
        """
        with self._reasoning_lock:
            if self._root is None:
                self._root = OpenAIClient(
                    base_url=self.root_url, model=self.model, timeout=self.timeout,
                    max_in_flight=0, max_retries=self.max_retries,
                    retry_wait=self.retry_wait,
                )
            return self._root

    def _tok_post(self, path: str, payload: dict) -> dict:
        try:
            return self._root_client()._post(path, payload)
        except OpenAIClientError:
            # Some builds mount them under /v1 as well; a 404 at the root is not proof
            # the endpoint is missing.
            return self._post(path, payload)

    def render_prompt(self, prompt: str, system: str | None = None) -> str:
        """The prompt as the MODEL's own chat template renders it, block closed.

        The round trip through /tokenize and /detokenize is deliberate: it is the token
        stream the server would build for a chat request, read back as text, so the only
        difference between this and the chat path is the tail this appends. A tokenizer
        that does not round-trip shows up here as mangled text rather than as a silently
        degenerate prompt, which is exactly how the R1-0528 defect was found
        (docs/REASONING-MODE-TEST.md section 3).
        """
        messages = ([{"role": "system", "content": system}] if system else [])
        messages = messages + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": True,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        body = self._tok_post("/tokenize", payload)
        ids = body.get("tokens")
        if not ids:
            raise OpenAIClientError(f"/tokenize returned no tokens: {str(body)[:200]}")
        back = self._tok_post("/detokenize", {"model": self.model, "tokens": ids})
        rendered = back.get("prompt")
        if rendered is None:
            raise OpenAIClientError(f"/detokenize returned no prompt: {str(back)[:200]}")
        closed, branch = close_reasoning_block(rendered)
        with self._reasoning_lock:
            self._branch = branch
            self._rendered_tail = closed[-RENDERED_TAIL_CHARS:]
        return closed

    # --- generation ------------------------------------------------------------
    def _completion_body(self, prompt: str, *, num_predict: int, n: int = 1,
                         temperature: float | None = None, seed: int | None = None,
                         stop: list[str] | None = None) -> dict:
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": num_predict,
            "n": n,
        }
        use_seed = self.seed if seed is None else seed
        if use_seed is not None:
            body["seed"] = use_seed
        if stop:
            body["stop"] = stop
        return body

    def _tally(self, texts: list[str], num_predict: int) -> None:
        """Count, per call, whether the completion reopened a reasoning block.

        A continuation is a call at the forced budget: that is the same discriminator the
        runner uses (``FORCE_TOKENS`` for every forced-answer call, ``ctx.num_predict``
        for every full generation), so the two counters partition the calls exactly.
        """
        is_continuation = num_predict <= self.force_tokens
        bucket = self._continuations if is_continuation else self._full_generations
        with self._reasoning_lock:
            for text in texts:
                bucket["n"] += 1
                if reopened_block(text):
                    bucket["n_reopened"] += 1
                    if is_continuation and len(bucket["examples"]) < MAX_REOPEN_EXAMPLES:
                        bucket["examples"].append((text or "")[-160:])

    def generate_many(self, prompt: str, *, num_predict: int = 320,
                      system: str | None = None,
                      stop: list[str] | None = None) -> list[str]:
        body = self._post("/completions", self._completion_body(
            self.render_prompt(prompt, system), num_predict=num_predict, stop=stop))
        try:
            texts = [c.get("text") or "" for c in body["choices"]]
        except (KeyError, TypeError) as exc:
            raise OpenAIClientError(f"unexpected response shape: {str(body)[:300]}") from exc
        self._tally(texts, num_predict)
        return texts

    def sample_completions(self, prompt: str, *, k: int, temperature: float,
                           seed: int | None = None, num_predict: int = 320,
                           system: str | None = None,
                           stop: list[str] | None = None) -> tuple[list[str], str]:
        """The element 9.2 draw, on the SAME closed prompt as every other call.

        Same two paths and the same recorded choice between them as the parent (``n = k``
        in one request, else k seeded singles), because a draw taken on a different prompt
        than the cell's generations would put the uncertain stratum on a different object
        than the arms that consume it.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        rendered = self.render_prompt(prompt, system)

        def _one(n: int, call_seed: int | None) -> list[str]:
            body = self._post("/completions", self._completion_body(
                rendered, num_predict=num_predict, n=n, temperature=temperature,
                seed=call_seed, stop=stop))
            try:
                return [c.get("text") or "" for c in body["choices"]]
            except (KeyError, TypeError) as exc:
                raise OpenAIClientError(
                    f"unexpected response shape: {str(body)[:300]}") from exc

        if self._sampling_mode != "seeded_calls":
            outputs = _one(k, seed)
            if len(outputs) >= k:
                self._sampling_mode = "n_parameter"
                self._tally(outputs[:k], num_predict)
                return outputs[:k], "n_parameter"
            # Everything the server did return is discarded rather than topped up:
            # mixing one n-parameter draw with seeded singles would make the k samples
            # two different draws.
            self._sampling_mode = "seeded_calls"

        outputs = []
        for j in range(k):
            call_seed = None if seed is None else seed + j
            got = _one(1, call_seed)
            if not got:
                raise OpenAIClientError("server returned zero choices for a sample")
            outputs.append(got[0])
        self._tally(outputs, num_predict)
        return outputs, "seeded_calls"

    # --- what the record has to carry -----------------------------------------
    def reasoning_paths(self) -> dict:
        """Which path each KIND of call took, so the record never has to be inferred."""
        return {
            "generate": "completions",
            "sample_completions": "completions",
            "forced_answer_logprobs": "chat",
        }

    def reasoning_report(self) -> dict:
        """The block-closing evidence for run_meta.json and the cell summary."""
        with self._reasoning_lock:
            cont = dict(self._continuations)
            cont["examples"] = list(cont["examples"])
            full = dict(self._full_generations)
            branch, tail = self._branch, self._rendered_tail
        n_cont = cont["n"]
        return {
            "prefill_branch": branch,
            "rendered_prompt_tail": tail,
            "block_closed_in_prompt": branch is not None,
            "paths": self.reasoning_paths(),
            "full_generations": full,
            "continuations": cont,
            # Never a rate without its n.
            "continuation_reopen_rate": (
                None if n_cont == 0 else cont["n_reopened"] / n_cont
            ),
        }
