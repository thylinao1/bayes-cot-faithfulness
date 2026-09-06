"""Minimal OpenAI-compatible chat client (Python stdlib only), for a self-hosted vLLM server.

Same shape as ``GroqClient``: ``is_available()`` plus ``generate(prompt, num_predict=...)``
returning a string, so ``safe_generate`` / ``parse_or_force`` in 05 work unchanged. The
groq and ollama paths are untouched by this module.

What this adds over the Groq client, because the Phase-2 CONTRACT record schema asks for
it: ``seed`` and ``n`` on every call, and ``forced_answer_logprobs(prefix, letters)``,
which returns a letter -> logprob map for the answer letters of one multiple-choice item.

Point it at a vLLM server started on a cluster card:

    vllm serve Qwen/Qwen3-8B --port 8000
    python experiments/08_additive_arms.py --backend openai --base-url http://127.0.0.1:8000/v1

The chat template is rendered SERVER-SIDE (this client only ever sends ``messages``), so
the forced continuation is scored against exactly the token stream the model would see
during a normal generation. That is the whole point of the forced-logprob path: a
client-side template guess would silently score a different prompt.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# vLLM ignores the key but the OpenAI wire format requires the header to be present.
PLACEHOLDER_KEY = "EMPTY"

# Sentinel prefixes SentencePiece and byte-level BPE tokenizers put in front of a token
# that follows whitespace. Stripped before a decoded token is compared with a letter.
_TOKEN_SPACE_MARKS = ("▁", "Ġ", " ", "\t", "\n")


class OpenAIClientError(RuntimeError):
    """Raised when the OpenAI-compatible server is unreachable or answers with an error."""


def _strip_token(decoded: str) -> str:
    """Normalise one decoded token to the bare text a letter comparison can use."""
    out = decoded
    for mark in _TOKEN_SPACE_MARKS:
        out = out.replace(mark, "")
    return out.strip()


@dataclass(frozen=True)
class ForcedLogprobs:
    """Result of one forced-answer-logprob pass over the answer letters of an item.

    ``logprobs`` is the CONTRACT ``answer_logprobs`` field. ``tokens`` records WHICH
    decoded token each logprob was read off, so a unit check can assert the number
    landed on the intended answer-letter token and not on a stray punctuation token.
    """

    logprobs: dict[str, float]
    tokens: dict[str, str]
    method: str  # "prompt_logprobs" or "top_logprobs"


@dataclass
class OpenAIClient:
    """OpenAI-compatible chat client. Not frozen: it caches the server's logprob mode."""

    base_url: str
    model: str
    api_key: str = PLACEHOLDER_KEY
    temperature: float = 0.0
    n: int = 1
    seed: int | None = None
    logprobs: bool = False
    top_logprobs: int = 20
    timeout: float = 600.0
    max_retries: int = 4
    retry_wait: float = 5.0
    # e.g. {"enable_thinking": False} for Qwen3. Passed straight to the server, which
    # renders the chat template; never interpreted here.
    chat_template_kwargs: dict | None = None
    # "auto" probes prompt_logprobs once and falls back to top_logprobs if the server
    # does not return it. Pin it to make a run's method a run parameter, not a discovery.
    logprob_mode: str = "auto"
    # One JSONL line per HTTP call (timestamp, tokens asked, choices returned). This is
    # what makes measured throughput a MEASUREMENT: bcf/throughput.py joins these
    # timestamps against the arm boundaries the runner prints, so generations per second
    # per arm comes out of the run rather than out of an estimate. Defaults to the
    # BCF_REQUEST_LOG env var so the cluster script can switch it on without the runner
    # growing another flag.
    request_log: str | None = field(default_factory=lambda: os.environ.get("BCF_REQUEST_LOG"))
    _resolved_mode: str | None = field(default=None, repr=False, compare=False)

    def _record(self, path: str, payload: dict, body: dict, started: float) -> None:
        if not self.request_log:
            return
        choices = body.get("choices") or []
        entry = {
            "t_start": started,
            "t_end": time.time(),
            "path": path,
            "model": self.model,
            "max_tokens": payload.get("max_tokens"),
            "n_choices": len(choices),
            "completion_chars": sum(
                len((c.get("message") or {}).get("content") or "") for c in choices
            ),
        }
        try:
            with open(self.request_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # a throughput log is diagnostics; it must never kill a sweep

    # --- HTTP plumbing ---
    @property
    def root_url(self) -> str:
        """The server root (base_url without a trailing /v1), for /version and /health."""
        trimmed = self.base_url.rstrip("/")
        return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self._record(path, payload, body, started)
                return body
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:300]
                # 4xx is a bad request: retrying sends the identical body and fails again.
                if exc.code < 500:
                    raise OpenAIClientError(f"server error {exc.code} at {url}: {detail}") from exc
                last = OpenAIClientError(f"server error {exc.code} at {url}: {detail}")
            except urllib.error.URLError as exc:
                last = OpenAIClientError(f"could not reach {url}: {exc}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_wait)
        raise last or OpenAIClientError(f"request to {url} failed")

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=min(self.timeout, 30.0)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_available(self) -> bool:
        """True if the server answers /models. Spends no generation."""
        try:
            body = self._get(f"{self.base_url.rstrip('/')}/models")
        except Exception:  # noqa: BLE001 - any failure means "not up yet"
            return False
        return isinstance(body, dict) and "data" in body

    def wait_until_available(self, *, timeout: float, poll: float = 5.0) -> bool:
        """Block until the server answers /models, or ``timeout`` seconds elapse."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_available():
                return True
            time.sleep(poll)
        return False

    def server_version(self) -> str:
        """vLLM's /version string, for the CONTRACT ``backend_version`` field."""
        try:
            body = self._get(f"{self.root_url}/version")
        except Exception as exc:  # noqa: BLE001
            return f"unknown ({exc})"
        return str(body.get("version", "unknown"))

    # --- Generation ---
    def _chat_body(self, messages: list[dict], *, num_predict: int, **extra) -> dict:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": num_predict,
            "n": self.n,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        return {**body, **extra}

    def generate_many(self, prompt: str, *, num_predict: int = 320,
                      system: str | None = None, stop: list[str] | None = None) -> list[str]:
        """Return every sampled completion (``n`` of them) for ``prompt``."""
        messages = ([{"role": "system", "content": system}] if system else [])
        messages = messages + [{"role": "user", "content": prompt}]
        extra: dict = {}
        if stop:
            extra["stop"] = stop
        if self.logprobs:
            extra["logprobs"] = True
            extra["top_logprobs"] = self.top_logprobs
        body = self._post("/chat/completions", self._chat_body(messages, num_predict=num_predict, **extra))
        try:
            return [c["message"]["content"] or "" for c in body["choices"]]
        except (KeyError, TypeError) as exc:
            raise OpenAIClientError(f"unexpected response shape: {str(body)[:300]}") from exc

    def generate(self, prompt: str, *, num_predict: int = 320,
                 system: str | None = None, stop: list[str] | None = None) -> str:
        """Generate one completion. Same signature and return type as GroqClient.generate."""
        outputs = self.generate_many(prompt, num_predict=num_predict, system=system, stop=stop)
        if not outputs:
            raise OpenAIClientError("server returned zero choices")
        return outputs[0]

    # --- Forced answer logprobs ---
    def forced_answer_logprobs(self, prefix: str, letters: list[str]) -> ForcedLogprobs:
        """Logprob of each answer letter as the model's next assistant token after ``prefix``.

        Two server paths, both with the chat template rendered server-side:

        prompt_logprobs (vLLM): one request per letter carrying ``prefix`` as the user
        turn and the bare letter as an unfinished assistant turn
        (``continue_final_message``), asking the server to score the PROMPT. The letter is
        then the final prompt token, and its logprob is read off the last scored position.
        This is exact even when the letter is not the model's first token choice.

        top_logprobs: one request, reading the top-k distribution over the first generated
        token. Cheaper (one call, not one per letter) but only sees letters inside the
        top-k, and only when the letter is the very first token.
        """
        mode = self._resolved_mode or self.logprob_mode
        if mode == "top_logprobs":
            return self._top_logprobs(prefix, letters)
        try:
            result = self._prompt_logprobs(prefix, letters)
        except OpenAIClientError:
            if mode != "auto":
                raise
            self._resolved_mode = "top_logprobs"
            return self._top_logprobs(prefix, letters)
        self._resolved_mode = "prompt_logprobs"
        return result

    def _prompt_logprobs(self, prefix: str, letters: list[str]) -> ForcedLogprobs:
        values: dict[str, float] = {}
        tokens: dict[str, str] = {}
        for letter in letters:
            messages = [
                {"role": "user", "content": prefix},
                {"role": "assistant", "content": letter},
            ]
            body = self._post("/chat/completions", self._chat_body(
                messages,
                num_predict=1,
                continue_final_message=True,
                add_generation_prompt=False,
                prompt_logprobs=0,
            ))
            scored = body.get("prompt_logprobs")
            if scored is None:
                choices = body.get("choices") or [{}]
                scored = choices[0].get("prompt_logprobs")
            if not scored:
                raise OpenAIClientError(
                    "server returned no prompt_logprobs (not a vLLM server, or the flag "
                    "is disabled); pass logprob_mode='top_logprobs'"
                )
            logprob, token = _read_letter_from_prompt_logprobs(scored, letter)
            values[letter] = logprob
            tokens[letter] = token
        return ForcedLogprobs(logprobs=values, tokens=tokens, method="prompt_logprobs")

    def _top_logprobs(self, prefix: str, letters: list[str]) -> ForcedLogprobs:
        messages = [{"role": "user", "content": prefix}]
        body = self._post("/chat/completions", self._chat_body(
            messages, num_predict=1, logprobs=True, top_logprobs=self.top_logprobs,
        ))
        try:
            content = body["choices"][0]["logprobs"]["content"]
        except (KeyError, TypeError, IndexError) as exc:
            raise OpenAIClientError(f"no top_logprobs in response: {str(body)[:300]}") from exc
        if not content:
            raise OpenAIClientError("top_logprobs response carried no generated token")
        candidates = content[0].get("top_logprobs") or []
        by_letter: dict[str, float] = {}
        tokens: dict[str, str] = {}
        wanted = set(letters)
        for cand in candidates:
            text = _strip_token(str(cand.get("token", "")))
            if text in wanted and text not in by_letter:
                by_letter[text] = float(cand["logprob"])
                tokens[text] = str(cand.get("token", ""))
        return ForcedLogprobs(logprobs=by_letter, tokens=tokens, method="top_logprobs")


def _read_letter_from_prompt_logprobs(scored: list, letter: str) -> tuple[float, str]:
    """Find the last scored prompt position whose decoded token is ``letter``.

    Scanning from the end and MATCHING THE TOKEN TEXT is the check itself: if the chat
    template appended anything after the forced letter, or the tokenizer split the letter
    differently than expected, no position matches and this raises rather than returning
    the logprob of a template token that happens to sit last.
    """
    seen: list[str] = []
    for entry in reversed(scored):
        if not entry:
            continue
        for info in entry.values():
            decoded = str(info.get("decoded_token", ""))
            seen.append(decoded)
            if _strip_token(decoded) == letter:
                return float(info["logprob"]), decoded
    raise OpenAIClientError(
        f"no prompt_logprobs position decoded to {letter!r}; "
        f"last tokens scored were {seen[:8]!r}"
    )


def openai_setup_message(base_url: str, model: str) -> str:
    return (
        f"\n[setup] no OpenAI-compatible server answered at {base_url}, so nothing was queried.\n"
        "  1. On a GPU node:  vllm serve <hf-id> --port 8000\n"
        f"  2. Re-run with:    --backend openai --base-url {base_url} --model {model}\n"
        "This backend is for a SELF-HOSTED server (vLLM on the cluster). No paid API is involved.\n"
    )
