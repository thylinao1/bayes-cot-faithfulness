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

import http.client
import json
import os
import threading
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
    # Exponential backoff factor between retries: attempt k waits
    # retry_wait * retry_backoff ** k, capped at retry_wait_max. Set retry_backoff to
    # 1.0 to get the constant retry_wait this client used before concurrency existed.
    # The schedule affects only the failure path; a successful call never sleeps, so a
    # concurrency-1 run produces the same records either way.
    retry_backoff: float = 2.0
    retry_wait_max: float = 60.0
    # Hard ceiling on requests this client has in flight at once, enforced inside _post
    # by a semaphore. 0 means "no client-side ceiling"; the caller's thread pool is then
    # the only bound. Non-zero is the belt for a caller that over-subscribes the server.
    max_in_flight: int = 0
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
    # Counters, and the locks that make this client safe to share across threads. The
    # request log is a single append-mode file handle per write, so two threads writing
    # at once can interleave a line; the lock is what keeps every JSONL line whole.
    _stats: dict = field(
        default_factory=lambda: {"requests_ok": 0, "retries": 0, "requests_failed": 0},
        repr=False, compare=False,
    )
    _stats_lock: "threading.Lock" = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    _log_lock: "threading.Lock" = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    _gate: "threading.Semaphore | None" = field(default=None, repr=False, compare=False)
    # Which path sample_completions resolved to for this client: "n_parameter" when the
    # server honored n > 1, "seeded_calls" when it did not. None until the first draw.
    _sampling_mode: str | None = field(default=None, repr=False, compare=False)

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + 1

    def stats(self) -> dict:
        """A snapshot of the request counters: successes, retries, hard failures.

        ``retries`` counts RETRY ATTEMPTS, not failed calls: one call that succeeded on
        its third try contributes 2 here and 1 to ``requests_ok``. A call that exhausted
        ``max_retries`` contributes ``max_retries - 1`` retries and 1 to
        ``requests_failed``, and raises.
        """
        with self._stats_lock:
            return dict(self._stats)

    def _acquire(self):
        """The in-flight gate, created on first use so the dataclass stays cheap."""
        if self.max_in_flight <= 0:
            return None
        with self._stats_lock:
            if self._gate is None:
                self._gate = threading.Semaphore(self.max_in_flight)
        return self._gate

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
            # Which OS thread issued the call. Two lines whose [t_start, t_end] overlap
            # AND whose thread ids differ are the proof that requests were concurrent;
            # without it an overlap could only be argued from the clock.
            "thread": threading.get_ident(),
        }
        try:
            line = json.dumps(entry) + "\n"
            with self._log_lock:
                with open(self.request_log, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError:
            pass  # a throughput log is diagnostics; it must never kill a sweep

    # --- HTTP plumbing ---
    @property
    def root_url(self) -> str:
        """The server root (base_url without a trailing /v1), for /version and /health."""
        trimmed = self.base_url.rstrip("/")
        return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed

    def _post(self, path: str, payload: dict) -> dict:
        gate = self._acquire()
        if gate is not None:
            gate.acquire()
        try:
            return self._post_inner(path, payload)
        finally:
            if gate is not None:
                gate.release()

    def _post_inner(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        # One Request object per attempt, not one per call: urllib mutates a Request
        # while it is being opened (redirect handling, host header), so reusing one
        # across threads is a data race waiting to happen.
        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=body_bytes, headers=headers)
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self._record(path, payload, body, started)
                self._bump("requests_ok")
                return body
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:300]
                # 4xx is a bad request: retrying sends the identical body and fails again.
                if exc.code < 500:
                    self._bump("requests_failed")
                    raise OpenAIClientError(f"server error {exc.code} at {url}: {detail}") from exc
                last = OpenAIClientError(f"server error {exc.code} at {url}: {detail}")
            except urllib.error.URLError as exc:
                last = OpenAIClientError(f"could not reach {url}: {exc}")
            # A reset or a short read DURING resp.read() is a bare OSError, not a
            # URLError: urllib wraps only the failures it sees while opening the
            # connection. Under concurrency that is the common transport failure, and
            # without this clause it escapes the retry loop and kills the arm on a
            # blip the next attempt would have survived. TimeoutError and
            # ConnectionResetError are both OSError subclasses, so one clause covers
            # them; a JSON body that will not decode is NOT retried, because the same
            # bytes decode the same way twice.
            except (OSError, http.client.HTTPException) as exc:
                # http.client.IncompleteRead is an HTTPException, NOT an OSError: a
                # server that announces a Content-Length and then hangs up lands here
                # and nowhere else. Both classes are the transport giving out mid-call,
                # which the next attempt can survive.
                last = OpenAIClientError(f"transport failure at {url}: "
                                         f"{type(exc).__name__}: {exc}")
            if attempt < self.max_retries - 1:
                self._bump("retries")
                wait = min(self.retry_wait * (self.retry_backoff ** attempt),
                           self.retry_wait_max)
                time.sleep(wait)
        self._bump("requests_failed")
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

    def sample_completions(
        self, prompt: str, *, k: int, temperature: float, seed: int | None = None,
        num_predict: int = 320, system: str | None = None,
        stop: list[str] | None = None,
    ) -> tuple[list[str], str]:
        """``k`` sampled completions of one prompt, and HOW they were drawn.

        The element 9.2 uncertain-item arm needs k = 32 draws at temperature 0.7 from
        the same clean prompt. Two ways exist and the run must record which one it got,
        because they are not the same draw: the endpoint's ``n`` parameter returns k
        samples from ONE request (one prefill, k sampler draws sharing one seed), while
        the fallback issues k requests with seeds ``seed, seed + 1, ...``.

        This tries ``n = k`` first and inspects the answer. A server that ignores ``n``
        replies with one choice, which is indistinguishable from k = 1 unless it is
        checked, so it is checked: fewer choices than asked for means the fallback runs
        and the returned method says ``seeded_calls``. The decision is cached on the
        client, so it costs at most one short request per run and every later item takes
        the same path.

        Returns ``(completions, method)`` with ``method`` in
        {"n_parameter", "seeded_calls"}. ``completions`` always has k entries.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        messages = ([{"role": "system", "content": system}] if system else [])
        messages = messages + [{"role": "user", "content": prompt}]
        extra: dict = {"temperature": temperature}
        if stop:
            extra["stop"] = stop

        def _one(n: int, call_seed: int | None) -> list[str]:
            body_extra = dict(extra, n=n)
            if call_seed is not None:
                body_extra["seed"] = call_seed
            body = self._post("/chat/completions", self._chat_body(
                messages, num_predict=num_predict, **body_extra))
            try:
                return [c["message"]["content"] or "" for c in body["choices"]]
            except (KeyError, TypeError) as exc:
                raise OpenAIClientError(
                    f"unexpected response shape: {str(body)[:300]}") from exc

        if self._sampling_mode != "seeded_calls":
            outputs = _one(k, seed)
            if len(outputs) >= k:
                self._sampling_mode = "n_parameter"
                return outputs[:k], "n_parameter"
            # The server answered with fewer completions than n asked for. Everything
            # it did return is discarded rather than topped up: mixing one n-parameter
            # draw with seeded singles would make the k samples two different draws.
            self._sampling_mode = "seeded_calls"

        outputs = []
        for j in range(k):
            call_seed = None if seed is None else seed + j
            got = _one(1, call_seed)
            if not got:
                raise OpenAIClientError("server returned zero choices for a sample")
            outputs.append(got[0])
        return outputs, "seeded_calls"

    def sampling_mode(self) -> str | None:
        """Which draw path ``sample_completions`` resolved to, or None if never called."""
        return self._sampling_mode

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
