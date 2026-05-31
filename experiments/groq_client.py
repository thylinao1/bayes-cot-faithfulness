"""Minimal Groq client (OpenAI-compatible). Reads GROQ_API_KEY from the environment.

Groq's free tier serves fast, capable Llama models (e.g. llama-3.3-70b-versatile) at
no cost: a free API key, no card required. Running the control on a capable cloud
model means a laptop is no longer the bottleneck, and a 70B is consistent enough for
the negative control (unlike a 3B). No charge on the free tier. If you add a payment
method to your Groq account, usage could start to bill, so keep an eye on that.

Get a free key at https://console.groq.com/keys, then:

    export GROQ_API_KEY=...   # never commit this
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq sits behind Cloudflare, which blocks the default urllib User-Agent with a 1010
# error. A normal browser UA gets the request through to the API.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


class GroqError(RuntimeError):
    """Raised when the Groq API is unreachable, unauthorised, or errors."""


def _retry_after(exc: urllib.error.HTTPError) -> float:
    """Seconds Groq asks us to wait after a 429 (from the Retry-After header, or a default)."""
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            return max(float(header), 1.0)
        except ValueError:
            pass
    return 8.0


@dataclass(frozen=True)
class GroqClient:
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 4
    max_wait: float = 25.0  # a 429 asking for longer than this aborts (don't grind for hours)

    def is_available(self) -> bool:
        """True if a GROQ_API_KEY is set. Does not spend a request to check."""
        return bool(os.environ.get("GROQ_API_KEY"))

    def generate(self, prompt: str, *, num_predict: int = 320) -> str:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise GroqError("GROQ_API_KEY is not set.")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": num_predict,
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": _USER_AGENT,
            },
        )
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                # 429 = free-tier rate limit (throttling, not a charge).
                if exc.code == 429:
                    wait = _retry_after(exc)
                    if wait > self.max_wait or attempt >= self.max_retries - 1:
                        raise GroqError(
                            f"rate limit: Groq asked to wait {wait:.0f}s. The free tier is "
                            "throttling, likely the per-minute or daily TOKEN budget (the "
                            "few-shot prompts are token-heavy and you have run several times). "
                            "Run fewer items (--n-items 20), wait for the reset, or use the "
                            "lighter --model llama-3.1-8b-instant. Not a charge."
                        ) from exc
                    print(f"  [groq] rate-limited, waiting {wait:.0f}s...", file=sys.stderr, flush=True)
                    time.sleep(wait)
                    continue
                detail = exc.read().decode("utf-8", "ignore")[:200]
                raise GroqError(f"Groq API error {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise GroqError(f"Could not reach Groq: {exc}") from exc
            except (KeyError, IndexError) as exc:  # pragma: no cover - unexpected shape
                raise GroqError(f"Unexpected Groq response: {str(body)[:200]}") from exc
        raise GroqError("Groq rate limit: retries exhausted.")
