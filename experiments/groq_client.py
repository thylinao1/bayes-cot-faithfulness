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
import urllib.error
import urllib.request
from dataclasses import dataclass

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(RuntimeError):
    """Raised when the Groq API is unreachable, unauthorised, or errors."""


@dataclass(frozen=True)
class GroqClient:
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    timeout: float = 60.0

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
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:200]
            raise GroqError(f"Groq API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise GroqError(f"Could not reach Groq: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - unexpected shape
            raise GroqError(f"Unexpected Groq response: {str(body)[:200]}") from exc
