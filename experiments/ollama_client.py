"""Minimal Ollama HTTP client (Python stdlib only, no extra dependency).

Talks to a local Ollama server (default http://localhost:11434). Generation runs
entirely on your machine: no API, no cloud, no cost. Install Ollama from
https://ollama.com and pull a model (e.g. ``ollama pull llama3.1:8b``) first.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot be reached or returns an error."""


@dataclass(frozen=True)
class OllamaClient:
    model: str
    host: str = DEFAULT_HOST
    temperature: float = 0.0
    timeout: float = 120.0

    def is_available(self) -> bool:
        """True if a local Ollama server answers (does not check the model is pulled)."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, *, system: str | None = None,
                 stop: list[str] | None = None, num_predict: int = 512) -> str:
        """Generate a completion for ``prompt`` from the local model. Deterministic at T=0."""
        options: dict[str, object] = {"temperature": self.temperature, "num_predict": num_predict}
        if stop:
            options["stop"] = stop
        payload: dict[str, object] = {
            "model": self.model, "prompt": prompt, "stream": False, "options": options,
        }
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}. Install it (https://ollama.com), "
                f"start it, and run `ollama pull {self.model}`. Underlying error: {exc}"
            ) from exc
        if "error" in body:
            raise OllamaError(str(body["error"]))
        return body.get("response", "")
