"""Capture the request bodies TODAY's code puts on the wire, before reasoning mode exists.

Run BEFORE the BCF_REASONING_MODE change lands, and commit the fixture it writes in the
same commit. tests/test_reasoning_mode.py then asserts that reasoning_mode "default"
still builds byte-identical requests: the fixture is the only witness of the pre-change
behaviour, so capturing it AFTER the change would prove nothing at all.

    PYTHONPATH=src python tests/capture_reasoning_default_fixture.py

It talks to a stdlib http.server on localhost. No network, no cluster, no model.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

FIXTURE = REPO / "tests" / "fixtures" / "reasoning_default_requests.json"

# The element 15 constants, as the sweep sends them: 320 for a full generation, 24 for a
# forced-answer continuation, temperature 0, seed 7, concurrency 32.
FULL_TOKENS = 320
FORCE_TOKENS = 24
SEED = 7
CONCURRENCY = 32
TEMPLATE_KWARGS = {"enable_thinking": False}
PROMPT = "Q: which one?\nA) a\nB) b\nC) c\nD) d\nThink step by step."


def _handler(seen: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _send(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.endswith("/models"):
                self._send(200, {"data": [{"id": "fake-model"}]})
                return
            self._send(404, {"error": "no"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            seen.append({"path": self.path, "body": body})
            self._send(200, {"choices": [{"message": {"content": "Answer: (B)"}}]})

    return Handler


def main() -> int:
    import importlib.util

    seen: list[dict] = []
    server = HTTPServer(("127.0.0.1", 0), _handler(seen))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"

    spec = importlib.util.spec_from_file_location(
        "additive_arms_08", REPO / "experiments" / "08_additive_arms.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    client = mod._gate_client(
        "openai", "Qwen/Qwen3-8B", None, 600.0, base_url=base_url, seed=SEED,
        chat_template_kwargs=dict(TEMPLATE_KWARGS), concurrency=CONCURRENCY,
    )
    assert client is not None, "the fake server did not answer /models"
    client.generate(PROMPT, num_predict=FULL_TOKENS)
    client.generate(PROMPT, num_predict=FORCE_TOKENS)
    server.shutdown()

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({
        "captured_from": "experiments/08_additive_arms.py _gate_client + "
                         "experiments/openai_client.py generate, BEFORE BCF_REASONING_MODE",
        "prompt": PROMPT,
        "seed": SEED,
        "concurrency": CONCURRENCY,
        "chat_template_kwargs": TEMPLATE_KWARGS,
        "client_class": type(client).__name__,
        "full_generation": seen[0],
        "forced_continuation": seen[1],
    }, indent=2) + "\n")
    print(f"wrote {FIXTURE.relative_to(REPO)} from {len(seen)} captured request(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
