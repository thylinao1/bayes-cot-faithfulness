"""Concurrency tests for the additive-arms runner and the OpenAI-compatible client.

Two claims are load-bearing here and both are asserted rather than argued.

1. ``--concurrency 4`` produces the SAME records as ``--concurrency 1``. The whole point
   of adding a thread pool to a pre-registered runner is that it changes the wall clock
   and nothing else, so the test drives every per-record arm at both settings against the
   same deterministic fake backend and compares the serialized records byte for byte.
2. A failing request is RETRIED with backoff and the retry is COUNTED. The client is
   pointed at a real local HTTP server that answers 503 a fixed number of times before it
   answers 200, so the retry path is exercised over a real socket rather than a mock.

No test here reaches the network beyond 127.0.0.1 and none loads a model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bayes_cot_faithfulness.interventions import QAItem

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "08_additive_arms.py"
sys.path.insert(0, str(REPO / "experiments"))
from openai_client import OpenAIClient, OpenAIClientError


def _load_arms_module():
    spec = importlib.util.spec_from_file_location("additive_arms_concurrency", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_arms_module()


# --------------------------------------------------------------------------- #
# map_in_order
# --------------------------------------------------------------------------- #
def test_map_in_order_consumes_in_index_order_at_concurrency_8():
    """Workers overlap; consume still sees 0, 1, 2, ... with no gaps."""
    seen: list[int] = []

    def worker(i, elem):
        # Reverse the natural completion order: late items finish first.
        time.sleep((20 - i) * 0.002)
        return elem * 2

    def consume(i, elem, result):
        seen.append(i)
        assert result == elem * 2
        return True

    assert mod.map_in_order(range(20), worker, concurrency=8, consume=consume) is True
    assert seen == list(range(20))


def test_map_in_order_concurrency_1_never_starts_a_thread():
    """The default path is the pre-concurrency loop, not a one-worker pool."""
    main_thread = threading.get_ident()
    threads: list[int] = []

    def worker(i, elem):
        threads.append(threading.get_ident())
        return elem

    assert mod.map_in_order([1, 2, 3], worker, concurrency=1,
                            consume=lambda i, e, r: True) is True
    assert threads == [main_thread] * 3


def test_map_in_order_bounds_workers_in_flight():
    lock = threading.Lock()
    live = {"now": 0, "peak": 0}

    def worker(i, elem):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.01)
        with lock:
            live["now"] -= 1
        return elem

    mod.map_in_order(range(40), worker, concurrency=4, consume=lambda i, e, r: True)
    assert live["peak"] <= 4
    assert live["peak"] > 1  # and it really did run more than one at a time


def test_map_in_order_abort_stops_consuming_at_the_failing_index():
    consumed: list[int] = []

    def consume(i, elem, result):
        consumed.append(i)
        return i < 3

    assert mod.map_in_order(range(30), lambda i, e: e, concurrency=4,
                            consume=consume) is False
    assert consumed == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# concurrency 1 vs 4 give identical records
# --------------------------------------------------------------------------- #
class _DeterministicClient:
    """A backend whose answer is a pure function of the prompt.

    Deterministic and stateless, so two runs at different concurrency levels can only
    differ if the RUNNER reordered something. The small sleep makes overlap real at
    concurrency 4 rather than accidental.
    """

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self._lock = threading.Lock()
        self.n_calls = 0

    def generate(self, prompt: str, num_predict: int = 320, **kw) -> str:
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.n_calls += 1
        letter = "ABCD"[sum(ord(c) for c in prompt) % 4]
        return f"reasoning for this prompt\nAnswer: ({letter})"


ITEM_BANK = [
    QAItem(question=f"Question number {n}: which one?",
           choices=("alpha", "beta", "gamma", "delta"), answer_index=n % 4)
    for n in range(12)
]


def _records():
    out = []
    for i, item in enumerate(ITEM_BANK):
        out.append({
            "item": item,
            "clean_correct": True,
            "clean_answer": "A",
            "clean_cot": f"clean chain {i}\nstep two\nAnswer: (A)",
            "hint_label": "B",
            "cue_text": "The best answer is: (B).",
            "cue_prepended": False,
            "hinted_cot": f"hinted chain {i}\nstep two\nAnswer: (B)",
            "hinted_answer": "B",
            "followed": True,
            "acknowledged": False,
            "silent": True,
        })
    return out


def _ctx(tmp_path, concurrency):
    return mod.RunCtx(
        n_choices=4, num_predict=64, out_dir=tmp_path, safe_model="fake",
        backend="ollama", model="fake", curve_cap=6, checkpoint=None,
        concurrency=concurrency,
    )


ARMS_UNDER_TEST = ["replay", "placebo", "direct", "twostep", "filler", "curves",
                   "transplant", "anchor"]


@pytest.mark.parametrize("arm", ARMS_UNDER_TEST)
def test_arm_records_identical_at_concurrency_1_and_4(arm, tmp_path):
    runner = mod.ARM_RUNNERS[arm]

    seq_records = _records()
    seq_client = _DeterministicClient()
    ok, err = runner(seq_client, seq_records, _ctx(tmp_path / "seq", 1))
    assert ok and err is None

    par_records = _records()
    par_client = _DeterministicClient(delay=0.002)
    ok, err = runner(par_client, par_records, _ctx(tmp_path / "par", 4))
    assert ok and err is None

    assert seq_client.n_calls == par_client.n_calls
    seq_json = json.dumps([mod.serialize_arm_record(r) for r in seq_records],
                          indent=2, sort_keys=True)
    par_json = json.dumps([mod.serialize_arm_record(r) for r in par_records],
                          indent=2, sort_keys=True)
    assert seq_json == par_json


def test_cue_pass_identical_at_concurrency_1_and_4(tmp_path):
    seq_records = [{"item": it, "clean_correct": True, "clean_answer": "A",
                    "clean_cot": "c"} for it in ITEM_BANK]
    assert mod.cue_pass(_DeterministicClient(), seq_records,
                        _ctx(tmp_path / "s", 1), None) is True

    par_records = [{"item": it, "clean_correct": True, "clean_answer": "A",
                    "clean_cot": "c"} for it in ITEM_BANK]
    assert mod.cue_pass(_DeterministicClient(delay=0.002), par_records,
                        _ctx(tmp_path / "p", 4), None) is True

    for a, b in zip(seq_records, par_records):
        for key in ("hint_label", "cue_text", "hinted_cot", "hinted_answer",
                    "followed", "acknowledged", "silent"):
            assert a[key] == b[key]


def test_substrate_pass_identical_at_concurrency_1_and_4():
    seq, ok_seq, attr_seq = mod.substrate_pass(
        _DeterministicClient(), ITEM_BANK, 4, 64, "ollama", "fake", concurrency=1)
    par, ok_par, attr_par = mod.substrate_pass(
        _DeterministicClient(delay=0.002), ITEM_BANK, 4, 64, "ollama", "fake",
        concurrency=4)
    assert ok_seq and ok_par
    assert attr_seq == attr_par
    assert [r["clean_answer"] for r in seq] == [r["clean_answer"] for r in par]
    assert [r["item"].question for r in seq] == [r["item"].question for r in par]


# --------------------------------------------------------------------------- #
# retry, backoff, and the counters
# --------------------------------------------------------------------------- #
class _FlakyHandler(BaseHTTPRequestHandler):
    """Answers 503 for the first ``fail_times`` requests, then 200. Set per server."""

    fail_times = 0
    seen = 0
    lock = threading.Lock()

    def log_message(self, *args):  # keep pytest output clean
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        with type(self).lock:
            type(self).seen += 1
            should_fail = type(self).seen <= type(self).fail_times
        if should_fail:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"server busy"}')
            return
        body = json.dumps({
            "choices": [{"message": {"content": "Answer: (A)"}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(fail_times):
    handler = type("H", (_FlakyHandler,), {"fail_times": fail_times, "seen": 0,
                                           "lock": threading.Lock()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler


def test_failing_request_is_retried_and_counted():
    server, handler = _serve(fail_times=2)
    try:
        port = server.server_address[1]
        client = OpenAIClient(base_url=f"http://127.0.0.1:{port}/v1", model="fake",
                              retry_wait=0.01, retry_backoff=2.0, max_retries=4)
        assert client.generate("hello", num_predict=8) == "Answer: (A)"
        stats = client.stats()
        assert stats["retries"] == 2       # two 503s, two retries
        assert stats["requests_ok"] == 1   # one call succeeded
        assert stats["requests_failed"] == 0
        assert handler.seen == 3           # the server really saw three requests
    finally:
        server.shutdown()


def test_exhausted_retries_raise_and_count_a_failure():
    server, handler = _serve(fail_times=100)
    try:
        port = server.server_address[1]
        client = OpenAIClient(base_url=f"http://127.0.0.1:{port}/v1", model="fake",
                              retry_wait=0.01, retry_backoff=2.0, max_retries=3)
        with pytest.raises(OpenAIClientError):
            client.generate("hello", num_predict=8)
        stats = client.stats()
        assert stats["requests_failed"] == 1
        assert stats["retries"] == 2       # max_retries 3 = 1 try + 2 retries
        assert stats["requests_ok"] == 0
        assert handler.seen == 3
    finally:
        server.shutdown()


def test_retry_wait_grows_with_the_backoff_factor(monkeypatch):
    """The waits are 0.01, 0.02, 0.04, not 0.01 three times."""
    server, _ = _serve(fail_times=100)
    waits: list[float] = []
    monkeypatch.setattr("openai_client.time.sleep", waits.append)
    try:
        port = server.server_address[1]
        client = OpenAIClient(base_url=f"http://127.0.0.1:{port}/v1", model="fake",
                              retry_wait=0.01, retry_backoff=2.0, max_retries=4)
        with pytest.raises(OpenAIClientError):
            client.generate("hello", num_predict=8)
    finally:
        server.shutdown()
    assert [round(w, 4) for w in waits] == [0.01, 0.02, 0.04]


def test_client_max_in_flight_caps_concurrent_requests():
    """The client's own semaphore, not just the caller's pool, bounds the server load."""
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class _SlowHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
            time.sleep(0.05)
            with lock:
                live["now"] -= 1
            body = json.dumps({"choices": [{"message": {"content": "Answer: (A)"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        client = OpenAIClient(base_url=f"http://127.0.0.1:{port}/v1", model="fake",
                              max_in_flight=3)
        mod.map_in_order(range(12), lambda i, e: client.generate("p", num_predict=8),
                         concurrency=12, consume=lambda i, e, r: True)
    finally:
        server.shutdown()
    assert live["peak"] <= 3
    assert live["peak"] > 1


def test_request_log_lines_stay_whole_under_concurrency(tmp_path):
    """Every JSONL line parses, and more than one thread wrote them."""
    server, _ = _serve(fail_times=0)
    log = tmp_path / "requests.jsonl"
    try:
        port = server.server_address[1]
        # Short retry wait: the toy server resets a connection now and then under eight
        # concurrent clients, which is exactly the transport failure the client retries.
        client = OpenAIClient(base_url=f"http://127.0.0.1:{port}/v1", model="fake",
                              retry_wait=0.01, request_log=str(log))
        mod.map_in_order(range(40), lambda i, e: client.generate("p", num_predict=8),
                         concurrency=8, consume=lambda i, e, r: True)
    finally:
        server.shutdown()
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    # One line per SUCCESSFUL call; a retried call logs only the attempt that returned.
    assert len(lines) == 40
    assert len({entry["thread"] for entry in lines}) > 1


def test_a_connection_reset_mid_read_is_retried_not_raised():
    """urllib wraps only open-time failures, so a reset during read is a bare OSError."""
    calls = {"n": 0}

    class _ResetOnce(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            calls["n"] += 1
            if calls["n"] == 1:
                # Announce a body and hang up without sending it: the client's read
                # raises ConnectionResetError or an incomplete-read OSError.
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "9999")
                self.end_headers()
                self.wfile.write(b"{")
                self.close_connection = True
                self.connection.close()
                return
            body = json.dumps({"choices": [{"message": {"content": "Answer: (A)"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ResetOnce)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        client = OpenAIClient(base_url=f"http://127.0.0.1:{port}/v1", model="fake",
                              retry_wait=0.01, max_retries=4)
        assert client.generate("hello", num_predict=8) == "Answer: (A)"
    finally:
        server.shutdown()
    assert calls["n"] == 2
    assert client.stats()["retries"] == 1
    assert client.stats()["requests_failed"] == 0
