"""Offline tests for the OpenAI-compatible backend, against a fake in-process server.

No network, no cluster, no model: a stdlib http.server on localhost records the request
bodies it receives and replays canned vLLM-shaped responses. Every assertion here is
about the WIRE FORMAT the client puts on the socket and how it reads the answer back,
which is exactly the part a cluster run cannot check cheaply.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from openai_client import (
    OpenAIClient,
    OpenAIClientError,
    _read_letter_from_prompt_logprobs,
    _strip_token,
)


class FakeServerState:
    """Canned responses plus the bodies the client actually sent."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: list[dict] = []
        self.status = 200
        self.models_ok = True


def _make_handler(state: FakeServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence the test log
            return

        def _send(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # http.server API, hence the uppercase name
            if self.path.endswith("/models"):
                if not state.models_ok:
                    self._send(503, {"error": "not ready"})
                    return
                self._send(200, {"data": [{"id": "fake-model"}]})
                return
            if self.path.endswith("/version"):
                self._send(200, {"version": "0.11.0"})
                return
            self._send(404, {"error": "no"})

        def do_POST(self):  # http.server API, hence the uppercase name
            length = int(self.headers.get("Content-Length", 0))
            state.requests.append(json.loads(self.rfile.read(length) or b"{}"))
            if state.status != 200:
                self._send(state.status, {"error": "boom"})
                return
            body = state.responses.pop(0) if state.responses else {}
            self._send(200, body)

    return Handler


@pytest.fixture()
def fake_server():
    state = FakeServerState()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield state, f"http://127.0.0.1:{port}/v1"
    server.shutdown()
    server.server_close()


def _chat_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _prompt_logprob_response(tokens: list[tuple[str, float]]) -> dict:
    """A vLLM chat response carrying prompt_logprobs for the given decoded tokens."""
    scored: list = [None]
    for idx, (decoded, logprob) in enumerate(tokens):
        scored.append({str(idx + 100): {"logprob": logprob, "rank": 1, "decoded_token": decoded}})
    return {"choices": [{"message": {"role": "assistant", "content": "x"}}], "prompt_logprobs": scored}


# --- availability and metadata ---
def test_is_available_true_when_models_endpoint_answers(fake_server):
    _state, base = fake_server
    assert OpenAIClient(base_url=base, model="m").is_available() is True


def test_is_available_false_when_server_not_ready(fake_server):
    state, base = fake_server
    state.models_ok = False
    assert OpenAIClient(base_url=base, model="m").is_available() is False


def test_is_available_false_when_nothing_is_listening():
    client = OpenAIClient(base_url="http://127.0.0.1:1/v1", model="m", timeout=2.0)
    assert client.is_available() is False


def test_server_version_reads_root_version_endpoint(fake_server):
    _state, base = fake_server
    assert OpenAIClient(base_url=base, model="m").server_version() == "0.11.0"


def test_root_url_strips_the_v1_suffix():
    assert OpenAIClient(base_url="http://h:8000/v1", model="m").root_url == "http://h:8000"
    assert OpenAIClient(base_url="http://h:8000", model="m").root_url == "http://h:8000"


# --- generate: the GroqClient-compatible surface ---
def test_generate_returns_the_first_choice_text(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("The answer is (C)."))
    out = OpenAIClient(base_url=base, model="m").generate("q", num_predict=320)
    assert out == "The answer is (C)."


def test_generate_sends_model_temperature_max_tokens_seed_and_n(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("ok"))
    client = OpenAIClient(base_url=base, model="Qwen/Qwen3-8B", temperature=0.0, n=1, seed=7)
    client.generate("hello", num_predict=320)
    sent = state.requests[0]
    assert sent["model"] == "Qwen/Qwen3-8B"
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 320
    assert sent["n"] == 1
    assert sent["seed"] == 7
    assert sent["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_omits_seed_when_unset(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("ok"))
    OpenAIClient(base_url=base, model="m").generate("hello")
    assert "seed" not in state.requests[0]


def test_chat_template_kwargs_are_forwarded_for_thinking_off(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("ok"))
    client = OpenAIClient(base_url=base, model="m", chat_template_kwargs={"enable_thinking": False})
    client.generate("hello")
    assert state.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_generate_many_returns_every_sampled_choice(fake_server):
    state, base = fake_server
    state.responses.append({"choices": [
        {"message": {"content": "a"}}, {"message": {"content": "b"}}, {"message": {"content": "c"}},
    ]})
    outs = OpenAIClient(base_url=base, model="m", n=3, temperature=0.7).generate_many("q")
    assert outs == ["a", "b", "c"]
    assert state.requests[0]["n"] == 3


def test_generate_forwards_system_and_stop(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("ok"))
    OpenAIClient(base_url=base, model="m").generate("q", system="be terse", stop=["\n\n"])
    sent = state.requests[0]
    assert sent["messages"][0] == {"role": "system", "content": "be terse"}
    assert sent["stop"] == ["\n\n"]


def test_a_4xx_error_raises_without_retrying(fake_server):
    state, base = fake_server
    state.status = 400
    client = OpenAIClient(base_url=base, model="m", max_retries=4, retry_wait=0.0)
    with pytest.raises(OpenAIClientError):
        client.generate("q")
    assert len(state.requests) == 1  # a bad body is not worth resending


def test_a_5xx_error_is_retried_then_raises(fake_server):
    state, base = fake_server
    state.status = 503
    client = OpenAIClient(base_url=base, model="m", max_retries=3, retry_wait=0.0)
    with pytest.raises(OpenAIClientError):
        client.generate("q")
    assert len(state.requests) == 3


# --- forced answer logprobs: prompt_logprobs (vLLM) ---
def test_forced_logprobs_sends_one_continue_final_message_request_per_letter(fake_server):
    state, base = fake_server
    for letter, lp in (("A", -3.0), ("B", -0.1), ("C", -4.0), ("D", -5.0)):
        state.responses.append(_prompt_logprob_response([("answer", -1.0), (letter, lp)]))
    client = OpenAIClient(base_url=base, model="m", seed=7)
    result = client.forced_answer_logprobs("Question ...", ["A", "B", "C", "D"])

    assert result.method == "prompt_logprobs"
    assert len(state.requests) == 4
    first = state.requests[0]
    assert first["continue_final_message"] is True
    assert first["add_generation_prompt"] is False
    assert first["prompt_logprobs"] == 0
    assert first["max_tokens"] == 1
    # The prefix goes as a USER turn and the forced letter as an unfinished ASSISTANT
    # turn: the server renders the chat template, this client never guesses it.
    assert first["messages"] == [
        {"role": "user", "content": "Question ..."},
        {"role": "assistant", "content": "A"},
    ]
    assert result.logprobs == {"A": -3.0, "B": -0.1, "C": -4.0, "D": -5.0}


def test_forced_logprobs_land_on_the_intended_answer_letter_token(fake_server):
    """The unit check the Phase-1 skeleton must pass: the number read is the LETTER's."""
    state, base = fake_server
    # A realistic tail: the letter, then template punctuation the model never chose.
    state.responses.append(_prompt_logprob_response(
        [("The", -2.0), (" answer", -1.0), (" is", -0.5), ("B", -0.11), (".", -0.9)]
    ))
    client = OpenAIClient(base_url=base, model="m")
    result = client.forced_answer_logprobs("Q", ["B"])
    assert _strip_token(result.tokens["B"]) == "B"
    assert math.isclose(result.logprobs["B"], -0.11)


def test_forced_logprobs_tolerate_a_sentencepiece_space_mark(fake_server):
    state, base = fake_server
    state.responses.append(_prompt_logprob_response([("x", -1.0), ("▁C", -0.25)]))
    result = OpenAIClient(base_url=base, model="m").forced_answer_logprobs("Q", ["C"])
    assert math.isclose(result.logprobs["C"], -0.25)
    assert result.tokens["C"] == "▁C"


def test_forced_logprobs_refuse_when_no_position_decodes_to_the_letter():
    with pytest.raises(OpenAIClientError, match="no prompt_logprobs position decoded"):
        _read_letter_from_prompt_logprobs(
            [None, {"1": {"logprob": -0.4, "decoded_token": "<|im_end|>"}}], "A"
        )


# --- forced answer logprobs: top_logprobs fallback ---
def _top_logprob_response(pairs: list[tuple[str, float]]) -> dict:
    return {"choices": [{
        "message": {"content": pairs[0][0]},
        "logprobs": {"content": [{
            "token": pairs[0][0],
            "logprob": pairs[0][1],
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs],
        }]},
    }]}


def test_top_logprobs_mode_reads_the_first_generated_token_distribution(fake_server):
    state, base = fake_server
    state.responses.append(_top_logprob_response([("B", -0.2), (" A", -1.8), ("C", -3.1), ("z", -9.0)]))
    client = OpenAIClient(base_url=base, model="m", logprob_mode="top_logprobs")
    result = client.forced_answer_logprobs("Q", ["A", "B", "C", "D"])
    assert result.method == "top_logprobs"
    assert result.logprobs == {"B": -0.2, "A": -1.8, "C": -3.1}  # D was outside top-k
    assert len(state.requests) == 1  # one call for every letter, not one per letter
    assert state.requests[0]["logprobs"] is True


def test_auto_mode_falls_back_to_top_logprobs_on_a_server_without_prompt_logprobs(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("B"))  # no prompt_logprobs key at all
    state.responses.append(_top_logprob_response([("B", -0.2), ("A", -1.4)]))
    client = OpenAIClient(base_url=base, model="m", logprob_mode="auto")
    result = client.forced_answer_logprobs("Q", ["A", "B"])
    assert result.method == "top_logprobs"
    assert result.logprobs == {"B": -0.2, "A": -1.4}


def test_auto_mode_remembers_the_fallback_and_does_not_reprobe(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("B"))
    state.responses.append(_top_logprob_response([("B", -0.2)]))
    state.responses.append(_top_logprob_response([("B", -0.3)]))
    client = OpenAIClient(base_url=base, model="m", logprob_mode="auto")
    client.forced_answer_logprobs("Q", ["B"])
    n_after_first = len(state.requests)
    client.forced_answer_logprobs("Q2", ["B"])
    assert len(state.requests) - n_after_first == 1  # one call, no repeated probe
    assert client._resolved_mode == "top_logprobs"


def test_pinned_prompt_logprobs_mode_never_falls_back(fake_server):
    state, base = fake_server
    state.responses.append(_chat_response("B"))
    client = OpenAIClient(base_url=base, model="m", logprob_mode="prompt_logprobs")
    with pytest.raises(OpenAIClientError, match="no prompt_logprobs"):
        client.forced_answer_logprobs("Q", ["B"])


# --- the runner wiring ---
def test_08_runner_builds_an_openai_client_for_the_openai_backend(fake_server):
    _state, base = fake_server
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "experiments" / "08_additive_arms.py"
    spec = importlib.util.spec_from_file_location("arms08_for_backend_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    client = module._gate_client("openai", "Qwen/Qwen3-8B", "http://localhost:11434", 60.0,
                                base_url=base, seed=7,
                                chat_template_kwargs={"enable_thinking": False})
    assert isinstance(client, OpenAIClient)
    assert client.model == "Qwen/Qwen3-8B"
    assert client.seed == 7
    assert client.chat_template_kwargs == {"enable_thinking": False}


def test_08_runner_returns_none_when_the_openai_server_is_down(capsys):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "experiments" / "08_additive_arms.py"
    spec = importlib.util.spec_from_file_location("arms08_for_down_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    client = module._gate_client("openai", "m", "http://localhost:11434", 2.0,
                                 base_url="http://127.0.0.1:1/v1")
    assert client is None
    assert "no OpenAI-compatible server answered" in capsys.readouterr().out


def test_08_parser_accepts_the_openai_backend_flags():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "experiments" / "08_additive_arms.py"
    spec = importlib.util.spec_from_file_location("arms08_for_parser_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([
        "--backend", "openai", "--base-url", "http://127.0.0.1:8000/v1",
        "--model", "Qwen/Qwen3-8B", "--seed", "7",
        "--chat-template-kwargs", '{"enable_thinking": false}',
    ])
    assert args.backend == "openai"
    assert args.base_url == "http://127.0.0.1:8000/v1"
    assert args.seed == 7
    assert args.chat_template_kwargs == '{"enable_thinking": false}'


# --- request log: the input to the measured-throughput reducer ---
def test_no_request_log_is_written_when_the_field_is_unset(fake_server, tmp_path):
    state, base = fake_server
    state.responses.append(_chat_response("ok"))
    OpenAIClient(base_url=base, model="m", request_log=None).generate("q")
    assert list(tmp_path.iterdir()) == []


def test_each_call_appends_one_line_with_a_timestamp_and_token_budget(fake_server, tmp_path):
    state, base = fake_server
    log = tmp_path / "requests.jsonl"
    state.responses.append(_chat_response("hello there"))
    state.responses.append(_chat_response("x"))
    client = OpenAIClient(base_url=base, model="m", request_log=str(log))
    client.generate("q", num_predict=320)
    client.generate("q2", num_predict=24)

    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert [r["max_tokens"] for r in rows] == [320, 24]
    assert all(r["t_end"] >= r["t_start"] for r in rows)
    assert rows[0]["completion_chars"] == len("hello there")
    assert rows[0]["n_choices"] == 1
    assert rows[0]["path"] == "/chat/completions"


def test_failed_calls_are_not_logged_as_generations(fake_server, tmp_path):
    state, base = fake_server
    log = tmp_path / "requests.jsonl"
    state.status = 400
    client = OpenAIClient(base_url=base, model="m", request_log=str(log),
                          max_retries=1, retry_wait=0.0)
    with pytest.raises(OpenAIClientError):
        client.generate("q")
    assert not log.exists(), "a failed call must not inflate the throughput numerator"


def test_an_unwritable_request_log_never_kills_a_sweep(fake_server, tmp_path):
    state, base = fake_server
    state.responses.append(_chat_response("ok"))
    unwritable = tmp_path / "no-such-dir" / "requests.jsonl"
    client = OpenAIClient(base_url=base, model="m", request_log=str(unwritable))
    assert client.generate("q") == "ok"  # diagnostics failing must not stop generation


def test_request_log_defaults_to_the_env_var(fake_server, tmp_path, monkeypatch):
    state, base = fake_server
    log = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("BCF_REQUEST_LOG", str(log))
    state.responses.append(_chat_response("ok"))
    OpenAIClient(base_url=base, model="m").generate("q")
    assert len(log.read_text().splitlines()) == 1
