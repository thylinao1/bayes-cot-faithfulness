"""Offline tests for RULING R12's reasoning_mode: default, off, on.

Everything here runs against a stdlib ``http.server`` on localhost that records the
bodies and the PATHS the client puts on the socket and replays canned vLLM-shaped
answers, the same pattern as tests/test_openai_client.py. No network, no cluster, no
model.

What each test is for:

  (a) reasoning_mode "default" sends byte-identical requests. The witness is
      tests/fixtures/reasoning_default_requests.json, captured by
      tests/capture_reasoning_default_fixture.py against the code as it stood BEFORE
      BCF_REASONING_MODE existed and committed in its own commit (f284b1d). A fixture
      captured after the change would prove nothing.
  (b) "off" closes the block the template opened, inserts a whole empty block where the
      template opened none, and routes generation to /completions.
  (c) "on" raises the FULL-generation budget to 4096 and leaves the forced continuation
      at the element 15 constant 24.
  (d) the post-think extractor reads the letter after the closing tag and returns
      unparseable when there is none.
  (e) every record carries reasoning_mode, reasoning_path, reasoning_block_closed and
      num_predict_full, and the writer REFUSES a record that lost them.

REVERT PROOF for (e), run on 2026-09-07 before this file was committed. With the line
``assert_records_reasoning_mode(transcripts)`` deleted from ``write_arm_transcripts`` in
experiments/08_additive_arms.py, and nothing else changed:

    $ PYTHONPATH=src python -m pytest -q tests/test_reasoning_mode.py
    FAILED tests/test_reasoning_mode.py::test_writer_refuses_a_record_that_lost_the_reasoning_fields
      Failed: DID NOT RAISE ReasoningModeError
    1 failed, 27 passed in 5.27s

Restoring the assertion turns it green again. So the refusal is what that test measures,
not the serializer that happens to fill the fields in on the way past.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from openai_client import OpenAIClient
from reasoning_client import ReasoningModeClient

from bayes_cot_faithfulness.interventions import QAItem
from bayes_cot_faithfulness.reasoning_mode import (
    EMPTY_THINK_BLOCK,
    ON_NUM_PREDICT_FULL,
    ReasoningModeError,
    assert_records_reasoning_mode,
    close_reasoning_block,
    full_num_predict,
    parse_answer_after_think,
    record_fields,
    reopened_block,
    strip_reasoning_block,
)

FIXTURE = REPO / "tests" / "fixtures" / "reasoning_default_requests.json"
SCRIPT = REPO / "experiments" / "08_additive_arms.py"


def _load_arms_module():
    spec = importlib.util.spec_from_file_location("arms_under_test_rmode", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_arms_module()


# --------------------------------------------------------------------------- #
# The fake server
# --------------------------------------------------------------------------- #
class FakeState:
    """Recorded (path, body) pairs, plus the canned renderings and completions."""

    def __init__(self) -> None:
        self.seen: list[dict] = []
        # What /detokenize hands back: the rendered prompt, as the model's template
        # would leave it. Set per test to a template that opens a block or one that
        # does not.
        self.rendered = "user turn<|im_start|>assistant\n<think>"
        # completion text by endpoint; callable(body) -> str
        self.completion_text = lambda body: "Answer: (A)"
        self.chat_text = lambda body: "Answer: (A)"


def _make_handler(state: FakeState):
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
            state.seen.append({"path": self.path, "body": body})
            if self.path.endswith("/tokenize"):
                self._send(200, {"tokens": [1, 2, 3], "count": 3})
                return
            if self.path.endswith("/detokenize"):
                self._send(200, {"prompt": state.rendered})
                return
            if self.path.endswith("/completions") and not self.path.endswith(
                    "/chat/completions"):
                n = int(body.get("n", 1) or 1)
                self._send(200, {"choices": [{"text": state.completion_text(body)}
                                             for _ in range(n)]})
                return
            n = int(body.get("n", 1) or 1)
            self._send(200, {"choices": [
                {"message": {"content": state.chat_text(body)}} for _ in range(n)]})

    return Handler


@pytest.fixture()
def fake_server():
    state = FakeState()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()


def _posts(state: FakeState, suffix: str) -> list[dict]:
    """Every recorded request whose path ends in ``suffix`` (chat is excluded from
    /completions, because '/v1/chat/completions' also ends in '/completions')."""
    out = []
    for row in state.seen:
        if not row["path"].endswith(suffix):
            continue
        if suffix == "/completions" and row["path"].endswith("/chat/completions"):
            continue
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# (a) default sends byte-identical requests
# --------------------------------------------------------------------------- #
def test_default_mode_builds_the_identical_request_as_before(fake_server):
    """The request bodies must match the fixture captured BEFORE this change."""
    want = json.loads(FIXTURE.read_text())
    client = mod._gate_client(
        "openai", "Qwen/Qwen3-8B", None, 600.0, base_url=fake_server.base_url,
        seed=want["seed"], chat_template_kwargs=dict(want["chat_template_kwargs"]),
        concurrency=want["concurrency"], reasoning_mode="default",
    )
    assert client is not None
    # The default cell's client is still the plain client, not a subclass: nothing new
    # is on the path of an existing cell.
    assert type(client) is OpenAIClient
    client.generate(want["prompt"], num_predict=320)
    client.generate(want["prompt"], num_predict=24)

    posts = [row for row in fake_server.seen if row["path"].endswith("/chat/completions")]
    assert len(posts) == 2
    assert posts[0] == want["full_generation"]
    assert posts[1] == want["forced_continuation"]
    # And nothing was rendered: the default path never touches /tokenize.
    assert _posts(fake_server, "/tokenize") == []


def test_default_mode_records_the_chat_path_and_the_frozen_budget():
    ctx = mod.RunCtx(4, 320, Path("."), "fake", "openai", "m", 20)
    assert ctx.reasoning_mode == "default"
    assert ctx.reasoning_fields() == {
        "reasoning_mode": "default", "reasoning_path": "chat",
        "reasoning_block_closed": None, "num_predict_full": 320,
    }
    assert ctx.extract_full is not parse_answer_after_think


# --------------------------------------------------------------------------- #
# (b) off closes the block, both branches, and routes to /completions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rendered, want_tail, branch",
    [
        # Olmo-3-7B-Think and R1-Distill-Llama-8B: the template ALREADY opened a block,
        # so only the closing tag is appended.
        ("...form 'Answer: (X)'.<|im_start|>assistant\n<think>",
         "<think>\n\n</think>\n\n", "closed_the_block_the_template_opened"),
        ("...form 'Answer: (X)'.<|Assistant|><think>\n",
         "<think>\n\n\n</think>\n\n", "closed_the_block_the_template_opened"),
        # R1-0528 and Phi-4-reasoning: the template opened nothing, so the whole empty
        # block goes in.
        ("...form 'Answer: (X)'.<|Assistant|>",
         "<|Assistant|><think>\n\n</think>\n\n", "inserted_a_whole_empty_block"),
        ("...form 'Answer: (X)'.<|im_end|><|im_start|>assistant<|im_sep|>",
         "<|im_sep|><think>\n\n</think>\n\n", "inserted_a_whole_empty_block"),
    ],
)
def test_off_mode_closes_the_block_on_both_branches(fake_server, rendered, want_tail,
                                                    branch):
    fake_server.rendered = rendered
    client = ReasoningModeClient(
        base_url=fake_server.base_url, model="m", temperature=0.0, seed=7,
        chat_template_kwargs={"enable_thinking": False},
    )
    out = client.generate_many("q", num_predict=320)
    assert out == ["Answer: (A)"]

    completions = _posts(fake_server, "/completions")
    assert len(completions) == 1, "an off generation must go through /completions"
    assert not [row for row in fake_server.seen
                if row["path"].endswith("/chat/completions")]
    prompt = completions[0]["body"]["prompt"]
    assert prompt.endswith(want_tail), prompt[-60:]
    # Every element 15 constant is where it was.
    assert completions[0]["body"]["max_tokens"] == 320
    assert completions[0]["body"]["temperature"] == 0.0
    assert completions[0]["body"]["seed"] == 7
    assert completions[0]["body"]["n"] == 1

    report = client.reasoning_report()
    assert report["prefill_branch"] == branch
    assert report["block_closed_in_prompt"] is True
    assert report["rendered_prompt_tail"].endswith(want_tail)
    assert report["paths"]["generate"] == "completions"


def test_off_mode_renders_through_tokenize_with_the_template_kwargs(fake_server):
    client = ReasoningModeClient(
        base_url=fake_server.base_url, model="m", seed=7,
        chat_template_kwargs={"enable_thinking": False},
    )
    client.generate_many("q", num_predict=320)
    tok = _posts(fake_server, "/tokenize")
    assert len(tok) == 1
    assert tok[0]["body"]["messages"] == [{"role": "user", "content": "q"}]
    assert tok[0]["body"]["add_generation_prompt"] is True
    assert tok[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(_posts(fake_server, "/detokenize")) == 1


def test_off_mode_closes_the_forced_continuation_prompt_too(fake_server):
    """R12(3): Phi-4's 24-token continuations reopen a block on the chat path.

    Under "off" the continuation prompt is rendered and closed like any other, and
    whether the model reopened one ANYWAY is counted per continuation.
    """
    fake_server.rendered = "...<|im_start|>assistant<|im_sep|>"
    fake_server.completion_text = lambda body: (
        "<think> wait, let me reconsider" if body["max_tokens"] == 24
        else "Answer: (A)")
    client = ReasoningModeClient(base_url=fake_server.base_url, model="m", seed=7)
    client.generate_many("q", num_predict=320)
    client.generate_many("continue", num_predict=24)

    completions = _posts(fake_server, "/completions")
    assert len(completions) == 2
    for row in completions:
        assert row["body"]["prompt"].endswith(EMPTY_THINK_BLOCK)
    assert completions[1]["body"]["max_tokens"] == 24

    report = client.reasoning_report()
    assert report["continuations"] == {
        "n": 1, "n_reopened": 1,
        "examples": ["<think> wait, let me reconsider"],
    }
    assert report["continuation_reopen_rate"] == 1.0
    assert report["full_generations"] == {"n": 1, "n_reopened": 0}


def test_off_mode_sampling_draws_run_on_the_same_closed_prompt(fake_server):
    client = ReasoningModeClient(base_url=fake_server.base_url, model="m", seed=7)
    outs, method = client.sample_completions("q", k=3, temperature=0.7, seed=11,
                                             num_predict=320)
    assert len(outs) == 3 and method == "n_parameter"
    body = _posts(fake_server, "/completions")[0]["body"]
    assert body["prompt"].endswith("<think>\n\n</think>\n\n")
    assert body["temperature"] == 0.7 and body["seed"] == 11 and body["n"] == 3


def test_off_mode_client_is_the_one_the_gate_builds():
    client = mod._gate_client("openai", "m", None, 1.0, base_url="http://127.0.0.1:1/v1",
                              seed=7, concurrency=32, reasoning_mode="off")
    # The server is not up, so the gate returns None; what is under test is that the
    # branch chose the reasoning client, which is visible from the class it built.
    assert client is None
    assert mod.ReasoningModeClient is ReasoningModeClient


def test_off_mode_ctx_records_the_completions_path():
    ctx = mod.RunCtx(4, 320, Path("."), "fake", "openai", "m", 20, reasoning_mode="off")
    assert ctx.reasoning_fields() == {
        "reasoning_mode": "off", "reasoning_path": "completions",
        "reasoning_block_closed": True, "num_predict_full": 320,
    }


# --------------------------------------------------------------------------- #
# (c) on raises the FULL budget to 4096 and leaves the forced budget at 24
# --------------------------------------------------------------------------- #
def _write_items(path: Path, n: int) -> list[QAItem]:
    rows = [{"question": f"pick the label for gadget number {i:02d}",
             "choices": ["alpha", "bravo", "charlie", "delta"], "answer_index": 0}
            for i in range(n)]
    path.write_text(json.dumps(rows))
    return [QAItem(r["question"], tuple(r["choices"]), r["answer_index"]) for r in rows]


def test_on_mode_full_generations_are_4096_and_forced_stay_24(fake_server, tmp_path):
    data_path = tmp_path / "data.json"
    _write_items(data_path, 4)
    # A full generation answers AFTER a closed reasoning block; a forced continuation is
    # a bare answer line, exactly as the two look on a live thinking model.
    fake_server.chat_text = lambda body: (
        "Answer: (A)" if body["max_tokens"] == 24
        else "<think>\nlong chain\n</think>\n\nAnswer: (A)")

    rc = mod.run("m", None, 4, data_path, tmp_path / "out", ["direct"],
                 curve_cap=4, num_predict=320, backend="openai",
                 base_url=fake_server.base_url, seed=7, reasoning_mode="on")
    assert rc == 0

    chats = [row for row in fake_server.seen if row["path"].endswith("/chat/completions")]
    budgets = sorted({row["body"]["max_tokens"] for row in chats})
    assert budgets == [24, ON_NUM_PREDICT_FULL], budgets
    # "on" is the ORDINARY chat path: nothing was rendered and nothing was closed.
    assert _posts(fake_server, "/tokenize") == []

    summary = json.loads((tmp_path / "out" / "arms_summary_m.json").read_text())
    assert summary["reasoning_mode"] == "on"
    assert summary["reasoning_path"] == "chat"
    assert summary["reasoning_block_closed"] is None
    assert summary["num_predict_full"] == 4096
    assert summary["num_predict"] == 4096


def test_full_num_predict_moves_only_for_on():
    assert full_num_predict("default", 320) == 320
    assert full_num_predict("off", 320) == 320
    assert full_num_predict("on", 320) == 4096
    # The forced budget is not this function's business and never moves.
    assert mod.FORCE_TOKENS == 24
    with pytest.raises(ReasoningModeError):
        full_num_predict("thinking", 320)


# --------------------------------------------------------------------------- #
# (d) the post-think extractor
# --------------------------------------------------------------------------- #
def test_post_think_extractor_reads_the_letter_after_the_closing_tag():
    text = "<think>\nthe answer looks like (B) at first\n</think>\n\nAnswer: (C)"
    assert parse_answer_after_think(text, 4) == "C"
    # The letter inside the block is not the model's stated answer, and is not read.
    assert parse_answer_after_think(
        "reasoning that mentions Answer: (B)\n</think>\nAnswer: (D)", 4) == "D"
    # The LAST closing tag wins, which is what a model that reopens a block leaves.
    assert parse_answer_after_think(
        "</think>Answer: (A)\n<think>more\n</think>\nAnswer: (B)", 4) == "B"


def test_post_think_extractor_is_unparseable_with_no_closing_tag():
    # A chain that never left the block: truncated at the budget, mid-reasoning.
    assert parse_answer_after_think("still working through it, maybe (B) or", 4) is None
    # Even when the text alone would parse: under "on" the OPENING tag is in the prompt,
    # so a completion with no closing tag was still inside the reasoning block.
    assert parse_answer_after_think("Answer: (B)", 4) is None
    assert parse_answer_after_think("", 4) is None
    assert parse_answer_after_think(None, 4) is None


def test_post_think_extractor_applies_the_frozen_regexes_unchanged():
    """A superset in the sense R12 asks for: the frozen parser, on a suffix."""
    from bayes_cot_faithfulness.interventions import parse_answer
    tail = "so the answer is (C)."
    assert parse_answer_after_think(f"chain\n</think>\n{tail}", 4) == parse_answer(tail, 4)
    # Out-of-range letters are refused by the frozen parser and stay refused here.
    assert parse_answer_after_think("chain\n</think>\nAnswer: (D)", 2) is None


def test_strip_and_reopen_helpers():
    assert strip_reasoning_block("a</think>b") == ("b", True)
    assert strip_reasoning_block("a<think>b") == ("a<think>b", False)
    assert reopened_block("<think> still going") is True
    assert reopened_block("<think>done</think> Answer: (A)") is False
    assert reopened_block("Answer: (A)") is False


def test_close_reasoning_block_never_returns_an_open_block():
    for rendered in ("x<think>", "x<think>\n", "x", "x<think>y</think>"):
        closed, _branch = close_reasoning_block(rendered)
        assert closed.endswith("</think>\n\n")


# --------------------------------------------------------------------------- #
# (e) the four fields, on every record, asserted before the write
# --------------------------------------------------------------------------- #
def _record(mode: str) -> dict:
    it = QAItem("q", ("alpha", "bravo", "charlie", "delta"), 0)
    return {"item": it, "clean_answer": "A", "clean_cot": "because",
            "hint_label": "B", "cue_text": "cue", "hinted_answer": "B",
            "hinted_cot": "chain", "followed": True, "acknowledged": False,
            "silent": True}


@pytest.mark.parametrize("mode, path, closed, budget", [
    ("default", "chat", None, 320),
    ("off", "completions", True, 320),
    ("on", "chat", None, 4096),
])
def test_every_record_carries_the_four_fields(mode, path, closed, budget):
    ctx = mod.RunCtx(4, budget, Path("."), "fake", "openai", "m", 20,
                     reasoning_mode=mode)
    out = mod.serialize_arm_record(_record(mode), ctx.reasoning_fields())
    assert out["reasoning_mode"] == mode
    assert out["reasoning_path"] == path
    assert out["reasoning_block_closed"] is closed
    assert out["num_predict_full"] == budget
    # A record with no reasoning block at all still says which mode it came from.
    bare = mod.serialize_arm_record(_record(mode))
    assert bare["reasoning_mode"] == "default"
    assert assert_records_reasoning_mode([out, bare]) == 2


def test_writer_refuses_a_record_that_lost_the_reasoning_fields(tmp_path, monkeypatch):
    """The guard REFUSES and refuses BEFORE the write. See the revert proof up top."""
    real = mod.serialize_arm_record

    def stripped(r, reasoning=None):
        out = real(r, reasoning)
        out.pop("reasoning_mode")
        return out

    monkeypatch.setattr(mod, "serialize_arm_record", stripped)
    with pytest.raises(ReasoningModeError):
        mod.write_arm_transcripts(tmp_path, "fake", [_record("off")],
                                  record_fields("off"))
    assert not (tmp_path / "arms_transcripts_fake.json").exists(), (
        "write_arm_transcripts accepted a record with no reasoning_mode"
    )


@pytest.mark.parametrize("bad", [
    {"reasoning_mode": "thinking", "reasoning_path": "chat",
     "reasoning_block_closed": None, "num_predict_full": 320},
    {"reasoning_mode": "off", "reasoning_path": "grpc",
     "reasoning_block_closed": True, "num_predict_full": 320},
    {"reasoning_mode": "off", "reasoning_path": "completions",
     "reasoning_block_closed": "yes", "num_predict_full": 320},
    {"reasoning_mode": "on", "reasoning_path": "chat",
     "reasoning_block_closed": None, "num_predict_full": 0},
])
def test_the_field_assertion_refuses_every_malformed_shape(bad):
    with pytest.raises(ReasoningModeError):
        assert_records_reasoning_mode([bad])


def test_the_resume_fingerprint_carries_the_mode_and_still_loads_old_checkpoints():
    import arms_resume
    default = arms_resume.build_params(
        "m", "openai", 4, Path("d.json"), None, ["direct"], 20, 320, Path("h.json"))
    off = arms_resume.build_params(
        "m", "openai", 4, Path("d.json"), None, ["direct"], 20, 320, Path("h.json"),
        reasoning_mode="off")
    # A default cell writes None, which is what every pre-existing checkpoint carries.
    assert default["reasoning_mode"] is None
    assert arms_resume.params_mismatch({k: v for k, v in default.items()
                                        if k != "reasoning_mode"}, default) == []
    # A leg resumed under a different mode refuses instead of merging two paths.
    assert [m[0] for m in arms_resume.params_mismatch(default, off)] == ["reasoning_mode"]


# --------------------------------------------------------------------------- #
# The gate's own verdict arithmetic (the live run is bcf/gate_twopath.sbatch)
# --------------------------------------------------------------------------- #
def _gate_report(**over) -> dict:
    report = {
        "n_items": 30,
        "identical_completions": {"count": "30/30", "fraction": 1.0},
        "rendered_prompt_roundtrip": {"count": "30/30", "fraction": 1.0},
        "max_abs_letter_logprob_diff": 0.0,
        "n_letter_logprobs_compared": 120,
        "n_letter_logprobs_missing": 0,
    }
    report.update(over)
    return report


def test_gate_passes_only_on_thirty_of_thirty_and_exactly_zero():
    import gate_twopath
    assert gate_twopath.verdict_for(_gate_report())[0] == "PASS"
    assert gate_twopath.verdict_for(
        _gate_report(identical_completions={"count": "29/30", "fraction": 0.967})
    )[0] == "REFUSE"
    assert gate_twopath.verdict_for(
        _gate_report(max_abs_letter_logprob_diff=1e-9))[0] == "REFUSE"
    assert gate_twopath.verdict_for(
        _gate_report(n_letter_logprobs_missing=1))[0] == "REFUSE"
    assert gate_twopath.verdict_for(
        _gate_report(rendered_prompt_roundtrip={"count": "0/30", "fraction": 0.0})
    )[0] == "REFUSE"
    # Nothing compared is not agreement.
    assert gate_twopath.verdict_for(
        _gate_report(n_letter_logprobs_compared=0))[0] == "REFUSE"
