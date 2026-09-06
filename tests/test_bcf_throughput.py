"""Offline tests for bcf/throughput.py, the measured-throughput reducer.

The number this script produces goes into CONTRACT.md as the compute budget of record,
replacing a first-order estimate. So the property that matters is CONSERVATION: every
logged call is assigned to exactly one arm, and no wall-clock second is invented or
lost. A rate whose denominator quietly drifts is worse than an honest estimate.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bcf_throughput", Path(__file__).resolve().parents[1] / "bcf" / "throughput.py"
)
throughput = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(throughput)

BASE = 1_757_170_000.0


def _stamp(offset: float) -> str:
    return datetime.fromtimestamp(BASE + offset, tz=timezone.utc).isoformat()


def _line(offset: float, text: str) -> str:
    return f"[{_stamp(offset)}] {text}"


def _call(offset: float, *, max_tokens: int = 320, chars: int = 500) -> dict:
    return {"t_start": BASE + offset, "t_end": BASE + offset + 1.0,
            "path": "/chat/completions", "max_tokens": max_tokens,
            "n_choices": 1, "completion_chars": chars}


@pytest.fixture()
def run_log(tmp_path: Path) -> Path:
    path = tmp_path / "run.log"
    path.write_text("\n".join([
        _line(0, "[1/3] Clean substrate: 30 items on Qwen/Qwen3-8B (filter to clean-correct)"),
        _line(100, "[2/3] Cue pass (stated-hint:strong) on 24 clean-correct items"),
        _line(200, "[3/3] Additive arms: replay, direct"),
        _line(201, "      running arm 'replay'..."),
        _line(300, "      running arm 'direct'..."),
        _line(400, "[arms] finished at ... with status 0"),
    ]) + "\n")
    return path


def test_every_logged_call_is_assigned_to_exactly_one_arm(run_log, tmp_path):
    calls = [_call(10), _call(20), _call(150), _call(250), _call(350), _call(360)]
    report = throughput.measure(throughput.parse_boundaries(run_log), calls)
    assigned = sum(row["n_calls"] for row in report["arms"])
    assert assigned == len(calls), f"{assigned}/{len(calls)} calls assigned"


def test_arm_seconds_sum_to_the_full_run_with_no_gap(run_log):
    report = throughput.measure(throughput.parse_boundaries(run_log), [_call(10)])
    total = sum(row["seconds"] for row in report["arms"])
    assert total == pytest.approx(400.0), "the 400 s between first mark and end must be covered once"


def test_rates_are_calls_over_that_arms_own_seconds(run_log):
    # 4 calls inside the 100-second clean-substrate window.
    calls = [_call(10), _call(20), _call(30), _call(40)]
    report = throughput.measure(throughput.parse_boundaries(run_log), calls)
    clean = next(r for r in report["arms"] if r["arm"] == "clean_substrate")
    assert clean["n_calls"] == 4
    assert clean["seconds"] == pytest.approx(100.0)
    assert clean["generations_per_second"] == pytest.approx(0.04)


def test_forced_continuations_are_counted_apart_from_full_generations(run_log):
    # 24 tokens is the frozen forced-answer continuation length; 320 is a full generation.
    calls = [_call(10, max_tokens=320), _call(20, max_tokens=24), _call(30, max_tokens=24)]
    report = throughput.measure(throughput.parse_boundaries(run_log), calls)
    clean = next(r for r in report["arms"] if r["arm"] == "clean_substrate")
    assert clean["n_calls"] == 3
    assert clean["n_full_generations"] == 1
    assert clean["n_forced_continuations"] == 2
    assert clean["full_generations_per_second"] == pytest.approx(0.01)


def test_an_arm_with_no_calls_reports_zero_not_a_rate(run_log):
    report = throughput.measure(throughput.parse_boundaries(run_log), [_call(250)])
    replay = next(r for r in report["arms"] if r["arm"] == "replay")
    direct = next(r for r in report["arms"] if r["arm"] == "direct")
    assert replay["n_calls"] == 1
    assert direct["n_calls"] == 0
    assert direct["generations_per_second"] == 0.0  # 0 calls over real seconds, not None


def test_a_run_with_no_marks_invents_no_rate(tmp_path):
    empty = tmp_path / "run.log"
    empty.write_text("no timestamps here\n")
    report = throughput.measure(throughput.parse_boundaries(empty), [_call(10)])
    assert report["arms"] == []
    assert report["n_intervals"] == 0


def test_unparseable_request_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(_call(10)) + "\n{not json}\n\n" + json.dumps(_call(20)) + "\n")
    assert len(throughput.load_requests(path)) == 2


def test_a_missing_request_log_is_zero_calls_not_a_crash(tmp_path):
    assert throughput.load_requests(tmp_path / "nope.jsonl") == []


def test_main_writes_throughput_json_with_denominators(run_log, tmp_path, capsys):
    (tmp_path / "requests.jsonl").write_text(
        "\n".join(json.dumps(_call(o)) for o in (10, 20, 250)) + "\n"
    )
    assert throughput.main(["--run-log", str(run_log), "--out-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "throughput.json").read_text())
    assert payload["total_calls"] == 3
    assert payload["total_seconds"] == pytest.approx(400.0)
    assert payload["overall_generations_per_second"] == pytest.approx(3 / 400.0)
    out = capsys.readouterr().out
    assert "3 calls / 400.0 s" in out  # numerator and denominator, both printed
