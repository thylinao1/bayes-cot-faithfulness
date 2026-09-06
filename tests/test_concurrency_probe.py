"""The probe's flag-off comparison and its refusals, offline.

The batch-invariant question is answered by comparing a flag-ON run against a flag-OFF
concurrency-1 run. Job 825548 saved no raw completions, so that baseline has to be
written by one phase and read by the next, and the two places this can go quietly wrong
are a missing file and a file from a different item count. Both refuse, both before any
model call, and both are asserted here. No test in this file reaches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bcf" / "concurrency_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("concurrency_probe_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load()


def _items_file(tmp_path: Path, n: int) -> Path:
    path = tmp_path / "items.json"
    path.write_text(json.dumps([
        {"question": f"Question {i}?", "choices": ["a", "b", "c", "d"],
         "answer_index": i % 4}
        for i in range(n)
    ]))
    return path


def _raw(completions, logprobs, label="batch_invariant_off"):
    return {"label": label, "batch_invariant_env": None,
            "completions": completions, "logprobs": logprobs}


def test_compare_raw_reads_a_saved_baseline_the_same_way_as_a_live_one():
    baseline = _raw(["one", "two"], [{"A": -1.0, "B": -2.0}, {"A": -3.0, "B": -4.0}])
    level = {"_completions": ["one", "TWO"],
             "_logprobs": [{"A": -1.0, "B": -2.5}, {"A": -3.0, "B": -4.0}]}
    out = probe.compare_raw(baseline, level)
    assert out["identical_completions"] == "1/2"
    assert out["identical_completion_fraction"] == 0.5
    assert out["n_letter_logprobs_compared"] == 4
    assert out["max_abs_letter_logprob_diff"] == 0.5
    assert out["n_exactly_zero_diff"] == 3


def test_compare_raw_counts_a_letter_the_level_never_produced():
    baseline = _raw(["one"], [{"A": -1.0, "B": -2.0}])
    level = {"_completions": ["one"], "_logprobs": [{"A": -1.0}]}
    out = probe.compare_raw(baseline, level)
    assert out["n_letter_logprobs_missing"] == 1
    assert out["n_letter_logprobs_compared"] == 1


def test_a_missing_baseline_file_refuses_before_any_model_call(tmp_path, capsys):
    code = probe.main([
        "--base-url", "http://127.0.0.1:1/v1", "--model", "nothing",
        "--data", str(_items_file(tmp_path, 4)), "--n-items", "2", "--levels", "1",
        "--baseline-raw", str(tmp_path / "does_not_exist.json"),
        "--out", str(tmp_path / "out.json"),
    ])
    assert code == 2
    assert "REFUSING" in capsys.readouterr().out


def test_a_baseline_of_the_wrong_length_refuses_before_any_model_call(tmp_path, capsys):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_raw(["only one"], [{"A": -1.0}])))
    code = probe.main([
        "--base-url", "http://127.0.0.1:1/v1", "--model", "nothing",
        "--data", str(_items_file(tmp_path, 4)), "--n-items", "2", "--levels", "1",
        "--baseline-raw", str(raw), "--out", str(tmp_path / "out.json"),
    ])
    assert code == 2
    out = capsys.readouterr().out
    assert "REFUSING" in out and "off by index" in out


def test_the_first_level_must_still_be_the_baseline(tmp_path, capsys):
    code = probe.main([
        "--base-url", "http://127.0.0.1:1/v1", "--model", "nothing",
        "--data", str(_items_file(tmp_path, 4)), "--n-items", "2", "--levels", "8,32",
        "--out", str(tmp_path / "out.json"),
    ])
    assert code == 2
    assert "first level must be 1" in capsys.readouterr().out
