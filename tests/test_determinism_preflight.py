"""The R1 preflight gate, and the proof that it can fail.

A gate that has only ever been seen to pass is not evidence. Every test below builds a
probe payload with ``concurrency_probe.compare`` on real completion and logprob lists,
so the fractions and the differences the gate reads are computed by the same code the
cluster runs, not typed in by hand. The falsification alters ONE completion out of the
thirty and asserts the gate refuses with its own exit code.

Nothing here reaches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("concurrency_probe_pf", REPO / "bcf" / "concurrency_probe.py")
pf = _load("determinism_preflight_test", REPO / "bcf" / "determinism_preflight.py")

N = 30
LETTERS = ("A", "B", "C", "D")


def _completions(n: int = N) -> list[str]:
    return [f"chain for item {i}\nANSWER: {LETTERS[i % 4]}" for i in range(n)]


def _logprobs(n: int = N) -> list[dict]:
    return [{L: -0.5 - i * 0.01 - j for j, L in enumerate(LETTERS)} for i in range(n)]


def _payload(rows_completions: dict[int, list[str]],
             rows_logprobs: dict[int, list[dict]] | None = None,
             *, env: str | None = "1", n_items: int = N) -> dict:
    """Build a probe_results.json exactly as concurrency_probe.main would write one."""
    logprobs = rows_logprobs or {}
    base = {"_completions": rows_completions[1],
            "_logprobs": logprobs.get(1, _logprobs(n_items))}
    rows = []
    for level, comps in rows_completions.items():
        level_res = {"_completions": comps,
                     "_logprobs": logprobs.get(level, _logprobs(n_items))}
        rows.append({
            "concurrency": level,
            "n_generations": len(comps),
            "generations_per_second": 0.1433 if level == 1 else 2.9196,
            "logprob_calls_per_second": 11.201 if level == 1 else 30.7345,
            "vs_concurrency_1": probe.compare(base, level_res),
        })
    return {"model": "Qwen/Qwen3-8B", "n_items": n_items, "label": "determinism_preflight",
            "batch_invariant_env": env, "rows": rows}


def test_gate_passes_when_every_level_reproduces_the_concurrency_1_run():
    payload = _payload({1: _completions(), 32: _completions()})
    v = pf.evaluate(payload)
    assert v["verdict"] == "PASS"
    assert v["exit_code"] == 0
    assert v["reasons"] == []
    checked = {row["concurrency"]: row for row in v["levels_checked"]}
    assert checked[32]["identical_completions"] == "30/30"
    assert checked[32]["max_abs_letter_logprob_diff"] == 0.0
    assert checked[32]["n_letter_logprobs_compared"] == 120
    assert checked[32]["n_letter_logprobs_missing"] == 0


def test_one_altered_completion_out_of_thirty_refuses_the_cell():
    """THE FALSIFICATION. Change one of the thirty completions and nothing else."""
    at_32 = _completions()
    at_32[17] = at_32[17] + " (one token different)"
    payload = _payload({1: _completions(), 32: at_32})
    v = pf.evaluate(payload)
    assert v["verdict"] == "REFUSE"
    assert v["exit_code"] == pf.EXIT_REFUSE == 10
    checked = {row["concurrency"]: row for row in v["levels_checked"]}
    assert checked[32]["identical_completions"] == "29/30"
    assert any("29/30" in r for r in v["reasons"])
    # and the level that was untouched is still reported as clean, so the refusal names
    # WHICH level moved rather than condemning the whole probe.
    assert checked[1]["identical_completions"] == "30/30"


def test_a_nonzero_letter_logprob_difference_refuses_even_with_identical_text():
    """The two halves of the check are independent: text can agree while a logprob moves.

    This is the 0.125-nat median the flag-off server showed at 8 in flight (job 825548)
    on completions that happened to match.
    """
    moved = _logprobs()
    moved[3]["B"] = moved[3]["B"] - 0.125
    payload = _payload({1: _completions(), 32: _completions()},
                       {1: _logprobs(), 32: moved})
    v = pf.evaluate(payload)
    assert v["verdict"] == "REFUSE"
    checked = {row["concurrency"]: row for row in v["levels_checked"]}
    assert checked[32]["identical_completions"] == "30/30"
    assert abs(checked[32]["max_abs_letter_logprob_diff"] - 0.125) < 1e-12
    assert any("0.125" in r for r in v["reasons"])


def test_a_flag_off_probe_refuses_a_powered_cell_and_is_allowed_only_when_asked():
    payload = _payload({1: _completions(), 32: _completions()}, env=None)
    assert pf.evaluate(payload)["verdict"] == "REFUSE"
    assert any("VLLM_BATCH_INVARIANT" in r for r in pf.evaluate(payload)["reasons"])
    # The exploratory label is reachable, and it is the ONLY way past this condition.
    assert pf.evaluate(payload, require_flag=False)["verdict"] == "PASS"


def test_a_missing_level_refuses_rather_than_passing_on_the_levels_present():
    payload = _payload({1: _completions()})
    v = pf.evaluate(payload)
    assert v["verdict"] == "REFUSE"
    assert any("concurrency 32 was not probed" in r for r in v["reasons"])


def test_a_short_probe_refuses_because_30_of_30_is_unreachable_on_29_items():
    payload = _payload({1: _completions(29), 32: _completions(29)}, n_items=29)
    v = pf.evaluate(payload)
    assert v["verdict"] == "REFUSE"
    assert any("29 items" in r for r in v["reasons"])


def test_a_missing_letter_logprob_is_not_an_equal_one():
    dropped = _logprobs()
    dropped[5] = {k: v for k, v in dropped[5].items() if k != "D"}
    payload = _payload({1: _completions(), 32: _completions()},
                       {1: _logprobs(), 32: dropped})
    v = pf.evaluate(payload)
    assert v["verdict"] == "REFUSE"
    checked = {row["concurrency"]: row for row in v["levels_checked"]}
    assert checked[32]["n_letter_logprobs_missing"] == 1
    assert any("missing" in r for r in v["reasons"])


def test_cli_writes_the_verdict_beside_the_cell_and_returns_the_refusal_code(tmp_path):
    at_32 = _completions()
    at_32[0] = "a different chain entirely"
    bad = tmp_path / "probe.json"
    bad.write_text(json.dumps(_payload({1: _completions(), 32: at_32})))
    out = tmp_path / "cell" / "determinism_preflight.json"
    rc = pf.main(["--probe-json", str(bad), "--out", str(out)])
    assert rc == 10
    written = json.loads(out.read_text())
    assert written["verdict"] == "REFUSE"
    assert written["probe_source"] == str(bad)

    good = tmp_path / "probe_ok.json"
    good.write_text(json.dumps(_payload({1: _completions(), 32: _completions()})))
    rc = pf.main(["--probe-json", str(good), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"


# --- the sbatch defaults the ruling changes, asserted rather than assumed ---------
#
# R1 makes the flag and concurrency 32 the DEFAULTS, not options a wave has to remember.
# A default that lives only in a comment is not a default, so the shipped submission
# script is read and checked, and the checker is shown able to reject.

SBATCH = REPO / "bcf" / "serve_and_run.sbatch"


def _serving_defaults(text: str) -> dict:
    """Read the three defaults out of the submission script's own parameter block."""
    import re

    def default_for(var: str) -> str | None:
        m = re.search(rf'^[A-Z_]+="\$\{{{var}:-([^}}]*)\}}"', text, re.MULTILINE)
        return m.group(1) if m else None

    return {
        "concurrency": default_for("BCF_CONCURRENCY"),
        "batch_invariant": default_for("BCF_BATCH_INVARIANT"),
        "preflight": default_for("BCF_PREFLIGHT"),
    }


def test_the_shipped_sbatch_defaults_to_the_pinned_serving_mode():
    got = _serving_defaults(SBATCH.read_text())
    assert got == {"concurrency": "32", "batch_invariant": "1", "preflight": "1"}, got


def test_the_defaults_check_rejects_a_script_that_reverts_them():
    """The falsification for the check above: revert one default, see it fail."""
    reverted = SBATCH.read_text().replace(
        'CONCURRENCY="${BCF_CONCURRENCY:-32}"', 'CONCURRENCY="${BCF_CONCURRENCY:-1}"'
    )
    assert _serving_defaults(reverted)["concurrency"] == "1"


def test_the_refusal_exit_code_is_wired_into_the_sbatch():
    text = SBATCH.read_text()
    assert "exit 10" in text
    assert "determinism_preflight.py" in text
    # and the preflight runs BEFORE the arms, not after them
    assert text.index("determinism_preflight.py") < text.index("08_additive_arms.py")
