"""Parse the EXACT argv bcf/serve_and_run.sbatch will build for the smoke run, off-cluster.

A flag the runner does not accept, or an arm name it rejects, costs a card allocation and
a model download before it is discovered. This builds the same argument list the sbatch
builds from bcf/a3f_smoke.sh's exports and hands it to the runner's own parser. It runs no
model, contacts no server and writes nothing.

    python bcf/a3f_smoke_argv_check.py     # exit 0 when the twelve arms parse, 1 otherwise
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ARMS = ("replay placebo direct twostep filler curves sampling repeat-curves "
        "chain-repeats transplant anchor specificity").split()

ARGV = [
    "--backend", "openai", "--base-url", "http://127.0.0.1:8000/v1",
    "--model", "Qwen/Qwen3-8B", "--seed", "7",
    "--chat-template-kwargs", '{"enable_thinking": false}',
    "--data", str(REPO / "experiments/data/arc_challenge.json"),
    "--specificity-holdout", str(REPO / "experiments/data/specificity_holdout.json"),
    "--n-items", "30", "--num-predict", "320", "--curve-cap", "30",
    "--concurrency", "32", "--sampling-k", "32", "--sampling-temperature", "0.7",
    "--curve-repeats", "3", "--repeat-temperatures", "0.0,0.7",
    "--chain-repeats", "3", "--chain-repeat-temperature", "0.7",
    "--chain-repeat-seed", "20260907",
    "--out", "/tmp/bcf-a3f-argv-check", "--timeout", "600", "--resume",
]
for arm in ARMS:
    ARGV += ["--arm", arm]


def main() -> int:
    spec = importlib.util.spec_from_file_location("aa", REPO / "experiments/08_additive_arms.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aa"] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    parser = module.build_parser()
    try:
        ns = parser.parse_args(ARGV)
    except SystemExit as exc:
        print(f"REFUSED by the runner's own parser (SystemExit {exc.code})")
        return 1
    ok = list(ns.arm) == ARMS
    print(f"arms parsed: {ns.arm}")
    print(f"n_items={ns.n_items} curve_cap={ns.curve_cap} concurrency={ns.concurrency} "
          f"chain_repeats={ns.chain_repeats} chain_repeat_temperature={ns.chain_repeat_temperature} "
          f"chain_repeat_seed={ns.chain_repeat_seed}")
    print(f"twelve arms in order: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
