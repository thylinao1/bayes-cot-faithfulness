#!/bin/bash
# Is the bcf conda env able to train a LoRA checkpoint? Answer with exact versions.
#
# NOT RUN YET. The cluster link was down for the lane that wrote this
# (2026-09-08), so every version below is printed by the script and none is
# recorded here. What IS recorded here is the pin and where the pin came from.
#
# WHY THIS EXISTS. docs/LADDER-IMPL.md, "what is NOT done", item 2: peft,
# transformers and torch are not in this repository's laptop venv, the real
# training path in bayes_cot_faithfulness.ladder.lora_train has never executed
# anywhere, and its test skips. On the cluster the first two of those three are
# present (bcf/env.sh header: the env was built by job 825198 with vllm 0.28.0 and
# torch 2.13.0+cu130, and vllm 0.28.0's own metadata requires transformers>=5.5.3).
# peft is the one that has to be added, and adding it is the one step that can
# break the serving env if it drags a different torch or transformers in behind it.
#
# Run it standalone on the login node:      bash bcf/ladder_env_check.sh
# bcf/ladder_train.sbatch runs it first and refuses when peft is missing.
#
# Exit codes: 0 ready; 20 peft is missing; 21 another required package is missing;
#             22 peft is present but transformers is older than the pin needs.
#
# ---------------------------------------------------------------------------------
# THE INSTALL, one line, into the bcf env and nothing else:
#
#     conda activate bcf && pip install --no-deps peft==0.20.0
#
# --no-deps is the whole point. peft's declared dependencies are numpy, packaging,
# psutil, pyyaml, torch, transformers, tqdm, accelerate>=0.21.0, safetensors and
# huggingface-hub>=0.25.0 (PyPI metadata for peft 0.20.0, read 2026-09-08). Resolving
# those is what would replace torch 2.13.0+cu130 or move transformers off the version
# vllm 0.28.0 was installed against, and a serving env that moved is a different
# serving mode under ruling R1. The five names this script checks are exactly the
# dependencies --no-deps then leaves the operator responsible for.
#
# WHY 0.20.0, from peft's own release notes (github.com/huggingface/peft/releases,
# read 2026-09-08, not from memory):
#   * v0.18.0, 2025-11-13: "This Transformers version [v5] will be incompatible with
#     PEFT < 0.18.0. If you plan to use Transformers v5 with PEFT, please upgrade PEFT
#     to 0.18.0+." vllm 0.28.0 requires transformers>=5.5.3, so the bcf env is on
#     transformers 5.x and every peft below 0.18.0 is ruled out.
#   * v0.20.0, 2026-07-28, the latest release: carries "FIX Clip module structure since
#     transformers > 5.5" (PR 3179) and three further transformers-weight-conversion
#     fixes (PRs 3197, 3262, 3230). Those are fixes AGAINST the transformers line this
#     env is on, which is why the pin is the newest release rather than the oldest
#     compatible one.
#
# VERIFY, and this is the half that is not settled: nobody has read the bcf env's
# actual transformers version. The pin above is correct for transformers >= 5.5.3,
# which is what vllm 0.28.0's metadata requires, but the env could carry something
# else. Run this script FIRST, read the transformers line it prints, and only then
# run the install. If transformers turns out to be 4.x, the pin is wrong and the
# right answer is peft 0.17.x, not this line. VERIFY.
# ---------------------------------------------------------------------------------

set -uo pipefail

PEFT_PIN="0.20.0"
TRANSFORMERS_MIN="5.5.3"

echo "=== bcf ladder env check on $(hostname) at $(date -Is) ==="
echo "python: $(command -v python || echo '<none>')"
echo "conda env: ${CONDA_DEFAULT_ENV:-<none>}"

python - "$PEFT_PIN" "$TRANSFORMERS_MIN" <<'PY'
import importlib
import sys

peft_pin, transformers_min = sys.argv[1], sys.argv[2]
REQUIRED = ("torch", "transformers", "peft", "accelerate", "safetensors")

def version_of(name):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:                      # ImportError, and anything a broken
        return None, f"{type(exc).__name__}: {exc}"   # native extension raises on import
    return getattr(mod, "__version__", "<no __version__ attribute>"), None

def as_tuple(v):
    out = []
    for part in str(v).split("+")[0].split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)

print(f"python: {sys.version.split()[0]}")
found, missing = {}, []
for name in REQUIRED:
    version, err = version_of(name)
    if version is None:
        missing.append(name)
        print(f"{name:<14} MISSING   ({err})")
    else:
        found[name] = version
        print(f"{name:<14} {version}")

if "peft" in missing:
    print("\nREFUSING: peft is not in this env. Install it with exactly:")
    print(f"    pip install --no-deps peft=={peft_pin}")
    print("  --no-deps so the resolver cannot replace torch or transformers under vLLM.")
    sys.exit(20)

others = [m for m in missing if m != "peft"]
if others:
    print(f"\nREFUSING: {', '.join(others)} missing. peft was installed with --no-deps, "
          "so these are the dependencies that stayed the operator's responsibility.")
    sys.exit(21)

t = as_tuple(found["transformers"])
want = as_tuple(transformers_min)
if t and t < want:
    print(f"\nREFUSING: transformers {found['transformers']} is older than {transformers_min}. "
          f"peft {peft_pin} is pinned FOR transformers 5.x (peft 0.18.0 release notes: "
          "PEFT < 0.18.0 is incompatible with Transformers v5). On a 4.x env the right "
          "pin is a 0.17.x peft, and this refusal is the signal to re-decide it, not to "
          "carry on.")
    sys.exit(22)

print(f"\nREADY: peft {found['peft']} against transformers {found['transformers']} "
      f"and torch {found['torch']}.")
sys.exit(0)
PY
RC=$?
echo "=== exit ${RC} ==="
exit "$RC"
