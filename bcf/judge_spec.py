"""Print one judge's serving parameters as shell assignments, from the single source.

The sbatch script must not carry its own copy of an HF id, a revision or a memory
fraction. It asks this, which reads `experiments/jury/family_map.py`, so a judge pin lives
in exactly one place and a shell typo cannot serve a different revision than the code
believes it served.

    eval "$(python bcf/judge_spec.py qwen3-32b)"
    echo "$JUDGE_HF_ID @ $JUDGE_REVISION on $JUDGE_GPU_TYPE"
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.jury.family_map import JUDGE_BY_KEY  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: judge_spec.py <judge_key>", file=sys.stderr)
        print(f"known: {' '.join(sorted(JUDGE_BY_KEY))}", file=sys.stderr)
        return 2
    key = argv[1]
    judge = JUDGE_BY_KEY.get(key)
    if judge is None:
        print(f"unknown judge key {key!r}; known: {sorted(JUDGE_BY_KEY)}", file=sys.stderr)
        return 2
    fields = {
        "JUDGE_KEY": judge.key,
        "JUDGE_FAMILY": judge.family,
        "JUDGE_HF_ID": judge.hf_id,
        "JUDGE_REVISION": judge.revision,
        "JUDGE_GPU_TYPE": judge.gpu_type,
        "JUDGE_GPU_UTIL": str(judge.gpu_memory_utilization),
        "JUDGE_MAX_LEN": str(judge.max_model_len),
        "JUDGE_QUANT": judge.quantization or "",
        "JUDGE_SERVING_LINE": judge.serving_line,
    }
    for name, value in fields.items():
        print(f"export {name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
