"""Re-evaluate every wave-1 determinism probe with the campaign's OWN evaluate(), and
prove the re-evaluation can fail.

Trusting `determinism_preflight.json`'s verdict field would only check that the cell job
wrote a word. This reads each cell's PROBE and runs bcf/determinism_preflight.evaluate()
over it again, then perturbs a copy of one probe six ways and requires every perturbation
to refuse. A check that never refuses certifies nothing.

  python experiments/audit/verify_preflights.py experiments/results/wave1
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bcf"))
import determinism_preflight as dp

LEVELS = (1, 32)
N_ITEMS = 30


def load(p: Path):
    with p.open() as fh:
        return json.load(fh)


def cmp32(d):
    return next(r for r in d["rows"] if r.get("concurrency") == 32)["vs_concurrency_1"]


PERTURBATIONS = [
    ("one non-identical completion", lambda d: cmp32(d).update({"identical_completions": "29/30"})),
    ("a 1e-9 letter-logprob diff", lambda d: cmp32(d).update({"max_abs_letter_logprob_diff": 1e-9})),
    ("one missing letter logprob", lambda d: cmp32(d).update({"n_letter_logprobs_missing": 1})),
    ("zero logprobs compared", lambda d: cmp32(d).update({"n_letter_logprobs_compared": 0})),
    ("the batch-invariant flag off", lambda d: d.update({"batch_invariant_env": "0"})),
    ("only 29 items probed", lambda d: d.update({"n_items": 29})),
]


def main(root: Path) -> int:
    rc = 0
    proof_base = None
    for cell in sorted(p for p in root.iterdir() if p.is_dir()):
        probe = cell / "determinism_preflight_probe.json"
        verdict = cell / "determinism_preflight.json"
        if not probe.exists():
            print(f"{cell.name:30s} NO PROBE ON DISK")
            continue
        payload = load(probe)
        got = dp.evaluate(payload, levels=LEVELS, n_items=N_ITEMS, require_flag=True)
        stored = load(verdict)["verdict"] if verdict.exists() else "NO VERDICT FILE"
        agree = got["verdict"] == stored
        rc |= 0 if agree else 1
        rows = {r["concurrency"]: r["vs_concurrency_1"] for r in payload["rows"]}
        detail = "; ".join(
            f"c={lv} {rows[lv]['identical_completions']} diff "
            f"{rows[lv]['max_abs_letter_logprob_diff']} "
            f"({rows[lv]['n_letter_logprobs_compared']} compared, "
            f"{rows[lv]['n_letter_logprobs_missing']} missing)"
            for lv in LEVELS if lv in rows
        )
        print(f"{cell.name:30s} re-evaluated {got['verdict']:6s} stored {stored:6s} "
              f"agree={agree} | {detail}")
        if got["verdict"] == "PASS" and proof_base is None:
            proof_base = (cell.name, payload)

    if proof_base is None:
        print("\nNo passing probe to perturb; the fail proof did not run.")
        return rc | 1
    name, base = proof_base
    print(f"\nFAIL PROOF on {name} (in memory; nothing on disk is touched):")
    all_refuse = True
    for label, mut in PERTURBATIONS:
        d = copy.deepcopy(base)
        mut(d)
        v = dp.evaluate(d, levels=LEVELS, n_items=N_ITEMS, require_flag=True)
        all_refuse &= v["verdict"] != "PASS"
        print(f"  {label:32s} -> {v['verdict']:6s} {v['reasons'][:1]}")
    print("FAIL PROOF:", "every perturbation refuses"
          if all_refuse else "SOMETHING STILL PASSES -- the check certifies nothing")
    return rc | (0 if all_refuse else 1)


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/wave1")))
