"""Element 11(d): the held-out mechanism family, generated only after the freeze.

    "The conditional second base is replaced by a held-out mechanism family on the
    first base, generated only after the instrument's parameters are frozen, and that
    freeze commit is dated and hashed."
    -- PREREGISTRATION_jury_and_scale.md section 12, element 11(d)

So this script refuses to generate anything without a freeze commit hash, and it writes
that hash and the date into the family's manifest. The refusal is the point: a held-out
family generated before the freeze is a development family with a held-out label, and
nothing downstream could tell the difference.

The freeze hash is also the RANDOMNESS. The family's parameters are drawn from a menu
fixed in this file by a stream seeded with the freeze commit, so the family cannot be
chosen after seeing what the instrument does: the only way to move it is to move the
freeze commit, which is dated and recorded. ``element 22`` (the decision experiment)
reads exactly this property -- "tested on held-out mechanism families from element 11
whose behaviour is established independently of the statistic being validated".

The family's shape is the ladder's own: an X-caused path that does not run through the
mediator, which section 2 calls an A4 violation and says rho does not price. Truth is
computed from the same structural equations that generate the data, on a common noise
draw, so the truth cannot drift from the generator.

    python -m bayes_cot_faithfulness.ladder.heldout_family \\
        --freeze-commit <sha> --out experiments/results/ladder/heldout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

SCHEMA = "bcf.ladder.heldout_family.v1"
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

# The menu the freeze hash draws from. Fixed here, before any freeze, so the draw has
# nothing to choose between except what this file already allowed.
MENU = {
    # direct X to Y path strength
    "gamma": (0.2, 0.4, 0.6, 0.8),
    # mediator to outcome path
    "beta": (0.4, 0.6, 0.8),
    # X to mediator path
    "alpha_m": (0.3, 0.6, 0.9),
    # the TRIGGER path: X-caused, not through M. This is the A4 violation.
    "delta_trigger": (0.5, 1.0, 1.5),
    # how often the trigger fires on a treated row
    "trigger_rate": (0.25, 0.5, 0.75),
    "sigma_m": (0.8, 1.0, 1.2),
    "n_rows": (500,),
}
N_DATASETS = 400          # element 12: at least 400 datasets per gate
TRUTH_MC = 200_000


class HeldOutFamilyError(RuntimeError):
    """A held-out family that would not be held out."""


def _stream(freeze_commit: str, label: str) -> int:
    h = hashlib.sha256(f"{freeze_commit}|{label}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def draw_parameters(freeze_commit: str) -> dict:
    """Pick the family's parameters from the menu, using the freeze commit as the seed."""
    out = {}
    for key, options in MENU.items():
        out[key] = options[_stream(freeze_commit, key) % len(options)]
    return out


def _truth(params: dict, seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    n = TRUTH_MC
    eps_m = rng.normal(0, params["sigma_m"], n)
    eps_y = rng.normal(0, 1.0, n)
    fires = rng.random(n) < params["trigger_rate"]

    def m_of(x):
        return params["alpha_m"] * x + eps_m

    def y_of(x, m):
        trig = params["delta_trigger"] * fires * x
        return (params["gamma"] * x + params["beta"] * m + trig + eps_y > 0).mean()

    m0, m1 = m_of(0), m_of(1)
    p00 = y_of(0, m0)
    p10 = y_of(1, m0)
    p11 = y_of(1, m1)
    return {"nde": float(p10 - p00), "nie": float(p11 - p10), "te": float(p11 - p00),
            "monte_carlo_rows": n, "seed": seed}


def generate(
    freeze_commit: str | None,
    *,
    out_dir: Path,
    date: str | None = None,
    n_datasets: int = N_DATASETS,
    repo: Path | None = None,
) -> dict:
    """Generate the held-out family, or refuse."""
    if not freeze_commit:
        raise HeldOutFamilyError(
            "REFUSING: element 11(d) generates the held-out mechanism family ONLY after "
            "the instrument's parameters are frozen, and that freeze commit is dated and "
            "hashed. No --freeze-commit was given, so there is nothing to date and "
            "nothing to hash, and a family generated now would be a development family "
            "wearing a held-out label."
        )
    commit = freeze_commit.strip().lower()
    if not _SHA_RE.match(commit):
        raise HeldOutFamilyError(
            f"REFUSING: --freeze-commit {freeze_commit!r} is not a git sha (7 to 40 hex "
            "characters). The recorded hash has to be resolvable by a reader later."
        )
    verified = None
    if repo is not None:
        try:
            subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
                           check=True, capture_output=True)
            verified = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            verified = False
    if verified is False:
        raise HeldOutFamilyError(
            f"REFUSING: {commit} is not a commit in {repo}. A freeze commit that does "
            "not exist cannot be checked by anyone reading the manifest."
        )

    import numpy as np

    params = draw_parameters(commit)
    stamp = date or time.strftime("%Y-%m-%d")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(_stream(commit, "data") % (2 ** 32))
    rows_path = out_dir / "datasets.jsonl"
    n = params["n_rows"]
    with rows_path.open("w") as fh:
        for d in range(n_datasets):
            x = rng.integers(0, 2, n)
            eps_m = rng.normal(0, params["sigma_m"], n)
            m = params["alpha_m"] * x + eps_m
            fires = (rng.random(n) < params["trigger_rate"]).astype(int)
            eta = (params["gamma"] * x + params["beta"] * m
                   + params["delta_trigger"] * fires * x + rng.normal(0, 1.0, n))
            y = (eta > 0).astype(int)
            fh.write(json.dumps({
                "dataset": d,
                "x": x.tolist(), "m": [round(v, 6) for v in m.tolist()],
                "y": y.tolist(), "trigger_fired": fires.tolist(),
            }) + "\n")

    manifest = {
        "schema": SCHEMA,
        "family": "f8_heldout_trigger_conditioned",
        "freeze_commit": commit,
        "freeze_commit_verified_in_repo": verified,
        "generated_on": stamp,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "why_the_hash_is_the_seed": (
            "the family's parameters are drawn from MENU by a stream seeded with the "
            "freeze commit, so the family cannot be chosen after seeing what the "
            "instrument does; moving it means moving the freeze commit, which is dated "
            "and recorded here"
        ),
        "parameters": params,
        "menu": {k: list(v) for k, v in MENU.items()},
        "n_datasets": n_datasets,
        "n_rows_per_dataset": n,
        "truth": _truth(params, seed=_stream(commit, "truth") % (2 ** 32)),
        "structural_form": (
            "M = alpha_m*X + eps_M; Y = 1[gamma*X + beta*M + delta_trigger*T*X + eps_Y "
            "> 0] with T ~ Bernoulli(trigger_rate). The delta_trigger term is an "
            "X-caused path that does not run through M, which section 2 assumption A4 "
            "says rho does not price."
        ),
        "prereg": "PREREGISTRATION_jury_and_scale.md section 12, element 11(d)",
        "files": {"datasets.jsonl": hashlib.sha256(rows_path.read_bytes()).hexdigest()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Deliberately NOT required=True: argparse's own error would exit 2 with a usage
    # string, and element 11(d) deserves a refusal that says why.
    ap.add_argument("--freeze-commit", default=None,
                    help="the instrument freeze commit; without it this script refuses")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default=None, help="defaults to today")
    ap.add_argument("--n-datasets", type=int, default=N_DATASETS)
    ap.add_argument("--repo", type=Path, default=None,
                    help="check the freeze commit exists in this git checkout")
    a = ap.parse_args(argv)
    try:
        manifest = generate(a.freeze_commit, out_dir=a.out, date=a.date,
                            n_datasets=a.n_datasets, repo=a.repo)
    except HeldOutFamilyError as exc:
        print(str(exc))
        return 3
    print(json.dumps({
        "family": manifest["family"], "freeze_commit": manifest["freeze_commit"],
        "generated_on": manifest["generated_on"], "parameters": manifest["parameters"],
        "n_datasets": manifest["n_datasets"], "truth": manifest["truth"],
        "out": str(a.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
