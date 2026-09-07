"""Record what was downloaded, from where, at which revision, and with which hashes.

FaithCoT-Bench (FINE-CoT) is THIRD-PARTY data used under a written, evaluation-only
permission recorded in ~/Developer/bayes-cot-phase2/PERMISSIONS.md. The data itself is
never committed: experiments/data/external/ is gitignored. Only this manifest and the
derived per-item predictions and aggregate metrics enter the repository.

Cite: Shen et al., "FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of
Chain-of-Thought Reasoning", arXiv:2510.04040.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys

# The release is a GitHub repository, not a Hugging Face dataset. The Hub has no
# se7esx/FaithCoT-BENCH repo (api/datasets returns 404 with a valid token, and a
# search for "FaithCoT" returns nothing), so the revision of record is a git commit.
DATASET_ID = "github.com/se7esx/FaithCoT-BENCH"
DATASET_REVISION = "6e3c004cbbde5bf47352df91e3ac399d2fb4593e"
DATASET_REVISION_DATE = "2026-07-27T04:12:06Z"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/se7esx/FaithCoT-BENCH/"
    f"{DATASET_REVISION}/faithcot.zip"
)
PAPER = "arXiv:2510.04040"

# Verbatim, as shipped. The repository carries no LICENSE file and the GitHub API
# reports license: null, which is exactly why written permission was required.
LICENCE_AS_SHIPPED = (
    "NONE. The repository at revision "
    f"{DATASET_REVISION} ships no LICENSE, LICENCE, COPYING or licence section: the "
    "recursive git tree lists 32 blobs and none of them is a licence file, and the "
    "GitHub repository API returns \"license\": null. Use therefore rests entirely on "
    "the written permission recorded in PERMISSIONS.md, not on a public licence."
)

PERMISSION_QUOTE = (
    "You are welcome to use the released data for the evaluation purposes described "
    "in your email. Please cite our paper when reporting the results."
)
PERMISSION_SOURCE = (
    "Corresponding author (Xu Shen), by email to the operator, 15:41 on 2026-09-07; "
    "recorded verbatim in ~/Developer/bayes-cot-phase2/PERMISSIONS.md"
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "data" / "external" / "faithcot"
OUT_DIR = ROOT / "experiments" / "external"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    archive = DATA_DIR / "faithcot.zip"
    if not archive.exists():
        raise SystemExit(f"archive missing: {archive}")

    files = sorted((DATA_DIR / "faithcot").rglob("*.json"))
    if not files:
        raise SystemExit("no extracted item files found")

    per_file = []
    counts: collections.Counter = collections.Counter()
    types: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    unfaith: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in files:
        rel = path.relative_to(DATA_DIR).as_posix()
        digest = sha256_file(path)
        per_file.append({"path": rel, "sha256": digest, "bytes": path.stat().st_size})
        parts = rel.split("/")
        split = f"{parts[1]}/{parts[2]}"
        counts[split] += 1
        record = json.loads(path.read_text())
        types[split][str(record.get("faithful_type", "unannotated"))] += 1
        unfaith[split][str(record.get("unfaithfulness", "unannotated"))] += 1

    # One digest that pins the whole extracted corpus, so a later run can prove it
    # read the same bytes without shipping 1,364 hashes into the prose.
    corpus_lines = "".join(f"{e['path']} {e['sha256']}\n" for e in per_file)
    corpus_digest = hashlib.sha256(corpus_lines.encode()).hexdigest()

    total_types: collections.Counter = collections.Counter()
    total_unfaith: collections.Counter = collections.Counter()
    for split, split_types in types.items():
        total_types.update(split_types)
        total_unfaith.update(unfaith[split])

    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_revision_date": DATASET_REVISION_DATE,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "paper": PAPER,
        "licence_as_shipped": LICENCE_AS_SHIPPED,
        "permission_quote": PERMISSION_QUOTE,
        "permission_source": PERMISSION_SOURCE,
        "permission_scope": (
            "Evaluation only. The data is never redistributed: the download path "
            "experiments/data/external/ is gitignored and only this manifest, derived "
            "per-item predictions and aggregate metrics are committed."
        ),
        "n_item_files": len(per_file),
        "corpus_sha256": corpus_digest,
        "counts_per_split": dict(sorted(counts.items())),
        "faithful_type_per_split": {k: dict(sorted(v.items())) for k, v in sorted(types.items())},
        "faithful_type_total": dict(sorted(total_types.items())),
        "unfaithfulness_total": dict(sorted(total_unfaith.items())),
        "per_file": per_file,
    }


def main() -> int:
    manifest = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_file = manifest.pop("per_file")
    (OUT_DIR / "faithcot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT_DIR / "faithcot_file_hashes.txt").write_text(
        "".join(f"{e['sha256']}  {e['path']}\n" for e in per_file)
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_file"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
