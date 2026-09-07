"""The 18-model roster, its family map, the four judges, and the panel routing rule.

Source of truth, copied field by field and never re-derived from memory:

* Family map: CONTRACT.md line 19 ("Family map (base architecture; distills belong to
  their base)").
* Calibrated strata: CONTRACT.md line 20.
* Panel rule and the five compositions with their subject counts: CONTRACT.md line 21 and
  PREREGISTRATION_jury_and_scale.md section 6.2.
* Judges, HF ids and pinned revisions: PREREGISTRATION_jury_and_scale.md section 6.1.

The routing rule is deterministic: every judge NOT of the subject model's family votes.
That gives a panel of 3 for the 12 subjects whose family supplies a judge and a panel of 4
for the 6 that have none. Because composition is a function of family alone, a
pre-registered 20 percent of every stratum's calibration rows is additionally scored by all
four judges; the own-family votes on those rows are recorded, excluded from the panel label
and kept for per-judge error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# --- families -------------------------------------------------------------------

QWEN = "Qwen"
LLAMA = "Llama"
GEMMA = "Gemma"
GPT_OSS = "gpt-oss"
OLMO = "OLMo"
MISTRAL = "Mistral"
PHI = "Phi"
GLM = "GLM"


@dataclass(frozen=True)
class RosterEntry:
    """One subject model on the canonical roster."""

    name: str
    family: str
    hf_id: str


# CONTRACT.md line 19, in the order the contract writes them. The hf_id column carries the
# identifier A2 verified live against the Hugging Face model API on 2026-09-07, including
# the one correction A2 recorded as an addition: the contract writes "OLMo-3-7B-Think" but
# the repository is `allenai/Olmo-3-7B-Think` (the uppercase form returns HTTP 307).
ROSTER: tuple[RosterEntry, ...] = (
    RosterEntry("Qwen3-8B", QWEN, "Qwen/Qwen3-8B"),
    RosterEntry("Qwen3-32B", QWEN, "Qwen/Qwen3-32B"),
    RosterEntry("Qwen3.6-35B-A3B", QWEN, "Qwen/Qwen3.6-35B-A3B"),
    RosterEntry("DeepSeek-R1-Distill-Llama-8B", LLAMA, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    RosterEntry("DeepSeek-R1-Distill-Llama-70B", LLAMA, "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    RosterEntry("DeepSeek-R1-0528-Qwen3-8B", QWEN, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
    RosterEntry("gpt-oss-20b", GPT_OSS, "openai/gpt-oss-20b"),
    RosterEntry("gpt-oss-120b", GPT_OSS, "openai/gpt-oss-120b"),
    RosterEntry("OLMo-3-7B-Think", OLMO, "allenai/Olmo-3-7B-Think"),
    RosterEntry("Olmo-3-32B-Think", OLMO, "allenai/Olmo-3-32B-Think"),
    RosterEntry("Mistral-Small-3.2-24B", MISTRAL, "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    RosterEntry("Phi-4-reasoning", PHI, "microsoft/Phi-4-reasoning"),
    RosterEntry("GLM-4.5-Air", GLM, "zai-org/GLM-4.5-Air"),
    RosterEntry("Llama-3.1-8B-Instruct", LLAMA, "meta-llama/Llama-3.1-8B-Instruct"),
    RosterEntry("Llama-3.3-70B-Instruct", LLAMA, "meta-llama/Llama-3.3-70B-Instruct"),
    RosterEntry("Gemma-2-9B-it", GEMMA, "google/gemma-2-9b-it"),
    RosterEntry("Gemma-3-27B-it", GEMMA, "google/gemma-3-27b-it"),
    RosterEntry("Magistral-Small-2509", MISTRAL, "mistralai/Magistral-Small-2509"),
)

# CONTRACT.md line 20. A verifier recomputes both from the map rather than trusting these.
CALIBRATED_STRATA: tuple[str, ...] = (QWEN, LLAMA, GEMMA, GPT_OSS, OLMO)
UNCALIBRATED_FAMILIES: tuple[str, ...] = (MISTRAL, PHI, GLM)


# --- judges ---------------------------------------------------------------------


@dataclass(frozen=True)
class Judge:
    """One panel member, pinned to an HF revision and a serving line."""

    key: str
    family: str
    hf_id: str
    revision: str
    serving_line: str
    gpu_type: str
    gpu_memory_utilization: float
    quantization: str | None = None
    # vLLM's --max-model-len for this judge. Judge prompts carry a full transcript, so
    # the window has to hold the longest banked reasoning plus the rubric.
    max_model_len: int = 8192


# PREREGISTRATION_jury_and_scale.md section 6.1, table copied verbatim.
JUDGES: tuple[Judge, ...] = (
    Judge(
        key="qwen3-32b",
        family=QWEN,
        hf_id="Qwen/Qwen3-32B",
        revision="9216db5781bf21249d130ec9da846c4624c16137",
        serving_line="bf16, 1 x a100-80",
        gpu_type="a100-80",
        gpu_memory_utilization=0.90,
    ),
    Judge(
        key="gemma-3-27b-it",
        family=GEMMA,
        hf_id="google/gemma-3-27b-it",
        revision="005ad3404e59d6023443cb575daa05336842228a",
        serving_line="bf16, shares one a100-80 at gpu-memory-utilization 0.65",
        gpu_type="a100-80",
        gpu_memory_utilization=0.65,
    ),
    Judge(
        key="gpt-oss-20b",
        family=GPT_OSS,
        hf_id="openai/gpt-oss-20b",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        serving_line="mxfp4, shares the same a100-80 at gpu-memory-utilization 0.25",
        gpu_type="a100-80",
        gpu_memory_utilization=0.25,
    ),
    Judge(
        key="llama-3.3-70b-fp8",
        family=LLAMA,
        hf_id="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        revision="f50dbad2c84590ca17dc51e207c34321b65ff14b",
        serving_line="FP8 dynamic, 1 x h200-141",
        gpu_type="h200-141",
        gpu_memory_utilization=0.90,
    ),
)

JUDGE_BY_KEY: dict[str, Judge] = {j.key: j for j in JUDGES}
JUDGE_FAMILIES: frozenset[str] = frozenset(j.family for j in JUDGES)


class RoutingError(ValueError):
    """Raised when a subject model is not on the roster or a panel is malformed."""


def family_of(subject: str) -> str:
    """Family of a roster subject, by display name or by HF id."""
    for entry in ROSTER:
        if subject in (entry.name, entry.hf_id):
            return entry.family
    raise RoutingError(
        f"{subject!r} is not on the canonical 18-model roster; "
        "the panel rule is defined only for roster subjects"
    )


def routing(subject: str) -> tuple[str, ...]:
    """Judge keys that vote on ``subject``: every judge NOT of the subject's family."""
    fam = family_of(subject)
    return tuple(j.key for j in JUDGES if j.family != fam)


def panel_size(subject: str) -> int:
    return len(routing(subject))


def all_judges() -> tuple[str, ...]:
    """Every judge key, for the 20 percent all-judge subsample rows."""
    return tuple(j.key for j in JUDGES)


def own_family_judges(subject: str) -> tuple[str, ...]:
    """Judges of the subject's own family: recorded on subsample rows, never in the label."""
    fam = family_of(subject)
    return tuple(j.key for j in JUDGES if j.family == fam)


def composition_counts() -> dict[frozenset[str], int]:
    """Map each panel composition that occurs to the number of subjects that draw it."""
    counts: dict[frozenset[str], int] = {}
    for entry in ROSTER:
        comp = frozenset(routing(entry.name))
        counts[comp] = counts.get(comp, 0) + 1
    return counts


def assert_panel(subject: str, panel: list[str] | tuple[str, ...]) -> None:
    """Fail loudly when a vote set does not match the routing rule for ``subject``.

    Called by the runner per vote. A vote from a same-family judge on a non-subsample row
    is the failure this exists to catch, and it raises rather than warns.
    """
    expected = set(routing(subject))
    got = set(panel)
    if got == expected:
        return
    same_family = got & set(own_family_judges(subject))
    detail = f"subject={subject!r} family={family_of(subject)!r} expected={sorted(expected)} got={sorted(got)}"
    if same_family:
        raise RoutingError(
            f"PANEL VIOLATION: same-family judge(s) {sorted(same_family)} in the panel. {detail}"
        )
    raise RoutingError(f"PANEL VIOLATION: {detail}")


# --- the seeded 20 percent all-judge subsample ----------------------------------

ALL_JUDGE_FRACTION = 0.20


def _rank_key(seed: int, stratum: str, item_id: str) -> str:
    """Stable per-row sort key. sha256, not hash(): Python's hash() is salted per process."""
    payload = f"{seed}|{stratum}|{item_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def draw_all_judge_subsample(
    stratum: str,
    item_ids: list[str] | tuple[str, ...],
    *,
    seed: int,
    fraction: float = ALL_JUDGE_FRACTION,
) -> tuple[str, ...]:
    """The pre-registered subsample of one stratum's rows, drawn by a seeded rule.

    Deterministic given (stratum, item_ids as a set, seed): the rows are ranked by a
    sha256 of ``seed|stratum|item_id`` and the first ``round(fraction * n)`` are taken, so
    the count is exact per stratum rather than binomial, the draw is reproducible from the
    seed alone, and two strata with the same item ids draw different rows.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    unique = sorted(set(item_ids))
    if len(unique) != len(item_ids):
        raise ValueError("item_ids contains duplicates; the subsample frame must be a set")
    n_draw = round(fraction * len(unique))
    ranked = sorted(unique, key=lambda i: _rank_key(seed, stratum, i))
    return tuple(sorted(ranked[:n_draw]))


def is_all_judge_row(
    stratum: str,
    item_id: str,
    frame: list[str] | tuple[str, ...],
    *,
    seed: int,
    fraction: float = ALL_JUDGE_FRACTION,
) -> bool:
    """Whether one row carries the all-judge flag, recomputed from the frame and the seed."""
    return item_id in draw_all_judge_subsample(stratum, frame, seed=seed, fraction=fraction)
