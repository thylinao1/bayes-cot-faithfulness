"""Judge backends: the pinned self-hosted vLLM endpoint, and the SoCLaaS fallback.

Ruling of record (DECISION-LOG.md 2026-09-06, after the outage and the re-probe, restated
in PREREGISTRATION_jury_and_scale.md section 6.7): SoCLaaS is ELIGIBLE as a backend for
jury VOTES and INELIGIBLE as a source of logprob outcomes. It is a fallback, never a
default: it activates only when the intended judge's own endpoint cannot be reached, the
key is present in the environment, and the eligibility ruling is on the record. Every
fallback vote is labeled `fallback=true`, never substitutes a judge of the SUBJECT's
family, and always queues a re-run on the intended judge.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai_client import OpenAIClient, OpenAIClientError  # noqa: E402

from .family_map import JUDGE_BY_KEY  # noqa: E402

SOCLAAS_ELIGIBLE_MARK = "ELIGIBLE for jury VOTES"

# Measured on the gateway 2026-09-07, not guessed: qwen3.8:27b returned an empty completion
# at 200 tokens on a gate item and the correct vote at 700.
SOCLAAS_MIN_NUM_PREDICT = 768
PERMISSIONS_ROW = "SoCLaaS eligibility for jury votes"

# Preference order per judge family, resolved against the gateway's live /models listing
# rather than assumed, because the tag set moves and a hardcoded id that no longer exists
# turns a fallback into a second failure.
#
# The order is explicit rather than alphabetical. A bare "qwen" prefix match against the
# live listing on 2026-09-07 picked `qwen3-coder-next`, a code model, as the stand-in for a
# reasoning judge, which is a worse vote than no vote. General chat models first, largest
# first, and the specialist variants excluded outright.
SOCLAAS_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "Qwen": ("qwen3.8:", "qwen3.6:35b", "qwen3.6:", "qwen3.5:", "qwen3:"),
    "Gemma": ("gemma4:", "gemma3:", "gemma2:", "gemma"),
    "Llama": ("llama3.3:", "llama3.1:", "llama3:", "llama"),
    # No gpt-oss model is served on this gateway, so a gpt-oss vote has no fallback and the
    # runner raises instead of substituting a judge from another family.
    "gpt-oss": (),
}

# Specialist variants that share a family prefix but are not general chat judges.
SOCLAAS_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "coder", "-vl", "vl:", "embed", "bge", "whisper", "vision", "rerank",
)


class BackendError(RuntimeError):
    """Raised when no backend can serve a judge and the fallback is not permitted."""


@dataclass(frozen=True)
class Eligibility:
    """Whether SoCLaaS may cast a vote right now, and why."""

    allowed: bool
    reason: str


def soclaas_eligibility(
    *,
    permissions_path: str | Path,
    decision_log_path: str | Path,
    env: dict | None = None,
) -> Eligibility:
    """Three conditions, all required, each checked against a file or the environment.

    1. ``SOCLAAS_API_KEY`` is present in the environment (it is on the Mac, not on the
       cluster). The key itself is never read, printed or logged, only tested for presence.
    2. PERMISSIONS.md carries the SoCLaaS row at all (the operator owns that file).
    3. DECISION-LOG.md carries the eligibility ruling for VOTES.
    """
    environ = os.environ if env is None else env
    if not environ.get("SOCLAAS_API_KEY"):
        return Eligibility(False, "SOCLAAS_API_KEY is not in the environment")
    perms = Path(permissions_path)
    if not perms.exists():
        return Eligibility(False, f"{perms} does not exist")
    perm_text = perms.read_text(encoding="utf-8")
    if PERMISSIONS_ROW not in perm_text:
        return Eligibility(False, f"PERMISSIONS.md carries no {PERMISSIONS_ROW!r} row")
    log = Path(decision_log_path)
    if not log.exists():
        return Eligibility(False, f"{log} does not exist")
    if SOCLAAS_ELIGIBLE_MARK not in log.read_text(encoding="utf-8"):
        return Eligibility(False, f"DECISION-LOG.md carries no {SOCLAAS_ELIGIBLE_MARK!r} ruling")
    return Eligibility(True, "key present, PERMISSIONS row present, DECISION-LOG ruling present")


def resolve_soclaas_model(client: OpenAIClient, judge_family: str) -> str | None:
    """Pick a live SoCLaaS model of the intended judge's family, or None."""
    prefixes = SOCLAAS_FAMILY_PREFIXES.get(judge_family, ())
    if not prefixes:
        return None
    try:
        body = client._get(f"{client.base_url.rstrip('/')}/models")  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return None
    ids = [str(m.get("id", "")) for m in (body.get("data") or [])]
    usable = [
        m for m in ids
        if not any(bad in m.lower() for bad in SOCLAAS_EXCLUDE_SUBSTRINGS)
    ]
    for prefix in prefixes:
        for model_id in sorted(usable):
            if model_id.lower().startswith(prefix):
                return model_id
    return None


@dataclass
class JudgeEndpoint:
    """One judge, bound to a backend. The runner asks this for text and nothing else."""

    judge_key: str
    backend: str  # "vllm" | "soclaas"
    model: str
    revision: str
    client: OpenAIClient
    fallback: bool = False
    fallback_of: str | None = None
    # A floor on the generation budget for this backend. The SoCLaaS chat models spend
    # tokens on a thinking channel before the JSON: measured 2026-09-07, qwen3.8:27b
    # returned an EMPTY completion at num_predict 200 on a gate item and the correct vote
    # at 700. Without a floor every such call burns its one retry and lands on `malformed`,
    # which would show up as a backend-specific unavailable rate and be read as a judge
    # property. The floor is recorded rather than the default quietly raised for everyone.
    min_num_predict: int = 0

    def for_seed(self, seed: int) -> "JudgeEndpoint":
        """A copy of this endpoint bound to one seed, safe to use from one thread."""
        client = copy.copy(self.client)
        client.seed = seed
        return JudgeEndpoint(
            judge_key=self.judge_key, backend=self.backend, model=self.model,
            revision=self.revision, client=client, fallback=self.fallback,
            fallback_of=self.fallback_of, min_num_predict=self.min_num_predict,
        )

    def generate(self, prompt: str, *, num_predict: int) -> str:
        return self.client.generate(prompt, num_predict=max(num_predict, self.min_num_predict))

    def is_available(self) -> bool:
        return self.client.is_available()


def vllm_endpoint(judge_key: str, base_url: str, *, seed: int, timeout: float = 600.0) -> JudgeEndpoint:
    """The pinned self-hosted endpoint for one judge, at temperature 0."""
    judge = JUDGE_BY_KEY[judge_key]
    client = OpenAIClient(
        base_url=base_url,
        model=judge.hf_id,
        temperature=0.0,
        n=1,
        seed=seed,
        timeout=timeout,
    )
    return JudgeEndpoint(
        judge_key=judge_key,
        backend="vllm",
        model=judge.hf_id,
        revision=judge.revision,
        client=client,
    )


def soclaas_endpoint(
    judge_key: str,
    *,
    seed: int,
    subject_family: str,
    env: dict | None = None,
    timeout: float = 120.0,
) -> JudgeEndpoint:
    """A fallback endpoint for one judge on SoCLaaS.

    Refuses when the resolved model's family is the SUBJECT's family: the family exclusion
    is the panel rule and a fallback may not quietly break it.
    """
    environ = os.environ if env is None else env
    judge = JUDGE_BY_KEY[judge_key]
    if judge.family == subject_family:
        raise BackendError(
            f"fallback refused: judge {judge_key} is of the subject family {subject_family}"
        )
    base = environ.get("SOCLAAS_BASE_URL")
    key = environ.get("SOCLAAS_API_KEY")
    if not base or not key:
        raise BackendError("SOCLAAS_BASE_URL or SOCLAAS_API_KEY missing from the environment")
    client = OpenAIClient(
        base_url=base, model="", api_key=key, temperature=0.0, n=1, seed=seed, timeout=timeout
    )
    model_id = resolve_soclaas_model(client, judge.family)
    if not model_id:
        raise BackendError(f"no live SoCLaaS model of family {judge.family} for {judge_key}")
    client.model = model_id
    return JudgeEndpoint(
        judge_key=judge_key,
        backend="soclaas",
        model=model_id,
        # The gateway pins no revision, and inventing one would put a false pin on the
        # record. The fallback flag plus the queued re-run is what carries the caveat.
        revision="unpinned:soclaas",
        client=client,
        fallback=True,
        fallback_of=judge.hf_id,
        min_num_predict=SOCLAAS_MIN_NUM_PREDICT,
    )


__all__ = [
    "BackendError",
    "Eligibility",
    "JudgeEndpoint",
    "OpenAIClientError",
    "resolve_soclaas_model",
    "soclaas_eligibility",
    "soclaas_endpoint",
    "vllm_endpoint",
]
