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
PERMISSIONS_ROW = "SoCLaaS eligibility for jury votes"

# Preference order per judge family. Resolved against the gateway's live /models listing
# rather than assumed, because the tag set moves and a hardcoded id that no longer exists
# turns a fallback into a second failure.
SOCLAAS_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "Qwen": ("qwen",),
    "Gemma": ("gemma",),
    "Llama": ("llama",),
    "gpt-oss": ("gpt-oss", "gptoss"),
}


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
    for prefix in prefixes:
        for model_id in sorted(ids):
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

    def for_seed(self, seed: int) -> "JudgeEndpoint":
        """A copy of this endpoint bound to one seed, safe to use from one thread."""
        client = copy.copy(self.client)
        client.seed = seed
        return JudgeEndpoint(
            judge_key=self.judge_key, backend=self.backend, model=self.model,
            revision=self.revision, client=client, fallback=self.fallback,
            fallback_of=self.fallback_of,
        )

    def generate(self, prompt: str, *, num_predict: int) -> str:
        return self.client.generate(prompt, num_predict=num_predict)

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
