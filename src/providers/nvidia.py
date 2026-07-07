"""NVIDIA hosted API provider (build.nvidia.com / integrate.api.nvidia.com).

OpenAI-compatible chat backend used as a strong-model rung for Planner and
Reviewer escalation. Key resolution: ``NVIDIA_API_KEY`` env first, falling
back to the key already configured in OpenCode's ``opencode.jsonc`` so the
key lives in exactly one place.

Mid-task model failover: the provider holds an *ordered ladder* of live
models. A per-request rate limit (429) or a dead/EOL model (404/410) only
sidelines *that* model — the same ``complete()`` call transparently retries
the next model in the ladder, so a task that gets rate-limited "in the middle
of doing something" keeps going on a different model instead of failing. Only
when the whole ladder is exhausted does the provider go on a persisted
cooldown. Account-level errors (401/403 auth, 402 credits) short-circuit
straight to a provider cooldown, since no other model would fare better.
"""

import os
import re
import time
from typing import Dict, List, Optional, Set

import httpx

from src.providers.base import Provider, Usage, log_usage

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Ordered failover ladder — strongest first, all verified live 2026-07-06 via
# /v1/models. Override the whole ladder with ROUTER_NVIDIA_MODELS (comma-sep),
# or just the primary with ROUTER_NVIDIA_MODEL.
DEFAULT_MODELS: List[str] = [
    "qwen/qwen3.5-397b-a17b",
    "deepseek-ai/deepseek-v4-pro",
    "qwen/qwen3.5-122b-a10b",
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.3-70b-instruct",
]

# How long to sideline a single model after a 429 before it re-enters the
# ladder (unless the response carried a longer retry-after).
_MODEL_COOLDOWN_S = 5 * 60

_OPENCODE_JSONC = os.path.expanduser("~/.config/opencode/opencode.jsonc")

# NVIDIA keys look like nvapi-<base64ish>. A targeted regex beats full JSONC
# parsing (naive comment-stripping corrupts "https://..." strings).
_NVAPI_RE = re.compile(r'"apiKey"\s*:\s*"(nvapi-[A-Za-z0-9_\-]+)"')


def resolve_api_key() -> Optional[str]:
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    try:
        with open(_OPENCODE_JSONC, "r", encoding="utf-8") as f:
            m = _NVAPI_RE.search(f.read())
        return m.group(1) if m else None
    except OSError:
        return None


def _resolve_models() -> List[str]:
    env = os.environ.get("ROUTER_NVIDIA_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    single = os.environ.get("ROUTER_NVIDIA_MODEL")
    if single:
        rest = [m for m in DEFAULT_MODELS if m != single]
        return [single] + rest
    return list(DEFAULT_MODELS)


class NvidiaProvider(Provider):
    name = "nvidia"
    shape = "chat"
    _COOLDOWN_TTL_DEFAULT = 60 * 60

    def __init__(
        self,
        models: Optional[List[str]] = None,
        timeout_s: float = 300.0,
    ):
        self.models: List[str] = models or _resolve_models()
        # Back-compat alias — the current primary model.
        self.model = self.models[0]
        self.timeout_s = timeout_s
        self._key: Optional[str] = None
        # model_id -> epoch until which it is rate-limited (429)
        self._model_until: Dict[str, float] = {}
        # models that returned 404/410 — permanently skipped this process
        self._dead: Set[str] = set()

    @property
    def api_key(self) -> Optional[str]:
        if self._key is None:
            self._key = resolve_api_key()
        return self._key

    def _probe(self) -> tuple:
        if not self.api_key:
            return False, "no NVIDIA_API_KEY (env or opencode.jsonc)"
        if not self._usable_models():
            return False, "all NVIDIA models cooling down"
        return True, "ok"

    def _usable_models(self, override: Optional[str] = None) -> List[str]:
        """Ladder of models that are neither dead nor currently rate-limited.
        If ``override`` is given (a caller pinned a specific model), only that
        model is considered."""
        candidates = [override] if override else self.models
        now = time.time()
        return [
            m
            for m in candidates
            if m not in self._dead and now >= self._model_until.get(m, 0.0)
        ]

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        ladder = self._usable_models(override=model)
        if not ladder:
            # Nothing usable right now — cool the whole provider so the router
            # skips us until at least one model is back.
            soonest = min(
                (t for t in self._model_until.values() if t > time.time()),
                default=time.time() + self._MODEL_COOLDOWN_S,
            )
            self.start_cooldown(
                max(60.0, soonest - time.time()), reason="all models rate-limited"
            )
            raise RuntimeError("NVIDIA: no usable model (all rate-limited/dead)")

        last_err: Optional[str] = None
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for candidate in ladder:
                payload = {
                    "model": candidate,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 8192,  # required: some NIM models omit content without it
                    "stream": False,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                started = time.time()
                try:
                    resp = await client.post(
                        f"{NVIDIA_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                except httpx.HTTPError as e:
                    # Network hiccup — try the next model rather than dying.
                    latency = time.time() - started
                    last_err = f"{candidate}: {type(e).__name__}"
                    log_usage(Usage(self.name, "complete", False, latency, last_err))
                    continue
                latency = time.time() - started

                # Account-level failures: another model won't help.
                if resp.status_code in (401, 403, 402):
                    self.start_cooldown(reason=f"HTTP {resp.status_code} (account)")
                    log_usage(
                        Usage(self.name, "complete", False, latency, f"{resp.status_code}")
                    )
                    raise RuntimeError(
                        f"NVIDIA account gated ({resp.status_code}); provider cooling down"
                    )

                # Rate limit: sideline THIS model, fail over to the next.
                if resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after")
                    secs = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else _MODEL_COOLDOWN_S
                    )
                    self._model_until[candidate] = time.time() + secs
                    last_err = f"{candidate}: 429 (cooling {int(secs)}s)"
                    log_usage(Usage(self.name, "complete", False, latency, last_err))
                    continue

                # Dead / EOL model: skip permanently, fail over.
                if resp.status_code in (404, 410):
                    self._dead.add(candidate)
                    last_err = f"{candidate}: {resp.status_code} EOL"
                    log_usage(Usage(self.name, "complete", False, latency, last_err))
                    continue

                if resp.status_code != 200:
                    last_err = f"{candidate}: {resp.status_code} {resp.text[:200]}"
                    log_usage(
                        Usage(self.name, "complete", False, latency, f"{resp.status_code}")
                    )
                    continue

                msg = resp.json()["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning_content") or ""
                log_usage(Usage(self.name, "complete", True, latency, detail=candidate))
                return content

        # Fell off the ladder — every candidate errored transiently. Cool the
        # provider briefly and report the last failure.
        self.start_cooldown(120.0, reason="all models errored")
        raise RuntimeError(f"NVIDIA: all models failed; last={last_err}")
