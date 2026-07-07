"""ProviderRouter — the escalation ladder (Phase 2.8).

Policy: local-first; external tiers are touched only after local has failed
`local_attempts(difficulty)` times, or on explicit request (@claude/@nvidia
or force_tier). Tier order among externals is cost-ordered by default
(claude_cli on subscription before metered nvidia) but flips for chat-shaped
roles where NVIDIA is the natural fit.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from src import difficulty
from src.providers.base import Provider
from src.providers.claude_cli import ClaudeCLIProvider
from src.providers.local import LocalProvider
from src.providers.nvidia import NvidiaProvider

_FORCE_RE = re.compile(r"@(claude|nvidia|local)\b", re.IGNORECASE)

_AVAIL_TTL = 60.0  # seconds to cache is_available() probes


class ProviderRouter:
    def __init__(
        self,
        local: Optional[LocalProvider] = None,
        claude: Optional[ClaudeCLIProvider] = None,
        nvidia: Optional[NvidiaProvider] = None,
    ):
        self.providers: Dict[str, Provider] = {
            "local": local or LocalProvider(),
            "claude_cli": claude or ClaudeCLIProvider(),
            "nvidia": nvidia or NvidiaProvider(),
        }
        self._avail_cache: Dict[str, Tuple[float, bool, str]] = {}
        # attempts[task_id] = number of local failures so far
        self.attempts: Dict[str, int] = {}

    # -- availability (cached) ------------------------------------------------

    def available(self, name: str) -> Tuple[bool, str]:
        cached = self._avail_cache.get(name)
        now = time.time()
        if cached and now - cached[0] < _AVAIL_TTL:
            return cached[1], cached[2]
        ok, reason = self.providers[name].is_available()
        self._avail_cache[name] = (now, ok, reason)
        return ok, reason

    def invalidate(self, name: str) -> None:
        self._avail_cache.pop(name, None)

    # -- escalation policy ----------------------------------------------------

    @staticmethod
    def parse_force(description: str) -> Optional[str]:
        """@claude / @nvidia / @local tag in the request forces a tier."""
        m = _FORCE_RE.search(description)
        if not m:
            return None
        tag = m.group(1).lower()
        return {"claude": "claude_cli", "nvidia": "nvidia", "local": "local"}[tag]

    def external_order(self, role: str) -> List[str]:
        """Tier order once escalating. Builder work wants the agentic Claude
        subagent (it actually edits files); planner/reviewer chat work suits
        NVIDIA's strong chat models first."""
        if role == "builder":
            return ["claude_cli", "nvidia"]
        return ["nvidia", "claude_cli"]

    def pick(
        self,
        task_id: str,
        description: str,
        role: str = "builder",
        dag: Optional[dict] = None,
        force_tier: Optional[str] = None,
    ) -> Tuple[Provider, str]:
        """Choose the provider for this attempt. Returns (provider, reason)."""
        forced = force_tier or self.parse_force(description)
        if forced:
            ok, why = self.available(forced)
            if ok:
                return self.providers[forced], f"explicitly requested {forced}"
            # fall through to normal policy if the forced tier is gated
            print(f"[Router] forced tier '{forced}' unavailable ({why}); using policy")

        fails = self.attempts.get(task_id, 0)
        budget = difficulty.local_attempts(difficulty.score(description, dag))

        if fails < budget:
            ok, why = self.available("local")
            if ok:
                return self.providers["local"], (
                    f"local attempt {fails + 1}/{budget}"
                )
            print(f"[Router] local unavailable ({why}); escalating early")

        for name in self.external_order(role):
            ok, why = self.available(name)
            if ok:
                return self.providers[name], (
                    f"escalated after {fails} local failure(s) -> {name}"
                )
            print(f"[Router] tier '{name}' unavailable: {why}")

        # Everything gated: hand back local as last resort (it may still work).
        return self.providers["local"], "all external tiers gated; retrying local"

    def record_failure(self, task_id: str) -> int:
        self.attempts[task_id] = self.attempts.get(task_id, 0) + 1
        return self.attempts[task_id]

    def record_success(self, task_id: str) -> None:
        self.attempts.pop(task_id, None)

    def status(self) -> List[dict]:
        out = []
        for name, p in self.providers.items():
            ok, reason = self.available(name)
            out.append(
                {"name": name, "shape": p.shape, "available": ok, "reason": reason}
            )
        return out


# Module-level singleton, mirroring the other backends.
router = ProviderRouter()
