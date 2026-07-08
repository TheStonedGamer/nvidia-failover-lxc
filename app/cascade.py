"""Cascade: builds the effective failover order for a request (special-mode
routing, sticky cursor, cooldowns) and tracks per-model cooldown/dead state."""

import time
from typing import Dict, List, Optional, Set

from app.config import (
    AUTO_MODEL,
    ONLY_MODEL,
    REFINER_MODEL_ID,
    LOCAL_ONLY,
    LOCAL_REFINE,
    AGENT_ROLES,
    LOCAL_ENABLE,
    LOCAL_BASE_URL,
    LOCAL_MODEL,
    _MODEL_COOLDOWN_S,
    ladder as config_ladder,
    strip_v1,
)
from app.ladder import ladder_config
from app.stats import Stats


class Cascade:
    def __init__(self, stats: Stats):
        self.models: List[str] = config_ladder()
        self.model_until: Dict[str, float] = {}
        self.dead: Set[str] = set()
        self.stats = stats

    def _local_model(self) -> Optional[str]:
        if ladder_config.local_tail:
            return ladder_config.local_tail
        for name, p in ladder_config.providers.items():
            if name == "ollama" and p.get("models"):
                return p["models"][0]
        return LOCAL_MODEL or None

    def _ollama_base_url(self) -> str:
        ollama = ladder_config.providers.get("ollama")
        if ollama and ollama.get("base_url"):
            return ollama["base_url"]
        return LOCAL_BASE_URL

    def is_local(self, model: Optional[str]) -> bool:
        return bool(model) and model == self._local_model()

    def _rotate_to_cursor(self, base: List[str]) -> List[str]:
        cursor = self.stats.sticky_model()
        if not cursor or cursor not in base:
            return base
        idx = base.index(cursor)
        return base[idx:] + base[:idx]

    def _serving_ladder(self) -> List[str]:
        return ladder_config.active_ladder() or list(self.models)

    def _live_now(self, now: float) -> List[str]:
        base = self._serving_ladder()
        return [m for m in base if m not in self.dead and self.model_until.get(m, 0) <= now]

    def order(self, preferred: Optional[str] = None) -> List[str]:
        now = time.time()
        base = self._serving_ladder()
        local_model = self._local_model()

        if preferred and preferred not in (AUTO_MODEL, "") and preferred not in AGENT_ROLES:
            if preferred in (ONLY_MODEL,):
                pass
            elif preferred in (LOCAL_ONLY, LOCAL_REFINE):
                return [local_model] if (LOCAL_ENABLE and local_model) else []
            elif preferred == REFINER_MODEL_ID:
                pass
            elif preferred in base:
                live = [preferred] + [m for m in self._rotate_to_cursor(base) if m != preferred]
                live = [m for m in live if m not in self.dead]
                if LOCAL_ENABLE and local_model:
                    live.append(local_model)
                return live

        cloud_only = preferred == ONLY_MODEL
        live = self._live_now(now)
        live = self._rotate_to_cursor(live)

        if not live:
            if cloud_only:
                return []
            if LOCAL_ENABLE and local_model:
                return [local_model]
            # Nothing live, no local tail: fall back to whichever cloud model
            # is closest to reviving, so the caller still gets a real attempt
            # (and a real error) instead of a hard-coded empty ladder.
            candidates = [m for m in base if m not in self.dead]
            if not candidates:
                return []
            candidates.sort(key=lambda m: self.model_until.get(m, 0))
            return [candidates[0]]

        if cloud_only:
            return live
        if LOCAL_ENABLE and local_model and local_model not in live:
            live = live + [local_model]
        return live

    def soonest_cooldown(self) -> Optional[float]:
        if not self.model_until:
            return None
        now = time.time()
        soonest = min(self.model_until.values())
        return max(0.0, soonest - now)

    def cool(self, model: str, secs: float) -> None:
        self.model_until[model] = max(self.model_until.get(model, 0.0), time.time() + secs)

    def reset_cooldowns(self) -> int:
        now = time.time()
        cleared = len([t for t in self.model_until.values() if t > now]) + len(self.dead)
        self.model_until.clear()
        self.dead.clear()
        return cleared

    def note_status(self, model: str, status: int, retry_after: Optional[float] = None) -> None:
        if status == 429:
            self.stats.record_429(model, retry_after)
            secs = retry_after or self.stats.learned_cooldown(model) or _MODEL_COOLDOWN_S
            self.cool(model, secs)
        elif status in (404, 410):
            self.dead.add(model)
        else:
            self.stats.record_error(model)
