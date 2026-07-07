"""Provider abstraction for the escalation ladder (Phase 2.8).

A Provider is one rung of the ladder: local llama.cpp, the headless Claude
Code CLI subagent, or the NVIDIA hosted API. Providers come in two *shapes*:

- ``chat``    — an OpenAI-style completion backend our own ReAct loop drives.
- ``agentic`` — a self-driving subagent (Claude CLI) that is handed a task and
                a workspace and does its own tool use.

Availability is usage-gated: a provider that is rate-limited or out of
credits reports unavailable (with a cooldown) instead of erroring mid-task.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Where cooldowns are persisted so restarts respect a provider's reset window.
STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "state",
)
USAGE_LOG = os.path.join(STATE_DIR, "provider_usage.jsonl")


@dataclass
class Usage:
    """One external (or notable local) call, for the usage log / Phase 12."""

    provider: str
    kind: str  # "complete" | "run_task"
    ok: bool
    latency_s: float
    detail: str = ""
    ts: float = field(default_factory=time.time)


def log_usage(usage: Usage) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(usage.__dict__) + "\n")
    except OSError:
        pass  # usage logging must never break the loop


class Provider(ABC):
    """One rung of the escalation ladder."""

    name: str = "base"
    shape: str = "chat"  # "chat" | "agentic"

    # -- availability / usage gating ----------------------------------------

    _COOLDOWN_TTL_DEFAULT = 15 * 60  # seconds, if the provider gives no reset

    def _cooldown_file(self) -> str:
        return os.path.join(STATE_DIR, f"cooldown_{self.name}.json")

    def start_cooldown(self, seconds: Optional[float] = None, reason: str = "") -> None:
        """Mark this provider unusable until now+seconds (persisted)."""
        until = time.time() + (seconds or self._COOLDOWN_TTL_DEFAULT)
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(self._cooldown_file(), "w", encoding="utf-8") as f:
                json.dump({"until": until, "reason": reason}, f)
        except OSError:
            pass

    def cooldown_remaining(self) -> float:
        """Seconds left on any active cooldown (0 when clear)."""
        try:
            with open(self._cooldown_file(), "r", encoding="utf-8") as f:
                data = json.load(f)
            return max(0.0, float(data.get("until", 0)) - time.time())
        except (OSError, json.JSONDecodeError, ValueError):
            return 0.0

    def is_available(self) -> tuple:
        """(available, reason). Checks cooldown first, then provider probe."""
        remaining = self.cooldown_remaining()
        if remaining > 0:
            return False, f"cooling down {int(remaining)}s"
        return self._probe()

    def _probe(self) -> tuple:
        """Provider-specific availability check. Default: available."""
        return True, "ok"

    # -- work ----------------------------------------------------------------

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """Chat completion (both shapes must implement; agentic may adapt)."""

    async def run_task(
        self, description: str, workspace: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Agentic task execution. Default: not supported for chat shape."""
        raise NotImplementedError(f"{self.name} is not an agentic provider")
