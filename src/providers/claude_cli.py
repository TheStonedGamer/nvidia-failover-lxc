"""Claude Code CLI provider — headless agentic subagent on the subscription.

Shells out to ``claude -p`` (non-interactive) inside the agent workspace so
Claude does its own file reads/edits/tool runs and returns a structured
result. Usage gating: a rate-limit/usage-limit response puts the provider on
a persisted cooldown until the reported reset (or a default TTL), so the
escalation router skips it instead of burning attempts.
"""

import asyncio
import json
import re
import shutil
import time
from typing import Any, Dict, Optional

from src.providers.base import Provider, Usage, log_usage

# Signals in CLI output that mean "out of usage / rate limited".
_LIMIT_RE = re.compile(
    r"rate.?limit|usage limit|limit reached|out of (usage|credits)|overloaded",
    re.IGNORECASE,
)
# e.g. "resets at 3pm" style hints are unreliable; default cooldown is used
# unless a unix reset timestamp is present.
_RESET_TS_RE = re.compile(r"reset[^0-9]*(\d{10})")


class ClaudeCLIProvider(Provider):
    name = "claude_cli"
    shape = "agentic"
    _COOLDOWN_TTL_DEFAULT = 30 * 60

    def __init__(self, model: str = "sonnet", max_turns: int = 25, timeout_s: int = 900):
        self.model = model
        self.max_turns = max_turns
        self.timeout_s = timeout_s

    def _probe(self) -> tuple:
        if not shutil.which("claude"):
            return False, "claude CLI not on PATH"
        return True, "ok"

    async def _run_cli(self, prompt: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Run claude -p and return the parsed JSON result envelope."""
        args = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--max-turns",
            str(self.max_turns),
            "--permission-mode",
            "acceptEdits",
        ]
        started = time.time()
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            log_usage(Usage(self.name, "run_task", False, time.time() - started, "timeout"))
            return {"ok": False, "error": f"claude CLI timed out after {self.timeout_s}s"}

        stdout = out.decode("utf-8", errors="replace")
        stderr = err.decode("utf-8", errors="replace")
        latency = time.time() - started

        # The CLI emits a JSON result envelope on stdout even when it exits
        # nonzero (errors arrive as {"is_error": true, "result": "<message>"}),
        # so parse stdout FIRST and only fall back to stderr.
        payload = None
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            pass

        if isinstance(payload, dict) and payload.get("is_error"):
            msg = str(payload.get("result", ""))
            status = payload.get("api_error_status")
            if status in (401, 403) or "disabled" in msg.lower():
                # Auth/entitlement problem (e.g. subscription access to headless
                # Claude Code disabled). Long cooldown — retrying won't help.
                self.start_cooldown(24 * 3600, reason=f"{status}: {msg[:200]}")
            elif _LIMIT_RE.search(msg):
                m = _RESET_TS_RE.search(msg)
                secs = max(60.0, float(m.group(1)) - time.time()) if m else None
                self.start_cooldown(secs, reason="usage limit reported by CLI")
            log_usage(Usage(self.name, "run_task", False, latency, msg[:200]))
            return {"ok": False, "error": f"claude CLI error: {msg[:300]}"}

        combined = stdout + "\n" + stderr
        if _LIMIT_RE.search(combined):
            m = _RESET_TS_RE.search(combined)
            secs = max(60.0, float(m.group(1)) - time.time()) if m else None
            self.start_cooldown(secs, reason="usage limit reported by CLI")
            log_usage(Usage(self.name, "run_task", False, latency, "rate-limited"))
            return {"ok": False, "error": "claude CLI usage limit; provider cooling down"}

        if proc.returncode != 0:
            log_usage(
                Usage(self.name, "run_task", False, latency, f"exit {proc.returncode}")
            )
            return {"ok": False, "error": f"claude exited {proc.returncode}: {stderr[:500]}"}

        if payload is None:
            # Some CLI versions emit plain text despite --output-format json.
            payload = {"result": stdout}

        log_usage(Usage(self.name, "run_task", True, latency))
        return {"ok": True, "payload": payload}

    # -- Provider interface ---------------------------------------------------

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """Single-shot completion via the CLI (no workspace tools needed)."""
        prompt = f"{system_prompt}\n\n{user_prompt}"
        if json_mode:
            prompt += "\n\nRespond with raw JSON only — no markdown fences."
        res = await self._run_cli(prompt)
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "claude CLI failed"))
        payload = res["payload"]
        return payload.get("result", "") if isinstance(payload, dict) else str(payload)

    async def run_task(
        self, description: str, workspace: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Hand a failing task to Claude as a full agentic subagent."""
        ctx = ""
        if context:
            ctx = (
                "\n\nContext from previous local attempts (errors/diagnostics):\n"
                + json.dumps(context)[:4000]
            )
        prompt = (
            "You are a subagent completing a task another (smaller) agent failed. "
            "Work directly in this directory: read, edit, and verify files as "
            f"needed, then summarize what you changed.\n\nTask: {description}{ctx}"
        )
        res = await self._run_cli(prompt, cwd=workspace)
        if not res.get("ok"):
            return {"status": "error", "message": res.get("error", "unknown failure")}
        payload = res["payload"]
        summary = payload.get("result", "") if isinstance(payload, dict) else str(payload)
        return {"status": "success", "summary": summary, "raw": payload}
