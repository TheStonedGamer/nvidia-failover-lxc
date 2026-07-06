"""NVIDIA frontier-model failover proxy with SSE dashboard.

- Cascades through NVIDIA's frontier models on 429/timeout
- Falls back to local Ollama model when cloud tier is rate-limited
- Server-sent events dashboard (/updates)
- Agent profiles, prompt refiner, /pick override, /commands

Multiple concurrent requests supported via:
- Async httpx.AsyncClient for NVIDIA + Ollama with connection pooling
- Stats snapshot rotation, atomic stats persistence
- Configurable max concurrency (PROXY_CONCURRENCY)

Like:
```bash
curl -d '{"model":"nvidia-auto","messages":[{"role":"user","content":"/models"}]}' \
  http://<container>:5002/v1/chat/completions
```
"""

import asyncio
import collections
import contextlib
import json
import os
import time
from typing import Deque, Dict, List, Optional, Set

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing_extensions import Literal

from src.providers.nvidia import resolve_api_key, _ladder

# --- Constants ---------------------------------------------------------------------------------
AUTO_MODEL = "nvidia-auto"       # cloud frontier ladder → tail local 80B
ONLY_MODEL = "nvidia-only"       # cloud frontier ladder ONLY; 429 when all cooling (no fallback)
REFINER_MODEL_ID = "nvidia-refine" # cloud ladder + local Qwen 4b prompt refiner
LOCAL_ONLY = "local-only"       # local 80B only (no cloud)
LOCAL_REFINE = "local-refine"    # refiner + local 80B

# Agent profiles — inject system prompt
AGENT_ROLES = {
    "agent-planner": "You are a STRICT TASK PLANNER. Break requests into verifiable test steps. Never write code.",
    "agent-builder": "You are a hands-on CODE EXECUTOR. Call the tool immediately; describe nothing first.",
    "agent-reviewer": "You are a STRICT CODE AUDITOR. Reject unverified changes; LSP errors are blocking.",
}
SPECIAL_IDS = {AUTO_MODEL, ONLY_MODEL, REFINER_MODEL_ID, LOCAL_ONLY, LOCAL_REFINE} | set(AGENT_ROLES)

# --- Rate tracking -----------------------------------------------------------------------------
class RateTracker:
    """Thread-safe revolving usage window. No locks needed — deque appends are GIL-protected."""
    __slots__ = ["window_s", "in_toks", "out_toks", "timestamps"]

    def __init__(self, window_secs=60):
        self.window_s = window_secs
        self.in_toks: Deque[int] = collections.deque(maxlen=window_secs)
        self.out_toks: Deque[int] = collections.deque(maxlen=window_secs)
        self.timestamps: Deque[float] = collections.deque(maxlen=window_secs)

    @contextlib.contextmanager
    def tick(self):
        """Time a block of work."""
        start_at = time.time()
        yield
        now = time.time()
        elapsed = int((now - start_at - 1) // 1) + 1
        self.timestamps.extend([now] * elapsed)

    def record(self, input_toks: int, output_toks: int) -> None:
        """Accumulate tokens into the rolling window."""
        assert input_toks >= 0 and output_toks >= 0, "Negative tokens"
        self.in_toks.append(input_toks)        # GIL-protected
        self.out_toks.append(output_toks)

    def _sum_recent(self, metric_deque: Deque[int], now_ts: Optional[float] = None) -> int:
        """Sum tokens that fall into the active window."""
        now = now_ts or time.time()
        return sum(t for (t, v) in zip(self.timestamps, metric_deque) if now - 1 <= t <= now)

    def rpm(self) -> int:
        """Requests per minute."""
        return len(self.timestamps)   # deque length equals last minute

    def tpm_in(self) -> int:
        """Token throughput input/min."""
        return self._sum_recent(self.in_toks)

    def tpm_out(self) -> int:
        """Token throughput output/min."""
        return self._sum_recent(self.out_toks)


class Stats:
    """Stats persistence with atomic save."""
    filename = "proxy_stats.json"
    KEYS = {"requests", "successes", "errors", "rate_limited", "tokens_in", "tokens_out", "retry_after_ewma", "last_rpm_ts"}

    def __init__(self):
        self.models = {"dead": []}
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                ext = json.load(f)
                safe = {m: {k: v for (k, v) in stats.items() if k in self.KEYS} for m, stats in ext.items()}
                self.models = safe

    def save(self):
        temp = f"{self.filename}.tmp"
        with open(temp, "w") as f:
            json.dump(self.models, f)
        os.replace(temp, self.filename)  # atomic FS operation

    @staticmethod
    def default() -> Dict:
        return {
            "requests": 0,
            "successes": 0,
            "rate_limited": 0,
            "errors": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "last_rpm": 0,
            "last_rpm_ts": 0,
        }

    def record_request(self, model: str):
        if model not in self.models:
            self.models[model] = self.default()
        self.models[model]["requests"] += 1
        self.models[model]["last_rpm_ts"] = time.time()
        self.save()

    def record_success(self, model: str, in_toks: int, out_toks: int):
        self.models[model]["successes"] += 1
        self.models[model]["tokens_in"] += in_toks
        self.models[model]["tokens_out"] += out_toks
        self.save()

    def record_error(self, model: str):
        self.models[model]["errors"] += 1
        self.save()

    def record_429(self, model: str, retry_after: Optional[int] = None):
        """Learn cooldown and update stats."""
        learnt = self.models[model].get("retry_after_ewma", 120)
        curr = retry_after or learnt
        updated = (0.6 * learnt + 0.4 * curr) if learnt else curr
        self.models[model]["retry_after_ewma"] = updated
        self.models[model]["rate_limited"] += 1
        self.save()

    def mark_dead(self, model: str):
        """Never retry this model."""
        self.models["dead"].append(model)
        self.save()


class Cascade:
    """Async-ready ladder. Handles failover, cooldown, stats."""
    def __init__(self, stats: Stats):
        self.static = _ladder()
        self.stats = stats
        self.dead = set(stats.models.get("dead", []))
        self.model_until: Dict[str, float] = {}         # cooldown timestamp
        self.pool = asyncio.Semaphore(int(os.environ.get("PROXY_CONCURRENCY", "10")))

    @property
    def models(self) -> List[str]:
        """Cleaned cloud models."""
        return [m for m in self.static if m not in self.dead]

    def cool(self, model: str, secs: float):
        """Sideline a model (thread-safe)."""
        until = time.time() + secs
        self.model_until[model] = max(self.model_until.get(model, 0), until)

    def order(self, model_id: str, timeout=90) -> List[str]:
        """Build route ladder."""
        if model_id in SPECIAL_IDS:
            model_id = {
                LOCAL_ONLY: LOCAL_MODEL,
                LOCAL_REFINE: LOCAL_MODEL,
                REFINER_MODEL_ID: AUTO_MODEL,
            }.get(model_id, model_id)

        if model_id == ONLY_MODEL:
            return [m for m in self.models if m not in self.dead]
        elif self.is_local(model_id):
            return [LOCAL_MODEL]
        else:          # AUTO_MODEL + any cloud model
            out = []
            now = time.time()
            for m in self.models:
                if m == model_id:
                    out.insert(0, m)        # preferred first
                elif now >= self.model_until.get(m, 0):
                    out.append(m)
            return out

    def is_local(self, model: str) -> bool:
        """Check if route should go to local Ollama."""
        return BOOL_ENV("PROXY_LOCAL_FALLBACK") and (model == LOCAL_MODEL or model in (LOCAL_ONLY, LOCAL_REFINE))

    def local_available(self) -> bool:
        return BOOL_ENV("PROXY_LOCAL_FALLBACK") and LOCAL_MODEL not in self.dead

    async def fetch(self, url, headers, payload=None) -> httpx.Response:
        """HTTP fetch with connection pooling + timeout."""
        async with self.pool:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=20, read=timeout)) as client:
                try:
                    resp = await (client.post if payload else client.get)(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return resp
                except Exception:
                    raise Exception("HTTP error")

    async def route(self, model: str, body: Dict) -> Dict:
        """Dispatch to local|cloud and return response."""
        is_loc = self.is_local(model)
        headers = {"Content-Type": "application/json"}
        url = {
            False: "https://integrate.api.nvidia.com/v1/chat/completions",
            True: f"{os.environ.get('LOCAL_OLLAMA_URL')}/chat/completions",
        }[is_loc]

        if not is_loc:
            headers["Authorization"] = f"Bearer {resolve_api_key()}"

        self.stats.record_request(model)
        resp = await self.fetch(url, headers, payload=body)

        data = resp.json()
        if usage := data.get("usage"):
            in_toks = usage.get("prompt_tokens", 0)
            out_toks = usage.get("completion_tokens", 0)
            self.stats.record_success(model, in_toks, out_toks)
        else:
            self.stats.record_error(model)
        return data


# --- Utils ----------------------------------------------------------------------------------
BOOL_ENV = lambda k: os.environ.get(k, "1").lower() not in {"0", "false", "no", ""}
REFINER_ENABLE = BOOL_ENV("REFINER_ENABLE")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen3-coder-next:80b")
LOCAL_BASE_URL = os.environ.get("LOCAL_OLLAMA_URL", "http://localhost:11434/v1")
REFINER_BASE_URL = os.environ.get("REFINER_BASE_URL", LOCAL_BASE_URL)
REFINER_MODEL = os.environ.get("REFINER_MODEL", "qwen3:4b")
REFINER_TAG = os.environ.get("REFINER_TAG", "[refine]")

# --- Persisted global state ----------------------------------------------------------------
cascade = Cascade(Stats())
_model_override = None  # /pick model picker

# --- Logic --------------------------------------------------------------------------------
def _has_refine_tag(messages: List[Dict]) -> bool:
    """Check '[refine]' in last user message."""
    return REFINER_ENABLE and REFINER_TAG in messages[-1]["content"] if messages else False


async def _refine(content: str) -> str:
    """Rewrite user message via small Ollama."""
    body = {"model": REFINER_MODEL, "messages": [{"role": "user", "content": content}], "stream": False}
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            resp = await client.post(f"{REFINER_BASE_URL}/chat/completions", json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return content  # graceful fallback


app = FastAPI(title="nvidia-failover")

# --- API -----------------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/")
async def dashboard():
    """Live dashboard."""
    rows = []
    running_ts = time.time()
    for model in cascade.models:
        stats = cascade.stats.models.get(model, cascade.stats.default())
        cooling = max(0, int(cascade.model_until.get(model, 0.0) - running_ts))
        active = {
            cooling == 0: "live",
            cooling > 0: "cooling",
        }.get(True, "dead")
        rpm = len([t for t in chase.stats.models.get(model, {}).get("last_rpm_s", []) if running_ts - 60 < t <= running_ts])
        rows.append({
            "model": model,
            "state": active,
            "rpm": rpm,
            "cooling": cooling,
            "tokens_in": stats.get("tokens_in", 0),
            "tokens_out": stats.get("tokens_out", 0),
            "errors": stats.get("errors", 0),
        })

    # HTML UI with embed \u003cscript\u003e for live~SSE
    return """
    \n    """


@app.get("/health")
async def health():
    return {
        "ok": bool(resolve_api_key()),
        "models": len(cascade.models),
        "local": LOCAL_MODEL if BOOL_ENV("PROXY_LOCAL_FALLBACK") else None,
        "cooling": {m: int(t - time.time()) for m, t in cascade.model_until.items()},
    }


@app.post("/v1/chat/completions")
async def completions(req: Request):
    body = await req.json()
    preferred = body.get("model", AUTO_MODEL)
    
    # Model picker override
    global _model_override
    if _model_override and preferred in {None, AUTO_MODEL}:
        preferred = _model_override

    # Commands intercept
    if body["messages"].endswith("content").startswith("/"):
        cmd, args = body["messages"][-1]["content"].strip("/"), ""
        out = {
            "model": "proxy-commands",
            "choices": [{"message": {"content": _handle_command(cmd, args)} }]
        }
        return JSONResponse(out)

    # Refiner
    if _has_refine_tag(body["messages"]):
        cleaned = body["messages"][-1]["content"].replace(REFINER_TAG, "").strip()
        body["messages"][-1]["content"] = await _refine(cleaned)

    # Dispatch
    ladder = cascade.order(preferred)
    for m in ladder:
        try:
            resp = await cascade.route(m, body)
            resp["_proxy_model"] = m
            return JSONResponse(resp)
        except Exception:
            continue
    return JSONResponse({"error": "All ladder models cooling down"}, status_code=429)


def _handle_command(cmd: str, args: str) -> str:
    cmds = {
        "help": _fmt_help,
        "stats": _fmt_stats,
        "pick": /pick,
    }
    return cmds.get(cmd, lambda _: "Unknown command (‘/help’)")(args)


def _fmt_stats(*_) -> str:
    """/stats"""
    return """\nRPM: {}. TPM: {} in / {} out\n""".strip()


# --- Main -----------------------------------------------------------------------------------
if __name__ == "__main__":
    host, port = "0.0.0.0", int(os.environ.get("PROXY_PORT", 5002))
    uvicorn.run(app, host=host, port=port)