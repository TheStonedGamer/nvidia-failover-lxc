"""Environment-derived configuration: provider defaults, curated ladder,
pricing/frequency-penalty tables, agent role prompts, and feature toggles.

Ported verbatim (values and logic) from nvidia_failover_proxy.py so behavior
is unchanged; only the module boundary is new.
"""

import json
import os
import re
from typing import Dict, List, Optional

import httpx

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# NVIDIA API key resolution: env NVIDIA_API_KEY first, else the key already in
# OpenCode's opencode.jsonc so it lives in exactly one place. A targeted regex
# beats naive JSONC comment-stripping, which would corrupt "https://..." strings.
_OPENCODE_JSONC = os.path.expanduser("~/.config/opencode/opencode.jsonc")
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


# Curated frontier ladder — strongest first. Any model that ever returns
# 404/410 is dropped automatically at runtime, so a stale entry is harmless.
# Override wholesale with ROUTER_NVIDIA_MODELS.
FRONTIER_MODELS: List[str] = [
    "deepseek-ai/deepseek-v4-pro",
    "qwen/qwen3.5-397b-a17b",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2",
    "minimaxai/minimax-m3",
    "openai/gpt-oss-120b",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "qwen/qwen3.5-122b-a10b",
    "nvidia/nemotron-3-super-120b-a12b",
    "mistralai/mistral-medium-3.5-128b",
    "meta/llama-4-maverick-17b-128e-instruct",
    "deepseek-ai/deepseek-v4-flash",
    "meta/llama-3.3-70b-instruct",
]

# Env-var seeding for the major OpenAI-compatible providers. name -> default base_url.
PROVIDER_ENV: Dict[str, str] = {
    "nvidia": NVIDIA_BASE_URL,
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "xai": "https://api.x.ai/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}

AUTO_MODEL = "nvidia-auto"
ONLY_MODEL = "nvidia-only"
REFINER_MODEL_ID = "nvidia-refine"
_MODEL_COOLDOWN_S = 5 * 60
_CONNECT_COOLDOWN_S = 20
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

_DEFAULT_MAX_TOKENS = int(os.environ.get("PROXY_MAX_TOKENS_DEFAULT", "8192"))

# --- Estimated money-saved accounting ----------------------------------------
_MODEL_PRICING_DEFAULT: Dict[str, tuple] = {
    "kimi-k2": (0.80, 3.50),
    "moonshot": (0.80, 3.50),
    "glm": (1.00, 3.20),
    "deepseek-v4-pro": (1.74, 3.48),
    "deepseek-pro": (1.74, 3.48),
    "deepseek": (0.60, 1.70),
    "minimax": (0.30, 1.20),
    "nemotron": (0.60, 1.80),
    "qwen": (0.35, 1.40),
    "mistral-large": (2.00, 6.00),
    "mistral-medium": (0.40, 2.00),
    "mistral": (0.40, 2.00),
    "mixtral": (0.60, 0.60),
    "llama-4": (0.20, 0.60),
    "llama": (0.35, 0.80),
    "gpt-oss": (0.30, 1.20),
    "phi": (0.15, 0.45),
    "gemma": (0.15, 0.45),
}


def _default_pricing_rate() -> tuple:
    raw = os.environ.get("PROXY_PRICING_DEFAULT", "").strip()
    if raw:
        try:
            a, b = raw.split(",")
            return (float(a), float(b))
        except Exception:
            print(f"[pricing] ignoring bad PROXY_PRICING_DEFAULT={raw!r}")
    return (0.50, 1.50)


def _load_pricing() -> Dict[str, tuple]:
    table = dict(_MODEL_PRICING_DEFAULT)
    raw = os.environ.get("PROXY_PRICING_JSON", "").strip()
    if raw:
        try:
            for k, v in json.loads(raw).items():
                table[k.lower()] = (float(v[0]), float(v[1]))
        except Exception as e:
            print(f"[pricing] ignoring bad PROXY_PRICING_JSON: {e}")
    return table


_MODEL_PRICING = _load_pricing()
_PRICING_DEFAULT_RATE = _default_pricing_rate()


def price_for(model: str) -> tuple:
    """(in, out) USD per 1M tokens the equivalent model would cost commercially."""
    ml = (model or "").lower()
    for key, rate in _MODEL_PRICING.items():
        if key in ml:
            return rate
    return _PRICING_DEFAULT_RATE


def saved_usd(model: str, tokens_in, tokens_out) -> float:
    pin, pout = price_for(model)
    return (tokens_in or 0) / 1_000_000 * pin + (tokens_out or 0) / 1_000_000 * pout


def fmt_money(v: float) -> str:
    """Compact USD: bigger numbers lose the cents, sub-dollar keeps precision."""
    if not v:
        return "$0"
    if v >= 1000:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    return f"${v:.4f}".rstrip("0").rstrip(".")


# --- Per-model repetition guard ----------------------------------------------
_FREQ_PENALTY_DEFAULT_TABLE: Dict[str, float] = {
    "kimi-k2": 0.3,
}


def _load_freq_penalties() -> Dict[str, float]:
    table = dict(_FREQ_PENALTY_DEFAULT_TABLE)
    raw = os.environ.get("PROXY_FREQ_PENALTY_JSON", "").strip()
    if raw:
        try:
            for k, v in json.loads(raw).items():
                table[k.lower()] = float(v)
        except Exception as e:
            print(f"[freq-penalty] ignoring bad PROXY_FREQ_PENALTY_JSON: {e}")
    return table


_FREQ_PENALTY_TABLE = _load_freq_penalties()
try:
    _FREQ_PENALTY_GLOBAL = float(os.environ.get("PROXY_FREQ_PENALTY_DEFAULT", "0") or 0)
except ValueError:
    _FREQ_PENALTY_GLOBAL = 0.0


def freq_penalty_for(model: str):
    """frequency_penalty to inject for this model when the client sent none."""
    ml = (model or "").lower()
    for key, val in _FREQ_PENALTY_TABLE.items():
        if key in ml:
            return val
    return _FREQ_PENALTY_GLOBAL or None


# Agent profile model IDs — same cloud ladder but inject a role system prompt.
_CJK_CODE_RULE = (
    " Write all code, identifiers, comments, and string literals in ASCII/English; "
    "never emit Chinese or other CJK characters in source files or tool arguments."
)
AGENT_ROLES = {
    "agent-planner": "You are a meticulous planning agent. Before answering, produce a clear numbered step-by-step plan. Then execute each step thoroughly."
    + _CJK_CODE_RULE,
    "agent-builder": "You are a builder agent. Implement solutions with clean, well-structured, "
    "production-ready code. Include error handling, logging, and tests. Do the work with tools "
    "rather than describing it: when the next step is clear, call the tool immediately. Never end "
    "your turn right after announcing an action — if you say you will create or edit a file, the "
    "matching tool call MUST be in the same response, or the action never happens and the loop "
    "stalls." + _CJK_CODE_RULE,
    "agent-reviewer": "You are a review agent. Scrutinize code and plans for bugs, edge cases, security flaws, and performance issues. Be thorough but constructive."
    + _CJK_CODE_RULE,
}
LOCAL_ONLY = "local-only"
LOCAL_REFINE = "local-refine"

SPECIAL_IDS = {
    AUTO_MODEL,
    ONLY_MODEL,
    REFINER_MODEL_ID,
    LOCAL_ONLY,
    LOCAL_REFINE,
} | set(AGENT_ROLES)

# Prompt refiner
REFINER_ENABLE = os.environ.get("PROXY_REFINER_ENABLE", "1").lower() not in (
    "0", "false", "no", "",
)
REFINER_BASE_URL = os.environ.get("REFINER_BASE_URL", "http://10.0.0.142:11434/v1")
REFINER_MODEL = os.environ.get("REFINER_MODEL", "qwen3:4b")
REFINER_TAG = os.environ.get("REFINER_TAG", "[refine]")

# Local Ollama tail rung — used only when the whole cloud ladder is cooling.
LOCAL_ENABLE = os.environ.get("PROXY_LOCAL_FALLBACK", "1").lower() not in (
    "0", "false", "no", "",
)
LOCAL_BASE_URL = os.environ.get("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen3-Coder-Next-80b-A3B:latest")


def strip_v1(url: str) -> str:
    """Base URL without a trailing /v1 — for Ollama's native (non-OpenAI) API.
    NOT rstrip("/v1"): that strips *characters* and would eat a port ending in
    1/v (e.g. http://host:11431/v1 -> http://host:1143)."""
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")].rstrip("/")
    return u


def ladder() -> List[str]:
    env = os.environ.get("ROUTER_NVIDIA_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    return list(FRONTIER_MODELS)


# Learned-limit tuning
_EWMA_ALPHA = 0.4
_RPM_WINDOW_S = 60.0
_THROTTLE_RATIO = 0.85
_RPM_LIMIT_FLOOR = float(os.environ.get("PROXY_RPM_LIMIT_FLOOR", "5"))
_TPM_IN_LIMIT_FLOOR = float(os.environ.get("PROXY_TPM_IN_LIMIT_FLOOR", "8000"))
_TPM_OUT_LIMIT_FLOOR = float(os.environ.get("PROXY_TPM_OUT_LIMIT_FLOOR", "2000"))
_SERVING_WINDOW_S = float(os.environ.get("PROXY_SERVING_WINDOW_S", "20"))
_STICKY_LADDER = os.environ.get("PROXY_STICKY_LADDER", "1") not in ("0", "false", "no")

# Guards
SKIP_EMPTY = os.environ.get("PROXY_SKIP_EMPTY", "1").lower() not in (
    "0", "false", "no", "",
)
GUARD_DEGENERATE = os.environ.get("PROXY_GUARD_DEGENERATE", "1").lower() not in (
    "0", "false", "no", "",
)
_REP_MIN_REPEATS = int(os.environ.get("PROXY_REP_MIN_REPEATS", "8"))
_REP_MAX_UNIT = int(os.environ.get("PROXY_REP_MAX_UNIT", "60"))
_REP_MIN_RUN = int(os.environ.get("PROXY_REP_MIN_RUN", "80"))

GUARD_CJK_CODE = os.environ.get("PROXY_GUARD_CJK_CODE", "1").lower() not in (
    "0", "false", "no", "",
)
_CJK_CODE_MIN = int(os.environ.get("PROXY_CJK_CODE_MIN", "2"))

_STREAM_STALL_S = float(os.environ.get("PROXY_STREAM_STALL_S", "180"))
