"""NVIDIA frontier-model failover proxy.

A tiny OpenAI-compatible endpoint you can point OpenCode (or anything that
speaks the OpenAI chat API) at. It forwards to NVIDIA's hosted API
(integrate.api.nvidia.com) and **auto-cycles through the frontier models**:
when one is rate-limited (429) or dead (404/410), the same request transparently
fails over to the next model in the ladder. Works for both streaming and
non-streaming requests. Rate-limited models are sidelined on a short cooldown
and rejoin the ladder automatically.

**Local tail rung:** a local Ollama model (default the qwen3-coder-next-80b in
Ollama) sits at the *end* of the ladder, so when the whole NVIDIA frontier tier
is rate-limited for the time being, the request keeps going on your local model
instead of failing — and picks the cloud models back up as soon as their
cooldowns expire. Disable with `PROXY_LOCAL_FALLBACK=0`.

**Context is preserved across every switch.** The proxy is stateless: the client
(OpenCode) resends the full `messages` history each turn and the proxy forwards
it verbatim to whichever model serves. So a conversation that starts on a cloud
frontier model, fails over to another, and eventually lands on the local model
carries its entire context along — no state is dropped at any hop.

Point OpenCode at it:
    base URL   http://127.0.0.1:5002/v1
    api key    anything (the real nvapi- key is resolved server-side)
    model      "nvidia-auto"  (start at the top of the ladder)
               or any real NVIDIA model id (tried first, then cascades)

Run:
    NVIDIA_API_KEY resolves from env, else from ~/.config/opencode/opencode.jsonc
    E:\\Projects\\model-router\\.venv\\Scripts\\python.exe nvidia_failover_proxy.py
    # optional overrides:
    #   PROXY_PORT=5002
    #   ROUTER_NVIDIA_MODELS="deepseek-ai/deepseek-v4-pro,qwen/qwen3.5-397b-a17b,..."
"""

import asyncio
import base64
import json
import os
import re
import sqlite3
import time
from typing import Dict, List, Optional, Set

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# NVIDIA API key resolution — inlined so this file is fully self-contained (the
# standalone installers, the Docker image, and the LXC helper all ship just this
# one module). Env NVIDIA_API_KEY first, else the key already in OpenCode's
# opencode.jsonc so it lives in exactly one place. A targeted regex beats naive
# JSONC comment-stripping, which would corrupt "https://..." strings.
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


# Curated frontier ladder — the top-tier flagship chat/reasoning models on the
# NVIDIA hosted API, strongest first (verified present 2026-07-06 via /v1/models).
# Any model that ever returns 404/410 is dropped automatically at runtime, so a
# stale entry here is harmless. Override wholesale with ROUTER_NVIDIA_MODELS.
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

# Env-var seeding for the major OpenAI-compatible providers. On first run (when
# the provider isn't already configured) the proxy reads <PREFIX>_API_KEY and, if
# present, adds that provider with its default base_url. Optional companions:
#   <PREFIX>_MODELS     comma-separated model ids to seed into the failover ladder
#   <PREFIX>_BASE_URL   override the default endpoint (e.g. a self-hosted gateway)
# name -> (env prefix, default base_url). NVIDIA also honors the legacy
# NVIDIA_API_KEY / ROUTER_NVIDIA_MODELS handled by _migrate_nvidia_provider().
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

AUTO_MODEL = "nvidia-auto"  # cloud frontier ladder, then local 80B tail rung
ONLY_MODEL = "nvidia-only"  # cloud frontier ladder ONLY; 429 when all cooling
REFINER_MODEL_ID = "nvidia-refine"  # cloud ladder + local Qwen 4b prompt refiner
_MODEL_COOLDOWN_S = 5 * 60  # sideline a 429'd model this long (or its retry-after)
# A failure to *connect* (vs. a 429 or a hung/slow model) is usually a transient
# network blip — NVIDIA publishes several A records and a fresh connection often
# lands on a healthy one. We retry the connect once and, if it still fails, only
# briefly sideline the model instead of cooling it for 5 minutes, so one blip
# can't drain the whole ladder and hard-fail an in-flight request.
_CONNECT_COOLDOWN_S = 20
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

# Default max_tokens when the client doesn't send one. NIM quirk: some models
# return empty content unless max_tokens is set, so we must supply *something*.
# 8192 is the safe floor — most frontier NIM models accept it; raising it higher
# makes lower-cap models 400 and burn a failover hop on every request (measured:
# kimi-k2.6 400s at 32768). Bump this only if every model in your ladder supports
# a larger output window. A client-supplied max_tokens always wins.
_DEFAULT_MAX_TOKENS = int(os.environ.get("PROXY_MAX_TOKENS_DEFAULT", "8192"))

# --- Estimated money-saved accounting ----------------------------------------
# NVIDIA NIM is free on a personal account, so every token this proxy serves is
# spend AVOIDED at a commercial API. To estimate the savings we price each model
# at a representative commercial rate (USD per 1M tokens, input / output) for the
# same open-weight model at typical hosts (Together / DeepInfra / OpenRouter).
# Rates below were checked against those hosts' published pricing in mid-2026 and
# are a representative mid-point across them — the point is a defensible "money
# saved" figure, not an invoice; providers vary and prices drift, so re-check or
# override. Matching is by case-insensitive substring against the model id, first
# hit wins, so keep more-specific keys earlier.
# Override any entry (or add models) with PROXY_PRICING_JSON, e.g.
#   PROXY_PRICING_JSON='{"kimi-k2": [0.6, 2.5], "my-model": [1.0, 3.0]}'
# and the unmatched-model fallback with PROXY_PRICING_DEFAULT="in,out".
# Sources (mid-2026): DeepInfra & Together pricing pages, pricepertoken.com,
# artificialanalysis.ai. Kimi K2.6 ~0.75/3.50 (DeepInfra) .. 1.20/4.50 (Together);
# GLM-5.2 ~1.00/3.20 (GLM-5.1 1.40/4.40, GLM-5 0.80/2.56); DeepSeek V4 Pro
# 1.74/3.48, V3.1 0.60/1.70; Qwen3.x 0.20/1.00 (small) .. higher for 397B;
# Nemotron Ultra 253B 0.60/1.80; MiniMax M3 0.30/1.20; Llama 3.3 70B ~0.88 flat,
# Llama-4 Maverick ~0.20/0.60; Mistral Large ~2/6, Medium ~0.40/2.00.
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
    return (0.50, 1.50)  # per 1M in / out for models with no table match


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


def _price_for(model: str) -> tuple:
    """(in, out) USD per 1M tokens the equivalent model would cost commercially."""
    ml = (model or "").lower()
    for key, rate in _MODEL_PRICING.items():
        if key in ml:
            return rate
    return _PRICING_DEFAULT_RATE


def _saved_usd(model: str, tokens_in, tokens_out) -> float:
    pin, pout = _price_for(model)
    return (tokens_in or 0) / 1_000_000 * pin + (tokens_out or 0) / 1_000_000 * pout


def _fmt_money(v: float) -> str:
    """Compact USD: bigger numbers lose the cents, sub-dollar keeps precision."""
    if not v:
        return "$0"
    if v >= 1000:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    return f"${v:.4f}".rstrip("0").rstrip(".")


# --- Per-model repetition guard ----------------------------------------------
# Some NIM models (notably moonshotai/kimi-k2.6) periodically fall into a
# degenerate repetition loop on long "explain thoroughly" prompts and, once
# looping, code-switch to Chinese — NIM surfaces this as finish_reason=repetition.
# A mild frequency_penalty breaks the loop. We inject one ONLY for models that
# need it, and ONLY when the client didn't send frequency_penalty (client wins).
# Matched case-insensitively by substring; override with PROXY_FREQ_PENALTY_JSON,
# e.g. PROXY_FREQ_PENALTY_JSON='{"kimi-k2":0.4,"some-model":0.3}'. A global
# default for all other models can be set with PROXY_FREQ_PENALTY_DEFAULT (0=off).
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


def _freq_penalty_for(model: str):
    """frequency_penalty to inject for this model when the client sent none."""
    ml = (model or "").lower()
    for key, val in _FREQ_PENALTY_TABLE.items():
        if key in ml:
            return val
    return _FREQ_PENALTY_GLOBAL or None


# Agent profile model IDs — same cloud ladder but inject a role system prompt.
# OpenCode picks these up as separate models for multi-agent workflows.
AGENT_ROLES = {
    "agent-planner": "You are a meticulous planning agent. Before answering, produce a clear numbered step-by-step plan. Then execute each step thoroughly.",
    "agent-builder": "You are a builder agent. Implement solutions with clean, well-structured, production-ready code. Include error handling, logging, and tests.",
    "agent-reviewer": "You are a review agent. Scrutinize code and plans for bugs, edge cases, security flaws, and performance issues. Be thorough but constructive.",
}
# Local model IDs — route directly to the local Ollama 80B instead of cloud.
LOCAL_ONLY = "local-only"  # direct to local 80B
LOCAL_REFINE = "local-refine"  # refiner → local 80B

# Model override: set by the /pick command. When non-None, this model ID is
# used for all subsequent requests (unless the request explicitly specifies one).
_model_override: Optional[str] = None

# All special model IDs that trigger custom routing (not a real NVIDIA model).
SPECIAL_IDS = {
    AUTO_MODEL,
    ONLY_MODEL,
    REFINER_MODEL_ID,
    LOCAL_ONLY,
    LOCAL_REFINE,
} | set(AGENT_ROLES)

# Prompt refiner — a tiny local model that rewrites user prompts before they
# reach the main model. Include `[refine]` anywhere in your message to activate.
# The tag is stripped and the refiner's output becomes the new user message.
REFINER_ENABLE = os.environ.get("PROXY_REFINER_ENABLE", "1").lower() not in (
    "0",
    "false",
    "no",
    "",
)
REFINER_BASE_URL = os.environ.get("REFINER_BASE_URL", "http://10.0.0.142:11434/v1")
REFINER_MODEL = os.environ.get("REFINER_MODEL", "qwen3:4b")
REFINER_TAG = os.environ.get("REFINER_TAG", "[refine]")

# Local Ollama tail rung — used only when the whole cloud ladder is cooling.
LOCAL_ENABLE = os.environ.get("PROXY_LOCAL_FALLBACK", "1").lower() not in (
    "0",
    "false",
    "no",
    "",
)
LOCAL_BASE_URL = os.environ.get("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen3-Coder-Next-80b-A3B:latest")


def _strip_v1(url: str) -> str:
    """Base URL without a trailing /v1 — for Ollama's native (non-OpenAI) API.
    NOT rstrip("/v1"): that strips *characters* and would eat a port ending in
    1/v (e.g. http://host:11431/v1 → http://host:1143)."""
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")].rstrip("/")
    return u


def _ladder() -> List[str]:
    env = os.environ.get("ROUTER_NVIDIA_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    return list(FRONTIER_MODELS)


# --- SQLite persistence -------------------------------------------------------------
# One DB holds both the config (order/toggles/providers/keys) and the learned
# per-model stats. Legacy JSON files are auto-imported once on first run.
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.environ.get("PROXY_DB_FILE", os.path.join(_HERE, "proxy.db"))


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS stats (model TEXT PRIMARY KEY, data TEXT)")
    return conn


def _db_secure() -> None:
    """DB holds API keys — keep it (and WAL sidecars) owner-only."""
    for suffix in ("", "-wal", "-shm"):
        try:
            os.chmod(DB_FILE + suffix, 0o600)
        except OSError:
            pass


def _kv_get_all(conn: sqlite3.Connection) -> Dict[str, str]:
    return {k: v for k, v in conn.execute("SELECT key, value FROM kv")}


def _kv_set(conn: sqlite3.Connection, mapping: Dict[str, str]) -> None:
    conn.executemany(
        "INSERT INTO kv(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        list(mapping.items()),
    )


# --- Ladder config (user-reorderable + toggleable via web UI) -----------------------
CONFIG_FILE = os.environ.get("PROXY_CONFIG_FILE", "proxy_config.json")


class LadderConfig:
    """Persists user-defined failover order, disabled models, custom
    OpenAI-compatible providers, and API-key overrides to proxy_config.json.

    The web UI calls /_config (order/toggles), /_settings (providers + keys),
    and /_models/available (discovery). Cascade.order() consults active_ladder()
    and _route() consults model_provider() for per-model base_url/key.

    Each custom provider: {name, base_url, api_key, models: [ids...]}. Its model
    ids route to that provider's base_url + key instead of NVIDIA.
    """

    def __init__(self) -> None:
        # No providers / no ladder by default — a fresh install is empty and the
        # user adds providers (NVIDIA, OpenAI, Anthropic, …) via the web UI.
        self.order: List[str] = []
        self.disabled: Set[str] = set()
        # name -> {"base_url": str, "api_key": str, "models": [str, ...]}
        self.providers: Dict[str, dict] = {}
        self.nvidia_key: Optional[str] = None  # legacy override (migrated → provider)
        # Selected local tail model (None = auto-pick first Ollama provider model).
        self.local_tail: Optional[str] = None
        self.load()

    def load(self) -> None:
        try:
            conn = _db()
            try:
                kv = _kv_get_all(conn)
                if not kv and os.path.exists(CONFIG_FILE):
                    # One-time migration from the legacy JSON config.
                    self._import_json()
                    self._write(conn)
                    conn.commit()
                    _db_secure()
                    kv = _kv_get_all(conn)
                if "order" in kv:
                    self.order = json.loads(kv["order"])
                if "disabled" in kv:
                    self.disabled = set(json.loads(kv["disabled"]))
                if "providers" in kv:
                    self.providers = json.loads(kv["providers"])
                if kv.get("nvidia_key"):
                    self.nvidia_key = kv["nvidia_key"]
                if kv.get("local_tail"):
                    self.local_tail = kv["local_tail"]
                changed = self._migrate_nvidia_provider()
                changed = self._seed_providers_from_env() or changed
                if changed:
                    self._write(conn)
                    conn.commit()
                    _db_secure()
            finally:
                conn.close()
        except (sqlite3.Error, json.JSONDecodeError, OSError):
            pass

    def _migrate_nvidia_provider(self) -> bool:
        """Move the legacy dedicated NVIDIA key + hardcoded frontier ladder into
        a first-class 'nvidia' provider, so NVIDIA is just another provider.
        Only fires when there's prior NVIDIA state (a key or an existing ladder);
        a genuinely fresh install stays empty. Returns True if anything changed."""
        if "nvidia" in self.providers:
            return False
        key = self.nvidia_key or resolve_api_key() or ""
        # nvidia frontier ids already in the saved order (from prior versions).
        nvidia_models = [m for m in self.order if m in FRONTIER_MODELS]
        if not (self.nvidia_key or nvidia_models):
            return False  # nothing to migrate — leave the install empty
        if not nvidia_models:
            nvidia_models = list(FRONTIER_MODELS)
        self.providers["nvidia"] = {
            "base_url": NVIDIA_BASE_URL,
            "api_key": key,
            "models": nvidia_models,
        }
        for m in nvidia_models:
            if m not in self.order:
                self.order.append(m)
        self.nvidia_key = None  # key now lives on the provider
        return True

    def _seed_providers_from_env(self) -> bool:
        """Seed any major OpenAI-compatible provider from environment variables.
        For each entry in PROVIDER_ENV, if <PREFIX>_API_KEY is set and that
        provider isn't already configured, add it (with optional <PREFIX>_MODELS
        and <PREFIX>_BASE_URL). Only seeds when absent, so the web UI stays the
        source of truth once a provider exists. Returns True if anything changed.

        Ollama is special: it does not require an API key (local) and its
        default model list comes from OLLAMA_MODELS (or the legacy LOCAL_MODEL)."""
        changed = False
        for name, default_url in PROVIDER_ENV.items():
            if name in self.providers:
                continue  # user/UI already owns this provider
            prefix = name.upper()
            key = (os.environ.get(f"{prefix}_API_KEY") or "").strip()
            # Ollama doesn't require an API key for local use
            if not key and name != "ollama":
                continue
            base_url = (os.environ.get(f"{prefix}_BASE_URL") or default_url).strip()
            # Legacy LOCAL_OLLAMA_URL env var — only used when OLLAMA_BASE_URL is not set
            if name == "ollama" and not os.environ.get(f"{prefix}_BASE_URL"):
                legacy = (os.environ.get("LOCAL_OLLAMA_URL") or "").strip()
                if legacy:
                    base_url = legacy

            raw = os.environ.get(f"{prefix}_MODELS") or ""
            # Legacy LOCAL_MODEL feeds into ollama model list if OLLAMA_MODELS not set
            if name == "ollama" and not raw:
                raw = os.environ.get("LOCAL_MODEL", "Qwen3-Coder-Next-80b-A3B:latest")
            if not raw and name == "nvidia":
                raw = os.environ.get("ROUTER_NVIDIA_MODELS") or ""
            models = [m.strip() for m in raw.split(",") if m.strip()]
            if name == "nvidia" and not models:
                models = list(FRONTIER_MODELS)
            self.providers[name] = {
                "base_url": base_url,
                "api_key": key,
                "models": models,
            }
            # Ollama models are not added to the failover order — they serve as
            # the local tail rung (handled by Cascade._local_model()) and can be
            # added to the ladder manually via the UI if the user wants.
            if name != "ollama":
                for m in models:
                    if m not in self.order:
                        self.order.append(m)
            changed = True
        return changed

    def _import_json(self) -> None:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("order"), list):
                self.order = data["order"]
            if isinstance(data.get("disabled"), list):
                self.disabled = set(data["disabled"])
            if isinstance(data.get("providers"), dict):
                self.providers = data["providers"]
            if isinstance(data.get("nvidia_key"), str) and data["nvidia_key"]:
                self.nvidia_key = data["nvidia_key"]
        except (OSError, json.JSONDecodeError):
            pass

    def _write(self, conn: sqlite3.Connection) -> None:
        data: Dict[str, str] = {
            "order": json.dumps(self.order),
            "disabled": json.dumps(list(self.disabled)),
            "providers": json.dumps(self.providers),
            "nvidia_key": self.nvidia_key or "",
        }
        if self.local_tail:
            data["local_tail"] = self.local_tail
        _kv_set(conn, data)

    def save(self) -> None:
        try:
            conn = _db()
            try:
                self._write(conn)
                conn.commit()
            finally:
                conn.close()
            _db_secure()
        except sqlite3.Error:
            pass

    def update(
        self, order: Optional[List[str]] = None, disabled: Optional[List[str]] = None
    ) -> None:
        if order is not None:
            self.order = order
        if disabled is not None:
            self.disabled = set(disabled)
        self.save()

    def is_enabled(self, model: str) -> bool:
        return model not in self.disabled

    def active_ladder(self) -> List[str]:
        """Return models in user-defined order, excluding disabled ones."""
        return [m for m in self.order if m not in self.disabled]

    # --- custom providers + keys ------------------------------------------
    def custom_models(self) -> List[str]:
        out: List[str] = []
        for p in self.providers.values():
            out.extend(m for m in p.get("models", []) if m not in out)
        return out

    def model_provider(self, model: str) -> Optional[tuple]:
        """(base_url, api_key) for a custom-provider model, else None (=NVIDIA)."""
        for p in self.providers.values():
            if model in p.get("models", []):
                return p.get("base_url", "").rstrip("/"), p.get("api_key") or None
        return None

    def set_nvidia_key(self, key: Optional[str]) -> None:
        k = key.strip() if key and key.strip() else None
        # NVIDIA is a provider now — keep its key in sync if it exists.
        if "nvidia" in self.providers:
            self.providers["nvidia"]["api_key"] = k or ""
        self.nvidia_key = k
        self.save()

    def add_provider(
        self, name: str, base_url: str, api_key: str, models: List[str]
    ) -> None:
        """Add or update a provider. Re-adding an existing name merges: a blank
        base_url/key keeps the old value, and models are unioned — so you can
        update just a provider's key without losing its model list."""
        name = name.strip()
        base_url = base_url.strip()
        models = [m.strip() for m in models if m.strip()]
        existing = self.providers.get(name, {})
        merged = list(existing.get("models", []))
        for m in models:
            if m not in merged:
                merged.append(m)
        self.providers[name] = {
            "base_url": base_url or existing.get("base_url", ""),
            "api_key": (api_key or "").strip() or existing.get("api_key", ""),
            "models": merged,
        }
        # Append new models to the failover order so they join the cascade.
        for m in models:
            if m not in self.order:
                self.order.append(m)
        self.save()

    def remove_provider(self, name: str) -> None:
        p = self.providers.pop(name, None)
        if p:
            gone = set(p.get("models", []))
            self.order = [m for m in self.order if m not in gone]
            self.disabled -= gone
        self.save()

    def set_local_tail(self, model: Optional[str]) -> None:
        """Set the local tail model (None = auto from first Ollama model)."""
        self.local_tail = model or None
        self.save()

    def resolved_nvidia_key(self) -> Optional[str]:
        return self.nvidia_key or resolve_api_key()


ladder_config = LadderConfig()


class Cascade:
    """Holds the frontier ladder plus per-model rate-limit / EOL state.

    The local tail rung is now sourced from the Ollama provider in
    ladder_config (instead of a hardcoded env var). When no Ollama provider
    exists, a legacy LOCAL_MODEL env fallback is used for backward compat.
    """

    def __init__(self) -> None:
        self.models = _ladder()  # cloud (NVIDIA) frontier ladder
        self.model_until: Dict[str, float] = {}  # model -> cooling-until epoch
        self.dead: Set[str] = set()  # 404/410 — dropped permanently

    def _local_model(self) -> Optional[str]:
        """First model from the Ollama provider, else legacy LOCAL_MODEL env.

        If the user has set a specific local_tail in the config (via the web UI
        dropdown), that model is used instead of the first Ollama provider model."""
        # User-selected local tail (from UI dropdown) takes priority.
        lt = ladder_config.local_tail
        if lt:
            return lt
        p = ladder_config.providers.get("ollama", {})
        models = p.get("models", [])
        if models:
            return models[0]
        # Backward compat: legacy env var fallback
        if LOCAL_ENABLE:
            return LOCAL_MODEL
        return None

    def _ollama_base_url(self) -> str:
        """Base URL for the Ollama provider, else legacy LOCAL_OLLAMA_URL."""
        p = ladder_config.providers.get("ollama", {})
        url = p.get("base_url", "").strip()
        if url:
            return url
        return LOCAL_BASE_URL

    def is_local(self, model: Optional[str]) -> bool:
        """True if model is the local Ollama fallback or a local-only alias."""
        if not model:
            return False
        if model in (LOCAL_ONLY, LOCAL_REFINE):
            return True
        # Check if the model belongs to the Ollama provider.
        p = ladder_config.providers.get("ollama", {})
        if model in p.get("models", []):
            return True
        return bool(LOCAL_ENABLE) and model == LOCAL_MODEL

    def order(self, preferred: Optional[str]) -> List[str]:
        """Try `preferred` first (if it's a real, usable model), then cascade
        through the rest of the cloud ladder (skipping dead / cooling models),
        and finally the local model as the guaranteed tail rung — so when the
        whole cloud tier is rate-limited the request still lands somewhere.

        Special ids control tail behavior:
          - AUTO_MODEL ("nvidia-auto"): cloud ladder, then the local 80B tail.
          - ONLY_MODEL ("nvidia-only"): cloud ladder ONLY. 429 when all cooling.
          - REFINER_MODEL_ID, AGENT_ROLES: same as nvidia-auto (cloud→local).
          - LOCAL_ONLY ("local-only"): local 80B only, no cloud.
          - LOCAL_REFINE ("local-refine"): refiner + local 80B (refiner runs upstream)."""
        now = time.time()
        cloud_only = preferred == ONLY_MODEL
        # User-defined order + toggles from the web UI (SQLite). Empty by
        # default — when no providers/models are configured only the local
        # tail rung (if any) remains.
        base = ladder_config.active_ladder()
        if preferred and preferred not in SPECIAL_IDS and not self.is_local(preferred):
            base = [preferred] + [m for m in base if m != preferred]
        cloud = [
            m
            for m in base
            if m not in self.dead
            and now >= self.model_until.get(m, 0.0)
            and not stats.is_near_limit(m)
        ]

        local = self._local_model()

        # If the caller explicitly asked for a local model, honor it first.
        # local-only and local-refine route ONLY to local (no cloud fallback).
        if self.is_local(preferred):
            if not local:
                return []
            if preferred in (LOCAL_ONLY, LOCAL_REFINE):
                return [local]
            return [local] + cloud

        if cloud_only:
            # Cloud-only: no local tail, no "closest to reviving" grace — if every
            # cloud model is cooling, return empty so the caller returns 429.
            return cloud

        ladder = cloud + ([local] if local else [])
        if not ladder:
            # No local rung and every cloud model cooling: try the one closest
            # to coming back rather than hard-failing.
            alive = [m for m in base if m not in self.dead]
            ladder = sorted(alive, key=lambda m: self.model_until.get(m, 0.0))[:1]
        return ladder

    def soonest_cooldown(self) -> int:
        """Seconds until the nearest cloud model comes back (for retry-after)."""
        now = time.time()
        live = [
            t for m, t in self.model_until.items() if m not in self.dead and t > now
        ]
        return int(min(live) - now) if live else _MODEL_COOLDOWN_S

    def cool(self, model: str, secs: float) -> None:
        """Sideline a model for `secs` (e.g. after a timeout/transport error)."""
        self.model_until[model] = max(
            self.model_until.get(model, 0.0), time.time() + secs
        )

    def note_status(self, model: str, status: int, retry_after: Optional[str]) -> None:
        if status == 429:
            stats.record_429(model, retry_after)
            if retry_after and retry_after.isdigit():
                secs = float(retry_after)
            else:
                # No Retry-After header: use what we've *learned* this model's
                # cooldown to be, falling back to the flat default until we know.
                secs = stats.learned_cooldown(model) or _MODEL_COOLDOWN_S
            self.model_until[model] = time.time() + secs
        elif status in (404, 410):
            self.dead.add(model)
        else:
            stats.record_error(model)


# ---- learned rate-limit tracker -------------------------------------------
STATS_PATH = os.environ.get(
    "PROXY_STATS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy_stats.json"),
)
_EWMA_ALPHA = 0.4  # weight on the newest observation
_RPM_WINDOW_S = 60.0  # rolling window for requests-per-minute
# Skip a model if its live rate is at/above this fraction of its learned ceiling.
# 0.85 = give 15% headroom before we expect a throttle.
_THROTTLE_RATIO = 0.85
# Floors under the learned ceilings. A 429 that fires under low traffic (e.g. a
# daily quota, not a rate limit) would otherwise teach limit_rpm≈1 and the
# throttle check would then skip the model at ~1 rpm forever — a death spiral.
# The learned value is kept as-is; the floor applies only when *checking*.
_RPM_LIMIT_FLOOR = float(os.environ.get("PROXY_RPM_LIMIT_FLOOR", "5"))
_TPM_IN_LIMIT_FLOOR = float(os.environ.get("PROXY_TPM_IN_LIMIT_FLOOR", "8000"))
_TPM_OUT_LIMIT_FLOOR = float(os.environ.get("PROXY_TPM_OUT_LIMIT_FLOOR", "2000"))


class Stats:
    """Learns each model's rate-limit behavior from live traffic and persists it.

    For every model it tracks:
      - retry_after_ewma: learned cooldown seconds, from Retry-After headers when
        present, else the measured gap between a 429 and the next success.
      - limit_rpm: learned requests-per-minute ceiling — the rpm observed at the
        moment a 429 fired, EWMA'd so it converges on the real throttle point.
      - peak_rpm / counters for display.
    Saved to proxy_stats.json so the learning survives restarts.
    """

    def __init__(self, data: Optional[dict] = None) -> None:
        self.models: Dict[str, dict] = dict(data or {})
        self._recent: Dict[str, List[float]] = {}
        # Rolling token windows: list of (timestamp, token_count) tuples
        self._recent_tok_in: Dict[str, List[tuple]] = {}
        self._recent_tok_out: Dict[str, List[tuple]] = {}
        self._last_save = 0.0

    def _m(self, model: str) -> dict:
        # setdefault returns the existing dict (from DB or prior calls) or the
        # default.  Existing dicts may be missing newer fields (e.g.
        # limit_tpm_in added in a later version), so patch any that are absent.
        defaults = {
            "requests": 0,
            "successes": 0,
            "rate_limited": 0,
            "errors": 0,
            "last_429": 0.0,
            "last_success": 0.0,
            "retry_after_ewma": 0.0,
            "limit_rpm": None,
            "peak_rpm": 0,
            "limit_tpm_in": None,
            "limit_tpm_out": None,
            "peak_tpm_in": 0,
            "peak_tpm_out": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "tokens_total": 0,
            "last_in": 0,
            "last_out": 0,
        }
        m = self.models.setdefault(model, dict(defaults))
        # Patch any keys the existing dict may be missing (legacy DB data).
        for k, v in defaults.items():
            if k not in m:
                m[k] = v
        return m

    def _rpm(self, model: str, now: float) -> int:
        w = self._recent.setdefault(model, [])
        cutoff = now - _RPM_WINDOW_S
        while w and w[0] < cutoff:
            w.pop(0)
        return len(w)

    def _tpm(self, window: Dict[str, List[tuple]], model: str, now: float) -> float:
        """Sum tokens in the rolling window (per minute)."""
        w = window.setdefault(model, [])
        cutoff = now - _RPM_WINDOW_S
        while w and w[0][0] < cutoff:
            w.pop(0)
        return sum(t[1] for t in w)

    def _snapshot_tpm(self, model: str, now: float) -> tuple:
        """Return (tpm_in, tpm_out) at the current instant."""
        return (
            self._tpm(self._recent_tok_in, model, now),
            self._tpm(self._recent_tok_out, model, now),
        )

    def live_tpm_in(self, model: str) -> float:
        return self._tpm(self._recent_tok_in, model, time.time())

    def live_tpm_out(self, model: str) -> float:
        return self._tpm(self._recent_tok_out, model, time.time())

    def learned_tpm_in(self, model: str) -> Optional[float]:
        m = self.models.get(model)
        if m and m.get("limit_tpm_in") is not None:
            return m["limit_tpm_in"]
        return None

    def learned_tpm_out(self, model: str) -> Optional[float]:
        m = self.models.get(model)
        if m and m.get("limit_tpm_out") is not None:
            return m["limit_tpm_out"]
        return None

    def is_near_limit(self, model: str) -> bool:
        """True if model's current throughput is at/above the learned ceiling.
        Learned ceilings are floored (see _RPM_LIMIT_FLOOR) so a 429 observed
        under low traffic can't teach a ceiling so low the model is never tried."""
        # RPM check
        lim_rpm = self.models.get(model, {}).get("limit_rpm")
        if lim_rpm is not None and self.live_rpm(model) >= (
            max(lim_rpm, _RPM_LIMIT_FLOOR) * _THROTTLE_RATIO
        ):
            return True
        # TPM-in check
        lim_tpi = self.models.get(model, {}).get("limit_tpm_in")
        if lim_tpi is not None and self.live_tpm_in(model) >= (
            max(lim_tpi, _TPM_IN_LIMIT_FLOOR) * _THROTTLE_RATIO
        ):
            return True
        # TPM-out check
        lim_tpo = self.models.get(model, {}).get("limit_tpm_out")
        if lim_tpo is not None and self.live_tpm_out(model) >= (
            max(lim_tpo, _TPM_OUT_LIMIT_FLOOR) * _THROTTLE_RATIO
        ):
            return True
        return False

    def record_request(self, model: str) -> None:
        now = time.time()
        self._recent.setdefault(model, []).append(now)
        m = self._m(model)
        m["requests"] += 1
        m["peak_rpm"] = max(m["peak_rpm"], self._rpm(model, now))

    def record_success(self, model: str) -> None:
        now = time.time()
        m = self._m(model)
        m["successes"] += 1
        # A success right after a 429 reveals the actual cooldown length.
        if (
            m["last_429"]
            and m["last_success"] < m["last_429"]
            and now - m["last_429"] < 3600
        ):
            self._learn_cooldown(m, now - m["last_429"])
        m["last_success"] = now
        self._maybe_save()

    def record_429(self, model: str, retry_after: Optional[str]) -> None:
        now = time.time()
        m = self._m(model)
        m["rate_limited"] += 1
        m["last_429"] = now
        rpm = self._rpm(model, now)
        if rpm > 0:  # rpm at a 429 is at/above the real ceiling — learn toward it
            m["limit_rpm"] = (
                rpm
                if m["limit_rpm"] is None
                else _EWMA_ALPHA * rpm + (1 - _EWMA_ALPHA) * m["limit_rpm"]
            )
        # Snapshot token throughput at the moment of throttle
        tpi, tpo = self._snapshot_tpm(model, now)
        if tpi > 0:
            m["limit_tpm_in"] = (
                tpi
                if m["limit_tpm_in"] is None
                else _EWMA_ALPHA * tpi + (1 - _EWMA_ALPHA) * m["limit_tpm_in"]
            )
        if tpo > 0:
            m["limit_tpm_out"] = (
                tpo
                if m["limit_tpm_out"] is None
                else _EWMA_ALPHA * tpo + (1 - _EWMA_ALPHA) * m["limit_tpm_out"]
            )
        if retry_after and str(retry_after).isdigit():
            self._learn_cooldown(m, float(retry_after))
        self._save()  # always persist rate-limit events

    def record_error(self, model: str) -> None:
        self._m(model)["errors"] += 1

    def record_usage(self, model: str, usage: Optional[dict]) -> None:
        """Accumulate token counts from an OpenAI `usage` block."""
        if not usage:
            return
        m = self._m(model)
        now = time.time()
        pin = int(usage.get("prompt_tokens", 0) or 0)
        pout = int(usage.get("completion_tokens", 0) or 0)
        tot = int(usage.get("total_tokens", pin + pout) or (pin + pout))
        m["tokens_in"] = m.get("tokens_in", 0) + pin
        m["tokens_out"] = m.get("tokens_out", 0) + pout
        m["tokens_total"] = m.get("tokens_total", 0) + tot
        m["last_in"], m["last_out"] = pin, pout
        # Feed rolling token windows for TPM tracking
        self._recent_tok_in.setdefault(model, []).append((now, pin))
        self._recent_tok_out.setdefault(model, []).append((now, pout))
        tpi = self.live_tpm_in(model)
        tpo = self.live_tpm_out(model)
        m["peak_tpm_in"] = max(m.get("peak_tpm_in", 0), tpi)
        m["peak_tpm_out"] = max(m.get("peak_tpm_out", 0), tpo)
        self._maybe_save()

    def _learn_cooldown(self, m: dict, secs: float) -> None:
        if secs <= 0:
            return
        m["retry_after_ewma"] = (
            secs
            if m["retry_after_ewma"] <= 0
            else _EWMA_ALPHA * secs + (1 - _EWMA_ALPHA) * m["retry_after_ewma"]
        )

    def learned_cooldown(self, model: str) -> Optional[float]:
        m = self.models.get(model)
        if m and m.get("retry_after_ewma", 0) > 0:
            return m["retry_after_ewma"]
        return None

    def live_rpm(self, model: str) -> int:
        return self._rpm(model, time.time())

    def _maybe_save(self) -> None:
        if time.time() - self._last_save > 10:
            self._save()

    def _save(self) -> None:
        self._last_save = time.time()
        try:
            conn = _db()
            try:
                conn.executemany(
                    "INSERT INTO stats(model, data) VALUES(?, ?) "
                    "ON CONFLICT(model) DO UPDATE SET data=excluded.data",
                    [(m, json.dumps(d)) for m, d in self.models.items()],
                )
                conn.commit()
            finally:
                conn.close()
            _db_secure()
        except sqlite3.Error:
            pass

    @classmethod
    def load(cls) -> "Stats":
        try:
            conn = _db()
            try:
                rows = list(conn.execute("SELECT model, data FROM stats"))
                if not rows and os.path.exists(STATS_PATH):
                    # One-time migration from the legacy JSON stats file.
                    inst = cls(_load_json_stats())
                    inst._save()
                    return inst
                return cls({m: json.loads(d) for m, d in rows})
            finally:
                conn.close()
        except (sqlite3.Error, ValueError):
            return cls()


def _load_json_stats() -> dict:
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


cascade = Cascade()
stats = Stats.load()
app = FastAPI(title="nvidia-failover-proxy")


def _serving_ladder() -> List[str]:
    """The cloud models the cascade will actually try, in order. Routing uses
    the user-configured ladder (web UI / SQLite); the built-in env ladder is
    only a display fallback for a fresh, unconfigured install."""
    return ladder_config.active_ladder() or list(cascade.models)


def _headers() -> Dict[str, str]:
    key = ladder_config.resolved_nvidia_key()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _route(model: str) -> tuple:
    """(base_url, headers) for a model — local Ollama needs no auth, custom
    providers use their own base_url + key, everything else goes to NVIDIA."""
    if cascade.is_local(model):
        return cascade._ollama_base_url(), {"Content-Type": "application/json"}
    prov = ladder_config.model_provider(model)
    if prov:
        base_url, key = prov
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return base_url, headers
    return NVIDIA_BASE_URL, _headers()


# ---------------------------------------------------------------------------
# Prompt refiner — calls a tiny local model to rewrite user prompts before the
# main model sees them. Activated by including `[refine]` anywhere in messages.
# ---------------------------------------------------------------------------
REFINER_SYSTEM = (
    "You are a prompt engineering assistant. Rewrite the user's request into a "
    "clear, structured, highly-specific prompt that will get the best possible "
    "result from a large language model. Preserve the original intent. Add "
    "context, break it into steps, and be explicit about the desired output "
    "format. Output ONLY the rewritten prompt — no explanations, no greetings."
)


def _has_refine_tag(messages: List[dict]) -> bool:
    """Check if any user message contains `[refine]` anywhere (in any msg)."""
    if not REFINER_ENABLE:
        return False
    tag_lower = REFINER_TAG.lower()
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            if tag_lower in msg["content"].lower():
                return True
    return False


def _strip_refine_tag(text: str) -> str:
    """Remove all occurrences of the refine tag (case-insensitive)."""
    import re

    return re.sub(re.escape(REFINER_TAG), "", text, flags=re.IGNORECASE).strip()


async def _refine_prompt(messages: List[dict], timeout: httpx.Timeout) -> List[dict]:
    """Send messages to the local refiner model and return improved messages.

    The refiner sees the full conversation and produces an improved version of
    the last user message. Returns a copy of messages with the last user message
    replaced by the refiner's output (tag stripped from both).
    """
    out = list(messages)
    # Find the last user message index
    last_user_idx = -1
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        return out  # no user message to refine

    original = out[last_user_idx]
    text = original.get("content", "")
    if not isinstance(text, str) or not text.strip():
        return out

    # Strip tag from the outgoing message
    cleaned = _strip_refine_tag(text)
    out[last_user_idx] = {**original, "content": cleaned}

    # Call the refiner
    refiner_body = {
        "model": REFINER_MODEL,
        "messages": [
            {"role": "system", "content": REFINER_SYSTEM},
            {"role": "user", "content": cleaned},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
        "stream": False,
    }
    print(f"[refiner] calling {REFINER_MODEL} with {len(cleaned)} chars...")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Warmup: unload whatever is in VRAM by loading the refiner model
            # with keep_alive=0 (loads then immediately unloads). This forces
            # Ollama to evict any other model from GPU memory first.
            try:
                await client.post(
                    f"{_strip_v1(REFINER_BASE_URL)}/api/generate",
                    json={"model": REFINER_MODEL, "prompt": "", "keep_alive": 0},
                    timeout=httpx.Timeout(connect=10.0, read=10.0, write=5.0),
                )
            except Exception:
                pass  # warmup failure is non-fatal; the real call below will still try

            resp = await client.post(
                f"{REFINER_BASE_URL}/chat/completions",
                json=refiner_body,
            )
        if resp.status_code == 200:
            data = resp.json()
            improved = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if improved:
                print(f"[refiner] improved from {len(cleaned)} → {len(improved)} chars")
                out[last_user_idx] = {**out[last_user_idx], "content": improved}
            else:
                print("[refiner] empty response, keeping original")
        else:
            print(f"[refiner] HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[refiner] error: {type(e).__name__}: {e}")
    return out


# Fail over when a model returns a 200 but no usable output (NVIDIA occasionally
# returns an empty completion / empty stream). Tool-call responses legitimately
# carry empty content, so those count as "has content".
SKIP_EMPTY = os.environ.get("PROXY_SKIP_EMPTY", "1").lower() not in (
    "0",
    "false",
    "no",
    "",
)


def _msg_has_content(msg: dict) -> bool:
    if (msg.get("content") or "").strip():
        return True
    if msg.get("tool_calls"):
        return True
    return False


def _resp_has_content(data: dict) -> bool:
    choices = data.get("choices") or []
    if not choices:
        return False
    return _msg_has_content(choices[0].get("message") or {})


def _delta_has_content(obj: dict) -> bool:
    choices = obj.get("choices") or []
    if not choices:
        return False
    d = choices[0].get("delta") or {}
    if (d.get("content") or "").strip():
        return True
    if d.get("tool_calls"):
        return True
    return False


# --- Degenerate-output guard -------------------------------------------------
# Detect the kimi-k2.6 (and similar) failure mode: a repetition loop that
# code-switches to CJK. Used to fail over on the non-stream path and to abort a
# fresh (not-yet-committed) stream before garbage reaches the client.
GUARD_DEGENERATE = os.environ.get("PROXY_GUARD_DEGENERATE", "1").lower() not in (
    "0", "false", "no", "",
)
# A degenerate model code-switches to CJK. We only guard when the *prompt* had
# (essentially) no CJK, so genuine Chinese/Japanese/Korean requests are never
# touched. In that case ANY meaningful amount of CJK in the answer is drift, so
# we trip on a small absolute count rather than a fraction — kimi interleaves
# Chinese with Latin/code, which defeats run- or fraction-based thresholds.
_CJK_MIN_CHARS = int(os.environ.get("PROXY_CJK_MIN_CHARS", "4"))

# Anti-hang watchdog. httpx's read timeout resets on *any* byte, so an upstream
# that stalls after partial output while still trickling SSE keepalives (or that
# goes silent under our per-read `read` timeout window) will hang the stream with
# no new tokens ever arriving — the classic "model just stops mid-output". We
# additionally track wall-clock time since the last *content-bearing* delta and
# treat too long a gap as a stall: fail over if nothing was delivered yet, else
# end the stream cleanly so the client isn't left waiting forever.
_STREAM_STALL_S = float(os.environ.get("PROXY_STREAM_STALL_S", "30"))


class _StreamStall(Exception):
    """Raised when an upstream stream produces no new content for too long."""


def _cjk_count(s: str) -> int:
    # CJK Unified + common Japanese kana ranges; enough to catch a code-switch.
    return sum(
        1 for ch in s
        if "一" <= ch <= "鿿"
        or "぀" <= ch <= "ヿ"
        or "가" <= ch <= "힣"
    )


def _prompt_has_cjk(body: dict) -> bool:
    # Only user/system messages express the user's intent. Assistant history is
    # excluded on purpose: a prior degenerate CJK reply in the conversation must
    # not disarm the guard for every following turn.
    for m in body.get("messages") or []:
        if m.get("role") not in ("user", "system"):
            continue
        c = m.get("content")
        if isinstance(c, str) and _cjk_count(c) >= 3:
            return True
        if isinstance(c, list):  # multimodal content parts
            for part in c:
                if isinstance(part, dict) and _cjk_count(part.get("text") or "") >= 3:
                    return True
    return False


def _unexpected_cjk(text: str, body: dict) -> bool:
    """True if the response is heavily CJK but the prompt wasn't (a code-switch)."""
    if not text or _prompt_has_cjk(body):
        return False
    # Absolute count only: kimi interleaves Chinese with Latin/code when it
    # degenerates, so a run- or fraction-based test lets large amounts through.
    return _cjk_count(text) >= _CJK_MIN_CHARS


def _delta_text(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


def _degenerate_reason(text: str, finish_reason, body: dict):
    """Return a short reason string if this looks degenerate, else None."""
    if not GUARD_DEGENERATE:
        return None
    if finish_reason == "repetition":
        return "finish_reason=repetition"
    if _unexpected_cjk(text or "", body):
        return "unexpected CJK code-switch"
    return None


def _prep_body(body: dict, model: str) -> dict:
    out = dict(body)
    out["model"] = model
    # NIM quirk: some models return empty content unless max_tokens is set.
    # Client-supplied value always wins; see _DEFAULT_MAX_TOKENS for the rationale.
    out.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
    # Repetition guard: inject a mild frequency_penalty for models prone to
    # degenerate loops (e.g. kimi-k2.6 → Chinese). Client value always wins.
    if "frequency_penalty" not in out:
        fp = _freq_penalty_for(model)
        if fp:
            out["frequency_penalty"] = fp
    # Streaming: ask for a final usage chunk so we can track tokens per model.
    if out.get("stream"):
        so = dict(out.get("stream_options") or {})
        so.setdefault("include_usage", True)
        out["stream_options"] = so
    return out


@app.get("/health")
async def health() -> dict:
    key = ladder_config.resolved_nvidia_key()
    # Healthy = any configured provider (keys live per-provider now), or the
    # legacy NVIDIA key for installs that pre-date providers.
    agent_descs = {k: v.split(".")[0][:60] for k, v in AGENT_ROLES.items()}
    return {
        "ok": bool(ladder_config.providers) or bool(key),
        "modes": {
            AUTO_MODEL: "cloud ladder then local 80B",
            ONLY_MODEL: "cloud ladder only, 429 when all cooling",
            REFINER_MODEL_ID: f"cloud ladder + local {REFINER_MODEL} prompt refiner",
            LOCAL_ONLY: "local 80B only, no cloud",
            LOCAL_REFINE: f"[refine]→{REFINER_MODEL} → local 80B",
            **agent_descs,
        },
        "models": _serving_ladder(),
        "local_fallback": cascade._local_model(),
        "refiner": {
            "enabled": REFINER_ENABLE,
            "model": REFINER_MODEL,
            "tag": REFINER_TAG,
        },
        "cooling": {
            m: int(t - time.time())
            for m, t in cascade.model_until.items()
            if t > time.time()
        },
        "dead": sorted(cascade.dead),
    }


def _model_view() -> List[dict]:
    """Per-model snapshot joining live cascade state with learned stats."""
    now = time.time()
    rows: List[dict] = []
    # Follow the user-defined failover order from the web UI so the metrics table
    # matches the config panel; disabled models are hidden; local tail stays last.
    local_model = cascade._local_model()
    names = [m for m in _known_models() if ladder_config.is_enabled(m)]
    if local_model and local_model not in names:
        names.append(local_model)
    # The "active" model is the first live rung in ladder order — the one that
    # serves the next request. Mark it so the dashboard can show a green dot.
    active_seen = False
    for name in names:
        m = stats.models.get(name, {})
        until = cascade.model_until.get(name, 0.0)
        cooling = max(0, int(until - now)) if until > now else 0
        if cascade.is_local(name):
            state = "local"
        elif name in cascade.dead:
            state = "dead"
        elif cooling:
            state = "cooling"
        else:
            state = "live"
        is_active = state == "live" and not active_seen
        if is_active:
            active_seen = True
        rows.append(
            {
                "model": name,
                "state": state,
                "active": is_active,
                "cooling_s": cooling,
                "requests": m.get("requests", 0),
                "successes": m.get("successes", 0),
                "rate_limited": m.get("rate_limited", 0),
                "errors": m.get("errors", 0),
                "live_rpm": stats.live_rpm(name),
                "peak_rpm": m.get("peak_rpm", 0),
                "learned_limit_rpm": (
                    round(m["limit_rpm"], 1) if m.get("limit_rpm") is not None else None
                ),
                "live_tpm_in": round(stats.live_tpm_in(name), 0),
                "live_tpm_out": round(stats.live_tpm_out(name), 0),
                "peak_tpm_in": m.get("peak_tpm_in", 0),
                "peak_tpm_out": m.get("peak_tpm_out", 0),
                "learned_limit_tpm_in": (
                    round(m["limit_tpm_in"], 0)
                    if m.get("limit_tpm_in") is not None
                    else None
                ),
                "learned_limit_tpm_out": (
                    round(m["limit_tpm_out"], 0)
                    if m.get("limit_tpm_out") is not None
                    else None
                ),
                "learned_cooldown_s": (
                    round(m["retry_after_ewma"], 1)
                    if m.get("retry_after_ewma", 0) > 0
                    else None
                ),
                "tokens_in": m.get("tokens_in", 0),
                "tokens_out": m.get("tokens_out", 0),
                "tokens_total": m.get("tokens_total", 0),
                "last_in": m.get("last_in", 0),
                "last_out": m.get("last_out", 0),
                "saved_usd": _saved_usd(
                    name, m.get("tokens_in", 0), m.get("tokens_out", 0)
                ),
            }
        )
    return rows


@app.get("/stats")
async def stats_json() -> dict:
    models = _model_view()
    return {
        "models": models,
        "window_s": int(_RPM_WINDOW_S),
        "db_file": DB_FILE,
        "saved_usd_total": round(sum(m.get("saved_usd", 0.0) for m in models), 4),
    }


def _fmt_tpm(val) -> str:
    """Format a TPM number: integer with commas (0 is valid, show it)."""
    return f"{int(val):,}"


def _fmt_ceiling(val) -> str:
    """Format a learned ceiling: human-readable int, or 'learning…'."""
    return f"{int(val):,}/min" if val is not None else "learning…"


def _totals_row(rows: list, tot: dict) -> str:
    """The TOTAL <tr> — kept inside <tbody> so SSE refreshes it live."""
    return (
        f"<tr class=tot><td></td><td>TOTAL ({len(rows)} models)</td><td></td><td></td>"
        f"<td>{tot['requests']}</td><td>{tot['successes']}</td><td>{tot['rate_limited']}</td>"
        f"<td></td><td></td><td></td><td></td><td></td><td></td><td></td>"
        f"<td class=num>{_fmt_num(tot['tokens_in'])}</td><td class=num>{_fmt_num(tot['tokens_out'])}</td>"
        f"<td class=num>{_fmt_num(tot['tokens_total'])}</td>"
        f"<td class=num><span class=save>{_fmt_money(tot['saved_usd'])}</span></td></tr>"
    )


async def _live_tbody() -> str:
    """HTML <tr>...</tr> rows for the current stats snapshot, for SSE."""
    rows = _model_view()
    color = {
        "live": "#2e7d32",
        "cooling": "#e65100",
        "dead": "#b71c1c",
        "local": "#1565c0",
        "disabled": "#5b6472",
    }
    tot = {
        "requests": 0,
        "successes": 0,
        "rate_limited": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_total": 0,
        "saved_usd": 0.0,
    }
    trs = []
    for i, r in enumerate(rows):
        for k in tot:
            tot[k] += r.get(k, 0)
        avail = (
            "now"
            if r["state"] == "live"
            else (
                _fmt_dur(r["cooling_s"])
                if r["state"] == "cooling"
                else (
                    "last rung"
                    if r["state"] == "local"
                    else ("off" if r["state"] == "disabled" else "dropped")
                )
            )
        )
        limit_rpm = (
            f"{r['learned_limit_rpm']}/min"
            if r["learned_limit_rpm"] is not None
            else "learning…"
        )
        cd = (
            _fmt_dur(r["learned_cooldown_s"])
            if r["learned_cooldown_s"]
            else "learning…"
        )
        badge = f'<span style="color:{color.get(r["state"], "#555")};font-weight:600">{r["state"].upper()}</span>'
        tin = (
            f"{_fmt_num(r['tokens_in'])}<span class=dim> (+{r['last_in']})</span>"
            if r["tokens_in"]
            else "—"
        )
        tout = (
            f"{_fmt_num(r['tokens_out'])}<span class=dim> (+{r['last_out']})</span>"
            if r["tokens_out"]
            else "—"
        )
        tpi = f"{_fmt_tpm(r['live_tpm_in'])} <span class=dim>(peak {_fmt_tpm(r['peak_tpm_in'])})</span>"
        tpo = f"{_fmt_tpm(r['live_tpm_out'])} <span class=dim>(peak {_fmt_tpm(r['peak_tpm_out'])})</span>"
        saved = f'<span class=save>{_fmt_money(r["saved_usd"])}</span>' if r["saved_usd"] else "—"
        trs.append(
            f"<tr><td class=n>{i + 1}</td><td class=m>{_ACTIVE_DOT if r.get('active') else ''}{r['model']}</td><td>{badge}</td>"
            f"<td>{avail}</td><td>{r['requests']}</td><td>{r['successes']}</td>"
            f"<td>{r['rate_limited']}</td><td>{r['live_rpm']} <span class=dim>(peak {r['peak_rpm']})</span></td>"
            f"<td>{limit_rpm}</td><td>{_fmt_ceiling(r['learned_limit_tpm_in'])}</td>"
            f"<td>{_fmt_ceiling(r['learned_limit_tpm_out'])}</td><td>{tpi}</td><td>{tpo}</td>"
            f"<td>{cd}</td>"
            f"<td class=num>{tin}</td><td class=num>{tout}</td><td class=num>{_fmt_num(r['tokens_total'])}</td>"
            f"<td class=num>{saved}</td></tr>"
        )
    return "".join(trs) + _totals_row(rows, tot)


@app.get("/updates")
async def updates(request: Request):
    """Server-sent events: pushes tbody HTML every 0.5 seconds."""

    async def watch_stats():
        # Honor client disconnect so closed dashboard tabs don't leak a coroutine
        # and, crucially, so `systemctl restart` doesn't hang ~90s waiting for
        # these long-lived connections to drain (which briefly makes the proxy
        # unreachable and drops in-flight OpenCode requests).
        while not await request.is_disconnected():
            await asyncio.sleep(0.5)  # sub-second push for snappy feel
            tbody = await _live_tbody()
            yield f"data: {sse_escape(tbody)}\n\n"

    return StreamingResponse(
        watch_stats(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        background=None,
    )


def _fmt_dur(secs) -> str:
    if not secs:
        return "—"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    return f"{m}m{s:02d}s"


def _fmt_num(n) -> str:
    return f"{int(n):,}" if n else "—"


# Pulsing green dot shown next to the active (head-of-ladder, live) model.
_ACTIVE_DOT = '<span class="dot" title="active model — serves the next request"></span>'


# Stylized green-on-black "eye" mark (NVIDIA brand colors, #76b900) — used as the
# browser-tab favicon and the header logo on the dashboard.
_NV_LOGO_PNG = "data:image/png;base64,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"
_NV_FAVICON = "data:image/png;base64,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"
# The official NVIDIA logo, used as the app's own branding (header + favicon).
_NV_LOGO_SVG = f'<img class="nvlogo" alt="NVIDIA" src="{_NV_LOGO_PNG}">'


# --- Provider brand icons + presets ------------------------------------------
# Small inline SVG marks shown next to each provider in the settings UI. Brand
# colors with a simple recognizable glyph. provider_icon() matches by name.
def _badge(bg: str, inner: str) -> str:
    return (
        '<svg class=pvicon viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="24" height="24" rx="6" fill="{bg}"/>{inner}</svg>'
    )


def _mono(bg: str, glyph: str, fill: str = "#fff", size: int = 12) -> str:
    return _badge(
        bg,
        f'<text x="12" y="16.5" font-family="system-ui,Segoe UI,sans-serif" '
        f'font-size="{size}" font-weight="700" fill="{fill}" '
        f'text-anchor="middle">{glyph}</text>',
    )


# OpenAI-style six-spoke asterisk mark.
_OPENAI_MARK = _badge(
    "#0b0d10",
    '<g stroke="#fff" stroke-width="2.2" stroke-linecap="round">'
    '<line x1="12" y1="6" x2="12" y2="18"/>'
    '<line x1="6.8" y1="9" x2="17.2" y2="15"/>'
    '<line x1="6.8" y1="15" x2="17.2" y2="9"/></g>',
)
# NVIDIA eye, badge-sized.
_NVIDIA_MARK = _badge(
    "#0b0d10",
    '<path fill="#76b900" d="M7 12 C7 8.5 10 6.5 13.6 7 '
    "C10.8 8.2 9.6 10.4 10 13 C10.4 16 12.8 17.2 16 16.2 "
    'C13.4 18.8 9 18.8 6.8 16 C5.4 14.6 5.6 12.6 7 12 Z"/>'
    '<circle cx="16" cy="9.4" r="1.5" fill="#76b900"/>',
)

PROVIDER_ICONS: Dict[str, str] = {
    "nvidia": _NVIDIA_MARK,
    "openai": _OPENAI_MARK,
    "codex": _badge(
        "#0b0d10",
        '<text x="12" y="16" font-family="ui-monospace,Consolas,monospace" '
        'font-size="10" font-weight="700" fill="#10a37f" '
        'text-anchor="middle">&gt;_</text>',
    ),
    "anthropic": _mono("#d97757", "A"),
    "claude": _mono("#d97757", "A"),
    "cerebras": _mono("#f26722", "C"),
    "groq": _mono("#f55036", "G"),
    "openrouter": _mono("#6467f2", "OR", size=9),
    "mistral": _mono("#fa5310", "M"),
    "deepseek": _mono("#4d6bfe", "D"),
    "google": _mono("#1a73e8", "G"),
    "gemini": _mono("#1a73e8", "G"),
    "xai": _mono("#0b0d10", "X"),
    "grok": _mono("#0b0d10", "X"),
    "together": _mono("#0f6fff", "T"),
    "perplexity": _mono("#20808d", "P"),
    "fireworks": _mono("#5019c5", "F"),
    "ollama": _mono("#1f2937", "O"),
    "local": _mono("#334155", "&#8962;"),
}
_ICON_FALLBACK_BG = "#3a4150"


def provider_icon(name: str) -> str:
    """Best-effort brand mark for a provider name (substring match, else a
    gray monogram badge of the first letter)."""
    key = (name or "").strip().lower()
    if key in PROVIDER_ICONS:
        return PROVIDER_ICONS[key]
    for k, svg in PROVIDER_ICONS.items():
        if k in key:
            return svg
    letter = (key[:1] or "?").upper()
    return _mono(_ICON_FALLBACK_BG, letter)


# One-click provider presets for the settings UI: name -> OpenAI-compatible base
# URL. Selecting one prefills the add-provider form (still needs an API key).
PROVIDER_PRESETS: List[Dict[str, str]] = [
    {"name": "nvidia", "base_url": NVIDIA_BASE_URL},
    {"name": "openai", "base_url": "https://api.openai.com/v1"},
    {"name": "codex", "base_url": "https://api.openai.com/v1"},
    {"name": "anthropic", "base_url": "https://api.anthropic.com/v1"},
    {"name": "cerebras", "base_url": "https://api.cerebras.ai/v1"},
    {"name": "groq", "base_url": "https://api.groq.com/openai/v1"},
    {"name": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
    {"name": "mistral", "base_url": "https://api.mistral.ai/v1"},
    {"name": "deepseek", "base_url": "https://api.deepseek.com/v1"},
    {
        "name": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
    {"name": "xai", "base_url": "https://api.x.ai/v1"},
    {"name": "together", "base_url": "https://api.together.xyz/v1"},
    {"name": "ollama", "base_url": "http://127.0.0.1:11434/v1"},
]


CFG_SCRIPT = """<script>
(function(){
  var list=document.getElementById("cfglist");
  if(!list) return;
  var dragEl=null;
  list.addEventListener("dragstart",function(e){
    dragEl=e.target.closest(".cfgrow");
    if(dragEl) dragEl.classList.add("dragging");
  });
  list.addEventListener("dragend",function(){
    if(dragEl) dragEl.classList.remove("dragging");
    dragEl=null;
  });
  list.addEventListener("dragover",function(e){
    e.preventDefault();
    if(!dragEl) return;
    var after=getAfter(list,e.clientY);
    if(after==null) list.appendChild(dragEl);
    else list.insertBefore(dragEl,after);
  });
  function getAfter(container,y){
    var els=[].slice.call(container.querySelectorAll(".cfgrow:not(.dragging)"));
    var closest={offset:-Infinity,el:null};
    els.forEach(function(c){
      var box=c.getBoundingClientRect();
      var off=y-box.top-box.height/2;
      if(off<0&&off>closest.offset){closest={offset:off,el:c};}
    });
    return closest.el;
  }
  function collect(){
    var order=[],disabled=[];
    [].slice.call(list.querySelectorAll(".cfgrow")).forEach(function(r){
      var m=r.getAttribute("data-model");
      order.push(m);
      if(!r.querySelector(".tog").checked) disabled.push(m);
    });
    return {order:order,disabled:disabled};
  }
  var st=document.getElementById("cfgstatus");
  document.getElementById("cfgsave").addEventListener("click",function(){
    st.textContent="saving…";
    fetch("/_config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collect())})
      .then(function(r){return r.json();})
      .then(function(d){ st.textContent=d.ok?("saved — "+d.active.length+" active model(s)"):("error: "+(d.error||"?")); })
      .catch(function(){ st.textContent="save failed"; });
  });
  document.getElementById("cfgreset").addEventListener("click",function(){ location.reload(); });
})();
</script>"""


SET_SCRIPT = """<script>
(function(){
  var st=document.getElementById("setstatus");
  function esc(s){ return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}[c];}); }
  function say(m){ if(st) st.textContent=m; }

  function addLadderRow(model){
    var list=document.getElementById("cfglist");
    if(!list) return false;
    var have=[].slice.call(list.querySelectorAll(".cfgrow")).some(function(r){return r.getAttribute("data-model")===model;});
    if(have) return false;
    var li=document.createElement("li");
    li.className="cfgrow"; li.setAttribute("draggable","true"); li.setAttribute("data-model",model);
    li.innerHTML='<span class="grip">&#8942;&#8942;</span><input type="checkbox" class="tog" checked><span class="cfgname">'+esc(model)+'</span>';
    list.appendChild(li);
    return true;
  }

  var ICONS={};
  function renderSettings(d){
    (d.providers||[]).forEach(function(p){ ICONS[p.name]=p.icon; });
    (d.presets||[]).forEach(function(p){ if(!ICONS[p.name]) ICONS[p.name]=p.icon; });
    // Preset quick-add chips — click to prefill name + base URL.
    var pr=document.getElementById("presetrow");
    if(pr && !pr.dataset.done){
      pr.dataset.done="1";
      (d.presets||[]).forEach(function(p){
        var c=document.createElement("button"); c.type="button"; c.className="presetchip";
        c.innerHTML=p.icon+"<span>"+esc(p.name)+"</span>";
        c.addEventListener("click",function(){
          document.getElementById("pvname").value=p.name;
          document.getElementById("pvurl").value=p.base_url;
          document.getElementById("pvkey").focus();
          say("prefilled "+p.name+" — add your API key + model ids");
        });
        pr.appendChild(c);
      });
    }
    var ul=document.getElementById("provlist");
    if(ul){
      ul.innerHTML="";
      if(!d.providers.length){ ul.innerHTML='<li class="pvmeta">no providers yet — pick one above or add a custom API</li>'; }
      d.providers.forEach(function(p){
        var li=document.createElement("li");
        li.innerHTML=(p.icon||"")+'<b>'+esc(p.name)+'</b> <span class="pvmeta">'+esc(p.base_url)+' · key '+(p.key_masked?esc(p.key_masked):"none")+' · '+p.models.length+' model(s)</span>';
        var b=document.createElement("button"); b.className="rm"; b.textContent="remove";
        b.addEventListener("click",function(){ post({action:"remove_provider",name:p.name}); });
        li.appendChild(b); ul.appendChild(li);
      });
    }
    // Local tail model dropdown — populate from ollama provider models.
    var sel=document.getElementById("tailsel");
    if(sel && d.local_tail){
      sel.innerHTML="";
      var opts=d.local_tail.options||[];
      if(!opts.length){
        var o=document.createElement("option"); o.textContent="(no Ollama models)"; o.disabled=true; sel.appendChild(o);
      } else {
        opts.forEach(function(m){
          var o=document.createElement("option"); o.value=m; o.textContent=m;
          if(m===d.local_tail.current) o.selected=true;
          sel.appendChild(o);
        });
      }
    }
  }

  function load(){ fetch("/_settings").then(function(r){return r.json();}).then(renderSettings).catch(function(){}); }
  function post(payload){
    say("saving…");
    return fetch("/_settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)})
      .then(function(r){return r.json();})
      .then(function(d){ if(d.error){say("error: "+d.error);} else {say("saved"); renderSettings(d);} return d; })
      .catch(function(){ say("request failed"); });
  }

  var pa=document.getElementById("pvadd");
  if(pa) pa.addEventListener("click",function(){
    var name=document.getElementById("pvname").value.trim();
    var url=document.getElementById("pvurl").value.trim();
    var key=document.getElementById("pvkey").value;
    var models=document.getElementById("pvmodels").value;
    if(!name||!url){ say("name and base URL required"); return; }
    post({action:"add_provider",name:name,base_url:url,api_key:key,models:models}).then(function(d){
      if(d&&!d.error){
        (models.split(",")).forEach(function(m){ m=m.trim(); if(m) addLadderRow(m); });
        say("provider added — models appended to ladder");
        document.getElementById("pvname").value=""; document.getElementById("pvurl").value="";
        document.getElementById("pvkey").value=""; document.getElementById("pvmodels").value="";
      }
    });
  });

  var ts=document.getElementById("tailsave");
  if(ts) ts.addEventListener("click",function(){
    var sel=document.getElementById("tailsel");
    if(!sel) return;
    var model=sel.value||"";
    post({action:"set_local_tail",model:model}).then(function(d){
      if(d&&!d.error) say("local tail model updated to: "+model);
    });
  });

  var discData=null, discHave={};
  function renderDisc(){
    var box=document.getElementById("discresult");
    if(!box||!discData) return;
    var pv=discData.providers||{};
    var names=Object.keys(pv);
    box.innerHTML="";
    if(!names.length){ box.appendChild(document.createTextNode("no providers configured — add one above")); return; }
    var q=(document.getElementById("discfilter")||{}).value||"";
    q=q.trim().toLowerCase();
    var sortMode=(document.getElementById("discsort")||{}).value||"name-asc";
    names.forEach(function(name){
      var res=pv[name]||{};
      var models=(res.models||[]).slice();
      if(q) models=models.filter(function(m){ return m.toLowerCase().indexOf(q)>=0; });
      if(sortMode==="name-asc") models.sort();
      else if(sortMode==="name-desc") models.sort().reverse();
      else if(sortMode==="new-first"){ models.sort(); models.reverse(); models.sort(function(a,b){ return (discHave[b]?1:0)-(discHave[a]?1:0); }); }
      // Hide a provider entirely when a filter excludes all of its models.
      if(q && !models.length && !res.error) return;
      var det=document.createElement("details"); det.className="discprov"; det.open=true;
      var sum=document.createElement("summary");
      var total=(res.models||[]).length;
      var label=q?(models.length+"/"+total):(""+total);
      sum.innerHTML=(ICONS[name]||"")+" <b>"+esc(name)+"</b> "+(res.error?('<span class="pvmeta">error: '+esc(res.error)+'</span>'):('<span class="pvmeta">'+label+' model(s)</span>'));
      det.appendChild(sum);
      var body=document.createElement("div"); body.className="discbody";
      var addable=models.filter(function(m){ return !discHave[m]; });
      if(addable.length>1){
        var all=document.createElement("button"); all.type="button"; all.className="addall";
        all.textContent="＋ add all "+addable.length+" shown";
        all.addEventListener("click",function(){ var n=0; addable.forEach(function(m){ if(addLadderRow(m)){ discHave[m]=1; n++; } }); say("added "+n+' model(s) — hit "Save order & toggles"'); renderDisc(); });
        body.appendChild(all);
      }
      models.forEach(function(m){
        var chip=document.createElement("span"); chip.className="mdlchip"+(discHave[m]?" have":"");
        chip.appendChild(document.createTextNode(m));
        if(!discHave[m]){
          var add=document.createElement("button"); add.textContent="＋";
          add.addEventListener("click",function(){ if(addLadderRow(m)){ discHave[m]=1; chip.className="mdlchip have"; add.remove(); say("added "+m+' — hit "Save order & toggles"'); } });
          chip.appendChild(add);
        }
        body.appendChild(chip);
      });
      det.appendChild(body);
      box.appendChild(det);
    });
  }

  var db=document.getElementById("discbtn");
  if(db) db.addEventListener("click",function(){
    var box=document.getElementById("discresult");
    box.textContent="querying providers…";
    var tb=document.getElementById("disctools"); if(tb) tb.style.display="none";
    fetch("/_models/available").then(function(r){return r.json();}).then(function(d){
      discData=d; discHave={}; (d.in_ladder||[]).forEach(function(m){discHave[m]=1;});
      if(tb){ tb.style.display=""; }
      renderDisc();
    }).catch(function(){ box.textContent="discovery failed"; });
  });
  var df=document.getElementById("discfilter"); if(df) df.addEventListener("input",renderDisc);
  var ds=document.getElementById("discsort"); if(ds) ds.addEventListener("change",renderDisc);

  load();
})();
</script>"""


def _known_models() -> List[str]:
    """Saved user order, with any provider models not yet in the order appended
    so the config UI shows every configured model. Empty when no providers.

    Ollama-provider models are excluded — they serve as the local tail rung
    (handled by Cascade._local_model()) and are not part of the failover order
    unless the user explicitly adds them via the UI."""
    ollama_models = set(ladder_config.providers.get("ollama", {}).get("models", []))
    known = list(ladder_config.order)
    for m in ladder_config.custom_models():
        if m not in known and m not in ollama_models:
            known.append(m)
    return known


@app.get("/_config")
async def get_config():
    """Current failover order + disabled set for the web UI."""
    return JSONResponse(
        {
            "order": _known_models(),
            "disabled": sorted(ladder_config.disabled),
        }
    )


@app.post("/_config")
async def post_config(request: Request):
    """Persist a user-defined failover order and/or disabled model set."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    order = body.get("order")
    disabled = body.get("disabled")
    if order is not None and (
        not isinstance(order, list) or not all(isinstance(m, str) for m in order)
    ):
        return JSONResponse(
            {"error": "order must be a list of strings"}, status_code=400
        )
    if disabled is not None and (
        not isinstance(disabled, list) or not all(isinstance(m, str) for m in disabled)
    ):
        return JSONResponse(
            {"error": "disabled must be a list of strings"}, status_code=400
        )
    ladder_config.update(order=order, disabled=disabled)
    return JSONResponse(
        {
            "ok": True,
            "order": ladder_config.order,
            "disabled": sorted(ladder_config.disabled),
            "active": ladder_config.active_ladder(),
        }
    )


def _mask_key(key: Optional[str]) -> str:
    if not key:
        return ""
    return (key[:6] + "…" + key[-4:]) if len(key) > 12 else "•" * len(key)


def _ollama_live_models() -> List[str]:
    """Fetch model names from the Ollama native /api/tags endpoint.

    Falls back to the provider's configured model list on any error so the
    UI dropdown is never empty when Ollama is temporarily unreachable."""
    p = ladder_config.providers.get("ollama", {})
    base = _strip_v1(p.get("base_url", ""))
    if not base:
        return list(p.get("models", []))
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return list(p.get("models", []))
        data = resp.json()
        names = sorted(
            name
            for m in data.get("models", [])
            if isinstance(m, dict) and (name := m.get("name"))
        )
        return names or list(p.get("models", []))
    except (httpx.RequestError, ValueError, KeyError):
        return list(p.get("models", []))


def _settings_view() -> dict:
    """Providers + key status + local tail config for the UI — never returns
    raw keys.

    The local tail model dropdown is populated from the Ollama native API so
    the user can pick any model currently available in Ollama, not just the
    ones pre-configured in the provider."""
    ollama_models = _ollama_live_models()
    return {
        "providers": [
            {
                "name": name,
                "base_url": p.get("base_url", ""),
                "models": p.get("models", []),
                "key_masked": _mask_key(p.get("api_key")),
                "icon": provider_icon(name),
            }
            for name, p in ladder_config.providers.items()
        ],
        "presets": [
            {
                "name": p["name"],
                "base_url": p["base_url"],
                "icon": provider_icon(p["name"]),
            }
            for p in PROVIDER_PRESETS
        ],
        "local_tail": {
            "current": ladder_config.local_tail or ollama_models[0]
            if ollama_models
            else None,
            "options": ollama_models,
        },
    }


@app.get("/_settings")
async def get_settings():
    # _settings_view does a blocking Ollama /api/tags probe (up to 5s when the
    # host is unreachable) — run it off the event loop so the dashboard SSE and
    # in-flight completions never stall behind the settings panel.
    return JSONResponse(await asyncio.to_thread(_settings_view))


@app.post("/_settings")
async def post_settings(request: Request):
    """Manage the NVIDIA key override and custom OpenAI-compatible providers.

    Body: {"action": "set_nvidia_key"|"add_provider"|"remove_provider", ...}."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    action = body.get("action")
    if action == "set_nvidia_key":
        ladder_config.set_nvidia_key(body.get("key"))
    elif action == "add_provider":
        name = (body.get("name") or "").strip()
        base_url = (body.get("base_url") or "").strip()
        models = body.get("models") or []
        if isinstance(models, str):
            models = [m.strip() for m in models.replace("\n", ",").split(",")]
        if not name or (not base_url and name not in ladder_config.providers):
            return JSONResponse(
                {"error": "name and base_url are required"}, status_code=400
            )
        if not isinstance(models, list):
            return JSONResponse({"error": "models must be a list"}, status_code=400)
        ladder_config.add_provider(name, base_url, body.get("api_key") or "", models)
    elif action == "remove_provider":
        ladder_config.remove_provider((body.get("name") or "").strip())
    elif action == "set_local_tail":
        model = (body.get("model") or "").strip()
        ladder_config.set_local_tail(model if model else None)
    else:
        return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
    return JSONResponse({"ok": True, **(await asyncio.to_thread(_settings_view))})


async def _fetch_model_ids(base_url: str, key: Optional[str]) -> dict:
    """GET {base_url}/models from an OpenAI-compatible endpoint.

    Retries once on a transport/timeout error: hosts like NVIDIA publish several
    A records, and if the first IP blackholes the SYN httpx can burn the whole
    connect budget before trying the next — a fresh client on retry picks another
    address, so a transient ConnectTimeout doesn't fail discovery outright."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{base_url.rstrip('/')}/models"
    last: Optional[Exception] = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=10.0)
            ) as client:
                r = await client.get(url, headers=headers)
            if r.status_code >= 400:
                return {"error": f"HTTP {r.status_code}"}
            data = r.json()
            ids = sorted(
                m.get("id")
                for m in data.get("data", [])
                if isinstance(m, dict) and m.get("id")
            )
            return {"models": ids}
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = e  # transient — try a fresh connection once more
            continue
        except (httpx.HTTPError, ValueError, KeyError) as e:
            msg = str(e).strip() or type(e).__name__
            return {"error": msg[:140]}
    msg = str(last).strip() or type(last).__name__ if last else "unknown error"
    return {"error": msg[:140]}


@app.get("/_models/available")
async def models_available():
    """Live-discover model ids from NVIDIA + each custom provider so the UI can
    offer new models to add to the ladder."""
    in_ladder = set(_known_models())
    providers = {}
    for name, p in ladder_config.providers.items():
        providers[name] = await _fetch_model_ids(
            p.get("base_url", ""), p.get("api_key")
        )
    return JSONResponse({"in_ladder": sorted(in_ladder), "providers": providers})


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    rows = _model_view()
    color = {
        "live": "#2e7d32",
        "cooling": "#e65100",
        "dead": "#b71c1c",
        "local": "#1565c0",
        "disabled": "#5b6472",
    }
    tot = {
        "requests": 0,
        "successes": 0,
        "rate_limited": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_total": 0,
        "saved_usd": 0.0,
    }
    trs = []
    for i, r in enumerate(rows):
        for k in tot:
            tot[k] += r.get(k, 0)
        avail = (
            "now"
            if r["state"] == "live"
            else (
                _fmt_dur(r["cooling_s"])
                if r["state"] == "cooling"
                else (
                    "last rung"
                    if r["state"] == "local"
                    else ("off" if r["state"] == "disabled" else "dropped")
                )
            )
        )
        limit_rpm = (
            f"{r['learned_limit_rpm']}/min"
            if r["learned_limit_rpm"] is not None
            else "learning…"
        )
        cd = (
            _fmt_dur(r["learned_cooldown_s"])
            if r["learned_cooldown_s"]
            else "learning…"
        )
        badge = f'<span style="color:{color.get(r["state"], "#555")};font-weight:600">{r["state"].upper()}</span>'
        tin = (
            f"{_fmt_num(r['tokens_in'])}<span class=dim> (+{r['last_in']})</span>"
            if r["tokens_in"]
            else "—"
        )
        tout = (
            f"{_fmt_num(r['tokens_out'])}<span class=dim> (+{r['last_out']})</span>"
            if r["tokens_out"]
            else "—"
        )
        tpi = f"{_fmt_tpm(r['live_tpm_in'])} <span class=dim>(peak {_fmt_tpm(r['peak_tpm_in'])})</span>"
        tpo = f"{_fmt_tpm(r['live_tpm_out'])} <span class=dim>(peak {_fmt_tpm(r['peak_tpm_out'])})</span>"
        saved = f'<span class=save>{_fmt_money(r["saved_usd"])}</span>' if r["saved_usd"] else "—"
        trs.append(
            f"<tr><td class=n>{i + 1}</td><td class=m>{_ACTIVE_DOT if r.get('active') else ''}{r['model']}</td><td>{badge}</td>"
            f"<td>{avail}</td><td>{r['requests']}</td><td>{r['successes']}</td>"
            f"<td>{r['rate_limited']}</td><td>{r['live_rpm']} <span class=dim>(peak {r['peak_rpm']})</span></td>"
            f"<td>{limit_rpm}</td><td>{_fmt_ceiling(r['learned_limit_tpm_in'])}</td>"
            f"<td>{_fmt_ceiling(r['learned_limit_tpm_out'])}</td><td>{tpi}</td><td>{tpo}</td>"
            f"<td>{cd}</td>"
            f"<td class=num>{tin}</td><td class=num>{tout}</td><td class=num>{_fmt_num(r['tokens_total'])}</td>"
            f"<td class=num>{saved}</td></tr>"
        )
    foot = _totals_row(rows, tot)
    saved_hero = _fmt_money(tot["saved_usd"])
    key_ok = bool(ladder_config.providers)
    prov_n = len(ladder_config.providers)
    cfg_items = []
    for m in _known_models():
        checked = "" if m in ladder_config.disabled else "checked"
        cfg_items.append(
            f'<li class=cfgrow draggable=true data-model="{m}">'
            f"<span class=grip>&#8942;&#8942;</span>"
            f"<input type=checkbox class=tog {checked}>"
            f"<span class=cfgname>{m}</span></li>"
        )
    cfg_html = "".join(cfg_items)
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>NVIDIA failover proxy — live updates</title>
<link rel="icon" type="image/svg+xml" href="{_NV_FAVICON}">
<style>
body{{font:14px/1.45 system-ui,Segoe UI,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:22px}}
body.online{{opacity:1; animation:online 2s infinite ease-in-out}}
body.offline{{opacity:.7; animation:none}}
@keyframes online{{0%,10%,50%,80%,100%{{opacity:.8}} 30%{{opacity:.9}} 70%{{opacity:1}}}}
h1{{font-size:18px;margin:0 0 2px}} .sub{{color:#8b95a5;font-size:12px;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;max-width:1280px}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #232733}}
th{{color:#8b95a5;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
td.n{{color:#5b6472}} td.m{{font-family:ui-monospace,Consolas,monospace;font-size:13px}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.dim{{color:#5b6472}} tr:hover td{{background:#161a22}}
tr.tot td{{border-top:2px solid #2a2f3a;font-weight:600;color:#c8cfda}}
.save{{color:#76b900;font-weight:600;font-variant-numeric:tabular-nums}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:#76b900;margin-right:7px;vertical-align:middle;box-shadow:0 0 0 0 rgba(118,185,0,.7);animation:pulse 1.8s infinite}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(118,185,0,.6)}}70%{{box-shadow:0 0 0 6px rgba(118,185,0,0)}}100%{{box-shadow:0 0 0 0 rgba(118,185,0,0)}}}}
.hero{{display:inline-flex;align-items:baseline;gap:10px;margin:0 0 16px;padding:10px 16px;background:#141822;border:1px solid #233318;border-left:3px solid #76b900;border-radius:8px}}
.hero .amt{{font-size:22px;font-weight:700;color:#76b900;font-variant-numeric:tabular-nums}}
.hero .lbl{{color:#8b95a5;font-size:12px}}
.ok{{color:#2e7d32}} .bad{{color:#b71c1c}}
.live{{color:#2e7d32}} .dead{{color:#b71c1c}} .connection{{font-size:9px;margin-left:8px;color:#5b6472}}
h1{{display:flex;align-items:center;gap:10px}}
img.nvlogo,svg.nvlogo{{height:38px;width:auto;flex:0 0 auto;display:block}}
.nvbrand{{color:#76b900;font-weight:700;letter-spacing:.01em}}
details#cfgpanel{{max-width:1280px;margin:0 0 18px;background:#141822;border:1px solid #232733;border-radius:8px;padding:6px 12px}}
details#cfgpanel summary{{cursor:pointer;color:#c8cfda;font-weight:600;font-size:13px;padding:6px 0}}
ul#cfglist{{list-style:none;margin:8px 0;padding:0;max-width:640px}}
li.cfgrow{{display:flex;align-items:center;gap:10px;padding:6px 10px;margin:4px 0;background:#0f1115;border:1px solid #232733;border-radius:6px;cursor:grab}}
li.cfgrow.dragging{{opacity:.4;border-color:#1565c0}}
li.cfgrow .grip{{color:#5b6472;cursor:grab;user-select:none;font-size:14px}}
li.cfgrow .cfgname{{font-family:ui-monospace,Consolas,monospace;font-size:13px}}
li.cfgrow input.tog:not(:checked) ~ .cfgname{{color:#5b6472;text-decoration:line-through}}
.cfgbar{{display:flex;align-items:center;gap:12px;padding:6px 0 4px}}
.cfgbar button{{background:#1565c0;color:#fff;border:0;border-radius:6px;padding:6px 14px;font:13px system-ui;cursor:pointer}}
.cfgbar button#cfgreset{{background:#2a2f3a}}
details#setpanel{{max-width:1280px;margin:0 0 18px;background:#141822;border:1px solid #232733;border-radius:8px;padding:6px 12px}}
details#setpanel summary{{cursor:pointer;color:#c8cfda;font-weight:600;font-size:13px;padding:6px 0}}
.setsec{{margin:12px 0;padding-bottom:12px;border-bottom:1px solid #232733}}
.setlbl{{color:#8b95a5;font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px}}
.setrow{{display:flex;gap:8px;flex-wrap:wrap}}
.setgrid{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}
#setpanel input{{background:#0f1115;border:1px solid #2a2f3a;border-radius:6px;color:#e6e6e6;padding:7px 10px;font:13px system-ui;min-width:180px;flex:1 1 180px}}
#setpanel button{{background:#1565c0;color:#fff;border:0;border-radius:6px;padding:7px 14px;font:13px system-ui;cursor:pointer;flex:0 0 auto}}
#setpanel button.small{{background:#2a2f3a;padding:3px 10px;font-size:12px;margin-left:8px}}
#setpanel button.rm{{background:#7a1f1f}}
ul#provlist{{list-style:none;margin:0 0 8px;padding:0}}
ul#provlist li{{display:flex;align-items:center;gap:10px;padding:6px 10px;margin:4px 0;background:#0f1115;border:1px solid #232733;border-radius:6px;font-size:13px}}
ul#provlist li .pvmeta{{color:#8b95a5;font-size:12px}}
svg.pvicon{{width:18px;height:18px;flex:0 0 auto;vertical-align:middle}}
.presetrow{{display:flex;flex-wrap:wrap;gap:8px}}
.presetchip{{display:inline-flex;align-items:center;gap:6px;background:#0f1115;border:1px solid #2a2f3a;border-radius:20px;padding:4px 12px 4px 6px;color:#e6e6e6;font:12px system-ui;cursor:pointer}}
.presetchip:hover{{border-color:#1565c0}}
.setlbl svg.pvicon{{margin-right:4px}}
.mdlchip{{display:inline-block;background:#0f1115;border:1px solid #2a2f3a;border-radius:12px;padding:2px 8px 2px 10px;margin:3px;font-family:ui-monospace,Consolas,monospace;font-size:12px}}
.mdlchip button{{background:#1b5e20;padding:1px 7px;font-size:11px;margin-left:6px;border-radius:10px}}
.mdlchip.have{{opacity:.5}}
.disctools{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}}
#disctools input{{flex:1 1 200px;min-width:160px}}
#disctools select{{background:#0f1115;border:1px solid #2a2f3a;border-radius:6px;color:#e6e6e6;padding:7px 10px;font:13px system-ui;cursor:pointer;flex:0 0 auto}}
details.discprov{{background:#0f1115;border:1px solid #232733;border-radius:6px;margin:6px 0;padding:2px 10px}}
details.discprov>summary{{cursor:pointer;padding:6px 2px;list-style:none;font-size:13px;display:flex;align-items:center;gap:6px}}
details.discprov>summary::-webkit-details-marker{{display:none}}
details.discprov>summary::before{{content:"\\25B8";color:#5b6472;margin-right:2px}}
details.discprov[open]>summary::before{{content:"\\25BE"}}
details.discprov .pvmeta{{color:#8b95a5;font-size:12px;font-weight:400}}
.discbody{{padding:4px 0 8px}}
.discbody .addall{{display:inline-block;background:#155e2f !important;padding:3px 10px !important;font-size:12px;margin:3px 3px 6px}}
</style>
<script>
(function(){{
 let b=document.body, t=document.getElementById("timer"), c=document.querySelector(".connection");
 let lm=0, ils=0, rc, src;

function beat(){{
   lm=Date.now(); b.classList.add("online"); b.classList.remove("offline");
   if(c) c.textContent="down-tick";
  }}

  function chk(){{
   let now=Date.now();
   if(now-lm>2000){{ b.classList.remove("online"); b.classList.add("offline"); if(c) c.textContent=rc?"disconnect":"wait"; }}
   if(t){{ let d=(now-ils)/1000; t.textContent=d<2?((d*1000)|0)+"ms":(d|0)+"s"; }}
  }}

 function ssrc(){{
  if(src) src.close();
  src=new EventSource("/updates");
  src.onmessage=function(e){{
   var tb=document.querySelector("tbody");
   if(tb) tb.innerHTML=e.data;
   ils=Date.now(); beat();
  }};
src.onerror=function(){{
    b.classList.remove("online"); b.classList.add("offline");
    clearTimeout(rc); rc=setTimeout(ssrc,500);
    src.close();
   }};
 }}

 ils=Date.now(); ssrc(); beat();
 setInterval(chk,1000);
}})();
</script></head><body>
<h1>{_NV_LOGO_SVG}<span><span class=nvbrand>failover proxy</span> — live state <span class=connection style="font-size:10px">connecting...</span></span></h1>
<div class=sub>{f"<span class=ok>{prov_n} provider(s)</span>" if key_ok else "<span class=bad>no providers — add one in Providers &amp; API keys</span>"}
 · live updates via SSE · models: auto, only, refine (+local refiner), local-only, local-refine, agent-*<span id=timer style="margin-left:10px;color:#5b6472">0s</span></div>
<div class=hero title="Estimated spend avoided vs. running the same tokens on a commercial API (NVIDIA NIM is free on a personal account). Rough per-model rates — tune with PROXY_PRICING_JSON."><span class=amt>{saved_hero}</span><span class=lbl>estimated saved vs. commercial API pricing</span></div>
<details id=cfgpanel><summary>&#9881; Failover ladder — drag to reorder, uncheck to disable</summary>
<ul id=cfglist>{cfg_html}</ul>
<div class=cfgbar><button id=cfgsave>Save order &amp; toggles</button><button id=cfgreset>Reload</button><span id=cfgstatus class=dim></span></div>
</details>
<details id=setpanel><summary>&#128273; Providers &amp; API keys — add NVIDIA, OpenAI, Anthropic, Cerebras… or any OpenAI-compatible API</summary>
<div class=setsec><div class=setlbl>Quick add a provider</div>
<div id=presetrow class=presetrow></div></div>
<div class=setsec><div class=setlbl>Providers</div>
<ul id=provlist></ul>
<div class=setgrid><input id=pvname placeholder="name (e.g. openrouter)"><input id=pvurl placeholder="base URL (…/v1)"><input id=pvkey type=password placeholder="API key"><input id=pvmodels placeholder="model ids, comma-separated"><button id=pvadd>Add / update provider</button></div></div>
<div class=setsec style="border-bottom:0"><div class=setlbl>Discover available models <button id=discbtn class=small>Refresh</button></div>
<div id=disctools class=disctools style="display:none"><input id=discfilter placeholder="filter models…"><select id=discsort><option value="name-asc">sort: name A→Z</option><option value="name-desc">sort: name Z→A</option><option value="new-first">sort: not-yet-added first</option></select></div>
<div id=discresult class=dim>Click Refresh to query each provider's /v1/models. Each provider collapses into its own dropdown; use the filter and sort to narrow things down, then click ＋ to add a model to the ladder.</div></div>
<div class=setsec style="border-bottom:0"><div class=setlbl>Local tail model (Ollama fallback)</div>
<div class=setrow><select id=tailsel class=tailselect style="background:#0f1115;border:1px solid #2a2f3a;border-radius:6px;color:#e6e6e6;padding:7px 10px;font:13px system-ui;flex:1 1 300px;cursor:pointer;min-width:200px"></select>
<button id=tailsave class=small>Save</button></div></div>
<span id=setstatus class=dim></span>
</details>
<table><thead><tr>
<th>#</th><th>model</th><th>state</th><th>available</th><th>req</th><th>ok</th>
<th>429s</th><th>rpm now</th><th>learned RPM</th><th>learned TPM in</th><th>learned TPM out</th>
<th class=num>TPM in now</th><th class=num>TPM out now</th><th>learned cooldown</th>
<th class=num>tokens in</th><th class=num>tokens out</th><th class=num>tokens total</th>
<th class=num title="Estimated spend avoided vs. commercial API pricing for the same tokens">$ saved</th>
</tr></thead><tbody>{"".join(trs)}{foot}</tbody></table>
</body></html>"""
    return html.replace("</body>", CFG_SCRIPT + SET_SCRIPT + "</body>")


@app.get("/v1/models")
async def list_models() -> dict:
    # Core modes + agent profiles + refiner + local-only + real NVIDIA models + local
    special = [
        AUTO_MODEL,
        ONLY_MODEL,
        REFINER_MODEL_ID,
        LOCAL_ONLY,
        LOCAL_REFINE,
    ] + sorted(AGENT_ROLES)
    lm = cascade._local_model()
    ids = special + _serving_ladder() + ([lm] if lm else [])
    seen: Set[str] = set()
    ids = [m for m in ids if not (m in seen or seen.add(m))]
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "nvidia-failover-proxy"}
            for m in ids
        ],
    }


def _inject_agent_prompt(body: dict) -> dict:
    """If the model is an agent role, inject its system prompt first."""
    model = body.get("model", "")
    sys_prompt = AGENT_ROLES.get(model)
    if not sys_prompt:
        return body
    out = dict(body)
    msgs = list(out.get("messages", []))
    # Don't double-inject if the user already has a system message
    has_sys = any(m.get("role") == "system" for m in msgs)
    if not has_sys:
        msgs.insert(0, {"role": "system", "content": sys_prompt})
    out["messages"] = msgs
    return out


def _fmt_header(text: str) -> str:
    return f"\n━━━ {text} ━━━\n\n"


def _fmt_bold(text: str) -> str:
    return f"**{text}**"


def _get_last_user_msg(messages: List[dict]) -> Optional[str]:
    """Return the text content of the most recent user message, or None."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"].strip()
    return None


def _format_model_list() -> str:
    """Pretty model listing with state badges."""
    special = [
        AUTO_MODEL,
        ONLY_MODEL,
        REFINER_MODEL_ID,
        LOCAL_ONLY,
        LOCAL_REFINE,
    ] + sorted(AGENT_ROLES)
    lines = [_fmt_header("Proxy Modes")]
    for m in special:
        desc = {
            AUTO_MODEL: "cloud cascade → local 80B",
            ONLY_MODEL: "cloud cascade only, 429 when all cooling",
            REFINER_MODEL_ID: f"cloud cascade + [refine]→{REFINER_MODEL}",
            LOCAL_ONLY: "local 80B only, no cloud",
            LOCAL_REFINE: f"[refine]→{REFINER_MODEL} → local 80B",
            **{k: v.split(".")[0] for k, v in AGENT_ROLES.items()},
        }.get(m, "")
        lines.append(f"  {m:<30} {desc}")
    lines.append(_fmt_header("Failover Ladder"))
    now = time.time()
    for m in _serving_ladder():
        cooling = cascade.model_until.get(m, 0)
        if m in cascade.dead:
            tag = "☠ dead"
        elif cooling > now:
            tag = f"❄ {int(cooling - now)}s"
        else:
            tag = "✓ live"
        lines.append(f"  {m:<55} {tag}")
    lm = cascade._local_model()
    if lm:
        lines.append(f"\n  {lm:<55} local tail (always last)")
    return "\n".join(lines)


async def _handle_command(body: dict) -> Optional[JSONResponse]:
    """Check if the last user message starts with / and handle it.

    Returns a JSONResponse (command handled) or None (pass through).
    """
    global _model_override
    text = _get_last_user_msg(body.get("messages", []))
    if not text or not text.startswith("/"):
        return None

    parts = text[1:].split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []

    if cmd in ("help", "?"):
        msg = (
            _fmt_header("Proxy Commands")
            + "Type any of these as your message:\n\n"
            + "  /help          Show this help\n"
            + "  /pick [N|off]  Interactive model picker (numbered list)\n"
            + "  /stats         Current usage stats (rpms, tokens, cooling)\n"
            + "  /models        List available models with state\n"
            + "  /health        Proxy + key health\n"
            + "  /cool <model>  Manually sideline a model for 5min\n"
            + "  /uncool <model> Remove a model from cooldown\n"
            + "  /switch <model> Hint to switch active model\n"
            + "  /warm [model]   Preload an Ollama model into GPU memory\n"
            + "  /unload [model] Evict an Ollama model from GPU memory\n"
        )

    elif cmd == "stats":
        rows = _model_view()
        lines = [_fmt_header("Usage Stats (last 60s window)")]
        for r in rows:
            if r["requests"] == 0 and r["state"] == "live":
                continue  # skip idle models for brevity
            rpm = (
                f"{r['live_rpm']}/{_fmt_tpm(r['learned_limit_rpm'])}"
                if r["learned_limit_rpm"]
                else f"{r['live_rpm']}/?"
            )
            tpi = (
                f"{_fmt_tpm(r['live_tpm_in'])}/{_fmt_tpm(r['learned_limit_tpm_in'] or 0)}"
                if r["learned_limit_tpm_in"]
                else f"{_fmt_tpm(r['live_tpm_in'])}/?"
            )
            tpo = (
                f"{_fmt_tpm(r['live_tpm_out'])}/{_fmt_tpm(r['learned_limit_tpm_out'] or 0)}"
                if r["learned_limit_tpm_out"]
                else f"{_fmt_tpm(r['live_tpm_out'])}/?"
            )
            state_icon = {"live": "✓", "cooling": "❄", "dead": "☠", "local": "⌂"}.get(
                r["state"], "?"
            )
            lines.append(
                f"  {state_icon} {r['model']:<50} RPM {rpm:<12} TPMi {tpi:<12} TPMo {tpo:<12} req={r['requests']} ok={r['successes']} 429={r['rate_limited']}"
            )
        msg = "\n".join(lines)

    elif cmd == "models":
        msg = _format_model_list()

    elif cmd == "health":
        key = resolve_api_key()
        cooling = {
            m: int(t - time.time())
            for m, t in cascade.model_until.items()
            if t > time.time()
        }
        msg = (
            _fmt_header("Proxy Health")
            + f"  API key: {'✓ loaded' if key else '✗ missing'}\n"
            + f"  Dead models: {len(cascade.dead)}\n"
            + f"  Cooling models: {len(cooling)}\n"
            + f"  Ladder size: {len(cascade.models)} cloud + {'yes' if cascade._local_model() else 'no'} local\n"
            + f"  Refiner: {'enabled' if REFINER_ENABLE else 'disabled'} ({REFINER_MODEL})\n"
        )

    elif cmd == "cool" and args:
        model = " ".join(args)
        if model in _serving_ladder():
            cascade.cool(model, _MODEL_COOLDOWN_S)
            msg = f"  ❄ **{model}** sidelined for {_fmt_dur(_MODEL_COOLDOWN_S)}"
        else:
            msg = f"  ✗ Unknown model: {model}. Use /models to see valid models."

    elif cmd == "uncool" and args:
        model = " ".join(args)
        if model in cascade.model_until:
            cascade.model_until.pop(model, None)
            msg = f"  ✓ **{model}** removed from cooldown"
        else:
            msg = f"  ✗ {model} is not on cooldown."

    elif cmd == "switch" and args:
        model = " ".join(args)
        msg = f"  ↻ Switch hint noted: **{model}**\n  (OpenCode controls model selection; set it in the UI or config)"

    elif cmd in ("warm", "unload"):
        ollama_prov = ladder_config.providers.get("ollama", {})
        ollama_models = ollama_prov.get("models", [])
        if not ollama_models:
            msg = "  ✗ No Ollama provider configured. Add one via the settings UI or set OLLAMA_MODELS."
        else:
            # Determine which model to warm/unload
            target = " ".join(args) if args else ollama_models[0]
            if target not in ollama_models:
                known = ", ".join(ollama_models)
                msg = f"  ✗ **{target}** is not in the Ollama model list.\n    Known Ollama models: {known}"
            else:
                # Build native Ollama API base URL (strip /v1 suffix)
                ollama_url = (
                    ollama_prov.get("base_url", "http://127.0.0.1:11434/v1") or ""
                ).strip()
                native_base = _strip_v1(ollama_url)
                keep_alive = -1 if cmd == "warm" else 0
                action = "warming" if cmd == "warm" else "unloading"
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.post(
                            f"{native_base}/api/generate",
                            json={"model": target, "keep_alive": str(keep_alive)},
                        )
                    if resp.status_code in (200, 201):
                        msg = f"  ✓ **{target}** {action} via Ollama (keep_alive={keep_alive})"
                    else:
                        msg = (
                            f"  ✗ **{target}** {action} failed: HTTP {resp.status_code}\n"
                            f"    {resp.text[:300]}"
                        )
                except httpx.RequestError as exc:
                    msg = f"  ✗ **{target}** {action} failed — connection error: {exc}"

    elif cmd == "pick":
        # Build numbered list of all available models
        lm = cascade._local_model()
        all_models = (
            [AUTO_MODEL, ONLY_MODEL, REFINER_MODEL_ID, LOCAL_ONLY, LOCAL_REFINE]
            + sorted(AGENT_ROLES)
            + _serving_ladder()
            + ([lm] if lm else [])
        )
        now = time.time()

        if args and args[0].isdigit():
            idx = int(args[0]) - 1
            if 0 <= idx < len(all_models):
                picked = all_models[idx]
                _model_override = picked
                msg = f"  ✓ Picked **{picked}**\n  (Override set for subsequent messages. Use `/pick off` to clear.)"
            else:
                msg = f"  ✗ Invalid number. Use /pick to see available models (1-{len(all_models)})."
        elif args and args[0].lower() == "off":
            _model_override = None
            msg = "  ✓ Model override cleared — using default model."
        else:
            current = _model_override or "(default)"
            lines = [_fmt_header(f"Model Picker — current: {current}")]
            lines.append("  Type `/pick N` to select, `/pick off` to clear:\n")
            for idx, m in enumerate(all_models, 1):
                # State badge
                if m == _model_override:
                    badge = " ◀"
                elif m in cascade.dead:
                    badge = " ☠"
                elif m in (LOCAL_ONLY, LOCAL_REFINE) and cascade._local_model():
                    badge = " ⌂"
                elif (
                    m in (AUTO_MODEL, ONLY_MODEL, REFINER_MODEL_ID) or m in AGENT_ROLES
                ):
                    badge = ""
                elif cascade.model_until.get(m, 0) > now:
                    badge = " ❄"
                else:
                    badge = ""
                lines.append(f"  {idx:>2}. {m:<50}{badge}")
            msg = "\n".join(lines)

    else:
        msg = (
            _fmt_header("Unknown Command")
            + f"  `/{cmd}` not recognized. Try /help for available commands.\n"
        )

    return JSONResponse(
        {
            "id": f"cmd-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "proxy-commands",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": msg.strip(),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return JSONResponse(
            {"error": {"message": "request body is not valid JSON"}}, status_code=400
        )
    preferred = body.get("model")
    stream = bool(body.get("stream"))

    # ---- Commands: intercept /command in last user message ------------
    cmd_resp = await _handle_command(body)
    if cmd_resp is not None:
        return cmd_resp

    # ---- Model override: /pick sets a global default ------------------
    if _model_override and (
        preferred is None or preferred == AUTO_MODEL or preferred == ""
    ):
        preferred = _model_override

    # ---- Agent system prompt injection --------------------------------
    body = _inject_agent_prompt(body)

    # ---- Prompt refiner: triggered by [refine] tag ------------------------
    should_refine = _has_refine_tag(body.get("messages", []))
    if should_refine:
        ref_timeout = httpx.Timeout(connect=5.0, read=30.0, write=15.0, pool=5.0)
        body["messages"] = await _refine_prompt(body.get("messages", []), ref_timeout)
        # Route through the appropriate ladder after refinement.
        # If the user picked local-refine, keep routing to local only.
        if preferred == LOCAL_REFINE:
            effective_model = LOCAL_REFINE
        else:
            effective_model = AUTO_MODEL
    else:
        effective_model = preferred

    ladder = cascade.order(effective_model)
    if not ladder:
        # Only reachable in nvidia-only mode: every cloud model is cooling and
        # there is no local tail rung. Report a real rate-limit.
        retry = cascade.soonest_cooldown()
        msg = f"all NVIDIA frontier models are rate-limited; retry in ~{retry}s"
        print(f"[proxy] nvidia-only exhausted; 429 (retry {retry}s)")
        if stream:
            return StreamingResponse(
                _sse_once(_sse_error(msg)), media_type="text/event-stream"
            )
        return JSONResponse(
            {
                "error": {
                    "message": msg,
                    "type": "rate_limit_exceeded",
                    "code": "rate_limited",
                }
            },
            status_code=429,
            headers={"Retry-After": str(retry)},
        )
    # Per-model read timeout: if a model stalls (slow/hung upstream), fail over
    # to the next one instead of hanging the whole request. connect stays short
    # so an unreachable rung (e.g. local Ollama down) is skipped near-instantly.
    read_t = float(os.environ.get("PROXY_MODEL_TIMEOUT_S", "90"))
    timeout = httpx.Timeout(connect=15.0, read=read_t, write=30.0, pool=15.0)

    if stream:
        return StreamingResponse(
            _stream_cascade(body, ladder, timeout), media_type="text/event-stream"
        )

    # ---- non-streaming: try each model until one answers -------------------
    tried: List[str] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in ladder:
            tried.append(model)
            base_url, headers = _route(model)
            resp = None
            for attempt in range(2):  # attempt 1 is the connect-retry
                stats.record_request(model)
                try:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=_prep_body(body, model),
                    )
                    break
                except _CONNECT_ERRORS as e:
                    if attempt == 0:
                        print(f"[proxy] {model}: {type(e).__name__}; retrying")
                        continue
                    stats.record_error(model)
                    cascade.cool(model, _CONNECT_COOLDOWN_S)
                    print(
                        f"[proxy] {model}: {type(e).__name__}; cooling {_CONNECT_COOLDOWN_S}s; next"
                    )
                except httpx.HTTPError as e:
                    stats.record_error(model)
                    cascade.cool(model, _MODEL_COOLDOWN_S)
                    print(
                        f"[proxy] {model}: {type(e).__name__}; cooling {_MODEL_COOLDOWN_S}s; next"
                    )
                    break
            if resp is None:
                continue
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    stats.record_error(model)
                    body_preview = (await resp.aread())[:200]
                    print(
                        f"[proxy] {model}: non-JSON 200 response "
                        f"({body_preview}); failing over"
                    )
                    continue
                stats.record_usage(model, data.get("usage"))
                if SKIP_EMPTY and not _resp_has_content(data):
                    stats.record_error(model)
                    print(f"[proxy] {model}: empty completion; failing over")
                    continue
                _choice0 = (data.get("choices") or [{}])[0]
                _txt = (_choice0.get("message") or {}).get("content") or ""
                _deg = _degenerate_reason(_txt, _choice0.get("finish_reason"), body)
                if _deg:
                    stats.record_error(model)
                    cascade.cool(model, _CONNECT_COOLDOWN_S)
                    print(f"[proxy] {model}: {_deg}; failing over")
                    continue
                stats.record_success(model)
                data["_proxy_model"] = model  # trace which model actually served
                print(f"[proxy] served by {model}")
                return JSONResponse(data)
            if resp.status_code in (401, 402, 403):
                # account-level — no other model will help
                return JSONResponse(
                    {
                        "error": {
                            "message": f"NVIDIA account gated ({resp.status_code}): {resp.text[:300]}"
                        }
                    },
                    status_code=resp.status_code,
                )
            cascade.note_status(
                model, resp.status_code, resp.headers.get("retry-after")
            )
            print(f"[proxy] {model}: {resp.status_code}; failing over")
    return JSONResponse(
        {"error": {"message": f"all frontier models unavailable; tried {tried}"}},
        status_code=502,
    )


async def _stream_cascade(body: dict, ladder: List[str], timeout: httpx.Timeout):
    """Yield SSE bytes from the first model that connects with 200; on a 429/EOL
    or a connect failure, fail over to the next model.

    Two robustness rules that keep OpenCode from stopping mid-task:
      * A connect failure is retried once (fresh connection → likely a different
        A record) and, if still failing, only short-cooled — a transient blip
        can't drain the ladder and hard-fail the request.
      * Once we've streamed real bytes to the client, an upstream error can NOT
        restart on a fresh model (that would splice a second answer onto the
        first). We end the SSE stream cleanly with [DONE] instead."""
    client_sent = False  # have we delivered any content bytes to the client?
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in ladder:
            base_url, headers = _route(model)
            for attempt in range(2):  # attempt 1 is the connect-retry
                stats.record_request(model)
                try:
                    async with client.stream(
                        "POST",
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=_prep_body(body, model),
                    ) as resp:
                        if resp.status_code != 200:
                            await resp.aread()
                            if resp.status_code in (401, 402, 403):
                                yield _sse_error(
                                    f"NVIDIA account gated ({resp.status_code})"
                                )
                                return
                            cascade.note_status(
                                model, resp.status_code, resp.headers.get("retry-after")
                            )
                            print(
                                f"[proxy/stream] {model}: {resp.status_code}; failing over"
                            )
                            break  # not a connect issue — next model
                        # Forward line-by-line (equivalent SSE framing) while
                        # sniffing usage for token accounting. Buffer the leading
                        # chunks until we see real content/tool_calls: if the
                        # stream ends without any (an empty 200 completion),
                        # discard and fail over instead of emitting nothing.
                        #
                        # NOTE: NVIDIA may send the usage block in the first SSE
                        # chunk (with role="assistant" delta) AND again in the
                        # final chunk (with choices=[]).  We only record the
                        # *last* occurrence to avoid double-counting
                        # prompt_tokens.
                        buffered: List[bytes] = []
                        committed = not SKIP_EMPTY
                        _stream_usage: Optional[dict] = None
                        # Mid-stream code-switch guard: count unexpected CJK chars
                        # cumulatively (a degenerate model drifting into Chinese)
                        # and truncate cleanly the moment the count crosses the
                        # threshold — checking *before* the offending chunk is
                        # forwarded, so the Chinese never reaches the client.
                        guard_cjk = GUARD_DEGENERATE and not _prompt_has_cjk(body)
                        cjk_total = 0
                        # Content-idle watchdog: the moment the last real content
                        # delta arrived. A fully-silent socket is already caught by
                        # httpx's read timeout; this catches the other hang, where
                        # the upstream keeps the stream open (often trickling SSE
                        # keepalives) but stops producing tokens after some real
                        # output. We only enforce it once content has started, so a
                        # slow first token on a big reasoning model isn't cut.
                        last_content = time.monotonic()
                        async for line in resp.aiter_lines():
                            if (
                                committed
                                and time.monotonic() - last_content > _STREAM_STALL_S
                            ):
                                raise _StreamStall()
                            drifted = False
                            if line.startswith("data:"):
                                payload = line[5:].strip()
                                if payload and payload != "[DONE]":
                                    try:
                                        obj = json.loads(payload)
                                        if isinstance(obj, dict):
                                            if obj.get("usage"):
                                                # Save — record once after the
                                                # loop, using the last (most
                                                # complete) occurrence.
                                                _stream_usage = obj["usage"]
                                            if _delta_has_content(obj):
                                                # Real token(s) landed — reset the
                                                # idle watchdog and commit.
                                                last_content = time.monotonic()
                                                committed = True
                                            if guard_cjk:
                                                cjk_total += _cjk_count(
                                                    _delta_text(obj)
                                                )
                                                if cjk_total >= _CJK_MIN_CHARS:
                                                    drifted = True
                                    except ValueError:
                                        pass
                            if committed and drifted:
                                # Drifted into Chinese — stop WITHOUT emitting this
                                # (CJK) chunk. Better a clean, truncated answer than
                                # pages of CJK leaking to the client.
                                if _stream_usage is not None:
                                    stats.record_usage(model, _stream_usage)
                                stats.record_success(model)
                                print(
                                    f"[proxy/stream] {model}: unexpected CJK "
                                    f"code-switch mid-stream; truncating"
                                )
                                yield b"data: [DONE]\n\n"
                                return
                            emit = (line + "\n").encode("utf-8")
                            if committed:
                                for b in buffered:
                                    yield b
                                buffered.clear()
                                yield emit
                                client_sent = True
                            else:
                                buffered.append(emit)
                        # Record usage now — exactly once per stream, using the
                        # last (most complete) usage block seen.
                        if _stream_usage is not None:
                            stats.record_usage(model, _stream_usage)
                        if committed:
                            stats.record_success(model)
                            print(f"[proxy/stream] served by {model}")
                            return
                        stats.record_error(model)
                        print(f"[proxy/stream] {model}: empty stream; failing over")
                        break  # next model
                except _StreamStall:
                    stats.record_error(model)
                    if client_sent:
                        # Real tokens already delivered, then the stream wedged.
                        # We can't restart on another model (would splice a second
                        # answer); record the partial and end cleanly so the client
                        # gets a finished — if truncated — message instead of an
                        # endless spinner.
                        if _stream_usage is not None:
                            stats.record_usage(model, _stream_usage)
                        stats.record_success(model)
                        print(
                            f"[proxy/stream] {model}: stalled >{_STREAM_STALL_S:.0f}s "
                            f"mid-output; ending cleanly"
                        )
                        yield b"data: [DONE]\n\n"
                        return
                    # Stalled before any content reached the client — sideline
                    # briefly and fail over to the next rung.
                    cascade.cool(model, _CONNECT_COOLDOWN_S)
                    print(
                        f"[proxy/stream] {model}: stalled >{_STREAM_STALL_S:.0f}s "
                        f"before output; cooling {_CONNECT_COOLDOWN_S}s; next"
                    )
                    break
                except _CONNECT_ERRORS as e:
                    # Connect never established, so nothing was sent. Retry once on
                    # a fresh connection, then only briefly sideline the model.
                    if attempt == 0:
                        print(f"[proxy/stream] {model}: {type(e).__name__}; retrying")
                        continue
                    stats.record_error(model)
                    cascade.cool(model, _CONNECT_COOLDOWN_S)
                    print(
                        f"[proxy/stream] {model}: {type(e).__name__}; cooling {_CONNECT_COOLDOWN_S}s; next"
                    )
                    break
                except httpx.HTTPError as e:
                    stats.record_error(model)
                    if client_sent:
                        # Partial answer already delivered — can't cleanly restart
                        # on another model. End the stream so the client sees a
                        # finished (if truncated) message rather than corruption.
                        print(
                            f"[proxy/stream] {model}: {type(e).__name__} mid-stream; ending cleanly"
                        )
                        yield b"data: [DONE]\n\n"
                        return
                    cascade.cool(model, _MODEL_COOLDOWN_S)
                    print(
                        f"[proxy/stream] {model}: {type(e).__name__}; cooling {_MODEL_COOLDOWN_S}s; next"
                    )
                    break
        yield _sse_error("all frontier models unavailable")


async def _sse_once(payload: bytes):
    yield payload


def _sse_error(message: str) -> bytes:
    payload = json.dumps({"error": {"message": message}})
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode("utf-8")


def sse_escape(s: str) -> str:
    """Escape newlines in SSE event data."""
    return s.replace("\n", "\\n").replace("\r", "")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PROXY_PORT", "5002"))
    host = os.environ.get("PROXY_HOST", "127.0.0.1")
    print(
        f"NVIDIA failover proxy on http://{host}:{port}/v1  (models: {AUTO_MODEL} + {len(cascade.models)} frontier)"
    )
    uvicorn.run(app, host=host, port=port)
