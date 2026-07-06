"""Minimal NVIDIA provider for the failover proxy.
Resolves API key from env or OpenCode config.
"""

import os
import re
from typing import Optional, List

# --- Resolve API key ------------------------------------------------------------------
_NVAPI_RE = re.compile(r"nvapi-([a-zA-Z0-9_-]{50,})")
_OPENCODE_JSONC = os.path.expanduser("~/.config/opencode/opencode.jsonc")


def resolve_api_key() -> Optional[str]:
    """Resolve NVIDIA API key. Returns None if not found (proxy serves /v1/models)."""
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    try:
        with open(_OPENCODE_JSONC, "r", encoding="utf-8") as f:
            m = _NVAPI_RE.search(f.read())
            return m.group(1) if m else None
    except OSError:
        return None


# --- Curated frontier ladder ------------------------------------------------------------
FRONTIER_MODELS: List[str] = [
    # strongest first, verified present 2026-07 via /v1/models
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


def _ladder() -> List[str]:
    """Build cloud model ladder."""
    # override via ROUTER_NVIDIA_MODELS
    env = os.environ.get("ROUTER_NVIDIA_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    # single override ROUTER_NVIDIA_MODEL
    single = os.environ.get("ROUTER_NVIDIA_MODEL")
    if single:
        rest = [m for m in FRONTIER_MODELS if m != single]
        return [single] + rest
    # default
    return list(FRONTIER_MODELS)
