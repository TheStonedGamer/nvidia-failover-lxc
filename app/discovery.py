"""Live provider model discovery, shared by /v1/models, /_models/available,
and per-model chat routing. A model discovered here but absent from a
provider's curated `models` list is still resolvable to the right base_url
and key, so any individual model a provider offers can be requested directly
without first being added to the ladder."""

import asyncio
import time
from typing import Dict, List, Optional

import httpx

from app.config import FRONTIER_MODELS

_DISCOVERY_CACHE: Dict[str, List[str]] = {}
_MODEL_PROVIDER: Dict[str, str] = {}
_LAST_REFRESH: Dict[str, float] = {}
_REFRESH_INTERVAL_S = 30.0


def _is_nvidia_provider(name: str, base_url: str) -> bool:
    return name == "nvidia" or "integrate.api.nvidia.com" in (base_url or "")


async def _fetch_model_ids(base_url: str, key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last_err = None
    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt in range(3):
            try:
                resp = await client.get(f"{base_url}/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
                return {"models": ids}
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = e
                await asyncio.sleep(0.4 * (attempt + 1))
            except Exception as e:
                return {"error": str(e)}
    return {"error": str(last_err) if last_err else "unknown error"}


async def discover_provider(name: str, p: dict) -> dict:
    result = await _fetch_model_ids(p.get("base_url", ""), p.get("api_key", ""))
    if "error" in result:
        cached = _DISCOVERY_CACHE.get(name)
        if cached:
            return {"models": cached, "stale": True}
        if _is_nvidia_provider(name, p.get("base_url", "")):
            return {"models": list(FRONTIER_MODELS), "fallback": "static"}
        return result
    _DISCOVERY_CACHE[name] = result["models"]
    for mid in result["models"]:
        _MODEL_PROVIDER[mid] = name
    return result


async def discover_all(providers: Dict[str, dict], force: bool = False) -> Dict[str, dict]:
    """Refresh each provider's model list at most once per _REFRESH_INTERVAL_S
    — /v1/models is polled frequently by clients and shouldn't pay a network
    round-trip (or a slow provider's retry/backoff) on every single call."""
    now = time.time()
    due = [n for n in providers if force or now - _LAST_REFRESH.get(n, 0) >= _REFRESH_INTERVAL_S]
    if due:
        results = await asyncio.gather(*(discover_provider(n, providers[n]) for n in due))
        for n in due:
            _LAST_REFRESH[n] = now

    out: Dict[str, dict] = {}
    fresh = dict(zip(due, results)) if due else {}
    for name in providers:
        if name in fresh:
            out[name] = fresh[name]
        elif name in _DISCOVERY_CACHE:
            out[name] = {"models": _DISCOVERY_CACHE[name], "stale": True}
        else:
            out[name] = {"error": "not yet discovered"}
    return out


def cached_provider_for(model: str) -> Optional[str]:
    return _MODEL_PROVIDER.get(model)


def all_discovered_models() -> List[str]:
    seen: List[str] = []
    for ids in _DISCOVERY_CACHE.values():
        for m in ids:
            if m not in seen:
                seen.append(m)
    return seen
