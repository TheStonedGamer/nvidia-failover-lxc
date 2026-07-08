"""/v1/models plus provider model discovery (/_models/available)."""

import asyncio
import time
from typing import Dict, List

import httpx
from fastapi import APIRouter

from app.config import AUTO_MODEL, ONLY_MODEL, REFINER_MODEL_ID, LOCAL_ONLY, LOCAL_REFINE, AGENT_ROLES, FRONTIER_MODELS
from app.ladder import ladder_config
from app.state import cascade

router = APIRouter()

_DISCOVERY_CACHE: Dict[str, List[str]] = {}


def _known_models() -> List[str]:
    known = list(ladder_config.order)
    for name, p in ladder_config.providers.items():
        if name == "ollama":
            continue
        for m in p.get("models") or []:
            if m not in known:
                known.append(m)
    return known


@router.get("/v1/models")
async def list_models():
    ids = [AUTO_MODEL, ONLY_MODEL, REFINER_MODEL_ID, LOCAL_ONLY, LOCAL_REFINE]
    ids += sorted(AGENT_ROLES)
    for m in _known_models():
        if m not in ids:
            ids.append(m)
    local = cascade._local_model()
    if local and local not in ids:
        ids.append(local)
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "owned_by": "model-proxy"} for mid in ids],
    }


def inject_agent_prompt(body: dict, model) -> dict:
    role_prompt = AGENT_ROLES.get(model)
    if not role_prompt:
        return body
    messages = body.get("messages") or []
    if any(m.get("role") == "system" for m in messages):
        return body
    out = dict(body)
    out["messages"] = [{"role": "system", "content": role_prompt}] + list(messages)
    return out


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


async def _discover_provider(name: str, p: dict) -> dict:
    result = await _fetch_model_ids(p.get("base_url", ""), p.get("api_key", ""))
    if "error" in result:
        cached = _DISCOVERY_CACHE.get(name)
        if cached:
            return {"models": cached, "stale": True}
        if _is_nvidia_provider(name, p.get("base_url", "")):
            return {"models": list(FRONTIER_MODELS), "fallback": "static"}
        return result
    _DISCOVERY_CACHE[name] = result["models"]
    return result


@router.get("/_models/available")
async def models_available():
    providers = ladder_config.providers
    names = list(providers.keys())
    results = await asyncio.gather(*(_discover_provider(n, providers[n]) for n in names))
    in_ladder = set(_known_models())
    out = {}
    for name, result in zip(names, results):
        out[name] = result
    return {"in_ladder": sorted(in_ladder), "providers": out}
