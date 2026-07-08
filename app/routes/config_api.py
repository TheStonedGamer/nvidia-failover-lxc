"""Ladder + provider CRUD API backing the settings/config dashboard panels."""

from typing import Dict

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import strip_v1
from app.discovery import discover_provider
from app.ladder import ladder_config
from app.state import cascade, stats
from app.routes.models import _known_models

router = APIRouter()


async def _populate_from_live_catalog(name: str) -> None:
    """Immediately fetch and persist a provider's full model list, rather
    than waiting on the next throttled /v1/models discovery pass to surface
    what it offers."""
    p = ladder_config.providers.get(name)
    if not p:
        return
    result = await discover_provider(name, p)
    discovered = result.get("models") or []
    if discovered:
        ladder_config.add_provider(name, models=discovered)


@router.get("/_config")
async def get_config():
    return {"order": _known_models(), "disabled": sorted(ladder_config.disabled)}


@router.post("/_config")
async def post_config(request: Request):
    body = await request.json()
    order = body.get("order")
    disabled = body.get("disabled")
    if not isinstance(order, list) or not all(isinstance(m, str) for m in order):
        return JSONResponse({"error": "order must be a list of strings"}, status_code=400)
    if not isinstance(disabled, list) or not all(isinstance(m, str) for m in disabled):
        return JSONResponse({"error": "disabled must be a list of strings"}, status_code=400)
    ladder_config.update(order, disabled)
    return {"ok": True, "order": ladder_config.order, "disabled": sorted(ladder_config.disabled), "active": ladder_config.active_ladder()}


@router.post("/_reset_cooldowns")
async def reset_cooldowns():
    try:
        cleared = cascade.reset_cooldowns()
        return {"ok": True, "cleared": cleared}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/_reset_stats")
async def reset_stats():
    try:
        stats.reset_all()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


def _mask_key(key: str) -> str:
    if key and len(key) > 12:
        return key[:6] + "…" + key[-4:]
    return "•" * 8


def _ollama_live_models(base_url: str) -> list:
    """Blocking probe of Ollama's native /api/tags; runs off the event loop
    via asyncio.to_thread since httpx's sync client is used here."""
    native = strip_v1(base_url)
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{native}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        ollama = ladder_config.providers.get("ollama") or {}
        return list(ollama.get("models") or [])


def _settings_view_sync() -> dict:
    from app.routes.dashboard import provider_icon, PROVIDER_PRESETS

    providers = []
    for name, p in ladder_config.providers.items():
        providers.append(
            {
                "name": name,
                "base_url": p.get("base_url", ""),
                "key_masked": _mask_key(p.get("api_key", "")),
                "models": p.get("models", []),
                "icon": provider_icon(name),
            }
        )
    presets = [{**preset, "icon": provider_icon(preset["name"])} for preset in PROVIDER_PRESETS]

    ollama = ladder_config.providers.get("ollama")
    options = _ollama_live_models(ollama.get("base_url", "")) if ollama else []

    return {
        "providers": providers,
        "presets": presets,
        "local_tail": {"current": ladder_config.local_tail, "options": options},
    }


@router.get("/_settings")
async def get_settings():
    import asyncio

    return await asyncio.to_thread(_settings_view_sync)


@router.post("/_settings")
async def post_settings(request: Request):
    body = await request.json()
    action = body.get("action")
    if action == "set_nvidia_key":
        key = body.get("key", "")
        ladder_config.set_nvidia_key(key)
        await _populate_from_live_catalog("nvidia")
    elif action == "add_provider":
        name = body.get("name", "").strip()
        if not name:
            return JSONResponse({"error": "name required"}, status_code=400)
        ladder_config.add_provider(
            name,
            base_url=body.get("base_url", ""),
            api_key=body.get("api_key", ""),
            models=body.get("models") or [],
        )
        await _populate_from_live_catalog(name)
    elif action == "remove_provider":
        name = body.get("name", "").strip()
        ladder_config.remove_provider(name)
    elif action == "set_local_tail":
        ladder_config.set_local_tail(body.get("model", ""))
    else:
        return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)

    import asyncio

    settings = await asyncio.to_thread(_settings_view_sync)
    return {"ok": True, **settings}
