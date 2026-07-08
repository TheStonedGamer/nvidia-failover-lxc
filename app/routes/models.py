"""/v1/models plus provider model discovery (/_models/available). /v1/models
exposes every individual model each configured provider actually offers
(live-discovered, not just the curated failover ladder) alongside the
aggregate "auto"/"only"/agent-role models that still drive the cascade."""

from typing import List

from fastapi import APIRouter

from app.config import AUTO_MODEL, ONLY_MODEL, REFINER_MODEL_ID, LOCAL_ONLY, LOCAL_REFINE, AGENT_ROLES
from app.ladder import ladder_config
from app.state import cascade
from app.discovery import discover_all, all_discovered_models

router = APIRouter()


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

    await discover_all(ladder_config.providers)
    for m in all_discovered_models():
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


@router.get("/_models/available")
async def models_available():
    # Manually-triggered from the dashboard — always force a live check
    # rather than serving the /v1/models throttle-cached result.
    providers = ladder_config.providers
    results = await discover_all(providers, force=True)
    in_ladder = set(_known_models())
    return {"in_ladder": sorted(in_ladder), "providers": results}
