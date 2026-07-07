import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

import httpx

from src.llama_backend import moe_backend

app = FastAPI(title="Model-Router MOE Proxy API Server", version="0.3.0")


class ModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int = 1686935002
    owned_by: str = "local-platform"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelItem]


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Returns the orchestrator + all registered expert models."""
    models = await moe_backend.get_models()
    return ModelList(
        data=[ModelItem(id=m["id"], owned_by=m.get("alias", "general")) for m in models]
    )


@app.get("/v1/experts")
async def list_experts():
    """Admin endpoint: shows expert state (loaded/ports) for debugging swaps."""
    return {"experts": moe_backend.registry.list_experts()}


@app.post("/v1/experts/{expert_name}/unload")
async def unload_expert(expert_name: str):
    """Manually unload an expert to free VRAM."""
    state = moe_backend.registry.get_expert_by_name(expert_name)
    if state is None:
        raise HTTPException(
            status_code=404, detail=f"Expert '{expert_name}' not registered"
        )
    await moe_backend.registry._unload(state)
    return {"status": "unloaded", "expert": expert_name}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """MOE-aware proxy endpoint.

    The model field in the body selects the routing target:
      - "qwen3-4b" / "orchestrator": routed to the small orchestrator on CPU
      - "builder" / "planner" / "reviewer" / expert name: triggers a VRAM
        swap if needed, then forwards to the expert's dedicated llama-server.
    """
    body = await request.json()

    model_name = body.get("model", "qwen3-4b")
    stream = body.get("stream", False)
    messages = body.get("messages", [])

    print(f"[MOE Proxy] Request for model '{model_name}' (Stream: {stream})")

    target_role = _infer_target_role(model_name)

    if stream:
        return StreamingResponse(
            _stream_response(messages, target_role),
            media_type="text/event-stream",
        )

    result = await moe_backend.route_request(messages, target_role=target_role)
    if "error" in result:
        raise HTTPException(
            status_code=result.get("status_code", 500), detail=result["error"]
        )
    return result


async def _stream_response(messages: list, target_role: str):
    """Stream a response from either the orchestrator or the active expert."""
    # Resolve which backend + model this request goes to
    if target_role == "orchestrator":
        base_url = moe_backend.orchestrator_url
        model_name = moe_backend.orchestrator_model
        temperature = 0.3
    else:
        expert_state = await moe_backend.registry.ensure_expert_loaded(target_role)
        if expert_state is None or not expert_state.loaded:
            print(
                f"[MOE Proxy] No expert for role '{target_role}', streaming from orchestrator"
            )
            base_url = moe_backend.orchestrator_url
            model_name = moe_backend.orchestrator_model
            temperature = 0.1
        else:
            base_url = f"http://localhost:{expert_state.port}/v1"
            model_name = expert_state.expert.name
            messages = await moe_backend._inject_context(messages, target_role)
            temperature = 0.1

    print(
        f"[MOE Proxy] Streaming from {target_role} via {base_url} (model={model_name})"
    )

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True,
                },
                timeout=300.0,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    err_msg = f"Downstream {response.status_code}: {body.decode(errors='replace')}"
                    yield f"data: {json.dumps({'error': {'message': err_msg}})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            parsed = json.loads(data)
                            parsed["model"] = model_name
                            yield f"data: {json.dumps(parsed)}\n\n"
                        except json.JSONDecodeError:
                            yield f"{line}\n"
                    else:
                        yield f"{line}\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': {'message': f'Stream Error: {e}'}})}\n\n"


def _infer_target_role(model_name: str) -> str:
    """Map a requested model string to a routing role.

    Recognized patterns:
      - orchestrator, qwen3-4b, anything with "4b" or "router" -> orchestrator
      - planner / "plan"                                  -> planner
      - builder / "build" / "coder" / "80b"               -> builder
      - reviewer / "review" / "verify"                    -> reviewer
      - expert name (e.g. "builder-80b")                  -> passed through to
                                                            registry name lookup
    """
    m = model_name.lower()

    if "orchestrator" in m or "router" in m or m == "qwen3-4b" or "4b" in m:
        return "orchestrator"
    if "planner" in m or "plan" in m:
        return "planner"
    if "builder" in m or "build" in m or "coder" in m or "80b" in m:
        return "builder"
    if "reviewer" in m or "review" in m or "verify" in m:
        return "reviewer"
    # Unknown - try as an exact expert name (registry will fall back if not found)
    return model_name


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5001)
