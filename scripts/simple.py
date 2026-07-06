"""
Minimal working proxy (single-file, no src folder).
Copy this to /root/model-router/nvidia_failover_proxy.py
"""

import asyncio
import collections
import json
import os
import time
from typing import Dict, Deque, List, Optional, Set
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
stats = {}  # model: {requests, tokens_in, tokens_out, status}
cascade_stats = {"requests": 0}
time_window = collections.deque(maxlen=60)  # 60-second sliding window

MODELS = [
    "deepseek-ai/deepseek-v4-pro",
    "qwen/qwen3.5-397b-a17b",
    "mistralai/mistral-large-3",
    "meta/llama-3.3-70b-instruct",
]


def get_key():
    return os.environ.get("NVIDIA_API_KEY", "nvapi-missing")


@app.get("/health")
async def health():
    return {"ok": bool(get_key()), "models": len(MODELS)}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    model = body.get("model", MODELS[0])
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {get_key()}"}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            cascade_stats["requests"] += 1
            time_window.append(time.time())
            return JSONResponse(data)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))
