"""The main OpenAI-compatible chat endpoint: command interception, agent-role
injection, prompt refiner, per-model routing/headers, and the streaming
failover generator that stitches together attempts across the cascade
without ever splicing two partial answers into one client-visible stream."""

import json
import re
import time
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import (
    AUTO_MODEL,
    ONLY_MODEL,
    REFINER_MODEL_ID,
    LOCAL_ONLY,
    LOCAL_REFINE,
    AGENT_ROLES,
    NVIDIA_BASE_URL,
    _CONNECT_ERRORS,
    _CONNECT_COOLDOWN_S,
    _MODEL_COOLDOWN_S,
    _DEFAULT_MAX_TOKENS,
    SKIP_EMPTY,
    GUARD_DEGENERATE,
    GUARD_CJK_CODE,
    REFINER_ENABLE,
    REFINER_BASE_URL,
    REFINER_MODEL,
    REFINER_TAG,
    freq_penalty_for,
    strip_v1,
)
from app.guards.stall import StreamStall, STREAM_STALL_S
from app.guards.degenerate import degenerate_reason, looping_suffix
from app.guards.cjk import cjk_count, cjk_in_code_blocks, cjk_in_tool_calls, prompt_has_cjk
from app.ladder import ladder_config
from app.state import cascade, stats, get_model_override, set_model_override
from app.routes.commands import handle_command
from app.routes.models import inject_agent_prompt

router = APIRouter()

_CJK_SCAN_CAP = 262144


# --- routing -----------------------------------------------------------------
def _headers() -> Dict[str, str]:
    key = ladder_config.resolved_nvidia_key()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _route(model: str):
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


def _prep_body(body: dict, model: str) -> dict:
    out = dict(body)
    out["model"] = model
    out.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
    if "frequency_penalty" not in out:
        fp = freq_penalty_for(model)
        if fp:
            out["frequency_penalty"] = fp
    if out.get("stream"):
        so = dict(out.get("stream_options") or {})
        so["include_usage"] = True
        out["stream_options"] = so
    return out


# --- content-presence helpers --------------------------------------------------
def _msg_has_content(msg: dict) -> bool:
    if not msg:
        return False
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return True
    if msg.get("tool_calls"):
        return True
    return False


def _resp_has_content(data: dict) -> bool:
    for choice in (data or {}).get("choices", []):
        if _msg_has_content(choice.get("message") or {}):
            return True
    return False


def _delta_has_content(delta: dict) -> bool:
    if not delta:
        return False
    content = delta.get("content")
    if isinstance(content, str) and content:
        return True
    if delta.get("tool_calls"):
        return True
    return False


def _delta_is_active(delta: dict) -> bool:
    if _delta_has_content(delta):
        return True
    if not delta:
        return False
    if delta.get("reasoning_content") or delta.get("reasoning"):
        return True
    return False


def _delta_text(delta: dict) -> str:
    content = (delta or {}).get("content")
    return content if isinstance(content, str) else ""


def _approx_tokens_from_chars(n_chars: int) -> int:
    return max(0, round(n_chars / 4))


def _estimate_usage(body: dict, out_chars: int) -> dict:
    prompt_chars = sum(len(str(m.get("content") or "")) for m in body.get("messages") or [])
    tin = _approx_tokens_from_chars(prompt_chars)
    tout = _approx_tokens_from_chars(out_chars)
    return {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}


# --- prompt refiner ------------------------------------------------------------
REFINER_SYSTEM = (
    "You rewrite the user's most recent message into a clearer, more complete, "
    "unambiguous version of the same request. Preserve intent exactly. Output "
    "only the rewritten prompt, nothing else."
)

_REFINE_TAG_RE = None


def _has_refine_tag(text: str) -> bool:
    return bool(text) and REFINER_TAG.lower() in text.lower()


def _strip_refine_tag(text: str) -> str:
    global _REFINE_TAG_RE
    if _REFINE_TAG_RE is None:
        _REFINE_TAG_RE = re.compile(re.escape(REFINER_TAG), re.IGNORECASE)
    return _REFINE_TAG_RE.sub("", text).strip()


async def _refine_prompt(body: dict) -> dict:
    messages = body.get("messages") or []
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return body

    original = messages[last_user_idx].get("content")
    if not isinstance(original, str):
        return body
    stripped = _strip_refine_tag(original)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                await client.post(
                    f"{strip_v1(REFINER_BASE_URL)}/api/generate",
                    json={"model": REFINER_MODEL, "prompt": "", "keep_alive": 0},
                )
            except Exception:
                pass
            resp = await client.post(
                f"{REFINER_BASE_URL}/chat/completions",
                json={
                    "model": REFINER_MODEL,
                    "messages": [
                        {"role": "system", "content": REFINER_SYSTEM},
                        {"role": "user", "content": stripped},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            refined = data["choices"][0]["message"]["content"]
    except Exception:
        return body

    out = dict(body)
    out_messages = list(messages)
    out_messages[last_user_idx] = {**out_messages[last_user_idx], "content": refined}
    out["messages"] = out_messages
    return out


# --- SSE helpers ---------------------------------------------------------------
async def _sse_once(payload: bytes):
    yield payload


def _sse_error(message: str) -> bytes:
    payload = json.dumps({"error": {"message": message}})
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode("utf-8")


def sse_escape(s: str) -> str:
    return s.replace("\n", "\\n").replace("\r", "")


def _model_timeout() -> httpx.Timeout:
    import os

    read_s = float(os.environ.get("PROXY_MODEL_TIMEOUT_S", "300"))
    return httpx.Timeout(connect=15.0, read=read_s, write=30.0, pool=15.0)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)

    messages = body.get("messages") or []
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user = m["content"]
            break

    cmd_response = handle_command(last_user)
    if cmd_response is not None:
        return cmd_response

    preferred = body.get("model")
    if preferred in (None, "", AUTO_MODEL):
        override = get_model_override()
        if override:
            preferred = override

    body = inject_agent_prompt(body, preferred)

    if preferred in (None, "", AUTO_MODEL) or _has_refine_tag(last_user):
        if _has_refine_tag(last_user) and REFINER_ENABLE:
            body = await _refine_prompt(body)
            preferred = LOCAL_REFINE if preferred == LOCAL_REFINE else AUTO_MODEL

    effective_model = preferred or AUTO_MODEL
    ladder = cascade.order(effective_model)
    if not ladder:
        return JSONResponse(
            {"error": {"message": "all models cooling down"}},
            status_code=429,
            headers={"Retry-After": str(int(cascade.soonest_cooldown() or 30))},
        )

    if body.get("stream"):
        return StreamingResponse(
            _stream_cascade(body, ladder), media_type="text/event-stream"
        )

    timeout = _model_timeout()
    tried = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in ladder:
            tried.append(model)
            base_url, headers = _route(model)
            req_body = _prep_body(body, model)
            stats.record_request(model)
            attempts = 2 if not cascade.is_local(model) else 1
            for attempt in range(attempts):
                try:
                    resp = await client.post(f"{base_url}/chat/completions", json=req_body, headers=headers)
                except _CONNECT_ERRORS:
                    if attempt == 0 and attempts > 1:
                        continue
                    cascade.cool(model, _CONNECT_COOLDOWN_S)
                    break
                if resp.status_code in (401, 402, 403):
                    return JSONResponse(
                        {"error": {"message": f"{model}: account-gated ({resp.status_code})"}},
                        status_code=resp.status_code,
                    )
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    cascade.note_status(model, 429, float(retry_after) if retry_after else None)
                    break
                if resp.status_code >= 400:
                    cascade.note_status(model, resp.status_code)
                    cascade.cool(model, _MODEL_COOLDOWN_S)
                    break

                data = resp.json()
                usage = data.get("usage") or {}
                stats.record_usage(model, usage)

                if not _resp_has_content(data):
                    stats.record_error(model)
                    break

                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                reason = choice.get("finish_reason")
                text = msg.get("content") or ""
                if GUARD_DEGENERATE or GUARD_CJK_CODE:
                    bad = degenerate_reason(text, reason, body, msg)
                    if bad:
                        stats.record_error(model)
                        cascade.cool(model, _CONNECT_COOLDOWN_S)
                        break

                stats.record_success(model)
                cascade.note_status(model, resp.status_code)
                stats.note_serving(model)
                data["_proxy_model"] = model
                return JSONResponse(data)
            else:
                continue
            continue

    return JSONResponse(
        {"error": {"message": "all frontier models unavailable", "tried": tried}},
        status_code=502,
    )


async def _stream_cascade(body: dict, ladder: List[str]):
    timeout = _model_timeout()
    client_sent = False

    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in ladder:
            base_url, headers = _route(model)
            req_body = _prep_body(body, model)
            stats.record_request(model)

            committed = not SKIP_EMPTY
            buffered: List[bytes] = []
            out_chars = 0
            rep_tail = ""
            cjk_scan_buf = ""
            cjk_code_seen = False
            _stream_usage = None
            last_content = time.monotonic()
            connect_retried = False

            try:
                async with client.stream("POST", f"{base_url}/chat/completions", json=req_body, headers=headers) as resp:
                    if resp.status_code != 200:
                        raw = await resp.aread()
                        if resp.status_code in (401, 402, 403):
                            yield _sse_error(f"{model}: account-gated ({resp.status_code})")
                            return
                        retry_after = resp.headers.get("Retry-After")
                        cascade.note_status(model, resp.status_code, float(retry_after) if retry_after else None)
                        if resp.status_code >= 400 and resp.status_code != 429:
                            cascade.cool(model, _MODEL_COOLDOWN_S)
                        continue

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if time.monotonic() - last_content > STREAM_STALL_S and committed:
                            raise StreamStall()

                        if line.startswith("data: "):
                            payload = line[len("data: "):]
                            if payload.strip() == "[DONE]":
                                if committed:
                                    for b in buffered:
                                        yield b
                                    buffered = []
                                    yield b"data: [DONE]\n\n"
                                    client_sent = True
                                break
                            try:
                                chunk = json.loads(payload)
                            except Exception:
                                if committed:
                                    yield (line + "\n\n").encode("utf-8")
                                    client_sent = True
                                continue

                            if chunk.get("usage"):
                                _stream_usage = chunk["usage"]

                            choices = chunk.get("choices") or []
                            delta = (choices[0].get("delta") if choices else {}) or {}
                            text = _delta_text(delta)
                            out_chars += len(text)

                            if GUARD_DEGENERATE and text:
                                rep_tail = (rep_tail + text)[-2000:]
                                if committed and looping_suffix(rep_tail):
                                    if _stream_usage:
                                        stats.record_usage(model, _stream_usage)
                                    else:
                                        stats.record_usage(model, _estimate_usage(body, out_chars))
                                    stats.record_success(model)
                                    cascade.note_status(model, 200)
                                    cascade.cool(model, _CONNECT_COOLDOWN_S)
                                    for b in buffered:
                                        yield b
                                    yield b"data: [DONE]\n\n"
                                    client_sent = True
                                    return

                            if GUARD_CJK_CODE and not prompt_has_cjk(body.get("messages") or []):
                                tool_calls = delta.get("tool_calls")
                                if tool_calls and cjk_in_tool_calls(tool_calls, 2):
                                    cjk_code_seen = True
                                elif text:
                                    cjk_scan_buf = (cjk_scan_buf + text)[-_CJK_SCAN_CAP:]
                                    if cjk_in_code_blocks(cjk_scan_buf, 2):
                                        cjk_code_seen = True
                                if cjk_code_seen and committed:
                                    if _stream_usage:
                                        stats.record_usage(model, _stream_usage)
                                    else:
                                        stats.record_usage(model, _estimate_usage(body, out_chars))
                                    stats.record_success(model)
                                    cascade.note_status(model, 200)
                                    cascade.cool(model, _CONNECT_COOLDOWN_S)
                                    for b in buffered:
                                        yield b
                                    yield b"data: [DONE]\n\n"
                                    client_sent = True
                                    return

                            if _delta_is_active(delta):
                                last_content = time.monotonic()

                            line_bytes = (line + "\n\n").encode("utf-8")
                            if not committed:
                                if _delta_has_content(delta):
                                    committed = True
                                    stats.note_serving(model)
                                    for b in buffered:
                                        yield b
                                    buffered = []
                                    yield line_bytes
                                    client_sent = True
                                else:
                                    buffered.append(line_bytes)
                            else:
                                yield line_bytes
                                client_sent = True
                        else:
                            if committed:
                                yield (line + "\n\n").encode("utf-8")
                                client_sent = True

                    if _stream_usage:
                        stats.record_usage(model, _stream_usage)
                    elif committed:
                        stats.record_usage(model, _estimate_usage(body, out_chars))

                    if committed:
                        stats.record_success(model)
                        cascade.note_status(model, 200)
                        return
                    else:
                        stats.record_error(model)
                        continue

            except StreamStall:
                if client_sent:
                    yield b"data: [DONE]\n\n"
                    return
                cascade.cool(model, _CONNECT_COOLDOWN_S)
                continue
            except _CONNECT_ERRORS:
                if not connect_retried:
                    connect_retried = True
                    continue
                cascade.cool(model, _CONNECT_COOLDOWN_S)
                continue
            except httpx.HTTPError:
                if client_sent:
                    yield b"data: [DONE]\n\n"
                    return
                cascade.cool(model, _MODEL_COOLDOWN_S)
                continue

    yield _sse_error("all frontier models unavailable")
