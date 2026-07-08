"""In-chat slash commands: /help /stats /models /health /cool /uncool
/switch /warm /unload /pick. Intercepted before the request reaches the
cascade — ported verbatim from nvidia_failover_proxy.py's _handle_command."""

import time
from typing import Optional

import httpx
from fastapi.responses import JSONResponse

from app.config import (
    AGENT_ROLES,
    AUTO_MODEL,
    LOCAL_ONLY,
    LOCAL_REFINE,
    ONLY_MODEL,
    REFINER_ENABLE,
    REFINER_MODEL,
    REFINER_MODEL_ID,
    _MODEL_COOLDOWN_S,
    resolve_api_key,
    strip_v1,
)
from app.ladder import ladder_config
from app.state import cascade, get_model_override, set_model_override
from app.routes.dashboard import _fmt_dur, _fmt_tpm, _model_view


def _fmt_header(text: str) -> str:
    return f"\n━━━ {text} ━━━\n\n"


def _fmt_bold(text: str) -> str:
    return f"**{text}**"


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
    for m in cascade._serving_ladder():
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


def _cmd_response(msg: str) -> JSONResponse:
    return JSONResponse(
        {
            "id": f"cmd-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "proxy-commands",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": msg.strip()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


def handle_command(text: str) -> Optional[JSONResponse]:
    """Check if the last user message starts with / and handle it.

    Returns a JSONResponse (command handled) or None (pass through).
    """
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
        if model in cascade._serving_ladder():
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
            target = " ".join(args) if args else ollama_models[0]
            if target not in ollama_models:
                known = ", ".join(ollama_models)
                msg = f"  ✗ **{target}** is not in the Ollama model list.\n    Known Ollama models: {known}"
            else:
                ollama_url = (
                    ollama_prov.get("base_url", "http://127.0.0.1:11434/v1") or ""
                ).strip()
                native_base = strip_v1(ollama_url)
                keep_alive = -1 if cmd == "warm" else 0
                action = "warming" if cmd == "warm" else "unloading"
                try:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.post(
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
        lm = cascade._local_model()
        all_models = (
            [AUTO_MODEL, ONLY_MODEL, REFINER_MODEL_ID, LOCAL_ONLY, LOCAL_REFINE]
            + sorted(AGENT_ROLES)
            + cascade._serving_ladder()
            + ([lm] if lm else [])
        )
        now = time.time()
        current_override = get_model_override()

        if args and args[0].isdigit():
            idx = int(args[0]) - 1
            if 0 <= idx < len(all_models):
                picked = all_models[idx]
                set_model_override(picked)
                msg = f"  ✓ Picked **{picked}**\n  (Override set for subsequent messages. Use `/pick off` to clear.)"
            else:
                msg = f"  ✗ Invalid number. Use /pick to see available models (1-{len(all_models)})."
        elif args and args[0].lower() == "off":
            set_model_override(None)
            msg = "  ✓ Model override cleared — using default model."
        else:
            current = current_override or "(default)"
            lines = [_fmt_header(f"Model Picker — current: {current}")]
            lines.append("  Type `/pick N` to select, `/pick off` to clear:\n")
            for idx, m in enumerate(all_models, 1):
                if m == current_override:
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

    return _cmd_response(msg)
