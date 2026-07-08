"""Dashboard shell, live SSE updates, and /stats — plus the provider icon
system shared with the settings panel (config_api.py imports provider_icon
and PROVIDER_PRESETS from here)."""

import asyncio
import time
from typing import Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import (
    NVIDIA_BASE_URL,
    AUTO_MODEL,
    ONLY_MODEL,
    REFINER_MODEL_ID,
    LOCAL_ONLY,
    LOCAL_REFINE,
    REFINER_MODEL,
    REFINER_ENABLE,
    REFINER_TAG,
    AGENT_ROLES,
    fmt_money,
    saved_usd,
    _RPM_WINDOW_S,
)
from app.db import DB_FILE
from app.ladder import ladder_config
from app.state import cascade, stats
from app.routes.models import _known_models

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/health")
async def health() -> dict:
    key = ladder_config.resolved_nvidia_key()
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
        "models": cascade._serving_ladder(),
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
    local_model = cascade._local_model()
    names = [m for m in _known_models() if ladder_config.is_enabled(m)]
    if local_model and local_model not in names:
        names.append(local_model)
    serving = stats.current_serving(now)
    idle_next = None
    if serving is None:
        _next = cascade.order(AUTO_MODEL)
        idle_next = _next[0] if _next else None
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
        if serving is not None:
            is_active = name == serving
        else:
            is_active = name == idle_next
        rows.append(
            {
                "model": name,
                "state": state,
                "active": is_active,
                "serving_now": serving is not None and name == serving,
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
                "saved_usd": saved_usd(
                    name, m.get("tokens_in", 0), m.get("tokens_out", 0)
                ),
            }
        )
    return rows


@router.get("/stats")
async def stats_json() -> dict:
    models = _model_view()
    return {
        "models": models,
        "window_s": int(_RPM_WINDOW_S),
        "db_file": DB_FILE,
        "saved_usd_total": round(sum(m.get("saved_usd", 0.0) for m in models), 4),
    }


def _fmt_tpm(val) -> str:
    return f"{int(val):,}"


def _fmt_ceiling(val) -> str:
    return f"{int(val):,}/min" if val is not None else "learning…"


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


_ACTIVE_DOT = '<span class="dot" title="active model — serves the next request"></span>'


def _dot(row: dict) -> str:
    if not row.get("active"):
        return ""
    if row.get("serving_now"):
        return '<span class="dot serving" title="currently serving requests"></span>'
    return _ACTIVE_DOT


_ROW_COLOR = {
    "live": "#2e7d32",
    "cooling": "#e65100",
    "dead": "#b71c1c",
    "local": "#1565c0",
    "disabled": "#5b6472",
}


def _totals_row(rows: list, tot: dict) -> str:
    return (
        f"<tr class=tot><td></td><td>TOTAL ({len(rows)} models)</td><td></td><td></td>"
        f"<td>{tot['requests']}</td><td>{tot['successes']}</td><td>{tot['rate_limited']}</td>"
        f"<td></td><td></td><td></td><td></td><td></td><td></td><td></td>"
        f"<td class=num>{_fmt_num(tot['tokens_in'])}</td><td class=num>{_fmt_num(tot['tokens_out'])}</td>"
        f"<td class=num>{_fmt_num(tot['tokens_total'])}</td>"
        f"<td class=num><span class=save>{fmt_money(tot['saved_usd'])}</span></td></tr>"
    )


def _rows_html(rows: list) -> tuple:
    """Build (trs_joined, tot) shared by the dashboard shell and SSE refresh."""
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
        badge = f'<span style="color:{_ROW_COLOR.get(r["state"], "#555")};font-weight:600">{r["state"].upper()}</span>'
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
        saved = f'<span class=save>{fmt_money(r["saved_usd"])}</span>' if r["saved_usd"] else "—"
        trs.append(
            f"<tr><td class=n>{i + 1}</td><td class=m>{_dot(r)}{r['model']}</td><td>{badge}</td>"
            f"<td>{avail}</td><td>{r['requests']}</td><td>{r['successes']}</td>"
            f"<td>{r['rate_limited']}</td><td>{r['live_rpm']} <span class=dim>(peak {r['peak_rpm']})</span></td>"
            f"<td>{limit_rpm}</td><td>{_fmt_ceiling(r['learned_limit_tpm_in'])}</td>"
            f"<td>{_fmt_ceiling(r['learned_limit_tpm_out'])}</td><td>{tpi}</td><td>{tpo}</td>"
            f"<td>{cd}</td>"
            f"<td class=num>{tin}</td><td class=num>{tout}</td><td class=num>{_fmt_num(r['tokens_total'])}</td>"
            f"<td class=num>{saved}</td></tr>"
        )
    return "".join(trs), tot


async def _live_tbody() -> str:
    rows = _model_view()
    trs, tot = _rows_html(rows)
    return trs + _totals_row(rows, tot)


@router.get("/updates")
async def updates(request: Request):
    async def watch_stats():
        while not await request.is_disconnected():
            await asyncio.sleep(0.5)
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


def sse_escape(s: str) -> str:
    return s.replace("\n", "\\n")




# --- Branding + provider icon system --------------------------------------
from app.branding import NV_LOGO_PNG as _NV_LOGO_PNG, NV_FAVICON as _NV_FAVICON

# The official NVIDIA logo, used as the app's own branding (header + favicon).
_NV_LOGO_SVG = f'<img class="nvlogo" alt="NVIDIA" src="{_NV_LOGO_PNG}">'


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


# --- Dashboard shell route --------------------------------------------------
@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    rows = _model_view()
    trs, tot = _rows_html(rows)
    foot = _totals_row(rows, tot)
    saved_hero = fmt_money(tot["saved_usd"])
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
    return templates.TemplateResponse(
        request,
        "dashboard.html.j2",
        {
            "nv_favicon": _NV_FAVICON,
            "nv_logo_svg": _NV_LOGO_SVG,
            "key_ok": key_ok,
            "prov_n": prov_n,
            "saved_hero": saved_hero,
            "cfg_html": cfg_html,
            "tbody_html": trs + foot,
        },
    )
