# NVIDIA frontier-model failover proxy

A tiny OpenAI-compatible endpoint that forwards to NVIDIA's hosted API and
**auto-cycles through the frontier models**. When a model is rate-limited (429)
or dead (404/410), the same request transparently fails over to the next model
in the ladder. Works for streaming and non-streaming requests.

File: `nvidia_failover_proxy.py`

## Two selectable models

- **`nvidia-auto`** — cloud frontier ladder, then the local Qwen3-Coder-Next-80B
  as the tail rung. When the whole cloud tier is rate-limited it keeps going
  locally instead of failing.
- **`nvidia-only`** — the same frontier ladder with **no local fallback**. When
  every cloud model is rate-limited it returns a real `429`
  (`all NVIDIA frontier models are rate-limited; retry in ~Ns`, with a
  `Retry-After` header) instead of dropping to local.

Both appear side by side in OpenCode's model picker.

## Learned rate limits + dashboard

The proxy **learns each model's rate limits from live traffic** and shows them:

- **`http://127.0.0.1:5002/`** — an auto-refreshing (4s) HTML dashboard: per-model
  state (live / cooling / dead / local), requests, successes, 429s, current rpm
  (and peak), the **learned requests-per-minute ceiling**, the **learned
  cooldown**, and **cumulative tokens in / out / total** (with the last request's
  token counts). A TOTAL row sums requests, 429s, and tokens across all models.
  Columns read "learning…" until a model has actually been throttled.
- **`GET /stats`** — the same data as JSON.

Token usage is read from each response's `usage` block. Streaming requests get
`stream_options.include_usage` set automatically so the final usage chunk is
captured. Token totals persist to `proxy_stats.json` alongside the learned limits.

How it learns: the rpm observed at the moment a `429` fires is an upper bound on
the real limit, EWMA'd so it converges (`learned limit`). The cooldown comes from
the `Retry-After` header when present, else the measured gap between a 429 and the
next success. Once learned, a 429 **without** a `Retry-After` header sidelines the
model for the learned duration instead of the flat 5-minute default. State
persists to `proxy_stats.json` (override with `PROXY_STATS_FILE`) so learning
survives restarts.

## Local tail rung + context continuity

A **local Ollama model sits at the end of the ladder** (default
`Qwen3-Coder-Next-80b-A3B:latest`). When the entire NVIDIA frontier tier is
rate-limited for the time being, the request keeps going on your local model
instead of failing, and re-picks the cloud models as soon as their cooldowns
expire. Disable with `PROXY_LOCAL_FALLBACK=0`; point elsewhere with
`LOCAL_OLLAMA_URL` / `LOCAL_MODEL`.

**Context is preserved across every hop.** The proxy is stateless — OpenCode
resends the full `messages` history each turn and the proxy forwards it verbatim
to whichever model serves. A conversation can start on a cloud frontier model,
fail over to another, and land on the local 80B, carrying its full context the
whole way.

## Run

```powershell
# key resolves from $env:NVIDIA_API_KEY, else from ~/.config/opencode/opencode.jsonc
E:\Projects\model-router\.venv\Scripts\python.exe E:\Projects\model-router\nvidia_failover_proxy.py
```

Optional env:
- `PROXY_PORT` (default `5002`)
- `ROUTER_NVIDIA_MODELS` — comma-separated ladder override (else the built-in
  frontier list, strongest first)
- `PROXY_TIMEOUT_S` (default `300`)

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`
(`/health` shows the ladder plus which models are currently cooling/dead).

## One-click launcher (proxy + OpenCode GUI)

Double-click the **`Router + OpenCode`** desktop shortcut (created by
`make_shortcut.ps1`, icon = OpenCode). It runs `launch_hidden.vbs` →
`launch_router_opencode.ps1` **fully hidden** (no console window):

1. Starts the router proxy on `:5002` (skips if already running).
2. Waits for `/health` to go green.
3. Opens the **OpenCode desktop app**
   (`%LOCALAPPDATA%\Programs\@opencode-aidesktop\OpenCode.exe`), already
   configured for the `nvidia-failover` provider.
4. Waits for OpenCode to fully close, then **stops the proxy** — the router
   never lingers after you quit OpenCode.

The config default model in `~/.config/opencode/opencode.jsonc` is
`nvidia-failover/nvidia-auto`. To revert to the pure local model, set it back to
`ollama/Qwen3-Coder-Next-80b-A3B`.

## Point OpenCode at it

In `~/.config/opencode/opencode.jsonc`, add a custom OpenAI-compatible provider:

```jsonc
{
  "provider": {
    "nvidia-failover": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NVIDIA Failover",
      "options": {
        "baseURL": "http://127.0.0.1:5002/v1",
        "apiKey": "unused"   // real nvapi- key is resolved server-side
      },
      "models": {
        "nvidia-auto": { "name": "NVIDIA auto (frontier cascade)" }
      }
    }
  }
}
```

Then select the `nvidia-auto` model. Requests start at the top of the frontier
ladder and cascade down on rate limits. You can also select any real NVIDIA
model id — it's tried first, then the cascade takes over.

## Frontier ladder (default, strongest first)

deepseek-v4-pro → qwen3.5-397b → mistral-large-3-675b → nemotron-3-ultra-550b →
kimi-k2.6 → glm-5.2 → minimax-m3 → gpt-oss-120b → nemotron-ultra-253b →
qwen3.5-122b → nemotron-3-super-120b → mistral-medium-3.5 → llama-4-maverick →
deepseek-v4-flash → llama-3.3-70b

Verified live 2026-07-06: non-streaming, streaming, and forced failover all work;
served responses tag the actual model in `_proxy_model` and proxy stdout logs
`[proxy] served by <model>`.
