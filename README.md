# NVIDIA Failover Proxy

An OpenAI-compatible proxy that cascades requests across many model providers and
**automatically fails over** when a model is rate-limited (429), times out, or dies.
Point any OpenAI-compatible client (OpenCode, Continue, the `openai` SDK, `curl`) at
it and get resilient, multi-provider inference behind a single endpoint.

- **Multi-provider** — NVIDIA, OpenAI, Anthropic, Cerebras, Groq, OpenRouter, Mistral,
  DeepSeek, xAI, Together, or **any** OpenAI-compatible API. NVIDIA is just another
  provider; **there are no providers by default** — you add them in the web UI.
- **Learned rate limits** — the proxy learns each model's RPM/TPM ceiling and cooldown
  from live 429s and proactively routes around models about to throttle.
- **Local tail rung** — an optional local Ollama model as the guaranteed last resort
  when the whole cloud tier is cooling.
- **Live dashboard** — SSE dashboard with per-model state, token accounting, an estimated
  **money-saved** figure (what the same tokens would cost at a commercial API), drag-and-drop
  failover ladder, model toggles, provider management (with brand icons), and live model
  discovery.
- **SQLite persistence** — provider config, API keys, and learned stats live in a single
  `proxy.db` (WAL, `chmod 600`).

---

## Quick start

### Docker (recommended)

```bash
docker run -d --name nvidia-failover-proxy \
  -p 5002:5002 \
  -v proxy-data:/data \
  shinyjesus/nvidia-failover-proxy:latest
```

Then open **http://localhost:5002/** and add a provider under
**🔑 Providers & API keys** (click a preset — NVIDIA, OpenAI, Anthropic, Cerebras, … —
then paste your API key and the model ids you want).

Optionally seed an NVIDIA provider at first run:

```bash
docker run -d --name nvidia-failover-proxy \
  -p 5002:5002 -v proxy-data:/data \
  -e NVIDIA_API_KEY=nvapi-xxxxxxxx \
  shinyjesus/nvidia-failover-proxy:latest
```

### Docker Compose

```bash
# optionally: export NVIDIA_API_KEY=nvapi-xxxx
docker compose up -d
```

See [`docker-compose.yml`](docker-compose.yml) — the `proxy-data` volume keeps your
providers, keys, and learned stats across upgrades.

### Run from source

```bash
pip install -r requirements.txt
PROXY_HOST=0.0.0.0 python nvidia_failover_proxy.py
# → http://0.0.0.0:5002/v1
```

### Proxmox VE (helper script)

Run [`scripts/proxmox-lxc.sh`](scripts/proxmox-lxc.sh) **on a Proxmox VE host** (as
root). It's an interactive helper — it creates an unprivileged Debian LXC, installs
the proxy as a systemd service bound to `0.0.0.0`, and optionally seeds any provider
API keys — with sensible defaults (next free CTID, DHCP, 2 core / 2 GB / 6 GB):

```bash
bash -c "$(wget -qLO - https://raw.githubusercontent.com/TheStonedGamer/nvidia-failover-lxc/main/scripts/proxmox-lxc.sh)"
```

It prompts (via whiptail) for container specs and, optionally, an API key for each
major provider. For unattended runs, set `AUTO=1` and export the vars you want
(`CT_ID`, `CT_NET`, `CT_STORAGE`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, …). The older
workstation-driven [`scripts/deploy.sh`](scripts/deploy.sh) (push from your machine
over SSH) is still available.

**Updating an existing container** — pull the latest proxy code into a running LXC
and restart the service (run on the PVE host). It syntax-checks the new code before
bouncing the service and keeps a `.bak` for rollback:

```bash
bash -c "$(wget -qLO - https://raw.githubusercontent.com/TheStonedGamer/nvidia-failover-lxc/main/scripts/proxmox-lxc.sh)" -- update <CTID>
```

It auto-detects the install location (`/opt/nvidia-failover` for helper installs,
`/root/model-router` for the legacy `deploy.sh` layout).

### Seeding providers from environment variables

Any of the major OpenAI-compatible providers can be pre-configured at first run by
setting `<PROVIDER>_API_KEY` — no web-UI step needed. Recognized prefixes:
`NVIDIA`, `OPENAI`, `ANTHROPIC`, `CEREBRAS`, `GROQ`, `OPENROUTER`, `MISTRAL`,
`DEEPSEEK`, `GOOGLE`, `XAI`, `TOGETHER`. Optional companions per provider:

| Variable | Purpose |
| --- | --- |
| `<PREFIX>_API_KEY` | seeds that provider on first run (if not already configured) |
| `<PREFIX>_MODELS` | comma-separated model ids to add to the failover ladder |
| `<PREFIX>_BASE_URL` | override the default endpoint (e.g. a self-hosted gateway) |

```bash
docker run -d --name nvidia-failover-proxy -p 5002:5002 -v proxy-data:/data \
  -e OPENAI_API_KEY=sk-... -e OPENAI_MODELS="gpt-5,gpt-5-mini" \
  -e GROQ_API_KEY=gsk-... -e GROQ_MODELS="llama-3.3-70b-versatile" \
  shinyjesus/nvidia-failover-proxy:latest
```

Seeding only fires when a provider isn't already configured, so the web UI remains
the source of truth once you've added or edited it there.

### Standalone installers (no Python required)

Prebuilt, self-contained builds are produced by GitHub Actions
([`.github/workflows/build-installers.yml`](.github/workflows/build-installers.yml))
and attached to the [Releases](../../releases) page. **Every push to `main`
auto-publishes a rolling [`latest`](../../releases/tag/latest) prerelease** (the tag
is moved to the newest build); pushing a `v*` tag publishes a permanent versioned
release:

| Platform | Download | Notes |
| --- | --- | --- |
| **Windows** | `nvidia-failover-proxy-setup-windows-x64.exe` (installer) or the bare `…-windows-x64.exe` | Inno Setup installer with Start-menu shortcuts; the bare exe is portable |
| **Linux** | `nvidia-failover-proxy-linux-x64.AppImage` | `chmod +x` then run; bundles its own Python |

Each is a single executable that starts the proxy on `http://localhost:5002/`.
The SQLite store (`proxy.db`) is written next to the executable unless you set
`PROXY_DB_FILE`. To build them yourself locally:

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --onefile --name nvidia-failover-proxy --collect-submodules uvicorn nvidia_failover_proxy.py
# → dist/nvidia-failover-proxy
```

---

## Publishing the Docker image (CI)

[`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml) builds a
multi-arch (`amd64` + `arm64`) image and pushes it to Docker Hub on every push to `main`
and every `v*` tag. To enable it, add two **repository secrets**
(Settings → Secrets and variables → Actions):

| Secret | Value |
| --- | --- |
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | a Docker Hub **access token** (Account Settings → Security → New Access Token) |

The image is published as `<DOCKERHUB_USERNAME>/nvidia-failover-proxy` (currently
[`shinyjesus/nvidia-failover-proxy`](https://hub.docker.com/r/shinyjesus/nvidia-failover-proxy)),
tagged `latest`, the branch name, the short commit SHA, and semver tags for `v*`
releases. When the secrets are absent the workflow still builds the image (to prove
it compiles) but skips the push instead of failing. Pull requests build but do
**not** push.

---

## Using the proxy

It speaks the OpenAI API. Use any model id you've added to the ladder, or one of the
special routing ids below.

```bash
curl http://localhost:5002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia-auto","messages":[{"role":"user","content":"hi"}]}'
```

**Special model ids** (routing modes, not real models):

| id | behavior |
| --- | --- |
| `nvidia-auto` | full cloud ladder, then the local model as the tail rung |
| `nvidia-only` | cloud ladder only — returns `429` when everything is cooling |
| `nvidia-refine` | cloud ladder with a local prompt-refiner pass first |
| `local-only` | route straight to the local Ollama model |
| `local-refine` | refiner → local model |
| `agent-planner` / `agent-builder` / `agent-reviewer` | cloud ladder with a role system prompt injected |

Slash commands work inside a message too: `/help`, `/stats`, `/models`, `/health`,
`/cool <model>`, `/uncool <model>`, `/pick <n>`.

### Point OpenCode at it

```jsonc
// ~/.config/opencode/opencode.jsonc
{
  "provider": {
    "nvidia-failover": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:5002/v1", "apiKey": "unused" },
      "models": { "nvidia-auto": {}, "nvidia-only": {} }
    }
  }
}
```

---

## Configuration

Most configuration is done **in the web UI** and persisted to `proxy.db`. Environment
variables set defaults and infrastructure:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROXY_HOST` | `127.0.0.1` | bind address (Docker sets `0.0.0.0`) |
| `PROXY_PORT` | `5002` | listen port |
| `PROXY_DB_FILE` | `./proxy.db` (`/data/proxy.db` in Docker) | SQLite store for config + stats |
| `<PREFIX>_API_KEY` | — | seed any major provider on first run — see [Seeding providers from environment variables](#seeding-providers-from-environment-variables) (`NVIDIA`, `OPENAI`, `ANTHROPIC`, `GROQ`, …) |
| `<PREFIX>_MODELS` / `<PREFIX>_BASE_URL` | — | model ids / endpoint override for a seeded provider |
| `ROUTER_NVIDIA_MODELS` | built-in frontier list | comma-separated NVIDIA model ids (alias for `NVIDIA_MODELS`) |
| `LOCAL_OLLAMA_URL` | `http://10.0.0.142:11434/v1` | local Ollama base URL for the tail rung |
| `LOCAL_MODEL` | `Qwen3-Coder-Next-80b-A3B:latest` | local model id |
| `PROXY_LOCAL_FALLBACK` | `1` | set `0` to disable the local tail rung |
| `PROXY_MODEL_TIMEOUT_S` | `90` | per-model read timeout (fully silent socket) before failing over |
| `PROXY_STREAM_STALL_S` | `60` | max gap with no new content tokens mid-stream before the stream is treated as hung — fails over if nothing was sent yet, else ends the stream cleanly. Only armed after the first token so slow-starting reasoning models aren't cut |
| `PROXY_STICKY_LADDER` | `1` | keep serving from the model that last served and only roll forward (wrapping) through the ladder as models rate-limit, instead of restarting at the top of the ladder every request. Set `0` for strict top-priority ordering. An explicit model id in the request always overrides the cursor |
| `PROXY_SERVING_WINDOW_S` | `20` | how long (seconds) after its last content delta a model still counts as "currently serving" for the dashboard's green dot before it falls back to marking the model that would serve the next request |
| `PROXY_MAX_TOKENS_DEFAULT` | `8192` | `max_tokens` used when the client sends none (a client value always wins). Safe floor — raising it makes lower-cap models `400` and burn a failover hop per request; only bump it if every model in your ladder supports a larger output window |
| `PROXY_PRICING_JSON` | built-in table | JSON object of `{"<model-substring>": [in_per_1M, out_per_1M]}` (USD) used for the dashboard's estimated **money-saved** figure. Matched case-insensitively by substring, first hit wins. Overrides/extends the built-in per-family rates |
| `PROXY_PRICING_DEFAULT` | `0.50,1.50` | `in,out` USD per 1M tokens for models not matched by the pricing table |
| `PROXY_GUARD_DEGENERATE` | `1` | fail over when a model degenerates — `finish_reason=repetition` or an unexpected **CJK code-switch** (e.g. kimi drifting into Chinese on long coding prompts). Streaming is truncated cleanly *before* the CJK reaches the client |
| `PROXY_CJK_MIN_CHARS` | `4` | how many unexpected CJK chars (in a non-CJK prompt) count as a code-switch. Cumulative, not a run — kimi interleaves Chinese with Latin/code, which defeats run/fraction thresholds. Genuine CJK requests are never touched (guard keys off the prompt) |
| `PROXY_FREQ_PENALTY_JSON` / `PROXY_FREQ_PENALTY_DEFAULT` | `{"kimi-k2":0.3}` / `0` | per-model / global `frequency_penalty` injected when the client sends none (mild repetition mitigation) |
| `PROXY_RPM_LIMIT_FLOOR` / `PROXY_TPM_IN_LIMIT_FLOOR` / `PROXY_TPM_OUT_LIMIT_FLOOR` | `5` / `8000` / `2000` | floors under the *learned* rate-limit ceilings when deciding to skip a model. A 429 under low traffic (a quota, not a rate limit) would otherwise teach a ceiling so low the model is skipped forever |
| `REFINER_BASE_URL` / `REFINER_MODEL` | Ollama / `qwen3:4b` | prompt-refiner endpoint |
| `PROXY_REFINER_ENABLE` | `1` | enable the `[refine]` prompt refiner |

### Web UI

- **⚙ Failover ladder** — drag to reorder, uncheck to disable. The order is the real
  cascade order; disabled models never receive traffic and are hidden from the metrics.
- **🔑 Providers & API keys** — one-click presets for NVIDIA/OpenAI/Anthropic/Cerebras/…
  (each with a brand icon), add/update/remove providers, and set API keys. Re-adding a
  provider with a blank key/URL updates just the fields you change.
- **Discover available models** — live-queries each provider's `/v1/models`; click ＋ to
  add a discovered model to the ladder.

### Endpoints

| Path | Description |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `GET /v1/models` | model list |
| `GET /` · `GET /dashboard` | live SSE dashboard |
| `GET /health` | health + mode info |
| `GET /stats` | JSON metrics (reports `db_file`) |
| `GET/POST /_config` | failover order + toggles |
| `GET/POST /_settings` | providers + keys (keys returned masked) |
| `GET /_models/available` | live model discovery per provider |

---

## Migrating from the JSON build

Older builds stored `proxy_config.json` / `proxy_stats.json`. On first run the proxy
**auto-imports** those into `proxy.db`, and moves any legacy NVIDIA key + frontier ladder
into a first-class `nvidia` provider. Nothing to do manually.
