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
- **Live dashboard** — SSE dashboard with per-model state, token accounting, drag-and-drop
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
  thestonedgamer/nvidia-failover-proxy:latest
```

Then open **http://localhost:5002/** and add a provider under
**🔑 Providers & API keys** (click a preset — NVIDIA, OpenAI, Anthropic, Cerebras, … —
then paste your API key and the model ids you want).

Optionally seed an NVIDIA provider at first run:

```bash
docker run -d --name nvidia-failover-proxy \
  -p 5002:5002 -v proxy-data:/data \
  -e NVIDIA_API_KEY=nvapi-xxxxxxxx \
  thestonedgamer/nvidia-failover-proxy:latest
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

### Proxmox LXC

[`scripts/deploy.sh`](scripts/deploy.sh) creates an unprivileged Ubuntu container on a
Proxmox host, installs the proxy as a systemd service, and binds it to `0.0.0.0`:

```bash
./scripts/deploy.sh <pve-host-ip> <container-ip> <local-ollama-ip> 'nvapi-YOUR-KEY'
```

### Standalone installers (no Python required)

Prebuilt, self-contained builds are produced by GitHub Actions for every `v*`
release ([`.github/workflows/build-installers.yml`](.github/workflows/build-installers.yml))
and attached to the [Releases](../../releases) page:

| Platform | Download | Notes |
| --- | --- | --- |
| **Windows** | `nvidia-failover-proxy-setup-windows-x64.exe` (installer) or the bare `…-windows-x64.exe` | Inno Setup installer with Start-menu shortcuts; the bare exe is portable |
| **macOS (Intel)** | `nvidia-failover-proxy-macos-x64.dmg` | unsigned — first launch: right-click → Open, or `xattr -dr com.apple.quarantine` the binary |
| **macOS (Apple Silicon)** | `nvidia-failover-proxy-macos-arm64.dmg` | same as above |
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

The image is published as `<DOCKERHUB_USERNAME>/nvidia-failover-proxy`, tagged
`latest`, the branch name, the short commit SHA, and semver tags for `v*` releases.
Pull requests build but do **not** push.

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
| `NVIDIA_API_KEY` | — | seeds a first-run `nvidia` provider (optional) |
| `ROUTER_NVIDIA_MODELS` | built-in frontier list | comma-separated NVIDIA model ids used to seed migration |
| `LOCAL_OLLAMA_URL` | `http://10.0.0.142:11434/v1` | local Ollama base URL for the tail rung |
| `LOCAL_MODEL` | `Qwen3-Coder-Next-80b-A3B:latest` | local model id |
| `PROXY_LOCAL_FALLBACK` | `1` | set `0` to disable the local tail rung |
| `PROXY_MODEL_TIMEOUT_S` | `90` | per-model read timeout before failing over |
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
