# NVIDIA Failover Proxy LXC Deployment

Deploy a **24/7 auto-scaling NVIDIA frontier-model failover proxy** to a Proxmox container with a single command.

## Features

- **Cascades through all 15 NVIDIA frontier models** on rate-limits (429/timeout)
- **Auto-fails over to local Ollama 80B** when cloud tier is rate-limited
- **Learns limits** from live traffic, cooldowns from Retry-After headers
- **Supports `local-only`, `local-refine`, agent profiles** (`agent-planner`, `agent-builder`, `agent-reviewer`)
- **Prompt refiner** via small Qwen (qwen3:4b) for `[refine]` tagged messages
- **Interactive `/pick` command** + slash commands (`/help`, `/stats`, `/cool`)
- **SSE dashboard** at `/` with per-model state, tokens/sec, rate limits
- **Multi-request concurrency** via async httpx + semaphore

## Deploy

```bash
# Quickstart
./scripts/deploy.sh 10.0.0.98 10.0.0.199 10.0.0.127 "nvapi-YOUR_KEY_HERE"
```

## Architecture

```
raw-router (LXC 10.0.0.199)
├─ python systemd ←→ 15× NVIDIA frontier
├─ proxy (5002) ← local Ollama (your desktop)
└── SSE /stats ←→ OpenCode client
```

## Usage

```bash
# Hit the proxy (any OpenAI-compatible client)
curl -v \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia-auto","messages":[{"role":"user","content":"/help"}]}' \
  http://10.0.0.199:5002/v1/chat/completions
```

## Environment

| Variable            | Default                     | Description                       |
| ------------------- | --------------------------- | --------------------------------- |
| `PROXY_PORT`        | `5002`                      | Bind port                         |
| `PROXY_CONCURRENCY` | `10`                        | Max concurrent requests           |
| `REFINER_BASE_URL`  | `http://localhost:11434/v1` | Qwen4B refiner (Ollama)           |
| `REFINER_MODEL`     | `qwen3:4b`                  | Small Ollama model for `[refine]` |
| `REFINER_TAG`       | `[refine]`                  | Tag to trigger refinement         |
| `LOCAL_OLLAMA_URL`  | `http://127.0.0.1:11434/v1` | Ollama 80B tail rung              |
| `LOCAL_MODEL`       | `qwen3-coder-next:80b`      | Local Ollama model                |
| `NVIDIA_API_KEY`    | (via env)                   | Your nvapi- key                   |
