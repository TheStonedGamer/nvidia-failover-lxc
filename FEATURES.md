# Features

Everything `nvidia_failover_proxy.py` does, in one place. It's a single-file,
OpenAI-compatible proxy (FastAPI + httpx) you point OpenCode — or anything that
speaks the OpenAI chat API — at. Default port **5002**.

## Routing & failover

- **Failover ladder cascade** — requests go to the first live model in the
  ladder; on 429 / 404 / 410 / connect failure / timeout / empty completion /
  degenerate output the *same request* transparently fails over to the next
  rung. Works for streaming and non-streaming.
- **Special model ids**
  | id | behavior |
  |---|---|
  | `nvidia-auto` | cloud ladder top-down, then the local Ollama tail |
  | `nvidia-only` | cloud ladder only — returns a real 429 (with `Retry-After`) when everything is cooling |
  | `nvidia-refine` | cloud ladder + `[refine]` prompt-refiner support |
  | `local-only` | straight to the local Ollama model, no cloud |
  | `local-refine` | refiner → local model |
  | `agent-planner` / `agent-builder` / `agent-reviewer` | cloud ladder with a role system prompt injected (skipped if the client already sent one) |
- **Explicit real model id** — starts the cascade *at that rung* and fails over
  downward from there.
- **Local tail rung** — an Ollama model sits at the end of the ladder so the
  request still lands somewhere when the whole cloud tier is rate-limited.
  Selectable from the dashboard dropdown (live list from Ollama `/api/tags`).
- **Context preserved across every switch** — the proxy is stateless; the full
  `messages` history is forwarded verbatim to whichever model serves.
- **Last-resort grace** — if every cloud model is cooling and there's no local
  rung, the model closest to reviving is tried instead of hard-failing.

## Reliability guards

- **Cooldowns** — 429 sidelines a model for `Retry-After`, else its *learned*
  cooldown, else 5 min. Connect blips get a one-shot retry on a fresh
  connection, then only a 20 s sideline so one blip can't drain the ladder.
  404/410 drops the model permanently for the process lifetime.
- **Learned rate limits** — per model, from live traffic: requests/min ceiling,
  input & output tokens/min ceilings (EWMA at the moment each 429 fired), and
  the real cooldown length. Models near their learned ceiling are skipped
  *before* they 429 (15% headroom). Ceilings are floored when checking
  (`PROXY_RPM_LIMIT_FLOOR` etc.) so a quota 429 under low traffic can't teach a
  ceiling so low the model is never tried again.
- **Empty-completion guard** — a 200 with no content/tool_calls fails over
  instead of returning nothing (`PROXY_SKIP_EMPTY`). Streams buffer the leading
  chunks until real content appears, so an empty stream is discarded silently.
- **Degenerate-output guard** (`PROXY_GUARD_DEGENERATE`) — catches the
  kimi-k2.6 failure mode (repetition loop that code-switches into Chinese):
  - non-stream: `finish_reason=repetition` or unexpected CJK → fail over.
  - streaming: cumulative CJK counter checked **before** each chunk is
    forwarded — the stream is truncated cleanly with `[DONE]` and the Chinese
    never reaches the client (`PROXY_CJK_MIN_CHARS`, default 4).
  - Keyed off the *user/system prompt*: genuine Chinese/Japanese/Korean
    requests are never touched, and degenerate assistant history can't disarm
    the guard.
- **Per-model frequency penalty** — a mild `frequency_penalty` is injected for
  models prone to loops (default `kimi-k2 → 0.3`); a client-supplied value
  always wins (`PROXY_FREQ_PENALTY_JSON`).
- **Mid-stream error handling** — once real bytes reached the client, an
  upstream error ends the SSE stream cleanly with `[DONE]` (truncated but
  coherent) instead of splicing a second model's answer on.
- **max_tokens floor** — NIM quirk: some models return empty output without
  `max_tokens`; the proxy injects 8192 when the client sends none
  (`PROXY_MAX_TOKENS_DEFAULT`).
- **Per-model read timeout** — a hung rung fails over after
  `PROXY_MODEL_TIMEOUT_S` (default 90 s) instead of hanging the request.

## Providers

- **Multi-provider** — any OpenAI-compatible API can be added (NVIDIA, OpenAI,
  Anthropic, Cerebras, Groq, OpenRouter, Mistral, DeepSeek, Google, xAI,
  Together, Ollama…). Each provider carries its own base URL + API key; each of
  its models routes there. One-click presets with brand icons in the UI.
- **Env seeding** — on first run, `<PREFIX>_API_KEY` (+ optional
  `<PREFIX>_MODELS`, `<PREFIX>_BASE_URL`) auto-creates a provider. Legacy
  `NVIDIA_API_KEY` / `ROUTER_NVIDIA_MODELS` / `LOCAL_OLLAMA_URL` /
  `LOCAL_MODEL` still work. The NVIDIA key can also be read from OpenCode's
  `opencode.jsonc` so it lives in one place.
- **Model discovery** — the UI queries every provider's `/v1/models` live and
  offers filter/sort/add-all chips to append models to the ladder.
- **Ollama as a first-class provider** — no key needed; its models serve as
  the local tail (not auto-added to the cloud ladder).

## Web dashboard (`/`)

- **Live metrics table via SSE** (0.5 s push): per model — state
  (LIVE/COOLING/DEAD/LOCAL), availability countdown, requests, successes, 429s,
  live+peak RPM, learned RPM/TPM ceilings, live TPM in/out, learned cooldown,
  token totals, and estimated $ saved. Auto-reconnects; dims when offline.
- **Green pulsing dot** on the active model — the first live rung, i.e. the
  one that serves the next request.
- **Money-saved estimate** — every token priced at a representative commercial
  rate for the same open-weight model (per-family table, substring-matched;
  `PROXY_PRICING_JSON` / `PROXY_PRICING_DEFAULT`). Shown per model, as a TOTAL
  row, and as a green hero banner. Also in `/stats` as `saved_usd_total`.
- **Failover ladder panel** — drag to reorder, uncheck to disable; persisted
  instantly and used by the very next request.
- **Providers & API keys panel** — add/update/remove providers (keys are
  masked in every response), quick-add presets, model discovery, local-tail
  dropdown.

## In-chat commands

Type these as a chat message through the proxy:

| command | effect |
|---|---|
| `/help` | list commands |
| `/pick [N\|off]` | numbered model picker; sets a sticky override for subsequent requests |
| `/stats` | usage stats (RPM/TPM vs learned limits, tokens, 429s) |
| `/models` | ladder with live/cooling/dead badges |
| `/health` | key + ladder health |
| `/cool <model>` / `/uncool <model>` | manually sideline / revive a rung |
| `/warm [model]` / `/unload [model]` | load / evict an Ollama model from GPU memory |

## Prompt refiner

Include `[refine]` anywhere in a message and a small local model
(`REFINER_MODEL`, default `qwen3:4b`) rewrites the prompt into a clearer,
more specific one before the main model sees it. The tag is stripped either
way. Disable with `PROXY_REFINER_ENABLE=0`.

## Persistence & accounting

- **SQLite (`proxy.db`, WAL)** holds the ladder order, toggles, providers,
  keys, and learned per-model stats; chmod 0600 including WAL sidecars. Legacy
  `proxy_config.json` / `proxy_stats.json` are imported once automatically.
- **Token accounting** — `usage` blocks tracked per model for streaming
  (`stream_options.include_usage` is injected) and non-streaming; the *last*
  usage chunk wins so prompt tokens aren't double-counted.
- **Endpoints** — `/v1/chat/completions`, `/v1/models`, `/health`, `/stats`
  (JSON), `/updates` (SSE), `/_config`, `/_settings`, `/_models/available`.

## Configuration reference

See the environment-variable table in [README.md](README.md).
