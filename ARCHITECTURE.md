# Architecture

`model-proxy` is a structural rewrite of `model-router`'s single-file
`nvidia_failover_proxy.py` (~3500 lines) into a normal FastAPI package. Behavior is
unchanged — see [FEATURES.md](FEATURES.md) for what it does and
[README.md](README.md) for how to run/configure it. This doc is the map for anyone
extending or debugging the code itself.

## Module map

```
app/
  config.py       # env var parsing, constants (FRONTIER_MODELS, provider defaults,
                   # pricing/freq-penalty JSON loaders). No state, no I/O beyond os.environ.
  db.py            # sqlite3 connection to proxy.db, WAL pragma, kv table get/set,
                   # chmod 0600 lockdown (secure_db_file, called once from main.py).
  ladder.py        # LadderConfig: provider CRUD (add/remove/merge), failover order +
                   # disabled set, persisted to the kv table. One-time env-var provider
                   # seeding on first boot (seeded_providers_v1 flag in kv).
  cascade.py       # Cascade: which model serves next. order() applies the sticky
                   # cursor, dead-model exclusion, cooldown filtering, and the local
                   # tail rung. cool()/note_status() record 429/404 outcomes.
  stats.py         # Stats: per-model counters, learned RPM/TPM ceilings from live
                   # 429s, token/cost accounting, sticky "serving now" tracking.
  state.py         # Constructs the shared singletons (ladder_config, cascade, stats)
                   # at import time, plus get/set_model_override() for /pick.
  guards/
    degenerate.py  # Repetition-loop + degenerate-output detection (looping_tail,
                   # looping_suffix, degenerate_reason) — pure functions over text.
    cjk.py         # CJK-in-code/tool-call detection, used to catch models that
                   # code-switch into Chinese/Japanese/Korean mid-code-block.
    stall.py       # Stream-stall watchdog (silent-stream detection for SSE).
  routes/
    chat.py        # POST /v1/chat/completions — the cascade loop itself: try each
                   # model in order, stream or buffer, apply guards, fail over on
                   # 429/5xx/stall/degenerate output, record stats.
    models.py       # GET /v1/models, GET /_models/available (live discovery).
    config_api.py   # GET/POST /_config (ladder order/disabled), GET/POST /_settings
                     # (providers + masked keys).
    dashboard.py     # GET / (SSE dashboard shell), GET /updates (SSE stream),
                     # GET /stats, GET /health, POST /_reset_cooldowns, /_reset_stats.
    commands.py      # In-chat slash commands (/help /stats /models /health /cool
                     # /uncool /switch /warm /unload /pick), intercepted in chat.py
                     # before any upstream call is made.
  templates/
    dashboard.html.j2   # Jinja2 shell (was inline Python string-building).
  static/
    dashboard.css, dashboard.js   # SSE client, drag-and-drop ladder reorder.
main.py            # FastAPI app factory: mounts static/, includes the 4 routers,
                    # calls secure_db_file(), has the uvicorn entrypoint under
                    # `if __name__ == "__main__"` (preserves PROXY_HOST/PROXY_PORT).
```

## Data flow for a chat request

1. `routes/chat.py` receives `POST /v1/chat/completions`.
2. If the last user message is a slash command, `routes/commands.py` handles it and
   returns immediately — no upstream call, no stats recorded.
3. Otherwise `cascade.order(preferred)` returns the ladder to try, starting from the
   sticky cursor (last model that actually served content) unless a specific model
   was requested.
4. For each model in order: resolve its provider via `ladder_config.model_provider()`,
   make the upstream call, and either stream (SSE passthrough with the stall watchdog
   and guards applied per-delta) or return a buffered response.
5. On 429 → `cascade.note_status(model, 429)` cools it and `stats` records the hit
   (feeding the learned-limit ceiling). On 404/410 → marked dead, permanently skipped.
   On a clean success → `stats.note_serving()` moves the sticky cursor.
6. `dashboard.py`'s `/updates` SSE endpoint polls the same `stats`/`cascade` singletons
   every 500ms to drive the live table.

## Why singletons + `importlib.reload` in tests

`ladder_config`, `cascade`, and `stats` are constructed once at import time in
`app/state.py`, keyed off `PROXY_DB_FILE` at construction. This matches the original
script's global-instance pattern and keeps route handlers simple (no dependency
injection plumbing). The cost: tests need a fresh DB file per test, which means
popping the 12 app modules from `sys.modules` and re-importing them under a
`monkeypatch`ed `PROXY_DB_FILE` — see `tests/conftest.py`'s `app_modules` fixture.
Don't try to mutate the module-level singletons directly across tests; reload instead.

## Known non-obvious behavior

- **Clock source**: cooldowns (`cascade.model_until`) are wall-clock
  (`time.time()`), not `time.monotonic()`, matching `dashboard.py`/`commands.py`'s
  countdown math. Don't reintroduce `monotonic()` here — it was a bug caught and
  fixed during this rewrite (silently desyncs cooldown displays from actual state).
- **`cool()` never shortens an existing cooldown** — `model_until[model] =
  max(existing, time.time() + secs)`. A model already cooling for 5 minutes that
  gets a fresh 30s 429 stays cooled for the original 5 minutes.
- **`reset_cooldowns()`'s return count** includes both live (not-yet-expired)
  cooldowns and dead-marked models, not just the `model_until` dict size — matches
  what the dashboard button reports.

## Testing

`tests/` covers the pure-function guards, `LadderConfig` CRUD/persistence/env-seeding,
`Cascade` ordering/cooldown/sticky-cursor behavior, and the FastAPI routes via
`TestClient`. Run with `pytest` from the repo root. All fixtures use a fresh temp
`proxy.db` per test — nothing touches the real database.
