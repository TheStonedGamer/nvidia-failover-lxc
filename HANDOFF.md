# Handoff: model-proxy (structural rewrite of nvidia-failover-proxy)

> **Status:** Rewrite complete, tested, and deployed live.
> **Date:** 2026-07-08

---

## What this is

`E:\Projects\model-router\nvidia_failover_proxy.py` was a working, deployed
OpenAI-compatible failover proxy — but a single 3482-line file (routing, guards,
SQLite persistence, stats, in-chat commands, and the entire dashboard all in one
module). This repo, `model-proxy`, is a **behavior-identical rewrite** of that file
into a normal FastAPI package (`app/`) — see [ARCHITECTURE.md](ARCHITECTURE.md) for
the module map. No new features were added; feature parity was the bar (see the
original plan, `imperative-waddling-whisper` in `~/.claude/plans/` if it still
exists). Three real parity bugs were found and fixed during the port — see
"Bugs fixed during the port" below.

## Current state

- **Deployed and live** at CT 3000 (10.0.0.199:5002), replacing the old
  `nvidia_failover_proxy.py` process. This is the proxy OpenCode and other clients
  actually talk to right now.
- 42 tests in `tests/`, all passing, run against fresh temp DBs (never the real one).
- Verified against a **copy** of the real `proxy.db` before deploying (health,
  stats, models, dashboard, SSE, slash commands, config round-trip all matched).
- Verified live after deploying: `/health`, `/stats` (showed the pre-existing
  historical counters intact), `/v1/models`, dashboard HTML, and a real
  `/v1/chat/completions` request against NVIDIA all returned correct results with
  usage recorded.

## Where things live on CT 3000

| What | Where |
| --- | --- |
| PVE host | `10.0.0.98` (`ssh root@10.0.0.98`) |
| Container | CTID 3000, IP `10.0.0.199` |
| **New install** | `/root/model-proxy/` — `app/`, `.venv/`, `requirements.txt`, `proxy.db` |
| **Old install (kept for rollback)** | `/root/model-router/` — untouched, still has `nvidia_failover_proxy.py` + its own `proxy.db` copy from before the swap |
| systemd unit | `/etc/systemd/system/nvidia-failover-proxy.service` — `ExecStart` now points at `/root/model-proxy/.venv/bin/python -m app.main`, `WorkingDirectory=/root/model-proxy` |
| **Pre-swap unit backup** | `/etc/systemd/system/nvidia-failover-proxy.service.bak-pre-model-proxy` |
| Drop-in | `/etc/systemd/system/nvidia-failover-proxy.service.d/maxtokens.conf` (`PROXY_MAX_TOKENS_DEFAULT=16384`) — unchanged, still applies |

Env vars in the unit (`NVIDIA_API_KEY`, `REFINER_BASE_URL`, `LOCAL_OLLAMA_URL`,
`PROXY_HOST=0.0.0.0`, `PROXY_MODEL_TIMEOUT_S=45`) were carried over unchanged.

## Rollback

If the new deployment misbehaves:

```bash
ssh root@10.0.0.98 "pct exec 3000 -- bash -c '\
  cp /etc/systemd/system/nvidia-failover-proxy.service.bak-pre-model-proxy \
     /etc/systemd/system/nvidia-failover-proxy.service && \
  systemctl daemon-reload && systemctl restart nvidia-failover-proxy'"
```

That points `ExecStart` back at `/root/model-router/nvidia_failover_proxy.py`. The
old install's own `proxy.db` wasn't touched, so it resumes with its last known
state (it will be missing anything served only through the new install in the
interim).

## How to redeploy an update

There's no `scripts/proxmox-lxc.sh` equivalent wired up for this repo yet (the one
in `model-router/scripts/` targets the old single-file layout / `/opt/nvidia-failover`
or `/root/model-router` paths — it doesn't know about `/root/model-proxy`). Until
that's written, redeploy by hand:

```bash
# from E:\Projects\model-proxy
tar --exclude="__pycache__" -cf /tmp/model-proxy-app.tar app requirements.txt
scp /tmp/model-proxy-app.tar root@10.0.0.98:/root/
ssh root@10.0.0.98 "pct push 3000 /root/model-proxy-app.tar /root/model-proxy-app.tar && \
  pct exec 3000 -- bash -c 'cd /root/model-proxy && tar --no-same-owner -xf /root/model-proxy-app.tar && rm /root/model-proxy-app.tar && chown -R root:root . && .venv/bin/pip install -q -r requirements.txt && systemctl restart nvidia-failover-proxy'"
ssh root@10.0.0.98 "pct exec 3000 -- curl -s http://127.0.0.1:5002/health"
```

`proxy.db` lives in `/root/model-proxy/` on the container and is **not** part of
the tarball — don't overwrite it on redeploy.

## Bugs fixed during the port (not in the original scope, but real deviations)

Found by careful line-by-line comparison against the original source while writing
`tests/test_cascade.py`, not by test failures:

1. **Clock source mismatch** — an earlier draft of `app/cascade.py` used
   `time.monotonic()` for `model_until` bookkeeping in `order()`,
   `soonest_cooldown()`, and `cool()`. The original (and `dashboard.py`/
   `commands.py`) all use `time.time()`. Fixed to `time.time()` everywhere.
2. **Missing cooldown-extension guard** — `cool()` now does
   `model_until[model] = max(existing, time.time() + secs)`, matching the
   original: a fresh short cooldown must never shorten an existing longer one.
3. **`reset_cooldowns()` return count** — now counts live (unexpired) cooldowns
   plus dead-marked models, matching what the original reports to the dashboard
   button, instead of just `len(model_until)`.

## What's genuinely different from the original

Nothing behaviorally. Structurally: Jinja2 templates instead of inline Python
f-string HTML building for the dashboard; `app/routes/commands.py`'s slash-command
handler uses a synchronous `httpx.Client` (matching its sync call site in
`chat.py`) instead of the original's `httpx.AsyncClient`; `/pick`'s model override
is stored via `get_model_override()`/`set_model_override()` in `app/state.py`
instead of a bare module-level `global`.

## If you need more detail

- [ARCHITECTURE.md](ARCHITECTURE.md) — module map, request data flow, singleton
  test-reload pattern.
- [FEATURES.md](FEATURES.md) — full feature list.
- [README.md](README.md) — install/config/usage.
- `E:\Projects\model-router\HANDOFF.md` — an older, task-specific handoff from
  when the ladder-config web UI was added to the original single-file proxy;
  useful for pre-rewrite history, not for this codebase's structure.
