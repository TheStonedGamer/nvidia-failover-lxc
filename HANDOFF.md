# Handoff: Web UI for Failover Order + Model Toggle

> **Status:** In progress — Claude Code is picking this up.
> **Date:** 2026-07-06
> **Repo:** https://github.com/TheStonedGamer/nvidia-failover-lxc (branch `main`)

---

## What we're building

A web UI on the NVIDIA failover proxy dashboard that lets you:

1. **Drag-and-drop reorder** the failover ladder (which cloud model is tried first, second, etc.)
2. **Toggle individual models on/off** (disabled models are skipped entirely)
3. **Persist** the config to `proxy_config.json` so it survives restarts

---

## What's already DONE

### GitHub repo

- Repo created at `https://github.com/TheStonedGamer/nvidia-failover-lxc`
- Initial commit pushed (contain `deploy.sh`, `README.md`, a simplified proxy stub)
- Remote `origin` → `https://github.com/TheStonedGamer/nvidia-failover-lxc.git` on branch `main`
- Local repo dir: `E:\Projects\nvidia-failover-lxc\`

### LXC container (Proxmox CT 3000, IP 10.0.0.199)

- Container running, systemd service `nvidia-failover-proxy` is enabled + active
- Proxy accessible at `http://10.0.0.199:5002/v1` (health endpoint returns `ok:true`)
- Bound to `0.0.0.0:5002` (reachable from LAN)
- Environment vars set in systemd unit: `NVIDIA_API_KEY`, `REFINER_BASE_URL`, `LOCAL_OLLAMA_URL`
- Code deployed at `/root/model-router/nvidia_failover_proxy.py` + `/root/model-router/src/providers/nvidia.py`
- Deploy commands:
  ```powershell
  scp E:\Projects\model-router\nvidia_failover_proxy.py root@10.0.0.98:/root/
  ssh root@10.0.0.98 "pct push 3000 /root/nvidia_failover_proxy.py /root/model-router/nvidia_failover_proxy.py && pct exec 3000 -- systemctl restart nvidia-failover-proxy"
  ssh root@10.0.0.98 "pct exec 3000 -- curl -s http://127.0.0.1:5002/health"
  ```

### Proxy code (`E:\Projects\model-router\nvidia_failover_proxy.py` — the REAL working version, ~1500 lines)

- `LadderConfig` class added at lines ~135–186 with:
  - `order: List[str]` — user-defined failover order
  - `disabled: Set[str]` — models toggled off
  - `load()` / `save()` — atomic JSON persistence to `proxy_config.json`
  - `update(order, disabled)` — update + persist in one call
  - `active_ladder()` — returns enabled models in user-defined order
- Global instance: `ladder_config = LadderConfig()` at line ~186
- Config file path: `CONFIG_FILE = os.environ.get("PROXY_CONFIG_FILE", "proxy_config.json")`

### OpenCode config

- `C:\Users\BrianTheMint\.config\opencode\opencode.jsonc` already points `nvidia-failover` provider at `http://10.0.0.199:5002/v1`

---

## What's NOT done (pick up here)

### 1. Wire `LadderConfig` into `Cascade.order()`

In `E:\Projects\model-router\nvidia_failover_proxy.py`, find `Cascade.order()` at line ~203.

Currently the base ladder is:

```python
base = list(self.models)  # line ~217
```

Change it to use the config-aware order:

```python
base = ladder_config.active_ladder()  # respects user order + disabled set
if preferred and preferred not in SPECIAL_IDS and not self.is_local(preferred):
    base = [preferred] + [m for m in base if m != preferred]
cloud = [
    m for m in base
    if m not in self.dead
    and now >= self.model_until.get(m, 0.0)
    and not stats.is_near_limit(m)
]
```

The dead/cooling/near-limit filters stay the same — just apply them to the config-aware base.

### 2. Add `/_config` API endpoints

Add these near the other `@app.get` endpoints (search for `@app.get("/health")`):

```python
from fastapi import Body

@app.get("/_config")
async def get_config() -> dict:
    return {
        "order": ladder_config.order,
        "disabled": list(ladder_config.disabled),
        "all_models": list(cascade.models),
    }

@app.post("/_config")
async def set_config(body: dict = Body(...)) -> dict:
    valid = set(cascade.models)
    order = [m for m in body.get("order", []) if m in valid]
    # append any models missing from the submitted order
    for m in cascade.models:
        if m not in order:
            order.append(m)
    disabled = [m for m in body.get("disabled", []) if m in valid]
    ladder_config.update(order=order, disabled=disabled)
    return {"ok": True, "order": ladder_config.order, "disabled": list(ladder_config.disabled)}
```

### 3. Add drag-and-drop UI to the dashboard

In the `dashboard()` function (search `async def dashboard`), the HTML is a big f-string. The SSE handler swaps `<tbody>` innerHTML every 500ms, so the config bar must be **outside** the `<tbody>`.

Add a config bar above the table:

```html
<div
  id="config-bar"
  style="margin-bottom:16px;padding:12px;background:#1a1e28;border-radius:8px"
>
  <b>Failover Order</b> (drag to reorder, toggle to enable):
  <div id="ladder-list"></div>
  <button onclick="saveConfig()">Save Order</button>
</div>
```

JavaScript (add to the existing `<script>` block):

```javascript
async function loadConfig() {
  const r = await fetch("/_config");
  const c = await r.json();
  const el = document.getElementById("ladder-list");
  el.innerHTML = c.order
    .map(function (m, i) {
      const checked = !c.disabled.includes(m) ? "checked" : "";
      return (
        '<div draggable=true style="padding:6px;margin:4px 0;background:#222;cursor:move;border-radius:4px"' +
        ' ondragstart="drag(event,' +
        i +
        ')" ondrop="drop(event,' +
        i +
        ')" ondragover="event.preventDefault()">' +
        "<input type=checkbox " +
        checked +
        ' data-model="' +
        m +
        '"> ' +
        m +
        "</div>"
      );
    })
    .join("");
}

let dragIdx = null;
function drag(e, i) {
  dragIdx = i;
}
function drop(e, i) {
  e.preventDefault();
  const items = [...document.querySelectorAll("#ladder-list > div")];
  const moved = items[dragIdx],
    target = items[i];
  if (dragIdx < i) target.parentNode.insertBefore(moved, target.nextSibling);
  else target.parentNode.insertBefore(moved, target);
  dragIdx = null;
}

async function saveConfig() {
  const items = [...document.querySelectorAll("#ladder-list > div")];
  const order = items.map(function (d) {
    return d.querySelector("input").dataset.model;
  });
  const disabled = items
    .filter(function (d) {
      return !d.querySelector("input").checked;
    })
    .map(function (d) {
      return d.querySelector("input").dataset.model;
    });
  await fetch("/_config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order: order, disabled: disabled }),
  });
  alert("Failover order saved!");
}

loadConfig();
```

### 4. Verify locally

```powershell
# Compile check
E:\Projects\model-router\.venv\Scripts\python.exe -c "import py_compile; py_compile.compile('E:\\Projects\\model-router\\nvidia_failover_proxy.py', doraise=True); print('OK')"
```

### 5. Deploy to the LXC

```powershell
scp E:\Projects\model-router\nvidia_failover_proxy.py root@10.0.0.98:/root/
ssh root@10.0.0.98 "pct push 3000 /root/nvidia_failover_proxy.py /root/model-router/nvidia_failover_proxy.py && pct exec 3000 -- systemctl restart nvidia-failover-proxy && sleep 3 && pct exec 3000 -- curl -s http://127.0.0.1:5002/health"
ssh root@10.0.0.98 "pct exec 3000 -- curl -s http://127.0.0.1:5002/_config"
```

Verify:

- `GET /_config` returns `{"order": [...all 15 models...], "disabled": []}`
- Dashboard at `http://10.0.0.199:5002/` shows the config bar with draggable model rows
- Drag a model, uncheck a model, click Save, refresh page — the order/toggle persists

### 6. Push the real proxy to GitHub

The repo at `E:\Projects\nvidia-failover-lxc\` currently has a simplified stub. Replace it with the real working proxy:

```powershell
# Copy the REAL proxy + src into the repo
Copy-Item "E:\Projects\model-router\nvidia_failover_proxy.py" "E:\Projects\nvidia-failover-lxc\nvidia_failover_proxy.py" -Force
Copy-Item "E:\Projects\model-router\src\providers" "E:\Projects\nvidia-failover-lxc\src\providers" -Recurse -Force

Set-Location E:\Projects\nvidia-failover-lxc
git add -A
git commit -m "Add web UI for failover order + model toggle; replace stub with real proxy"
git push
```

---

## Key files

| File                                                    | Role                                                   |
| ------------------------------------------------------- | ------------------------------------------------------ |
| `E:\Projects\model-router\nvidia_failover_proxy.py`     | **Edit this** — the real working proxy (~1500 lines)   |
| `E:\Projects\model-router\src\providers\nvidia.py`      | API key resolver                                       |
| `E:\Projects\nvidia-failover-lxc\`                      | GitHub repo dir (push the real proxy here after edits) |
| `C:\Users\BrianTheMint\.config\opencode\opencode.jsonc` | OpenCode config (already points to LXC)                |

## Infrastructure

| What             | Where                                                         |
| ---------------- | ------------------------------------------------------------- |
| Proxmox PVE host | `10.0.0.98` (root, SSH key auth via `ssh root@10.0.0.98`)     |
| LXC container    | CTID 3000, IP `10.0.0.199`                                    |
| Proxy service    | systemd `nvidia-failover-proxy` (enabled, auto-start on boot) |
| Desktop Ollama   | `10.0.0.127:11434` (80B tail + qwen3:4b refiner)              |
| Dashboard        | `http://10.0.0.199:5002/`                                     |

## Gotchas

- The SSE EventSource at `/updates` swaps `<tbody>` innerHTML every 500ms — the config bar MUST be outside the tbody or it gets overwritten.
- `LadderConfig` was added to the proxy but is NOT yet wired into `Cascade.order()`. The `base = list(self.models)` line at ~217 needs to become `base = ladder_config.active_ladder()`.
- The repo's `nvidia_failover_proxy.py` is a simplified stub — it needs to be REPLACED with the real one before pushing.
- The container's `nvidia_failover_proxy.py` was modified with `sed` to bind `0.0.0.0` instead of `127.0.0.1`. After deploying the real proxy, you need to either set the `host="0.0.0.0"` in the file OR set `PROXY_HOST=0.0.0.0` (if you add that env var).
