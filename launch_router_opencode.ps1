# Launch the NVIDIA failover router proxy + the OpenCode desktop GUI.
#
#   - Starts nvidia_failover_proxy.py on :5002 (skips if already running)
#   - Waits for the proxy /health to go green
#   - Opens the OpenCode desktop app (already configured for nvidia-failover)
#
# Meant to be run from the desktop shortcut created by make_shortcut.ps1, which
# launches it hidden so there's no console window. Run directly to see output:
#   powershell -ExecutionPolicy Bypass -File launch_router_opencode.ps1
#
# Env overrides (optional): PROXY_PORT, ROUTER_NVIDIA_MODELS, LOCAL_MODEL,
#   PROXY_LOCAL_FALLBACK, PROXY_MODEL_TIMEOUT_S (see PROXY_README.md).

$ErrorActionPreference = "Stop"
$Root = "E:\Projects\model-router"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Proxy = Join-Path $Root "nvidia_failover_proxy.py"
$Port = if ($env:PROXY_PORT) { $env:PROXY_PORT } else { "5002" }
$OpenCode = "$env:LOCALAPPDATA\Programs\@opencode-aidesktop\OpenCode.exe"

if (-not (Test-Path $Py)) { Write-Host "venv python not found at $Py" -ForegroundColor Red; exit 1 }

# 1) Start the proxy only if nothing is already listening on the port.
$listening = Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    Write-Host "Router proxy already listening on :$Port" -ForegroundColor Green
} else {
    Write-Host "Starting NVIDIA failover router proxy on :$Port ..." -ForegroundColor Cyan
    Start-Process -FilePath $Py -ArgumentList $Proxy -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $Root "proxy.out.log") `
        -RedirectStandardError (Join-Path $Root "proxy.err.log") -WindowStyle Hidden | Out-Null
}

# 2) Wait for /health to report ok.
$healthy = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($h.ok) { $healthy = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
if (-not $healthy) {
    Write-Host "Proxy did not become healthy on :$Port. Check proxy.err.log." -ForegroundColor Red
    exit 1
}
Write-Host "Router healthy on :$Port  (dashboard http://127.0.0.1:$Port/)" -ForegroundColor Green

# 3) Open the OpenCode desktop GUI, then wait for it to close and shut the
#    proxy down with it — so the router never lingers after you quit OpenCode.
if (-not (Test-Path $OpenCode)) {
    Write-Host "OpenCode desktop app not found at $OpenCode" -ForegroundColor Red
    exit 1
}
Write-Host "Launching OpenCode desktop app ..." -ForegroundColor Cyan
Start-Process -FilePath $OpenCode | Out-Null

# The Electron app forks helper processes; give them a moment to appear, then
# wait until every OpenCode process is gone (i.e. the app has fully closed).
Start-Sleep -Seconds 3
while (Get-Process -Name "OpenCode" -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 2
}

# OpenCode has closed — stop the router proxy listening on the port.
Write-Host "OpenCode closed; stopping router proxy on :$Port ..." -ForegroundColor Cyan
$owners = Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $owners) { try { Stop-Process -Id $procId -Force -ErrorAction Stop } catch {} }
