@echo off
REM Double-click launcher: starts the NVIDIA failover router proxy and opens the
REM OpenCode web GUI configured for it. All logic lives in the .ps1 next to this.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_router_opencode.ps1"
if errorlevel 1 (
  echo.
  echo Launch failed. See the messages above and proxy.err.log.
  pause
)
