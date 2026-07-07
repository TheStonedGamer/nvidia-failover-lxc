# Create/refresh the "Router + OpenCode" desktop shortcut.
#   - Target: wscript.exe launch_hidden.vbs  (runs the launcher fully hidden)
#   - Icon:   the OpenCode desktop app
# Re-run any time to recreate it.

$Root    = "E:\Projects\model-router"
$vbs     = Join-Path $Root "launch_hidden.vbs"
$icon    = "$env:LOCALAPPDATA\Programs\@opencode-aidesktop\OpenCode.exe"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk     = Join-Path $desktop "Router + OpenCode.lnk"

$ws = New-Object -ComObject WScript.Shell
$s  = $ws.CreateShortcut($lnk)
$s.TargetPath       = "wscript.exe"
$s.Arguments        = """$vbs"""
$s.WorkingDirectory = $Root
$s.IconLocation     = "$icon,0"
$s.Description       = "Start NVIDIA failover router + OpenCode desktop GUI (router closes when OpenCode does)"
$s.WindowStyle      = 7   # minimized; wscript itself shows nothing
$s.Save()

if (Test-Path $lnk) { Write-Host "Created shortcut: $lnk" -ForegroundColor Green }
else { Write-Host "FAILED to create shortcut" -ForegroundColor Red }
