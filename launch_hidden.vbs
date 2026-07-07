' Zero-flash wrapper: runs the launcher PowerShell script fully hidden so the
' desktop shortcut opens the router + OpenCode with no console window at all.
Set sh = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptDir & "\launch_router_opencode.ps1""", 0, False
