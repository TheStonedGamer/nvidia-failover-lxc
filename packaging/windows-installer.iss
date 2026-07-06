; Inno Setup script for the NVIDIA Failover Proxy.
; Compiled in CI (see .github/workflows/build-installers.yml). The one-file
; PyInstaller exe is expected at dist\nvidia-failover-proxy.exe.
#define MyAppName "NVIDIA Failover Proxy"
#define MyAppExe "nvidia-failover-proxy.exe"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=TheStonedGamer
DefaultDirName={autopf}\NVIDIA Failover Proxy
DefaultGroupName=NVIDIA Failover Proxy
UninstallDisplayIcon={app}\{#MyAppExe}
OutputDir=installer
OutputBaseFilename=nvidia-failover-proxy-setup
SetupIconFile=packaging\icon.ico
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Files]
Source: "dist\nvidia-failover-proxy.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\NVIDIA Failover Proxy"; Filename: "{app}\{#MyAppExe}"
Name: "{group}\Open dashboard"; Filename: "http://localhost:5002/"
Name: "{group}\Uninstall NVIDIA Failover Proxy"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "Launch the proxy now"; Flags: nowait postinstall skipifsilent
