#!/usr/bin/env bash
# ============================================================================
#  NVIDIA Failover Proxy — Proxmox VE LXC helper script
# ----------------------------------------------------------------------------
#  Run this ON a Proxmox VE host (as root). It interactively creates an
#  unprivileged Debian LXC, installs the proxy as a systemd service, and
#  (optionally) seeds any major OpenAI-compatible provider's API key.
#
#    bash -c "$(wget -qLO - https://raw.githubusercontent.com/TheStonedGamer/nvidia-failover-lxc/main/scripts/proxmox-lxc.sh)"
#
#  Non-interactive: export the CT_* / provider *_API_KEY vars and pipe with a
#  closed stdin, or set AUTO=1 to accept all defaults.
# ============================================================================
set -Eeuo pipefail

APP="NVIDIA Failover Proxy"
REPO_RAW="https://raw.githubusercontent.com/TheStonedGamer/nvidia-failover-lxc/main"
TEMPLATE="debian-12-standard_12.7-1_amd64.tar.zst"
PROXY_PORT=5002

# --- pretty output ----------------------------------------------------------
RD=$'\033[01;31m'; GN=$'\033[1;92m'; YW=$'\033[33m'; BL=$'\033[36m'; CL=$'\033[m'
msg_info()  { echo -e " ${YW}•${CL} $*"; }
msg_ok()    { echo -e " ${GN}✓${CL} $*"; }
msg_error() { echo -e " ${RD}✗${CL} $*" >&2; }
die()       { msg_error "$*"; exit 1; }
trap 'msg_error "failed at line $LINENO"' ERR

header() {
  echo -e "${GN}"
  cat <<'EOF'
   _   ___   _____ ___ ___    _   ___ _   _ _ _____   _____ ___
  | \ | \ \ / /_ _|   \_ _|  /_\ | __/_\ | | |_   _| |_   _/ _ \
  |  \| |\ V / | || |) | |  / _ \| _/ _ \| | | | |     | || (_) |
  |_|\__| \_/ |___|___/___|/_/ \_\_/_/ \_\_|_| |_|     |_| \___/
            F A I L O V E R   P R O X Y   —   L X C
EOF
  echo -e "${CL}"
}

# --- prompt helpers (whiptail if a TTY, else defaults/env) -------------------
have_tty() { [ -t 0 ] || [ -e /dev/tty ]; }
AUTO="${AUTO:-0}"

ask() { # ask VAR "Prompt" "default"
  local __var="$1" __prompt="$2" __def="${3:-}" __val
  __val="${!__var:-}"
  if [ -n "$__val" ]; then printf -v "$__var" '%s' "$__val"; return; fi
  if [ "$AUTO" = "1" ] || ! have_tty; then printf -v "$__var" '%s' "$__def"; return; fi
  __val=$(whiptail --title "$APP" --inputbox "$__prompt" 9 70 "$__def" 3>&1 1>&2 2>&3 </dev/tty) \
    || die "cancelled"
  printf -v "$__var" '%s' "${__val:-$__def}"
}

ask_secret() { # ask_secret VAR "Prompt"  (blank = skip)
  local __var="$1" __prompt="$2" __val
  __val="${!__var:-}"
  if [ -n "$__val" ]; then return; fi
  if [ "$AUTO" = "1" ] || ! have_tty; then printf -v "$__var" '%s' ""; return; fi
  __val=$(whiptail --title "$APP" --passwordbox "$__prompt\n(leave blank to skip)" 10 70 3>&1 1>&2 2>&3 </dev/tty) || __val=""
  printf -v "$__var" '%s' "$__val"
}

# --- preflight --------------------------------------------------------------
header
command -v pct >/dev/null 2>&1 || die "must be run on a Proxmox VE host (pct not found)"
[ "$(id -u)" -eq 0 ] || die "must be run as root"

MODE="${1:-install}"

# --- update: pull the latest proxy code into an existing container ----------
do_update() {
  local ctid="${2:-${CT_ID:-}}"
  if [ -z "$ctid" ]; then
    ask ctid "Container ID to update" ""
  fi
  [ -n "$ctid" ] || die "no container ID given (usage: proxmox-lxc.sh update <CTID>)"
  pct status "$ctid" >/dev/null 2>&1 || die "CT $ctid does not exist"
  pct start "$ctid" >/dev/null 2>&1 || true

  # Find where the proxy is installed (helper layout, else legacy deploy.sh).
  local dir=""
  for cand in /opt/nvidia-failover /root/model-router; do
    if pct exec "$ctid" -- test -f "$cand/nvidia_failover_proxy.py"; then dir="$cand"; break; fi
  done
  [ -n "$dir" ] || die "couldn't find nvidia_failover_proxy.py in CT $ctid (looked in /opt/nvidia-failover, /root/model-router)"
  msg_info "Updating CT ${BL}$ctid${CL} at ${dir}"

  # Back up the current file, fetch the latest, refresh deps if a venv exists.
  pct exec "$ctid" -- bash -c "
    set -e
    cd '$dir'
    cp nvidia_failover_proxy.py nvidia_failover_proxy.py.bak 2>/dev/null || true
    curl -fsSL '${REPO_RAW}/nvidia_failover_proxy.py' -o nvidia_failover_proxy.py
    curl -fsSL '${REPO_RAW}/requirements.txt' -o requirements.txt 2>/dev/null || true
    if [ -x .venv/bin/pip ]; then .venv/bin/pip install -q -r requirements.txt || true; fi
    # Syntax-check before we bounce the service; roll back on failure.
    PY=.venv/bin/python; [ -x \"\$PY\" ] || PY=python3
    if ! \$PY -c 'import ast,sys; ast.parse(open(\"nvidia_failover_proxy.py\",encoding=\"utf-8\").read())'; then
      echo 'SYNTAX_FAIL'; mv -f nvidia_failover_proxy.py.bak nvidia_failover_proxy.py 2>/dev/null || true; exit 1
    fi
  " || die "update failed (code fetched did not parse; rolled back)"
  msg_ok "Code updated"

  # Restart whichever service unit is present.
  local svc="nvidia-failover-proxy"
  pct exec "$ctid" -- systemctl list-unit-files 2>/dev/null | grep -q "$svc" || svc="model-router"
  pct exec "$ctid" -- systemctl restart "$svc" 2>/dev/null \
    || die "couldn't restart service (tried '$svc') — check: pct exec $ctid -- systemctl status $svc"
  msg_ok "Service '$svc' restarted"

  sleep 4
  local ip; ip="$(pct exec "$ctid" -- hostname -I 2>/dev/null | awk '{print $1}')"
  if pct exec "$ctid" -- curl -fsS --max-time 8 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
    msg_ok "Health check passed"
  else
    msg_error "Health check did not pass — check: pct exec $ctid -- journalctl -u $svc -e"
  fi
  echo
  msg_ok "${APP} updated"
  echo -e "   Dashboard : ${BL}http://${ip}:${PROXY_PORT}/${CL}"
  echo -e "   Rollback  : ${YW}pct exec $ctid -- bash -c 'cd $dir && mv -f nvidia_failover_proxy.py.bak nvidia_failover_proxy.py && systemctl restart $svc'${CL}"
  exit 0
}

if [ "$MODE" = "update" ]; then do_update "$@"; fi
if [ "$MODE" != "install" ]; then die "unknown command '$MODE' (use: install | update <CTID>)"; fi

DEFAULT_CTID="$(pvesh get /cluster/nextid 2>/dev/null || echo 3000)"
DEFAULT_STORAGE="$(pvesm status -content rootdir 2>/dev/null | awk 'NR==2{print $1}')"
DEFAULT_STORAGE="${DEFAULT_STORAGE:-local-lvm}"

ask CT_ID       "Container ID"                 "$DEFAULT_CTID"
ask CT_HOSTNAME "Hostname"                     "nvidia-failover"
ask CT_DISK     "Disk size (GB)"               "6"
ask CT_CORES    "CPU cores"                    "2"
ask CT_RAM      "RAM (MB)"                      "2048"
ask CT_BRIDGE   "Network bridge"               "vmbr0"
ask CT_STORAGE  "Storage pool"                 "$DEFAULT_STORAGE"
ask CT_NET      "IPv4 (CIDR or 'dhcp')"        "dhcp"
ask CT_GW       "Gateway (blank if dhcp)"      ""
ask CT_PASS     "Root password"                "changeme"
ask LOCAL_OLLAMA_URL "Local Ollama URL (blank = none, for the tail rung)" ""

# --- provider API keys (optional) — seeded via env into the service ---------
msg_info "Provider API keys (optional — leave blank to add later in the web UI)"
declare -A PROVIDERS=(
  [NVIDIA]=NVIDIA_API_KEY   [OPENAI]=OPENAI_API_KEY   [ANTHROPIC]=ANTHROPIC_API_KEY
  [CEREBRAS]=CEREBRAS_API_KEY [GROQ]=GROQ_API_KEY     [OPENROUTER]=OPENROUTER_API_KEY
  [MISTRAL]=MISTRAL_API_KEY [DEEPSEEK]=DEEPSEEK_API_KEY [GOOGLE]=GOOGLE_API_KEY
  [XAI]=XAI_API_KEY         [TOGETHER]=TOGETHER_API_KEY
)
for name in NVIDIA OPENAI ANTHROPIC CEREBRAS GROQ OPENROUTER MISTRAL DEEPSEEK GOOGLE XAI TOGETHER; do
  var="${PROVIDERS[$name]}"
  ask_secret "$var" "$name API key"
done

# --- summary ----------------------------------------------------------------
echo
msg_info "About to create CT ${BL}$CT_ID${CL} (${CT_HOSTNAME}) — ${CT_CORES} core / ${CT_RAM}MB / ${CT_DISK}GB on ${CT_STORAGE}"

# --- template ---------------------------------------------------------------
if ! pveam list local 2>/dev/null | grep -q "$TEMPLATE"; then
  msg_info "Downloading template $TEMPLATE"
  pveam update >/dev/null 2>&1 || true
  pveam download local "$TEMPLATE" >/dev/null || die "template download failed"
fi
msg_ok "Template ready"

# --- create container -------------------------------------------------------
if [ "$CT_NET" = "dhcp" ]; then
  NETCFG="name=eth0,bridge=${CT_BRIDGE},ip=dhcp"
else
  NETCFG="name=eth0,bridge=${CT_BRIDGE},ip=${CT_NET}"
  [ -n "$CT_GW" ] && NETCFG="${NETCFG},gw=${CT_GW}"
fi

pct status "$CT_ID" >/dev/null 2>&1 && die "CT $CT_ID already exists — pick another ID or destroy it"

msg_info "Creating LXC"
pct create "$CT_ID" "local:vztmpl/${TEMPLATE}" \
  --hostname "$CT_HOSTNAME" \
  --cores "$CT_CORES" --memory "$CT_RAM" --swap 512 \
  --rootfs "${CT_STORAGE}:${CT_DISK}" \
  --net0 "$NETCFG" \
  --password "$CT_PASS" \
  --unprivileged 1 --features nesting=1 \
  --onboot 1 --ostype debian >/dev/null
pct start "$CT_ID"
msg_ok "Container created & started"

# wait for network/apt
sleep 5
msg_info "Installing dependencies (this takes a minute)"
pct exec "$CT_ID" -- bash -c "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq python3-venv python3-pip curl ca-certificates >/dev/null"
msg_ok "Dependencies installed"

# --- fetch app --------------------------------------------------------------
msg_info "Fetching proxy code"
pct exec "$CT_ID" -- bash -c "
  set -e
  mkdir -p /opt/nvidia-failover
  cd /opt/nvidia-failover
  curl -fsSL '${REPO_RAW}/nvidia_failover_proxy.py' -o nvidia_failover_proxy.py
  curl -fsSL '${REPO_RAW}/requirements.txt' -o requirements.txt
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
"
msg_ok "Proxy installed to /opt/nvidia-failover"

# --- build env file (provider keys + host binding) --------------------------
ENVLINES=$'PROXY_HOST=0.0.0.0\nPROXY_PORT='"${PROXY_PORT}"$'\nPROXY_DB_FILE=/opt/nvidia-failover/proxy.db\n'
[ -n "${LOCAL_OLLAMA_URL:-}" ] && ENVLINES+="LOCAL_OLLAMA_URL=${LOCAL_OLLAMA_URL}"$'\n'
for name in NVIDIA OPENAI ANTHROPIC CEREBRAS GROQ OPENROUTER MISTRAL DEEPSEEK GOOGLE XAI TOGETHER; do
  var="${PROVIDERS[$name]}"; val="${!var:-}"
  [ -n "$val" ] && ENVLINES+="${var}=${val}"$'\n'
done

msg_info "Writing service"
pct exec "$CT_ID" -- bash -c "cat > /opt/nvidia-failover/proxy.env" <<< "$ENVLINES"
pct exec "$CT_ID" -- chmod 600 /opt/nvidia-failover/proxy.env

pct exec "$CT_ID" -- bash -c "cat > /etc/systemd/system/nvidia-failover-proxy.service" <<'UNIT'
[Unit]
Description=NVIDIA Failover Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/nvidia-failover
EnvironmentFile=/opt/nvidia-failover/proxy.env
ExecStart=/opt/nvidia-failover/.venv/bin/python /opt/nvidia-failover/nvidia_failover_proxy.py
Restart=always
RestartSec=5
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
UNIT

pct exec "$CT_ID" -- systemctl daemon-reload
pct exec "$CT_ID" -- systemctl enable -q --now nvidia-failover-proxy
msg_ok "Service started"

# --- verify -----------------------------------------------------------------
sleep 4
IP="$(pct exec "$CT_ID" -- hostname -I 2>/dev/null | awk '{print $1}')"
if pct exec "$CT_ID" -- curl -fsS --max-time 8 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
  msg_ok "Health check passed"
else
  msg_error "Health check did not pass yet — check: pct exec $CT_ID -- journalctl -u nvidia-failover-proxy -e"
fi

echo
msg_ok "${APP} is ready"
echo -e "   Dashboard : ${BL}http://${IP}:${PROXY_PORT}/${CL}"
echo -e "   API base  : ${BL}http://${IP}:${PROXY_PORT}/v1${CL}"
echo -e "   Logs      : ${YW}pct exec ${CT_ID} -- journalctl -u nvidia-failover-proxy -f${CL}"
echo -e "   Update    : re-run this script's fetch step, or edit /opt/nvidia-failover/proxy.env then restart"
echo
