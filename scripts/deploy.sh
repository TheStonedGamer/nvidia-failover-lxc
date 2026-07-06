#!/bin/bash
# Deploy NVIDIA Failover Proxy to Proxmox LXC Container
# 
# Usage: ./deploy.sh <pve-host-ip> <container-ip> <desktop-ollama-ip> <nvapi-key>
# Example: ./deploy.sh 10.0.0.98 10.0.0.199 10.0.0.127 'nvapi-YOUR-KEY-HERE'

set -e

PVE_HOST="${1:-10.0.0.98}"
CONTAINER_IP="${2:-10.0.0.199}"
DESKTOP_IP="${3:-10.0.0.127}"
NVAPI_KEY="${4}"

CTID=3000
PROXY_PORT=5002

if [ -z "$NVAPI_KEY" ]; then
    echo "ERROR: NVIDIA API key required. Pass as 4th argument."
    echo "Usage: $0 <pve-host> <container-ip> <desktop-ip> <nvapi-key>"
    exit 1
fi

echo "=== NVIDIA Failover Proxy LXC Deployment ==="
echo "PVE Host: $PVE_HOST"
echo "Container IP: $CONTAINER_IP"
echo "Desktop Ollama: $DESKTOP_IP"
echo "Proxy Port: $PROXY_PORT"
echo ""

# Step 1: Download Ubuntu template
echo "[1/8] Downloading Ubuntu 24.04 template..."
ssh root@$PVE_HOST "pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst" || true

# Step 2: Create container
echo "[2/8] Creating LXC container $CTID..."
ssh root@$PVE_HOST "<< 'CREATEEOF'
pct destroy $CTID 2>/dev/null || true
pct create $CTID local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst \\
    --rootfs local-lvm:4 \\
    --cores 2 \\
    --memory 2048 \\
    --swap 1024 \\
    --unprivileged 1 \\
    --features nesting=1 \\
    --password changeme \\
    --hostname nvidia-proxy \\
    --net0 name=eth0,bridge=vmbr0,ip=$CONTAINER_IP/24,gw=$(echo $CONTAINER_IP | sed 's/\\.[0-9]*\\$/\\.1/') \\
    --arch amd64 \\
    --ostype ubuntu

cat >> /etc/pve/lxc/$CTID.conf << EOF
lxc.apparmor.profile: unconfined
lxc.apparmor.allow_nesting: 1
EOF

pct start $CTID
sleep 5
CREATEEOF

# Step 3: Install dependencies
echo "[3/8] Installing Python and dependencies..."
ssh root@$PVE_HOST "pct exec $CTID -- apt-get update && apt-get install -y python3-venv git curl tmux jq"

# Step 4: Deploy proxy code
echo "[4/8] Copying proxy code to container..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Create tarball of src folder
tar -cf /tmp/src.tar src --exclude="__pycache__" --exclude="*.pyc"

# Copy files via PVE host
scp nvidia_failover_proxy.py root@$PVE_HOST:/root/
scp /tmp/src.tar root@$PVE_HOST:/root/

ssh root@$PVE_HOST << 'COPYEOF'
pct exec $CTID -- mkdir -p /root/model-router
pct push $CTID /root/nvidia_failover_proxy.py /root/model-router/nvidia_failover_proxy.py
pct push $CTID /root/src.tar /root/model-router/src.tar
pct exec $CTID -- bash -c 'cd /root/model-router && tar -xf src.tar && rm src.tar'
COPYEOF

rm -f /tmp/src.tar

# Step 5: Setup virtual environment
echo "[5/8] Creating Python virtual environment..."
ssh root@$PVE_HOST "pct exec $CTID -- bash -c 'cd /root/model-router && python3 -m venv .venv && source .venv/bin/activate && pip install httpx fastapi uvicorn'"

# Step 6: Create systemd service
echo "[6/8] Creating systemd service..."
ssh root@$PVE_HOST << SERVICEEOF
pct exec $CTID -- bash -c 'cat > /etc/systemd/system/nvidia-failover-proxy.service << EOF
[Unit]
Description=NVIDIA Failover Proxy
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/model-router
Environment=NVIDIA_API_KEY=$NVAPI_KEY
Environment=REFINER_BASE_URL=http://$DESKTOP_IP:11434/v1
Environment=LOCAL_OLLAMA_URL=http://$DESKTOP_IP:11434/v1
ExecStart=/root/model-router/.venv/bin/python /root/model-router/nvidia_failover_proxy.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Fix host binding (0.0.0.0 for external access)
sed -i "s/127.0.0.1/0.0.0.0/g" /root/model-router/nvidia_failover_proxy.py

systemctl daemon-reload
systemctl enable nvidia-failover-proxy
systemctl start nvidia-failover-proxy'
SERVICEEOF

# Step 7: Wait and verify
echo "[7/8] Waiting for service to start..."
sleep 5

echo "[8/8] Verifying deployment..."
HEALTH=$(ssh root@$PVE_HOST "pct exec $CTID -- curl -s --max-time 10 http://127.0.0.1:$PROXY_PORT/health")
OK=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok', False))" 2>/dev/null || echo "false")

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Container: $CTID"
echo "Proxy URL: http://$CONTAINER_IP:$PROXY_PORT/v1"
echo "Dashboard: http://$CONTAINER_IP:$PROXY_PORT/"
echo "Status: $([ "$OK" = "True" ] && echo '✓ Running' || echo '✗ Check logs')"
echo ""
echo "To check logs:"
echo "  ssh root@$PVE_HOST \"pct exec $CTID -- journalctl -u nvidia-failover-proxy -f\""
echo ""
echo "To enter container:"
echo "  ssh root@$PVE_HOST \"pct enter $CTID\""