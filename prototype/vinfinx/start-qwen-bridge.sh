#!/usr/bin/env bash
set -euo pipefail

# Run this on the Mac. The SSH host alias must reach the server over NetBird or the LAN,
# never through a public relay (VinSoc policy). Forwards the Qwen API (8061) and the
# pipeline API (8062); a port that is already listening is left alone.
HOST="${VINFINX_SSH_HOST:-vishc-server-1}"
PORTS=("${VINFINX_QWEN_PORT:-8061}" "${VINFINX_PIPELINE_PORT:-8062}")

FORWARDS=()
for PORT in "${PORTS[@]}"; do
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $PORT is already forwarded (http://127.0.0.1:$PORT)"
  else
    FORWARDS+=(-L "$PORT:127.0.0.1:$PORT")
  fi
done
if [[ ${#FORWARDS[@]} -eq 0 ]]; then
  exit 0
fi

# ControlPath=none: a dedicated connection, independent of any shared ControlMaster session.
# ServerAlive*: a keep-alive every 30 s, so an idle tunnel over NetBird is not dropped silently; after 3 missed replies ssh
# exits and frees the ports, and running this script again restores the bridge.
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ControlPath=none \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -f -N "${FORWARDS[@]}" "$HOST"

echo "VinFinx bridge started (${FORWARDS[*]}) via $HOST"
