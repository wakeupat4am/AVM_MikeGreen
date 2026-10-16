#!/usr/bin/env bash
set -euo pipefail

# Run this on the Mac. The SSH host alias must reach the server over NetBird or the LAN,
# never through a public relay (VinSoc policy).
HOST="${VINFINX_SSH_HOST:-vishc-1}"
PORT="${VINFINX_QWEN_PORT:-8061}"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "VinFinx Qwen bridge is already listening on http://127.0.0.1:$PORT"
  exit 0
fi

# ControlPath=none: a dedicated connection, independent of any shared ControlMaster session.
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ControlPath=none -f -N \
  -L "$PORT:127.0.0.1:$PORT" "$HOST"

echo "VinFinx Qwen bridge started on http://127.0.0.1:$PORT (via $HOST)"
