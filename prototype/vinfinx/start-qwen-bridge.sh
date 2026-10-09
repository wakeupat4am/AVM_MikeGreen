#!/usr/bin/env bash
set -euo pipefail

if lsof -nP -iTCP:8061 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "VinFinx Qwen bridge is already listening on http://127.0.0.1:8061"
  exit 0
fi

ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -f -N \
  -L 8061:127.0.0.1:8061 cecs-server1

echo "VinFinx Qwen bridge started on http://127.0.0.1:8061"
