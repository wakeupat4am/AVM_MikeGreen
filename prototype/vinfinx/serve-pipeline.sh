#!/usr/bin/env bash
set -euo pipefail

# Run this on the GPU server (vishc-server-1), in tmux `vinfinx-api`. CPU only. Binds to loopback only; reach it
# through SSH -L (start-qwen-bridge.sh forwards this port too). docs/PIPELINE_API_PLAN.md.
ROOT="${VINFINX_ROOT:-/mnt/disk1/backup_user/24chuong.ta/vinfinx}"
PYTHON="${VINFINX_PYTHON:-$ROOT/env/bin/python}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=""
export VINFINX_ROOT="$ROOT"
export VINFINX_PIPELINE_PATH="${VINFINX_PIPELINE_PATH:-vinfinx-pipeline}"
export VINFINX_ALLOWED_ORIGINS="${VINFINX_ALLOWED_ORIGINS:-http://127.0.0.1:4173,http://localhost:4173}"
export VINFINX_REVIEWERS="${VINFINX_REVIEWERS:-jimmytrn154}"

cd "$APP_DIR"
exec "$PYTHON" -m uvicorn pipeline_api:app \
  --host 127.0.0.1 \
  --port "${VINFINX_PIPELINE_PORT:-8062}"
