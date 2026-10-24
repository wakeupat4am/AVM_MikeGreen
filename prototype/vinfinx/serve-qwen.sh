#!/usr/bin/env bash
set -euo pipefail

# Run this on the GPU server (vishc-server-1). Binds to loopback only; reach it through SSH -L.
# Override any path or GPU assignment through environment variables.
# Paths come from the environment or ~/.config/vinfinx/paths.env (see ai/paths.py), never from this file.
if [[ -f "$HOME/.config/vinfinx/paths.env" ]]; then set -a; . "$HOME/.config/vinfinx/paths.env"; set +a; fi
ROOT="${VINFINX_ROOT:-$HOME/vinfinx}"
PYTHON="${VINFINX_PYTHON:-$ROOT/env/bin/python}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the agreed GPU(s), for example CUDA_VISIBLE_DEVICES=7" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES
export VINFINX_QWEN_PATH="${VINFINX_QWEN_PATH:-$ROOT/models/Qwen3.5-9B}"
export VINFINX_GPU_MAX_MEMORY="${VINFINX_GPU_MAX_MEMORY:-21GiB}"
export VINFINX_API_PATH="${VINFINX_API_PATH:-vinfinx-api}"
export VINFINX_ALLOWED_ORIGINS="${VINFINX_ALLOWED_ORIGINS:-http://127.0.0.1:4173,http://localhost:4173}"
export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export HF_HUB_OFFLINE=1

cd "$APP_DIR"
exec "$PYTHON" -m uvicorn qwen_api:app \
  --host 127.0.0.1 \
  --port "${VINFINX_QWEN_PORT:-8061}"
