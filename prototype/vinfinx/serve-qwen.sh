#!/usr/bin/env bash
set -euo pipefail

# Run this on cecs-server1 from the llms-conformity-rq1 repository root.
# Override any path or GPU assignment through environment variables.
ROOT="${VINFINX_RQ1_ROOT:-/home/ubuntu/llms-conformity-rq1}"
PYTHON="${VINFINX_PYTHON:-$ROOT/server_experiments/.venv/bin/python}"
MODEL_PATH="${VINFINX_QWEN_PATH:-$ROOT/multimodal_rq1/models/Qwen3.5-9B}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$ROOT/multimodal_rq1/.deps-qwen35:$ROOT:${PYTHONPATH:-}"
export VINFINX_QWEN_PATH="$MODEL_PATH"
export VINFINX_API_PATH="${VINFINX_API_PATH:-vinfinx-api}"
export VINFINX_ALLOWED_ORIGINS="${VINFINX_ALLOWED_ORIGINS:-http://127.0.0.1:4173,http://localhost:4173}"

exec "$PYTHON" -m uvicorn vinfinx_qwen_api:app \
  --host 127.0.0.1 \
  --port "${VINFINX_QWEN_PORT:-8061}"
