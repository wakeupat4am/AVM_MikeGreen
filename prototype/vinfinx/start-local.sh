#!/bin/bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$APP_DIR/.venv-local/bin/python"
if [ ! -x "$PY" ]; then
  echo "Create .venv-local and install requirements/local.txt first (see LOCAL_DEMO.md)."
  exit 1
fi
cd "$APP_DIR"
mkdir -p .local-demo
if [ ! -f local_app/assets/market.json ]; then "$PY" local_app/prepare_data.py; fi
if [ ! -f ../../output/pdf/vinfinx-sample-payslip.pdf ]; then "$PY" local_app/generate_sample.py; fi
if [ ! -x .local-demo/vision-ocr ] && command -v swiftc >/dev/null 2>&1; then
  swiftc -module-cache-path .local-demo/swift-cache local_app/ocr.swift -o .local-demo/vision-ocr || echo "Image OCR unavailable; PDF text extraction still works."
fi
exec "$PY" local_app/server.py
