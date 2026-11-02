#!/usr/bin/env bash
# Run on cecs-server1 only, inside a scoped tmux session. No public listener.
set -euo pipefail
umask 077
PILOT_ROOT=/home/ubuntu/vinfinx-pilot
PILOT_RELEASE="$(readlink -f "$PILOT_ROOT/current")"
case "$PILOT_RELEASE" in "$PILOT_ROOT"/releases/*) ;; *) echo 'Invalid release pointer'; exit 1 ;; esac
export VINFINX_LOCAL_STATE="$PILOT_ROOT/state"
export VINFINX_LOCAL_PORT=4175
export VINFINX_MODEL_DIR="$PILOT_RELEASE/model"
export VINFINX_MODEL_MANIFEST="$PILOT_RELEASE/model/manifest.json"
export VINFINX_HYBRID_PATH="$PILOT_RELEASE/model/hierarchy.json"
export VINFINX_WORKSPACE_ROOT="$PILOT_RELEASE/fixtures"
export VINFINX_TESSERACT="$PILOT_ROOT/env/bin/tesseract"
export VINFINX_TESS_LANG=eng
export TESSDATA_PREFIX="$PILOT_ROOT/tessdata_best-e12c65a"
export OMP_THREAD_LIMIT=2
export TZ=Asia/Ho_Chi_Minh
export VINFINX_REQUIRE_OCR=1
export VINFINX_REQUIRE_QWEN=0
export VINFINX_QWEN_URL=http://127.0.0.1:8061/vinfinx-api
export VINFINX_QWEN_ORIGIN=http://127.0.0.1:4173
test -f "$VINFINX_LOCAL_STATE/pilot_accounts.json" || { echo 'Provision a fresh pilot state explicitly first.'; exit 1; }
cd "$PILOT_RELEASE/local_app"
case "${1:-serve}" in
  serve) exec "$PILOT_ROOT/env/bin/python" -u supervise.py ;;
  check) exec "$PILOT_ROOT/env/bin/python" check_deployed.py ;;
  test) exec "$PILOT_ROOT/env/bin/python" -m unittest -v test_pilot ;;
  restart-check) exec "$PILOT_ROOT/env/bin/python" check_restart.py ;;
  backup) exec "$PILOT_ROOT/env/bin/python" backup.py create "${2:?New private backup directory required}" ;;
  verify) exec "$PILOT_ROOT/env/bin/python" backup.py verify "${2:?Backup directory required}" ;;
  restore) exec "$PILOT_ROOT/env/bin/python" backup.py restore "${2:?Backup directory required}" --to "${3:?New restore directory required}" ;;
  *) echo 'Use serve, check, test, restart-check, backup, verify or restore.'; exit 2 ;;
esac
