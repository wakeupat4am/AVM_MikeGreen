#!/usr/bin/env bash
# Each tester uses their own authorized VPN + SSH access. Never share SSH keys.
set -euo pipefail
echo 'Pilot: http://127.0.0.1:4175/ (keep this SSH session open)'
exec ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -N -L 127.0.0.1:4175:127.0.0.1:4175 cecs-server1
