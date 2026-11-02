#!/bin/bash
set -euo pipefail
# Only forwarding: never starts/stops a model or touches the shared GPU.
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8061:127.0.0.1:8061 \
  -L 127.0.0.1:8062:127.0.0.1:8062 cecs-server1
