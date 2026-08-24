#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${HOME}/miniforge3/envs/arc-yolo/bin/python"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -m capture_sim.sil "$@"
