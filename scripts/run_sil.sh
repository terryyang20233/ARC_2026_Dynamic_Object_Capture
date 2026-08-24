#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONNOUSERSITE=1

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -x "${HOME}/miniforge3/envs/arc-yolo/bin/python" ]]; then
  PY="${HOME}/miniforge3/envs/arc-yolo/bin/python"
  # Jetson Nano OpenMP preload (harmless to skip on x86_64)
  if [[ -f /usr/lib/aarch64-linux-gnu/libgomp.so.1 ]]; then
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  fi
else
  PY="${PYTHON:-python3}"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
exec "$PY" -m capture_sim.sil "$@"
