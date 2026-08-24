#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate ros2-humble
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CAPTURE_ROOT="$ROOT"
cd "$ROOT/ros2_ws"
colcon build --merge-install --packages-select capture_bringup
set +u
source install/setup.bash
export AMENT_PREFIX_PATH="$ROOT/ros2_ws/install:${AMENT_PREFIX_PATH:-}"
set -u
exec ros2 launch capture_bringup sil.launch.py
