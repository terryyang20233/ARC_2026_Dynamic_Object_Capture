#!/usr/bin/env bash
# Workstation Gazebo Harmonic + ROS 2 Humble capture pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONNOUSERSITE=1
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-3.3}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-26}"

# NVIDIA GLX is broken on this machine; force EGL headless unless the user
# explicitly keeps DISPLAY and has a working GPU driver.
if [[ "${FORCE_GZ_GUI:-}" != "1" ]]; then
  unset DISPLAY WAYLAND_DISPLAY
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # Isolate from other colcon overlays on this machine (e.g. fsc_autopilot_ws).
  export AMENT_PREFIX_PATH=/opt/ros/humble
  export CMAKE_PREFIX_PATH=/opt/ros/humble
  export COLCON_PREFIX_PATH=/opt/ros/humble
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CAPTURE_ROOT="$ROOT"
export GZ_SIM_RESOURCE_PATH="$ROOT/gazebo/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

cd "$ROOT/ros2_ws"
colcon build --merge-install --packages-select capture_bringup --symlink-install
set +u
# shellcheck disable=SC1091
source install/local_setup.bash
export AMENT_PREFIX_PATH="$ROOT/ros2_ws/install:${AMENT_PREFIX_PATH:-}"
set -u
exec ros2 launch capture_bringup gazebo.launch.py
