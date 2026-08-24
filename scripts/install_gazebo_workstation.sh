#!/usr/bin/env bash
# Workstation installer for Gazebo Classic 11 + ROS 2 camera bridge.
# This Jetson Nano cannot run Gazebo; copy the repo to Ubuntu 22.04 x86_64 and run:
#
#   sudo apt update
#   sudo apt install -y gazebo11 libgazebo11-dev ros-humble-gazebo-ros-pkgs \
#       ros-humble-gazebo-ros ros-humble-ros-gz-bridge
#   echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$(cd "$(dirname "$0")/.." && pwd)/gazebo/models" >> ~/.bashrc
#   gazebo worlds/dynamic_capture.world
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo: $ROOT"
echo "On this Nano: sudo is required and Gazebo is not feasible (4 GB RAM)."
echo "Workstation commands are printed above. Refusing to apt-install here."
exit 1
