#!/usr/bin/env bash
# This repo targets Gazebo Harmonic (gz-sim 8) + ROS 2 Humble on Ubuntu 22.04 x86_64.
# Gazebo Classic 11 is not installed on this workstation and gazebo11 has no apt candidate.
#
# Do not apt-install from this script (machine-wide changes need an explicit request).
# Python deps go in the repo venv:  python3 -m venv --system-site-packages .venv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo: $ROOT"
echo "Use:  $ROOT/scripts/run_gazebo.sh"
echo "Requires: Ubuntu 22.04, ROS 2 Humble, Gazebo Harmonic 8, python3-venv."
exit 0
