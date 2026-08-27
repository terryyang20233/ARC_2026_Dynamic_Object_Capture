# Dynamic Object Detection and Capture for UAVs

Summer research from the University of Toronto Institute for Aerospace Studies (UTIAS) Aerial Robotics Club (ARC).

The stack closes the loop from a single camera to a catch setpoint: **detect → range → world frame → Kalman filter → intercept planner**. The target is a tennis ball in freefall. Verification is software-in-the-loop (Python SIL and Gazebo Harmonic / ROS 2 Humble). This is a research prototype, not a flight-ready autopilot.

Authors: Terry Yang, Tommy Yang, and Felicia Zhou.

## Intended use

This repository is for **academic research and education**: simulated and lab capture of a tennis ball with a netted research quadrotor.

It is **not** a targeting, surveillance, or weapons system. Do not use it to intercept people, vehicles, or anything other than the documented tennis-ball scenario. Real flight requires institutional approval, airspace authorization, and a safety-reviewed flight stack. The intercept planner publishes setpoints only; it does not include a motor mixer or a production PX4/ArduPilot integration.

## Privacy and data

- No API keys, passwords, or private credentials are stored in this repository.
- Jetson camera test dumps are excluded (see `.gitignore`).
- YOLO training visualizations and `Object_Detection_Test_1/predict/` images come from a public Roboflow tennis-ball dataset (`Tennis-Ball-1` in Colab). They may include people who appear in that third-party dataset. Weights are in `Object_Detection_Test_1/train3/weights/`.
- Hardware photos in `ARC_2026_presentation/` show a lab/office prototype, not a private address.

If you fork this work, do not commit webcam captures, flight logs with GPS, or personal photos.

## Repository layout

| Path | Role |
| --- | --- |
| `capture_core/` | Geometry, pinhole/PnP ranging, Kalman filter, intercept planner |
| `capture_sim/` | Closed-loop Python SIL and synthetic camera |
| `ros2_ws/src/capture_bringup/` | ROS 2 Humble nodes and launch files |
| `gazebo/` | Catcher drone and tennis-ball models, capture world |
| `scripts/` | SIL and Gazebo launch helpers |
| `tests/` | Unit tests for perception, filter, planner, and SIL |
| `Object_Detection_Test_1/` | YOLO11s training artifacts and live-camera demo |
| `Distance_Detection/`, `3D_Reconstruction /` | Early ranging and filter experiments |
| `Nano_Files/` | ONNX export for Jetson (`best.onnx`). TensorRT `.engine` files are local-only |
| `ARC_2026_presentation/` | Beamer source and slides |

## Quick start

Python 3.10+ with NumPy is enough for the core SIL and tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
python -m unittest tests.test_capture_core
./scripts/run_sil.sh
```

Optional live-camera demos (`Object_Detection_Test_1/live_demo.py` and the ranging scripts) need `opencv-python` and `ultralytics`, plus the weights at `Object_Detection_Test_1/train3/weights/best.pt`.

Gazebo + ROS 2 (Ubuntu 22.04, ROS 2 Humble, Gazebo Harmonic):

```bash
./scripts/run_gazebo.sh
```

## License

MIT. See [LICENSE](LICENSE).
