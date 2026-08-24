"""Closed-loop software-in-the-loop capture scenario.

Vision -> pinhole range -> world frame -> Kalman -> intercept planner -> drone.
Uses the synthetic camera by default. Pass --yolo to run the trained weights.
"""

from dataclasses import dataclass
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from capture_core.geometry import camera_to_world
from capture_core.intercept import DroneLimits, InterceptPlanner
from capture_core.kalman import KalmanFilter3D
from capture_core.perception import estimate_3d_pinhole
from capture_core.physics import DroneBody, TennisBall
from capture_sim.renderer import Camera, render_frame


@dataclass
class SilConfig:
    dt: float = 1.0 / 30.0
    duration: float = 2.8
    use_yolo: bool = False
    yolo_weights: str = ""
    yolo_imgsz: int = 320
    save_video: str = ""
    seed: int = 0
    catch_radius: float = 0.22
    meas_noise: float = 0.03


def _colour_detect(img, proj_hint=None):
    """Optic-yellow blob. OpenCV if present, otherwise a NumPy BGR threshold."""
    try:
        import cv2

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 80, 80), (45, 255, 255))
        mask = cv2.medianBlur(mask, 5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None if proj_hint is None else proj_hint["bbox"]
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 12:
            return None if proj_hint is None else proj_hint["bbox"]
        x, y, w, h = cv2.boundingRect(c)
        return [float(x), float(y), float(x + w), float(y + h)]
    except Exception:
        pass
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    mask = (g > 140) & (r > 90) & (b < 130) & (g + r > 2 * b + 40)
    ys, xs = np.where(mask)
    if xs.size < 12:
        return None if proj_hint is None else proj_hint["bbox"]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _load_yolo(weights, imgsz):
    from ultralytics import YOLO

    model = YOLO(weights)
    def _infer(img):
        res = model.predict(img, imgsz=imgsz, verbose=False, conf=0.25)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return None
        xyxy = res[0].boxes.xyxy.cpu().numpy()
        conf = res[0].boxes.conf.cpu().numpy()
        i = int(np.argmax(conf))
        return [float(x) for x in xyxy[i]]
    return _infer


def run_sil(cfg: SilConfig = None, detector=None):
    cfg = cfg or SilConfig()
    rng = np.random.default_rng(cfg.seed)
    cam = Camera()
    ball = TennisBall(position=[0.4, 9.5, 2.4], velocity=[-0.2, -7.2, 3.1], drag=True)
    drone = DroneBody(position=[0.0, 0.0, 1.55], yaw=np.pi / 2.0, v_max=8.0, a_max=6.5)
    kf = KalmanFilter3D(meas_var=0.08, process_var=0.04)
    planner = InterceptPlanner(DroneLimits(v_max=8.0, a_max=6.5, catch_offset=0.32))

    if detector is None and cfg.use_yolo:
        weights = cfg.yolo_weights or os.path.join(
            ROOT, "Object_Detection_Test_1", "train3", "weights", "best.pt"
        )
        detector = _load_yolo(weights, cfg.yolo_imgsz)

    writer = None
    if cfg.save_video:
        import cv2

        os.makedirs(os.path.dirname(os.path.abspath(cfg.save_video)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.save_video, fourcc, 1.0 / cfg.dt, (cam.width, cam.height))

    caught = False
    min_net = 1e9
    log = []
    t = 0.0
    plan = None
    t0 = time.time()
    steps = int(cfg.duration / cfg.dt)
    for k in range(steps):
        img, proj = render_frame(ball.p, drone.p, drone.yaw, cam)
        bbox = None
        if detector is not None:
            bbox = detector(img)
        else:
            bbox = _colour_detect(img, proj)

        if bbox is not None:
            X, Y, Z = estimate_3d_pinhole(bbox, cam.K)
            if Z is not None:
                meas = camera_to_world(X, Y, Z, drone.p, drone.yaw)
                meas = meas + rng.normal(0.0, cfg.meas_noise, size=3)
                if kf.is_initialized:
                    kf.predict(cfg.dt)
                kf.update(meas)

        if kf.is_initialized:
            plan = planner.plan(kf.position, kf.velocity, drone.p, drone.v)

        if plan is not None:
            drone.step(cfg.dt, plan.drone_target, v_ff=plan.drone_vel_ff, yaw_cmd=plan.drone_yaw)
        else:
            drone.step(cfg.dt, drone.p, yaw_cmd=drone.yaw)

        ball.step(cfg.dt)

        forward = np.array([np.cos(drone.yaw), np.sin(drone.yaw), 0.0])
        net = drone.p + planner.limits.catch_offset * forward
        net[2] += 0.04
        dist = float(np.linalg.norm(net - ball.p))
        min_net = min(min_net, dist)
        if dist < cfg.catch_radius and ball.p[2] > 0.4:
            caught = True

        log.append(
            {
                "t": t,
                "ball": ball.p.copy(),
                "drone": drone.p.copy(),
                "dist": dist,
                "detected": bbox is not None,
                "feasible": None if plan is None else plan.feasible,
                "t_go": None if plan is None else plan.t_go,
            }
        )
        if writer is not None:
            import cv2

            vis = img.copy()
            cv2.putText(
                vis,
                f"t={t:.2f} d={dist:.2f}m caught={int(caught)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            writer.write(vis)
        if caught:
            break
        t += cfg.dt

    if writer is not None:
        writer.release()
    return {
        "caught": caught,
        "min_net_dist": min_net,
        "t_catch": t if caught else None,
        "steps": len(log),
        "elapsed_s": time.time() - t0,
        "log": log,
        "final_ball": ball.p.copy(),
        "final_drone": drone.p.copy(),
    }


def main():
    import argparse

    p = argparse.ArgumentParser(description="Closed-loop capture SIL")
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--weights", default="")
    p.add_argument("--save", default="")
    p.add_argument("--seconds", type=float, default=2.8)
    args = p.parse_args()
    cfg = SilConfig(
        use_yolo=args.yolo,
        yolo_weights=args.weights,
        save_video=args.save,
        duration=args.seconds,
    )
    out = run_sil(cfg)
    print(
        f"caught={out['caught']} min_net={out['min_net_dist']:.3f}m "
        f"t={out['t_catch']} steps={out['steps']} wall={out['elapsed_s']:.2f}s"
    )
    if not out["caught"]:
        sys.exit(2)


SilConfig = SilConfig


if __name__ == "__main__":
    main()
