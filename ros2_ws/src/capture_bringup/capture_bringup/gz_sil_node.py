"""Gazebo Harmonic SIL: physics + camera in gz-sim, drone body in Python.

Publishes the same ROS graph as sil_node so perception and guidance are unchanged:
  /catcher/image_raw, /capture/drone_pose, /capture/drone_twist, /capture/ball_truth
  /capture/setpoint  -> pose-hold the quadrotor in the sim
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np


def _repo_root():
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "capture_core")):
            return cur
        cur = os.path.dirname(cur)
    return os.environ.get("CAPTURE_ROOT", os.getcwd())


ROOT = _repo_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from capture_core.physics import DroneBody, TennisBall

from gz.math7 import Pose3d, Vector3d
from gz.msgs10 import image_pb2
from gz.sim8 import K_NULL_ENTITY, Link, Model, TestFixture, World, world_entity
from gz.transport13 import Node as GzNode


RGB_INT8 = 3


def _seconds(value, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return v * 1e-9 if v > 1e6 else v
    if hasattr(value, "total_seconds"):
        try:
            return float(value.total_seconds())
        except Exception:
            pass
    if hasattr(value, "seconds"):
        sec = value.seconds
        sec = sec() if callable(sec) else sec
        ns = getattr(value, "nanoseconds", 0)
        ns = ns() if callable(ns) else ns
        return float(sec) + float(ns or 0) * 1e-9
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pose(p, stamp, frame, yaw=None):
    msg = PoseStamped()
    msg.header = Header(stamp=stamp, frame_id=frame)
    msg.pose.position.x = float(p[0])
    msg.pose.position.y = float(p[1])
    msg.pose.position.z = float(p[2])
    if yaw is None:
        msg.pose.orientation.w = 1.0
    else:
        msg.pose.orientation.z = float(np.sin(yaw / 2.0))
        msg.pose.orientation.w = float(np.cos(yaw / 2.0))
    return msg


class GzSilNode(Node):
    def __init__(self):
        super().__init__("capture_sil")
        self.dt = float(self.declare_parameter("dt", 0.004).value)
        world_default = os.path.join(ROOT, "gazebo", "worlds", "dynamic_capture.world")
        self.world_path = str(self.declare_parameter("world", world_default).value)
        self.catch_radius = float(self.declare_parameter("catch_radius", 0.22).value)
        self.save_frame = str(self.declare_parameter("save_frame", "").value)

        models = os.path.join(ROOT, "gazebo", "models")
        existing = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
        os.environ["GZ_SIM_RESOURCE_PATH"] = models + (os.pathsep + existing if existing else "")

        self.drone = DroneBody(position=[0.0, 0.0, 1.55], yaw=np.pi / 2.0, v_max=8.0, a_max=6.5)
        self.ball = TennisBall(position=[0.4, 9.5, 2.4], velocity=[-0.2, -7.2, 3.1], drag=True)
        self.setpoint = self.drone.p.copy()
        self.set_yaw = self.drone.yaw
        self._lock = threading.Lock()
        self._ball_p = self.ball.p.copy()
        self._ball_v = self.ball.v.copy()
        self._caught = False
        self._min_net = 1e9
        self._n_img = 0
        self._latest_rgb = None

        self.image_pub = self.create_publisher(Image, "/catcher/image_raw", 10)
        self.ball_pub = self.create_publisher(PoseStamped, "/capture/ball_truth", 10)
        self.drone_pub = self.create_publisher(PoseStamped, "/capture/drone_pose", 10)
        self.twist_pub = self.create_publisher(TwistStamped, "/capture/drone_twist", 10)
        self.create_subscription(PoseStamped, "/capture/setpoint", self._on_setpoint, 10)

        self._gz = GzNode()
        if not self._gz.subscribe(image_pb2.Image, "/catcher/image", self._on_gz_image):
            raise RuntimeError("failed to subscribe to /catcher/image")

        self._fixture = TestFixture(self.world_path)
        self._fixture.on_pre_update(self._on_pre_update)
        self._fixture.finalize()
        self._server = self._fixture.server()
        if not self._server.run(False, 0, False):
            raise RuntimeError(f"gz-sim failed to start {self.world_path}")

        self.create_timer(1.0 / 30.0, self._publish_state)
        self.create_timer(1.0, self._status)
        self.get_logger().info(f"Gazebo Harmonic SIL started ({self.world_path})")

    def _on_setpoint(self, msg: PoseStamped):
        with self._lock:
            self.setpoint = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
            )
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w
            self.set_yaw = float(np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz))

    def _on_pre_update(self, info, ecm):
        try:
            dt = _seconds(info.dt, self.dt)
            if dt <= 0.0 or dt > 0.1:
                dt = self.dt
            world = World(world_entity(ecm))
            ball_ent = world.model_by_name(ecm, "tennis_ball")
            drone_ent = world.model_by_name(ecm, "catcher_drone")
            if ball_ent == K_NULL_ENTITY or drone_ent == K_NULL_ENTITY:
                return
            ball = Model(ball_ent)
            drone = Model(drone_ent)
            ball_link = Link(ball.link_by_name(ecm, "ball"))
            ball_link.enable_velocity_checks(ecm, True)

            with self._lock:
                setpoint = self.setpoint.copy()
                yaw_cmd = self.set_yaw

            self.ball.step(dt)
            self.drone.step(dt, setpoint, yaw_cmd=yaw_cmd)

            ball.set_world_pose_cmd(
                ecm,
                Pose3d(
                    float(self.ball.p[0]),
                    float(self.ball.p[1]),
                    float(self.ball.p[2]),
                    0.0,
                    0.0,
                    0.0,
                ),
            )
            ball_link.set_linear_velocity(
                ecm,
                Vector3d(float(self.ball.v[0]), float(self.ball.v[1]), float(self.ball.v[2])),
            )
            drone.set_world_pose_cmd(
                ecm,
                Pose3d(
                    float(self.drone.p[0]),
                    float(self.drone.p[1]),
                    float(self.drone.p[2]),
                    0.0,
                    0.0,
                    float(self.drone.yaw),
                ),
            )
            with self._lock:
                self._ball_p = self.ball.p.copy()
                self._ball_v = self.ball.v.copy()
        except Exception as exc:  # pragma: no cover - sim thread
            self.get_logger().error(f"gz pre_update: {exc}")

    def _on_gz_image(self, msg):
        try:
            h, w = int(msg.height), int(msg.width)
            buf = bytes(msg.data)
            if h <= 0 or w <= 0 or len(buf) < h * w * 3:
                return
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3)).copy()
            if int(msg.pixel_format_type) == RGB_INT8:
                bgr = rgb[:, :, ::-1].copy()
            else:
                bgr = rgb
            n = 0
            with self._lock:
                self._latest_rgb = bgr
                self._n_img = getattr(self, "_n_img", 0) + 1
                n = self._n_img
            if self.save_frame and n == 25:
                self._save_bgr(bgr)
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"gz image: {exc}")

    def _save_bgr(self, bgr):
        path = self.save_frame
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            import cv2

            cv2.imwrite(path, bgr)
            self.get_logger().info(f"saved camera frame to {path}")
        except Exception as exc:
            self.get_logger().warn(f"could not save frame: {exc}")

    def _publish_state(self):
        now = self.get_clock().now().to_msg()
        with self._lock:
            ball_p = self._ball_p.copy()
            img = None if self._latest_rgb is None else self._latest_rgb.copy()
            drone_p = self.drone.p.copy()
            drone_v = self.drone.v.copy()
            yaw = self.drone.yaw

        if img is not None:
            msg = Image()
            msg.header = Header(stamp=now, frame_id="front_camera_optical")
            msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
            msg.encoding = "bgr8"
            msg.step = msg.width * 3
            msg.data = img.tobytes()
            self.image_pub.publish(msg)

        self.ball_pub.publish(_pose(ball_p, now, "map"))
        self.drone_pub.publish(_pose(drone_p, now, "map", yaw=yaw))
        tw = TwistStamped()
        tw.header = Header(stamp=now, frame_id="map")
        tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = [float(v) for v in drone_v]
        self.twist_pub.publish(tw)

        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        net = drone_p + 0.32 * forward
        net[2] += 0.04
        dist = float(np.linalg.norm(net - ball_p))
        self._min_net = min(self._min_net, dist)
        if dist < self.catch_radius and ball_p[2] > 0.4 and not self._caught:
            self._caught = True
            self.get_logger().info(f"CATCH dist={dist:.3f}m drone={drone_p} ball={ball_p}")

    def _status(self):
        with self._lock:
            ball_p = self._ball_p.copy()
            ball_v = self._ball_v.copy()
            drone_p = self.drone.p.copy()
            kicked = True
            has_img = self._latest_rgb is not None
        self.get_logger().info(
            f"gz status kicked={kicked} img={int(has_img)} "
            f"ball=({ball_p[0]:.2f},{ball_p[1]:.2f},{ball_p[2]:.2f}) "
            f"v=({ball_v[0]:.2f},{ball_v[1]:.2f},{ball_v[2]:.2f}) "
            f"drone=({drone_p[0]:.2f},{drone_p[1]:.2f},{drone_p[2]:.2f}) "
            f"min_net={self._min_net:.2f} caught={int(self._caught)}"
        )

    def destroy_node(self):
        try:
            if getattr(self, "_server", None) is not None and self._server.is_running():
                # Stop accepting new image callbacks before tearing the server down.
                self._latest_rgb = None
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    rclpy.init(args=args)
    node = GzSilNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
