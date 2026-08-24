"""Image -> YOLO/colour blob -> pinhole depth -> Kalman world state."""

import os
import sys

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
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from capture_core.geometry import camera_to_world
from capture_core.kalman import KalmanFilter3D
from capture_core.perception import estimate_3d_pinhole
from capture_sim.renderer import Camera
from capture_sim.sil import _colour_detect


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("capture_perception")
        self.cam = Camera()
        self.kf = KalmanFilter3D(meas_var=0.08, process_var=0.04)
        self.last_t = None
        self.drone_p = np.array([0.0, 0.0, 1.55])
        self.drone_yaw = np.pi / 2.0
        self.use_yolo = bool(self.declare_parameter("use_yolo", False).value)
        weights = self.declare_parameter(
            "yolo_weights",
            os.path.join(ROOT, "Object_Detection_Test_1", "train3", "weights", "best.pt"),
        ).value
        self.detect = _colour_detect
        if self.use_yolo:
            from capture_sim.sil import _load_yolo

            self.detect = _load_yolo(weights, 320)
            self.get_logger().info(f"YOLO loaded from {weights}")
        self.pose_pub = self.create_publisher(PoseStamped, "/capture/ball_pose", 10)
        self.twist_pub = self.create_publisher(TwistStamped, "/capture/ball_twist", 10)
        self.create_subscription(Image, "/catcher/image_raw", self._on_image, 10)
        self.create_subscription(PoseStamped, "/capture/drone_pose", self._on_drone, 10)
        self.get_logger().info("perception node ready")

    def _on_drone(self, msg: PoseStamped):
        self.drone_p = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        qz, qw = msg.pose.orientation.z, msg.pose.orientation.w
        self.drone_yaw = float(np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz))

    def _on_image(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        bbox = self.detect(img) if self.detect is not _colour_detect else _colour_detect(img)
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.last_t is not None and self.kf.is_initialized:
            self.kf.predict(max(t - self.last_t, 1e-3))
        self.last_t = t
        if bbox is not None:
            X, Y, Z = estimate_3d_pinhole(bbox, self.cam.K)
            if Z is not None:
                meas = camera_to_world(X, Y, Z, self.drone_p, self.drone_yaw)
                self.kf.update(meas)
        if not self.kf.is_initialized:
            return
        now = self.get_clock().now().to_msg()
        pmsg = PoseStamped()
        pmsg.header = Header(stamp=now, frame_id="map")
        p = self.kf.position
        pmsg.pose.position.x, pmsg.pose.position.y, pmsg.pose.position.z = map(float, p)
        pmsg.pose.orientation.w = 1.0
        self.pose_pub.publish(pmsg)
        tmsg = TwistStamped()
        tmsg.header = pmsg.header
        v = self.kf.velocity
        tmsg.twist.linear.x, tmsg.twist.linear.y, tmsg.twist.linear.z = map(float, v)
        self.twist_pub.publish(tmsg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
