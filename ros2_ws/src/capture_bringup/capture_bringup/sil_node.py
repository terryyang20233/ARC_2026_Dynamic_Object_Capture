"""Publish synthetic camera frames from the SIL renderer (no Gazebo required)."""

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

from capture_core.physics import DroneBody, TennisBall
from capture_sim.renderer import Camera, render_frame


class SilNode(Node):
    def __init__(self):
        super().__init__("capture_sil")
        self.dt = float(self.declare_parameter("dt", 0.033).value)
        self.cam = Camera()
        self.ball = TennisBall(position=[0.4, 9.5, 2.4], velocity=[-0.2, -7.2, 3.1], drag=True)
        self.drone = DroneBody(position=[0.0, 0.0, 1.55], yaw=np.pi / 2.0)
        self.image_pub = self.create_publisher(Image, "/catcher/image_raw", 10)
        self.ball_pub = self.create_publisher(PoseStamped, "/capture/ball_truth", 10)
        self.drone_pub = self.create_publisher(PoseStamped, "/capture/drone_pose", 10)
        self.twist_pub = self.create_publisher(TwistStamped, "/capture/drone_twist", 10)
        self.setpoint = self.drone.p.copy()
        self.set_yaw = self.drone.yaw
        self.create_subscription(PoseStamped, "/capture/setpoint", self._on_setpoint, 10)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info("SIL camera/physics node started")

    def _on_setpoint(self, msg: PoseStamped):
        self.setpoint = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.set_yaw = float(np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz))

    def _tick(self):
        self.drone.step(self.dt, self.setpoint, yaw_cmd=self.set_yaw)
        self.ball.step(self.dt)
        img, _proj = render_frame(self.ball.p, self.drone.p, self.drone.yaw, self.cam)
        now = self.get_clock().now().to_msg()
        self.image_pub.publish(_bgr_to_image(img, now))
        self.ball_pub.publish(_pose(self.ball.p, now, "map"))
        self.drone_pub.publish(_pose(self.drone.p, now, "map", yaw=self.drone.yaw))
        tw = TwistStamped()
        tw.header = Header(stamp=now, frame_id="map")
        tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = [float(v) for v in self.drone.v]
        self.twist_pub.publish(tw)


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


def _bgr_to_image(img, stamp):
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id="front_camera_optical")
    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = img.tobytes()
    return msg


def main(args=None):
    rclpy.init(args=args)
    node = SilNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
