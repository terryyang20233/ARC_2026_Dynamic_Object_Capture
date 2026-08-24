"""Kalman ball state -> intercept setpoint for the drone."""

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
from std_msgs.msg import Header

from capture_core.intercept import DroneLimits, InterceptPlanner


class GuidanceNode(Node):
    def __init__(self):
        super().__init__("capture_guidance")
        self.planner = InterceptPlanner(DroneLimits(v_max=8.0, a_max=6.5, catch_offset=0.32))
        self.drone_p = np.array([0.0, 0.0, 1.55])
        self.drone_v = np.zeros(3)
        self.ball_p = None
        self.ball_v = np.zeros(3)
        self.pub = self.create_publisher(PoseStamped, "/capture/setpoint", 10)
        self.create_subscription(PoseStamped, "/capture/ball_pose", self._on_ball, 10)
        self.create_subscription(TwistStamped, "/capture/ball_twist", self._on_ball_twist, 10)
        self.create_subscription(PoseStamped, "/capture/drone_pose", self._on_drone, 10)
        self.create_subscription(TwistStamped, "/capture/drone_twist", self._on_drone_twist, 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info("guidance node ready")

    def _on_ball(self, msg: PoseStamped):
        self.ball_p = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )

    def _on_ball_twist(self, msg: TwistStamped):
        self.ball_v = np.array(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        )

    def _on_drone(self, msg: PoseStamped):
        self.drone_p = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )

    def _on_drone_twist(self, msg: TwistStamped):
        self.drone_v = np.array(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        )

    def _tick(self):
        if self.ball_p is None:
            return
        plan = self.planner.plan(self.ball_p, self.ball_v, self.drone_p, self.drone_v)
        now = self.get_clock().now().to_msg()
        msg = PoseStamped()
        msg.header = Header(stamp=now, frame_id="map")
        t = plan.drone_target
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, t)
        msg.pose.orientation.z = float(np.sin(plan.drone_yaw / 2.0))
        msg.pose.orientation.w = float(np.cos(plan.drone_yaw / 2.0))
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GuidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
