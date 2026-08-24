import os

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if not os.path.isdir(os.path.join(ROOT, "capture_core")):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))


def generate_launch_description():
    env_py = SetEnvironmentVariable("PYTHONPATH", ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env_root = SetEnvironmentVariable("CAPTURE_ROOT", ROOT)
    kwargs = dict(output="screen", additional_env={"PYTHONPATH": os.environ.get("PYTHONPATH", ""), "CAPTURE_ROOT": ROOT})
    return LaunchDescription(
        [
            env_py,
            env_root,
            Node(package="capture_bringup", executable="sil_node", name="capture_sil", output="screen"),
            Node(package="capture_bringup", executable="perception_node", name="capture_perception", output="screen"),
            Node(package="capture_bringup", executable="guidance_node", name="capture_guidance", output="screen"),
        ]
    )
