import os

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def _repo_root():
    env = os.environ.get("CAPTURE_ROOT", "")
    if env and os.path.isdir(os.path.join(env, "capture_core")):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.realpath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(here, "capture_core")):
            return here
        here = os.path.dirname(here)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", ".."))


ROOT = _repo_root()


def generate_launch_description():
    py = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    extra = {"PYTHONPATH": py, "CAPTURE_ROOT": ROOT, "PYTHONNOUSERSITE": "1"}
    env_py = SetEnvironmentVariable("PYTHONPATH", py)
    env_root = SetEnvironmentVariable("CAPTURE_ROOT", ROOT)
    return LaunchDescription(
        [
            env_py,
            env_root,
            Node(package="capture_bringup", executable="sil_node", name="capture_sil", output="screen", additional_env=extra),
            Node(package="capture_bringup", executable="perception_node", name="capture_perception", output="screen", additional_env=extra),
            Node(package="capture_bringup", executable="guidance_node", name="capture_guidance", output="screen", additional_env=extra),
        ]
    )
