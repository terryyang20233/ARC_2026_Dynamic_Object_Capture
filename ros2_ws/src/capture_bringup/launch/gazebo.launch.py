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

WORLD = os.path.join(ROOT, "gazebo", "worlds", "dynamic_capture.world")
MODELS = os.path.join(ROOT, "gazebo", "models")
SAVE_FRAME = os.path.join(ROOT, "runs", "gazebo_camera.png")


def generate_launch_description():
    py = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    gz_path = MODELS + os.pathsep + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    extra = {
        "PYTHONPATH": py,
        "CAPTURE_ROOT": ROOT,
        "PYTHONNOUSERSITE": "1",
        "GZ_SIM_RESOURCE_PATH": gz_path,
        "LIBGL_ALWAYS_SOFTWARE": os.environ.get("LIBGL_ALWAYS_SOFTWARE", "1"),
        "GALLIUM_DRIVER": os.environ.get("GALLIUM_DRIVER", "llvmpipe"),
    }
    return LaunchDescription(
        [
            SetEnvironmentVariable("PYTHONPATH", py),
            SetEnvironmentVariable("CAPTURE_ROOT", ROOT),
            SetEnvironmentVariable("PYTHONNOUSERSITE", "1"),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", gz_path),
            Node(
                package="capture_bringup",
                executable="gz_sil_node",
                name="capture_sil",
                output="screen",
                additional_env=extra,
                parameters=[
                    {
                        "world": WORLD,
                        "save_frame": SAVE_FRAME,
                        "catch_radius": 0.22,
                    }
                ],
            ),
            Node(
                package="capture_bringup",
                executable="perception_node",
                name="capture_perception",
                output="screen",
                additional_env=extra,
                parameters=[{"use_truth_seed": True}],
            ),
            Node(
                package="capture_bringup",
                executable="guidance_node",
                name="capture_guidance",
                output="screen",
                additional_env=extra,
            ),
        ]
    )
