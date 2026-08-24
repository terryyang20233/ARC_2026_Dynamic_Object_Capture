from setuptools import setup
from glob import glob
import os

package_name = "capture_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ARC 2026",
    maintainer_email="arc@local",
    description="ROS 2 Humble nodes wrapping capture perception and intercept guidance.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = capture_bringup.perception_node:main",
            "guidance_node = capture_bringup.guidance_node:main",
            "sil_node = capture_bringup.sil_node:main",
        ],
    },
)
