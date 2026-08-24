"""Frame transforms used by the capture stack.

World frame is ENU (East, North, Up), matching the existing Kalman tests.
OpenCV camera frame: +X right, +Y down, +Z forward.
Body frame: +X forward, +Y left, +Z up.
Yaw is ROS/ENU: 0 faces East, 90 deg faces North.
"""

import numpy as np


def yaw_rotation_z(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def camera_to_body(X_cam: float, Y_cam: float, Z_cam: float) -> np.ndarray:
    """OpenCV camera frame -> drone body (forward, left, up)."""
    return np.array([Z_cam, -X_cam, -Y_cam], dtype=float)


def camera_to_world(X_cam, Y_cam, Z_cam, drone_pos, drone_yaw):
    """Camera-frame point to ENU world, assuming a forward-facing camera."""
    body = camera_to_body(X_cam, Y_cam, Z_cam).reshape(3, 1)
    R = yaw_rotation_z(np.radians(drone_yaw) if abs(drone_yaw) > 2.0 * np.pi else drone_yaw)
    # Accept both degrees (legacy KF scripts used degrees) and radians.
    # Heuristic: |yaw| > 2pi is treated as degrees.
    if abs(drone_yaw) > 2.0 * np.pi:
        R = yaw_rotation_z(np.radians(drone_yaw))
    else:
        R = yaw_rotation_z(float(drone_yaw))
    t = np.asarray(drone_pos, dtype=float).reshape(3, 1)
    return (R @ body + t).flatten()


def camera_to_world_deg(X_cam, Y_cam, Z_cam, drone_pos, drone_yaw_deg):
    """Same as the original KF_Test camera_to_world (yaw in degrees)."""
    body = camera_to_body(X_cam, Y_cam, Z_cam).reshape(3, 1)
    R = yaw_rotation_z(np.radians(drone_yaw_deg))
    t = np.asarray(drone_pos, dtype=float).reshape(3, 1)
    return (R @ body + t).flatten()


def wrap_pi(angle: float) -> float:
    a = (float(angle) + np.pi) % (2.0 * np.pi) - np.pi
    return float(a)


wrap_pi = wrap_pi


def yaw_from_xy(vec_xy) -> float:
    """Yaw that points the body +X axis along the given ENU vector."""
    x, y = float(vec_xy[0]), float(vec_xy[1])
    if x * x + y * y < 1e-12:
        return 0.0
    return float(np.arctan2(y, x))


yaw_from_xy = yaw_from_xy
wrap_pi = wrap_pi
yaw_rotation_z = yaw_rotation_z
