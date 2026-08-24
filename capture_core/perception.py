"""Tennis-ball range from a bounding box: pinhole (primary) and PnP (backup)."""

import numpy as np


TENNIS_DIAMETER_M = 0.067
TENNIS_RADIUS_M = TENNIS_DIAMETER_M / 2.0


def square_bbox(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    size = max(w, h)
    half = size / 2.0
    return [cx - half, cy - half, cx + half, cy + half]


def estimate_3d_pinhole(bbox, camera_matrix, real_diameter=TENNIS_DIAMETER_M):
    """Pinhole similar-triangles depth. Returns (X, Y, Z) in the camera frame."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pixel_diameter = max(w, h)
    if pixel_diameter <= 1.0:
        return None, None, None

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    Z = (real_diameter * fx) / pixel_diameter
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return float(X), float(Y), float(Z)


def estimate_3d_pnp(bbox, camera_matrix, dist_coeffs=None):
    """solvePnP on a squared bbox. OpenCV is optional so unit tests stay light."""
    try:
        import cv2
    except Exception:
        return None, None

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    x_min, y_min, x_max, y_max = square_bbox(bbox)
    R = TENNIS_RADIUS_M
    object_points = np.array(
        [[-R, -R, 0.0], [R, -R, 0.0], [R, R, 0.0], [-R, R, 0.0]],
        dtype=np.float32,
    )
    image_points = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )
    ok, _rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(camera_matrix, dtype=np.float32),
        np.asarray(dist_coeffs, dtype=np.float32),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None
    return tvec, float(np.linalg.norm(tvec))
