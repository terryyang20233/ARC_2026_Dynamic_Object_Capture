"""Pinhole camera that paints a tennis ball into a BGR image.

The renderer is intentionally simple: sky + ground + a lit yellow sphere with
a white seam. That is enough for colour-threshold detection and, at moderate
range, for the trained YOLO11s weights.
"""

from dataclasses import dataclass

import numpy as np

from capture_core.geometry import yaw_rotation_z
from capture_core.physics import TENNIS_RADIUS


@dataclass
class Camera:
    width: int = 640
    height: int = 480
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0

    @property
    def K(self):
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


def world_to_camera(p_world, drone_pos, yaw):
    """ENU world point -> OpenCV camera frame (X right, Y down, Z forward)."""
    R = yaw_rotation_z(yaw)
    body = R.T @ (np.asarray(p_world, dtype=float) - np.asarray(drone_pos, dtype=float))
    return np.array([-body[1], -body[2], body[0]], dtype=float)


def project_ball(p_world, drone_pos, yaw, cam: Camera, radius=TENNIS_RADIUS):
    p_cam = world_to_camera(p_world, drone_pos, yaw)
    z = float(p_cam[2])
    if z < 0.15:
        return None
    u = cam.fx * p_cam[0] / z + cam.cx
    v = cam.fy * p_cam[1] / z + cam.cy
    r_px = cam.fx * radius / z
    if r_px < 1.5:
        return None
    if u < -r_px or u > cam.width + r_px or v < -r_px or v > cam.height + r_px:
        return None
    bbox = [u - r_px, v - r_px, u + r_px, v + r_px]
    return {"u": u, "v": v, "r": r_px, "z": z, "p_cam": p_cam, "bbox": bbox}


def _sky_ground(cam: Camera):
    img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    horizon = int(cam.height * 0.58)
    for row in range(horizon):
        t = row / max(horizon, 1)
        img[row, :] = (int(210 - 40 * t), int(170 - 20 * t), int(90 + 40 * t))
    ground = img[horizon:]
    gh, gw = ground.shape[:2]
    yy, xx = np.mgrid[0:gh, 0:gw]
    checker = ((xx // 32) + (yy // 32)) % 2
    ground[:] = np.where(checker[..., None] == 0, (70, 140, 70), (50, 110, 55))
    return img


def _draw_ball(img, proj, cam: Camera):
    h, w = img.shape[:2]
    u, v, r = proj["u"], proj["v"], proj["r"]
    r_i = max(int(np.ceil(r)) + 1, 3)
    y0 = max(int(v) - r_i, 0)
    y1 = min(int(v) + r_i + 1, h)
    x0 = max(int(u) - r_i, 0)
    x1 = min(int(u) + r_i + 1, w)
    if x1 <= x0 or y1 <= y0:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = (xx - u) / r
    dy = (yy - v) / r
    rr = dx * dx + dy * dy
    mask = rr <= 1.0
    if not np.any(mask):
        return img
    nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
    # Camera-frame lighting from upper-left.
    light = dx * -0.35 + dy * -0.55 + nz * 0.75
    light = np.clip(light, 0.15, 1.0)
    seam = np.abs(np.sin(2.2 * np.arctan2(dy, dx) + 0.9 * nz)) < 0.16
    optic = np.array([40.0, 240.0, 215.0])  # BGR optic yellow
    white = np.array([245.0, 245.0, 245.0])
    shade = optic[None, None, :] * light[..., None]
    shade[seam] = white * light[seam][..., None]
    patch = img[y0:y1, x0:x1].astype(np.float32)
    patch[mask] = shade[mask]
    img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return img


def render_frame(ball_pos, drone_pos, yaw, cam: Camera = None):
    cam = cam or Camera()
    img = _sky_ground(cam)
    proj = project_ball(ball_pos, drone_pos, yaw, cam)
    if proj is not None:
        img = _draw_ball(img, proj, cam)
    return img, proj
