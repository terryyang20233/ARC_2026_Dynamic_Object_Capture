"""Ballistic tennis ball and a velocity-limited quadrotor model."""

from dataclasses import dataclass

import numpy as np

from .geometry import wrap_pi


G = np.array([0.0, 0.0, -9.81])
TENNIS_MASS = 0.057
TENNIS_RADIUS = 0.0335
TENNIS_CD = 0.55
AIR_RHO = 1.225


@dataclass
class WorldLimits:
    x_min: float = -20.0
    x_max: float = 20.0
    y_min: float = -5.0
    y_max: float = 40.0
    z_min: float = 0.35
    z_max: float = 8.0


class TennisBall:
    def __init__(self, position, velocity, drag=True):
        self.p = np.asarray(position, dtype=float).reshape(3).copy()
        self.v = np.asarray(velocity, dtype=float).reshape(3).copy()
        self.drag = bool(drag)
        self.radius = TENNIS_RADIUS
        self.on_ground = False

    def acceleration(self):
        a = G.copy()
        if self.drag:
            speed = np.linalg.norm(self.v)
            if speed > 1e-6:
                area = np.pi * self.radius ** 2
                fd = 0.5 * AIR_RHO * TENNIS_CD * area * speed * speed
                a = a - (fd / TENNIS_MASS) * (self.v / speed)
        return a

    def step(self, dt, z_floor=0.0, restitution=0.55):
        if self.on_ground:
            self.v[:] = 0.0
            self.p[2] = z_floor + self.radius
            return self.p.copy(), self.v.copy()
        a = self.acceleration()
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        if self.p[2] <= z_floor + self.radius:
            self.p[2] = z_floor + self.radius
            if self.v[2] < 0.0:
                self.v[2] = -self.v[2] * restitution
                self.v[0] *= 0.7
                self.v[1] *= 0.7
            if abs(self.v[2]) < 0.4:
                self.on_ground = True
                self.v[:] = 0.0
        return self.p.copy(), self.v.copy()


class DroneBody:
    """Point-mass quadrotor with speed, accel, and yaw-rate limits."""

    def __init__(
        self,
        position,
        velocity=None,
        yaw=0.0,
        v_max=8.0,
        a_max=6.0,
        yaw_rate_max=2.5,
    ):
        self.p = np.asarray(position, dtype=float).reshape(3).copy()
        self.v = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).reshape(3).copy()
        self.yaw = float(yaw)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.yaw_rate_max = float(yaw_rate_max)

    def step(self, dt, p_cmd, v_ff=None, yaw_cmd=None, kp=2.2, kv=2.8):
        v_ff = np.zeros(3) if v_ff is None else np.asarray(v_ff, dtype=float).reshape(3)
        p_cmd = np.asarray(p_cmd, dtype=float).reshape(3)
        v_des = kp * (p_cmd - self.p) + v_ff
        speed = np.linalg.norm(v_des)
        if speed > self.v_max:
            v_des *= self.v_max / speed
        a = kv * (v_des - self.v)
        an = np.linalg.norm(a)
        if an > self.a_max:
            a *= self.a_max / an
        self.v = self.v + a * dt
        vn = np.linalg.norm(self.v)
        if vn > self.v_max:
            self.v *= self.v_max / vn
        self.p = self.p + self.v * dt
        self.p[2] = max(self.p[2], 0.15)
        if yaw_cmd is not None:
            err = wrap_pi(float(yaw_cmd) - self.yaw)
            max_step = self.yaw_rate_max * dt
            self.yaw += float(np.clip(err, -max_step, max_step))
        return self.p.copy(), self.v.copy(), self.yaw


WorldLimits = WorldLimits
