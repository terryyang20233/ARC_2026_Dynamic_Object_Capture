"""Capture-point planner: when and where the drone should meet the ball.

The planner samples the ballistic trajectory from the Kalman state, checks
which future points the drone can actually reach (speed + accel limits), and
returns the earliest cheap feasible catch pose. The net is a point a fixed
offset in front of the drone, so the published setpoint is the drone CoM.
"""

from dataclasses import dataclass, field

import numpy as np

from .geometry import yaw_from_xy
from .physics import WorldLimits


@dataclass
class DroneLimits:
    v_max: float = 8.0
    a_max: float = 6.0
    catch_offset: float = 0.35
    catch_z_bias: float = -0.04
    catch_radius: float = 0.18
    t_min: float = 0.18
    t_horizon: float = 2.2
    dt: float = 0.04
    reaction: float = 0.05
    z_floor: float = 0.45
    z_ceil: float = 6.5


@dataclass
class InterceptPlan:
    feasible: bool
    t_go: float
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    drone_target: np.ndarray
    drone_yaw: float
    drone_vel_ff: np.ndarray
    cost: float
    reason: str
    samples_checked: int = 0
    extra: dict = field(default_factory=dict)


class InterceptPlanner:
    def __init__(self, limits: DroneLimits = None, world: WorldLimits = None):
        self.limits = limits or DroneLimits()
        self.world = world or WorldLimits()

    def plan(self, ball_pos, ball_vel, drone_pos, drone_vel, gravity=-9.81) -> InterceptPlan:
        p0 = np.asarray(ball_pos, dtype=float).reshape(3)
        v0 = np.asarray(ball_vel, dtype=float).reshape(3)
        pd = np.asarray(drone_pos, dtype=float).reshape(3)
        vd = np.asarray(drone_vel, dtype=float).reshape(3)
        lim = self.limits

        best_feasible = None
        best_effort = None
        n = 0
        t = lim.t_min
        while t <= lim.t_horizon + 1e-9:
            n += 1
            p_b, v_b = _ballistic(p0, v0, t, gravity)
            if p_b[2] < lim.z_floor:
                break
            if not _in_world(p_b, self.world, lim.z_floor, lim.z_ceil):
                t += lim.dt
                continue

            p_d, yaw, forward = capture_pose(p_b, v_b, lim.catch_offset, lim.catch_z_bias)
            if p_d[2] < 0.2:
                t += lim.dt
                continue

            t_pos = time_to_reach(pd, vd, p_d, lim.v_max, lim.a_max)
            dv = v_b - vd
            t_match = float(np.linalg.norm(dv)) / max(lim.a_max, 0.2)
            t_need = max(t_pos, 0.35 * t_match) + lim.reaction

            v_ff = 0.45 * v_b
            speed = np.linalg.norm(v_ff)
            if speed > lim.v_max:
                v_ff = v_ff * (lim.v_max / speed)

            closing = float(np.linalg.norm(v_b - v_ff))
            cost = (
                1.0 * t
                + 0.08 * np.linalg.norm(p_d - pd)
                + 0.04 * closing
                + 0.05 * abs(p_d[2] - 1.6)
            )
            candidate = InterceptPlan(
                feasible=t_need <= t,
                t_go=float(t),
                ball_pos=p_b,
                ball_vel=v_b,
                drone_target=p_d,
                drone_yaw=float(yaw),
                drone_vel_ff=v_ff,
                cost=float(cost),
                reason="ok" if t_need <= t else "too_far",
                samples_checked=n,
                extra={"t_need": float(t_need), "forward": forward},
            )
            if candidate.feasible:
                if best_feasible is None or candidate.cost < best_feasible.cost:
                    best_feasible = candidate
            else:
                slack = t_need - t
                candidate.extra["slack"] = float(slack)
                if best_effort is None or slack < best_effort.extra.get("slack", 1e9):
                    best_effort = candidate
            t += lim.dt

        if best_feasible is not None:
            best_feasible.samples_checked = n
            return best_feasible
        if best_effort is not None:
            best_effort.reason = "chase_best_effort"
            best_effort.samples_checked = n
            return best_effort
        return InterceptPlan(
            feasible=False,
            t_go=0.0,
            ball_pos=p0,
            ball_vel=v0,
            drone_target=pd,
            drone_yaw=0.0,
            drone_vel_ff=np.zeros(3),
            cost=1e9,
            reason="no_sample",
            samples_checked=n,
        )


def _ballistic(p0, v0, t, g=-9.81):
    p = p0 + v0 * t + np.array([0.0, 0.0, 0.5 * g * t * t])
    v = v0 + np.array([0.0, 0.0, g * t])
    return p, v


def time_to_reach(p0, v0, p_goal, v_max, a_max):
    """Min time for a speed-limited double integrator to fly through p_goal."""
    delta = np.asarray(p_goal, dtype=float) - np.asarray(p0, dtype=float)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        return 0.0
    direction = delta / dist
    v_along = float(np.dot(np.asarray(v0, dtype=float), direction))
    v_along = max(0.0, v_along)
    v_max = max(float(v_max), 0.2)
    a_max = max(float(a_max), 0.2)

    if v_along >= v_max - 1e-6:
        return dist / v_along

    t_acc = (v_max - v_along) / a_max
    d_acc = v_along * t_acc + 0.5 * a_max * t_acc * t_acc
    if d_acc >= dist:
        disc = v_along * v_along + 2.0 * a_max * dist
        return (-v_along + np.sqrt(max(disc, 0.0))) / a_max
    return t_acc + (dist - d_acc) / v_max


def capture_pose(ball_pos, ball_vel, catch_offset, z_bias):
    """Drone CoM pose such that the net meets the ball, facing incoming flight."""
    v_h = np.array([ball_vel[0], ball_vel[1], 0.0], dtype=float)
    speed_h = np.linalg.norm(v_h)
    if speed_h > 0.25:
        forward = -v_h / speed_h
    else:
        forward = np.array([1.0, 0.0, 0.0])
    yaw = yaw_from_xy(forward)
    drone_target = np.asarray(ball_pos, dtype=float).copy()
    drone_target -= catch_offset * np.array([forward[0], forward[1], 0.0])
    drone_target[2] += z_bias
    return drone_target, yaw, forward


def _in_world(p, limits: WorldLimits, z_floor, z_ceil):
    return (
        limits.x_min <= p[0] <= limits.x_max
        and limits.y_min <= p[1] <= limits.y_max
        and z_floor <= p[2] <= z_ceil
    )


time_to_reach = time_to_reach
DroneLimits = DroneLimits
