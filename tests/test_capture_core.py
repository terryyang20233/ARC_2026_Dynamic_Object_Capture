"""Unit tests for perception, Kalman, intercept, and closed-loop SIL."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capture_core.geometry import camera_to_world, yaw_from_xy
from capture_core.intercept import DroneLimits, InterceptPlanner, time_to_reach
from capture_core.kalman import KalmanFilter3D
from capture_core.perception import estimate_3d_pinhole
from capture_core.physics import DroneBody, TennisBall
from capture_sim.renderer import Camera, project_ball, render_frame
from capture_sim.sil import SilConfig, run_sil


class TestPerception(unittest.TestCase):
    def test_pinhole_recovers_depth(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        Z = 4.0
        d_px = 0.067 * 500.0 / Z
        bbox = [320 - d_px / 2, 240 - d_px / 2, 320 + d_px / 2, 240 + d_px / 2]
        X, Y, z = estimate_3d_pinhole(bbox, K)
        self.assertAlmostEqual(z, Z, places=5)
        self.assertAlmostEqual(X, 0.0, places=5)
        self.assertAlmostEqual(Y, 0.0, places=5)

    def test_camera_to_world_forward(self):
        # Camera +Z (forward) with yaw=pi/2 (facing North) -> +Y world.
        p = camera_to_world(0.0, 0.0, 3.0, [0.0, 0.0, 1.5], np.pi / 2.0)
        self.assertAlmostEqual(p[0], 0.0, places=5)
        self.assertAlmostEqual(p[1], 3.0, places=5)
        self.assertAlmostEqual(p[2], 1.5, places=5)


class TestKalman(unittest.TestCase):
    def test_tracks_ballistic_arc(self):
        kf = KalmanFilter3D(meas_var=0.01, process_var=0.02)
        p = np.array([0.0, 8.0, 2.0])
        v = np.array([0.0, -6.0, 4.0])
        g = np.array([0.0, 0.0, -9.81])
        dt = 0.05
        for _ in range(25):
            p = p + v * dt + 0.5 * g * dt * dt
            v = v + g * dt
            meas = p + np.array([0.01, -0.01, 0.0])
            if kf.is_initialized:
                kf.predict(dt)
            kf.update(meas)
        self.assertTrue(kf.is_initialized)
        err = np.linalg.norm(kf.position - p)
        self.assertLess(err, 0.25)


class TestIntercept(unittest.TestCase):
    def test_time_to_reach_at_rest(self):
        t = time_to_reach([0, 0, 1], [0, 0, 0], [8, 0, 1], v_max=8.0, a_max=6.0)
        self.assertGreater(t, 1.0)
        self.assertLess(t, 2.2)

    def test_finds_feasible_catch(self):
        planner = InterceptPlanner(DroneLimits(v_max=10.0, a_max=8.0, catch_offset=0.3))
        # Ball incoming from +Y toward origin; drone hovering at origin.
        plan = planner.plan(
            ball_pos=[0.2, 8.0, 2.2],
            ball_vel=[0.0, -7.0, 2.5],
            drone_pos=[0.0, 0.0, 1.6],
            drone_vel=[0.0, 0.0, 0.0],
        )
        self.assertTrue(plan.feasible, plan.reason)
        self.assertGreater(plan.t_go, 0.2)
        self.assertLess(plan.t_go, 2.0)
        self.assertGreater(plan.drone_target[2], 0.4)

    def test_yaw_faces_incoming(self):
        yaw = yaw_from_xy([0.0, 1.0])
        self.assertAlmostEqual(yaw, np.pi / 2.0, places=5)


class TestRendererAndSil(unittest.TestCase):
    def test_ball_projects_in_front(self):
        cam = Camera()
        drone = np.array([0.0, 0.0, 1.5])
        yaw = np.pi / 2.0
        ball = np.array([0.0, 6.0, 1.5])
        proj = project_ball(ball, drone, yaw, cam)
        self.assertIsNotNone(proj)
        self.assertGreater(proj["r"], 2.0)
        self.assertTrue(0 < proj["u"] < cam.width)
        self.assertTrue(0 < proj["v"] < cam.height)
        img, proj2 = render_frame(ball, drone, yaw, cam)
        self.assertEqual(img.shape, (cam.height, cam.width, 3))
        self.assertIsNotNone(proj2)

    def test_closed_loop_catch(self):
        out = run_sil(SilConfig(duration=2.8, seed=0, catch_radius=0.28, meas_noise=0.01))
        self.assertTrue(
            out["caught"] or out["min_net_dist"] < 0.35,
            f"missed catch: min_net={out['min_net_dist']:.3f} m",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
