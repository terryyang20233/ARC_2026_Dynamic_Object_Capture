"""Constant-acceleration Kalman filter for a ballistic tennis ball in ENU."""

import numpy as np


class KalmanFilter3D:
    """State: [x, y, z, vx, vy, vz]. Gravity is applied as a control input on Z."""

    def __init__(self, gravity=-9.81, meas_var=0.1, process_var=0.05, **kwargs):
        meas_var = kwargs.get("meas_var", meas_var)
        process_var = kwargs.get("process_var", process_var)
        self.g = float(gravity)
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 500.0
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.R = np.eye(3) * float(meas_var)
        self.Q = np.eye(6) * float(process_var)
        self.is_initialized = False

    @property
    def position(self):
        return self.x[:3].flatten() if self.is_initialized else None

    @property
    def velocity(self):
        return self.x[3:].flatten() if self.is_initialized else None

    def predict(self, dt):
        if not self.is_initialized:
            return None
        dt = float(max(dt, 1e-4))
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        B = np.zeros((6, 1))
        B[2, 0] = 0.5 * dt * dt
        B[5, 0] = dt
        u = np.array([[self.g]])
        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:3].flatten()

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape((3, 1))
        if not self.is_initialized:
            self.x[:3] = z
            self.x[3:] = 0.0
            self.is_initialized = True
            return self.x[:3].flatten()
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        return self.x[:3].flatten()

    def get_future_trajectory(self, future_time=1.5, step=0.05, z_floor=0.0):
        if not self.is_initialized:
            return [], [], []
        px, py, pz = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        vx, vy, vz = self.x[3, 0], self.x[4, 0], self.x[5, 0]
        xs, ys, zs = [], [], []
        t = 0.0
        while t <= future_time + 1e-9:
            x = px + vx * t
            y = py + vy * t
            z = pz + vz * t + 0.5 * self.g * t * t
            if z < z_floor:
                break
            xs.append(x)
            ys.append(y)
            zs.append(z)
            t += step
        return xs, ys, zs
