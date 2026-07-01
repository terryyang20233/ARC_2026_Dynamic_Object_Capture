import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

import matplotlib
# Force Matplotlib to use a native window, bypassing PyCharm's SciView
matplotlib.use('macosx')
import matplotlib.pyplot as plt


# ==========================================
# 1. KALMAN FILTER FOR TRAJECTORY TRACKING
# ==========================================
class KalmanFilter3D:
    def __init__(self):
        # State vector: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 500.0

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.R = np.eye(3) * 0.1
        self.Q = np.eye(6) * 0.05

        self.is_initialized = False

    def predict(self, dt):
        if not self.is_initialized:
            return None

        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        B = np.zeros((6, 1))
        B[2, 0] = 0.5 * (dt ** 2)
        B[5, 0] = dt

        u = np.array([[-9.81]])  # Gravity

        self.x = np.dot(F, self.x) + np.dot(B, u)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        return self.x[:3].flatten()

    def update(self, measurement):
        z = np.array(measurement).reshape((3, 1))

        if not self.is_initialized:
            self.x[:3] = z
            self.is_initialized = True
            return self.x[:3].flatten()

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x[:3].flatten()

    def get_future_trajectory(self, future_time=1.5, step=0.1):
        """
        Simulates the future path of the ball based on current velocity and gravity.
        Returns lists of predicted X, Y, and Z coordinates.
        """
        if not self.is_initialized:
            return [], [], []

        curr_x, curr_y, curr_z = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        vx, vy, vz = self.x[3, 0], self.x[4, 0], self.x[5, 0]

        pred_x, pred_y, pred_z = [], [], []

        # Project forward in time
        for t in np.arange(0, future_time, step):
            next_x = curr_x + (vx * t)
            next_y = curr_y + (vy * t)
            next_z = curr_z + (vz * t) + (0.5 * -9.81 * (t ** 2))

            # Stop predicting if it hits the ground (Z < 0)
            if next_z < 0:
                break

            pred_x.append(next_x)
            pred_y.append(next_y)
            pred_z.append(next_z)

        return pred_x, pred_y, pred_z


# ==========================================
# 2. COORDINATE TRANSFORMATIONS & PERCEPTION
# ==========================================
def camera_to_world(X_cam, Y_cam, Z_cam, drone_pos, drone_yaw):
    x_body = Z_cam
    y_body = -X_cam
    z_body = -Y_cam
    body_point = np.array([[x_body], [y_body], [z_body]])

    yaw_rad = np.radians(drone_yaw)
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    drone_translation = np.array([[drone_pos[0]], [drone_pos[1]], [drone_pos[2]]])
    world_point = np.dot(R_z, body_point) + drone_translation

    return world_point.flatten()


def estimate_3d_position(bbox, camera_matrix, real_diameter=0.067):
    x1, y1, x2, y2 = bbox
    pixel_diameter = max(x2 - x1, y2 - y1)
    if pixel_diameter <= 0: return None, None, None

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    Z = (real_diameter * fx) / pixel_diameter
    X = ((x1 + (x2 - x1) / 2.0) - cx) * Z / fx
    Y = ((y1 + (y2 - y1) / 2.0) - cy) * Z / fy

    return X, Y, Z


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Initialize YOLO and Camera
    model = YOLO(
        "/Users/terryyang/Documents/GitHub/ARC_2026_Dynamic_Object_Capture/Object_Detection_Test_1/train3/weights/best.pt")

    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if not ret: exit()

    frame_height, frame_width = frame.shape[:2]
    FOCAL_LENGTH = 1100.0
    camera_matrix = np.array([
        [FOCAL_LENGTH, 0.0, frame_width / 2.0],
        [0.0, FOCAL_LENGTH, frame_height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    kf = KalmanFilter3D()

    # Dummy Drone Telemetry
    drone_world_x = 0.0
    drone_world_y = 0.0
    drone_world_z = 2.0  # Drone hovering 2 meters high
    drone_yaw_deg = 90.0  # Facing North (+Y in ENU)

    # 2. Setup Matplotlib 3D Environment
    plt.ion()  # Interactive mode on
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Store history for drawing trails (max 30 points so it doesn't lag)
    history_raw = deque(maxlen=30)
    history_flt = deque(maxlen=30)

    PLOT_EVERY_N_FRAMES = 3  # Update graph every 3 frames to keep CV running smoothly
    frame_count = 0
    last_time = time.time()

    print("Starting video and 3D simulation... Press 'q' in CV window to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        predicted_world_pos = kf.predict(dt)
        results = model.predict(source=frame, conf=0.5, stream=True, verbose=False)

        ball_detected = False
        raw_world_pos = None
        filtered_world_pos = None

        # 3. Perception and Tracking
        for result in results:
            for box in result.boxes:
                ball_detected = True
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                X_cam, Y_cam, Z_cam = estimate_3d_position([x1, y1, x2, y2], camera_matrix)

                if Z_cam is not None:
                    drone_pos = [drone_world_x, drone_world_y, drone_world_z]
                    raw_world_x, raw_world_y, raw_world_z = camera_to_world(
                        X_cam, Y_cam, Z_cam, drone_pos, drone_yaw_deg)
                    raw_world_pos = (raw_world_x, raw_world_y, raw_world_z)
                    history_raw.append(raw_world_pos)

                    filtered_world_pos = kf.update([raw_world_x, raw_world_y, raw_world_z])
                    history_flt.append(tuple(filtered_world_pos))

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.imshow("UAV Dynamic Capture - Camera View", frame)

        # 4. 3D Simulation Plotting
        if frame_count % PLOT_EVERY_N_FRAMES == 0:
            ax.cla()  # Clear previous plot

            # Plot Drone Position
            ax.scatter(drone_world_x, drone_world_y, drone_world_z, c='black', marker='^', s=100, label='Drone')

            # Plot Historic Raw Detections (Red X)
            if len(history_raw) > 0:
                raw_xs, raw_ys, raw_zs = zip(*history_raw)
                ax.scatter(raw_xs, raw_ys, raw_zs, c='r', marker='x', alpha=0.5, label='Raw YOLO')

            # Plot Filtered Trajectory Trail (Green Line)
            if len(history_flt) > 0:
                flt_xs, flt_ys, flt_zs = zip(*history_flt)
                ax.plot(flt_xs, flt_ys, flt_zs, c='g', linewidth=2, label='KF Path')
                ax.scatter(flt_xs[-1], flt_ys[-1], flt_zs[-1], c='g', marker='o', s=50, label='Current Pos')

            # Plot Future Prediction Parabola (Blue Dashed Line)
            if kf.is_initialized:
                pred_x, pred_y, pred_z = kf.get_future_trajectory(future_time=1.5, step=0.05)
                if len(pred_x) > 0:
                    ax.plot(pred_x, pred_y, pred_z, c='b', linestyle='--', linewidth=2, label='Predicted Arc')

            # Standardize axes to represent a 10x10x5 meter room around the drone
            ax.set_xlim([drone_world_x - 5, drone_world_x + 5])  # Left/Right (East-West)
            ax.set_ylim([drone_world_y, drone_world_y + 10])  # Forward (North)
            ax.set_zlim([0, 5])  # Floor to Ceiling

            ax.set_xlabel('X (East/West) [m]')
            ax.set_ylabel('Y (North/South) [m]')
            ax.set_zlabel('Z (Altitude) [m]')
            ax.set_title('Live 3D Trajectory & Prediction')
            ax.legend(loc='upper right')

            plt.pause(0.001)  # Forces Matplotlib to refresh without blocking OpenCV

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()