from pathlib import Path
import cv2
import numpy as np
import time
from ultralytics import YOLO


# ==========================================
# 1. KALMAN FILTER FOR TRAJECTORY TRACKING
# ==========================================
class KalmanFilter3D:
    def __init__(self):
        # State vector: [x, y, z, vx, vy, vz] (World Frame)
        self.x = np.zeros((6, 1))

        # Uncertainty covariance matrix (Starts with high uncertainty)
        self.P = np.eye(6) * 500.0

        # Measurement matrix (We only measure position [x, y, z], not velocity)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise covariance (How much we trust YOLO/Camera depth)
        self.R = np.eye(3) * 0.1  # 10cm variance

        # Process noise covariance (Accounts for wind/aerodynamic drag not in the math)
        self.Q = np.eye(6) * 0.05

        self.is_initialized = False
        self.last_time = time.time()

    def predict(self, dt):
        if not self.is_initialized:
            return None

        # State Transition Matrix (Kinematics: x = x + vx*dt)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Control Input Matrix (Effect of gravity on position and velocity)
        # Gravity affects Z-axis (Index 2 for position, Index 5 for velocity)
        B = np.zeros((6, 1))
        B[2, 0] = 0.5 * (dt ** 2)
        B[5, 0] = dt

        # Control vector: Gravity is -9.81 m/s^2 (Downwards in ENU frame)
        u = np.array([[-9.81]])

        # Predict State
        self.x = np.dot(F, self.x) + np.dot(B, u)

        # Predict Covariance
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        return self.x[:3].flatten()  # Return predicted x, y, z

    def update(self, measurement):
        z = np.array(measurement).reshape((3, 1))

        if not self.is_initialized:
            # Initialize position to first measurement, velocities to 0
            self.x[:3] = z
            self.is_initialized = True
            return self.x[:3].flatten()

        # Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update State with measurement residual
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        # Update Covariance
        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x[:3].flatten()


# ==========================================
# 2. COORDINATE TRANSFORMATIONS
# ==========================================
def camera_to_world(X_cam, Y_cam, Z_cam, drone_pos, drone_yaw):
    """
    Transforms Camera coordinates to World Coordinates (ENU: East, North, Up).
    Assumes camera is mounted perfectly facing FORWARD on the drone.
    """
    # 1. Camera Frame (OpenCV) to Drone Body Frame (Forward-Left-Up)
    # OpenCV: +Z is forward, +X is right, +Y is down
    # Drone:  +X is forward, +Y is left, +Z is up
    x_body = Z_cam
    y_body = -X_cam
    z_body = -Y_cam

    body_point = np.array([[x_body], [y_body], [z_body]])

    # 2. Drone Body Frame to World Frame (ENU)
    # Rotation matrix around Z-axis (Yaw)
    yaw_rad = np.radians(drone_yaw)
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # Apply rotation and add drone translation (position in world)
    drone_translation = np.array([[drone_pos[0]], [drone_pos[1]], [drone_pos[2]]])
    world_point = np.dot(R_z, body_point) + drone_translation

    return world_point.flatten()


# ==========================================
# 3. PERCEPTION PIPELINE
# ==========================================
def estimate_3d_position(bbox, camera_matrix, real_diameter=0.067):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pixel_diameter = max(w, h)

    if pixel_diameter <= 0:
        return None, None, None

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    Z = (real_diameter * fx) / pixel_diameter
    u = x1 + (w / 2.0)
    v = y1 + (h / 2.0)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return X, Y, Z


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # Load YOLO
    weights = (
        Path(__file__).resolve().parent.parent
        / "Object_Detection_Test_1"
        / "train3"
        / "weights"
        / "best.pt"
    )
    model = YOLO(str(weights))

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

    # Initialize Kalman Filter
    kf = KalmanFilter3D()
    last_time = time.time()

    # ---------------------------------------------------------
    # DUMMY DRONE TELEMETRY (Replace with real MAVROS/PX4 data later)
    # Let's pretend the drone is hovering 5 meters high, exactly
    # at the origin of the world, facing "North" (Yaw = 90 deg in ENU)
    # ---------------------------------------------------------
    drone_world_x = 0.0
    drone_world_y = 0.0
    drone_world_z = 5.0
    drone_yaw_deg = 90.0

    print("Starting video stream... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Calculate dt for Kalman Filter kinematics
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # 1. PREDICT: Always predict the ball's next position, even if not detected
        predicted_world_pos = kf.predict(dt)

        # Run YOLO Inference
        results = model.predict(source=frame, conf=0.5, stream=True, verbose=False)

        ball_detected_this_frame = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                ball_detected_this_frame = True
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_bbox = [x1, y1, x2, y2]

                # 2. PERCEPTION: Get camera-relative coordinates
                X_cam, Y_cam, Z_cam = estimate_3d_position(raw_bbox, camera_matrix)

                if Z_cam is not None:
                    # 3. TRANSFORMATION: Convert to Absolute World Coordinates
                    drone_pos = [drone_world_x, drone_world_y, drone_world_z]
                    raw_world_x, raw_world_y, raw_world_z = camera_to_world(
                        X_cam, Y_cam, Z_cam, drone_pos, drone_yaw_deg)

                    # 4. UPDATE: Correct the Kalman Filter with the new measurement
                    measurement = [raw_world_x, raw_world_y, raw_world_z]
                    filtered_world_pos = kf.update(measurement)

                    # Visualization
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Display Raw vs Filtered World Coordinates
                    label_raw = f"Raw W-Frame: X:{raw_world_x:.2f} Y:{raw_world_y:.2f} Z:{raw_world_z:.2f}"
                    label_flt = f"KF  W-Frame: X:{filtered_world_pos[0]:.2f} Y:{filtered_world_pos[1]:.2f} Z:{filtered_world_pos[2]:.2f}"

                    cv2.putText(frame, label_raw, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, label_flt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If YOLO missed the ball, we can still show where the KF *thinks* the ball is!
        if not ball_detected_this_frame and kf.is_initialized:
            label_pred = f"KF Predict: X:{predicted_world_pos[0]:.2f} Y:{predicted_world_pos[1]:.2f} Z:{predicted_world_pos[2]:.2f}"
            cv2.putText(frame, "BALL LOST - TRACKING BLIND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, label_pred, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("UAV Dynamic Capture - World Frame Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()