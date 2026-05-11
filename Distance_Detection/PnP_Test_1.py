import cv2
import numpy as np


def estimate_distance_pnp(bbox, camera_matrix, dist_coeffs):
    x_min, y_min, x_max, y_max = bbox

    # Tennis ball radius = 0.0335 meters
    R = 0.0335

    # 1. 3D Object Points (in meters)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    object_points = np.array([
        [-R, -R, 0.0], [R, -R, 0.0],
        [R, R, 0.0], [-R, R, 0.0]
    ], dtype=np.float32)

    # 2. 2D Image Points (in pixels)
    # Order MUST match the 3D points exactly
    image_points = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max], [x_min, y_max]
    ], dtype=np.float32)

    # 3. Solve PnP (Using ITERATIVE for maximum stability)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        # Calculate straight-line distance
        euclidean_distance = np.linalg.norm(tvec)
        return tvec, euclidean_distance
    else:
        return None, None


# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # YOLO Bounding Box (40x40 pixel box, perfectly centered)
    yolo_bbox = [280.0, 200.0, 320.0, 240.0]

    # Camera Intrinsics Matrix
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0],
                              [0.0, 0.0, 1.0]
                              ], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Run Function
    tvec, distance = estimate_distance_pnp(yolo_bbox, camera_matrix, dist_coeffs)

    if tvec is not None:
        print(f"3D Position (X, Y, Z) in meters:\n{tvec}")
        print(f"True Distance to camera: {distance:.3f} meters")
    else:
        print("PnP failed to find a solution.")