from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def estimate_3d_position(bbox, camera_matrix, real_diameter=0.067):
    """
    Estimates the 3D position of a spherical object using the Pinhole Camera Model.
    Standard tennis ball diameter is ~0.067 meters (6.7 cm).

    Returns: X, Y, Z (in meters) and Euclidean Distance.
    Coordinate System (OpenCV Standard):
    +X = Right
    +Y = Down
    +Z = Forward (Depth)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # For a sphere, the width and height of the bounding box should theoretically be equal.
    # We use the maximum of the two to account for YOLO drawing a slightly squished box.
    pixel_diameter = max(w, h)

    if pixel_diameter <= 0:
        return None, None, None, None

    # Extract camera intrinsic parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 1. Calculate Z (Depth) using Similar Triangles:
    # Z = (Real_Size * Focal_Length) / Pixel_Size
    Z = (real_diameter * fx) / pixel_diameter

    # 2. Find the pixel center of the bounding box
    u = x1 + (w / 2.0)
    v = y1 + (h / 2.0)

    # 3. Calculate X and Y (in meters) based on the center of the image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # 4. Calculate total straight-line distance from the camera lens to the ball
    euclidean_distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    return X, Y, Z, euclidean_distance


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Load your trained YOLO model
    weights = (
        Path(__file__).resolve().parent.parent
        / "Object_Detection_Test_1"
        / "train3"
        / "weights"
        / "best.pt"
    )
    model = YOLO(str(weights))

    # 2. Open Video Source (0 or 1 for MacBook built-in webcam)
    cap = cv2.VideoCapture(1)

    # Allow the camera to warm up and fetch actual resolution
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check camera index.")
        exit()

    frame_height, frame_width = frame.shape[:2]
    print(f"Camera Initialized at Resolution: {frame_width}x{frame_height}")

    # 3. Define Camera Intrinsics dynamically based on Mac resolution
    # Mac webcams typically have a Horizontal Field of View (HFOV) of ~60 degrees.
    # Focal length formula: f = (width / 2) / tan(HFOV / 2)
    # For a 1080p or 720p macbook camera, ~1100 is a very solid starting estimate for fx.
    FOCAL_LENGTH = 1100.0

    camera_matrix = np.array([
        [FOCAL_LENGTH, 0.0, frame_width / 2.0],  # cx is perfectly in the center width
        [0.0, FOCAL_LENGTH, frame_height / 2.0],  # cy is perfectly in the center height
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    print("Starting video stream... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Run YOLO Inference
        results = model.predict(source=frame, conf=0.5, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_bbox = [x1, y1, x2, y2]

                # 5. Estimate 3D Position mathematically (No PnP needed)
                X, Y, Z, distance = estimate_3d_position(raw_bbox, camera_matrix)

                if Z is not None:
                    # 6. Visualization
                    # Draw YOLO box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Draw center dot
                    cx_box, cy_box = int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)
                    cv2.circle(frame, (cx_box, cy_box), 4, (0, 0, 255), -1)

                    # Display the Coordinates
                    # Z is depth (forward), X is horizontal (left/right), Y is vertical (up/down)
                    label = f"Dist: {distance:.2f}m | Z:{Z:.2f} X:{X:.2f} Y:{Y:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("UAV Dynamic Capture - Distance Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()