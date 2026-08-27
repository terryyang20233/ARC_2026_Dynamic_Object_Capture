from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def estimate_distance_pnp(bbox, camera_matrix, dist_coeffs):
    """
    Estimates the 3D position of a tennis ball using solvePnP.
    """
    x_min, y_min, x_max, y_max = bbox

    # Standard tennis ball radius = 0.0335 meters
    R = 0.0335

    # 1. 3D Object Points (in meters)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    object_points = np.array([
        [-R, -R, 0.0],
        [R, -R, 0.0],
        [R, R, 0.0],
        [-R, R, 0.0]
    ], dtype=np.float32)

    # 2. 2D Image Points (in pixels)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    image_points = np.array([
        [x_min, y_min],
        [x_max, y_min], [x_max, y_max],
        [x_min, y_max]
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
        euclidean_distance = np.linalg.norm(tvec)
        return tvec, euclidean_distance
    else:
        return None, None



def square_bbox(bbox):
    """
    Forces a YOLO rectangular bounding box into a perfect square centered
    on the original box. This drastically improves PnP accuracy for spheres.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Find the center
    cx = x1 + (w / 2)
    cy = y1 + (h / 2)

    # Use the largest dimension to ensure the whole ball is encompassed
    size = max(w, h)

    # Calculate new square coordinates
    new_x1 = cx - (size / 2)
    new_y1 = cy - (size / 2)
    new_x2 = cx + (size / 2)
    new_y2 = cy + (size / 2)

    return [new_x1, new_y1, new_x2, new_y2]


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

    # 2. Define Camera Intrinsics (MUST BE CALIBRATED FOR YOUR REAL DRONE CAMERA)
    # These are dummy values for a standard 640x480 webcam
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0],
                              [0.0, 0.0, 1.0]
                              ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # 3. Open Video Source (0 = standard webcam, or put a path to an .mp4 file)
    cap = cv2.VideoCapture(1)

    print("Starting video stream... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # 4. Run YOLO Inference on the current frame
        # stream=True keeps memory low, verbose=False stops console spam
        results = model.predict(source=frame, conf=0.5, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates from the YOLO tensor
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_bbox = [x1, y1, x2, y2]

                # Force the bounding box into a perfect square for better PnP math
                sq_bbox = square_bbox(raw_bbox)

                # 5. Estimate Distance
                tvec, distance = estimate_distance_pnp(sq_bbox, camera_matrix, dist_coeffs)

                if tvec is not None:
                    # Extract True 3D coordinates
                    X = tvec[0][0]
                    Y = tvec[1][0]
                    Z = tvec[2][0]

                    # 6. Visualization (Draw on the frame)
                    # Draw original YOLO box (Blue)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                    # Draw Squared box used for math (Green)
                    cv2.rectangle(frame, (int(sq_bbox[0]), int(sq_bbox[1])), (int(sq_bbox[2]), int(sq_bbox[3])),
                                  (0, 255, 0), 1)

                    # Display the Distance and Coordinates
                    label = f"Dist: {distance:.2f}m | Z: {Z:.2f}m"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Print to console for debugging
                    # print(f"Ball detected at: X:{X:.2f}m, Y:{Y:.2f}m, Z:{Z:.2f}m | Euclidean: {distance:.2f}m")

        # Display the frame
        cv2.imshow("UAV Dynamic Capture - Distance Estimation", frame)

        # Press 'q' to exit the video loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()