from pathlib import Path

import cv2
from ultralytics import YOLO

# 1. Load YOUR custom YOLO model
_WEIGHTS = Path(__file__).resolve().parent / "train3" / "weights" / "best.pt"
model = YOLO(str(_WEIGHTS))

# 2. Open the built-in MacBook webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the video window.")

while True:
    # 3. Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # 4. Run YOLO inference using your custom model
    # Added 'conf=0.5' to hide weak predictions (adjust from 0.1 to 0.9 as needed)
    results = model(frame, stream=True, conf=0.5)

    # 5. Process and display the results
    for result in results:
        # Plot the bounding boxes and your custom labels onto the frame
        annotated_frame = result.plot()

        # Display the frame in a window
        cv2.imshow("My Custom YOLO Model", annotated_frame)

    # 6. Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Release resources
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)