
import cv2
from ultralytics import YOLO

# Task 8: Display Vehicle Names and Confidence Scores

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# Create display window
cv2.namedWindow("Task 8 - Confidence Scores", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Task 8 - Confidence Scores", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            # Get class name
            class_id = int(box.cls[0])
            vehicle_name = model.names[class_id]

            # Get confidence score
            confidence = float(box.conf[0])
            score = confidence * 100

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Create label with confidence
            label = f"{vehicle_name} {score:.2f}%"

            # Draw bounding box
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            # Display name and confidence
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    # Resize frame
    display_frame = cv2.resize(frame, (960, 540))

    cv2.imshow("Task 8 - Confidence Scores", display_frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 8 completed!")
