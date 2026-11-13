
import cv2
from ultralytics import YOLO

# Task 6: Detect Cars Using YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# Create display window
cv2.namedWindow("Task 6 - Car Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Task 6 - Car Detection", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect objects in the frame
    results = model(frame)

    # Draw detection boxes
    for result in results:
        for box in result.boxes:

            # COCO class ID 2 = Car
            class_id = int(box.cls[0])

            if class_id == 2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw green bounding box
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                # Display car label
                cv2.putText(
                    frame, "Car",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

    # Resize frame for display
    display_frame = cv2.resize(frame, (960, 540))

    cv2.imshow("Task 6 - Car Detection", display_frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 6 completed!")
