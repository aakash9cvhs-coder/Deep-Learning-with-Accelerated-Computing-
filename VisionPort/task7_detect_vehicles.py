
import cv2
from ultralytics import YOLO

# Task 7: Detect Bikes, Buses, and Trucks

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# COCO class IDs
vehicle_classes = {
    1: "Bicycle",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

cv2.namedWindow("Task 7 - Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Task 7 - Vehicle Detection", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect vehicles
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id in vehicle_classes:
                label = vehicle_classes[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                # Display vehicle name
                cv2.putText(
                    frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

    # Resize for display
    display_frame = cv2.resize(frame, (960, 540))

    cv2.imshow("Task 7 - Vehicle Detection", display_frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 7 completed!")
