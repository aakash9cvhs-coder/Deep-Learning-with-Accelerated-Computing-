
import cv2
from ultralytics import YOLO

# Task 10: Calculate Traffic Density

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# Vehicle class IDs: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

cv2.namedWindow("Task 10 - Traffic Density", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Task 10 - Traffic Density", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect vehicles
    results = model(frame, verbose=False)

    vehicle_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id in vehicle_classes:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

    # Determine traffic density
    if vehicle_count <= 5:
        density = "LOW"
        color = (0, 255, 0)

    elif vehicle_count <= 15:
        density = "MEDIUM"
        color = (0, 255, 255)

    else:
        density = "HIGH"
        color = (0, 0, 255)

    # Display count and density
    cv2.putText(
        frame, f"Vehicle Count: {vehicle_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

    cv2.putText(
        frame, f"Traffic Density: {density}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, color, 2
    )

    # Resize frame for display
    display_frame = cv2.resize(frame, (960, 540))

    cv2.imshow("Task 10 - Traffic Density", display_frame)

    # Press ESC to stop
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 10 completed!")
