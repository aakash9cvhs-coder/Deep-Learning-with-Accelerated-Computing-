
# TASK 11: Detect Congestion at Signal Junctions
# Intelligent Vehicle Detection and Traffic Monitoring

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# Create resizable window
cv2.namedWindow("Traffic Congestion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Congestion Detection", 960, 540)

# Vehicle class IDs: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Resize video frame
    frame = cv2.resize(frame, (960, 540))

    # Detect vehicles using YOLO
    results = model(frame, imgsz=640, verbose=False)

    vehicle_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id in vehicle_classes:
                vehicle_count += 1

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

    # Determine congestion level
    if vehicle_count <= 5:
        congestion = "LOW"
        color = (0, 255, 0)

    elif vehicle_count <= 15:
        congestion = "MEDIUM"
        color = (0, 255, 255)

    else:
        congestion = "HIGH"
        color = (0, 0, 255)

    # Display vehicle count and congestion
    cv2.putText(
        frame, "Vehicles: " + str(vehicle_count),
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 2
    )

    cv2.putText(
        frame, "Congestion: " + congestion,
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
        1, color, 2
    )

    # Display output
    cv2.imshow("Traffic Congestion Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 11 completed!")
