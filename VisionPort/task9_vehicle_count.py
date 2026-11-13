
import cv2
from ultralytics import YOLO

# Task 9: Count Vehicles in Traffic Video

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# Vehicle class IDs in COCO
vehicle_classes = [2, 3, 5, 7]

cv2.namedWindow("Task 9 - Vehicle Count", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Task 9 - Vehicle Count", 960, 540)

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

            # Count cars, motorcycles, buses, and trucks
            if class_id in vehicle_classes:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                # Display vehicle name
                name = model.names[class_id]
                cv2.putText(
                    frame, name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

    # Display total vehicle count
    cv2.putText(
        frame, f"Vehicle Count: {vehicle_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2
    )

    # Resize for display
    display_frame = cv2.resize(frame, (960, 540))

    cv2.imshow("Task 9 - Vehicle Count", display_frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 9 completed!")
