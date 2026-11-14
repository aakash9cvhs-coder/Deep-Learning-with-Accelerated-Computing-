
# TASK 20: Triple-Riding Detection
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

cv2.namedWindow("Triple Riding Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Triple Riding Detection", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    # Detect objects
    results = model(frame, imgsz=640, verbose=False)

    persons = []
    motorcycles = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Class 0: Person
            if class_id == 0:
                persons.append((x1, y1, x2, y2))

            # Class 3: Motorcycle
            elif class_id == 3:
                motorcycles.append((x1, y1, x2, y2))

    # Check motorcycles
    for mx1, my1, mx2, my2 in motorcycles:

        rider_count = 0

        # Count persons near the motorcycle
        for px1, py1, px2, py2 in persons:

            person_center_x = (px1 + px2) // 2
            person_center_y = (py1 + py2) // 2

            if (
                mx1 < person_center_x < mx2
                and my1 - 100 < person_center_y < my2 + 50
            ):
                rider_count += 1

        # Detect triple riding
        if rider_count > 2:
            color = (0, 0, 255)
            label = "TRIPLE RIDING: " + str(rider_count) + " Persons"
        else:
            color = (0, 255, 0)
            label = "Riders: " + str(rider_count)

        # Draw motorcycle box
        cv2.rectangle(
            frame, (mx1, my1), (mx2, my2),
            color, 2
        )

        cv2.putText(
            frame,
            label,
            (mx1, my1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Display output
    cv2.imshow("Triple Riding Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 20 completed!")
