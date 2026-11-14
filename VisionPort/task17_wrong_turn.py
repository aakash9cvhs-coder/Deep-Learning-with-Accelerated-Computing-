
# TASK 17: Wrong-Turn Detection
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

cv2.namedWindow("Wrong Turn Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Wrong Turn Detection", 960, 540)

# Vehicle classes: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Store previous positions and movement
previous_positions = {}
wrong_turn_ids = set()

# Define a simple junction area
junction_x1 = 300
junction_y1 = 150
junction_x2 = 660
junction_y2 = 450

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        verbose=False
    )

    # Draw junction area
    cv2.rectangle(
        frame,
        (junction_x1, junction_y1),
        (junction_x2, junction_y2),
        (255, 0, 0),
        2
    )

    for result in results:

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, track_id, class_id in zip(boxes, ids, classes):

                if int(class_id) in vehicle_classes:

                    x1, y1, x2, y2 = map(int, box)

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    track_id = int(track_id)

                    if track_id in previous_positions:

                        old_x, old_y = previous_positions[track_id]

                        # Detect sharp direction change
                        dx = center_x - old_x
                        dy = center_y - old_y

                        # Mark a sharp turn inside junction area
                        if (
                            junction_x1 < center_x < junction_x2
                            and junction_y1 < center_y < junction_y2
                            and abs(dx) > 15
                            and abs(dy) > 15
                        ):
                            wrong_turn_ids.add(track_id)

                    previous_positions[track_id] = (
                        center_x, center_y
                    )

                    # Draw vehicle box
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    # Display status
                    if track_id in wrong_turn_ids:
                        label = "Wrong Turn ID: " + str(track_id)
                        color = (0, 0, 255)
                    else:
                        label = "Vehicle ID: " + str(track_id)
                        color = (0, 255, 255)

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

    # Display alert count
    cv2.putText(
        frame,
        "Wrong Turns: " + str(len(wrong_turn_ids)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Wrong Turn Detection", frame)

    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Wrong Turn Detection Completed!")
print("Detected Wrong Turns:", len(wrong_turn_ids))
print("Task 17 completed!")
