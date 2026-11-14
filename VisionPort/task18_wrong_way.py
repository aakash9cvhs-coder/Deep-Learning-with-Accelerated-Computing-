
# TASK 18: Wrong-Way Vehicle Alerts
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

cv2.namedWindow("Wrong Way Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Wrong Way Detection", 960, 540)

# Vehicle classes: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Store previous positions
previous_positions = {}

# Assume normal traffic moves from left to right
# Change this direction according to your video
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

    wrong_way_count = 0

    for result in results:

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, track_id, class_id in zip(boxes, ids, classes):

                if int(class_id) in vehicle_classes:

                    x1, y1, x2, y2 = map(int, box)

                    # Calculate vehicle center
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    track_id = int(track_id)

                    wrong_way = False

                    if track_id in previous_positions:

                        old_x, old_y = previous_positions[track_id]

                        # Calculate movement direction
                        movement_x = center_x - old_x

                        # Wrong direction: right to left
                        if movement_x < -5:
                            wrong_way = True

                    # Update position
                    previous_positions[track_id] = (
                        center_x, center_y
                    )

                    # Draw bounding box
                    if wrong_way:
                        color = (0, 0, 255)
                        label = "WRONG WAY ID: " + str(track_id)
                        wrong_way_count += 1
                    else:
                        color = (0, 255, 0)
                        label = "Vehicle ID: " + str(track_id)

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        color, 2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

    # Display wrong-way alert
    cv2.putText(
        frame,
        "Wrong Way Alerts: " + str(wrong_way_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Wrong Way Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 18 completed!")
