
# TASK 15: Estimate Vehicle Waiting Time
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

# Get video FPS
fps = video.get(cv2.CAP_PROP_FPS)

# Store previous vehicle positions
previous_positions = {}

# Store waiting time for each vehicle
waiting_times = {}

# Vehicle class IDs: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

cv2.namedWindow("Vehicle Waiting Time", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Waiting Time", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    # Track vehicles
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        verbose=False
    )

    for result in results:

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, track_id, class_id in zip(boxes, ids, classes):

                if int(class_id) in vehicle_classes:

                    x1, y1, x2, y2 = map(int, box)

                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    track_id = int(track_id)

                    waiting_times.setdefault(track_id, 0)

                    # Check vehicle movement
                    if track_id in previous_positions:

                        old_x, old_y = previous_positions[track_id]

                        movement = (
                            (center_x - old_x) ** 2 +
                            (center_y - old_y) ** 2
                        ) ** 0.5

                        # If vehicle is almost stationary
                        if movement < 3:
                            waiting_times[track_id] += 1

                        else:
                            waiting_times[track_id] = 0

                    # Update position
                    previous_positions[track_id] = (
                        center_x, center_y
                    )

                    # Convert frames into seconds
                    waiting_seconds = waiting_times[track_id] / fps

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    # Display waiting time
                    cv2.putText(
                        frame,
                        "ID " + str(track_id) +
                        " Wait: " + str(round(waiting_seconds, 1)) + " sec",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

    # Display video
    cv2.imshow("Vehicle Waiting Time", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 15 completed!")
