
# TASK 19: Illegal Parking Detection
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

cv2.namedWindow("Illegal Parking Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Illegal Parking Detection", 960, 540)

# Vehicle class IDs: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Define restricted parking area
zone_x1 = 300
zone_y1 = 150
zone_x2 = 650
zone_y2 = 450

# Store vehicle positions
previous_positions = {}

# Store stationary frame counts
stationary_frames = {}

# Parking threshold
parking_threshold = 100

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

    # Draw restricted zone
    cv2.rectangle(
        frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
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

                    # Calculate vehicle center
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    track_id = int(track_id)

                    # Check if vehicle is inside restricted zone
                    inside_zone = (
                        zone_x1 < center_x < zone_x2
                        and zone_y1 < center_y < zone_y2
                    )

                    if track_id in previous_positions:

                        old_x, old_y = previous_positions[track_id]

                        movement = (
                            (center_x - old_x) ** 2 +
                            (center_y - old_y) ** 2
                        ) ** 0.5

                        if inside_zone and movement < 3:
                            stationary_frames[track_id] = (
                                stationary_frames.get(track_id, 0) + 1
                            )
                        else:
                            stationary_frames[track_id] = 0

                    else:
                        stationary_frames[track_id] = 0

                    # Update position
                    previous_positions[track_id] = (
                        center_x, center_y
                    )

                    # Detect illegal parking
                    if stationary_frames[track_id] > parking_threshold:
                        color = (0, 0, 255)
                        label = "ILLEGAL PARKING ID: " + str(track_id)
                    else:
                        color = (0, 255, 0)
                        label = "Vehicle ID: " + str(track_id)

                    # Draw bounding box
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

    cv2.imshow("Illegal Parking Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 19 completed!")
