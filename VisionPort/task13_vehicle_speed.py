
# TASK 13: Approximate Vehicle Speed
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

# Video settings
fps = video.get(cv2.CAP_PROP_FPS)

# Approximate real-world distance covered per pixel
# Adjust this value after camera calibration
meters_per_pixel = 0.05

# Store previous vehicle positions
previous_positions = {}

cv2.namedWindow("Vehicle Speed Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Speed Estimation", 960, 540)

vehicle_classes = [2, 3, 5, 7]

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

                    speed = 0

                    if track_id in previous_positions:

                        old_x, old_y = previous_positions[track_id]

                        # Calculate pixel movement
                        distance_pixels = (
                            (center_x - old_x) ** 2 +
                            (center_y - old_y) ** 2
                        ) ** 0.5

                        # Estimate speed in km/h
                        distance_meters = (
                            distance_pixels * meters_per_pixel
                        )

                        speed = (
                            distance_meters * fps * 3.6
                        )

                    # Update vehicle position
                    previous_positions[track_id] = (
                        center_x, center_y
                    )

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    # Display speed
                    cv2.putText(
                        frame,
                        "ID " + str(track_id) +
                        " Speed: " + str(round(speed, 1)) + " km/h",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

    cv2.imshow("Vehicle Speed Estimation", frame)

    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 13 completed!")
