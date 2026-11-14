
# TASK 16: Entry and Exit Vehicle Count
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

# Create window
cv2.namedWindow("Entry Exit Vehicle Count", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Entry Exit Vehicle Count", 960, 540)

# Vehicle class IDs: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Entry and exit counts
entry_count = 0
exit_count = 0

# Previous vehicle positions
previous_positions = {}

# Counting line
line_y = 270

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

    # Draw counting line
    cv2.line(
        frame, (0, line_y), (960, line_y),
        (255, 0, 0), 2
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

                    # Check if vehicle crossed the line
                    if track_id in previous_positions:

                        old_y = previous_positions[track_id]

                        if old_y < line_y and center_y >= line_y:
                            entry_count += 1

                        elif old_y >= line_y and center_y < line_y:
                            exit_count += 1

                    # Update position
                    previous_positions[track_id] = center_y

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    # Display vehicle ID
                    cv2.putText(
                        frame,
                        "ID: " + str(track_id),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

    # Display entry and exit count
    cv2.putText(
        frame,
        "Entry Count: " + str(entry_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Exit Count: " + str(exit_count),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Show output
    cv2.imshow("Entry Exit Vehicle Count", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Total Entry:", entry_count)
print("Total Exit:", exit_count)
print("Task 16 completed!")
