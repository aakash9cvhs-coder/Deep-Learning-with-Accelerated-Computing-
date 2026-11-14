
# TASK 12: Track Individual Vehicles
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

# Create display window
cv2.namedWindow("Vehicle Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Tracking", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (960, 540))

    # Track vehicles using YOLO
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        verbose=False
    )

    # Vehicle class IDs: car, motorcycle, bus, truck
    vehicle_classes = [2, 3, 5, 7]

    for result in results:
        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, track_id, class_id in zip(boxes, ids, classes):

                if int(class_id) in vehicle_classes:

                    x1, y1, x2, y2 = map(int, box)

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    # Display unique vehicle ID
                    cv2.putText(
                        frame,
                        "Vehicle ID: " + str(int(track_id)),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

    # Show tracking output
    cv2.imshow("Vehicle Tracking", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 12 completed!")
