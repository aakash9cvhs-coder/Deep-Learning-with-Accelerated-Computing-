
# TASK 14: Traffic Signal Color Identification
# Intelligent Vehicle Detection and Traffic Monitoring

import cv2
import numpy as np

# Open traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# Create display window
cv2.namedWindow("Traffic Signal Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Signal Detection", 960, 540)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Resize video
    frame = cv2.resize(frame, (960, 540))

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Yellow color range
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Green color range
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    # Create color masks
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Count pixels for each color
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    # Identify signal color
    if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels > 500:
        signal = "RED"
        color = (0, 0, 255)

    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > 500:
        signal = "YELLOW"
        color = (0, 255, 255)

    elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > 500:
        signal = "GREEN"
        color = (0, 255, 0)

    else:
        signal = "SIGNAL NOT DETECTED"
        color = (255, 255, 255)

    # Display signal status
    cv2.putText(
        frame,
        "Traffic Signal: " + signal,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # Display video
    cv2.imshow("Traffic Signal Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()

print("Task 14 completed!")
