
import cv2
import os

# Task 5: Extract Individual Frames from Traffic Video

# Open the traffic video
video = cv2.VideoCapture("traffic_video.mp4")

# Check if video is opened
if not video.isOpened():
    print("Traffic video not found!")
    exit()

# Create folder to save frames
os.makedirs("frames", exist_ok=True)

frame_count = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Extract and save every 30th frame
    if frame_count % 30 == 0:
        filename = f"frames/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)

    frame_count += 1

# Release video
video.release()

print("Task 5 completed successfully!")
