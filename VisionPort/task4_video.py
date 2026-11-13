
import cv2

# Load traffic video
video = cv2.VideoCapture("traffic_video.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Resize video to fit the screen
    frame = cv2.resize(frame, (960, 540))

    # Display resized video frame
    cv2.imshow("Traffic Video", frame)

    # Press ESC to stop
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()
