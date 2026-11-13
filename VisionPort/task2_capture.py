
import cv2

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

print("Press SPACE to capture image")
print("Press ESC to exit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Traffic Signal Capture", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE key
        cv2.imwrite("traffic_image.jpg", frame)
        print("Image saved as traffic_image.jpg")
        break

    elif key == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()
