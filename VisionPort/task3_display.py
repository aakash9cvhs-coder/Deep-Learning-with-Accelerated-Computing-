
import cv2

# Read the traffic image
img = cv2.imread("traffic_image.jpg")

# Check if image is loaded
if img is None:
    print("Image not found!")
else:
    # Display image
    cv2.imshow("Traffic Image", img)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Close window
    cv2.destroyAllWindows()
