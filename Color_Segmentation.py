import cv2
import numpy as np

# Load image
image = cv2.imread("sandalwood_leaf.jpg")

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for yellow/reddish infected areas
lower_bound = np.array([15, 50, 50])  # Adjust these values
upper_bound = np.array([35, 255, 255])

# Create a mask
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Apply mask to extract infected areas
result = cv2.bitwise_and(image, image, mask=mask)

# Show results
cv2.imshow("Original", image)
cv2.imshow("Masked", mask)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()