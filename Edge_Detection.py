import cv2
import numpy as np

# Load image
image = cv2.imread("sandalwood_branch.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Canny Edge Detection
edges = cv2.Canny(image, 50, 150)  # Adjust thresholds as needed

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Show results
cv2.imshow("Edges", edges)
cv2.imshow("Contours", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
