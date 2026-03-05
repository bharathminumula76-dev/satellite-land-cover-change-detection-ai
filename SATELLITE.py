import cv2
import numpy as np

# Load satellite images
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Find difference between images
difference = cv2.absdiff(gray1, gray2)

# Threshold the difference
_, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

# Highlight changes
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 500:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save result
cv2.imwrite("output.jpg", image2)

cv2.imshow("Detected Changes", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
