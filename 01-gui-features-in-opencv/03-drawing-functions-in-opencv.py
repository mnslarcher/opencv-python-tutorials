import cv2 as cv
import numpy as np

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

# img, center, axes, angle, startAngle, endAgle (180 = only the first half), color, thickness
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0, 255, 255))  # True = close the shape

font = cv.FONT_HERSHEY_SIMPLEX
# img, text, bottom-left, font type, font scale, color, thickness, line type
cv.putText(img, "OpenCV", (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow("Display window", img)
cv.waitKey(0)
