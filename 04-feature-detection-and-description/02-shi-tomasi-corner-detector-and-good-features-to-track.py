import cv2 as cv
import numpy as np

img = cv.imread("../samples/data/blox.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# maxCorners = 25
# qualityLevel = 0.01
# minDistance = 10
corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(  # Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    corners
)

for i in corners:
    x, y = i.ravel()  # Flatten
    cv.circle(img, (x, y), 3, (0, 0, 255), -1)

cv.imshow("image", img)
cv.waitKey(0)
cv.destroyAllWindows()
