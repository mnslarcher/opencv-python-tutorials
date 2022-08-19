import sys

import cv2 as cv

img = cv.imread("../samples/data/starry_night.jpg")

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
# Delay in milliseconds. 0 is the special value meaning "forever". When the user presses a key, the window is destroyed.
# The return value is the key that was pressed.
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
