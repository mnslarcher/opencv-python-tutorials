import os

import cv2 as cv
import numpy as np

DATA_DIR = os.path.join(
    os.sep, *os.path.realpath(__file__).split(os.sep)[:-2], "samples", "data"
)

filename = os.path.join(DATA_DIR, "home.jpg")
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

dst = cv.drawKeypoints(gray, kp, img)

cv.imshow("sift_keypoints", dst)
cv.waitKey()
cv.destroyAllWindows()

dst = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("sift_keypoints", dst)
cv.waitKey()
cv.destroyAllWindows()
