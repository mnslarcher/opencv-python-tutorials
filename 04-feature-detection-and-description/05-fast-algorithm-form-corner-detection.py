import os

import cv2 as cv

DATA_DIR = os.path.join(
    os.sep, *os.path.realpath(__file__).split(os.sep)[:-2], "samples", "data"
)

filename = os.path.join(DATA_DIR, "blox.jpg")
img = cv.imread(filename, 0)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Print all default params
print(f"Threshold: {fast.getThreshold()}")
print(f"Non-max suppression: {fast.getNonmaxSuppression()}")
print(f"Neighborhood: {fast.getType()}")
print(f"Total keypoints with non-max suppression: {len(kp)}")

cv.imshow("fast_true", img2)
cv.waitKey()
cv.destroyAllWindows()

# Disable non-max suppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print(f"Total keypoints without non-max suppression: {len(kp)}")

img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

cv.imshow("fast_False", img3)
cv.waitKey()
cv.destroyAllWindows()
