import cv2 as cv
import numpy as np

# Scaling

# Note: use cv.INTER_AREA for shrinking and cv.INTER_CUBIC (slow) or cv.INTER_LINEAR (fast, default) for zooming.

img = cv.imread("../samples/data/messi5.jpg")
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

height, width = img.shape[:2]
res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)

img_padded = cv.copyMakeBorder(img, 0, height, 0, 0, cv.BORDER_CONSTANT)

cv.imshow("image", cv.hconcat([img_padded, res]))
cv.waitKey(0)
cv.destroyAllWindows()


# Translation

img = cv.imread("../samples/data/messi5.jpg", 0) # 0 = grayscale
rows, cols = img.shape

# Translation matrix with tx = 100, ty = 50
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv.warpAffine(img, M, (cols, rows)) # third arg is width, height

cv.imshow("image", cv.hconcat([img, dst]))
cv.waitKey(0)
cv.destroyAllWindows()


# Rotation


