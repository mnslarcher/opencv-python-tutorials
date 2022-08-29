import cv2 as cv
import matplotlib.pyplot as plt
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

# Rotation by a theta angle:
# [[cosO, -sinO], 
#  [sinO, cosO]]

# Scaled rotation with adjustable center of rotation
# [[a,  b, (1 - a) * center_x - b * center_y],
#  [-b, a, b * center_x + (1 - a) * center_y]]

img = cv.imread("../samples/data/messi5.jpg", 0)
rows, cols = img.shape

# cols - 1 and rows - 1 are the coordinates limits
# center, angle, scale
M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
dst = cv.warpAffine(img, M, (cols, rows))

cv.imshow("image", cv.hconcat([img, dst]))
cv.waitKey(0)
cv.destroyAllWindows()


# Affine transformation

# [[A,   t], 
#  [0^T, 1]
# Invariance: parallelism, volume ratio

img = cv.imread("../samples/data/sudoku.png")
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

for pt in pts1:
    img = cv.circle(img, pt.astype(int), 5, color=(0, 255, 0), thickness=-1)

M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, (cols, rows))

plt.figure("Affine transformation")
plt.subplot(121)
plt.imshow(img)
plt.title("Input")

plt.subplot(122)
plt.imshow(dst)
plt.title("Ouput")

plt.show()


# Perspective Transformation

# [[A,   t], 
#  [a^T, v]
# Invariance: plane intersection and tangency

img = cv.imread("../samples/data/sudoku.png")
rows, cols, ch = img.shape

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

for pt in pts1:
    img = cv.circle(img, pt.astype(int), 5, color=(0, 255, 0), thickness=-1)

M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, M, (cols, rows))

plt.figure("Perspective Transformation")
plt.subplot(121)
plt.imshow(img)
plt.title("Input")

plt.subplot(122)
plt.imshow(dst)
plt.title("Ouput")

plt.show()



