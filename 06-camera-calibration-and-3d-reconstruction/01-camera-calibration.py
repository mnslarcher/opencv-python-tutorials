import glob

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# See also: https://learnopencv.com/camera-calibration-using-opencv/

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0, 0, 0), (1, 0, 0), ..., (6, 5, 0)
objp = np.zeros((6 * 7, 3), np.float32)
# [[[0, 0, 0, 0, 0, 0],
#   [1, 1, 1, 1, 1, 1],
#   [2, 2, 2, 2, 2, 2],
#   [3, 3, 3, 3, 3, 3],
#   [4, 4, 4, 4, 4, 4],
#   [5, 5, 5, 5, 5, 5],
#   [6, 6, 6, 6, 6, 6]],

# [[0, 1, 2, 3, 4, 5],
#  [0, 1, 2, 3, 4, 5],
#  [0, 1, 2, 3, 4, 5],
#  [0, 1, 2, 3, 4, 5],
#  [0, 1, 2, 3, 4, 5],
#  [0, 1, 2, 3, 4, 5],
#  [0, 1, 2, 3, 4, 5]]]
# After reshape is the combination of the elements of the first grid with the elements of the second grid in the same
# position.
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

images = glob.glob("../samples/data/left*")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 6), corners, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

cv.destroyAllWindows()


# Calibration

# mtx: Input/output 3x3 floating-point camera intrinsic matrix
# dist: Input/output vector of distortion coefficients
# rvecs: Output vector of rotation vectors (Rodrigues) estimated for each pattern view
# tvecs: Output vector of translation vectors estimated for each pattern view
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)


# Undistortion

img = cv.imread("../samples/data/left12.jpg")
h, w = img.shape[:2]
# If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels. So it may even remove
# some pixels at image corners. If alpha=1, all pixels are retained with some extra black images. This function also
# returns an image ROI which can be used to crop the result.
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]


# Remapping

# Undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Crop the image
x, y, w, h = roi
dst2 = dst2[y : y + h, x : x + w]

plt.figure("Undistort")
plt.subplot(131)
plt.imshow(img)
plt.title("Original")

plt.subplot(132)
plt.imshow(dst)
plt.title("Undistortion")

plt.subplot(133)
plt.imshow(dst2)
plt.title("Remapping")

plt.show()


# Re-projection Error

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints)
    mean_error += error

print(f"Total error: {mean_error / len(objpoints)}")
