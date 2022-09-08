import glob

import cv2 as cv
import numpy as np

# Camera Calibration

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
objpts = []
imgpts = []
images = glob.glob("../samples/data/left*")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret:
        objpts.append(objp)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpts.append(corners)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpts, imgpts, gray.shape[::-1], None, None
)


# Draw


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners.astype(int), imgpts.astype(int))
        cv.imshow("img", img)
        k = cv.waitKey(0) & 0xFF
        if k == ord("s"):
            cv.imwrite(fname[:6] + ".png", img)

cv.destroyAllWindows()


# Render a Cube


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # Draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # Draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


axis = np.float32(
    [
        [0, 0, 0],
        [0, 3, 0],
        [3, 3, 0],
        [3, 0, 0],
        [0, 0, -3],
        [0, 3, -3],
        [3, 3, -3],
        [3, 0, -3],
    ]
).reshape(-1, 3)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners.astype(int), imgpts.astype(int))
        cv.imshow("img", img)
        k = cv.waitKey(0) & 0xFF
        if k == ord("s"):
            cv.imwrite(fname[:6] + ".png", img)

cv.destroyAllWindows()
