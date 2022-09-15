import cv2 as cv

img = cv.imread("../samples/data/messi_2.png")
mask = cv.imread("../samples/data/mask2.png", 0)


dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()

