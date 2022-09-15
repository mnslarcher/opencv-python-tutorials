import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Depth Map from Stereo Images

# disparity = x - x' = B * f / Z

imgL = cv.imread("../samples/data/tsukuba_l.png", 0)
imgR = cv.imread("../samples/data/tsukuba_r.png", 0)

# numDisparities: the disparity search range. For each pixel algorithm will find the best disparity from 0 (default 
#     minimum disparity) to numDisparities. The search range can then be shifted by changing the minimum disparity

# blockSize: the linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered 
#     at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size 
#     gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence.
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, "gray")
plt.show()
