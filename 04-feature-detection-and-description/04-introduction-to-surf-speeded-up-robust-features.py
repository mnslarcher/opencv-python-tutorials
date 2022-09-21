import os

import cv2 as cv
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(
    os.sep, *os.path.realpath(__file__).split(os.sep)[:-2], "samples", "data"
)

filename = os.path.join(DATA_DIR, "butterfly.jpg")

img = cv.imread(filename, 0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img, None)

len(kp)

# Check present Hessian threshold
print(surf.getHessianThreshold())

# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500

# Again compute keypoints and check its number
kp, des = surf.detectAndCompute(img, None)

print(len(kp))

img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)

plt.imshow(img2)
plt.show()

# Check upright flag, if it is False, set it to True
print(surf.getUpright())

surf.setUpright(True)

# Recompute the feature points and draw it
kp = surf.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)

plt.imshow(img2)
plt.show()

# Find size of descriptor
print(surf.descriptorSize())

# That means flag, "extended" is False.
surf.getExtended()

# So we make it to True to get 128-dim descriptors.
surf.setExtended(True)
kp, des = surf.detectAndCompute(img, None)
print(surf.descriptorSize())

print(des.shape)
