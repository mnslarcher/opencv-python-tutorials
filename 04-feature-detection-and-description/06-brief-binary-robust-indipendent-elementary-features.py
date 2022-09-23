import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

DATA_DIR = os.path.join(
    os.sep, *os.path.realpath(__file__).split(os.sep)[:-2], "samples", "data"
)

filename = os.path.join(DATA_DIR, "blox.jpg")

img = cv.imread(filename, 0)

# Initiate STAR detector
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# Find the keypoints with STAR
kp = star.detect(img, None)

# Compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print(brief.descriptorSize())
print(des.shape)
