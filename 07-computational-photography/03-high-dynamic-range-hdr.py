import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 1. Loading exposure images into a list
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv.imread(os.path.join("../samples/data", fn)) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

# 2. Merge exposures into HDR image

merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())

merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# 3. Tonemap HDR image

# Tonemap HDR image
tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
res_robertson = tonemap1.process(hdr_robertson.copy())

# 4. Merge exposures using Mertens fusion

# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# 5. Convert to 8-bit and save

# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec * 255, 0, 255).astype("uint8")
res_robertson_8bit = np.clip(res_robertson * 255, 0, 255).astype("uint8")
res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype("uint8")

plt.subplot(221)
plt.imshow(cv.cvtColor(res_debevec_8bit, cv.COLOR_BGR2RGB))

plt.subplot(222)
plt.imshow(cv.cvtColor(res_robertson_8bit, cv.COLOR_BGR2RGB))

plt.subplot(223)
plt.imshow(cv.cvtColor(res_mertens_8bit, cv.COLOR_BGR2RGB))

plt.show()

# Estimating Camera Response Function

cal_debevec = cv.createCalibrateDebevec()
crf_debevec = cal_debevec.process(img_list, times=exposure_times)
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy(), response=crf_debevec.copy())

cal_robertson = cv.createCalibrateRobertson()
crf_robertson = cal_debevec.process(img_list, times=exposure_times)
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())
