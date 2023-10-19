import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('./calibration_pic/left/left_20231019_163553_7.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('./calibration_pic/right/right_20231019_163553_7.jpg', cv.IMREAD_GRAYSCALE)
stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()