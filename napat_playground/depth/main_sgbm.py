import cv2
import sys
import numpy as np
import time
from stereo_camera import *

# Create stereo camera object
stereo_camera = StereoCamera()

keep_processing = True

# Define display window name
window_name = "Stereo Camera Input"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

frameL, frameR = stereo_camera.get_frames()

# Set the size of the window
heightL, widthL, channelsL = frameL.shape
heightR, widthR, channelsR = frameR.shape

combined_width = widthL + widthR

print("Combined Image size = ", combined_width, " x ", heightL)
cv2.resizeWindow(window_name, combined_width, heightL)

window_size = 7  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

left_matcher = cv2.StereoSGBM_create(
    minDisparity=-1,
    numDisparities=16*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=window_size,
    P1=9 * 3 * window_size,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=128 * 3 * window_size,
    disp12MaxDiff=12,
    uniquenessRatio=40,
    speckleWindowSize=50,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# FILTER Parameters
lmbda = 70000
sigma = 1.7
visual_multiplier = 6

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

while keep_processing:

    # Get frames from camera
    frameL, frameR = stereo_camera.get_frames()

    displ = left_matcher.compute(frameL, frameR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(frameR, frameL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filteredImg = wls_filter.filter(displ, frameL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.cvtColor(filteredImg, cv2.COLOR_GRAY2BGR)
    color_map = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)

    # Concatenate the frames
    bottom_combined = np.hstack((frameL, frameR))
    top_combined = np.hstack((filteredImg, color_map))
    combined_frame = np.vstack((top_combined, bottom_combined))

    # Display the combined frame
    cv2.imshow(window_name, combined_frame)

    # Start the event loop
    key = cv2.waitKey(40) & 0xFF

    # Loop control
    if key == ord(' '):
        keep_processing = False
    elif key == ord('x'):
        exit()
    elif key == ord('s'):
        print("SWAP")
        stereo_camera.swap_cameras()

