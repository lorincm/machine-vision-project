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

while keep_processing:

    # Get frames from camera
    frameL, frameR = stereo_camera.get_frames()

    # Concatenate the frames
    bottom_combined = np.hstack((frameL, frameR))

    # Display the combined frame
    cv2.imshow(window_name, bottom_combined)

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
    elif key == ord('p'):
        print("Saving image...")
        base_path = "/Users/napat/Documents/GitHub/machine-vision-project/napat_playground/depth/images"
        cv2.imwrite(base_path+"/left/left_" + time.strftime("%Y%m%d-%H%M%S") + ".jpg", frameL)
        cv2.imwrite(base_path+"/right/right_" + time.strftime("%Y%m%d-%H%M%S") + ".jpg", frameR)
        print("Image saved!")

