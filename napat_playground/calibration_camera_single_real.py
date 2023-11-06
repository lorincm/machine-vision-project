import cv2
import numpy as np
import json

# Load the stereo parameters from the JSON file
with open('./stereoParams.json', 'r') as f:
    data = json.load(f)

camera_matrix_l = np.array(data['intrinsicMatrix1'])
dist_coeffs_l = np.array(data['distortionCoefficients1'])
camera_matrix_r = np.array(data['intrinsicMatrix2'])
dist_coeffs_r = np.array(data['distortionCoefficients2'])
R = np.array(data['rotationOfCamera2'])
T = np.array(data['translationOfCamera2'])

# Load the stereo images
left_image = cv2.imread('./calibration_pic/left/left_20231106_144800_0.jpg', cv2.IMREAD_COLOR)  # Replace with your image path
right_image = cv2.imread('./calibration_pic/right/right_20231106_144800_0.jpg', cv2.IMREAD_COLOR)  # Replace with your image path

# Compute the rectification transforms
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, left_image.shape[:2][::-1], R, T
)

# Compute the undistortion and rectification transformation maps
map1_l, map2_l = cv2.initUndistortRectifyMap(
    camera_matrix_l, dist_coeffs_l, R1, P1, left_image.shape[:2][::-1], cv2.CV_16SC2
)
map1_r, map2_r = cv2.initUndistortRectifyMap(
    camera_matrix_r, dist_coeffs_r, R2, P2, right_image.shape[:2][::-1], cv2.CV_16SC2
)

np.save('./mapL1.npy', map1_l)
np.save('./mapL2.npy', map2_l)
np.save('./mapR1.npy', map1_r)
np.save('./mapR2.npy', map2_r)

# Apply the rectification
left_rectified = cv2.remap(left_image, map1_l, map2_l, interpolation=cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_image, map1_r, map2_r, interpolation=cv2.INTER_LINEAR)

# Display the rectified images side by side
real_image = np.hstack((left_image, right_image))
rectified_image = np.hstack((left_rectified, right_rectified))
combined_image = np.vstack((real_image, rectified_image))
cv2.imshow('Rectified Stereo Images', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
