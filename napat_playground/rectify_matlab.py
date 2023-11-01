import cv2
import numpy as np
from scipy.io import loadmat

# Load the MATLAB data
data = loadmat('simplifiedStereoParams.mat')

# Extract the parameters
camera_matrix1 = data['CameraMatrix1']
camera_matrix2 = data['CameraMatrix2']
dist_coeffs1 = data['DistCoeffs1'][0]
dist_coeffs2 = data['DistCoeffs2'][0]
R = data['RotationOfCamera2']
T = data['TranslationOfCamera2'].T

# Image size (replace with your image dimensions)
img_size = (1080, 1920)

# Apply stereo rectification
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, img_size, R, T)

# Load left and right images
left_img = cv2.imread('./calibration_pic/left/left_20231027_171901_0.jpg', cv2.IMREAD_GRAYSCALE)  # replace with your image path
right_img = cv2.imread('./calibration_pic/right/right_20231027_171901_0.jpg', cv2.IMREAD_GRAYSCALE)  # replace with your image path

# Rectify images
left_map1, left_map2 = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, img_size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, img_size, cv2.CV_16SC2)

rectified_left = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

# Ensure all images have the same size as the original left and right images
rectified_left = cv2.resize(rectified_left, (left_img.shape[1], left_img.shape[0]))
rectified_right = cv2.resize(rectified_right, (right_img.shape[1], right_img.shape[0]))

# Stitch images for display: left original, right original, left rectified, right rectified
top_row = np.hstack((left_img, right_img))
bottom_row = np.hstack((rectified_left, rectified_right))
combined_img = np.vstack((top_row, bottom_row))

# Display the result
cv2.imshow('Comparison', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
