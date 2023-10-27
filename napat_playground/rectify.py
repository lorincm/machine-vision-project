import cv2
import numpy as np
from scipy.io import loadmat

# Load stereo parameters from .mat file
mat_data = loadmat('processedStereoParams.mat')

# Extract the stereo parameters from the loaded data
left_cam_matrix = mat_data['LeftCam']['IntrinsicMatrix'].T  # Transpose because MATLAB uses column-major order
right_cam_matrix = mat_data['RightCam']['IntrinsicMatrix'].T

left_dist_coeffs = mat_data['LeftCam']['RadialDistortion'][:2]
right_dist_coeffs = mat_data['RightCam']['RadialDistortion'][:2]

R = mat_data['RotationOfCamera2']
T = mat_data['TranslationOfCamera2']

# Assuming left camera is index 0 and right camera is index 1
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# Get the resolution of the video feed (assuming both cameras have the same resolution)
ret, frame = cap_left.read()
ret_right, frame_right = cap_right.read()
if not ret:
    print("Failed to grab frame from left camera.")
    exit()

height, width = frame.shape[:2]
height_right, width_right = frame.shape[:2]
print(f"CAP LEFT height: {height} width: {width}")
print(f"CAP RIGHT height: {height_right} width: {width_right}")


# Compute rectification transforms for stereo cameras
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    left_cam_matrix, left_dist_coeffs, right_cam_matrix, right_dist_coeffs, imageSize=(width, height), R=R, T=T
)

# Compute the mapping matrices
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_cam_matrix, left_dist_coeffs, R1, P1, (width, height), cv2.CV_16SC2
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_cam_matrix, right_dist_coeffs, R2, P2, (width, height), cv2.CV_16SC2
)

# Create a single window
cv2.namedWindow("Stereo Rectification", cv2.WINDOW_NORMAL)

while True:
    ret_left, left_img = cap_left.read()
    ret_right, right_img = cap_right.read()

    if not ret_left or not ret_right:
        print("Failed to grab the frame.")
        break

    left_rectified = cv2.remap(left_img, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)

    # Concatenate the images for display
    top_row = np.hstack((left_img, right_img))
    bottom_row = np.hstack((left_rectified, right_rectified))
    combined = np.vstack((top_row, bottom_row))

    # Show the concatenated images
    cv2.imshow("Stereo Rectification", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
