import cv2
import numpy as np
from scipy.io import loadmat

# Load stereo parameters from .mat file
mat_data = loadmat('processedStereoParams.mat')

# Print available keys
print(mat_data.keys())

# Extract the stereo parameters from the loaded data
left_cam_matrix = mat_data['LeftCam']['IntrinsicMatrix'].T  # Transpose because MATLAB uses column-major order
right_cam_matrix = mat_data['RightCam']['IntrinsicMatrix'].T

left_dist_coeffs = mat_data['LeftCam']['RadialDistortion'][:2]
right_dist_coeffs = mat_data['RightCam']['RadialDistortion'][:2]

R = mat_data['RotationOfCamera2']
T = mat_data['TranslationOfCamera2']

# Create StereoSGBM object
window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

cap1 = cv2.VideoCapture(1)  # Assuming left camera is device 1
cap2 = cv2.VideoCapture(2)  # Assuming right camera is device 2

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # Display the images
    cv2.imshow('Left Image', gray1)
    cv2.imshow('Right Image', gray2)
    cv2.imshow('Disparity Map', (disparity-min_disp)/num_disp)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
