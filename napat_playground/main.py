import cv2
import numpy as np
from scipy.io import loadmat


# Load stereo parameters from .mat file
mat_data = loadmat('stereoParams.mat')

# Print available keys
print(mat_data.keys())

stereo_params = mat_data['stereoParams'][0,0]

left_cam_matrix = stereo_params['CameraParameters1'][0,0]['IntrinsicMatrix'].T  # Transpose because MATLAB uses column-major order
right_cam_matrix = stereo_params['CameraParameters2'][0,0]['IntrinsicMatrix'].T

left_dist_coeffs = stereo_params['CameraParameters1'][0,0]['RadialDistortion'][0,:2]
right_dist_coeffs = stereo_params['CameraParameters2'][0,0]['RadialDistortion'][0,:2]

R = stereo_params['RotationOfCamera2']
T = stereo_params['TranslationOfCamera2']


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

cap1 = cv2.VideoCapture(0)  # Assuming left camera is device 0
cap2 = cv2.VideoCapture(1)  # Assuming right camera is device 1

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    cv2.imshow('Disparity', (disparity-min_disp)/num_disp)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
