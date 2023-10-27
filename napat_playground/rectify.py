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
T = data['TranslationOfCamera2']

# Image size (replace with your image dimensions)
img_size = (1080, 1920)

# Apply stereo rectification
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, img_size, R, T)

# Now you can use the R1, R2, P1, P2 matrices for rectifying the stereo images.
