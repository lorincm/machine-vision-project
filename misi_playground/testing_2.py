import cv2
import numpy as np
import os

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpointsL = [] # 2d points in image plane for left camera
imgpointsR = [] # 2d points in image plane for right camera

#
#left_imgs = os.listdir("./calibration_pic/left")
#right_imgs = os.listdir("./calibration_pic/right")

#file_names = zip(left_imgs, right_imgs)

dir_path = 'calibration_pic'
left_path = os.path.join(dir_path, 'left')
right_path = os.path.join(dir_path, 'right')

# List all files in the left and right subfolders and sort them
left_files = sorted(os.listdir(left_path))
right_files = sorted(os.listdir(right_path))

# It's assumed that for every left image there's a corresponding right image with the same index
assert len(left_files) == len(right_files), "Mismatch in number of left and right images"

image_pairs = []

for left_file, right_file in zip(left_files, right_files):
    left_filename = os.path.join(left_path, left_file)
    right_filename = os.path.join(right_path, right_file)

    left_image = cv2.imread(left_filename)
    right_image = cv2.imread(right_filename)
    
    if left_image is not None and right_image is not None:
        image_pairs.append((left_image, right_image))

i = 1
for img_pair in image_pairs: 
    print("calculating img pair "+str(i)+"/"+str(len(image_pairs)))
    grayL = cv2.cvtColor(img_pair[0], cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(img_pair[1], cv2.COLOR_BGR2GRAY)
    
    retL, cornersL = cv2.findChessboardCorners(grayL, (7,6), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (7,6), None)

    if retL and retR:
        objpoints.append(objp)

        refinedCornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(refinedCornersL)

        refinedCornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(refinedCornersR)

retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here's how to calibrate the stereo pair
retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1], flags=flags, criteria=criteria)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T)

mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)

#to rectify any image pair
dstL = cv2.remap(img_pair[0], mapL1, mapL2, cv2.INTER_LINEAR)
dstR = cv2.remap(img_pair[1], mapR1, mapR2, cv2.INTER_LINEAR)

#saving calibration data
np.savez_compressed('calibration_data.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F)

#loading calibration data 
# with np.load('calibration_data.npz') as data:
#     mtxL = data['mtxL']
#     distL = data['distL']
#     mtxR = data['mtxR']
#     distR = data['distR']
#     R = data['R']
#     T = data['T']
#     E = data['E']
#     F = data['F']


