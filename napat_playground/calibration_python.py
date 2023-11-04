import numpy as np
import cv2
import glob

# Chessboard dimensions and square size (in meters)
chessboard_size = (4, 5)  # Number of corners
square_size = 0.030  # e.g., 25 mm or 0.025 m

# Prepare object points with actual square size
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Termination criteria for the iterative optimization algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpointsL = [] # 2d points in image plane for the left camera
imgpointsR = [] # 2d points in image plane for the right camera

# Load the left and right images
images_left = sorted(glob.glob("./calibration_pic/left/left*.jpg"))
images_right = sorted(glob.glob('./calibration_pic/right/right*.jpg'))

for imgL, imgR in zip(images_left, images_right):
    img1 = cv2.imread(imgL)
    img2 = cv2.imread(imgR)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    # if ret1:
    #     cv2.drawChessboardCorners(img1, chessboard_size, corners1, ret1)
    #     cv2.imshow('Detected Corners Left', img1)
    #     cv2.waitKey(0)

    # if ret2:
    #     cv2.drawChessboardCorners(img2, chessboard_size, corners2, ret2)
    #     cv2.imshow('Detected Corners Right', img2)
    #     cv2.waitKey(0)


    if ret1 and ret2:
        print("FOUND")
        objpoints.append(objp)

        refinedCorners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
        refinedCorners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)

        imgpointsL.append(refinedCorners1)
        imgpointsR.append(refinedCorners2)


# Perform stereo calibration
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, gray1.shape[::-1], None, None, None, None)

# Stereo Rectification
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, gray1.shape[::-1], R, T)

mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, gray1.shape[::-1], cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, gray1.shape[::-1], cv2.CV_16SC2)

# Now, you can use the remap function to rectify your stereo images:
imgL_rectified = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR)
imgR_rectified = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR)

# Stack the images
top_row = np.hstack([img1, img2])
bottom_row = np.hstack([imgL_rectified, imgR_rectified])
all_images = np.vstack([top_row, bottom_row])

cv2.imshow('Top: Original Images | Bottom: Rectified Images', all_images)
cv2.waitKey(0)
cv2.destroyAllWindows()