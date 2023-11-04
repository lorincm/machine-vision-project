import cv2
import os
import numpy as np

class StereoCamera:

    def __init__(self):

        self.camera_opened = False
        
        print("Loading Calibration data")
        self.calib_path = "./calibration/03-11-23-23-03-rms-2.91-zed-0-ximea-0"
        os.chdir(self.calib_path)
        self.mapL1 = np.load('mapL1.npy')
        self.mapL2 = np.load('mapL2.npy')
        self.mapR1 = np.load('mapR1.npy')
        self.mapR2 = np.load('mapR2.npy')

        ## FOR CAMERA FINDING
        target_resolution = (1920, 1080)
        opened_cams = []
        found_indices = []

        # Loop through each camera index to find stereo cameras
        for i in range(3):
            cam = cv2.VideoCapture(i)
            if cam.isOpened():
                ret, frame = cam.read()
                if ret and frame.shape[1] == target_resolution[0] and frame.shape[0] == target_resolution[1]:
                    found_indices.append(i)
                    opened_cams.append(cam)
                    if len(opened_cams) == 2:  # Break if we found both cameras
                        break
                else:
                    cam.release()

        if len(opened_cams) != 2:
            print("Could not find both stereo cameras.")
            exit()

        # Assign the opened cameras to the desired variables
        self.camL, self.camR = opened_cams

        print("Cameras initialized at indices:", found_indices)

    def swap_cameras(self):
        # swap the cameras - for all but ZED camera
        tmp = self.camL
        self.camL = self.camR
        self.camR = tmp

    def get_frames(self):  # return left, right

        self.camL.grab()
        self.camR.grab()

        # then retrieve the images in slow(er) time
        # (do not be tempted to use read() !)

        _, frameL = self.camL.retrieve()
        _, frameR = self.camR.retrieve()

        undistorted_rectifiedL = cv2.remap(frameL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(frameR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

        # return frameL, frameR
        return undistorted_rectifiedL, undistorted_rectifiedR