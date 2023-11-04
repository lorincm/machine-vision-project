import cv2
import tensorflow as tf
import numpy as np
import time as time
from stereo_camera import *
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img

def init_hitnet():

    # Select model type
    # model_type = ModelType.middlebury
    # model_type = ModelType.flyingthings
    model_type = ModelType.eth3d

    if model_type == ModelType.middlebury:
        model_path = "hitnet/models/middlebury_d400.pb"
    elif model_type == ModelType.flyingthings:
        model_path = "hitnet/models/flyingthings_finalpass_xl.pb"
    elif model_type == ModelType.eth3d:
        model_path = "hitnet/models/eth3d.pb"

    print("Initializing Hitnet model")
    # Initialize model
    hitnet_depth = HitNet(model_path, model_type)
    return hitnet_depth


if __name__ == '__main__':

    hitnet_depth = init_hitnet()
    stereo_camera = StereoCamera()