import cv2
import tensorflow as tf
import numpy as np
import time as time
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from stereo_camera import *
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from cobot import *


def init_hitnet():

    focal_length = 0.03
    baseline = 600

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
    hitnet_depth = HitNet(model_path, model_type,
                          camera_config=CameraConfig(focal_length, baseline))
    return hitnet_depth


def init_sgbm():
    # Call OpenCV passive stereo algorithms
    stereo = cv2.StereoSGBM_create(
        minDisparity=10, numDisparities=85, blockSize=11)

    lmbda = 8000
    sigma = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    return stereo, wls_filter


def object_detection(object_detection_model, frameL):

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Start time
    start_time = time.time()

    # Detect objects in the left frame
    img_tensor = F.to_tensor(frameL).unsqueeze(
        0)  # Convert frame to tensor format
    detections = object_detection_model(img_tensor)

    # End time
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

    frameL_with_detections = frameL.copy()
    detection_result = []

    for element in range(len(detections[0]['boxes'])):
        box = detections[0]['boxes'][element].cpu().detach().numpy()
        score = detections[0]['scores'][element].cpu().detach().numpy()

        label_index = detections[0]['labels'][element].item()
        if 0 <= label_index < len(COCO_INSTANCE_CATEGORY_NAMES):
            label = COCO_INSTANCE_CATEGORY_NAMES[label_index]
        else:
            print(
                f"Unexpected label index {label_index}. Defaulting to 'unknown'.")
            label = "unknown"

        if label in ["apple", "cat", "clock"] and score > 0.5:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2

            cv2.rectangle(frameL_with_detections, (int(box[0]), int(
                box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.circle(frameL_with_detections, (int(x_center),
                       int(y_center)), 5, (255, 0, 0), -1)

            # Draw label text at the center
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = int(x_center - text_width / 2)
            text_y = int(y_center + text_height / 2)
            cv2.putText(frameL_with_detections, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 1)

            print(f'bounding box midpoint x,y: {x_center}, {y_center}')
            detection_result.append([label, x_center, y_center])

    # Add inference time to the image
    text = f"Inference Time: {elapsed_time:.2f} ms"
    position = (10, 30)
    cv2.putText(frameL_with_detections, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frameL_with_detections, detection_result #label, x, y, z


def add_text(img, text, position):
    cv2.putText(img, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


def disparity_to_depth(disparity_map):

    f = 100
    B = 0.06

    # To prevent division by zero
    valid_disparity = disparity_map.copy()
    # replace zeros with a small value
    valid_disparity[valid_disparity == 0] = 0.01

    depth_map = f * B / valid_disparity
    return depth_map


def depth_estimation(hitnet_depth, stereo, wls_filter, frameL, frameR, method="hitnet"):
    # Start time
    start_time = time.time()

    if method == "sgbm":

        grayFrameL = cv2.cvtColor(frameL.copy(), cv2.COLOR_BGR2GRAY)
        grayFrameR = cv2.cvtColor(frameR.copy(), cv2.COLOR_BGR2GRAY)

        disparityMap = stereo.compute(
            grayFrameL, grayFrameR).astype(np.float32)/16

        # Use StereoSGBM to compute right disparity as well for WLS filtering
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        disparityMap_right = right_matcher.compute(
            grayFrameR, grayFrameL).astype(np.float32)/16

        # Apply WLS filtering
        filtered_disp = wls_filter.filter(
            disparityMap, grayFrameL, None, disparityMap_right)

        # Normalize the filtered disparity map
        filtered_disp_vis = cv2.normalize(
            src=filtered_disp, dst=None, beta=255, alpha=0, norm_type=cv2.NORM_MINMAX)
        # Convert to 8-bit image (necessary for equalization and color mapping)
        filtered_disp_vis = filtered_disp_vis.astype(np.uint8)
        # Apply histogram equalization to improve contrast
        filtered_disp_vis = cv2.equalizeHist(filtered_disp_vis)

        # Apply a color map to the equalized filtered disparity map
        filtered_disp_color = cv2.applyColorMap(
            filtered_disp_vis, cv2.COLORMAP_MAGMA)

        disparityMap_vis = cv2.normalize(
            src=disparityMap, dst=None, beta=255, alpha=0, norm_type=cv2.NORM_MINMAX)
        disparityMap_vis = cv2.equalizeHist(disparityMap_vis.astype(np.uint8))
        disparity_color = cv2.applyColorMap(
            disparityMap_vis, cv2.COLORMAP_MAGMA)

        color_disparity = filtered_disp_color.copy()

        depth_map = disparity_to_depth(filtered_disp)

    else:
        # Get the depth map from the Hitnet model
        disparity_map = hitnet_depth(frameL, frameR)
        color_disparity = draw_disparity(disparity_map)
        depth_map = hitnet_depth.get_depth()

    # End time
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Add inference time to the image
    text = f"Inference Time: {elapsed_time:.2f} ms"
    position = (10, 30)
    cv2.putText(color_disparity, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return color_disparity, depth_map


def depth_estimation_mouse_callback(event, x, y, flags, param):
    if param is None:
        return

    combined_image, depth_map = param

    if event == cv2.EVENT_LBUTTONDOWN:  # Left button clicked
        # Check if the click is within the boundaries of the depth estimation window
        # This assumes your depth estimation is on the right side.
        if x >= frameL.shape[1]:
            # Adjust the x-coordinate to match the depth map
            adjusted_x = x - frameL.shape[1]

            depth_value = depth_map[y, adjusted_x]
            print(f"Depth at ({adjusted_x}, {y}) is: {depth_value}")

            # Draw the depth value on the depth estimation image
            text = f"Depth: {depth_value:.2f}"
            cv2.putText(frameR_with_detections, text, (adjusted_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update the combined image to reflect the new annotation
            top_combined = np.hstack(
                (frameL_with_detections, frameR_with_detections))
            bottom_combined = np.hstack((frameL, frameR))
            combined = np.vstack((top_combined, bottom_combined))

            # Refresh the display
            cv2.imshow(window_name, combined)


def overlay_guidelines(frameL):
    guidelines = [
        ("'o' - Object Detection", (10, frameL.shape[0] - 150)),
        ("'d' - Depth Estimation", (10, frameL.shape[0] - 120)),
        ("'b' - Both", (10, frameL.shape[0] - 90)),
        ("'s' - Swap Cameras", (10, frameL.shape[0] - 60)),
        ("'x' - Exit", (10, frameL.shape[0] - 30))
    ]

    for text, position in guidelines:
        cv2.putText(frameL, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frameL


def overlay_depth_on_detections(depth_frame, disparity_map, detections):

    updated_detections = []

    for detection in detections:
        # Assume each detection has 'bbox' for bounding box [x1, y1, x2, y2] and you want the central point
        x_center = detection[1].astype(np.int32)
        y_center = detection[2].astype(np.int32)

        # Extract depth value from disparity map
        # Be sure to handle cases where the disparity map might not have a valid value (e.g., -inf, inf, nan)
        depth_value = disparity_map[y_center, x_center]

        # Overlay the depth value onto the depth frame
        cv2.putText(depth_frame,
                    f"Depth: {depth_value:.2f}",
                    (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red color
                    3)

        updated_detections.append(
            [detection[0], detection[1], detection[2], depth_value])

    return depth_frame, updated_detections


def object_detection_mouse_callback(event, x, y, flags, param):
    # Check for left mouse button click event
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: x = {x}, y = {y}")


def generate_grasping_overlay(frame, total_objects, current_object, status, coords, object_class):

    overlay_frame = frame.copy()

    # Define the list of texts and their positions
    info_texts = [
        (f"Objects to grasp: {total_objects}", (10, frame.shape[0] - 150)),
        (f"Grasping object {current_object} of {total_objects}",
         (10, frame.shape[0] - 120)),
        (status, (10, frame.shape[0] - 90)),
        (f"Coord: ({coords[0]:.2f}, {coords[1]:.2f})",
         (10, frame.shape[0] - 60)),
        (f"Grasping {object_class}", (10, frame.shape[0] - 30))
    ]

    # Loop over each text and its position, then overlay it on the frame
    for text, position in info_texts:
        cv2.putText(overlay_frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return overlay_frame


def aruco_detection(frame, aruco_dictionary, aruco_parameters):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, aruco_dictionary, parameters=aruco_parameters)

    center_coordinates = []

    if len(corners) > 0:

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for corner in corners:
            x_center = int((corner[0][0][0] + corner[0][1]
                           [0] + corner[0][2][0] + corner[0][3][0]) / 4)
            y_center = int((corner[0][0][1] + corner[0][1]
                           [1] + corner[0][2][1] + corner[0][3][1]) / 4)

            center_coordinates.append((x_center, y_center))
            # Drawing the center point
            # 5 is the radius, (0, 255, 0) is the color
            cv2.circle(frame, (x_center, y_center), 10, (0, 255, 0), -1)

    return frame, center_coordinates

# HOT FIX

def check_black_scene(disp_map_img):

    # Assuming 'filtered_disp_vis' is the normalized disparity map
    # Compute the histogram
    hist = cv2.calcHist([disp_map_img], [0], None, [256], [0, 256])

    # Threshold for black (can be adjusted)
    black_threshold = 5

    # Count black pixels
    black_pixels = hist[0:black_threshold+1].sum()

    # Calculate the percentage of black pixels
    total_pixels = disp_map_img.size
    black_percentage = (black_pixels / total_pixels) * 100

    print(f"Percentage of black pixels: {black_percentage}%")

    # Decision
    if black_percentage > 90:  # This threshold can be adjusted
        print("The image is predominantly black.")
        return True
    else:
        print("The image is not predominantly black.")
        return False

    return


if __name__ == '__main__':

    hitnet_depth = init_hitnet()
    stereo_camera = StereoCamera()

    print("Init Object Detection Model")
    object_detection_model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    object_detection_model.eval()

    print("Init COBOT")
    cobot = Cobot("/dev/tty.usbserial-54790107521")

    print("Init Aruco")
    # Load the predefined dictionary
    aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    print("Init SGBM")
    stereo_sgbm, wls_filter = init_sgbm()

    # Initialize the detector parameters using default values
    aruco_parameters = cv2.aruco.DetectorParameters_create()

    checking_process = True

    # Define display window name
    window_name = "Robot Grasping"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frameL, frameR = stereo_camera.get_frames()

    frameL_with_detections = None
    frameR_with_detections = None

    # Set the size of the window
    heightL, widthL, channelsL = frameL.shape
    heightR, widthR, channelsR = frameR.shape

    combined_width = widthL + widthR
    combined_height = heightL + heightR

    cv2.resizeWindow(window_name, combined_width, combined_height)

    while checking_process:

        # Get frames from camera
        frameL, frameR = stereo_camera.get_frames()
        frameL = overlay_guidelines(frameL)

        add_text(frameR, "Cam Right", (frameR.shape[1] // 4, 30))
        add_text(frameL, "Cam Left", (frameL.shape[1] // 4, 30))

        # Concatenate the frames
        if frameL_with_detections is not None and frameR_with_detections is not None:
            top_combined = np.hstack(
                (frameL_with_detections, frameR_with_detections))
        elif frameR_with_detections is not None:
            top_combined = np.hstack((frameL, frameR_with_detections))
        elif frameL_with_detections is not None:
            top_combined = np.hstack((frameL_with_detections, frameR))
        else:
            top_combined = np.hstack((frameL, frameR))

        bottom_combined = np.hstack((frameL, frameR))
        combined = np.vstack((top_combined, bottom_combined))

        # Display the combined frame
        cv2.imshow(window_name, combined)

        # Start the event loop
        key = cv2.waitKey(40) & 0xFF

        # Loop control
        if key == ord(' '):
            checking_process = False
            robot_process = True
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        elif key == ord('x'):
            exit()

        elif key == ord('s'):
            stereo_camera.swap_cameras()

        # OBJECT DETECTION ONLY
        elif key == ord('o'):
            frameL_with_detections, object_detection_result = object_detection(
                object_detection_model, frameL)

        # DEPTH ESTIMATION ONLY
        # HOT FIX
        elif key == ord('d'):

            frameR_with_detections, disparity_map = depth_estimation(
                hitnet_depth, stereo_sgbm, wls_filter, frameL, frameR)

            # check_bug = True
            # count_black = 0
            # while check_bug:
            #     frameR_with_detections, disparity_map = depth_estimation(
            #         hitnet_depth, stereo_sgbm, wls_filter, frameL, frameR)
            #     count_black += 1
            #     check_bug = check_black_scene(disparity_map)
            #     if count_black > 10:
            #         check_bug = False

            cv2.setMouseCallback(
                window_name, depth_estimation_mouse_callback, param=(combined, disparity_map))

        # BOTH
        elif key == ord('b'):
            frameL_with_detections, object_detection_result = object_detection(
                object_detection_model, frameL)
            frameR_with_detections, disparity_map = depth_estimation(
                hitnet_depth, stereo_sgbm, wls_filter, frameL, frameR)
            cv2.setMouseCallback(
                window_name, depth_estimation_mouse_callback, param=(combined, disparity_map))

    # Robot Process
    while robot_process:

        # Define display window name
        window_name = "Robot Process"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, object_detection_mouse_callback)

        frameL, frameR = stereo_camera.get_frames()

        object_detection_frame, object_detection_result = object_detection(
            object_detection_model, frameL)
        depth_estimation_frame, disparity_map = depth_estimation(
            hitnet_depth, stereo_sgbm, wls_filter, frameL, frameR)

        # Overlay the depth on detections
        depth_estimation_frame, object_detection_result = overlay_depth_on_detections(
            depth_estimation_frame, disparity_map, object_detection_result)

        # Aruco
        aruco_result, aruco_coord = aruco_detection(
            object_detection_frame, aruco_dictionary, aruco_parameters)
            
        if len(aruco_coord) == 2:
            cobot.set_aruco_coord(aruco_coord)

        print(f"aruco_coord: {aruco_coord}")

        combined = np.vstack((aruco_result, depth_estimation_frame))

        cv2.imshow(window_name, combined)
        cv2.waitKey(0)

        robot_process = False
        # Number of objects to grasp
        total_objects = len(object_detection_result)

        for index, detection in enumerate(object_detection_result):

            print("Grasping object {} of {} : {}".format(
                index+1, total_objects, detection[0]))

            status = "Grasping"
            overlay = generate_grasping_overlay(
                object_detection_frame, total_objects, index+1, status, (detection[1], detection[2]), detection[0])
            combined = np.vstack((overlay, depth_estimation_frame))

            cv2.imshow(window_name, combined)
            cv2.waitKey(1)

            # Grasp the object using the robot
            cobot.grab_object(detection[1], detection[2], detection[3])

            # Update the status to "Done" after grasping
            status = "Putting in Bin"
            overlay = generate_grasping_overlay(
                object_detection_frame, total_objects, index+1, status, (detection[1], detection[2]), detection[0])
            combined = np.vstack((overlay, depth_estimation_frame))

            cv2.imshow(window_name, combined)
            cv2.waitKey(1)

            if detection[0] == "cat":
                cobot.put_off(position="box_left_middle")
            elif detection[0] == "clock":
                cobot.put_off(position="box_right_middle")
            elif detection[0] == "apple":
                cobot.put_off(position="box_right_back")
            else:
                cobot.put_off(position="box_left_back")

            # Update the status to "Done" after grasping
            status = "Done"
            overlay = generate_grasping_overlay(
                object_detection_frame, total_objects, index+1, status, (detection[1], detection[2]), detection[0])
            combined = np.vstack((overlay, depth_estimation_frame))
            cv2.imshow(window_name, combined)
            cv2.waitKey(1)

        cobot.init()
