import cv2
import time

def capture_on_spacebar(cam1_id=0, cam2_id=2, base_path1='./calibration_pic/right/right_', base_path2='./calibration_pic/left/left_', ext='.jpg'):
    
    ## FOR CAMERA FINDING
    target_resolution = (1920, 1080)
    opened_cams = []
    found_indices = []

    # Open the cameras
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

    camL, camR = opened_cams
    counter = 0

    while True:

        camL.grab()
        camR.grab()

        # Read from the cameras
        ret1, frame1 = camL.retrieve()
        ret2, frame2 = camR.retrieve()

        # Display the frames
        cv2.imshow('Camera 1 RIGHT', frame1)
        cv2.imshow('Camera 2 LEFT', frame2)

        key = cv2.waitKey(1) & 0xFF

        # If spacebar is pressed, save the frames
        if key == ord(' '):
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            save_path1 = base_path1 + timestamp + f'_{counter}' + ext
            save_path2 = base_path2 + timestamp + f'_{counter}' + ext
            cv2.imwrite(save_path1, frame1)
            cv2.imwrite(save_path2, frame2)
            print(f"Images saved as {save_path1} and {save_path2}")
            counter += 1

        # If 'ESC' key is pressed, close the windows and exit
        elif key == 27:
            break

    # Close the windows and cameras
    cv2.destroyAllWindows()
    camL.release()
    camR.release()

if __name__ == '__main__':
    capture_on_spacebar()
