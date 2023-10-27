import cv2
import time

def capture_on_spacebar(cam1_id=0, cam2_id=2, base_path1='./calibration_pic/right/right_', base_path2='./calibration_pic/left/left_', ext='.jpg'):
    # Open the cameras
    cam1 = cv2.VideoCapture(cam1_id) 
    cam2 = cv2.VideoCapture(cam2_id) 


    # Check if the cameras are opened successfully
    if not (cam1.isOpened() and cam2.isOpened()):
        print("Error: Unable to open cameras.")
        return

    counter = 0

    while True:
        # Read from the cameras
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

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
    cam1.release()
    cam2.release()

if __name__ == '__main__':
    capture_on_spacebar()
