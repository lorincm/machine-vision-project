import cv2
import numpy as np

def get_marker_center(corners):
    """Compute the center of a marker given its corners."""
    x_center = int((corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4)
    y_center = int((corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4)
    return x_center, y_center

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

image = cv2.imread('./images/main.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

center_coordinates = []

if len(corners) > 0:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    for corner in corners:
        x, y = get_marker_center(corner[0])
        center_coordinates.append((x, y))
        
        # Drawing the center point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 5 is the radius, (0, 255, 0) is the color

cv2.imshow('Image with detected ArUco markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(center_coordinates)  # This will print the coordinates of the centers of the detected markers
