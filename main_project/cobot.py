from pymycobot.mycobot import MyCobot
import time


class Cobot():

    def __init__(self, id):

        self.id = id
        self.mc = MyCobot(id, 115200)
        self.MODE = 0
        self.aruco_coord = [(879, 936), (1289, 534)]

        self.POS_LIST = {
            "left_back_corner":
                [[225, 75, 230, -180, 0, 0],      # up
                 [225, 75, 100, -180, 0, 0]],   # down

            "right_back_corner":
                [[225, -75, 230, -180, 0, 0],      # up
                 [225, -75, 100, -180, 0, 0]],   # down

            "center":
                [[150, 0, 230, -180, 0, 0],      # up
                 [150, 0, 100, -180, 0, 0]],   # down

            "box_left_middle":
                [[120, 170, 240, -180, 0, 0],      # up
                 [120, 170, 175, -180, 0, 0]],   # down

            "box_left_back":
                [[215, 120, 250, -180, 0, 0],      # up
                 [225, 150, 220, -150, 0, -60]],   # down

            "box_right_middle":
                [[120, -150, 240, -180, 0, 0],      # up
                 [120, -150, 175, -180, 0, 0]],   # down

            "box_right_back":
                [[200, -100, 235, -180, 0, 0],      # up
                 [220, -140, 220, -150, -10, -80]],   # down
        }

        self.init()
    
    def set_aruco_coord(self,aruco_coord):
        self.aruco_coord = aruco_coord

    def pump_on(self):
        self.mc.set_basic_output(2, 0)
        self.mc.set_basic_output(5, 0)

    def pump_off(self):
        self.mc.set_basic_output(2, 1)
        self.mc.set_basic_output(5, 1)

    def init(self):
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 40)
        time.sleep(3)
        self.pump_off()

    def grasp(self,position):
        self.pump_on()
        self.mc.send_coords(self.POS_LIST[position][0], 40, self.MODE)
        time.sleep(4)
        self.mc.send_coords(self.POS_LIST[position][1], 40, self.MODE)
        time.sleep(6)
        self.mc.send_coords(self.POS_LIST[position][0], 40, self.MODE)
        time.sleep(2)

    def put_off(self,position):
        self.mc.send_coords(self.POS_LIST[position][0], 40, self.MODE)
        time.sleep(2)
        self.mc.send_coords(self.POS_LIST[position][1], 40, self.MODE)
        time.sleep(2)
        self.pump_off()
        time.sleep(4)
        self.mc.send_coords(self.POS_LIST[position][0], 40, self.MODE)
        time.sleep(2)

    def map_camera_to_robot(self,x_cam, y_cam):
        # Robot bounds
        robot_left, robot_right = 75, -25
        robot_bottom, robot_top = 100, 210
        
        ## OPPOSITE
        # robot_left, robot_right = -75, 75
        # robot_bottom, robot_top = 225, 0

        # Camera ROI bounds
        camera_left, camera_right = self.aruco_coord[0][0], self.aruco_coord[1][0]
        camera_top, camera_bottom = self.aruco_coord[1][1], self.aruco_coord[0][1]

        # Calculate scaling factors
        scale_x = (robot_right - robot_left) / (camera_right - camera_left)
        scale_y = (robot_bottom - robot_top) / (camera_bottom - camera_top)

        # Calculate translations
        trans_x = robot_left - camera_left * scale_x
        trans_y = robot_top - camera_top * scale_y

        # Transform the input coordinates
        x_rob = x_cam * scale_x + trans_x
        y_rob = y_cam * scale_y + trans_y

        print(f"Camera coordinates: ({x_cam}, {y_cam}) -> Robot coordinates: ({x_rob}, {y_rob})")

        return x_rob, y_rob


    def grab_object(self,x_camera,y_camera):

        x_robot,y_robot = self.map_camera_to_robot(x_camera,y_camera)

        ## GRASPING
        print(f"Grasping object  at ({x_robot}, {y_robot})")
        self.pump_on()
        print("UP")
        self.mc.send_coords([y_robot, x_robot, 200, -180, 0, 0], 40, self.MODE)
        time.sleep(4)
        print("DOWN")
        self.mc.send_coords([y_robot, x_robot, 100, -180, 0, 0], 40, self.MODE)
        time.sleep(6)
        print("UP")
        self.mc.send_coords([y_robot, x_robot, 240, -180, 0, 0], 40, self.MODE)
        time.sleep(2)

        
