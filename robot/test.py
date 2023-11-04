from pymycobot.mycobot import MyCobot
import time
Half_Block_Size=20

mc = MyCobot('/dev/tty.usbserial-56230104151',115200)
POS_LIST = {            
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
        [[200, 120, 230, -180, 0,0],      # up
        [215, 150, 220, -150, 0, -60]],   # down
        
    "box_right_middle":
        [[120, -150, 240, -180, 0, 0],      # up
        [120, -150, 175, -180, 0, 0]],   # down
    
    "box_right_back":
        [[200, -100, 235, -180, 0,0],      # up
        [220, -140, 220, -150, -10, -80]],   # down
}
MODE = 0


def pump_on():
    mc.set_basic_output(2, 0)
    mc.set_basic_output(5, 0)

def pump_off():
    mc.set_basic_output(2, 1)
    mc.set_basic_output(5, 1)
    
def init():
    mc.send_angles([0, 0, 0, 0, 0, 0], 40) 
    time.sleep(3)
    
def grasp(position):
    pump_on()
    mc.send_coords(POS_LIST[position][0], 40, MODE)
    time.sleep(4)
    mc.send_coords(POS_LIST[position][1], 40, MODE)
    time.sleep(6)
    mc.send_coords(POS_LIST[position][0], 40, MODE)
    time.sleep(2)

def put_off(position):
    mc.send_coords(POS_LIST[position][0], 40, MODE)
    time.sleep(2)
    mc.send_coords(POS_LIST[position][1], 40, MODE)
    time.sleep(2)
    pump_off()
    time.sleep(4)
    mc.send_coords(POS_LIST[position][0], 40, MODE)
    time.sleep(2)
    
################begins#################
init()
grasp(position="center")
put_off(position="box_left_middle")
grasp(position="left_back_corner")
put_off(position="box_right_middle")
grasp(position="right_back_corner")
put_off(position="box_left_back")
init()
################ends###################




