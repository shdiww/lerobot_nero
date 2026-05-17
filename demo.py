from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
import time
import os
robot_cfg = create_agx_arm_config(robot=ArmModel.NERO,
                                   firmeware_version=NeroFW.DEFAULT, 
                                   channel="can0")
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()

# while not robot.enable():
#     robot.set_normal_mode()
#     print(robot.get_joint_angles())
#     time.sleep(0.01)
def main():
    print("nero demo start")
    ret= os.system("sudo ip link set can0 up type can bitrate 1000000 2>/dev/null")
    if ret == 0:
        print("  CAN0 ready (socketcan, 1Mbps)")
    else:
        print("  CAN0 setup may have failed, continuing anyway...")

    robot_cfg = create_agx_arm_config(robot=ArmModel.NERO,
                                   firmeware_version=NeroFW.DEFAULT, 
                                   channel="can0")
    robot = AgxArmFactory.create_arm(robot_cfg)
    print("Robot created successfully")
    robot.connect()
    