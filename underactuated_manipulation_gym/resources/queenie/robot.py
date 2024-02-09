import pybullet as p
import os
import math
import numpy as np
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, parent_directory)
from .robot_sensors.camera import Camera_Sensor
from .robot_sensors.proprioception import Proprioception_Sensor

class QueenieRobot():

    def __init__(self, client, config, for_meta_env=False) -> None:
        self.client = client
        self._config = config

        f_name = os.path.join(os.path.dirname(__file__), "urdfs/queenie_pb.urdf")
        self.robot = p.loadURDF(f_name, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.client)

        self._setup_joint_motors()
        # self._setup_gear_constraints()
    """
    Returns the client and robot ids
    """
    def get_ids(self):
        return self.client, self.robot
    
    def get_robot(self):
        return self.robot
    
    def _setup_joint_motors(self):
        # set each joint to velocity control with 0 target velocity and 0 force
        for joint in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(self.robot, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            print(p.getJointInfo(self.robot, joint))
        
        for joint in self._config["parameters"]["joint_params"].keys():
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, targetPosition=self._config["parameters"]["joint_params"][joint]["init"])
        for joint in self._config["parameters"]["gripper"]["joints"]:
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, targetPosition=self._config["parameters"]["gripper"]["init"])
        # p.setJointMotorControl2(self.robot, 1, p.POSITION_CONTROL, targetPosition=0)
        # p.setJointMotorControl2(self.robot, 2, p.POSITION_CONTROL, targetPosition=0)
        # p.setJointMotorControl2(self.robot, 4, p.POSITION_CONTROL, targetPosition=0.5, force=20)
        # p.setJointMotorControl2(self.robot, 5, p.POSITION_CONTROL, targetPosition=0.5, force=20)
        return
        
    def _setup_gear_constraints(self):
        front_left_wheel = 8
        rear_left_wheel = 10
        front_right_wheel = 9
        rear_right_wheel = 11
        # Set up gear constraints so that wheels on the same side rotate at the same speed:
        c1 = p.createConstraint(self.robot, front_left_wheel, self.robot, rear_left_wheel,
                                    jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c1, gearRatio=1, maxForce=100)

        c2 = p.createConstraint(self.robot, front_right_wheel, self.robot, rear_right_wheel,
                                    jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c2, gearRatio=1, maxForce=100)

        c = p.createConstraint(self.robot, 4, self.robot, 5,
                            jointType=p.JOINT_GEAR,
                            jointAxis=[0, 1, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, maxForce=1000)

        return






if __name__ == "__main__":
    qr = QueenieRobot(None)
