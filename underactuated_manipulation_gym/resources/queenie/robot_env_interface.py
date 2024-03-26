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
from .robot_sensors.point_cloud import PointCloudSensor
from .robot import QueenieRobot

class QueenieRobotEnvInterface():

    def __init__(self, client, config, robot_object=None, object_handler=None) -> None:
        self.client = client
        self._config = config
        
        if robot_object is None:
            robot_object = QueenieRobot(self.client, self._config)
        self.robot = robot_object.get_robot()

        self.camera_link_index = self._get_link_index("camera")
        self.palm_link_index = self._get_link_index("palm")
        self.left_finger_link_index = self._get_link_index("left_finger")
        self.right_finger_link_index = self._get_link_index("right_finger")

        self.object_handler = object_handler

        self._setup_sensors()
    """
    Returns the client and robot ids
    """
    def get_ids(self):
        return self.client, self.robot
    
    def get_base_pose(self):
        return p.getBasePositionAndOrientation(self.robot, self.client)
    
    def get_wheel_positions(self):
        return [p.getLinkState(self.robot, i, self.client)[0] for i in [8, 9]]
    """
    Resets the robot to the given position and orientation
    """
    def reset(self, position, orientation):
        # set wheel velocity to zero
        for joint in self._config["actuators"]["wheel_joints"]:
            p.setJointMotorControl2(self.robot, joint, p.VELOCITY_CONTROL, targetVelocity=0)
        # set joint positions to initial positions
        for joint in self._config["actuators"]["joints"]:
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, targetPosition=self._config["parameters"]["joint_params"][joint]["init"])
        # set gripper to initial position
        if self._config["actuators"]["gripper"]:
            for joint in self._config["parameters"]["gripper"]["joints"]:
                p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, targetPosition=self._config["parameters"]["gripper"]["init"])
        p.resetBasePositionAndOrientation(self.robot, position, orientation)
    
    def apply_action(self, action, use_gripper=False):
        # we do not know the size of action, but we know first two are linear and angular velocity
        # this is followed by actuated neck joints and gripper (if gripper is enabled)
        v, w_angular = action[:2]
        # scale the linear and angular velocity
        v = v * self._config["parameters"]["max_linear_velocity"]
        w_angular = w_angular * self._config["parameters"]["max_angular_velocity"]
        v_left = v - w_angular * 0.6 / 2
        v_right = v + w_angular * 0.6 / 2
        # wheels
        p.setJointMotorControl2(self.robot, 8, p.VELOCITY_CONTROL, targetVelocity=v_left)
        p.setJointMotorControl2(self.robot, 9, p.VELOCITY_CONTROL, targetVelocity=v_right)
        p.setJointMotorControl2(self.robot, 10, p.VELOCITY_CONTROL, targetVelocity=v_left)
        p.setJointMotorControl2(self.robot, 11, p.VELOCITY_CONTROL, targetVelocity=v_right)
        
        # actions predicted by network are between -1 and 1
        a = -1
        b = 1
        for i, joint in enumerate(self._config["actuators"]["joints"]):
            c = self._config["parameters"]["joint_params"][joint]["min"]
            d = self._config["parameters"]["joint_params"][joint]["max"]
            # formula for converting from one range to another
            pos = c + ((d - c)*(action[i+2] - a)/(b - a))
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, targetPosition=pos)
        
        # gripper
        if self._config["actuators"]["gripper"] or use_gripper:
            position = self._config["parameters"]["gripper"]["max"] if action[-1] > 0 else self._config["parameters"]["gripper"]["min"]
            for joint in self._config["parameters"]["gripper"]["joints"]:
                if self._config["parameters"]["gripper"]["use_dynamic_force"]:
                    force = action[-2]
                else:
                    force = self._config["parameters"]["gripper"]["force"]
                p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, targetPosition=position, force=force)
            # p.setJointMotorControl2(self.robot, 4, p.POSITION_CONTROL, targetPosition=position)
            # p.setJointMotorControl2(self.robot, 5, p.POSITION_CONTROL, targetPosition=position)


    """
    Applies the action to the robot
    """
    def apply_action_old(self, action):
        v_left, v_right, neck_pos, neck_x_pos, gripper = action
        # wheels
        p.setJointMotorControl2(self.robot, 8, p.VELOCITY_CONTROL, targetVelocity=v_left)
        p.setJointMotorControl2(self.robot, 9, p.VELOCITY_CONTROL, targetVelocity=v_right)
        p.setJointMotorControl2(self.robot, 10, p.VELOCITY_CONTROL, targetVelocity=v_left)
        p.setJointMotorControl2(self.robot, 11, p.VELOCITY_CONTROL, targetVelocity=v_right)

        # neck
        p.setJointMotorControl2(self.robot, 1, p.POSITION_CONTROL, targetPosition=neck_pos)
        p.setJointMotorControl2(self.robot, 2, p.POSITION_CONTROL, targetPosition=neck_x_pos)

        # gripper
        position = 0.1 if gripper > 0 else 0
        p.setJointMotorControl2(self.robot, 4, p.POSITION_CONTROL, targetPosition=position, force=20)
        p.setJointMotorControl2(self.robot, 5, p.POSITION_CONTROL, targetPosition=position, force=20)

    def get_observation_space_size(self):
        return {"image": self.sensors["camera"].get_observation_space_size(), "vector": self.sensors["proprioception"].get_observation_space_size()}
    
    """
    Returns the observation of the robot in a dictionary format with
    separate keys for camera, joint states, and contact points
    """
    def get_state(self):
            observation = {}
            for sensor_name, sensor in self.sensors.items():
                if sensor_name == "proprioception":
                    observation[sensor_name], indices = sensor.get_observation()
                    observation[sensor_name + "_indices"] = indices
                else:
                    observation[sensor_name] = sensor.get_observation()
            # image_obs = self.sensors["camera"].get_observation()
            # proprioception, indices = self.sensors["proprioception"].get_observation()
            # proprioception = np.array(proprioception, dtype=np.float32)

            # Construct observation dictionary
            # observation = {
            #     "image_obs": image_obs,
            #     "proprioception": proprioception,
            #     "proprioception_indices": indices
            # }
            return observation


    def _setup_sensors(self):
        self.sensors = {}
        for sensor_name, sensor_params in self._config["sensors"].items():
            if sensor_name == "camera":
                self.sensors[sensor_name] = Camera_Sensor(self.client, self.robot, sensor_name, sensor_params, self._config["parameters"])
            elif sensor_name == "proprioception":
                self.sensors[sensor_name] = Proprioception_Sensor(self.client, self.robot, sensor_name, sensor_params, self._config["parameters"])
            elif sensor_name == "point_cloud":
                self.sensors[sensor_name] = PointCloudSensor(self.client, self.robot, sensor_name, sensor_params, self._config["parameters"], self.sensors["camera"], object_handler=self.object_handler)
            else:
                raise NotImplementedError(f"Sensor {sensor_name} not implemented")
        return

    def _get_link_index(self, link_name):
        for joint in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, joint)
            if info[12].decode('utf-8') == link_name:
                return joint
        return -1  # Link not found
    

    def rotate_vector(self, vector, theta):
        """Rotates 2D vector by theta degrees"""
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return rotation_matrix.dot(vector)
    
    def get_action_space_size(self):
        # 2 incldues linear and angular velocity
        len_action_space = 2 + len(self._config["actuators"]["joints"]) + int(self._config["actuators"]["gripper"])
        return len_action_space
    
    def render_camera_image(self):
        return self.sensors["camera"].get_render_images()






if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    sys.path.insert(0, parent_directory)
    qr = QueenieRobotEnvInterface(None)
