import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2

from underactuated_manipulation_gym.resources.queenie import Queenie_Robot
from underactuated_manipulation_gym.resources.plane import Plane
from underactuated_manipulation_gym.resources.man_object import ObjectMan

class DifferentialDriveEnv(gym.Env):
    def __init__(self):
        super(DifferentialDriveEnv, self).__init__()

        self.client = p.connect(p.GUI)
        p.setGravity(0,0,-10)
        p.setRealTimeSimulation(0)
        self.robot = Queenie_Robot(self.client)
        self.plane = Plane(self.client)
        self.object = ObjectMan(self.client)

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.step_i = 0


    def reset(self):
        # Reset the environment to its initial state
        self.step_i = 0
        pos = [0, 0, 0.4]
        orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot.reset(pos, orn)
        for _ in range(100):
            p.stepSimulation()
        # Return the initial observation
        return self._get_observation()

    def step(self, action):
        # Extract the action components
        v, w_angular, neck_pos, neck_x_pos = action

        v_left = v - w_angular * 0.6 / 2
        v_right = v + w_angular * 0.6 / 2

        action = [v_left, v_right, neck_pos, neck_x_pos]

        self.robot.apply_action(action)

        
        for i in range(50):
            p.stepSimulation()

        # Get the new observation after taking the action
        observation = self._get_observation()

        # Define your reward and done criteria
        reward = self._reward(observation, action)
        done = self.step_i >=1000
        self.step_i += 1

        return observation, reward, done, False,{}
    
    def _reward(self, observation, action):
        # Define your reward function
        return -2

    def _get_observation(self):
        robot_state = self.robot.get_observation()
        # convert robot_state to environment_state (i.e. observation)
        queenie_pos, queenie_orn = robot_state["base_pose"]
        contact_points = robot_state["contact_points"]
        print(contact_points)
        try:
            camera_img = robot_state["camera_rgb"]
            cv2.imwrite("test.png", camera_img)
        except:
            print("no image")
            print(robot_state)

        x, y = queenie_pos[0:2]
        return np.array([x, y])


    def render(self, mode='human'):
        # If you want to visualize the robot's behavior, you can implement this method
        pass

    def close(self):
        # Disconnect from PyBullet
        p.disconnect()

    def seed(self, seed=None):
        # Set the random seed for reproducibility
        pass

    def _parse_config(self):
        pass

    def _get_observation_space(self):
        # Define the observation space
        return spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    
    """
    Define the action space
    """
    def _get_action_space(self):
        min_linear_vel = -2
        max_linear_vel = 2
        min_angular_vel = -2
        max_angular_vel = 2
        min_neck_joint = -1
        max_neck_joint = 0.21
        min_neck_joint = -0.5
        max_neck_joint = 0.5
        min_action = [min_linear_vel, min_angular_vel, min_neck_joint, min_neck_joint]
        max_action = [max_linear_vel, max_angular_vel, max_neck_joint, max_neck_joint]
        return spaces.Box(low=np.array(min_action), high=np.array(max_action), dtype=np.float32)