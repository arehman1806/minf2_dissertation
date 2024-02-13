import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import yaml

from underactuated_manipulation_gym.resources.queenie.robot_env_interface import QueenieRobotEnvInterface
from underactuated_manipulation_gym.resources.queenie.robot import QueenieRobot
from underactuated_manipulation_gym.resources.plane import Plane
from underactuated_manipulation_gym.resources.objects.object_loader import ObjectLoader
from underactuated_manipulation_gym.resources.target import Target

class BaseEnvironment(gym.Env):
    def __init__(self, config_file):
        super(BaseEnvironment, self).__init__()
        self._config = self._parse_config(config_file)
        self._robot_config = self._config["robot"]
        self._environment_config = self._config["environment"]
    

    def reset(self, seed=None):
        # Reset the environment to its initial state
        self.step_i = 0
        pos = [0, 0, 0.4]
        orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot.reset(pos, orn)
        self.current_object = self.object_loader.change_object()
        self.target.reset_position(None)
        for _ in range(100):
            p.stepSimulation()
        # Return the initial observation
        return self._get_observation()[0], {}

    def step(self, action):

        # action = self._calculate_action(action)

        self.robot.apply_action(action)

        self.current_object = self.object_loader.get_current_object()

        
        for i in range(50):
            p.stepSimulation()

        # Get the new observation after taking the action
        observation, proprioception_indices = self._get_observation()

        # Define your reward and done criteria
        reward, done = self._reward(observation, proprioception_indices, action)
        done = done or self.step_i >= self._episode_length
        if done:
            self.previous_distance = None
        self.step_i += 1

        return observation, reward, done, False,{}
    
    def _reward(self, observation, proprioception_indices, action):
        raise NotImplementedError
    
    def _calculate_robot_object_distance(self):
        object_id = self.current_object.get_ids()[1]
        robot_id = self.robot.get_ids()[1]
        object_link_state = p.getBasePositionAndOrientation(object_id)[0]
        robot_link_state = p.getLinkState(robot_id, 3)[0]
        distance = ((object_link_state[0] - robot_link_state[0]) ** 2 +
                    (object_link_state[1] - robot_link_state[1]) ** 2 )** 0.5
        
        return distance
    
    def _calculate_object_target_distance(self):
        object_id = self.current_object.get_ids()[1]
        object_link_state = p.getBasePositionAndOrientation(object_id)[0]
        target_link_state = self.target.get_base_position()
        distance = ((object_link_state[0] - target_link_state[0]) ** 2 +
                    (object_link_state[1] - target_link_state[1]) ** 2 )** 0.5
        
        return distance

    def _get_observation(self):
        raise NotImplementedError
    
    def _calculate_action(self, action):
        raise NotImplementedError


    def cartesian_to_polar_2d(self, x_target, y_target, x_origin = 0, y_origin = 0):
        """Transform 2D cartesian coordinates to 2D polar coordinates.

        Args:
            x_target (type): x coordinate of target point.
            y_target (type): y coordinate of target point.
            x_origin (type): x coordinate of origin of polar system. Defaults to 0.
            y_origin (type): y coordinate of origin of polar system. Defaults to 0.

        Returns:
            float, float: r,theta polard coordinates.

        """

        delta_x = x_target - x_origin
        delta_y = y_target - y_origin
        polar_r = np.sqrt(delta_x**2+delta_y**2)
        polar_theta = np.arctan2(delta_y,delta_x)

        return polar_r, polar_theta
    
    def render(self, mode='human'):
        # If you want to visualize the robot's behavior, you can implement this method
        pass

    def close(self):
        self.object_loader.empty_scene()
        # Disconnect from PyBullet
        p.disconnect()

    def seed(self, seed=None):
        # Set the random seed for reproducibility
        pass

    def get_robot(self):
        return self.robot

    """loads yaml config file and returns a dictionary"""
    def _parse_config(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
        

    def _get_observation_space(self):
        raise NotImplementedError
    
    """
    Define the action space
    """
    def _get_action_space(self):

        # Define the action space
        # use the robot parameters from self.robot_params to define the action space
        len_action_space = self.robot.get_action_space_size()
        
        # the actions will always be normalised, so the action space is always between -1 and 1
        min_action = np.full(len_action_space, -1)
        max_action = np.full(len_action_space, 1)
        return spaces.Box(low=min_action, high=max_action, dtype=np.float32)