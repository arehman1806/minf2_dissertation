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

from underactuated_manipulation_gym.envs.base_environment import BaseEnvironment

class BaseOptionEnvironment(BaseEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        if config_file is None:
            raise Exception("No config file provided")
        super(BaseOptionEnvironment, self).__init__()
        config = self._parse_config(config_file)
        self.robot_config = config["robot"]
        self.environment_config = config["environment"]

        if not as_subpolicy:
            render_gui = self.environment_config["gui"]
            connection_mode = p.GUI if render_gui else p.DIRECT
            self.client = p.connect(connection_mode)
            p.setGravity(0,0,-10)
            p.setRealTimeSimulation(0)
            # p.setTimeStep(1./500

            robot_object = QueenieRobot(self.client, self.robot_config)
            self.robot = QueenieRobotEnvInterface(self.client, self.robot_config, robot_object)
            self.plane = Plane(self.client)
            self.object_loader = ObjectLoader(self.client, "random_urdfs", 
                                            num_objects=self.environment_config["num_objects"], 
                                            specific_objects=self.environment_config["specific_objects"],
                                            global_scale=self.environment_config["global_scale"])
            self.current_object = self.object_loader.change_object()
            self.target = Target(self.client, ([5,5,0.0], [0,0,0,1]))
        else:
            self.robot = controllers["robot"]
            self.plane = controllers["plane"]
            self.object_loader = controllers["object_loader"]
            self.target = controllers["target"]
            self.current_object = self.object_loader.get_current_object()

        self._episode_length = self.environment_config["episode_length"]
        
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.step_i = 0

        self.previous_distance = None
    
    def _reward(self, observation, proprioception_indices, action):
        raise NotImplementedError
    

    def _get_observation(self):
        raise NotImplementedError
    
    def _calculate_action(self, action):
        raise NotImplementedError
    
    def render(self, mode='human'):
        # If you want to visualize the robot's behavior, you can implement this method
        pass

    def seed(self, seed=None):
        # Set the random seed for reproducibility
        pass

    def get_robot(self):
        return self.robot
        

    def _get_observation_space(self):
        raise NotImplementedError
    