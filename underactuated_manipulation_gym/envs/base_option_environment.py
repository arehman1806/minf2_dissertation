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
        self.step_i = 0
        if config_file is None:
            raise Exception("No config file provided")
        super(BaseOptionEnvironment, self).__init__(config_file)
        self._as_subpolicy = as_subpolicy
        if not self._as_subpolicy:
            render_gui = self._environment_config["gui"]
            connection_mode = p.GUI if render_gui else p.DIRECT
            self.client = p.connect(connection_mode)
            p.setGravity(0,0,-10)
            p.setRealTimeSimulation(0)
            # p.setTimeStep(1./500

            self.plane = Plane(self.client)
            self.object_loader = ObjectLoader(self.client, "random_urdfs", 
                                            num_objects=self._environment_config["num_objects"], 
                                            specific_objects=self._environment_config["specific_objects"],
                                            global_scale=self._environment_config["global_scale"])
            self.current_object = self.object_loader.change_object()
            robot_object = QueenieRobot(self.client, self._robot_config)
            self.robot = QueenieRobotEnvInterface(self.client, self._robot_config, robot_object, self.object_loader)
            self.target = Target(self.client, ([5,5,0.0], [0,0,0,1]))
        else:
            self.client = controllers["client"]
            self.plane = controllers["plane"]
            self.object_loader = controllers["object_loader"]
            self.robot = QueenieRobotEnvInterface(self.client, self._robot_config, controllers["robot"], self.object_loader)
            self.target = controllers["target"]
            self.current_object = self.object_loader.get_current_object()

        self._episode_length = self._environment_config["episode_length"]
        
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()


        self.previous_distance = None
        self.proprioception_indices = None
    
    def reset(self, seed=None):
        # Reset the environment to its initial state
        self.reset_env_memory()
        self.step_i = 0
        if self._as_subpolicy:
            self.current_object = self.object_loader.get_current_object()
            observation, self.proprioception_indices = self.get_observation()
            return observation, {}
        pos = [0, 0, 0.4]
        orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot.reset(pos, orn)
        self.current_object = self.object_loader.change_object()
        self.target.reset_position(None)
        for _ in range(100):
            p.stepSimulation()
        observation, self.proprioception_indices = self.get_observation()
        return observation, {}
    
    def reset_env_memory(self):
        pass
    
    def _reward(self, observation, proprioception_indices, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError
    
    def _calculate_action(self, action):
        raise NotImplementedError
    
    def terminal_state(self, s):
        raise NotImplementedError
    
    def initial_state(self, s):
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
    
    