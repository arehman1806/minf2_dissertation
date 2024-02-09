from underactuated_manipulation_gym.envs.base_option_environment import BaseEnvironment
from policy_executors.base_policy_executor import BasePolicyExecutor
from resources.queenie.robot import QueenieRobot
from resources.queenie.robot_env_interface import QueenieRobotEnvInterface
from resources.plane import Plane
from resources.objects.object_loader import ObjectLoader
from resources.target import Target
import gymnasium as gym
import pybullet as p
import yaml
import numpy as np

class MetaEnvironment(BaseEnvironment):

    def __init__(self, config):
        super(MetaEnvironment, self).__init__()

        # figure out a way to loadsubpolicies from yaml file. each policy or subenv will be given a number and a name
        # network will predict action 3. what will that action correspond to? so we need to do the mapping
        # we need to figure out how to do the mapping of the action to the subpolicy

        self._config = self._parse_config(config)
        self._robot_config = self._config["robot"]
        self._environment_config = self._config["environment"]

        render_gui = self.environment_config["gui"]
        connection_mode = p.GUI if render_gui else p.DIRECT
        self.client = p.connect(connection_mode)
        p.setGravity(0,0,-10)
        p.setRealTimeSimulation(0)
        # p.setTimeStep(1./500

        robot_object = QueenieRobot(self.client, self._robot_config)
        self.robot = QueenieRobotEnvInterface(self.client, self._robot_config, robot_object)
        self.target = Target(self.client, ([5,5,0.0], [0,0,0,1]))
        self.plane = Plane(self.client)
        self.object_loader = ObjectLoader(self.client, self._environment_config["object_dataset"], 
                                        num_objects=self._environment_config["num_objects"], 
                                        specific_objects=self._environment_config["specific_objects"],
                                        global_scale=self._environment_config["global_scale"])
        self.current_object = self.object_loader.change_object()

        self._episode_length = self._environment_config["episode_length"]
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.step_i = 0
        self.previous_distance = None
        self.robot_state = None

        self._policy_executors = dict()
        self._load_policy_executors()



    
    def step(self, action):
        # action will be the index of the subpolicy to be used
        # Two options:
        # 1. run the subpolicy until it finishes and then return the observation, reward, done, info
        # 2. run the subpolicy for a fixed number of steps and then return the observation, reward, done, info
        # we will go with the first option for now
        # we will call the execute_policy method of the subpolicy and return the results
        policy_executor = self._policy_executors[action]
        obs, reward, done = policy_executor.execute_policy(steps=-1)

        return obs, reward, done, False, {}
        
    def _get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["image_obs"]
        vect_obs = self.robot_state["proprioception"]
        observation_indices = self.robot_state["proprioception_indices"]
        object_pose = self.current_object.get_base_pose()[0]
        robot_pose = self.robot.get_base_pose()[0]
        target_position = self.target.get_base_position()

        # object and target polar coordinates
        object_target_polar_r, object_target_polar_theta = self.cartesian_to_polar_2d(target_position[0], target_position[1], object_pose[0], object_pose[1])
        robot_target_polar_r, robot_target_polar_theta = self.cartesian_to_polar_2d(target_position[0], target_position[1], robot_pose[0], robot_pose[1])
        robot_object_polar_r, robot_object_polar_theta = self.cartesian_to_polar_2d(object_pose[0], object_pose[1], robot_pose[0], robot_pose[1])

        # add polar coordinates to observation indices
        observation_indices["polar_object_target"] = len(vect_obs)
        observation_indices["polar_robot_target"]  = len(vect_obs) + 2
        observation_indices["polar_robot_object"]  = len(vect_obs) + 4
        
        # add polar coordinates to vector observation
        vect_obs = np.append(vect_obs, [object_target_polar_r, object_target_polar_theta, robot_target_polar_r, robot_target_polar_theta, robot_object_polar_r, robot_object_polar_theta])
        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation, observation_indices
    

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass

    def get_robot(self):
        pass

    def get_target(self):
        pass
    
    def _load_policy_executors(self):
        # load the subpolicies from the config file
        # we have a name for each env and then we have the path to the model
        # we need a completely new set of objects. these objects will have the model and the environments
        for policy_executor in self._config["policy_executors"]:
            self._policy_executors[policy_executor["index"]] = BasePolicyExecutor(policy_executor["env_name"], policy_executor["vec_normalze_path"], policy_executor["model_path"], policy_executor["model_class"])