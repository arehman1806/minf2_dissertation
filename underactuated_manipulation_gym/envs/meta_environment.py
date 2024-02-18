from underactuated_manipulation_gym.envs.base_option_environment import BaseEnvironment
from underactuated_manipulation_gym.policy_executors.base_policy_executor import BasePolicyExecutor
from underactuated_manipulation_gym.policy_executors.rule_based_policy_executor import RuleBasedPolicyExecutor
from underactuated_manipulation_gym.resources.queenie.robot import QueenieRobot
from underactuated_manipulation_gym.resources.queenie.robot_env_interface import QueenieRobotEnvInterface
from underactuated_manipulation_gym.resources.plane import Plane
from underactuated_manipulation_gym.resources.objects.object_loader import ObjectLoader
from underactuated_manipulation_gym.resources.target import Target
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import yaml
import numpy as np

class MetaEnvironment(BaseEnvironment):

    def __init__(self, config_file):
        super(MetaEnvironment, self).__init__(config_file)

        render_gui = self._environment_config["gui"]
        connection_mode = p.GUI if render_gui else p.DIRECT
        self.client = p.connect(connection_mode)
        p.setGravity(0,0,-10)
        p.setRealTimeSimulation(0)
        # p.setTimeStep(1./500

        self.target = Target(self.client, ([5,5,0.0], [0,0,0,1]))
        self.plane = Plane(self.client)
        self.object_loader = ObjectLoader(self.client, self._environment_config["object_dataset"], 
                                        num_objects=self._environment_config["num_objects"], 
                                        specific_objects=self._environment_config["specific_objects"],
                                        global_scale=self._environment_config["global_scale"])
        self.current_object = self.object_loader.change_object()
        self.robot_object = QueenieRobot(self.client, self._robot_config)
        self.robot = QueenieRobotEnvInterface(self.client, self._robot_config, self.robot_object, self.object_loader)
        
        self._policy_executors = dict()
        self._load_policy_executors()
        self._policy_executor_mapping = self._config["policy_executor_mapping"]

        self._episode_length = self._environment_config["episode_length"]
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.step_i = 0
        self.previous_distance = None
        self.robot_state = None
        self.policy_params = self._config["policy_parameters"]
        self.joint_params = self._robot_config["parameters"]["joint_params"]



    def _reward(self, observation, proprioception_indices, action):

        #reward is based on how far the object is from its target
        reward = 0
        distance = self._calculate_object_target_distance()
        if self.previous_distance is None:
            self.previous_distance = distance
        reward += 10 * (self.previous_distance - distance)
        self.previous_distance = distance
        
        # done 
        done = distance < self._environment_config["target_radius"] or self.step_i >= self._episode_length

        return reward, done

    
    def step(self, action):
        # action will be the index of the subpolicy to be used
        action = action[0]
        policy_executor = self._policy_executors[action]
        observation, proprioception_indices = self.get_observation()
        obs_policy, reward_policy, done_policy = policy_executor.execute_policy(steps=-1)

        reward, object_reahed_target = self._reward(observation, proprioception_indices, action)
        done = object_reahed_target or self.step_i >= self._episode_length

        return self.get_observation()[0], reward, done, False, {}
        
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
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
    
    def get_num_actions(self):
        return len(self._policy_executors.keys())
    

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
        controllers = {"robot": self.robot_object, "plane": self.plane, "object_loader": self.object_loader, "target": self.target, "client": self.client}
        for type_policy, policy_executors in self._config["policy_executors"].items():
            if type_policy == "learning_based":
                for name, policy_executor in policy_executors.items():
                    self._policy_executors[policy_executor["index"]] = BasePolicyExecutor(policy_executor["env_config_file"], controllers)
            elif type_policy == "rule_based":
                for name, policy_executor in policy_executors.items():
                    self._policy_executors[policy_executor["index"]] = RuleBasedPolicyExecutor(policy_executor["env_config_file"], controllers, policy_executor["policy_class"])

    
    def _get_action_space(self):
        # Define the action space for the meta environment
        num_options = len(self._policy_executors)
        return spaces.Discrete(num_options)
    
    def _get_observation_space(self):
        obs_space_dry_run = self.get_observation()
        vect_obs_size = obs_space_dry_run[0]["vect_obs"].shape
        image_obs_size = obs_space_dry_run[0]["image_obs"].shape
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})