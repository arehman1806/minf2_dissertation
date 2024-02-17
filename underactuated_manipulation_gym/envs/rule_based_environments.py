# Packages
import numpy as np
import gymnasium as gym
from gymnasium import spaces
# Local dependencies
from underactuated_manipulation_gym.envs.base_option_environment import BaseOptionEnvironment


class RuleBasedEnvironment(BaseOptionEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
        self.previous_vels = np.array([0, 0])
        self.previous_joint_commands = np.array(len(self._robot_config["actuators"]["joints"]) * [0])
        self.consecutive_graps = 0
        self.robot_state = None
        self._gripper_enabled = self._robot_config["actuators"]["gripper"]
        self.joint_params = self._robot_config["parameters"]["joint_params"]

    """
    Dummy function to conform to the Gym API
    """
    def _reward(self, observation, proprioception_indices, action):
        raise NotImplementedError
    

class PickEnvironment(RuleBasedEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
        self.speed = self._robot_config["parameters"]["speed"]
    
    def _reward(self, observation, proprioception_indices, action):
        done = False
        joint_positions = observation["vect_obs"][proprioception_indices["joint_position"]: proprioception_indices["joint_position"] + 2]
        targets = np.array([-1, 0])
        distance = np.linalg.norm(joint_positions - targets)
        if distance < 0.1:
            done = True
            return 1, done
        return 0, done
    
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
        vect_obs = self.robot_state["proprioception"]
        proprioception_indices = self.robot_state["proprioception_indices"]
        object_state = self.current_object.get_state()
        # convert robot_state to environment_state (i.e. observation)
        # queenie_pos, queenie_orn = self.robot_state["base_pose"]

        # convert to polar coordinates:
        # polar_r, polar_theta = self.cartesian_to_polar_2d(object_state[0][0], object_state[0][1], queenie_pos[0], queenie_pos[1])

        # for now its good enough but maybe extend the vector space to include polar coordinates for closest point on the object

        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation, proprioception_indices
    
    def _get_observation_space(self):
        # Define the observation space
        obs_space_size = self.robot.get_observation_space_size()
        vect_obs_size = obs_space_size["vector"]
        image_obs_size = obs_space_size["image"]
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})
        # return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    

class DragEnvironment(RuleBasedEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
        self.speed = self._robot_config["parameters"]["speed"]
    
    def _reward(self, observation, proprioception_indices, action):
        done = False
        joint_positions = observation["vect_obs"][proprioception_indices["joint_position"]: proprioception_indices["joint_position"] + 2]
        targets = np.array([1, 0])
        distance = np.linalg.norm(joint_positions - targets)
        if distance < 0.1:
            done = True
            return 1, done
        return 0, done
    
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
        vect_obs = self.robot_state["proprioception"]
        observation_indices = self.robot_state["proprioception_indices"]
        object_pose = self.current_object.get_base_pose()[0]
        robot_pose = self.robot.get_base_pose()[0]
        target_position = self.target.get_base_position()

        # object and target polar coordinates
        # object_target_polar_r, object_target_polar_theta = self.cartesian_to_polar_2d(target_position[0], target_position[1], object_pose[0], object_pose[1])
        robot_target_polar_r, robot_target_polar_theta = self.cartesian_to_polar_2d(target_position[0], target_position[1], robot_pose[0], robot_pose[1])
        # robot_object_polar_r, robot_object_polar_theta = self.cartesian_to_polar_2d(object_pose[0], object_pose[1], robot_pose[0], robot_pose[1])

        # add polar coordinates to observation indices
        # observation_indices["polar_object_target"] = len(vect_obs)
        observation_indices["polar_robot_target"] = len(vect_obs) + 2
        # observation_indices["polar_robot_object"] = len(vect_obs) + 4
        # add polar coordinates to vector observation
        vect_obs = np.append(vect_obs, [robot_target_polar_r, robot_target_polar_theta])

        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation, observation_indices
    
    def _get_observation_space(self):
        obs_space_dry_run = self.get_observation()
        vect_obs_size = obs_space_dry_run[0]["vect_obs"].shape
        image_obs_size = obs_space_dry_run[0]["image_obs"].shape
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})
    

class ReorintEnvironment(RuleBasedEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
        self.speed = self._robot_config["parameters"]["speed"]
    
    def _reward(self, observation, proprioception_indices, action):
        done = False
        joint_positions = observation["vect_obs"][proprioception_indices["joint_position"]: proprioception_indices["joint_position"] + 2]
        targets = np.array([0, 1])
        distance = np.linalg.norm(joint_positions - targets)
        if distance < 0.1:
            done = True
            return 1, done
        return 0, done
    
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
        vect_obs = self.robot_state["proprioception"]
        observation_indices = self.robot_state["proprioception_indices"]
        object_pose = self.current_object.get_base_pose()[0]
        robot_pose = self.robot.get_base_pose()[0]
        target_position = self.target.get_base_position()

        # object and target polar coordinates
        # object_target_polar_r, object_target_polar_theta = self.cartesian_to_polar_2d(target_position[0], target_position[1], object_pose[0], object_pose[1])
        # robot_target_polar_r, robot_target_polar_theta = self.cartesian_to_polar_2d(target_position[0], target_position[1], robot_pose[0], robot_pose[1])
        robot_object_polar_r, robot_object_polar_theta = self.cartesian_to_polar_2d(object_pose[0], object_pose[1], robot_pose[0], robot_pose[1])

        # add polar coordinates to observation indices
        # observation_indices["polar_object_target"] = len(vect_obs)
        # observation_indices["polar_robot_target"] = len(vect_obs) + 2
        observation_indices["polar_robot_object"] = len(vect_obs) + 4
        # add polar coordinates to vector observation
        vect_obs = np.append(vect_obs, [robot_object_polar_r, robot_object_polar_theta])

        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation, observation_indices
    
    def _get_observation_space(self):
        obs_space_dry_run = self.get_observation()
        vect_obs_size = obs_space_dry_run[0]["vect_obs"].shape
        image_obs_size = obs_space_dry_run[0]["image_obs"].shape
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})

