from gymnasium import spaces
import numpy as np
import cv2
import pybullet as p

from underactuated_manipulation_gym.envs.base_environment import BaseManipulationEnvironment

class PushEnvironment(BaseManipulationEnvironment):
    
    def __init__(self, config_file):
        super().__init__(config_file)
        self.previous_vels = np.array([0, 0])
        self.previous_joint_commands = np.array(len(self.robot_config["actuators"]["joints"]) * [0])
        self.consecutive_graps = 0
        self.robot_state = None
        self._gripper_enabled = self.robot_config["actuators"]["gripper"]

    def render(self, mode="human"):
        # Render the environment for logging to tensorboard
        return self.robot.render_camera_image()
        
    def _reward(self, observation, proprioception_indices, action):

        # reward is based on 2 things: contact with the object, and object movement towards the goal
        # observation space will consist of image and vector. vector will contain information 
        # about object and target polar coordinates, robot and target polar coordinates, robot and 
        #object polar coordinates, and proprioception
        reward = 0

        # penalize large movements, both planer movement and joint angles
        linear_vel, angular_vel = action[:2]
        if self._gripper_enabled:
            gripper = action[-1]
            current_joint_commands = action[2:-1]
        else:
            current_joint_commands = action[2:]
        current_vels = np.array([linear_vel, angular_vel])
        diff_vels = abs(current_vels - self.previous_vels)
        reward += -0.01 * np.sum(diff_vels)
        self.previous_vels = current_vels
        diff_joint_positions = abs(current_joint_commands - self.previous_joint_commands)
        reward += -0.9 * np.sum(diff_joint_positions)
        self.previous_joint_commands = current_joint_commands

        # Penalise movement of the object away from the target and reward movement closer to the target
        distance = self._calculate_object_target_distance()
        if self.previous_distance is None:
            self.previous_distance = distance
        reward += 1 * (self.previous_distance - distance)
        self.previous_distance = distance


        # reward contact to encourage exploration
        contacts = observation["vect_obs"][proprioception_indices["contact"]: proprioception_indices["contact"] + 3]
        num_contacts = sum(contacts)
        reward += 0.01 * num_contacts
        
        # done 
        done = distance < self.environment_config["target_radius"] or self.step_i >= self._episode_length
        if done:
            print("object within 1m of target: ", distance < self.environment_config["target_radius"])
            self.previous_distance = None

        return reward, done
    
    def _calculate_action(self, action):
        # we do not know the size of action, but we know first two are linear and angular velocity
        # this is followed by actuated neck joints and gripper (if gripper is enabled)
        v, w_angular = action[:2]
        # scale the linear and angular velocity
        v = v * self.robot_config["max_linear_velocity"]
        w_angular = w_angular * self.robot_config["max_angular_velocity"]
        v_left = v - w_angular * 0.6 / 2
        v_right = v + w_angular * 0.6 / 2
        
        # actions predicted by network are between -1 and 1
        a = -1
        b = 1
        for i, joint in enumerate(self.robot_config["actuaters"]["joints"]):
            c = self.robot_config["parameters"][joint]["min"]
            d = self.robot_config["parameters"][joint]["max"]
            # formula for converting from one range to another
            action[i+2] = c + ((d - c)*(action[i+2] - a)/(b - a))
        
        return action

        # Extract the action components
        # v, w_angular, neck_y, neck_x, gripper = action

        # v_left = v - w_angular * 0.6 / 2
        # v_right = v + w_angular * 0.6 / 2

        # action = [v_left, v_right, neck_y, neck_x, gripper]
        # return action

    
    """
    This function is called by the environment to get the observation.
    The function converts the robot state to environment state (i.e. observation)
    
    Returns:
        observation: dict of image_obs and vect_obs
        observation_indices: dict of indices of different proprioception values in the vector observation
    """
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
        observation_indices["polar_robot_target"] = len(vect_obs) + 2
        observation_indices["polar_robot_object"] = len(vect_obs) + 4
        # add polar coordinates to vector observation
        vect_obs = np.append(vect_obs, [object_target_polar_r, object_target_polar_theta, robot_target_polar_r, robot_target_polar_theta, robot_object_polar_r, robot_object_polar_theta])

        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation, observation_indices
    
    def _get_observation_space(self):
        obs_space_dry_run = self._get_observation()
        vect_obs_size = obs_space_dry_run[0]["vect_obs"].shape
        image_obs_size = obs_space_dry_run[0]["image_obs"].shape
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})
    
    # def _get_observation_space(self):
    #     # Define the observation space
    #     obs_space_size = self.robot.get_observation_space_size()
    #     vect_obs_size = obs_space_size["vector"]
    #     image_obs_size = obs_space_size["image"]
    #     min_obs = np.full(vect_obs_size[0], -np.inf)
    #     max_obs = np.full(vect_obs_size[0], np.inf)
    #     image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
    #     vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
    #     return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})
        # return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)