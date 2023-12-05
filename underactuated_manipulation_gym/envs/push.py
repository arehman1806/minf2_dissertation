from gymnasium import spaces
import numpy as np
import cv2
import pybullet as p

from underactuated_manipulation_gym.envs.base_env import BaseManipulationEnvironment

class PushFixedJoints(BaseManipulationEnvironment):
    
    def __init__(self, config_file):
        super().__init__(config_file)
        self.previous_vels = np.array([0, 0])
        self.previous_joint_commands = np.array(len(self.robot_config["actuators"]["joints"]) * [0])
        self.consecutive_graps = 0
        self.robot_state = None
        self._gripper_enabled = self.robot_config["actuators"]["gripper"]
        self.goal = np.array([0.5, 0.5])

    def render(self, mode="human"):
        # Render the environment for logging to tensorboard
        return self.robot.render_camera_image()
        
    def _reward(self, observation, proprioception_indices, action):
        reward = 0

        # encourage contact with the object
        contacts = observation["vect_obs"][proprioception_indices["contact"]: proprioception_indices["contact"] + 3]
        num_contacts = sum(contacts)
        reward += 0.01 * num_contacts

        # encourage movement towards the goal

        # Penalise movement away from the object and reward movement closer to the object
        distance = self._calculate_robot_object_distance()
        if self.previous_distance is None:
            self.previous_distance = distance
        reward += 0.01 * self.previous_distance - distance
        self.previous_distance = distance


        # reward contact to encourage exploration
        contacts = observation["vect_obs"][proprioception_indices["contact"]: proprioception_indices["contact"] + 3]
        num_contacts = sum(contacts)
        if num_contacts > 0:
            if contacts[0] == 1:
                print("contact with palm")
                reward += 1000
                return reward, True
            else:
                reward += 0.01 * num_contacts
        done = self.consecutive_graps > 3

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

    
    def _get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["image_obs"]
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