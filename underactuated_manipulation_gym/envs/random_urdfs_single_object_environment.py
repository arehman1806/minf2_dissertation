from gymnasium import spaces
import numpy as np
import cv2

from underactuated_manipulation_gym.envs.base_environment import BaseManipulationEnvironment

class RandomURDFsSOEnvironment(BaseManipulationEnvironment):
    
    def __init__(self):
        super().__init__()
        self.previous_action = np.array([0, 0, 0, 0, 0])

    def _reward(self, observation, action):
        linear_vel, angular_vel, neck_y, neck_x, gripper = action
        current_action = np.array([linear_vel, angular_vel, neck_y, neck_x, gripper])
        difference_in_actions = np.linalg.norm(current_action - self.previous_action)
        self.previous_action = current_action
        # Define your reward function
        distance = self._calculate_robot_object_distance()
        # print(str(observation["image_obs"].shape) + "\n\n\n\n\n\n\n\n\n\n\n\n")
        # print(distance)
        if self.previous_distance is None:
            self.previous_distance = distance
        reward = self.previous_distance - distance
        self.previous_distance = distance
        contacts = observation["vect_obs"][-3:]
        num_contacts = sum(contacts)
        if num_contacts > 1:
            reward += num_contacts
        done = False

        return reward, done
    
    def _calculate_action(self, action):
        # Extract the action components
        v, w_angular, neck_y, neck_x, gripper = action

        v_left = v - w_angular * 0.6 / 2
        v_right = v + w_angular * 0.6 / 2

        action = [v_left, v_right, neck_y, neck_x, gripper]
        return action

    
    def _get_observation(self):
        robot_state = self.robot.get_state()
        object_state = self.current_object.get_state()
        # convert robot_state to environment_state (i.e. observation)
        queenie_pos, queenie_orn = robot_state["base_pose"]
        # convert to polar coordinates:
        polar_r, polar_theta = self.cartesian_to_polar_2d(object_state[0][0], object_state[0][1], queenie_pos[0], queenie_pos[1])
        # proprioception
        proprioception = robot_state["proprioception"]
        # contact points
        contacts = robot_state["contact_points"]
        contact_palm = int(len(contacts[0]) > 0)
        contact_left_finger = int(len(contacts[1]) > 0)
        contact_right_finger = int(len(contacts[2]) > 0)
        # try:
        #     camera_img = robot_state["camera_rgb"]
        #     cv2.imwrite("test.png", camera_img)
        # except:
        #     print("no image")
        #     print(robot_state)

        # combine all observations
        vect_obs = np.concatenate((np.array([polar_r, polar_theta], dtype=np.float32), np.array(proprioception, dtype=np.float32), np.array([contact_palm, contact_left_finger, contact_right_finger], np.float32)))
        rgb_image_obs = np.array(robot_state["camera_rgb"], dtype=np.uint8)
        depth_image_obs = np.array(robot_state["camera_depth"], dtype=np.float32)
        #merge rgb and depth images
        image_obs = np.concatenate((rgb_image_obs, depth_image_obs[..., np.newaxis]), axis=-1)
        image_obs = rgb_image_obs
        # Transpose to make it channel-first
        image_obs = np.transpose(image_obs, (2, 0, 1))
        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation
    
    def _get_observation_space(self):
        # Define the observation space
        min_obs = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        max_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        image_obs = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})
        # return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    
    def _get_action_space(self):
        min_linear_vel = -2
        max_linear_vel = 2
        min_angular_vel = -2
        max_angular_vel = 2
        min_neck_joint = -1
        max_neck_joint = 0.21
        min_neck_joint = -0.5
        max_neck_joint = 0.5
        max_gripper = 1
        min_gripper = -1
        min_action = [min_linear_vel, min_angular_vel, min_neck_joint, min_neck_joint, min_gripper]
        max_action = [max_linear_vel, max_angular_vel, max_neck_joint, max_neck_joint, max_gripper]
        return spaces.Box(low=np.array(min_action), high=np.array(max_action), dtype=np.float32)